"""The reading pass -- Qwen as its own OCR, made reliable by construction.

With the OCR specialists de-scoped (``pipeline_design.md`` §2), the design gets a
trustworthy fragment set from Qwen itself, not by hoping a full-page look reads
every digit but by structuring the perception:

* **Structure-blind contract.** The reading prompt asks for fragments only --
  text + box, guided JSON, required fields -- never rows, columns, or meaning.
  Perception must be *unconditioned* on structure, or verification later would
  check the model against its own theory instead of against the page.
* **Tiling.** Overlapping high-resolution crops (``plan_tiles``) so small print
  reaches the encoder at a legible scale; per-tile reading; boxes mapped back to
  full-image coordinates and de-duplicated across the overlaps (``merge_fragments``).
* **Consensus.** Several reads (different tiling offsets) are merged; agreement
  raises confidence, disagreement is carried as ``alternates`` and a
  ``contested`` flag rather than silently resolved.
* **Unread-ink sweep.** ``find_unread_ink`` binarizes the crop and returns any
  ink region no fragment box covers, to be re-read at maximum zoom -- the recall
  net for faint print.

The output is a list of :class:`Fragment`, which is **``OcrWord``-compatible** (same
``x/y/w/h/cx/cy`` interface) so ``ocr/layout.py`` and the grounding block consume
it unchanged, plus an ``id`` (referenced by the grounded cell list in
``model/cells.py``), ``alternates``, and ``page``.

The pure geometry -- tiling, IoU merge, consensus, the fragment block -- is
plain Python and tested on the Mac. The model call and PIL are injected /
deferred, so this module imports with nothing installed; only a live
:class:`ReadingPass` needs them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --- the structure-blind reading contract -------------------------------------

READING_INSTRUCTION = (
    "Transcribe EVERY printed text fragment visible in this image, with its pixel "
    "box. A fragment is one short run of text with no large gap inside it -- a "
    "word, a number, a code, a date, or a short phrase that sits together in one "
    "place. Read everything printed, including faint, small, or repeated text, and "
    "including stamps, handwriting, and any text in or beside a QR code or barcode "
    "-- deciding what belongs to the table happens later, not now.\n"
    "Do NOT group fragments into rows, columns, or cells. Do NOT interpret headers "
    "or meaning. Do NOT reorder, merge across gaps, skip, or invent text.\n"
    'Output ONE JSON object and nothing else: {"fragments": [{"t": "<exact text>", '
    '"b": [x1, y1, x2, y2]}, ...]}. b is the pixel box enclosing that fragment, '
    "(x1,y1) top-left and (x2,y2) bottom-right, in THIS image's pixels. No markdown "
    "fences, no explanation."
)


def fragments_json_schema() -> dict:
    """Guided-JSON schema for the reading pass: a flat list of ``{t, b}``.

    Text and box are required on every fragment; nothing structural is allowed,
    which is what keeps the pass unconditioned on the table's structure.
    """
    return {
        "type": "object",
        "properties": {
            "fragments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "t": {"type": "string", "minLength": 1},
                        "b": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["t", "b"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["fragments"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Fragment:
    """One read text span: ``OcrWord``-compatible plus id/alternates/page.

    ``bbox`` is ``(x, y, w, h)`` in full-image pixels -- the convention
    ``layout.py`` expects -- so a list of ``Fragment`` drops straight into the
    survey. ``id`` is the stable handle the grounded cell list references;
    ``alternates`` and ``contested`` carry cross-read disagreement.
    """

    id: int
    text: str
    bbox: tuple[float, float, float, float]
    conf: float = 1.0
    page: int = 1
    alternates: tuple[str, ...] = ()
    contested: bool = False

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def w(self) -> float:
        return self.bbox[2]

    @property
    def h(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2

    @property
    def cy(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2

    @property
    def box_xyxy(self) -> tuple[float, float, float, float]:
        return (self.bbox[0], self.bbox[1], self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3])


def _norm(text: str) -> str:
    return " ".join(str(text).split())


# --- tiling -------------------------------------------------------------------


def plan_tiles(
    width: int, height: int, tile: int = 1024, overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Cover a ``width x height`` image with overlapping square-ish tiles.

    Returns ``(x1, y1, x2, y2)`` crops with a ``overlap`` fraction shared between
    neighbours, so a fragment straddling a tile edge is read whole in at least
    one tile. An image no larger than ``tile`` yields a single full-image crop.
    Pure geometry; the caller does the cropping.
    """
    if width <= 0 or height <= 0:
        return []
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        pts = list(range(0, extent - tile + 1, step))
        if pts[-1] != extent - tile:
            pts.append(extent - tile)  # a final tile flush to the edge
        return pts

    tiles = []
    for y in starts(height):
        for x in starts(width):
            tiles.append((x, y, min(x + tile, width), min(y + tile, height)))
    return tiles


# --- parsing a single read ----------------------------------------------------


def parse_fragments(raw: str, origin: tuple[int, int] = (0, 0), page: int = 1) -> list[Fragment]:
    """Parse one reading-pass response into ``Fragment`` s in full-image space.

    ``origin`` is the tile's top-left in the full image; boxes are offset by it
    and converted from the model's ``[x1,y1,x2,y2]`` to ``(x,y,w,h)``. Empty text
    and degenerate boxes are dropped. ids are provisional (0-based within this
    read); ``merge_fragments`` assigns the final, stable ids.
    """
    import json
    import re

    if not raw:
        return []
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    if "```" in text:
        for block in text.split("```")[1::2]:
            body = block.split("\n", 1)[-1] if block[:12].lower().startswith("json") else block
            if '"fragments"' in body:
                text = body
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    ox, oy = origin
    out: list[Fragment] = []
    for i, frag in enumerate(payload.get("fragments", []) if isinstance(payload, dict) else []):
        if not isinstance(frag, dict):
            continue
        t = _norm(frag.get("t", ""))
        b = frag.get("b")
        if not t or not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            Fragment(id=i, text=t, bbox=(x1 + ox, y1 + oy, x2 - x1, y2 - y1), page=page)
        )
    return out


# --- cross-tile / cross-read merge and consensus ------------------------------


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def merge_fragments(fragments: list[Fragment], iou_thresh: float = 0.5) -> list[Fragment]:
    """De-duplicate fragments that describe the same physical text.

    Fragments from overlapping tiles (and from several consensus reads) that
    overlap by more than ``iou_thresh`` are one span. Each cluster becomes a
    single :class:`Fragment`: the **majority** normalized text wins, minority
    readings become ``alternates``, ``contested`` marks any disagreement, the box
    is the cluster average, and ``conf`` is the agreement fraction. Output is in
    reading order (top-to-bottom, left-to-right) with final ids ``0..n-1``.
    """
    items = list(fragments)
    used = [False] * len(items)
    clusters: list[list[Fragment]] = []
    # Greedy: seed a cluster, absorb every later fragment overlapping the seed.
    for i, frag in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        group = [frag]
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            if items[j].page == frag.page and _iou(frag.box_xyxy, items[j].box_xyxy) >= iou_thresh:
                used[j] = True
                group.append(items[j])
        clusters.append(group)

    merged: list[Fragment] = []
    for group in clusters:
        texts = [g.text for g in group]
        counts = Counter(_norm(t) for t in texts)
        best, votes = counts.most_common(1)[0]
        alternates = tuple(sorted(k for k in counts if k != best))
        n = len(group)
        cx1 = sum(g.x for g in group) / n
        cy1 = sum(g.y for g in group) / n
        cw = sum(g.w for g in group) / n
        ch = sum(g.h for g in group) / n
        merged.append(
            Fragment(
                id=0,
                text=best,
                bbox=(cx1, cy1, cw, ch),
                conf=votes / n,
                page=group[0].page,
                alternates=alternates,
                contested=len(counts) > 1,
            )
        )

    merged.sort(key=lambda f: (f.page, round(f.cy, 1), f.cx))
    return [
        Fragment(
            id=i,
            text=f.text,
            bbox=f.bbox,
            conf=f.conf,
            page=f.page,
            alternates=f.alternates,
            contested=f.contested,
        )
        for i, f in enumerate(merged)
    ]


def render_fragment_block(fragments: list[Fragment]) -> str:
    """The numbered fragment list handed to the grounded cell prompt.

    One line per fragment: ``id: "text" (x,y,w,h)`` (plus alternates when a read
    was contested), so the model can reference fragments by id. Pass this as the
    ``ocr_layout`` block alongside ``build_cells_instruction(with_fragments=True)``.
    """
    lines = []
    for f in fragments:
        alt = f" alt={list(f.alternates)}" if f.alternates else ""
        lines.append(
            f'{f.id}: "{f.text}" (x={f.x:.0f}, y={f.y:.0f}, w={f.w:.0f}, h={f.h:.0f}, p={f.page}){alt}'
        )
    return "\n".join(lines)


# --- unread-ink recall net ----------------------------------------------------


def find_unread_ink(
    image,
    fragments: list[Fragment],
    patch: int = 32,
    ink_frac: float = 0.04,
    dark: int = 128,
) -> list[tuple[int, int, int, int]]:
    """Regions of ink that no fragment box covers -- the recall net.

    Binarizes ``image`` (a path or PIL image), scans it on a ``patch`` grid,
    marks any patch with ink not overlapped by a fragment box, and merges
    adjacent marked patches into rectangles to re-read at zoom. Returns xyxy
    rects; ``[]`` if PIL is missing or nothing is uncovered. Deferred PIL import.
    """
    try:
        from PIL import Image
    except ImportError:
        return []
    img = image if hasattr(image, "crop") else Image.open(image)
    img = img.convert("L")
    W, H = img.width, img.height
    boxes = [f.box_xyxy for f in fragments]

    def covered(px1, py1, px2, py2) -> bool:
        for bx1, by1, bx2, by2 in boxes:
            if not (bx2 <= px1 or bx1 >= px2 or by2 <= py1 or by1 >= py2):
                return True
        return False

    nx, ny = (W + patch - 1) // patch, (H + patch - 1) // patch
    marked = [[False] * nx for _ in range(ny)]
    for gy in range(ny):
        for gx in range(nx):
            x1, y1 = gx * patch, gy * patch
            x2, y2 = min(x1 + patch, W), min(y1 + patch, H)
            if x2 - x1 < 2 or y2 - y1 < 2 or covered(x1, y1, x2, y2):
                continue
            crop = img.crop((x1, y1, x2, y2))
            hist = crop.histogram()
            area = (x2 - x1) * (y2 - y1)
            if area and sum(hist[:dark]) / area > ink_frac:
                marked[gy][gx] = True

    return _merge_marked_patches(marked, patch, W, H)


def _merge_marked_patches(marked, patch, W, H) -> list[tuple[int, int, int, int]]:
    """Connected-component merge of marked grid patches into xyxy rects (4-neighbour)."""
    ny, nx = len(marked), len(marked[0]) if marked else 0
    seen = [[False] * nx for _ in range(ny)]
    rects = []
    for sy in range(ny):
        for sx in range(nx):
            if not marked[sy][sx] or seen[sy][sx]:
                continue
            stack = [(sy, sx)]
            seen[sy][sx] = True
            minx, miny, maxx, maxy = sx, sy, sx, sy
            while stack:
                cy, cx = stack.pop()
                minx, maxx = min(minx, cx), max(maxx, cx)
                miny, maxy = min(miny, cy), max(maxy, cy)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ay, ax = cy + dy, cx + dx
                    if 0 <= ay < ny and 0 <= ax < nx and marked[ay][ax] and not seen[ay][ax]:
                        seen[ay][ax] = True
                        stack.append((ay, ax))
            rects.append(
                (minx * patch, miny * patch, min((maxx + 1) * patch, W), min((maxy + 1) * patch, H))
            )
    return rects


# --- orchestration (needs a model + PIL) --------------------------------------

# A predict callable: (image_or_path, instruction, guided_json) -> raw model text.
# Injected so the pass is backend-agnostic (local transformers, or a served vLLM
# client wrapped to accept a PIL crop) and testable with a fake.
PredictFn = Callable[..., str]


@dataclass
class ReadingPass:
    """Run the full reading pass for one page: tile -> read -> merge -> sweep.

    ``predict_fn(image, instruction, guided_json) -> raw`` is the only model
    dependency; pass a wrapper over ``VLLMTableReconstructor``/``TableReconstructor``
    that accepts a PIL crop. Everything else is deterministic. Not exercised on
    the Mac (needs a model + images); its helpers above are.
    """

    predict_fn: PredictFn
    tile: int = 1024
    overlap: float = 0.2
    consensus_offsets: tuple[int, ...] = (0,)  # extra x/y shifts for extra reads
    iou_thresh: float = 0.5
    sweep: bool = True
    _schema: dict = field(default_factory=fragments_json_schema)

    def read(self, image_path: str | Path, page: int = 1) -> list[Fragment]:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        raw_frags: list[Fragment] = []
        for off in self.consensus_offsets:
            for (x1, y1, x2, y2) in plan_tiles(img.width, img.height, self.tile, self.overlap):
                ox, oy = max(0, x1 - off), max(0, y1 - off)
                crop = img.crop((ox, oy, x2, y2))
                raw = self.predict_fn(crop, READING_INSTRUCTION, self._schema)
                raw_frags.extend(parse_fragments(raw, origin=(ox, oy), page=page))

        fragments = merge_fragments(raw_frags, self.iou_thresh)
        if self.sweep:
            fragments = self._sweep(img, fragments, page)
        return fragments

    def _sweep(self, img, fragments: list[Fragment], page: int) -> list[Fragment]:
        extra: list[Fragment] = list(fragments)
        for rect in find_unread_ink(img, fragments):
            x1, y1, x2, y2 = rect
            crop = img.crop((x1, y1, x2, y2))
            raw = self.predict_fn(crop, READING_INSTRUCTION, self._schema)
            extra.extend(parse_fragments(raw, origin=(x1, y1), page=page))
        return merge_fragments(extra, self.iou_thresh) if len(extra) != len(fragments) else fragments


def reread_regions(
    predict_fn: PredictFn, image_path: str | Path, regions, page: int = 1
) -> list[Fragment]:
    """Re-read specific xyxy regions at zoom -- for ``ocr_missed`` follow-ups.

    Referenced by ``cells.collect_ocr_missed``: hand it the boxes the model
    claimed carried text no fragment covered, and it returns the fragments read
    from just those crops (deduped). Deferred PIL import.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    frags: list[Fragment] = []
    for x1, y1, x2, y2 in regions:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        raw = predict_fn(img.crop((x1, y1, x2, y2)), READING_INSTRUCTION, fragments_json_schema())
        frags.extend(parse_fragments(raw, origin=(x1, y1), page=page))
    return merge_fragments(frags)
