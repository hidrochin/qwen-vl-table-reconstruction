"""OCR as a grounding source for the VLM (Option A).

The VLM is a poor character reader and a poor localizer, but a good structurer.
So we run a dedicated OCR engine first, hand the VLM the words *and where they
are*, and let it do only the part it is good at -- arranging known text into the
correct table grid with the right merged cells.

This module is the thin, swappable boundary to whatever OCR engine runs on the
box. Everything downstream (``layout.py``, the grounded prompt) speaks only
``OcrWord``, so switching PaddleOCR for Surya, or for the customer's existing
engine, is a single adapter here and nothing else changes.

Heavy imports (paddleocr, numpy, PIL) are deferred into method bodies so this
module imports cleanly on a Mac with no OCR installed -- matching the rest of
``src/``. Only ``read()`` needs the engine present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class OcrWord:
    """One recognized text span with an axis-aligned box and confidence.

    ``bbox`` is ``(x, y, w, h)`` in pixels, origin top-left -- the convention
    ``layout.py`` assumes. Engines that return quadrilaterals or ``(x1,y1,x2,y2)``
    are normalized to this in their adapter, so no downstream code has to care
    which engine produced the word.
    """

    text: str
    bbox: tuple[float, float, float, float]
    conf: float = 1.0

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
        """Horizontal center -- what column clustering keys on."""
        return self.bbox[0] + self.bbox[2] / 2

    @property
    def cy(self) -> float:
        """Vertical center -- what row clustering keys on."""
        return self.bbox[1] + self.bbox[3] / 2


@runtime_checkable
class OcrEngine(Protocol):
    """Anything that turns an image into ``OcrWord``s. Implement this to swap
    engines; ``layout.py`` and the grounded prompt depend only on the output."""

    def read(self, image_path: str | Path) -> list[OcrWord]: ...


def _poly_to_xywh(poly) -> tuple[float, float, float, float]:
    """Axis-aligned box from a 4-point polygon ``[[x,y], ...]``.

    PaddleOCR returns rotated quads; the layout algorithm works on upright
    boxes, and for near-horizontal document text the bounding rectangle loses
    nothing that matters.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0 = min(xs), min(ys)
    return (float(x0), float(y0), float(max(xs) - x0), float(max(ys) - y0))


class PaddleOcrEngine:
    """Default engine: PaddleOCR. Mature, offline, on-prem friendly, and its
    word boxes are good enough to ground the VLM.

    The engine is constructed lazily on first ``read`` so importing this module
    (and therefore all of ``src/``) never requires paddle to be installed. Tune
    ``lang`` for the invoice language; ``use_angle_cls`` corrects rotated scans.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, **kwargs):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._kwargs = kwargs
        self._ocr = None

    def _ensure(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls, lang=self.lang, show_log=False, **self._kwargs
            )
        return self._ocr

    def read(self, image_path: str | Path) -> list[OcrWord]:
        ocr = self._ensure()
        result = ocr.ocr(str(image_path), cls=self.use_angle_cls)
        # PaddleOCR nests per-image: result[0] is the page's line list, each
        # entry [poly, (text, conf)]. Guard against the empty-page shape.
        page = result[0] if result and result[0] is not None else []
        words: list[OcrWord] = []
        for poly, (text, conf) in page:
            text = text.strip()
            if not text:
                continue
            words.append(OcrWord(text=text, bbox=_poly_to_xywh(poly), conf=float(conf)))
        return words


def run_ocr(image_path: str | Path, engine: OcrEngine | None = None) -> list[OcrWord]:
    """Read one image, defaulting to PaddleOCR. Pass ``engine`` to override."""
    engine = engine or PaddleOcrEngine()
    return engine.read(image_path)
