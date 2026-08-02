"""Prompt construction for Qwen3-VL table reconstruction.

Three modes:

* ``structure`` -- tags and span attributes only, no cell text. The training
  target and the TEDS-Struct input. Roughly 5-10x shorter, which is what makes
  training fit on a small GPU.
* ``full`` -- HTML including cell text. Slower and mixes OCR into the output,
  but the rendered tables look real, so it is what the demo visuals use.
* ``schema`` -- the logical-reconstruction path for the invoice tables in
  ``layout_description.md``. The default is **document-agnostic**: it teaches the
  reconstruction *principles* (hierarchical + implicit headers, a variable
  logical-column count, sparse/blank cells, section and summary rows, semantic
  column assignment) and makes the model infer the schema from the body. It does
  **not** hard-code a column layout -- the header varies across documents, and a
  prompt pinned to one sample only reconstructs documents that match that sample.
  Pass a known header via ``build_schema_instruction(schema=...)`` only as an
  optional *hint to verify*, and ``with_bbox=True`` to also emit a per-cell
  ``data-bbox``. Pair with ``ocr_layout=`` (stage-1) when OCR is available.

Prompts are deliberately terse about format. Base VLMs like to wrap output in
markdown fences and add commentary, both of which break parsing; saying so
explicitly is cheaper than post-hoc cleanup.
"""

from __future__ import annotations

STRUCTURE_INSTRUCTION = (
    "Reconstruct this table's HTML structure. Preserve every rowspan, colspan, "
    "merged cell, header hierarchy, and the reading order. Emit only the tags "
    "with their rowspan/colspan attributes and leave all cells empty -- no cell "
    "text. Output raw HTML starting with <table>. No markdown fences, no "
    "explanation."
)

FULL_INSTRUCTION = (
    "Reconstruct the complete HTML representation of this table. Preserve every "
    "rowspan, colspan, merged cell, header hierarchy, and the reading order, and "
    "transcribe the text of each cell. Ignore anything that is not part of the "
    "printed table -- handwritten notes and signatures, stamps, QR codes and "
    "barcodes, and any text printed beside a QR/barcode. Output raw HTML starting "
    "with <table>. No markdown fences, no explanation."
)

# Option A: the model is handed the OCR text + positions, so it does *not*
# re-read pixels for text -- it only decides structure. Pass this via
# ``instruction=`` together with an ``ocr_layout=`` on predict.
GROUNDED_INSTRUCTION = (
    "You are given an image that a detector has already cropped to a single table "
    "(it contains only the table, and the crop may slightly clip the outer frame "
    "or the edge of the first/last column) and, below it, the OCR text with each "
    "snippet's position. Reconstruct the printed data table as HTML. Take each "
    "cell's text from the OCR -- do not re-transcribe from the image -- and use "
    "the image and the positions to decide the structure: which cells merge "
    "(rowspan/colspan), the header hierarchy (use <th> for headers), and the "
    "reading order. Do not treat a clipped or missing border as a column boundary "
    "or a merged cell -- reconstruct every column the body supports. Ignore "
    "anything that is not part of the printed data table, including content laid "
    "over it: handwritten notes and signatures, stamps, QR codes and barcodes, and "
    "any text printed as part of or beside a QR/barcode (reference IDs, URLs, "
    "codes). Output raw HTML starting with <table> and ending with </table>. No "
    "markdown fences, no explanation."
)

# Schema-conditioned prompting (the two-stage / logical-reconstruction path).
#
# These tables (see ``layout_description.md``) are not conventional TSR: the
# header is *hierarchical* and partly *implicit* (some child columns are never
# printed), and the *body* -- not the header -- fixes the logical schema, down to
# how many logical columns a record group actually has. Crucially the header
# **varies across documents**, so the prompt must NOT hard-code one layout: a
# prompt pinned to a single sample only reconstructs documents that match that
# sample (the observed "some worked, some failed" failure mode). The default
# instruction below is therefore document-agnostic -- it teaches the principles
# and makes the model infer the schema from body evidence. A specific header may
# be supplied later, but only as an optional *hint to verify*.

_SCHEMA_INTRO = (
    "Reconstruct the latent LOGICAL table of the printed data table as HTML -- "
    "recover the logical table, not its visual appearance. If OCR text with "
    "positions is provided below the image, take each cell's text from the OCR "
    "(do not re-transcribe from pixels) and use the positions to place cells; "
    "otherwise read the values from the image."
)

# The image is not a full page: a table detector has already cropped it to the
# table region, and that crop is often tight enough to shave off the outer frame
# or the edge of the first/last column. The failure this guards against is the
# model reading a missing border as "no column here" and dropping or merging an
# edge column.
_CROPPED_TABLE_NOTE = (
    "The image is a table region already cropped from the page by a detector, so "
    "it contains only the table. The crop can be tight and may clip the outer "
    "frame or the left/right edge of the first or last column. Do not treat a "
    "missing or clipped border as a column boundary, a missing column, or a merged "
    "cell: reconstruct every logical column the body supports even where the frame "
    "is cut, and still read a value that sits at a slightly clipped edge."
)

# The reconstruction principles, unnumbered. The final item is the only
# HTML-specific one; ``cells.py`` reuses items [:-1] and swaps in a JSON-output
# version of it, so edit the shared items here once for both output formats.
_SCHEMA_PRINCIPLE_ITEMS: tuple[str, ...] = (
    "Infer the logical schema from the whole table, not the header alone. The "
    "header is hierarchical: parent headers span several child columns, and some "
    "child columns are implied by the body and never printed. Decide how many "
    "logical columns exist, and what each means, from vertical alignment, value "
    "semantics, and neighbouring values.",
    "The logical column count is VARIABLE across documents. An optional or "
    "implicit column exists only when the body actually supplies values for it. "
    "Do not add a column the body does not support, and do not drop one it does. "
    "The printed header does not signal this -- the body does.",
    "Assign every printed text fragment to exactly one logical cell. Never "
    "invent, duplicate, move, or drop a value.",
    "Keep blank cells blank. A blank is real content: never shift a "
    "neighbouring value into an empty position, and never merge two logical "
    "columns just because one is sparse.",
    "Rows are grouped. A section/group row (a label that starts a block) "
    "carries text only in its first logical column; every other cell in that row "
    "is empty. A summary row (e.g. a total) is an ordinary row with values in "
    "only some columns.",
    "Rows and columns are often separated by alignment and spacing, not printed "
    "lines. Use value semantics (a numeric column stays numeric, a symbol column "
    "stays symbols, a percentage/amount column stays that type) together with "
    "geometry to keep each value in the correct row and column.",
    "Use <th> for header cells and preserve every parent/child relationship "
    "with rowspan/colspan. Preserve merged cells in headers, section rows, and "
    "summary rows.",
)


def number_principles(items: tuple[str, ...]) -> str:
    """Join principle items into the numbered 'How to reconstruct' block."""
    body = "\n".join(f"{i}. {text}" for i, text in enumerate(items, 1))
    return f"How to reconstruct:\n{body}"


_SCHEMA_PRINCIPLES = number_principles(_SCHEMA_PRINCIPLE_ITEMS)

_IGNORE_RULE = (
    "Ignore everything that is not part of the printed data table: handwritten "
    "notes, annotations and signatures; stamps; QR codes and barcodes; and any "
    "text printed as part of or beside a QR/barcode (reference IDs, URLs, or codes "
    "under or next to it). Never place any of it in a cell."
)

_SCHEMA_HINT_TEMPLATE = (
    "Optional header hint for this document family (it may be absent or different "
    "in this particular document). Treat it ONLY as a clue to recognise the "
    "columns, not a structure to force -- parent/child names may differ, and "
    "optional or implicit columns may be present or absent here. Verify every part "
    "against what is actually printed:\n{schema}"
)

_BBOX_RULE = (
    "For every non-empty cell, add a data-bbox attribute giving the pixel box that "
    'encloses that cell\'s content, as data-bbox="x1,y1,x2,y2" -- (x1,y1) is the '
    "top-left corner and (x2,y2) the bottom-right, in pixels of the image as given "
    "to you. Emit ONE box per logical cell: for a cell built from several words or "
    "OCR fragments, use the single box that encloses all of them (when OCR "
    "positions are given, the union of the fragments you assigned to that cell). "
    "Give a truly blank cell no data-bbox -- it has nothing to bound. Keep every "
    "coordinate inside the image: for a cell at a clipped edge, extend its box to "
    "the image border rather than past it. These data-bbox attributes are required "
    "in every response, on every request, even when a request repeats an earlier "
    "image -- never drop them."
)

# Appended automatically by ``build_messages`` when it receives more than one
# image: a long table split across page crops must come back as ONE logical
# table, and the classic failures are a re-emitted continuation header and a
# row at the page break either duplicated or torn in two.
_MULTI_PAGE_NOTE = (
    "You are given {n} images. They are consecutive crops of ONE table that "
    "continues across pages, in reading order (image 1 is the top). Reconstruct "
    "them as a SINGLE logical table:\n"
    "- If a later image repeats the header (a continuation header), recognise it "
    "and emit the header only once, at the top.\n"
    "- A row cut by the page break (its text starts at the bottom of one image "
    "and continues at the top of the next) is ONE row -- merge it, do not emit "
    "it twice or as two rows.\n"
    "- Column boundaries are the same in every image: align the pages by their "
    "columns, and carry section groups that started on an earlier page across "
    "the break.\n"
    "- Wherever a pixel position is reported (data-bbox or cell boxes), also "
    'give the 1-based image it belongs to as data-page="N", with coordinates in '
    "that image's own pixel space."
)

_OUTPUT_PLAIN = (
    "Output raw HTML starting with <table> and ending with </table>. No markdown "
    "fences, no explanation."
)
_OUTPUT_BBOX = (
    "Output raw HTML starting with <table> and ending with </table>, with a "
    "data-bbox on every non-empty cell. No markdown fences, no explanation."
)

# The example from ``layout_description.md``, kept ONLY as an optional hint to
# pass to ``build_schema_instruction(schema=...)``. It is deliberately NOT the
# default and NOT baked into ``mode="schema"`` -- baking it in is what overfit the
# prompt to a single sample layout.
DEFAULT_INVOICE_SCHEMA = (
    "Header (two levels), one observed example -- other documents differ:\n"
    "  - Currency  -> child columns: Description, Value\n"
    "  - Figure100 -> child columns: Number, Type, [Optional]\n"
    "  - Figures   -> child columns: Debit, Credit\n"
    "Here the 'Optional' child under Figure100 appears only when the body supplies "
    "a third value in that group (e.g. 'Nil'); otherwise Figure100 has just Number "
    "and Type. Number is numeric, Type is a symbol (D/C), Value is like '3%' or "
    "'1000', Debit/Credit are amounts."
)


def build_schema_instruction(schema: str | None = None, with_bbox: bool = False) -> str:
    """Assemble the schema-inference (logical-reconstruction) instruction.

    ``schema`` -- an *optional* header hint for a known document family, framed as
    a clue to verify rather than a layout to force. Omit it (the default) for the
    document-agnostic prompt that infers the schema from the body. Passing the old
    fixed header here is what previously overfit the prompt to one sample, so pass
    a hint only when you are sure the family's header is stable.

    ``with_bbox`` -- also require a per-cell ``data-bbox`` (one pixel box per
    logical cell, enclosing that cell's content) so the caller can draw cell
    boxes. If you fine-tune with this on, the teacher labels and the inference
    prompt must use the same ``with_bbox`` value -- the train/inference prompts
    have to match.

    Pass the result via ``instruction=``; editing this module mid-session has no
    effect once it is imported (see ``resolve_instruction``).
    """
    parts = [_SCHEMA_INTRO, _CROPPED_TABLE_NOTE, _SCHEMA_PRINCIPLES, _IGNORE_RULE]
    if schema:
        parts.append(_SCHEMA_HINT_TEMPLATE.format(schema=schema.strip()))
    if with_bbox:
        parts.append(_BBOX_RULE)
        parts.append(_OUTPUT_BBOX)
    else:
        parts.append(_OUTPUT_PLAIN)
    return "\n\n".join(parts)


def append_bbox_rule(instruction: str) -> str:
    """Add the per-cell ``data-bbox`` requirement to any instruction.

    For the ``full``/``grounded`` paths; the ``schema`` path uses
    ``build_schema_instruction(with_bbox=True)`` instead. The base instruction
    already says to output HTML, so this only adds the bbox rule.
    """
    return f"{instruction}\n\n{_BBOX_RULE}"


SCHEMA_INSTRUCTION = build_schema_instruction()

INSTRUCTIONS = {
    "structure": STRUCTURE_INSTRUCTION,
    "full": FULL_INSTRUCTION,
    "schema": SCHEMA_INSTRUCTION,
}


def format_layout_block(ocr_layout: str | list[str]) -> str:
    """Wrap the OCR layout in a delimited block for the user turn.

    Delimiters keep the grounding text from bleeding into the instruction and
    give the model a clear region to treat as source-of-truth for cell text.
    A list means one layout per page image, in the same order the images are
    given; each page gets its own tagged block so fragment positions stay tied
    to the right image's pixel space.
    """
    if isinstance(ocr_layout, list):
        blocks = "\n".join(
            f'<ocr_layout page="{i}">\n{layout}\n</ocr_layout>'
            for i, layout in enumerate(ocr_layout, 1)
        )
        return (
            "OCR text and layout extracted from each image (page numbers match "
            "the image order; positions are in that page's own pixels) -- use it "
            "as the source of cell text and positions; do not re-read the pixels "
            f"for text:\n{blocks}"
        )
    return (
        "OCR text and layout extracted from the image -- use it as the source of "
        "cell text and positions; do not re-read the pixels for text:\n"
        f"<ocr_layout>\n{ocr_layout}\n</ocr_layout>"
    )


def resolve_instruction(mode: str = "structure", instruction: str | None = None) -> str:
    """Pick the instruction text: an explicit override, else the mode default.

    The override exists so prompt iteration can happen in the notebook. Editing
    this file and re-running a cell does *not* pick up the change -- Python
    caches the imported module -- so a file-edit workflow silently scores the old
    prompt twice and reads as "prompt engineering did nothing".
    """
    if instruction is not None:
        return instruction
    if mode not in INSTRUCTIONS:
        raise ValueError(f"mode must be one of {sorted(INSTRUCTIONS)}, got {mode!r}")
    return INSTRUCTIONS[mode]


def as_image_list(image) -> list:
    """Normalize the ``image`` argument: one image or an ordered page list."""
    return list(image) if isinstance(image, (list, tuple)) else [image]


def build_prompt_text(
    n_images: int,
    mode: str = "structure",
    instruction: str | None = None,
    ocr_layout: str | list[str] | None = None,
) -> str:
    """Assemble the full user-turn text: instruction, multi-page note, OCR block.

    Shared by the local and the served (vllm_client) paths so a multi-page call
    is prompted identically regardless of backend.
    """
    text = resolve_instruction(mode, instruction)
    if n_images > 1:
        text = f"{text}\n\n{_MULTI_PAGE_NOTE.format(n=n_images)}"
    if ocr_layout:
        text = f"{text}\n\n{format_layout_block(ocr_layout)}"
    return text


def build_messages(
    image,
    mode: str = "structure",
    instruction: str | None = None,
    ocr_layout: str | list[str] | None = None,
) -> list[dict]:
    """Build a Qwen-VL chat message list for one table (one or more images).

    ``image`` may be a PIL image or a path (qwen-vl-utils accepts both), or a
    **list** of them when one long table is split across several page crops --
    the images go in reading order and the multi-page stitching note is added
    automatically.

    ``ocr_layout`` (Option A) is appended to the same text block rather than
    added as a second content item -- a single text turn is what every VL
    processor accepts, and it keeps the ``[images..., text]`` shape stable. Pass
    a list (one layout per page) with multi-image input. Omit it and a
    single-image message is byte-for-byte the ungrounded one.
    """
    images = as_image_list(image)
    text = build_prompt_text(len(images), mode, instruction, ocr_layout)
    content: list[dict] = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def build_training_example(
    image,
    target_html: str,
    mode: str = "structure",
    instruction: str | None = None,
    ocr_layout: str | None = None,
) -> dict:
    """Build a supervised example: prompt messages plus the assistant turn.

    ``ocr_layout`` must be included here too when training the grounded model,
    or the adapter learns from a prompt it will never see at inference -- the
    same train/inference prompt-match rule the LoRA config guards.
    """
    messages = build_messages(image, mode, instruction, ocr_layout)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": target_html}]})
    return {"messages": messages}


def clean_prediction(raw: str) -> str:
    """Strip the wrappers models add despite being told not to.

    Handles markdown fences, leading prose, and trailing commentary, then trims
    to the outermost <table>...</table>. Returns "" if no table is present --
    the caller scores that as 0 rather than guessing.
    """
    if not raw:
        return ""
    text = raw.strip()

    if "```" in text:
        blocks = text.split("```")
        # Odd indices are fenced bodies; prefer the first containing a table.
        for block in blocks[1::2]:
            body = block.split("\n", 1)[-1] if block[:12].lower().startswith("html") else block
            if "<table" in body:
                text = body
                break

    start = text.lower().find("<table")
    if start == -1:
        return ""
    end = text.lower().rfind("</table>")
    return text[start:] if end == -1 else text[start : end + len("</table>")]
