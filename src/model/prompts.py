"""Prompt construction for Qwen3-VL table reconstruction.

Two output modes:

* ``structure`` -- tags and span attributes only, no cell text. The training
  target and the TEDS-Struct input. Roughly 5-10x shorter, which is what makes
  training fit on a small GPU.
* ``full`` -- HTML including cell text. Slower and mixes OCR into the output,
  but the rendered tables look real, so it is what the demo visuals use.

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
    "transcribe the text of each cell. Output raw HTML starting with <table>. No "
    "markdown fences, no explanation."
)

# Option A: the model is handed the OCR text + positions, so it does *not*
# re-read pixels for text -- it only decides structure. Pass this via
# ``instruction=`` together with an ``ocr_layout=`` on predict.
GROUNDED_INSTRUCTION = (
    "You are given the document image and, below it, the OCR text with each "
    "snippet's position. Reconstruct the printed data table as HTML. Take each "
    "cell's text from the OCR -- do not re-transcribe from the image -- and use "
    "the image and the positions to decide the structure: which cells merge "
    "(rowspan/colspan), the header hierarchy (use <th> for headers), and the "
    "reading order. Ignore anything that is not part of the printed table. "
    "Output raw HTML starting with <table> and ending with </table>. No markdown "
    "fences, no explanation."
)

INSTRUCTIONS = {"structure": STRUCTURE_INSTRUCTION, "full": FULL_INSTRUCTION}


def format_layout_block(ocr_layout: str) -> str:
    """Wrap the OCR layout in a delimited block for the user turn.

    Delimiters keep the grounding text from bleeding into the instruction and
    give the model a clear region to treat as source-of-truth for cell text.
    """
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


def build_messages(
    image,
    mode: str = "structure",
    instruction: str | None = None,
    ocr_layout: str | None = None,
) -> list[dict]:
    """Build a Qwen-VL chat message list for one table image.

    ``image`` may be a PIL image or a path; qwen-vl-utils accepts both.

    ``ocr_layout`` (Option A) is appended to the same text block rather than
    added as a second content item -- a single text turn is what every VL
    processor accepts, and it keeps the ``[image, text]`` shape stable. Omit it
    and the message is byte-for-byte the ungrounded one.
    """
    text = resolve_instruction(mode, instruction)
    if ocr_layout:
        text = f"{text}\n\n{format_layout_block(ocr_layout)}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]


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
