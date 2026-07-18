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

INSTRUCTIONS = {"structure": STRUCTURE_INSTRUCTION, "full": FULL_INSTRUCTION}


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


def build_messages(image, mode: str = "structure", instruction: str | None = None) -> list[dict]:
    """Build a Qwen-VL chat message list for one table image.

    ``image`` may be a PIL image or a path; qwen-vl-utils accepts both.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": resolve_instruction(mode, instruction)},
            ],
        }
    ]


def build_training_example(
    image, target_html: str, mode: str = "structure", instruction: str | None = None
) -> dict:
    """Build a supervised example: prompt messages plus the assistant turn."""
    messages = build_messages(image, mode, instruction)
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
