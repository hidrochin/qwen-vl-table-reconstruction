"""Tests for OCR-grounded prompting (Option A).

Grounding must be *additive*: without an ``ocr_layout`` the message is exactly
the ungrounded one (so nothing about the existing zero-shot path changes), and
with one the layout reaches the user turn as source-of-truth text. The
train/inference prompt-match rule the LoRA config enforces means the training
example has to carry the layout too.
"""

from src.model.prompts import (
    GROUNDED_INSTRUCTION,
    STRUCTURE_INSTRUCTION,
    build_messages,
    build_training_example,
    format_layout_block,
)

LAYOUT = "A | B\nC | D"


class TestBackwardCompatible:
    def test_no_layout_is_unchanged(self):
        msgs = build_messages("img.png", "structure")
        kinds = [c["type"] for c in msgs[0]["content"]]
        assert kinds == ["image", "text"]
        assert msgs[0]["content"][1]["text"] == STRUCTURE_INSTRUCTION


class TestGroundedMessage:
    def test_layout_keeps_image_text_shape(self):
        msgs = build_messages("img.png", instruction=GROUNDED_INSTRUCTION, ocr_layout=LAYOUT)
        assert [c["type"] for c in msgs[0]["content"]] == ["image", "text"]

    def test_layout_and_instruction_both_present(self):
        text = build_messages(
            "img.png", instruction=GROUNDED_INSTRUCTION, ocr_layout=LAYOUT
        )[0]["content"][1]["text"]
        assert GROUNDED_INSTRUCTION in text
        assert "<ocr_layout>" in text and "</ocr_layout>" in text
        assert LAYOUT in text

    def test_format_layout_block_delimits(self):
        block = format_layout_block(LAYOUT)
        assert block.strip().endswith("</ocr_layout>")
        assert LAYOUT in block

    def test_grounded_instruction_forbids_reading_pixels_for_text(self):
        low = GROUNDED_INSTRUCTION.lower()
        assert "ocr" in low
        assert "do not re-transcribe" in low


class TestGroundedTrainingExample:
    def test_layout_reaches_the_user_turn(self):
        ex = build_training_example(
            "img.png", "<table></table>", mode="full",
            instruction=GROUNDED_INSTRUCTION, ocr_layout=LAYOUT,
        )
        assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
        assert LAYOUT in ex["messages"][0]["content"][1]["text"]
        assert ex["messages"][1]["content"][0]["text"] == "<table></table>"
