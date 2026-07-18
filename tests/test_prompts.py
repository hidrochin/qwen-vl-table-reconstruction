"""Tests for prompt construction and model-output cleaning.

``clean_prediction`` sits between raw generation and every score, so a bug here
shows up as a bad TEDS number rather than as an error. Base VLMs reliably wrap
output in markdown fences and add commentary despite being told not to; each
case below is a shape real models actually emit.
"""

import pytest

from src.model.prompts import (
    FULL_INSTRUCTION,
    STRUCTURE_INSTRUCTION,
    build_messages,
    build_training_example,
    clean_prediction,
    resolve_instruction,
)

TABLE = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'


class TestCleanPrediction:
    def test_bare_html_passes_through(self):
        assert clean_prediction(TABLE) == TABLE

    def test_strips_markdown_fence(self):
        assert clean_prediction(f"```html\n{TABLE}\n```") == TABLE

    def test_strips_unlabelled_fence(self):
        assert clean_prediction(f"```\n{TABLE}\n```") == TABLE

    def test_strips_leading_prose(self):
        raw = f"Here is the reconstructed table:\n\n{TABLE}"
        assert clean_prediction(raw) == TABLE

    def test_strips_trailing_commentary(self):
        raw = f"{TABLE}\n\nNote that the first cell spans two rows."
        assert clean_prediction(raw) == TABLE

    def test_strips_prose_on_both_sides(self):
        raw = f"Sure! Here it is:\n```html\n{TABLE}\n```\nLet me know if you need changes."
        assert clean_prediction(raw) == TABLE

    def test_no_table_returns_empty(self):
        """Scored as 0 by the caller rather than guessed at."""
        assert clean_prediction("I cannot read this image.") == ""

    def test_empty_input_returns_empty(self):
        assert clean_prediction("") == ""
        assert clean_prediction("   \n  ") == ""

    def test_unterminated_table_is_kept(self):
        """Truncation at max_new_tokens is common; lxml recovers the partial
        tree, so keeping it scores better than discarding it."""
        truncated = '<table><tr><td rowspan="2">A</td><td>B</td></tr>'
        assert clean_prediction(truncated).startswith("<table")

    def test_uppercase_tags_are_found(self):
        assert clean_prediction("<TABLE><TR><TD>x</TD></TR></TABLE>").startswith("<TABLE")

    def test_prefers_the_fenced_block_containing_a_table(self):
        raw = "```python\nprint('hi')\n```\nand the table:\n```html\n" + TABLE + "\n```"
        assert clean_prediction(raw) == TABLE


class TestInstructionResolution:
    def test_defaults_by_mode(self):
        assert resolve_instruction("structure") == STRUCTURE_INSTRUCTION
        assert resolve_instruction("full") == FULL_INSTRUCTION

    def test_override_wins(self):
        assert resolve_instruction("structure", "CUSTOM") == "CUSTOM"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            resolve_instruction("nonsense")

    def test_override_bypasses_mode_validation(self):
        """Prompt iteration should not require inventing a mode name."""
        assert resolve_instruction("nonsense", "CUSTOM") == "CUSTOM"

    def test_structure_prompt_forbids_cell_text(self):
        """The training target has no text; the prompt must ask for none."""
        assert "no cell" in STRUCTURE_INSTRUCTION.lower()


class TestMessageShape:
    def test_user_turn_carries_image_then_text(self):
        msgs = build_messages("img.png", "structure")
        assert msgs[0]["role"] == "user"
        kinds = [c["type"] for c in msgs[0]["content"]]
        assert kinds == ["image", "text"]

    def test_override_reaches_the_message(self):
        """The Day 2 path: iterate prompts from the notebook without editing
        prompts.py, which an already-imported module would ignore anyway."""
        msgs = build_messages("img.png", instruction="CUSTOM PROMPT")
        assert msgs[0]["content"][1]["text"] == "CUSTOM PROMPT"

    def test_training_example_appends_assistant_turn(self):
        ex = build_training_example("img.png", TABLE)
        assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
        assert ex["messages"][1]["content"][0]["text"] == TABLE
