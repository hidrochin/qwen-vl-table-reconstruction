"""The registry is the confidential-boundary gate. A wrong flag here could route
a private teacher into a Track-1 (cloud) driver, which is the one mistake the
whole two-track split exists to prevent -- so the gate is tested directly.
"""

import pytest

from src.model import registry as reg


class TestDataBoundary:
    def test_public_excludes_private_teachers(self):
        public = reg.models_for_data("public")
        assert "qwen3.6-27b" not in public and "qwen3.6-35b-a3b" not in public

    def test_private_excludes_public_anchor(self):
        private = reg.models_for_data("private")
        assert "qwen3-vl-30b-a3b" not in private

    def test_both_models_appear_on_each_side(self):
        for data in ("public", "private"):
            names = reg.models_for_data(data)
            assert "qwen3-vl-8b" in names  # data="both" student
            assert "mineru-2.5-pro" in names  # data="both" specialist

    def test_no_private_model_leaks_into_public(self):
        for name, spec in reg.models_for_data("public").items():
            assert spec.data in ("public", "both"), f"{name} leaked into public"

    def test_bad_data_raises(self):
        with pytest.raises(ValueError):
            reg.models_for_data("cloud")


class TestRoleSets:
    def test_generative_public_is_the_bakeoff_set(self):
        gen = reg.generative_models("public")
        assert set(gen) == {"qwen3-vl-30b-a3b", "qwen3-vl-8b", "qwen3-vl-4b"}

    def test_specialists_are_all_specialist_backend(self):
        assert all(s.backend == "specialist" for s in reg.specialist_models().values())

    def test_trainable_includes_teacher_and_students_not_anchor(self):
        trainable = reg.trainable_models()
        assert "qwen3.6-27b" in trainable and "qwen3-vl-4b" in trainable
        assert "qwen3-vl-30b-a3b" not in trainable  # anchor is not fine-tuned

    def test_no_hosted_api_models(self):
        """Kimi / frontier APIs were removed -- everything must be self-hostable."""
        joined = " ".join(s.repo_id.lower() for s in reg.MODELS.values())
        for banned in ("kimi", "gpt-", "gemini", "claude"):
            assert banned not in joined
