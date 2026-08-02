"""Client for a **locally served** vLLM endpoint (OpenAI-compatible, private).

This replaces the old hosted-API bake-off path. Nothing here calls a third-party
service: on the company server (Track 2) you serve a model once --

    vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --enable-lora --port 8000 \
        --no-enable-prefix-caching --limit-mm-per-prompt image=4

-- and hit ``http://localhost:8000/v1`` over the loopback. The 20 confidential
invoices never leave the network, and LoRA adapters hot-load via ``--enable-lora``
so base-vs-adapter A/B is possible against the same served weights.
``--no-enable-prefix-caching`` is for labelling runs: multimodal prefix-cache
hits have produced degraded repeat-call outputs on some vLLM versions, and a
labelling pass re-sends near-identical prompts constantly. ``image=4`` allows
multi-page tables (a list of page crops in one request).

It mirrors ``TableReconstructor``'s ``predict`` / ``predict_many`` surface and
returns the same ``Prediction``, so everything downstream -- TEDS-Struct, the new
schema metrics, the demo renderer -- is reused unchanged. Use it for the
persistent serving path; use ``TableReconstructor(backend="vllm")`` for one-shot
offline batch labelling.

Guided decoding (``guided_regex`` / ``guided_grammar`` / ``guided_json``) is
passed straight through to vLLM's ``extra_body`` -- the cheap way to force valid
HTML and stop the model shifting a value into a blank cell.

The ``openai`` / ``PIL`` imports are deferred, keeping this module importable on
the Mac with nothing installed.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from src.model.inference import Prediction, predictions_dict  # noqa: F401  (re-export)
from src.model.registry import MODEL_QWEN36_35B_FP8
from src.model.prompts import as_image_list, build_prompt_text, clean_prediction


def image_to_data_uri(image_path: str | Path, max_side: int | None = 1536) -> str:
    """Base64 data URI for the chat ``image_url`` field.

    ``max_side`` downscales the longer edge before encoding -- the served model
    still tokenizes by resolution, so capping it is the endpoint analogue of
    ``max_pixels``. Set ``None`` to send the image untouched. PIL is imported
    lazily so callers passing already-small images pay nothing.
    """
    path = Path(image_path)
    if max_side is not None:
        import io

        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            data = buf.getvalue()
        mime = "image/jpeg"
    else:
        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class VLLMTableReconstructor:
    """Generate table HTML through a locally served vLLM OpenAI endpoint.

    ``base_url`` points at your own server (default loopback). ``api_key`` is a
    dummy vLLM accepts; there is no real auth on a private box. ``model_id`` must
    match what you passed to ``vllm serve`` (or a hot-loaded LoRA adapter name).
    """

    def __init__(
        self,
        model_id: str = MODEL_QWEN36_35B_FP8,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_side: int | None = 1536,
        thinking: bool = False,
    ):
        self.model_id = model_id
        self.base_url = base_url
        self.max_side = max_side
        self.thinking = thinking
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            # Deferred: this module must import without the openai package.
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def predict(
        self,
        image_path: str | Path | list[str | Path],
        mode: str = "structure",
        max_new_tokens: int = 2048,
        instruction: str | None = None,
        ocr_layout: str | list[str] | None = None,
        guided_regex: str | None = None,
        guided_grammar: str | None = None,
        guided_json: dict | None = None,
    ) -> Prediction:
        """Generate HTML for one table -- one image, or a page list for a long
        table split across several crops (reading order; the multi-page
        stitching note is added automatically).

        Signature matches the local ``predict`` (``instruction`` overrides the
        mode prompt; ``ocr_layout`` supplies stage-1 grounding -- pass a list,
        one layout per page, with multi-image input) so a run script is agnostic
        to which backend it holds. The ``guided_*`` args force valid output via
        vLLM guided decoding.
        """
        paths = as_image_list(image_path)
        text = build_prompt_text(len(paths), mode, instruction, ocr_layout)

        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(p, self.max_side)},
            }
            for p in paths
        ]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        extra_body: dict = {}
        if guided_regex:
            extra_body["guided_regex"] = guided_regex
        if guided_grammar:
            extra_body["guided_grammar"] = guided_grammar
        if guided_json:
            extra_body["guided_json"] = guided_json
        # Qwen3.6 thinking toggle rides in extra_body too; harmless if unsupported.
        extra_body["chat_template_kwargs"] = {"enable_thinking": self.thinking}

        completion = self._get_client().chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,  # structure reconstruction has one right answer
            extra_body=extra_body or None,
        )
        raw = completion.choices[0].message.content or ""
        return Prediction(uid=Path(paths[0]).stem, raw=raw, html=clean_prediction(raw))

    def predict_many(
        self,
        image_paths: list,
        mode: str = "structure",
        ocr_layouts: list | None = None,
        **kwargs,
    ) -> list[Prediction]:
        """Sequential generation with progress, matching the local path.

        Each entry of ``image_paths`` is one *table*: a single path, or a list of
        page paths for a table split across crops. ``ocr_layouts`` is per-table
        and aligned to ``image_paths`` (itself a per-page list for a multi-page
        entry); ``**kwargs`` (``instruction``, ``guided_*``) is shared across all
        tables.
        """
        if ocr_layouts is not None and len(ocr_layouts) != len(image_paths):
            raise ValueError(
                f"ocr_layouts has {len(ocr_layouts)} entries, expected {len(image_paths)}"
            )
        preds = []
        for i, path in enumerate(image_paths, 1):
            per_image = {"ocr_layout": ocr_layouts[i - 1]} if ocr_layouts is not None else {}
            preds.append(self.predict(path, mode=mode, **per_image, **kwargs))
            if i % 10 == 0 or i == len(image_paths):
                print(f"  predicted {i}/{len(image_paths)}")
        return preds
