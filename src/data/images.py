"""Image preparation for inference inputs.

The hand-drawn hard sample (``data/samples/IMG_2677.HEIC``) is a phone photo: it
is HEIC (which most model pipelines and chat APIs will not decode) and it is on
its side. Both have to be fixed before any model sees it, or the model spends its
capacity fighting the format instead of the table structure.

PIL only, imported lazily so the rest of ``src/`` still imports without it. HEIC
decoding needs ``pillow-heif`` if present; on a Mac we fall back to the built-in
``sips`` converter, which is always available there.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _heic_to_png(src: Path, dst: Path) -> Path:
    """Decode HEIC to PNG. Try pillow-heif, else macOS ``sips``."""
    try:
        import pillow_heif  # noqa: F401
        from PIL import Image

        Image.open(src).convert("RGB").save(dst, format="PNG")
        return dst
    except ImportError:
        # No pillow-heif -- fall back to sips (present on every Mac).
        subprocess.run(
            ["sips", "-s", "format", "png", str(src), "--out", str(dst)],
            check=True,
            capture_output=True,
        )
        return dst


def prepare_image(
    src: str | Path,
    out_dir: str | Path,
    rotate_deg: int = 0,
    max_side: int | None = None,
) -> Path:
    """Normalise ``src`` into an upright RGB PNG under ``out_dir``.

    ``rotate_deg`` is counter-clockwise (PIL convention). IMG_2677 is on its
    side; try ``rotate_deg=90`` and ``-90`` and keep whichever reads upright --
    the correct one is obvious by eye and cheap to flip. EXIF orientation is
    applied first so camera auto-rotation is honoured before the manual rotate.

    ``max_side`` optionally downscales the longer edge (kept off by default so
    this stays a faithful copy; the API path caps resolution separately).
    """
    from PIL import Image, ImageOps

    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work = src
    if src.suffix.lower() in {".heic", ".heif"}:
        work = _heic_to_png(src, out_dir / f"{src.stem}.png")

    im = ImageOps.exif_transpose(Image.open(work)).convert("RGB")
    if rotate_deg:
        im = im.rotate(rotate_deg, expand=True)
    if max_side is not None and max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)))

    dst = out_dir / f"{src.stem}_upright.png"
    im.save(dst, format="PNG")
    return dst
