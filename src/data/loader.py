"""Build a hard-table corpus from FinTabNet without downloading the bulk dataset.

FinTabNet is financial tables from S&P 500 annual reports: borderless-heavy,
hierarchical headers, closer to invoice layouts than PubTabNet's biomedical
tables. Schema is ``{image: Image, html_table: str}`` under config ``en``.

The obvious approach -- ``load_dataset(..., streaming=True)`` -- pulls 10.28 GB
of parquet for the train split and stalls on a home connection. Difficulty can
be scored from the HTML text alone, so this module instead pages the
datasets-server ``/rows`` API (~100 rows per 2s, images returned as URLs) and
downloads image bytes only for the tables that pass the filter. That turns
~10 GB into tens of MB.

Train and eval come from the dataset's own ``train`` and ``validation`` splits,
so they cannot overlap by construction rather than by trusting a split function.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

from src.data.difficulty import (
    TableComplexity,
    assign_bins_by_percentile,
    score_table,
    summarize,
)

DATASET_ID = "apoidea/fintabnet-html"
CONFIG = "en"

# Biomedical rather than financial, same {image, html_table} shape. Fallback
# only -- financial tables are far more persuasive to a finance customer.
FALLBACK_DATASET_ID = "apoidea/pubtabnet-html"

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # API maximum
_TIMEOUT = 60


@dataclass
class TableRecord:
    uid: str
    image_path: str
    html: str
    complexity: TableComplexity

    def to_json(self) -> dict:
        return {
            "uid": self.uid,
            "image_path": self.image_path,
            "html": self.html,
            "complexity": self.complexity.as_dict(),
        }


def _get(url: str, params: dict | None = None, retries: int = 4):
    """GET with exponential backoff. The API rate-limits under sustained paging."""
    delay = 1.0
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 502, 503, 504):
                last = f"HTTP {resp.status_code}"
            else:
                resp.raise_for_status()
        except requests.RequestException as exc:
            last = str(exc)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"request failed after {retries} attempts ({last}): {url}")


def _fetch_page(dataset_id: str, split: str, offset: int, length: int) -> list[dict]:
    resp = _get(
        ROWS_API,
        {
            "dataset": dataset_id,
            "config": CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        },
    )
    return resp.json().get("rows", [])


def _image_url(row: dict) -> str | None:
    image = row.get("image")
    return image.get("src") if isinstance(image, dict) else None


def select_hard_tables(
    split: str,
    n_target: int,
    max_scanned: int = 20000,
    dataset_id: str = DATASET_ID,
    require_spanning: bool = True,
    progress_every: int = 1000,
) -> tuple[list[dict], list[float]]:
    """Scan a split and return the ``n_target`` hardest tables, plus all scores.

    Ranks by score and takes the top N rather than filtering on an absolute
    threshold. A fixed cutoff has to be guessed per dataset and guesses badly:
    0.55 matched 1 table in 3,000 on FinTabNet. Top-N always returns a full
    corpus and always returns the hardest available.

    Only HTML is scored here -- no image bytes are fetched -- which is what makes
    scanning tens of thousands of rows affordable.
    """
    candidates: list[dict] = []
    all_scores: list[float] = []
    seen_html: set[str] = set()
    n_duplicates = 0
    offset = 0

    while offset < max_scanned:
        rows = _fetch_page(dataset_id, split, offset, min(PAGE_SIZE, max_scanned - offset))
        if not rows:
            break

        for row in rows:
            data = row.get("row", {})
            html = data.get("html_table") or ""
            complexity = score_table(html)
            all_scores.append(complexity.score)

            if complexity.bin == "empty":
                continue
            # Merged cells are the capability under test; a corpus without them
            # would not stress the model regardless of how it scored otherwise.
            if require_spanning and complexity.n_spanning == 0:
                continue
            # FinTabNet repeats tables (~25% of one sampled eval selection).
            # Duplicates would weight a handful of layouts disproportionately in
            # a small eval set.
            key = " ".join(html.split())
            if key in seen_html:
                n_duplicates += 1
                continue
            seen_html.add(key)

            url = _image_url(data)
            if url:
                candidates.append(
                    {"html": html, "complexity": complexity, "image_url": url}
                )

        offset += len(rows)
        if progress_every and offset % progress_every < PAGE_SIZE:
            print(f"  scanned {offset:,}, {len(candidates)} candidates")

    candidates.sort(key=lambda c: c["complexity"].score, reverse=True)
    selected = candidates[:n_target]

    print(
        f"  scanned {offset:,} -> {len(candidates)} unique candidates "
        f"({n_duplicates} duplicates dropped) -> kept {len(selected)}"
    )
    if selected:
        lo = selected[-1]["complexity"].score
        hi = selected[0]["complexity"].score
        print(f"  selected score range: {lo:.3f} - {hi:.3f}")
    if len(selected) < n_target:
        print(
            f"  WARNING: wanted {n_target}, got {len(selected)}. "
            f"Raise max_scanned (now {max_scanned:,}) or set require_spanning=False."
        )
    return selected, all_scores


def _download_image(url: str, dest: Path) -> bool:
    try:
        resp = _get(url, retries=3)
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:  # noqa: BLE001 - one bad image must not kill the build
        print(f"  image download failed ({exc}): {dest.name}")
        return False


def build_split(
    split_name: str,
    source_split: str,
    n_target: int,
    out_dir: Path,
    max_scanned: int = 20000,
    dataset_id: str = DATASET_ID,
    workers: int = 8,
) -> list[TableRecord]:
    """Select hard tables, download their images concurrently, write a manifest."""
    out_dir = Path(out_dir) / split_name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{split_name}] scanning {dataset_id}:{source_split}")
    selected, all_scores = select_hard_tables(
        split=source_split,
        n_target=n_target,
        max_scanned=max_scanned,
        dataset_id=dataset_id,
    )
    if all_scores:
        print(f"  corpus-wide distribution: {summarize(all_scores)}")

    # Bin within the selection, so "hard" means hardest-of-the-selected rather
    # than a threshold tuned to a different dataset's conventions.
    rebinned = assign_bins_by_percentile([item["complexity"] for item in selected])
    for item, complexity in zip(selected, rebinned):
        item["complexity"] = complexity

    print(f"[{split_name}] downloading {len(selected)} images ({workers} workers)")
    records: list[TableRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, item in enumerate(selected):
            uid = f"{split_name}_{i:05d}"
            dest = images_dir / f"{uid}.jpg"
            futures[pool.submit(_download_image, item["image_url"], dest)] = (uid, dest, item)

        for future in as_completed(futures):
            uid, dest, item = futures[future]
            if future.result():
                records.append(
                    TableRecord(
                        uid=uid,
                        image_path=f"images/{dest.name}",
                        html=item["html"],
                        complexity=item["complexity"],
                    )
                )

    records.sort(key=lambda r: r.uid)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json()) + "\n")

    mb = sum(p.stat().st_size for p in images_dir.glob("*.jpg")) / 1e6
    print(f"[{split_name}] {len(records)} records, {mb:.1f} MB images -> {manifest}\n")
    return records


def build_corpus(
    out_dir: Path,
    n_train: int = 500,
    n_eval: int = 100,
    max_scanned: int = 20000,
    dataset_id: str = DATASET_ID,
) -> dict[str, int]:
    """Build both splits. Eval draws from `validation`, so it cannot overlap train."""
    out_dir = Path(out_dir)
    counts = {
        "train": len(
            build_split("train", "train", n_train, out_dir, max_scanned, dataset_id)
        ),
        "eval": len(
            build_split("eval", "validation", n_eval, out_dir, max_scanned, dataset_id)
        ),
    }
    assert_no_leakage(out_dir / "train", out_dir / "eval")
    return counts


def load_manifest(split_dir: Path) -> list[TableRecord]:
    """Read back a built split, resolving image paths to absolute."""
    split_dir = Path(split_dir)
    manifest = split_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"no manifest at {manifest} -- run build_corpus first")

    records = []
    with manifest.open() as f:
        for line in f:
            d = json.loads(line)
            records.append(
                TableRecord(
                    uid=d["uid"],
                    image_path=str(split_dir / d["image_path"]),
                    html=d["html"],
                    complexity=TableComplexity(**d["complexity"]),
                )
            )
    return records


def assert_no_leakage(train_dir: Path, eval_dir: Path) -> None:
    """Fail loudly if a table appears in both splits.

    Separate source splits should make this impossible, but a customer noticing
    inflated numbers costs more than the check does.
    """
    train_html = {r.html.strip() for r in load_manifest(train_dir)}
    eval_html = {r.html.strip() for r in load_manifest(eval_dir)}
    overlap = train_html & eval_html
    if overlap:
        raise AssertionError(f"{len(overlap)} tables appear in BOTH train and eval")
    print(f"leakage check passed: {len(train_html)} train / {len(eval_html)} eval, 0 overlap")
