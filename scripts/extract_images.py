"""
Extract main tables from arXiv papers referenced in the thesis
using the unstructured library.

Usage:
    python extract_images.py [--paper SLUG] [--timeout SECS]
"""

import argparse
import json
import logging
import os
import re
import signal
import tempfile
import time
from collections import Counter
from pathlib import Path

import requests
from unstructured.partition.pdf import partition_pdf

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = PROJECT_DIR / "references"
DATA_DIR = PROJECT_DIR / "data"
PROGRESS_FILE = SCRIPT_DIR / "extraction_progress.json"

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
DOWNLOAD_DELAY = 3
REQUEST_TIMEOUT = 120
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (thesis-research-image-extraction)"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class PaperTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PaperTimeout("Processing exceeded time limit")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def parse_reference(md_path: Path) -> dict | None:
    text = md_path.read_text()
    match = re.search(r"\*\*arXiv ID:\*\*\s*(\S+)", text)
    if not match:
        return None
    arxiv_id = match.group(1)
    if arxiv_id == "N/A":
        return None
    return {
        "arxiv_id": arxiv_id,
        "slug": md_path.stem,
        "title": text.split("\n")[0].lstrip("# ").strip(),
    }


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("text/html"):
            log.warning("Got HTML instead of PDF for %s (rate-limited?)", arxiv_id)
            return False
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        log.error("Failed to download %s: %s", arxiv_id, e)
        return False


def extract_tables(pdf_path: Path, output_dir: Path) -> int:
    """Extract only Table elements as images from a PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_image_block_types=["Table"],
        extract_image_block_output_dir=str(output_dir),
    )

    type_counts = Counter(type(e).__name__ for e in elements)
    n_tables = type_counts.get("Table", 0)
    total_pages = max(
        (e.metadata.page_number for e in elements if hasattr(e.metadata, "page_number") and e.metadata.page_number),
        default=0,
    )

    log.info("  %d elements, %d tables, %d pages", len(elements), n_tables, total_pages)

    saved = list(output_dir.glob("table-*.jpg")) + list(output_dir.glob("table-*.png"))

    # Remove any non-table images that might have been saved
    for f in output_dir.glob("figure-*"):
        f.unlink()

    return len(saved)


def process_paper(ref: dict, progress: dict, timeout: int) -> bool:
    slug = ref["slug"]
    arxiv_id = ref["arxiv_id"]
    output_dir = DATA_DIR / slug

    if progress.get(slug, {}).get("status") == "done":
        log.info("Skipping %s (already done)", slug)
        return True

    log.info("Processing: %s (arXiv: %s)", ref["title"], arxiv_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        if not download_pdf(arxiv_id, pdf_path):
            progress[slug] = {"status": "download_failed", "arxiv_id": arxiv_id}
            save_progress(progress)
            return False

        size_mb = pdf_path.stat().st_size / 1e6
        log.info("  Downloaded PDF (%.1f MB)", size_mb)

        num_tables = extract_tables(pdf_path, output_dir)
        log.info("  Saved %d table images -> %s", num_tables, output_dir.name)

        if num_tables == 0:
            if not any(output_dir.iterdir()):
                output_dir.rmdir()

        progress[slug] = {
            "status": "done",
            "arxiv_id": arxiv_id,
            "tables": num_tables,
        }
        save_progress(progress)
        return True

    except PaperTimeout:
        log.error("  TIMEOUT after %ds for %s — skipping", timeout, slug)
        progress[slug] = {"status": "timeout", "arxiv_id": arxiv_id}
        save_progress(progress)
        # Clean up partial output
        if output_dir.exists():
            for f in output_dir.iterdir():
                f.unlink()
            output_dir.rmdir()
        return False

    except Exception as e:
        log.error("  Failed to process %s: %s", slug, e)
        progress[slug] = {"status": "error", "arxiv_id": arxiv_id, "error": str(e)}
        save_progress(progress)
        return False

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        pdf_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Extract tables from arXiv papers")
    parser.add_argument("--paper", type=str, default=None,
                        help="Process only a specific paper by slug")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Max seconds per paper (default: 600)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(REFERENCES_DIR.glob("*.md"))
    log.info("Found %d reference files", len(md_files))

    refs = []
    for md in md_files:
        ref = parse_reference(md)
        if ref:
            refs.append(ref)
        else:
            log.warning("Skipping %s (no valid arXiv ID)", md.name)
    log.info("Parsed %d papers with valid arXiv IDs", len(refs))

    if args.paper:
        refs = [r for r in refs if r["slug"] == args.paper]
        if not refs:
            log.error("Paper '%s' not found", args.paper)
            return

    progress = load_progress()
    done, failed = 0, 0

    for i, ref in enumerate(refs, 1):
        log.info("=== [%d/%d] %s ===", i, len(refs), ref["slug"])

        if progress.get(ref["slug"], {}).get("status") == "done":
            log.info("Skipping (already done)")
            done += 1
            continue

        success = process_paper(ref, progress, args.timeout)
        if success:
            done += 1
        else:
            failed += 1

        if i < len(refs):
            time.sleep(DOWNLOAD_DELAY)

    log.info("=== COMPLETE: %d done, %d failed out of %d ===", done, failed, len(refs))

    failed_papers = {k: v for k, v in progress.items() if v.get("status") != "done"}
    if failed_papers:
        log.info("Failed papers:")
        for slug, info in failed_papers.items():
            log.info("  %s: %s", slug, info.get("error", info.get("status")))


if __name__ == "__main__":
    main()
