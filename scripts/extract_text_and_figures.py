"""
Extract full text and all images (tables + figures) from specific arXiv papers
using the unstructured library.

Usage:
    python extract_text_and_figures.py
    python extract_text_and_figures.py --paper gigachat-family-efficient-russian-language-modeling-through-mixture-of-experts-architecture
"""

import argparse
import logging
import tempfile
from collections import Counter
from pathlib import Path

import requests
from unstructured.partition.pdf import partition_pdf

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
REQUEST_TIMEOUT = 120
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (thesis-research-extraction)"}

PAPERS = [
    {
        "arxiv_id": "2506.09440",
        "slug": "gigachat-family-efficient-russian-language-modeling-through-mixture-of-experts-architecture",
        "title": "GigaChat Family",
    },
    {
        "arxiv_id": "2510.22369",
        "slug": "gigaembeddings-efficient-russian-language-embedding-model",
        "title": "GigaEmbeddings",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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


def extract_all(pdf_path: Path, output_dir: Path) -> list:
    """Extract text elements and save all images (tables + figures)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=str(output_dir),
    )

    type_counts = Counter(type(e).__name__ for e in elements)
    log.info("  Element types: %s", dict(type_counts))

    return elements


def elements_to_markdown(elements: list) -> str:
    """Convert unstructured elements into a markdown document."""
    lines = []
    current_page = None

    for el in elements:
        page = getattr(el.metadata, "page_number", None)
        if page and page != current_page:
            current_page = page
            lines.append(f"\n---\n**[Page {page}]**\n")

        el_type = type(el).__name__
        text = str(el).strip()

        if not text:
            continue

        if el_type == "Title":
            lines.append(f"\n## {text}\n")
        elif el_type == "Header":
            lines.append(f"\n### {text}\n")
        elif el_type in ("NarrativeText", "UncategorizedText"):
            lines.append(f"\n{text}\n")
        elif el_type == "ListItem":
            lines.append(f"- {text}")
        elif el_type == "Table":
            lines.append(f"\n**[Table]**\n{text}\n")
        elif el_type == "Image":
            lines.append(f"\n**[Figure]** {text}\n")
        elif el_type == "Formula":
            lines.append(f"\n$$\n{text}\n$$\n")
        elif el_type == "FigureCaption":
            lines.append(f"\n*Caption: {text}*\n")
        else:
            lines.append(f"\n{text}\n")

    return "\n".join(lines)


def process_paper(paper: dict):
    slug = paper["slug"]
    arxiv_id = paper["arxiv_id"]
    output_dir = DATA_DIR / slug

    log.info("Processing: %s (arXiv: %s)", paper["title"], arxiv_id)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)

    try:
        if not download_pdf(arxiv_id, pdf_path):
            return

        size_mb = pdf_path.stat().st_size / 1e6
        log.info("  Downloaded PDF (%.1f MB)", size_mb)

        elements = extract_all(pdf_path, output_dir)

        md_text = elements_to_markdown(elements)
        md_path = output_dir / "extracted_text.md"
        md_path.write_text(md_text, encoding="utf-8")
        log.info("  Saved extracted text -> %s", md_path)

        n_images = len(list(output_dir.glob("figure-*.jpg")) + list(output_dir.glob("figure-*.png")))
        n_tables = len(list(output_dir.glob("table-*.jpg")) + list(output_dir.glob("table-*.png")))
        log.info("  Images: %d figures, %d tables in %s", n_images, n_tables, output_dir.name)

    except Exception as e:
        log.error("  Failed to process %s: %s", slug, e)
    finally:
        pdf_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Extract text and images from arXiv papers")
    parser.add_argument("--paper", type=str, default=None, help="Process only a specific paper by slug")
    args = parser.parse_args()

    papers = PAPERS
    if args.paper:
        papers = [p for p in papers if p["slug"] == args.paper]
        if not papers:
            log.error("Paper '%s' not found", args.paper)
            return

    for paper in papers:
        process_paper(paper)


if __name__ == "__main__":
    main()
