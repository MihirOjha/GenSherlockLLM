"""
preprocess.py
-------------
Downloads and cleans the full Sherlock Holmes corpus from Project Gutenberg.
Then splits it into manageable text chunks ready for fine-tuning.

Usage:
    python src/data_prep/preprocess.py
"""

import re
import json
import requests
from pathlib import Path

# -------------------- Directory Setup --------------------
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
MANIFEST_DIR = Path("data/manifests")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Project Gutenberg Sources --------------------
BOOKS = {
    "study_in_scarlet": "https://www.gutenberg.org/cache/epub/244/pg244.txt",
    "sign_of_four": "https://www.gutenberg.org/cache/epub/2097/pg2097.txt",
    "hound_baskervilles": "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
    "valley_of_fear": "https://www.gutenberg.org/cache/epub/3289/pg3289.txt",
    "adventures": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "memoirs": "https://www.gutenberg.org/cache/epub/834/pg834.txt",
    "return": "https://www.gutenberg.org/cache/epub/221/pg221.txt",
    "his_last_bow": "https://www.gutenberg.org/cache/epub/2350/pg2350.txt",
    "casebook": "https://www.gutenberg.org/cache/epub/69700/pg69700.txt",
}


# -------------------- Step 1: Download Books --------------------
def download_books():
    """Download all Sherlock Holmes books into data/raw if not already present."""
    for name, url in BOOKS.items():
        dest = RAW_DIR / f"{name}.txt"
        if dest.exists():
            print(f"✔ Already downloaded: {name}")
            continue
        print(f"⬇ Downloading {name} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_text(r.text, encoding="utf-8")
    print("✅ All books downloaded.\n")


# -------------------- Step 2: Clean Gutenberg Text --------------------
def clean_gutenberg(text: str) -> str:
    """
    Extract only the main story text between Gutenberg START/END markers.
    Normalize whitespace and remove boilerplate, underscores, etc.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # More flexible regex to handle various Gutenberg marker formats
    start_pattern = r"\*\*\*\s*START\s+OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK.*?\*\*\*"
    end_pattern = r"\*\*\*\s*END\s+OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK.*?\*\*\*"
    
    # Find start and end positions
    start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
    end_match = re.search(end_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if start_match and end_match:
        # Extract text between the markers
        start_pos = start_match.end()
        end_pos = end_match.start()
        story_text = text[start_pos:end_pos].strip()
        print(f"✅ Found markers, extracted {len(story_text)} characters")
    else:
        print("⚠️ Could not find explicit START/END markers, using middle 80% fallback.")
        n = len(text)
        story_text = text[int(0.1 * n): int(0.9 * n)]

    # Remove underscores, normalize spacing and punctuation
    story_text = re.sub(r"_+", "", story_text)
    story_text = re.sub(r"\s+", " ", story_text)
    story_text = story_text.strip()

    # Optional: remove table of contents sections
    story_text = re.sub(r"(?i)contents\s+I\.", "", story_text)

    return story_text


# -------------------- Step 3: Chunking --------------------
def chunk_text(text: str, max_words: int = 200, min_words: int = 30, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping chunks of roughly max_words words.
    Overlap helps preserve context continuity between chunks.
    """

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
        start += max_words - overlap  # step forward with overlap

    return chunks


# -------------------- Step 4: Process All Books --------------------
def process_books():
    manifest = []
    for path in RAW_DIR.glob("*.txt"):
        print(f"🧹 Cleaning {path.name} ...")
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_gutenberg(raw_text)
        chunks = chunk_text(cleaned)

        out_path = CLEAN_DIR / f"{path.stem}.json"
        json_data = [{"id": f"{path.stem}_{i}", "text": chunk} for i, chunk in enumerate(chunks)]
        out_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest.append({"book": path.stem, "chunks": len(chunks)})
        print(f"✅ {path.stem}: {len(chunks)} chunks saved.")

    # Save manifest
    manifest_path = MANIFEST_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n📄 Manifest saved to {manifest_path}")


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    download_books()
    process_books()
    print("\n🏁 Preprocessing complete!")