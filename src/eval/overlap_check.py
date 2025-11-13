"""
overlap_check.py
----------------
Evaluates the cleaned Sherlock Holmes dataset for training readiness.

Usage:
    python src/eval/overlap_check.py
"""

import json
from pathlib import Path
from collections import Counter

CLEAN_DIR = Path("data/cleaned")

MIN_WORDS = 30
MAX_WORDS = 300

def load_chunks():
    """Load all JSON chunk files and return a list of (id, text)."""
    all_chunks = []
    for file in CLEAN_DIR.glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                all_chunks.append((item["id"], item["text"]))
    return all_chunks


def basic_stats(chunks):
    """Compute and print basic word count statistics."""
    word_counts = [len(text.split()) for _, text in chunks]
    total_chunks = len(chunks)
    avg_words = sum(word_counts) / total_chunks
    min_words = min(word_counts)
    max_words = max(word_counts)

    print(f"\n📊 Dataset Statistics:")
    print(f"Total chunks      : {total_chunks}")
    print(f"Average words/chunk: {avg_words:.1f}")
    print(f"Min words/chunk   : {min_words}")
    print(f"Max words/chunk   : {max_words}")


def detect_duplicates(chunks, top_n=5):
    """Detect exact duplicate chunks (by text) in the dataset."""
    texts = [text for _, text in chunks]
    counter = Counter(texts)
    duplicates = [(text, count) for text, count in counter.items() if count > 1]

    if duplicates:
        print(f"\n⚠️ Found {len(duplicates)} duplicate chunks. Showing top {top_n}:")
        for text, count in duplicates[:top_n]:
            snippet = text[:100].replace("\n", " ")
            print(f"- Count {count}: {snippet}...")
    else:
        print("\n✅ No exact duplicate chunks found.")


def sample_chunks(chunks, n=5):
    """Print a few random sample chunks for manual inspection."""
    print(f"\n📚 Sample chunks:")
    for i, (_, text) in enumerate(chunks[:n]):
        snippet = text[:300].replace("\n", " ")
        print(f"{i+1}. {snippet}...\n")


if __name__ == "__main__":
    all_chunks = load_chunks()
    basic_stats(all_chunks)
    detect_duplicates(all_chunks)
    sample_chunks(all_chunks, n=5)
