"""
search engine retrieval component — assignment 3 m2
boolean AND retrieval with tf-idf ranking over disk-based inverted index.
"""

import math
import re
import time
from pathlib import Path

from nltk.stem import PorterStemmer

# configuration
INDEX_DIR = Path("index")
IMPORTANCE_BOOST = 1.5  # multiplier for terms found in important tags

# globals
stemmer = PorterStemmer()
stem_cache: dict[str, str] = {}


def cached_stem(token: str) -> str:
    s = stem_cache.get(token)
    if s is None:
        s = stemmer.stem(token)
        stem_cache[token] = s
    return s


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_index_of_index() -> dict[str, int]:
    """load term -> byte offset mapping into memory."""
    ioi = {}
    with open(INDEX_DIR / "index_of_index.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            term = line[:sep]
            offset = int(line[sep + 1:])
            ioi[term] = offset
    return ioi


def load_doc_id_map() -> dict[int, str]:
    """load doc_id -> url mapping."""
    doc_map = {}
    with open(INDEX_DIR / "doc_id_map.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            did = int(line[:sep])
            url = line[sep + 1:]
            doc_map[did] = url
    return doc_map


def load_total_docs() -> int:
    with open(INDEX_DIR / "metadata.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("documents:"):
                return int(line.split(":")[1].strip())
    return 0


def fetch_postings(term: str, ioi: dict[str, int], index_fh) -> list[tuple[int, int, int]]:
    """seek into index.txt and read postings for a term.
    returns list of (doc_id, tf, important)."""
    offset = ioi.get(term)
    if offset is None:
        return []

    index_fh.seek(offset)
    line = index_fh.readline().decode("utf-8").rstrip("\n")
    sep = line.index("|")
    postings_str = line[sep + 1:]

    postings = []
    for entry in postings_str.split(","):
        parts = entry.split(":")
        doc_id = int(parts[0])
        tf = int(parts[1])
        imp = int(parts[2])
        postings.append((doc_id, tf, imp))
    return postings


def boolean_and_search(query: str, ioi: dict[str, int], doc_map: dict[int, str],
                       total_docs: int, index_fh) -> list[tuple[str, float]]:
    """perform AND retrieval with tf-idf ranking.
    returns list of (url, score) sorted by descending score."""
    tokens = tokenize(query)
    if not tokens:
        return []

    stems = [cached_stem(t) for t in tokens]
    # deduplicate while preserving order
    seen = set()
    unique_stems = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            unique_stems.append(s)

    # fetch postings for each term
    term_postings: list[dict[int, tuple[int, int]]] = []  # [{doc_id: (tf, imp)}]
    term_idfs: list[float] = []

    for stem in unique_stems:
        postings = fetch_postings(stem, ioi, index_fh)
        if not postings:
            return []  # AND semantics: if any term missing, no results
        pmap = {doc_id: (tf, imp) for doc_id, tf, imp in postings}
        term_postings.append(pmap)
        df = len(postings)
        idf = math.log(total_docs / df)
        term_idfs.append(idf)

    # intersect: start with smallest posting list
    order = sorted(range(len(term_postings)), key=lambda i: len(term_postings[i]))
    candidate_docs = set(term_postings[order[0]].keys())
    for i in order[1:]:
        candidate_docs &= term_postings[i].keys()
        if not candidate_docs:
            return []

    # score each candidate
    scored = []
    for doc_id in candidate_docs:
        score = 0.0
        for i, stem in enumerate(unique_stems):
            tf, imp = term_postings[i][doc_id]
            tf_weight = 1 + math.log(tf) if tf > 0 else 0
            idf = term_idfs[i]
            term_score = tf_weight * idf
            if imp:
                term_score *= IMPORTANCE_BOOST
            score += term_score
        url = doc_map.get(doc_id, f"doc_{doc_id}")
        scored.append((url, score))

    scored.sort(key=lambda x: -x[1])
    return scored


def main():
    print("Loading index metadata...")
    t0 = time.time()
    ioi = load_index_of_index()
    doc_map = load_doc_id_map()
    total_docs = load_total_docs()
    index_fh = open(INDEX_DIR / "index.txt", "rb")
    t1 = time.time()
    print(f"Ready. {len(ioi)} terms, {total_docs} docs loaded in {t1 - t0:.2f}s\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        t_start = time.time()
        results = boolean_and_search(query, ioi, doc_map, total_docs, index_fh)
        t_end = time.time()

        elapsed_ms = (t_end - t_start) * 1000
        print(f"  [{len(results)} results in {elapsed_ms:.1f}ms]")

        if not results:
            print("  No results found.\n")
            continue

        for rank, (url, score) in enumerate(results[:5], 1):
            print(f"  {rank}. {url}  (score: {score:.4f})")
        print()

    index_fh.close()


if __name__ == "__main__":
    main()
