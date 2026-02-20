"""
inverted index builder — assignment 3 m1
reads from developer.zip, builds a disk-based inverted index with partial offloading.
"""

import json
import os
import re
import time
import heapq
import zipfile
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

# configuration
CORPUS_ZIP = "/Users/robert/Downloads/developer.zip"
INDEX_DIR = Path("/Users/robert/a3/index")
BATCH_SIZE = 18_000  # docs per partial index (ensures >= 3 offloads for ~55k docs)

IMPORTANT_TAGS = {"title", "h1", "h2", "h3", "b", "strong"}

# globals
stemmer = PorterStemmer()
stem_cache: dict[str, str] = {}


def cached_stem(token: str) -> str:
    """porter-stem with memoisation."""
    s = stem_cache.get(token)
    if s is None:
        s = stemmer.stem(token)
        stem_cache[token] = s
    return s


def tokenize(text: str) -> list[str]:
    """return lowercased alphanumeric tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def parse_document(html: str):
    """
    returns (term_tf: dict[str,int], important_stems: set[str]).
    term_tf maps stemmed token -> raw term frequency.
    important_stems is the set of stems that appeared in important tags.
    """
    soup = BeautifulSoup(html, "lxml")

    # important stems
    important_stems: set[str] = set()
    for tag_name in IMPORTANT_TAGS:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(separator=" ", strip=True)
            for tok in tokenize(text):
                important_stems.add(cached_stem(tok))

    # full text tf
    full_text = soup.get_text(separator=" ", strip=True)
    term_tf: dict[str, int] = defaultdict(int)
    for tok in tokenize(full_text):
        term_tf[cached_stem(tok)] += 1

    return dict(term_tf), important_stems


def write_partial_index(partial_index: dict, partial_num: int) -> str:
    """write sorted partial index to disk. returns file path."""
    path = INDEX_DIR / f"partial_{partial_num}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(partial_index.keys()):
            postings = partial_index[term]
            # postings: list of (doc_id, tf, important)
            posting_strs = [f"{did}:{tf}:{imp}" for did, tf, imp in postings]
            f.write(f"{term}|{','.join(posting_strs)}\n")
    return str(path)


def parse_posting_line(line: str):
    """parse a line from a partial index file. returns (term, postings_str)."""
    sep = line.index("|")
    return line[:sep], line[sep + 1:]


def merge_partial_indexes(partial_paths: list[str]):
    """k-way merge of sorted partial index files into final index.txt.
    also builds index_of_index.txt with byte offsets."""

    index_path = INDEX_DIR / "index.txt"
    ioi_path = INDEX_DIR / "index_of_index.txt"
    unique_terms = 0

    # open all partial files
    file_handles = []
    heap = []  # (term, postings_str, file_index)

    for i, path in enumerate(partial_paths):
        fh = open(path, "r", encoding="utf-8")
        file_handles.append(fh)
        line = fh.readline()
        if line:
            term, postings = parse_posting_line(line.rstrip("\n"))
            heap.append((term, postings, i))

    heapq.heapify(heap)

    # write merged index in binary mode for accurate byte offsets
    with open(index_path, "wb") as idx_f, open(ioi_path, "w", encoding="utf-8") as ioi_f:
        current_term = None
        current_postings: list[str] = []

        def flush_term():
            nonlocal unique_terms
            if current_term is None:
                return
            offset = idx_f.tell()
            line_bytes = f"{current_term}|{','.join(current_postings)}\n".encode("utf-8")
            idx_f.write(line_bytes)
            ioi_f.write(f"{current_term}|{offset}\n")
            unique_terms += 1

        while heap:
            term, postings_str, fi = heapq.heappop(heap)

            # advance that file
            line = file_handles[fi].readline()
            if line:
                next_term, next_postings = parse_posting_line(line.rstrip("\n"))
                heapq.heappush(heap, (next_term, next_postings, fi))

            if term == current_term:
                current_postings.append(postings_str)
            else:
                flush_term()
                current_term = term
                current_postings = [postings_str]

        flush_term()  # last term

    for fh in file_handles:
        fh.close()

    return unique_terms


def main():
    start = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening corpus: {CORPUS_ZIP}")
    zf = zipfile.ZipFile(CORPUS_ZIP, "r")
    json_files = [n for n in zf.namelist() if n.endswith(".json")]
    total_docs = len(json_files)
    print(f"Found {total_docs} JSON files")

    doc_id_map: dict[int, str] = {}  # doc_id -> url
    partial_index: dict[str, list] = defaultdict(list)  # term -> [(doc_id, tf, imp)]
    partial_num = 0
    partial_paths: list[str] = []
    doc_id = 0

    for _, jf in enumerate(json_files):
        try:
            raw = zf.read(jf)
            data = json.loads(raw)
            url = data.get("url", "")
            content = data.get("content", "")
        except Exception as e:
            print(f"  SKIP {jf}: {e}")
            continue

        doc_id_map[doc_id] = url

        if content.strip():
            term_tf, important_stems = parse_document(content)
            for term, tf in term_tf.items():
                imp = 1 if term in important_stems else 0
                partial_index[term].append((doc_id, tf, imp))

        doc_id += 1

        # progress
        if doc_id % 5000 == 0:
            elapsed = time.time() - start
            print(f"  Processed {doc_id}/{total_docs} docs  ({elapsed:.1f}s)")

        # offload partial index
        if doc_id % BATCH_SIZE == 0:
            print(f"  >> Offloading partial index {partial_num} ({len(partial_index)} terms, doc {doc_id})")
            path = write_partial_index(partial_index, partial_num)
            partial_paths.append(path)
            partial_index = defaultdict(list)
            partial_num += 1

    # final batch
    if partial_index:
        print(f"  >> Offloading final partial index {partial_num} ({len(partial_index)} terms, doc {doc_id})")
        path = write_partial_index(partial_index, partial_num)
        partial_paths.append(path)
        partial_num += 1

    zf.close()
    parse_done = time.time()
    print(f"\nParsing complete: {doc_id} docs, {partial_num} partial indexes ({parse_done - start:.1f}s)")

    # merge
    print("Merging partial indexes...")
    unique_terms = merge_partial_indexes(partial_paths)
    merge_done = time.time()
    print(f"Merge complete: {unique_terms} unique terms ({merge_done - parse_done:.1f}s)")

    # write doc_id_map
    with open(INDEX_DIR / "doc_id_map.txt", "w", encoding="utf-8") as f:
        for did in range(len(doc_id_map)):
            f.write(f"{did}|{doc_id_map[did]}\n")

    # compute index size
    index_size_bytes = os.path.getsize(INDEX_DIR / "index.txt")
    index_size_kb = index_size_bytes / 1024

    # write metadata
    with open(INDEX_DIR / "metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"documents: {doc_id}\n")
        f.write(f"unique_terms: {unique_terms}\n")
        f.write(f"index_size_kb: {index_size_kb:.1f}\n")
        f.write(f"partial_indexes: {partial_num}\n")
        f.write(f"total_time_s: {time.time() - start:.1f}\n")

    # clean up partial files
    for path in partial_paths:
        os.remove(path)

    total_time = time.time() - start
    print(f"\n{'='*50}")
    print(f"  Documents indexed : {doc_id}")
    print(f"  Unique terms      : {unique_terms}")
    print(f"  Index size on disk: {index_size_kb:.1f} KB ({index_size_kb/1024:.1f} MB)")
    print(f"  Partial offloads  : {partial_num}")
    print(f"  Total time        : {total_time:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
