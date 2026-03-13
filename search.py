"""
search engine retrieval — assignment 3 m3
BM25 ranking with tiered importance, OR fallback, dedup filtering,
bigram boost, word position proximity, PageRank, URL quality signal,
and LRU postings cache.
"""

import math
import re
import time
from collections import OrderedDict
from pathlib import Path

from nltk.stem import PorterStemmer

# configuration
INDEX_DIR = Path("index")

# BM25 parameters
BM25_K1 = 1.2
BM25_B = 0.75

# tiered importance multipliers
TIER_MULTIPLIERS = {
    3: 3.0,   # title
    2: 2.0,   # h1
    1: 1.5,   # h2, h3, b, strong
    0: 1.0,   # body
}

# OR fallback threshold
AND_MIN_RESULTS = 5
AND_BONUS = 1.5

# scoring weights
PAGERANK_WEIGHT = 500.0
PROXIMITY_WEIGHT = 2.0

# LRU cache size
CACHE_MAX = 1000

# globals
stemmer = PorterStemmer()
stem_cache: dict[str, str] = {}


def cached_stem(token: str) -> str:
    """stem a token with memoization."""
    s = stem_cache.get(token)
    if s is None:
        s = stemmer.stem(token)
        stem_cache[token] = s
    return s


def tokenize(text: str) -> list[str]:
    """split text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_index_of_index() -> dict[str, int]:
    """load term -> byte offset map for disk-based index seeks."""
    ioi = {}
    with open(INDEX_DIR / "index_of_index.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            ioi[line[:sep]] = int(line[sep + 1:])
    return ioi


def load_doc_id_map() -> dict[int, str]:
    """load doc_id -> url mapping."""
    doc_map = {}
    with open(INDEX_DIR / "doc_id_map.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            doc_map[int(line[:sep])] = line[sep + 1:]
    return doc_map


def load_doc_lengths() -> dict[int, int]:
    """load doc_id -> token count for BM25 length normalization."""
    lengths = {}
    path = INDEX_DIR / "doc_lengths.txt"
    if not path.exists():
        return lengths
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            lengths[int(line[:sep])] = int(line[sep + 1:])
    return lengths


def load_duplicates() -> dict[int, int]:
    """load dup_doc_id -> canonical_doc_id map for dedup filtering."""
    dups = {}
    path = INDEX_DIR / "duplicates.txt"
    if not path.exists():
        return dups
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            dups[int(line[:sep])] = int(line[sep + 1:])
    return dups


def load_pagerank() -> dict[int, float]:
    """load doc_id -> pagerank score."""
    pr = {}
    path = INDEX_DIR / "pagerank.txt"
    if not path.exists():
        return pr
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            sep = line.index("|")
            pr[int(line[:sep])] = float(line[sep + 1:])
    return pr


def load_total_docs() -> int:
    """read total document count from metadata."""
    with open(INDEX_DIR / "metadata.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("documents:"):
                return int(line.split(":")[1].strip())
    return 0


class PostingsCache:
    """LRU cache for postings lookups to avoid redundant disk seeks."""

    def __init__(self, maxsize: int = CACHE_MAX):
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._maxsize = maxsize

    def get(self, term: str):
        if term in self._cache:
            self._cache.move_to_end(term)
            return self._cache[term]
        return None

    def put(self, term: str, postings: list):
        if term in self._cache:
            self._cache.move_to_end(term)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[term] = postings


def fetch_postings(term: str, ioi: dict[str, int], index_fh,
                   cache: PostingsCache) -> list[tuple[int, int, int, list[int]]]:
    """fetch postings for a term. returns list of (doc_id, tf, tier, positions)."""
    cached = cache.get(term)
    if cached is not None:
        return cached

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
        tier = int(parts[2])
        positions = []
        if len(parts) > 3 and parts[3]:
            positions = [int(p) for p in parts[3].split(".")]
        postings.append((doc_id, tf, tier, positions))

    cache.put(term, postings)
    return postings


def min_window_span(pos_lists: list[list[int]]) -> int:
    """find minimum window span containing at least one position from each list."""
    if not pos_lists or any(not pl for pl in pos_lists):
        return -1

    pointers = [0] * len(pos_lists)
    min_span = float('inf')

    while True:
        positions = [pos_lists[i][pointers[i]] for i in range(len(pos_lists))]
        span = max(positions) - min(positions)
        min_span = min(min_span, span)
        if min_span == 0:
            return 0
        min_idx = 0
        min_val = positions[0]
        for i in range(1, len(positions)):
            if positions[i] < min_val:
                min_val = positions[i]
                min_idx = i
        pointers[min_idx] += 1
        if pointers[min_idx] >= len(pos_lists[min_idx]):
            break

    return min_span if min_span != float('inf') else -1


def url_quality_score(url: str) -> float:
    """small additive bonus for URL quality signals."""
    bonus = 0.0
    path = url.split("//", 1)[-1] if "//" in url else url
    depth = path.count("/")
    bonus += max(0, 0.3 - depth * 0.05)
    if ".edu" in url.lower():
        bonus += 0.15
    return bonus


def search(query: str, ioi: dict[str, int], doc_map: dict[int, str],
           total_docs: int, index_fh, doc_lengths: dict[int, int],
           avgdl: float, duplicates: dict[int, int],
           cache: PostingsCache,
           url_to_did: dict[str, int] | None = None,
           pagerank: dict[int, float] | None = None) -> list[tuple[str, float]]:
    """BM25 search with proximity, tiered importance, bigram boost, dedup."""
    tokens = tokenize(query)
    if not tokens:
        return []

    stems = []
    seen = set()
    for t in tokens:
        s = cached_stem(t)
        if s not in seen:
            seen.add(s)
            stems.append(s)

    # fetch postings for each term
    term_postings: list[dict[int, tuple[int, int]]] = []
    term_positions: list[dict[int, list[int]]] = []
    term_idfs: list[float] = []

    for stem in stems:
        raw = fetch_postings(stem, ioi, index_fh, cache)
        pmap = {}
        posmap = {}
        for doc_id, tf, tier, positions in raw:
            pmap[doc_id] = (tf, tier)
            if positions:
                posmap[doc_id] = positions
        term_postings.append(pmap)
        term_positions.append(posmap)
        df = max(len(raw), 1)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        term_idfs.append(idf)

    # bigram boost: if query has 2+ terms, look up bigrams
    bigram_docs: dict[int, float] = {}
    stemmed_query = [cached_stem(t) for t in tokens]
    for i in range(len(stemmed_query) - 1):
        bg = f"{stemmed_query[i]}_{stemmed_query[i+1]}"
        bg_postings = fetch_postings(bg, ioi, index_fh, cache)
        if bg_postings:
            bg_df = len(bg_postings)
            bg_idf = math.log((total_docs - bg_df + 0.5) / (bg_df + 0.5) + 1)
            for doc_id, tf, tier, _ in bg_postings:
                dl = doc_lengths.get(doc_id, avgdl)
                bm25 = bg_idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
                mult = TIER_MULTIPLIERS.get(tier, 1.0)
                bigram_docs[doc_id] = bigram_docs.get(doc_id, 0.0) + bm25 * mult

    # AND intersection
    non_empty = [tp for tp in term_postings if tp]
    if len(non_empty) == len(stems) and len(stems) > 0:
        order = sorted(range(len(term_postings)), key=lambda i: len(term_postings[i]))
        candidate_docs = set(term_postings[order[0]].keys())
        for i in order[1:]:
            candidate_docs &= term_postings[i].keys()
    else:
        candidate_docs = set()

    use_or = len(candidate_docs) < AND_MIN_RESULTS

    def score_doc(doc_id: int) -> float:
        score = 0.0
        for i in range(len(stems)):
            if doc_id not in term_postings[i]:
                continue
            tf, tier = term_postings[i][doc_id]
            dl = doc_lengths.get(doc_id, avgdl)
            bm25 = term_idfs[i] * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
            score += bm25 * TIER_MULTIPLIERS.get(tier, 1.0)
        score += bigram_docs.get(doc_id, 0.0)
        # proximity bonus (multi-term queries)
        if len(stems) >= 2:
            pos_lists = [term_positions[i].get(doc_id, []) for i in range(len(stems))]
            if all(pos_lists):
                span = min_window_span(pos_lists)
                if span >= 0:
                    score += PROXIMITY_WEIGHT / (1 + span)
        if pagerank:
            score += pagerank.get(doc_id, 0.0) * PAGERANK_WEIGHT
        url = doc_map.get(doc_id, "")
        score += url_quality_score(url)
        return score

    if use_or:
        all_doc_ids: set[int] = set()
        for tp in term_postings:
            all_doc_ids.update(tp.keys())
        scored = []
        for doc_id in all_doc_ids:
            sc = score_doc(doc_id)
            if doc_id in candidate_docs:
                sc *= AND_BONUS
            scored.append((doc_map.get(doc_id, f"doc_{doc_id}"), sc))
    else:
        scored = [(doc_map.get(doc_id, f"doc_{doc_id}"), score_doc(doc_id))
                  for doc_id in candidate_docs]

    # filter duplicates
    filtered = []
    seen_canonical = set()
    scored.sort(key=lambda x: -x[1])
    for url, sc in scored:
        if not url_to_did or not duplicates:
            filtered.append((url, sc))
            continue
        did = url_to_did.get(url)
        if did is not None and did in duplicates:
            canon = duplicates[did]
            if canon in seen_canonical:
                continue
            seen_canonical.add(canon)
        elif did is not None:
            if did in seen_canonical:
                continue
            seen_canonical.add(did)
        filtered.append((url, sc))

    return filtered


def main():
    """interactive search REPL for testing queries from the terminal."""
    print("Loading index metadata...")
    t0 = time.time()
    ioi = load_index_of_index()
    doc_map = load_doc_id_map()
    total_docs = load_total_docs()
    doc_lengths = load_doc_lengths()
    duplicates = load_duplicates()
    pagerank = load_pagerank()
    index_fh = open(INDEX_DIR / "index.txt", "rb")
    cache = PostingsCache()

    url_to_did = {url: did for did, url in doc_map.items()}

    if doc_lengths:
        avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    else:
        avgdl = 1.0

    t1 = time.time()
    print(f"Ready. {len(ioi)} terms, {total_docs} docs, {len(duplicates)} dups loaded in {t1 - t0:.2f}s\n")

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
        results = search(query, ioi, doc_map, total_docs, index_fh,
                         doc_lengths, avgdl, duplicates, cache,
                         url_to_did, pagerank)
        t_end = time.time()

        elapsed_ms = (t_end - t_start) * 1000
        print(f"  [{len(results)} results in {elapsed_ms:.1f}ms]")

        if not results:
            print("  No results found.\n")
            continue

        for rank, (url, score) in enumerate(results[:5], 1):
            print(f"  {rank}. {url}  (score: {score:.4f})")
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")
        print()

    index_fh.close()


if __name__ == "__main__":
    main()
