"""
Inverted index builder for the ICS web corpus search engine.

Reads approximately 56,000 HTML pages from developer.zip and constructs a
disk-based inverted index. The indexer processes each document by parsing its
HTML, extracting tokens, stemming them with Porter stemmer, and recording term
frequencies, word positions, and importance tiers derived from HTML tag context
(title, h1, h2/h3/bold, body text).

The index is built in batches: every 18,000 documents the in-memory partial
index is flushed to a sorted file on disk, satisfying the requirement to offload
at least 3 times. After all documents are processed, a k-way heap merge combines
the sorted partial files into a single final index with an accompanying
index-of-index for O(1) byte-offset lookups at query time.

Additional signals computed during indexing:
  - SimHash fingerprints for near-duplicate detection (64-bit, 4-band LSH)
  - Bigram (2-gram) entries for consecutive token pair matching
  - Anchor text from <a> tags, attributed to the linked target document
  - PageRank scores via power iteration over the extracted link graph
  - Per-document token counts for BM25 length normalization
"""

import hashlib
import json
import os
import re
import struct
import time
import heapq
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urldefrag

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ── configuration ──────────────────────────────────────────────────────────────
CORPUS_ZIP = "/Users/robert/Downloads/developer.zip"
INDEX_DIR = Path("index")
# Number of documents to process before flushing the in-memory partial index to
# disk. With ~56k documents this guarantees at least 3 offloads as required.
BATCH_SIZE = 18_000

# ── importance tiers ───────────────────────────────────────────────────────────
# Words appearing in more prominent HTML tags are assigned a higher tier value.
# At search time each tier maps to a score multiplier (title=3x, h1=2x, etc.).
TIER_TITLE = 3
TIER_H1 = 2
TIER_EMPHASIS = 1  # h2, h3, b, strong
TIER_BODY = 0

# Maps HTML tag names to their importance tier. When the parser encounters text
# inside one of these tags, every stemmed token gets at least this tier value.
TAG_TIERS = {
    "title": TIER_TITLE,
    "h1": TIER_H1,
    "h2": TIER_EMPHASIS,
    "h3": TIER_EMPHASIS,
    "b": TIER_EMPHASIS,
    "strong": TIER_EMPHASIS,
}

# ── SimHash near-duplicate detection ──────────────────────────────────────────
SIMHASH_BITS = 64          # fingerprint length in bits
HAMMING_THRESHOLD = 3      # max bit differences to consider two pages near-duplicates

# ── PageRank ──────────────────────────────────────────────────────────────────
PR_DAMPING = 0.85          # probability of following a link vs. teleporting randomly
PR_ITERATIONS = 25         # number of power-iteration rounds (sufficient for convergence)

# globals
stemmer = PorterStemmer()
stem_cache: dict[str, str] = {}


def cached_stem(token: str) -> str:
    """Return the Porter-stemmed form of a token, using a dictionary cache.

    The Porter stemmer is computationally expensive relative to a dict lookup.
    Since the same surface-form words appear across thousands of documents,
    caching avoids millions of redundant stemmer invocations during indexing.
    """
    s = stem_cache.get(token)
    if s is None:
        s = stemmer.stem(token)
        stem_cache[token] = s
    return s


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    Uses the regex pattern [a-zA-Z0-9]+ to capture every contiguous sequence
    of letters and digits. Punctuation, whitespace, and special characters act
    as delimiters and are discarded. The result is lowercased so that matching
    is case-insensitive. No stop-word removal is performed (per the spec).
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def canonical_url(url: str) -> str:
    """Normalize a URL by removing fragment and trailing slash."""
    return urldefrag(url)[0].rstrip("/").lower()


def compute_simhash(term_tf: dict[str, int]) -> int:
    """Compute a 64-bit SimHash fingerprint from a document's term frequencies.

    SimHash is a locality-sensitive hash: documents with similar content produce
    fingerprints that differ in only a few bits. The algorithm works by treating
    each bit position as a weighted vote. For every term, its MD5 hash determines
    which way each bit votes (+tf or -tf). After all terms have voted, each bit
    is set to 1 if its accumulated weight is positive, 0 otherwise.

    The result is a single 64-bit integer that summarizes the document's content.
    Two documents can then be compared by counting the differing bits (Hamming
    distance) — a distance of 3 or fewer bits indicates near-duplicate content.
    """
    v = [0] * SIMHASH_BITS
    for term, tf in term_tf.items():
        # Hash the term with MD5 and take the first 8 bytes as a 64-bit integer
        h = struct.unpack("<Q", hashlib.md5(term.encode()).digest()[:8])[0]
        for i in range(SIMHASH_BITS):
            if h & (1 << i):
                v[i] += tf
            else:
                v[i] -= tf
    # Convert the vote tallies into a binary fingerprint
    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count the number of bit positions where two integers differ.

    XOR produces a 1 at every position where the inputs differ, then we count
    the 1-bits. Used to measure similarity between SimHash fingerprints.
    """
    return bin(a ^ b).count("1")


def compute_pagerank(outlinks: dict[int, set[int]], total_docs: int) -> list[float]:
    """Compute PageRank scores for all documents using power iteration.

    PageRank models a "random surfer" who follows links with probability d
    (the damping factor, 0.85) and jumps to a random page with probability
    1-d. Pages linked to by many others — especially by authoritative pages —
    accumulate higher scores.

    Dangling nodes (pages with no outlinks) are handled by redistributing
    their rank equally across all pages, preventing rank from leaking out of
    the graph.

    After 25 iterations the scores have converged sufficiently for ranking.
    Returns a list indexed by doc_id.
    """
    pr = [1.0 / total_docs] * total_docs
    for _ in range(PR_ITERATIONS):
        # Dangling nodes contribute their rank uniformly to all pages
        dangling_sum = sum(
            pr[i] for i in range(total_docs)
            if i not in outlinks or not outlinks[i]
        )
        dangling_contrib = PR_DAMPING * dangling_sum / total_docs
        # Every page gets a baseline from the random teleport + dangling redistribution
        base = (1 - PR_DAMPING) / total_docs + dangling_contrib
        new_pr = [base] * total_docs
        # Each page distributes its rank equally among the pages it links to
        for src, targets in outlinks.items():
            if not targets:
                continue
            share = PR_DAMPING * pr[src] / len(targets)
            for target in targets:
                if target < total_docs:
                    new_pr[target] += share
        pr = new_pr
    return pr


def parse_document(html: str):
    """Parse an HTML document and extract all indexing signals.

    Returns a tuple of:
      - term_tf: dict mapping each stemmed term to its frequency in the document
      - stem_max_tier: dict mapping each stem to the highest importance tier it
        was found in (e.g., a word in both title and body gets the title tier)
      - doc_length: total token count, used later for BM25 length normalization
      - stemmed_tokens: ordered list of all stems, used to extract bigrams
      - term_positions: dict mapping each stem to its list of word positions,
        used later for proximity scoring at query time
      - anchors: list of (href, [anchor_stems]) for outgoing links, used to
        index anchor text under the target document
    """
    soup = BeautifulSoup(html, "lxml")

    # First pass: scan important HTML tags (title, h1, h2, h3, b, strong) and
    # record the highest importance tier for each stemmed token. If "algorithm"
    # appears in both the title and body, it gets tier 3 (title).
    stem_max_tier: dict[str, int] = {}
    for tag_name, tier in TAG_TIERS.items():
        for tag in soup.find_all(tag_name):
            text = tag.get_text(separator=" ", strip=True)
            for tok in tokenize(text):
                s = cached_stem(tok)
                if tier > stem_max_tier.get(s, -1):
                    stem_max_tier[s] = tier

    # Second pass: extract the full visible text and compute term frequencies
    # and word positions. The position of each token (0-indexed) is recorded so
    # that proximity scoring can determine how close query terms appear.
    full_text = soup.get_text(separator=" ", strip=True)
    tokens = tokenize(full_text)
    doc_length = len(tokens)
    stemmed_tokens = [cached_stem(t) for t in tokens]
    term_tf: dict[str, int] = defaultdict(int)
    term_positions: dict[str, list[int]] = defaultdict(list)
    for pos, s in enumerate(stemmed_tokens):
        term_tf[s] += 1
        term_positions[s].append(pos)

    # Third pass: extract anchor text from outgoing <a> links. The visible text
    # of each link describes the target page from an external perspective —
    # these stems will later be indexed under the target document.
    anchors = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        anchor_text = a_tag.get_text(separator=" ", strip=True)
        if anchor_text:
            anchor_stems = [cached_stem(t) for t in tokenize(anchor_text)]
            if anchor_stems:
                anchors.append((href, anchor_stems))

    return dict(term_tf), stem_max_tier, doc_length, stemmed_tokens, dict(term_positions), anchors


def write_partial_index(partial_index: dict, partial_num: int) -> str:
    """Flush the in-memory partial index to a sorted file on disk.

    Each line has the format: term|doc:tf:tier:pos1.pos2.pos3,...
    Terms are written in sorted (alphabetical) order so that the k-way merge
    can efficiently combine multiple partial files using a min-heap. Positions
    are dot-separated integers representing word offsets within the document.
    """
    path = INDEX_DIR / f"partial_{partial_num}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(partial_index.keys()):
            postings = partial_index[term]
            strs = []
            for entry in postings:
                did, tf, tier = entry[0], entry[1], entry[2]
                positions = entry[3] if len(entry) > 3 else []
                if positions:
                    strs.append(f"{did}:{tf}:{tier}:{'.'.join(map(str, positions))}")
                else:
                    strs.append(f"{did}:{tf}:{tier}")
            f.write(f"{term}|{','.join(strs)}\n")
    return str(path)


def parse_posting_line(line: str):
    """Split an index line into (term, postings_str) at the first pipe delimiter."""
    sep = line.index("|")
    return line[:sep], line[sep + 1:]


def merge_partial_indexes(partial_paths: list[str]):
    """K-way merge of all sorted partial index files into the final index.

    Opens all partial files simultaneously and uses a min-heap to merge them
    in alphabetical order. When the same term appears in multiple partial files
    (because it occurred in documents across different batches), their posting
    lists are concatenated.

    Produces two output files:
      - index.txt: the complete inverted index, one line per term
      - index_of_index.txt: maps each term to its byte offset in index.txt,
        enabling O(1) lookups at query time via file.seek()

    This is the standard external merge sort approach for building indexes
    that are too large to fit in memory at once.
    """
    index_path = INDEX_DIR / "index.txt"
    ioi_path = INDEX_DIR / "index_of_index.txt"
    unique_terms = 0

    file_handles = []
    heap = []

    for i, path in enumerate(partial_paths):
        fh = open(path, "r", encoding="utf-8")
        file_handles.append(fh)
        line = fh.readline()
        if line:
            term, postings = parse_posting_line(line.rstrip("\n"))
            heap.append((term, postings, i))

    heapq.heapify(heap)

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

        flush_term()

    for fh in file_handles:
        fh.close()

    return unique_terms


def find_duplicates(fingerprints: dict[int, int]) -> dict[int, int]:
    """Find near-duplicate documents using multi-band SimHash.

    Returns dict mapping duplicate doc_id -> canonical doc_id.
    """
    duplicates = {}
    items = sorted(fingerprints.items())

    num_bands = 4
    band_bits = SIMHASH_BITS // num_bands
    band_tables: list[dict[int, list[tuple[int, int]]]] = [defaultdict(list) for _ in range(num_bands)]
    for doc_id, fp in items:
        for b in range(num_bands):
            band_key = (fp >> (b * band_bits)) & ((1 << band_bits) - 1)
            band_tables[b][band_key].append((doc_id, fp))

    for doc_id, fp in items:
        if doc_id in duplicates:
            continue
        candidates = set()
        for b in range(num_bands):
            band_key = (fp >> (b * band_bits)) & ((1 << band_bits) - 1)
            for other_id, _ in band_tables[b][band_key]:
                if other_id < doc_id and other_id not in duplicates:
                    candidates.add(other_id)
        for other_id in sorted(candidates):
            if hamming_distance(fp, fingerprints[other_id]) <= HAMMING_THRESHOLD:
                duplicates[doc_id] = other_id
                break

    return duplicates


def resolve_link_graph(
    raw_links: list[tuple[str, str]], url_to_doc_id: dict[str, int],
) -> dict[int, set[int]]:
    """Convert (source_url, target_url) pairs into a doc_id adjacency map."""
    outlinks: dict[int, set[int]] = {}
    for src_url, tgt_url in raw_links:
        src_did = url_to_doc_id.get(src_url)
        tgt_did = url_to_doc_id.get(tgt_url)
        if src_did is not None and tgt_did is not None and src_did != tgt_did:
            outlinks.setdefault(src_did, set()).add(tgt_did)
    return outlinks


def main():
    """Build the full inverted index from the corpus zip file."""
    start = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening corpus: {CORPUS_ZIP}")
    zf = zipfile.ZipFile(CORPUS_ZIP, "r")
    json_files = [n for n in zf.namelist() if n.endswith(".json")]
    total_docs = len(json_files)
    print(f"Found {total_docs} JSON files")

    doc_id_map: dict[int, str] = {}
    url_to_doc_id: dict[str, int] = {}
    doc_lengths: dict[int, int] = {}
    fingerprints: dict[int, int] = {}
    raw_links: list[tuple[str, str]] = []  # (source_url, target_url) for post-loop resolution
    partial_index: dict[str, list] = defaultdict(list)
    partial_num = 0
    partial_paths: list[str] = []
    doc_id = 0
    anchor_targets: dict[str, list[str]] = defaultdict(list)

    for jf in json_files:
        try:
            raw = zf.read(jf)
            data = json.loads(raw)
            url = data.get("url", "")
            content = data.get("content", "")
        except Exception as e:
            print(f"  SKIP {jf}: {e}")
            continue

        doc_id_map[doc_id] = url
        clean_url = canonical_url(url)
        url_to_doc_id[clean_url] = doc_id

        if content.strip():
            term_tf, stem_max_tier, doc_length, stemmed_tokens, term_positions, anchors = parse_document(content)
            doc_lengths[doc_id] = doc_length

            if term_tf:
                fingerprints[doc_id] = compute_simhash(term_tf)

            for term, tf in term_tf.items():
                tier = stem_max_tier.get(term, TIER_BODY)
                positions = term_positions.get(term, [])
                partial_index[term].append((doc_id, tf, tier, positions))

            # bigrams from consecutive stemmed tokens
            bigram_tf: dict[str, int] = defaultdict(int)
            for i in range(len(stemmed_tokens) - 1):
                bg = f"{stemmed_tokens[i]}_{stemmed_tokens[i+1]}"
                bigram_tf[bg] += 1
            for bg, tf in bigram_tf.items():
                partial_index[bg].append((doc_id, tf, TIER_BODY))

            # collect anchor text + raw links for post-loop resolution
            for href, anchor_stems in anchors:
                try:
                    resolved = urljoin(url, href)
                    resolved = canonical_url(resolved)
                    anchor_targets[resolved].extend(anchor_stems)
                    raw_links.append((clean_url, resolved))
                except Exception:
                    pass
        else:
            doc_lengths[doc_id] = 0

        doc_id += 1

        if doc_id % 5000 == 0:
            elapsed = time.time() - start
            print(f"  Processed {doc_id}/{total_docs} docs  ({elapsed:.1f}s)")

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

    # anchor text pass: add anchor stems to target doc postings
    print("Processing anchor text...")
    anchor_index: dict[str, list] = defaultdict(list)
    anchor_count = 0
    for target_url, stems in anchor_targets.items():
        target_did = url_to_doc_id.get(target_url)
        if target_did is None:
            continue
        atf: dict[str, int] = defaultdict(int)
        for s in stems:
            atf[s] += 1
        for stem, tf in atf.items():
            anchor_index[stem].append((target_did, tf, TIER_H1))
            anchor_count += 1
    print(f"  Added {anchor_count} anchor text postings")

    if anchor_index:
        print(f"  >> Offloading anchor text partial index {partial_num} ({len(anchor_index)} terms)")
        path = write_partial_index(anchor_index, partial_num)
        partial_paths.append(path)
        partial_num += 1

    # merge
    print("Merging partial indexes...")
    unique_terms = merge_partial_indexes(partial_paths)
    merge_done = time.time()
    print(f"Merge complete: {unique_terms} unique terms ({merge_done - parse_done:.1f}s)")

    # write doc_id_map
    with open(INDEX_DIR / "doc_id_map.txt", "w", encoding="utf-8") as f:
        for did in range(len(doc_id_map)):
            f.write(f"{did}|{doc_id_map[did]}\n")

    # write doc_lengths
    with open(INDEX_DIR / "doc_lengths.txt", "w", encoding="utf-8") as f:
        for did in range(len(doc_id_map)):
            f.write(f"{did}|{doc_lengths.get(did, 0)}\n")

    # simhash duplicate detection
    print("Detecting near-duplicates via SimHash...")
    duplicates = find_duplicates(fingerprints)
    with open(INDEX_DIR / "duplicates.txt", "w", encoding="utf-8") as f:
        for dup_id, canon_id in sorted(duplicates.items()):
            f.write(f"{dup_id}|{canon_id}\n")
    print(f"  Found {len(duplicates)} near-duplicates")

    # resolve link graph post-loop (so forward links are captured)
    print("Resolving link graph...")
    outlinks = resolve_link_graph(raw_links, url_to_doc_id)
    del raw_links

    # pagerank
    print("Computing PageRank...")
    link_count = sum(len(targets) for targets in outlinks.values())
    pr_scores = compute_pagerank(outlinks, doc_id)
    with open(INDEX_DIR / "pagerank.txt", "w", encoding="utf-8") as f:
        for did in range(doc_id):
            f.write(f"{did}|{pr_scores[did]:.10f}\n")
    top_pr = sorted(range(doc_id), key=lambda i: pr_scores[i], reverse=True)[:5]
    print(f"  {len(outlinks)} docs with outlinks, {link_count} edges")
    print(f"  Top PR: {[(doc_id_map[d], f'{pr_scores[d]:.6f}') for d in top_pr]}")

    # compute index size
    index_size_bytes = os.path.getsize(INDEX_DIR / "index.txt")
    index_size_kb = index_size_bytes / 1024

    # write metadata
    with open(INDEX_DIR / "metadata.txt", "w", encoding="utf-8") as f:
        f.write("group ids\n35485800, 79822855, 30679988, 32438497\n\n")
        f.write(f"documents: {doc_id}\n")
        f.write(f"unique_terms: {unique_terms}\n")
        f.write(f"index_size_kb: {index_size_kb:.1f}\n")
        f.write(f"partial_indexes: {partial_num}\n")
        f.write(f"total_time_s: {time.time() - start:.1f}\n")
        f.write(f"near_duplicates: {len(duplicates)}\n")
        f.write(f"link_edges: {link_count}\n")

    # clean up partial files
    for path in partial_paths:
        os.remove(path)

    total_time = time.time() - start
    print(f"\n{'='*50}")
    print(f"  Documents indexed : {doc_id}")
    print(f"  Unique terms      : {unique_terms}")
    print(f"  Index size on disk: {index_size_kb:.1f} KB ({index_size_kb/1024:.1f} MB)")
    print(f"  Partial offloads  : {partial_num}")
    print(f"  Near-duplicates   : {len(duplicates)}")
    print(f"  Link edges        : {link_count}")
    print(f"  Total time        : {total_time:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
