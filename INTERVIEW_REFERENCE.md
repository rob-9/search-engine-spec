# A3 Search Engine — Interview Reference

Complete feature-to-code mapping across all three source files. Use this to quickly locate and explain any feature during the demo.

---

## Architecture Overview

The system has three files forming a pipeline:

| File | Role | Lines |
|---|---|---|
| `indexer.py` (466 lines) | Reads the corpus zip, parses HTML, builds a disk-based inverted index with all extra-credit signals | Offline, run once |
| `search.py` (435 lines) | Loads lightweight metadata into memory, reads postings from disk on demand, scores and ranks results | Core retrieval engine |
| `web.py` (629 lines) | Flask web interface that wraps `search.py` with a styled UI | User-facing frontend |

### On-Disk Index Files (produced by `indexer.py`, consumed by `search.py`)

| File | Contents | Format |
|---|---|---|
| `index.txt` | Full inverted index (one line per term) | `term\|doc:tf:tier:positions,doc:tf:tier:positions,...` |
| `index_of_index.txt` | Byte offset of each term's line in `index.txt` | `term\|byte_offset` |
| `doc_id_map.txt` | Numeric doc ID to URL mapping | `doc_id\|url` |
| `doc_lengths.txt` | Token count per document (for BM25 normalization) | `doc_id\|length` |
| `duplicates.txt` | SimHash near-duplicate pairs | `dup_doc_id\|canonical_doc_id` |
| `pagerank.txt` | PageRank score per document | `doc_id\|score` |
| `metadata.txt` | Corpus statistics (doc count, term count, etc.) | key-value text |

---

## Feature Map

### 1. Tokenization

All text (both at index time and query time) is split into lowercase alphanumeric sequences. No stop-word removal is performed — the assignment spec explicitly forbids it.

| What | File | Lines | Explanation |
|---|---|---|---|
| Tokenizer function | `indexer.py` | 74–76 | Uses the regex `[a-zA-Z0-9]+` on the lowercased input string. This captures every contiguous run of letters and digits as a single token. Lowercasing ensures case-insensitive matching. |
| Tokenizer function (search-side copy) | `search.py` | 57–59 | Identical regex logic so that queries are tokenized the same way as documents were at index time. Consistency here is critical — if they differed, stems wouldn't match. |

### 2. Porter Stemming

Every token is reduced to its root form using NLTK's Porter Stemmer. This lets queries like "learning" match documents containing "learns" or "learned."

| What | File | Lines | Explanation |
|---|---|---|---|
| Stemmer with memoization (indexer) | `indexer.py` | 61–71 | A dictionary `stem_cache` maps raw tokens to their stemmed forms. Before calling `stemmer.stem()`, the cache is checked first. Since the same words appear across thousands of documents, this avoids tens of millions of redundant stemmer invocations and significantly speeds up indexing. |
| Stemmer with memoization (search) | `search.py` | 44–54 | Same memoization pattern on the search side. Repeated or overlapping queries benefit because stems computed for earlier queries are already cached in memory. |

### 3. Importance Tiers (Weighted HTML Tags)

Words found inside important HTML tags are assigned a higher "tier" value. During scoring, each tier gets a multiplier so that a term appearing in a page's title contributes more to that page's score than the same term buried in body text.

| What | File | Lines | Explanation |
|---|---|---|---|
| Tier constants | `indexer.py` | 37–50 | Four tiers are defined: `TIER_TITLE = 3` (title tags), `TIER_H1 = 2` (h1 headings), `TIER_EMPHASIS = 1` (h2, h3, b, strong tags), and `TIER_BODY = 0` (everything else). The `TAG_TIERS` dictionary maps each HTML tag name to its tier. |
| Extracting tiers during parsing | `indexer.py` | 127–135 | For each document, the parser iterates over every tag listed in `TAG_TIERS`. It tokenizes and stems the text inside each tag, and records the highest tier seen for every stem. So if "algorithm" appears in both the title and body, it gets tier 3 (title). |
| Tier multipliers at scoring time | `search.py` | 25–30 | The multipliers translate tiers into score weight: title terms get 3x, h1 gets 2x, emphasis gets 1.5x, body gets 1x. These are applied multiplicatively to each term's BM25 score. |
| Multiplier applied in BM25 scoring | `search.py` | 302 | Inside `score_doc()`, after computing a term's raw BM25 score, it is multiplied by `TIER_MULTIPLIERS.get(tier, 1.0)`. This means a term in the title of a page can contribute triple the score of the same term in body text. |

### 4. Disk-Based Inverted Index (No Full Index in Memory)

The assignment requires that search must NOT load the full inverted index into memory. Instead, only lightweight metadata is loaded. Actual posting lists are read from disk on demand using byte-offset seeks.

| What | File | Lines | Explanation |
|---|---|---|---|
| Writing the final merged index | `indexer.py` | 185–238 | The k-way merge writes each term's posting list as a single line in `index.txt`. Simultaneously, it records the byte offset of each line in `index_of_index.txt`. This is the mechanism that makes on-demand retrieval possible — each term's data can be located in O(1) by seeking to its byte offset. |
| Loading the index-of-index into memory | `search.py` | 62–70 | At startup, only the index-of-index (term → byte offset) is loaded. This is a small lookup table — just a string and an integer per term — not the posting lists themselves. |
| On-demand disk seek for postings | `search.py` | 157–184 | `fetch_postings()` is the core retrieval mechanism. Given a term, it looks up the byte offset, calls `seek()` on the open file handle to jump directly to that position, reads one line, and parses it into `(doc_id, tf, tier, positions)` tuples. The file handle is opened in binary mode (`"rb"`) so that `seek()` works with byte offsets. |
| LRU postings cache | `search.py` | 135–154 | `PostingsCache` wraps an `OrderedDict` to implement least-recently-used eviction. When a term's postings are fetched from disk, they're stored in the cache (up to 1000 entries). Subsequent lookups for the same term skip the disk entirely. This is especially helpful when users issue similar queries in succession. |
| Lazy position parsing | `search.py` | 159–160, 180 | Position data is stored as a dot-separated string (e.g., `"3.17.42"`) rather than being parsed into a list of integers immediately. Parsing only happens inside `score_doc()` (line 309) when proximity scoring is actually needed. For single-term queries, positions are never parsed at all. This optimization was specifically added to bring query response under 300ms. |

### 5. Partial Index Offloading and K-Way Merge

The assignment requires that the indexer must offload partial indexes to disk at least 3 times during construction, then merge them. This simulates building an index too large to fit in memory.

| What | File | Lines | Explanation |
|---|---|---|---|
| Batch size configuration | `indexer.py` | 35 | `BATCH_SIZE = 18_000` — after every 18,000 documents, the current in-memory partial index is flushed to disk as a sorted text file. |
| Offloading a partial index | `indexer.py` | 161–176, 346–351 | `write_partial_index()` sorts all terms alphabetically and writes one line per term in the format `term\|postings`. The main loop triggers this every `BATCH_SIZE` documents. With ~56k documents and batch size 18k, this produces at least 3 partial files (plus one for anchor text). |
| K-way merge using a min-heap | `indexer.py` | 185–238 | `merge_partial_indexes()` opens all partial files simultaneously and uses Python's `heapq` to perform an efficient k-way merge. It pops the smallest term from the heap, reads the next line from that file, and pushes it back. When the same term appears in multiple partial files, their posting lists are concatenated. The merge produces the final `index.txt` and `index_of_index.txt`. |
| Cleanup of partial files | `indexer.py` | 448–450 | After merging is complete, the temporary partial index files are deleted from disk. |

### 6. BM25 Ranking

BM25 (Best Matching 25) is the core relevance scoring formula, an improvement over raw TF-IDF that accounts for term saturation and document length normalization.

| What | File | Lines | Explanation |
|---|---|---|---|
| BM25 parameters | `search.py` | 20–22 | `k1 = 1.2` controls term frequency saturation — higher values give more credit to repeated terms. `b = 0.75` controls document length normalization — at 0.75, longer documents are penalized but not aggressively. These are standard values from IR literature. |
| IDF calculation | `search.py` | 263–265 | Inverse document frequency: `log((N - df + 0.5) / (df + 0.5) + 1)`. This measures how rare a term is across the corpus. Rare terms (low `df`) get high IDF, common terms get low IDF. The `+1` inside the log prevents negative values. |
| BM25 score per term | `search.py` | 301 | The full formula: `idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))`. The numerator grows with term frequency but saturates due to `k1`. The denominator normalizes by document length — a term appearing 5 times in a 100-word document scores higher than the same 5 occurrences in a 10,000-word document. |
| Average document length | `search.py` | 390–392 | `avgdl` is computed once at startup as the mean of all document lengths. This is the baseline for BM25's length normalization — documents shorter than average get a boost, longer ones get penalized. |

### 7. Boolean AND with OR Fallback

The assignment requires boolean AND at minimum. This implementation starts with AND (all query terms must appear) but falls back to OR when AND yields too few results.

| What | File | Lines | Explanation |
|---|---|---|---|
| AND intersection | `search.py` | 282–290 | The posting lists are sorted by size (smallest first) and intersected using Python set operations. Starting with the smallest list minimizes the size of the working set at each step. If any term has zero postings, the intersection is empty. |
| OR fallback threshold | `search.py` | 33, 292 | If the AND intersection yields fewer than `AND_MIN_RESULTS = 5` documents, the engine switches to OR mode. This prevents returning zero results for queries where all terms rarely co-occur. |
| OR mode: prioritize multi-term matches | `search.py` | 320–343 | In OR mode, the engine counts how many query terms each document matches. Documents matching 2+ terms are included first. If still below the threshold, single-term matches are added. Documents that were in the AND set get a 1.5x score bonus (`AND_BONUS`) so they still rank above partial matches. |

### 8. SimHash Near-Duplicate Detection (Extra Credit — 2 pts)

SimHash is a locality-sensitive hash that produces similar fingerprints for similar documents. Two documents whose fingerprints differ by only a few bits are considered near-duplicates.

| What | File | Lines | Explanation |
|---|---|---|---|
| SimHash fingerprint computation | `indexer.py` | 79–93 | For each document, a 64-bit fingerprint is computed from term frequencies. Each term is hashed with MD5 (truncated to 64 bits). For each of the 64 bit positions, the TF is added if the bit is 1, subtracted if 0. The final fingerprint sets bit `i` to 1 if the accumulated value at position `i` is positive. This makes the fingerprint a "weighted majority vote" across all terms. |
| Hamming distance | `indexer.py` | 96–98 | Counts differing bits between two fingerprints using `bin(a ^ b).count("1")`. XOR produces a 1 bit everywhere the fingerprints differ. |
| Multi-band duplicate detection | `indexer.py` | 241–269 | To avoid comparing every pair of documents (O(n^2)), the 64-bit fingerprint is split into 4 bands of 16 bits each. Documents sharing the same 16-bit band value are candidates. Only candidates within `HAMMING_THRESHOLD = 3` bits are marked as duplicates. This band-based approach makes duplicate detection tractable at scale. |
| Duplicate filtering at search time | `search.py` | 354–370 | After scoring, the results list is scanned. Each document is checked against the duplicate map. If a document is a duplicate, its canonical version is looked up. If that canonical version was already shown, the duplicate is skipped. This prevents the same content from occupying multiple result slots. |

### 9. PageRank (Extra Credit — 2.5 pts)

PageRank measures the "authority" of a page based on the link structure of the web. Pages linked to by many other pages (especially authoritative ones) get higher scores.

| What | File | Lines | Explanation |
|---|---|---|---|
| Link graph construction | `indexer.py` | 287, 328–336, 410–420 | During document parsing, every `<a href>` tag is resolved to an absolute URL. Source-target URL pairs are collected in `raw_links`. After all documents are parsed, these are resolved to doc IDs to build an `outlinks` adjacency map (doc_id → set of target doc_ids). Self-links are excluded. |
| Power iteration | `indexer.py` | 101–120 | Classic PageRank with damping factor `d = 0.85` and 25 iterations. Each iteration: (1) compute the "dangling sum" — total rank of pages with no outlinks, which would otherwise leak from the graph; (2) set a baseline for every page of `(1-d)/N + d * dangling_sum/N`; (3) distribute each page's rank equally among its outlinks. After 25 iterations, scores have converged. |
| PageRank in scoring | `search.py` | 314–315, 37 | Each document's PageRank score is multiplied by `PAGERANK_WEIGHT = 500.0` and added to its BM25 score. The weight of 500 was tuned so that PageRank provides a meaningful tiebreaker — authoritative pages rise in the rankings — without completely overpowering textual relevance. |

### 10. 2-Gram (Bigram) Indexing (Extra Credit — 1 pt)

Bigrams are pairs of consecutive stemmed tokens (e.g., "machine_learn"). Indexing them lets the search engine reward documents where query terms appear next to each other.

| What | File | Lines | Explanation |
|---|---|---|---|
| Bigram extraction at index time | `indexer.py` | 320–326 | After stemming all tokens in a document, consecutive pairs are joined with an underscore (e.g., `stemmed_tokens[i] + "_" + stemmed_tokens[i+1]`). Each bigram is counted and added to the partial index with `TIER_BODY`. |
| Bigram lookup at search time | `search.py` | 267–280 | For each consecutive pair of stemmed query terms, the corresponding bigram key is looked up in the index. If found, its postings are scored with BM25 (including tier multipliers) and accumulated in `bigram_docs`. This score is later added to each document's total, rewarding documents where query terms co-occur adjacently. |
| Bigram score integration | `search.py` | 303 | Inside `score_doc()`, any bigram bonus for the document is added: `score += bigram_docs.get(doc_id, 0.0)`. This acts as an additive boost on top of the per-term BM25 scores. |

### 11. Word Position Indexing + Proximity Scoring (Extra Credit — 2 pts)

Beyond knowing that a term appears in a document, positions tell us *where* it appears. This enables proximity scoring — documents where query terms appear close together rank higher.

| What | File | Lines | Explanation |
|---|---|---|---|
| Position tracking at index time | `indexer.py` | 143–146 | While iterating over stemmed tokens, each token's position index (0, 1, 2, ...) is recorded in `term_positions[stem]`. These lists are written into the posting format as dot-separated integers (e.g., `doc:tf:tier:3.17.42.108`). |
| Position format in postings | `indexer.py` | 171–172 | Positions are serialized as `".".join(map(str, positions))`. The dot separator was chosen because it's not used elsewhere in the posting format (which uses `:` for fields and `,` for entries). |
| Lazy position parsing at search time | `search.py` | 180, 308–309 | Positions are stored as raw strings and only split into integer lists when proximity scoring is actually needed. The parsing happens at line 309: `[int(p) for p in raw.split(".")]`. For single-term queries, this code never runs. |
| Minimum window span algorithm | `search.py` | 187–211 | Given one position list per query term, this finds the smallest window of text containing at least one occurrence of every term. It uses a pointer-advance technique: maintain one pointer per list, compute the current window (max - min of pointed values), then advance the pointer at the minimum value. This greedily shrinks the window. Returns -1 if any term is missing from the document. |
| Proximity score contribution | `search.py` | 305–313, 38 | The proximity bonus is `PROXIMITY_WEIGHT / (1 + span)` where `PROXIMITY_WEIGHT = 2.0`. If all query terms appear within the same position (span = 0), the bonus is 2.0. If they're 10 words apart, it's ~0.18. If any term is missing, no bonus is given. This rewards exact-phrase and near-phrase matches. |

### 12. Anchor Text Indexing (Extra Credit — 1 pt)

Anchor text is the visible text inside `<a>` tags. It describes the linked page from an external perspective. By indexing anchor text under the *target* page, we can find pages even when they don't contain the query terms in their own body text.

| What | File | Lines | Explanation |
|---|---|---|---|
| Anchor extraction during parsing | `indexer.py` | 148–156 | For every `<a href>` tag, the parser extracts both the `href` and the visible text. The anchor text is tokenized and stemmed. The result is a list of `(href, [anchor_stems])` tuples per document. |
| URL resolution | `indexer.py` | 329–334 | Each href is resolved to an absolute URL using `urljoin()` (handles relative paths), then defragmented and normalized. The anchor stems are accumulated in `anchor_targets[resolved_url]`. |
| Injecting anchor postings | `indexer.py` | 364–384 | After the main document loop, anchor text is processed in a separate pass. For each target URL, the accumulated anchor stems are counted and added to the index as postings for the target doc with `TIER_H1` (tier 2). This means anchor text is weighted the same as h1 headings — significant but below title weight. |

### 13. URL Quality Signal

A small heuristic bonus based on URL properties, used as a lightweight tiebreaker.

| What | File | Lines | Explanation |
|---|---|---|---|
| URL depth penalty | `search.py` | 217–219 | The number of `/` characters in the URL path is counted. Shallower pages (fewer slashes) get a small bonus: `max(0, 0.3 - depth * 0.05)`. A root page gets +0.3, a page 6 levels deep gets 0. The intuition is that important pages tend to live closer to the domain root. |
| `.edu` domain bonus | `search.py` | 220–221 | Pages on `.edu` domains get a +0.15 bonus. Since the corpus is ICS web content from a university, `.edu` pages are generally more authoritative. |

### 14. Top-K Selection and Heap-Based Sorting

| What | File | Lines | Explanation |
|---|---|---|---|
| Heap-based top-K | `search.py` | 349–353 | If there are more than 1000 scored documents, `heapq.nlargest(1000, ...)` is used instead of a full sort. This runs in O(n log k) instead of O(n log n), which matters when OR mode produces tens of thousands of candidates. For smaller result sets, a simple sort is used. |

### 15. Web Interface (Extra Credit — 2 pts)

A Flask-based web UI with a polished dark theme, served on `localhost:5000`.

| What | File | Lines | Explanation |
|---|---|---|---|
| Flask app setup | `web.py` | 1–19 | Imports Flask and all loader functions from `search.py`. The app is a standard single-file Flask application. |
| Index data loaded at module level | `web.py` | 576–594 | All metadata (index-of-index, doc map, lengths, duplicates, pagerank) is loaded once when the module is imported. The index file handle is kept open for the lifetime of the process. `atexit.register(index_fh.close)` ensures the file handle is closed on shutdown. |
| Thread safety | `web.py` | 593 | A `threading.Lock()` named `search_lock` guards the `search()` call because Flask may serve requests from multiple threads, and the shared file handle (`index_fh`) is not thread-safe — concurrent `seek()` and `read()` calls would corrupt results. |
| Landing page route | `web.py` | 597–600 | `GET /` renders the template with no query and no results, showing the centered landing page with a search box. |
| Search route | `web.py` | 603–624 | `GET /search?q=...` extracts the query, runs `search()` inside the lock, measures elapsed time, and renders the results page showing the top 20 results with rank badges, URLs, and scores. |
| HTML template | `web.py` | 21–573 | A single Jinja2 template (rendered via `render_template_string`) handles both the landing page and results page using `{% if not query %}` branching. The CSS is fully inline — no external files needed. |
| UI features | `web.py` | 28–440 | Dark theme with custom CSS variables, monospace font stack, animated result entries (staggered fade-in), rank badges with accent colors for top 3, URL parsing into scheme/host/path for display, responsive breakpoints at 600px and 400px, custom scrollbar styling. |
| Keyboard shortcut | `web.py` | 556–568 | JavaScript at the bottom of the template: pressing `/` when not focused on an input field jumps focus to the search box (similar to GitHub's shortcut). |
| No-results state | `web.py` | 525–540 | When a query returns zero results, a styled empty state is shown with suggestion chips linking to known-good queries (machine learning, cristina lopes, ACM, software engineering). |

### 16. Interactive CLI REPL

| What | File | Lines | Explanation |
|---|---|---|---|
| REPL loop | `search.py` | 375–434 | `main()` loads all metadata once, then enters a `while True` loop that reads queries from stdin, runs search, prints the top 5 results with scores and timing, and exits on `quit`/`exit`/`q`/EOF. The index file handle is explicitly closed on exit. |
| Timing measurement | `search.py` | 411–418 | Each query is timed with `time.time()` before and after the `search()` call, and the elapsed milliseconds are printed. This is used to verify the <300ms requirement. |

---

## Performance Design Decisions

These are the specific choices made to meet the <300ms query response time requirement:

| Decision | Where | Why |
|---|---|---|
| Index-of-index for O(1) seek | `search.py:62–70`, `165–169` | Avoids scanning the entire index file. A dictionary lookup gives the byte offset, then a single `seek()` + `readline()` retrieves the posting list. |
| LRU postings cache | `search.py:135–154` | Repeated or overlapping queries skip disk I/O entirely. The cache holds up to 1000 posting lists. |
| Lazy position parsing | `search.py:180`, `309` | Position strings (e.g., `"3.17.42"`) are not split into integer lists until proximity scoring actually needs them. Single-term queries never parse positions. |
| Smallest-first AND intersection | `search.py:285–288` | Intersecting posting lists starting with the shortest one minimizes intermediate set sizes and speeds up the intersection. |
| Heap-based top-K | `search.py:349–351` | When result sets are large, `heapq.nlargest` avoids sorting the full list. O(n log k) vs O(n log n). |
| Stem memoization | `search.py:48–54` | Porter stemming is expensive. Caching stems avoids redundant NLTK calls for repeated tokens across queries. |
| Binary file handle kept open | `search.py:385` | Opening the index file once and reusing the handle avoids filesystem overhead on every query. |
| Pre-computed avgdl | `search.py:390–392` | Average document length is computed once at startup rather than on every query. |

---

## Scoring Formula Summary

For a given query Q and document D, the total score is:

```
score(D, Q) =
    SUM over each term t in Q:
        BM25(t, D) * tier_multiplier(t, D)
  + SUM over each bigram b in Q:
        BM25(b, D) * tier_multiplier(b, D)
  + proximity_bonus(Q, D)
  + PageRank(D) * 500
  + url_quality(D)
```

Where:
- `BM25(t, D) = IDF(t) * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * dl/avgdl))`
- `tier_multiplier` = 3.0 (title), 2.0 (h1), 1.5 (h2/h3/b/strong), 1.0 (body)
- `proximity_bonus` = `2.0 / (1 + min_window_span)` if all terms present, else 0
- `url_quality` = up to +0.3 for shallow URLs, +0.15 for `.edu` domains
- In OR mode, documents in the AND set get an additional 1.5x multiplier

---

## Quick-Find Index

For fast lookup during the interview — every major function and where it lives:

| Function / Class | File | Line | Purpose |
|---|---|---|---|
| `tokenize()` | `indexer.py` | 74 | Split text into lowercase alphanumeric tokens |
| `cached_stem()` | `indexer.py` | 65 | Porter stem with memoization |
| `compute_simhash()` | `indexer.py` | 79 | 64-bit SimHash fingerprint from term frequencies |
| `hamming_distance()` | `indexer.py` | 96 | Bit difference between two fingerprints |
| `compute_pagerank()` | `indexer.py` | 101 | Power iteration PageRank over link graph |
| `parse_document()` | `indexer.py` | 123 | HTML → (tf, tiers, length, tokens, positions, anchors) |
| `write_partial_index()` | `indexer.py` | 161 | Flush in-memory postings to sorted disk file |
| `merge_partial_indexes()` | `indexer.py` | 185 | K-way heap merge of partial files → final index |
| `find_duplicates()` | `indexer.py` | 241 | Multi-band SimHash duplicate detection |
| `indexer.main()` | `indexer.py` | 272 | Full indexing pipeline orchestrator |
| `tokenize()` | `search.py` | 57 | Query-side tokenizer (same regex) |
| `cached_stem()` | `search.py` | 48 | Query-side stem memoization |
| `load_index_of_index()` | `search.py` | 62 | Load term → byte offset map |
| `load_doc_id_map()` | `search.py` | 73 | Load doc_id → URL map |
| `load_doc_lengths()` | `search.py` | 84 | Load doc_id → token count |
| `load_duplicates()` | `search.py` | 98 | Load duplicate → canonical map |
| `load_pagerank()` | `search.py` | 112 | Load doc_id → PageRank score |
| `load_total_docs()` | `search.py` | 126 | Read document count from metadata |
| `PostingsCache` | `search.py` | 135 | LRU cache (OrderedDict-based) |
| `fetch_postings()` | `search.py` | 157 | Disk seek + parse postings for a term |
| `min_window_span()` | `search.py` | 187 | Minimum window containing all query terms |
| `url_quality_score()` | `search.py` | 214 | URL depth + .edu bonus |
| `search()` | `search.py` | 225 | Main search function (BM25 + all signals) |
| `search.main()` | `search.py` | 375 | Interactive CLI REPL |
| `web.index()` | `web.py` | 597 | Flask landing page route |
| `web.search_page()` | `web.py` | 603 | Flask search results route |
