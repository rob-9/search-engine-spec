# A3 Search Engine

Full-text search engine over the ICS developer corpus. Disk-based inverted index with BM25 ranking, six extra credit features, and a web interface.

Group IDs: 35485800, 79822855, 30679988, 32438497

## Architecture

```
developer.zip ──► indexer.py ──► index/ ──► search.py (CLI)
                                        ──► web.py    (Flask UI)
```

1. **Indexer** (`indexer.py`) parses ~55k HTML documents, builds partial indexes in memory (offloaded every 18k docs), then k-way merges them into a single sorted index on disk.
2. **Search** (`search.py`) loads only lightweight metadata into memory (index-of-index, doc lengths, duplicates, PageRank). Postings are fetched from disk via byte-offset seeks. An LRU cache avoids redundant disk reads.
3. **Web UI** (`web.py`) wraps the search module in a Flask app at `localhost:5000`.

## Files

| File | What it does |
|---|---|
| `indexer.py` | builds the inverted index from `developer.zip` |
| `search.py` | BM25 search with CLI REPL (`indexer.py:342`) |
| `web.py` | Flask web interface on port 5000 |
| `queries.txt` | 22 test queries (10 GOOD, 12 FIXED) |

## Index Files (`index/`)

| File | Format | Contents |
|---|---|---|
| `index.txt` | `term\|doc:tf:tier[:positions],...` | merged inverted index (binary, seekable) |
| `index_of_index.txt` | `term\|byte_offset` | term → byte offset into `index.txt` |
| `doc_id_map.txt` | `doc_id\|url` | integer doc ID → original URL |
| `doc_lengths.txt` | `doc_id\|length` | token count per document (for BM25) |
| `duplicates.txt` | `dup_id\|canonical_id` | SimHash near-duplicate pairs |
| `pagerank.txt` | `doc_id\|score` | PageRank score per document |
| `metadata.txt` | key-value | corpus stats (doc count, term count, etc.) |

## Scoring Formula

Each document is scored by combining multiple signals:

```
score = Σ(BM25_i × tier_multiplier_i)    # per query term
      + Σ(BM25_bigram × tier_multiplier)  # bigram matches
      + proximity_weight / (1 + min_span) # word proximity
      + pagerank × 500.0                  # PageRank signal
      + url_quality_bonus                 # URL depth + .edu bonus
```

**BM25** (`search.py:289`):
```
IDF × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × dl/avgdl))
```
- `k1 = 1.2`, `b = 0.75`
- IDF = `log((N - df + 0.5) / (df + 0.5) + 1)`

**Tiered importance** (`search.py:24-29`):
| Tier | HTML tags | Multiplier |
|---|---|---|
| 3 | `<title>` | 3.0× |
| 2 | `<h1>` | 2.0× |
| 1 | `<h2>`, `<h3>`, `<b>`, `<strong>` | 1.5× |
| 0 | body text | 1.0× |

**AND/OR fallback** (`search.py:280`): if boolean AND yields < 5 results, fall back to OR union. Documents matching all terms get a 1.5× AND bonus.

### All scoring weights (`search.py:19-40`)

| Constant | Value | Purpose |
|---|---|---|
| `BM25_K1` | 1.2 | tf saturation — higher = more credit for repeated terms |
| `BM25_B` | 0.75 | length normalization — 0 = ignore doc length, 1 = full normalization |
| `TIER_MULTIPLIERS[3]` | 3.0× | title match boost |
| `TIER_MULTIPLIERS[2]` | 2.0× | h1 match boost |
| `TIER_MULTIPLIERS[1]` | 1.5× | h2/h3/b/strong match boost |
| `TIER_MULTIPLIERS[0]` | 1.0× | body text (no boost) |
| `AND_BONUS` | 1.5× | multiplier for docs matching all terms in OR fallback mode |
| `AND_MIN_RESULTS` | 5 | AND result count threshold before falling back to OR |
| `PAGERANK_WEIGHT` | 500.0 | scalar for raw PageRank values (~0.00002 per doc) |
| `PROXIMITY_WEIGHT` | 2.0 | numerator in proximity bonus: `2.0 / (1 + span)` |
| `CACHE_MAX` | 1000 | LRU postings cache size (not a scoring weight) |

**How weights were determined:**
- `BM25_K1=1.2`, `BM25_B=0.75` — standard values from the BM25 literature (Robertson et al.), used by most search engines as sensible defaults.
- `TIER_MULTIPLIERS` — set by descending HTML semantic importance. Title is the strongest signal of page topic (3×), h1 is a section-level signal (2×), emphasis tags are moderate (1.5×), body is baseline (1×). Validated by checking that queries like "dean" and "cristina lopes" surface the correct authoritative page.
- `PAGERANK_WEIGHT=500.0` — raw PageRank values are very small (~0.00002 per doc). 500× scales them to contribute ~0.5–3 points, making PageRank a meaningful tiebreaker without overpowering content relevance (BM25 typically scores 6–10+).
- `PROXIMITY_WEIGHT=2.0` — gives up to 2 points for co-located terms (adjacent → 1.0, same position → 2.0). Tuned so phrase-like queries ("information retrieval", "capstone project") rank exact-phrase pages higher without dominating the score.
- `AND_BONUS=1.5×` — in OR fallback mode, docs matching all terms should clearly outrank partial matches. 1.5× was chosen empirically: high enough to separate full matches from partial, low enough that a strong partial match with high PageRank can still compete.
- `AND_MIN_RESULTS=5` — threshold for OR fallback. Set low enough that multi-word queries like "master of software engineering" (which return 0–2 AND results) trigger fallback, but high enough that common two-word queries stay in AND mode.

## Test Queries (`queries.txt`)

We tested with 22 queries (10 GOOD, 12 FIXED), following the assignment requirement that at least half should start with poor relevance or speed, then be improved by general code changes while preserving the other half's good performance.

### GOOD queries (10) — relevant results, fast response

These queries worked well from the start due to strong corpus coverage and natural BM25 behavior. They serve as a regression baseline: every M3 change was validated against these to ensure no degradation.

| Query | Why it works well |
|---|---|
| machine learning | Common CS topic, many relevant pages, bigram boost helps |
| cristina lopes | Faculty name, title/h1 tier boost surfaces profile page |
| informatics | Department name, strong tf-idf signal |
| software engineering | Degree program, bigram `softwar_engin` boosts exact phrase |
| ACM | Well-known acronym with clear matches |
| computer science | Broad department topic, good coverage |
| artificial intelligence | Research area, cross-page coverage |
| data structures | Course topic, bigram helps exact phrase |
| python programming | Popular language, many course pages |
| algorithm | Fundamental CS concept, high recall |

### FIXED queries (12) — previously poor, improved by M3 changes

Each query below performed poorly on relevance and/or speed before M3. The table shows the specific problem, which general-purpose fix addressed it, and how the fix works. All fixes are general heuristics — none are query-specific.

| Query | Problem (M2) | Fix (M3) | How the fix is general |
|---|---|---|---|
| master of software engineering | AND over 4 terms → 0 results | OR fallback with AND bonus | Applies to any multi-term query where AND is too strict |
| undergraduate research opportunities in machine learning | AND over 7 terms → 0 results | OR fallback with AND bonus | Same OR fallback, scales to any query length |
| ICS | Short acronym biased toward long docs | BM25 length normalization (b=0.75) | Normalizes all queries, not just short ones |
| information retrieval | No phrase awareness, scattered matches ranked high | Bigram indexing (`inform_retriev`) | All consecutive term pairs get bigrams automatically |
| dean | Body-only matches outranked actual dean's page | Tiered importance (title 3×, h1 2×) | Applies to all terms in any important HTML tag |
| fellowship | Near-identical pages cluttered top results | SimHash dedup (Hamming ≤ 3) | Filters all near-duplicates corpus-wide |
| graduate admission | Long FAQ pages outranked short authoritative pages | BM25 length normalization | Same b=0.75 normalization as the ICS fix |
| capstone project | No adjacency signal for phrase | Bigram boost | Same bigram mechanism as information retrieval |
| Rina Dechter | Name not prominent in her own page body | Anchor text indexing (tier 2) | All anchor text from all links indexed automatically |
| database management | Generic long pages ranked first | Tiered importance + BM25 | Same tier multipliers + length normalization |
| web crawler | AND returned few results | OR fallback | Same fallback mechanism |
| uci computer science ranking | 5-term AND too restrictive → 0 results | OR fallback + AND bonus | Same fallback mechanism |

### Core M3 changes (general-purpose, not query-specific)

| M2 Problem | M3 Fix | Code |
|---|---|---|
| Raw tf-idf biased toward long documents | BM25 with length normalization (k1=1.2, b=0.75) | `search.py:302` |
| No distinction between title match and body match | Tiered importance: title 3×, h1 2×, emphasis 1.5× | `search.py:24-29`, `indexer.py:41-48` |
| Boolean AND too strict for multi-word queries | OR fallback when AND < 5 results, 1.5× AND bonus | `search.py:292, 318-340` |
| Repeated disk seeks for same terms | LRU postings cache (1000 entries) | `search.py:135-154` |

### Operational constraints (developer option)

| Constraint | How it's met |
|---|---|
| Index on disk, not in memory | `index.txt` stays on disk; search seeks by byte offset via `index_of_index.txt` |
| Offload partial index ≥ 3 times | Offloaded every 18k docs → 3-4 partial files for ~55k docs (`indexer.py:342-347`) |
| Merge partial indexes | k-way heap merge into final sorted index (`indexer.py:179-231`) |
| Response time ≤ 300ms | Typically < 100ms; LRU cache, single disk seek per term |
| Small memory footprint | Only index-of-index, doc lengths, duplicates, PageRank loaded (~few MB) |

## Extra Credit Features

### 1. Near-duplicate detection (`indexer.py:75-89`, `indexer.py:234-262`)
- Computes 64-bit SimHash fingerprint per document from term frequencies
- Multi-band blocking (4 bands × 16 bits) for candidate generation
- Pairs with Hamming distance ≤ 3 are marked as duplicates
- Search filters duplicates, keeping only the canonical copy (`search.py:319-337`)

### 2. PageRank (`indexer.py:96-115`)
- Builds a link graph from `<a href>` tags during parsing
- Power iteration: 25 rounds, damping factor 0.85, dangling node redistribution
- Score added to BM25 at search time with weight 500.0 (`search.py:299-300`)

### 3. 2-gram indexing (`indexer.py:312-318`, `search.py:255-268`)
- Consecutive stemmed tokens joined with `_` (e.g., "machine learning" → `machin_learn`)
- Stored in the same inverted index as unigrams
- At search time, bigram postings looked up and BM25-scored as a bonus signal

### 4. Word position indexing + proximity scoring (`indexer.py:138-141`, `search.py:179-203`)
- Token positions stored in postings as dot-separated integers (format: `doc:tf:tier:p1.p2.p3`)
- Min-window-span algorithm finds the smallest text window containing all query terms
- Proximity bonus: `2.0 / (1 + span)` — adjacent terms score highest

### 5. Anchor text indexing (`indexer.py:144-152`, `indexer.py:361-379`)
- During parsing, `<a>` tags yield `(target_url, stemmed_anchor_tokens)`
- After all docs are parsed, anchor tokens are added to target document postings at tier 2 (h1-level importance)
- Helps pages rank for terms that describe them from other pages (e.g., "Rina Dechter")

### 6. Web interface (`web.py`)
- Flask app with landing page and search results view
- Shows ranked URLs with scores and response time
- Dark monospace theme, top 20 results displayed
- Run with `python web.py`, visit `http://localhost:5000`

## Usage

```bash
# build the index (takes ~5-10 minutes)
python indexer.py

# search from terminal
python search.py

# search from browser
python web.py
# open http://localhost:5000
```

## Dependencies

- `beautifulsoup4`, `lxml` — HTML parsing
- `nltk` — Porter stemmer
- `flask` — web interface

## Performance

- Index stays on disk (~800 MB); only metadata loaded into memory
- Partial index offloading every 18k docs keeps memory bounded during indexing
- LRU postings cache (1000 entries) speeds up repeated/similar queries
- Lazy position parsing: position data in postings is kept as raw strings and only parsed when proximity scoring needs it, avoiding hundreds of thousands of unnecessary `int()` conversions per query
- OR fallback filters low-IDF terms from candidate expansion and prioritizes docs matching 2+ terms, keeping the candidate set small without sacrificing ranking quality
- Typical query response: < 100ms (worst case < 300ms)
