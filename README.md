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

## M3 Improvements

### Core changes from M2 → M3

| M2 Problem | M3 Fix | Code |
|---|---|---|
| Raw tf-idf biased toward long documents | BM25 with length normalization (k1=1.2, b=0.75) | `search.py:289` |
| No distinction between title match and body match | Tiered importance: title 3×, h1 2×, emphasis 1.5× | `search.py:24-29`, `indexer.py:36-48` |
| Boolean AND too strict for multi-word queries | OR fallback when AND < 5 results, 1.5× AND bonus | `search.py:280,305-314` |
| Repeated disk seeks for same terms | LRU postings cache (1000 entries) | `search.py:126-145` |

### Problems found via test queries and how they were fixed

| Query example | Problem | Fix |
|---|---|---|
| "master of software engineering" | AND over 4+ terms returned 0 results | OR fallback finds partial matches |
| "ICS" | Short acronym biased toward longest docs | BM25 normalizes by doc length |
| "information retrieval" | No phrase awareness, scattered matches ranked high | Bigram `inform_retriev` boosts exact phrase |
| "dean" | Body-only matches outranked the actual dean's page | Title/h1 tier boost surfaces authoritative page |
| "fellowship" | Near-identical pages cluttered results | SimHash dedup filters copies |
| "graduate admission" | Long FAQ pages outranked short focused pages | BM25 length normalization |
| "Rina Dechter" | Only found if exact AND match in body text | Anchor text from linking pages adds signal |
| "capstone project" | No adjacency signal | Bigram boost for consecutive terms |
| "uci computer science ranking" | 5-term AND too restrictive, 0 results | OR fallback + AND bonus for best matches |

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

- Index stays on disk (~500+ MB); only metadata loaded into memory
- Partial index offloading every 18k docs keeps memory bounded during indexing
- LRU postings cache (1000 entries) speeds up repeated/similar queries
- Typical query response: < 100ms
