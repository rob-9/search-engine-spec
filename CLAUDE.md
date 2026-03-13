# A3 Search Engine — Project Context

## Assignment Spec

Search engine built in a group of 4 (IDs: 35485800, 79822855, 30679988, 32438497).

**Flavor:** Algorithms and Data Structures Developer (CS/SE students — required).

**Corpus:** `developer.zip` — 88 sub-domains, ~56k pages of ICS web content. JSON files with `url`, `content`, `encoding` fields.

### Requirements

**Indexer:**
- Tokens: all alphanumeric sequences
- No stop word removal
- Porter stemming
- Important words: bold, headings (h1-h3), titles weighted higher
- Index stored in files on disk (no databases)
- Must offload partial index to disk at least 3 times during construction, then merge
- Search must NOT load full inverted index into memory — read postings from disk

**Search:**
- Prompt user for query, stem terms, look up index, rank results
- Ranking: tf-idf minimum, important words factored in
- Response time ≤ 300ms (ideally ≤ 100ms)
- Boolean AND at minimum

**Extra credit options (all implemented):**
1. Near-duplicate detection via SimHash (2 pts)
2. PageRank (2.5 pts)
3. 2-gram indexing (1 pt)
4. Word position indexing + proximity scoring (2 pts)
5. Anchor text indexing (1 pt)
6. Web interface via Flask (2 pts)

### Milestones

| Milestone | Status | What |
|---|---|---|
| M1 (due Feb 20) | Done | Index construction, report with analytics |
| M2 (due Feb 27) | Done | Retrieval component, boolean AND, top 5 URLs for test queries |
| M3 (due Mar 14) | Implementing | Final search engine, 20+ test queries, ranking improvements, live demo |

### M3 Deliverables
- Zip of all code
- Document with 20+ test queries: half that started poorly + explanation of fixes, half that work well
- Live demo with TA: screen share, walk through code, answer detailed questions about implementation and design choices

### M3 Evaluation Criteria
- Does search work as expected?
- How general are the heuristics?
- Response time under 300ms?
- Can you demonstrate in-depth knowledge and justify choices?

### M2 Test Queries (must work)
1. cristina lopes
2. machine learning
3. ACM
4. master of software engineering

### Late Policy
M3: 3 days late max, 25% penalty. M1/M2: no late submissions.
