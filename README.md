# Search Engine

Boolean AND search engine with tf-idf ranking over a disk-based inverted index.

## Components

- **`indexer.py`** — Builds the inverted index from a corpus of HTML documents (`developer.zip`). Parses HTML, tokenizes and stems text (Porter stemmer), tracks term importance from HTML tags (`<title>`, `<h1>`–`<h3>`, `<b>`, `<strong>`), and writes a merged index with byte-offset lookup.
- **`search.py`** — Interactive query interface. Loads the index-of-index into memory for O(1) term lookup via disk seek. Performs boolean AND intersection starting from the smallest posting list, then ranks results using tf-idf with an importance boost.

## Usage

```
python indexer.py   # build index from developer.zip → index/
python search.py    # interactive search REPL
```

## Index Format

| File | Contents |
|---|---|
| `index/index.txt` | Merged inverted index (`term\|doc:tf:imp,...`) |
| `index/index_of_index.txt` | Term → byte offset into `index.txt` |
| `index/doc_id_map.txt` | Doc ID → URL |
| `index/metadata.txt` | Corpus stats |

## Dependencies

- `beautifulsoup4`, `lxml`, `nltk`
