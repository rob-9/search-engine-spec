"""
web interface for search engine — assignment 3 m3
flask app with search box and ranked results.
"""

import atexit
import time
from pathlib import Path

from flask import Flask, request, render_template_string

from search import (
    load_index_of_index, load_doc_id_map, load_doc_lengths,
    load_duplicates, load_pagerank, load_total_docs, PostingsCache, search,
    INDEX_DIR,
)

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>A3 Search Engine</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #333; }
        .search-box { width: 100%; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .search-btn { padding: 12px 24px; font-size: 16px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 8px; }
        .search-btn:hover { background: #3367d6; }
        .meta { color: #666; margin: 16px 0; font-size: 14px; }
        .result { margin: 16px 0; }
        .result a { color: #1a0dab; font-size: 16px; text-decoration: none; }
        .result a:hover { text-decoration: underline; }
        .result .score { color: #006621; font-size: 13px; }
        .result .url { color: #006621; font-size: 14px; }
    </style>
</head>
<body>
    <h1>A3 Search Engine</h1>
    <form action="/search" method="get">
        <input class="search-box" type="text" name="q" value="{{ query }}" placeholder="Enter search query..." autofocus>
        <button class="search-btn" type="submit">Search</button>
    </form>
    {% if query %}
    <div class="meta">{{ num_results }} results in {{ elapsed_ms }}ms</div>
    {% for url, score in results %}
    <div class="result">
        <a href="{{ url }}" target="_blank">{{ url }}</a>
        <div class="score">Score: {{ "%.4f"|format(score) }}</div>
    </div>
    {% endfor %}
    {% if not results %}
    <p>No results found.</p>
    {% endif %}
    {% endif %}
</body>
</html>
"""

# load index data at module level
print("Loading index...")
ioi = load_index_of_index()
doc_map = load_doc_id_map()
total_docs = load_total_docs()
doc_lengths = load_doc_lengths()
duplicates = load_duplicates()
pagerank = load_pagerank()
url_to_did = {url: did for did, url in doc_map.items()}
index_fh = open(INDEX_DIR / "index.txt", "rb")
cache = PostingsCache()

if doc_lengths:
    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
else:
    avgdl = 1.0

atexit.register(index_fh.close)
print(f"Ready. {len(ioi)} terms, {total_docs} docs")


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, query="", results=[], num_results=0, elapsed_ms=0)


@app.route("/search")
def search_page():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template_string(HTML_TEMPLATE, query="", results=[], num_results=0, elapsed_ms=0)

    t_start = time.time()
    results = search(query, ioi, doc_map, total_docs, index_fh,
                     doc_lengths, avgdl, duplicates, cache, url_to_did, pagerank)
    elapsed_ms = f"{(time.time() - t_start) * 1000:.1f}"

    top_results = results[:20]

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=top_results,
        num_results=len(results),
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
