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
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>A3 Search</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace;
            background: #1a1a2e;
            color: #d4d4d8;
            min-height: 100vh;
        }

        .container {
            max-width: 760px;
            margin: 0 auto;
            padding: 0 24px;
        }

        /* ── header ── */
        .header {
            padding: 48px 0 32px;
            text-align: center;
        }

        .header.has-results {
            padding: 24px 0 16px;
            text-align: left;
        }

        .logo {
            font-size: 28px;
            font-weight: 700;
            color: #e8e8ed;
            letter-spacing: -0.5px;
        }

        .logo span {
            color: #d4875e;
        }

        .header.has-results .logo {
            font-size: 20px;
        }

        /* ── search form ── */
        .search-form {
            display: flex;
            gap: 0;
            margin-top: 24px;
        }

        .header.has-results .search-form {
            margin-top: 12px;
        }

        .search-box {
            flex: 1;
            padding: 14px 18px;
            font-size: 15px;
            font-family: inherit;
            background: #16162a;
            color: #e8e8ed;
            border: 1px solid #2e2e4a;
            border-right: none;
            border-radius: 8px 0 0 8px;
            outline: none;
            transition: border-color 0.2s;
        }

        .search-box::placeholder {
            color: #52526e;
        }

        .search-box:focus {
            border-color: #d4875e;
        }

        .search-box:-webkit-autofill,
        .search-box:-webkit-autofill:hover,
        .search-box:-webkit-autofill:focus {
            -webkit-text-fill-color: #e8e8ed;
            -webkit-box-shadow: 0 0 0 1000px #16162a inset;
            transition: background-color 5000s ease-in-out 0s;
        }

        .search-btn {
            padding: 14px 28px;
            font-size: 15px;
            font-family: inherit;
            font-weight: 600;
            background: #d4875e;
            color: #1a1a2e;
            border: 1px solid #d4875e;
            border-radius: 0 8px 8px 0;
            cursor: pointer;
            transition: background 0.2s;
        }

        .search-btn:hover {
            background: #e09870;
        }

        /* ── results meta ── */
        .meta {
            padding: 16px 0 12px;
            font-size: 13px;
            color: #6e6e8a;
            border-bottom: 1px solid #2e2e4a;
            margin-bottom: 8px;
        }

        .meta strong {
            color: #9e9eb8;
            font-weight: 600;
        }

        /* ── result items ── */
        .result {
            padding: 14px 16px;
            border-radius: 8px;
            transition: background 0.15s;
            margin: 2px 0;
        }

        .result:hover {
            background: #16162a;
        }

        .result-rank {
            display: inline-block;
            width: 24px;
            font-size: 12px;
            color: #52526e;
            font-weight: 600;
        }

        .result a {
            color: #8ab4f8;
            font-size: 15px;
            text-decoration: none;
            word-break: break-all;
        }

        .result a:hover {
            color: #aecbfa;
            text-decoration: underline;
            text-decoration-color: #4a4a6a;
            text-underline-offset: 3px;
        }

        .result-details {
            margin-top: 4px;
            padding-left: 24px;
            font-size: 12px;
            color: #52526e;
        }

        .no-results {
            text-align: center;
            padding: 48px 0;
            color: #6e6e8a;
            font-size: 14px;
        }

        /* ── landing state ── */
        .landing {
            text-align: center;
            padding-top: 20vh;
        }

        .landing .logo {
            font-size: 36px;
        }

        .landing .tagline {
            margin-top: 8px;
            font-size: 14px;
            color: #6e6e8a;
        }
    </style>
</head>
<body>
    <div class="container">
        {% if not query %}
        <div class="landing">
            <div class="logo">a3<span>.</span>search</div>
            <div class="tagline">search the web</div>
            <form class="search-form" action="/search" method="get">
                <input class="search-box" type="text" name="q" placeholder="search..." autofocus>
                <button class="search-btn" type="submit">Go</button>
            </form>
        </div>
        {% else %}
        <div class="header has-results">
            <a href="/" style="text-decoration:none"><div class="logo">a3<span>.</span>search</div></a>
            <form class="search-form" action="/search" method="get">
                <input class="search-box" type="text" name="q" value="{{ query }}" placeholder="search..." autofocus>
                <button class="search-btn" type="submit">Go</button>
            </form>
        </div>
        <div class="meta"><strong>{{ num_results }}</strong> results in <strong>{{ elapsed_ms }}ms</strong></div>
        {% for url, score in results %}
        <div class="result">
            <span class="result-rank">{{ loop.index }}</span>
            <a href="{{ url }}" target="_blank">{{ url }}</a>
            <div class="result-details">{{ "%.4f"|format(score) }}</div>
        </div>
        {% endfor %}
        {% if not results %}
        <div class="no-results">No results found.</div>
        {% endif %}
        {% endif %}
    </div>
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
    """render the landing page with empty search box."""
    return render_template_string(HTML_TEMPLATE, query="", results=[], num_results=0, elapsed_ms=0)


@app.route("/search")
def search_page():
    """handle search queries and render ranked results."""
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
