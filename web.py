"""
web interface for search engine — assignment 3 m3
flask app with search box and ranked results.
"""

import atexit
import threading
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
    <title>{% if query %}{{ query }} — a3.search{% else %}a3.search{% endif %}</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg:        #111118;
            --bg-card:   #18181f;
            --bg-input:  #13131a;
            --border:    #26263a;
            --border-hi: #3a3a56;
            --accent:    #c97c4a;
            --accent-hi: #e09268;
            --text:      #d0d0d8;
            --text-dim:  #6a6a88;
            --text-mute: #404058;
            --link:      #7eaff5;
            --link-hi:   #a8c8fa;
            --rank-bg:   #1e1e2c;
            --rank-fg:   #5a5a7a;
            --green:     #5a9e6f;
        }

        html { scroll-behavior: smooth; }

        body {
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code',
                         'Consolas', 'Courier New', monospace;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            -webkit-font-smoothing: antialiased;
            overflow: hidden;
        }

        /* ─────────────────────────────── layout ── */
        .page-wrap {
            flex: 1;
            width: 100%;
            max-width: 960px;
            margin: 0 auto;
            padding: 0 24px;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* ─────────────────────────────── landing ── */
        .landing {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
            gap: 0;
            animation: fadeUp 0.45s ease both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .logo-wrap { text-align: center; margin-bottom: 32px; }

        .logo {
            font-size: 2.6rem;
            font-weight: 700;
            color: #eaeaf0;
            letter-spacing: -1px;
            line-height: 1;
        }
        .logo .dot { color: var(--accent); }

        .tagline {
            margin-top: 10px;
            font-size: 0.78rem;
            color: var(--text-dim);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .landing .search-form { width: 100%; max-width: 560px; }

        /* keyboard hint */
        .hint {
            margin-top: 14px;
            font-size: 0.72rem;
            color: var(--text-mute);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .kbd {
            display: inline-block;
            padding: 1px 6px;
            border: 1px solid var(--border-hi);
            border-radius: 4px;
            font-size: 0.70rem;
            color: var(--text-dim);
            background: var(--bg-card);
        }

        /* ─────────────────────────────── top bar ── */
        .topbar {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 18px 0 14px;
            border-bottom: 1px solid var(--border);
            animation: fadeDown 0.3s ease both;
        }

        @keyframes fadeDown {
            from { opacity: 0; transform: translateY(-8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .topbar .logo {
            font-size: 1.15rem;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .topbar .logo { text-decoration: none; }
        .topbar .search-form { flex: 1; margin-top: 0; }

        /* ─────────────────────────────── search form ── */
        .search-form {
            display: flex;
            align-items: stretch;
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .search-form:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(201,124,74,0.12);
        }

        .search-box {
            flex: 1;
            padding: 13px 16px;
            font-size: 0.92rem;
            font-family: inherit;
            background: var(--bg-input);
            color: #eaeaf0;
            border: none;
            outline: none;
            min-width: 0;
        }
        .search-box::placeholder { color: var(--text-mute); }

        .search-box:-webkit-autofill,
        .search-box:-webkit-autofill:hover,
        .search-box:-webkit-autofill:focus {
            -webkit-text-fill-color: #eaeaf0;
            -webkit-box-shadow: 0 0 0 1000px var(--bg-input) inset;
            transition: background-color 5000s ease-in-out 0s;
        }

        .search-btn {
            padding: 13px 22px;
            font-size: 0.85rem;
            font-family: inherit;
            font-weight: 600;
            background: var(--accent);
            color: #111118;
            border: none;
            cursor: pointer;
            letter-spacing: 0.03em;
            transition: background 0.18s, transform 0.12s;
            flex-shrink: 0;
        }
        .search-btn:hover  { background: var(--accent-hi); }
        .search-btn:active { transform: scale(0.97); }

        /* ─────────────────────────────── meta bar ── */
        .meta {
            padding: 12px 0 10px;
            font-size: 0.75rem;
            color: var(--text-dim);
            display: flex;
            align-items: center;
            gap: 4px;
            flex-wrap: wrap;
        }
        .meta-count { color: var(--text); font-weight: 600; }
        .meta-sep   { color: var(--text-mute); margin: 0 4px; }
        .meta-time  { color: var(--green); font-weight: 600; }

        /* ─────────────────────────────── results list ── */
        .results-list {
            padding: 6px 0 20px;
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            min-height: 0;
        }

        .result {
            display: grid;
            grid-template-columns: 36px 1fr;
            gap: 0 10px;
            align-items: start;
            padding: 13px 10px;
            border-radius: 8px;
            margin: 3px -10px;
            transition: background 0.14s;
            animation: fadeIn 0.25s ease both;
        }

        {% for i in range(20) %}
        .result:nth-child({{ i + 1 }}) { animation-delay: {{ i * 0.03 }}s; }
        {% endfor %}

        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-6px); }
            to   { opacity: 1; transform: translateX(0); }
        }

        .result:hover { background: var(--bg-card); }

        /* rank badge */
        .result-rank {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            border-radius: 6px;
            background: var(--rank-bg);
            color: var(--rank-fg);
            font-size: 0.70rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 1px;
            border: 1px solid var(--border);
            transition: background 0.14s, color 0.14s;
        }
        .result:hover .result-rank {
            background: var(--border);
            color: var(--text-dim);
        }
        /* top 3 get a subtle accent tint */
        .result:nth-child(1) .result-rank { color: var(--accent); border-color: rgba(201,124,74,0.3); }
        .result:nth-child(2) .result-rank { color: #8888aa; border-color: #333348; }
        .result:nth-child(3) .result-rank { color: #7a6e55; border-color: #333328; }

        .result-body { min-width: 0; overflow: hidden; }

        /* domain line */
        .result-domain {
            font-size: 0.72rem;
            color: var(--text-dim);
            margin-bottom: 3px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .result-domain .scheme { color: var(--text-mute); }
        .result-domain .host   { color: var(--text-dim); }
        .result-domain .path   { color: var(--text-mute); }

        /* main link */
        .result-url {
            display: block;
            color: var(--link);
            font-size: 0.87rem;
            text-decoration: none;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: color 0.14s;
        }
        .result-url:hover {
            color: var(--link-hi);
            text-decoration: underline;
            text-underline-offset: 3px;
            text-decoration-color: var(--border-hi);
        }

        /* score pill */
        .result-score {
            margin-top: 5px;
            font-size: 0.68rem;
            color: var(--text-mute);
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .result-score::before {
            content: '';
            display: inline-block;
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--text-mute);
            opacity: 0.4;
        }

        /* divider between results */
        .result + .result { border-top: 1px solid transparent; }
        .result + .result:not(:hover) { border-color: var(--border); }

        /* ─────────────────────────────── empty / no-results ── */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            padding: 64px 20px 48px;
            text-align: center;
            animation: fadeIn 0.3s ease both;
        }

        .empty-icon {
            font-size: 2.4rem;
            opacity: 0.25;
            line-height: 1;
        }

        .empty-title {
            font-size: 0.95rem;
            color: var(--text-dim);
            font-weight: 600;
        }

        .empty-sub {
            font-size: 0.78rem;
            color: var(--text-mute);
            max-width: 340px;
            line-height: 1.6;
        }

        .empty-suggestions {
            margin-top: 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
        }

        .suggestion-chip {
            padding: 5px 12px;
            border: 1px solid var(--border-hi);
            border-radius: 999px;
            font-size: 0.75rem;
            color: var(--text-dim);
            cursor: pointer;
            background: none;
            font-family: inherit;
            transition: border-color 0.15s, color 0.15s, background 0.15s;
            text-decoration: none;
        }
        .suggestion-chip:hover {
            border-color: var(--accent);
            color: var(--accent-hi);
            background: rgba(201,124,74,0.06);
        }

        /* ─────────────────────────────── footer ── */
        footer {
            border-top: 1px solid var(--border);
            padding: 16px 20px;
            font-size: 0.70rem;
            color: var(--text-mute);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
            text-align: center;
            flex-shrink: 0;
            overflow: hidden;
        }
        .footer-sep { opacity: 0.3; }
        footer a {
            color: var(--text-dim);
            text-decoration: none;
        }
        footer a:hover { color: var(--text); }

        /* ─────────────────────────────── scrollbar ── */
        .results-list::-webkit-scrollbar { width: 6px; }
        .results-list::-webkit-scrollbar-track { background: transparent; }
        .results-list::-webkit-scrollbar-thumb {
            background: var(--border-hi);
            border-radius: 3px;
        }
        .results-list::-webkit-scrollbar-thumb:hover { background: var(--text-mute); }

        /* ─────────────────────────────── responsive ── */
        @media (max-width: 600px) {
            .topbar {
                flex-wrap: wrap;
                gap: 10px;
            }
            .topbar .logo { font-size: 1rem; }
            .search-btn { padding: 13px 16px; }
            .result { grid-template-columns: 28px 1fr; gap: 0 8px; }
            .result-rank { width: 22px; height: 22px; font-size: 0.65rem; }
            .result-url { font-size: 0.82rem; }
            footer { gap: 8px; }
        }

        @media (max-width: 400px) {
            .landing .logo { font-size: 2rem; }
            .page-wrap { padding: 0 14px; }
        }
    </style>
</head>
<body>

{% if not query %}
<!-- ═══════════════════════════════════ LANDING PAGE ══════════════════════════════════ -->
<div class="page-wrap">
    <div class="landing">
        <div class="logo-wrap">
            <div class="logo">a3<span class="dot">.</span>search</div>
            <div class="tagline">ICS web corpus &mdash; 55,393 documents</div>
        </div>
        <form class="search-form" action="/search" method="get">
            <input class="search-box" type="text" name="q" placeholder="type a query and press Enter..." autofocus autocomplete="off" spellcheck="false">
            <button class="search-btn" type="submit">Search</button>
        </form>
        <div class="hint">
            <span class="kbd">Enter</span> to search &nbsp;&middot;&nbsp;
            <span class="kbd">/</span> to focus
        </div>
    </div>
</div>

{% else %}
<!-- ═══════════════════════════════════ RESULTS PAGE ═════════════════════════════════ -->
<div class="page-wrap">
    <nav class="topbar">
        <a href="/" style="text-decoration:none">
            <span class="logo">a3<span class="dot">.</span>search</span>
        </a>
        <form class="search-form" action="/search" method="get">
            <input class="search-box" type="text" name="q" value="{{ query|e }}" placeholder="search..." id="q" autocomplete="off" spellcheck="false">
            <button class="search-btn" type="submit">Search</button>
        </form>
    </nav>

    {% if results %}
    <div class="meta">
        <span class="meta-count">{{ num_results }}</span>
        <span>&nbsp;results</span>
        <span class="meta-sep">&middot;</span>
        <span class="meta-time">{{ elapsed_ms }}ms</span>
        <span class="meta-sep">&middot;</span>
        <span>showing top {{ results|length }}</span>
    </div>

    <div class="results-list">
        {% for url, score in results %}
        {# ── parse URL into parts for display ── #}
        {% set parts = url.split('://') %}
        {% if parts|length > 1 %}
            {% set scheme = parts[0] %}
            {% set rest   = parts[1] %}
        {% else %}
            {% set scheme = 'https' %}
            {% set rest   = url %}
        {% endif %}
        {% set slash_idx = rest.find('/') %}
        {% if slash_idx > -1 %}
            {% set host = rest[:slash_idx] %}
            {% set path = rest[slash_idx:] %}
        {% else %}
            {% set host = rest %}
            {% set path = '' %}
        {% endif %}
        {# truncate path to at most 60 chars #}
        {% if path|length > 60 %}
            {% set path_display = path[:57] + '...' %}
        {% else %}
            {% set path_display = path %}
        {% endif %}

        <div class="result">
            <div class="result-rank">{{ loop.index }}</div>
            <div class="result-body">
                <div class="result-domain">
                    <span class="scheme">{{ scheme }}://</span><span class="host">{{ host }}</span><span class="path">{{ path_display }}</span>
                </div>
                <a class="result-url" href="{{ url }}" target="_blank" rel="noopener" title="{{ url }}">{{ url }}</a>
                <div class="result-score">score {{ "%.4f"|format(score) }}</div>
            </div>
        </div>
        {% endfor %}
    </div>

    {% else %}
    <!-- no results -->
    <div class="empty-state">
        <div class="empty-icon">&#x2205;</div>
        <div class="empty-title">No results for &ldquo;{{ query|e }}&rdquo;</div>
        <div class="empty-sub">
            No documents matched all query terms. Try fewer words,
            check spelling, or search for something else.
        </div>
        <div class="empty-suggestions">
            <a class="suggestion-chip" href="/search?q=machine+learning">machine learning</a>
            <a class="suggestion-chip" href="/search?q=cristina+lopes">cristina lopes</a>
            <a class="suggestion-chip" href="/search?q=ACM">ACM</a>
            <a class="suggestion-chip" href="/search?q=software+engineering">software engineering</a>
        </div>
    </div>
    {% endif %}
</div>
{% endif %}

<!-- ═══════════════════════════════════ FOOTER ═══════════════════════════════════════ -->
<footer>
    <span>a3<span style="color:var(--accent)">.</span>search</span>
    <span class="footer-sep">|</span>
    <span>55,393 documents indexed</span>
    <span class="footer-sep">|</span>
    <span>ICS web corpus &mdash; 88 sub-domains</span>
    <span class="footer-sep">|</span>
    <span>BM25 &middot; PageRank &middot; SimHash</span>
</footer>

<script>
    /* on results page, focus search box with cursor at end */
    var q = document.getElementById('q');
    if (q) { q.focus(); q.setSelectionRange(q.value.length, q.value.length); }

    /* "/" to focus search box */
    document.addEventListener('keydown', function(e) {
        if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
            e.preventDefault();
            var box = document.querySelector('.search-box');
            if (box) { box.focus(); box.select(); }
        }
    });
</script>

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
search_lock = threading.Lock()
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
    with search_lock:
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
