"""Web interface for the search engine — Flask app with ranked results."""

import atexit
import threading
import time

from flask import Flask, request, render_template

from search import SearchEngine

app = Flask(__name__)

# ── load index once at startup ────────────────────────────────────────────────
engine = SearchEngine()
atexit.register(engine.close)
search_lock = threading.Lock()
print(f"Ready. {len(engine.ioi)} terms, {engine.total_docs} docs")


@app.route("/")
def index():
    """Render the landing page."""
    return render_template(
        "search.html", query="", results=[], num_results=0, elapsed_ms=0,
    )


@app.route("/search")
def search_page():
    """Handle search queries and render ranked results."""
    query = request.args.get("q", "").strip()
    if not query:
        return render_template(
            "search.html", query="", results=[], num_results=0, elapsed_ms=0,
        )

    t_start = time.time()
    with search_lock:
        results = engine.query(query)
    elapsed_ms = f"{(time.time() - t_start) * 1000:.1f}"

    return render_template(
        "search.html",
        query=query,
        results=results[:20],
        num_results=len(results),
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
