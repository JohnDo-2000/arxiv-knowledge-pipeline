"""
query_demo.py -- Demonstrates semantic search + metadata filtering
against the Qdrant collection populated by run_local_demo.py.

Run after run_local_demo.py has upserted at least one document:

    python scripts/query_demo.py "your search query here"
    python scripts/query_demo.py "your search query" --category cs.CL
    python scripts/query_demo.py "your search query" --after 2024-01-01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from etl.embed import embed_texts  # noqa: E402
from vectorstore.qdrant_client import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search demo with metadata filters.")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default=None, help="Filter by arXiv category, e.g. cs.CL")
    parser.add_argument("--author", default=None, help="Filter by author name")
    parser.add_argument("--after", default=None, help="Filter to papers published after this date (YYYY-MM-DD)")
    parser.add_argument("--before", default=None, help="Filter to papers published before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    store = VectorStore(url="http://localhost:6333")

    query_vector = embed_texts([args.query])[0].tolist()

    results = store.search(
        query_vector,
        top_k=args.top_k,
        category=args.category,
        author=args.author,
        published_after=args.after,
        published_before=args.before,
    )

    if not results:
        print("No results found. Have you run scripts/run_local_demo.py yet?")
        return

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
    for i, r in enumerate(results, 1):
        payload = r["payload"]
        print(f"{i}. score={r['score']:.4f}  [{payload.get('category', '?')}]  "
              f"{payload.get('title', 'Untitled')}")
        print(f"   chunk: {payload.get('text', '')[:160].replace(chr(10), ' ')}...")
        print()


if __name__ == "__main__":
    main()
