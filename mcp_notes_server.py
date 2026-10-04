"""
mcp_notes_server.py
====================

A real, minimal MCP (Model Context Protocol) server. It exposes one tool,
`search_notes`, over the standard MCP stdio transport, so any MCP-compatible
client (an IDE assistant, a desktop chat app or any other MCP client) can call it
without custom integration code.

This is the direct extension of the rag-retriever-skill project: the same
TF-IDF chunking-and-retrieval logic, now exposed as an actual MCP tool
instead of just documented as a future step.

No external API calls, no paid services, no signups. Everything runs
locally over stdio.

Run it directly to test:
    python3 mcp_notes_server.py

It will then wait on stdio for MCP protocol messages -- it's not meant to
print anything on its own. Use an MCP client, or the included test script
(test_client.py), to actually call the tool.
"""

import glob
import os
import re

from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NOTES_DIR = os.path.join(os.path.dirname(__file__), "notes")

mcp = FastMCP("notes-search")


def _load_documents(notes_dir: str) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(notes_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def _chunk_text(text: str, max_words: int = 60) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
    return chunks


@mcp.tool()
def search_notes(query: str, top_k: int = 3) -> str:
    """
    Search the local notes folder for chunks relevant to the query, and
    return them as a labelled context block.

    Args:
        query: The question or topic to search the notes for.
        top_k: How many top-matching chunks to return (default 3).
    """
    documents = _load_documents(NOTES_DIR)
    chunk_records = []
    for doc in documents:
        for chunk in _chunk_text(doc["text"]):
            chunk_records.append({"source": doc["source"], "chunk": chunk})

    if not chunk_records:
        return "No notes found to search."

    corpus = [r["chunk"] for r in chunk_records]
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(corpus)

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]

    lines = [f"Top {top_k} matches for: {query}\n"]
    for rank, idx in enumerate(ranked_indices, start=1):
        record = chunk_records[idx]
        lines.append(
            f"[{rank}] (source: {record['source']}, relevance: {scores[idx]:.3f})\n{record['chunk']}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
