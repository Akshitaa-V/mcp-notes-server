# mcp-notes-server

A real MCP (Model Context Protocol) server that exposes a note-search tool. Built as the direct follow-up to my rag-retriever-skill project — same retrieval logic, now actually wired up to the real protocol instead of just documented as a "next step."

## What it does

Runs an MCP server over stdio with one tool, `search_notes(query, top_k)`, which searches a local folder of `.txt` notes using TF-IDF and cosine similarity and returns the top-matching chunks, labeled with source and relevance score.

Any MCP-compatible client can call this tool without custom integration code — that's the whole point of the protocol. I included a real test client (`test_client.py`) that connects to the server, lists its tools, and calls `search_notes`, to prove the server actually speaks MCP correctly rather than just looking right on paper.

## Why this exists

In my first project (rag-retriever-skill), I documented how the retriever *could* be wrapped as an MCP server but didn't build it — I didn't want to overclaim something I hadn't actually done. This project closes that gap for real.

## Running it

```bash
pip install "mcp<2" scikit-learn
python3 test_client.py
```

`test_client.py` starts the server itself (over stdio) and runs a sample query end to end. Sample output:

```
Tools exposed by the server:
  - search_notes: Search the local notes folder for chunks relevant to the query...

Calling search_notes('What is a hook in an AI assistant workflow?')...

Top 2 matches for: What is a hook in an AI assistant workflow?

[1] (source: skills_hooks_and_mcp.txt, relevance: 0.349)
A hook is a way to run custom logic automatically at specific points in an assistant's workflow...
```

## Project structure

```
mcp-notes-server/
├── mcp_notes_server.py   # The actual MCP server (FastMCP, one tool)
├── test_client.py        # Real MCP client that tests it end to end
└── notes/                # Same sample notes from rag-retriever-skill
```

## What I'd add next

- More tools beyond search (e.g. listing available note files, adding new notes)
- Swapping TF-IDF for proper embeddings for semantic search
- Packaging this so it can be registered directly as an MCP server in any MCP-compatible client
