"""
test_client.py
==============

A real MCP client that connects to mcp_notes_server.py over stdio,
lists its tools, and calls search_notes with a sample query. This proves
the server actually speaks the MCP protocol correctly, rather than just
having the right-looking code.

Run:
    python3 test_client.py
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_notes_server.py")


async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("Tools exposed by the server:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\nCalling search_notes('What is a hook in an AI assistant workflow?')...\n")
            result = await session.call_tool(
                "search_notes",
                arguments={"query": "What is a hook in an AI assistant workflow?", "top_k": 2},
            )
            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
