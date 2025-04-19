"""
hello_server.py
A **minimal MCP server** that exposes one tool called `say_hello`.
When the tool is invoked it returns "Hello, world!" (or "Hello, <name>!").

Run locally:  python hello_server.py          # dev / manual test
Use with MCP CLI: mcp dev hello_server.py     # inspector / Claude Desktop
"""
from mcp.server.fastmcp import FastMCP

# Create the server instance
mcp = FastMCP("Hello‑World‑Server")


@mcp.tool()
def say_hello(name: str | None = None) -> str:
    """
    input   : optional name
    process : build greeting string
    output  : greeting text
    """
    return f"Hello, {name}!" if name else "Hello, world!"


# Allows `python hello_server.py` to behave like `mcp run hello_server.py`
if __name__ == "__main__":
    mcp.run()
