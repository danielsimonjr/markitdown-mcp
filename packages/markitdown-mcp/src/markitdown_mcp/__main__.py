import contextlib
import sys
import os
import markdown
from urllib.parse import urlparse, unquote
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Optional
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from markitdown import MarkItDown
from anthropic import Anthropic
import uvicorn

# Initialize FastMCP server for MarkItDown (SSE)
mcp = FastMCP("markitdown")

# Global configuration for Anthropic integration
_anthropic_client: Optional[Anthropic] = None
_llm_model: str = "claude-haiku-4-5-20251001"

# Default path to API key file (fallback if ANTHROPIC_API_KEY env var not set)
DEFAULT_API_KEY_FILE = "C:/mcp-servers/Claude_Key.txt"


def get_api_key() -> Optional[str]:
    """Get API key from environment variable or fallback to key file."""
    # First try environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return api_key.strip()

    # Fallback to key file
    key_file = os.getenv("ANTHROPIC_API_KEY_FILE", DEFAULT_API_KEY_FILE)
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                return f.read().strip()
        except Exception:
            pass

    return None


def get_anthropic_client() -> Optional[Anthropic]:
    """Get or create the Anthropic client if API key is available."""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = get_api_key()
        if api_key:
            _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def check_plugins_enabled() -> bool:
    return os.getenv("MARKITDOWN_ENABLE_PLUGINS", "false").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def get_markitdown_instance() -> MarkItDown:
    """Create a MarkItDown instance with optional Anthropic LLM support for image descriptions."""
    client = get_anthropic_client()
    if client:
        return MarkItDown(
            enable_plugins=check_plugins_enabled(),
            llm_client=client,
            llm_model=_llm_model
        )
    return MarkItDown(enable_plugins=check_plugins_enabled())


def markdown_to_html(md_content: str, title: str = "Document") -> str:
    """Convert markdown to self-contained HTML with inline CSS."""
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
            background-color: #fff;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            color: #111;
        }}
        h1 {{ font-size: 2em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
        p {{ margin: 1em 0; }}
        code {{
            background-color: #f4f4f4;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            margin: 1em 0;
            padding-left: 1em;
            color: #666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 0.5em;
            text-align: left;
        }}
        th {{
            background-color: #f4f4f4;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        ul, ol {{
            margin: 1em 0;
            padding-left: 2em;
        }}
        li {{
            margin: 0.25em 0;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>'''


@mcp.tool()
async def convert_to_markdown(uri: str) -> str:
    """Convert a resource described by an http:, https:, file: or data: URI to markdown.

    When ANTHROPIC_API_KEY is set, images will be processed using Claude for
    intelligent descriptions of technical diagrams, charts, and visual content.

    For file:// URIs, the markdown output is also saved to a .md file in the same folder.
    """
    md_content = get_markitdown_instance().convert_uri(uri).markdown

    # For file:// URIs, write both .md and .html to the same folder
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        # Extract file path from URI (handle Windows paths)
        file_path = unquote(parsed.path)
        if file_path.startswith("/") and len(file_path) > 2 and file_path[2] == ":":
            file_path = file_path[1:]  # Remove leading slash for Windows paths like /C:/...

        source_path = Path(file_path)
        md_path = source_path.with_suffix(".md")
        html_path = source_path.with_suffix(".html")

        # Write markdown file
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        except Exception as e:
            return f"Error writing {md_path}: {e}\n\n{md_content}"

        # Write HTML file
        try:
            html_content = markdown_to_html(md_content, title=source_path.stem)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            return f"Error writing {html_path}: {e}\n\n{md_content}"

        return f"Saved to:\n  - {md_path}\n  - {html_path}\n\n{md_content}"

    return md_content


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
    )

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        async with session_manager.run():
            print("Application started with StreamableHTTP session manager!")
            if get_anthropic_client():
                print(f"Anthropic integration enabled with model: {_llm_model}")
            else:
                print("Anthropic integration disabled (no ANTHROPIC_API_KEY set)")
            try:
                yield
            finally:
                print("Application shutting down...")

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/mcp", app=handle_streamable_http),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=lifespan,
    )


# Main entry point
def main():
    global _llm_model
    import argparse

    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description="Run a MarkItDown MCP server with Anthropic Claude integration")

    parser.add_argument(
        "--http",
        action="store_true",
        help="Run the server with Streamable HTTP and SSE transport rather than STDIO (default: False)",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="(Deprecated) An alias for --http (default: False)",
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (default: 3001)"
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model to use for image descriptions (default: claude-haiku-4-5-20251001)",
        choices=[
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-4-5-20251001",
        ]
    )
    args = parser.parse_args()

    # Set the model from CLI argument
    _llm_model = args.model

    use_http = args.http or args.sse

    if not use_http and (args.host or args.port):
        parser.error(
            "Host and port arguments are only valid when using streamable HTTP or SSE transport (see: --http)."
        )
        sys.exit(1)

    # Print status
    if get_anthropic_client():
        key_source = "environment variable" if os.getenv("ANTHROPIC_API_KEY") else f"key file ({DEFAULT_API_KEY_FILE})"
        print(f"Anthropic integration enabled with model: {_llm_model}")
        print(f"API key loaded from: {key_source}")
    else:
        print("Note: Set ANTHROPIC_API_KEY or place key in C:/mcp-servers/Claude_Key.txt to enable Claude image descriptions")

    if use_http:
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(
            starlette_app,
            host=args.host if args.host else "127.0.0.1",
            port=args.port if args.port else 3001,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
