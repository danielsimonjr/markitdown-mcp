# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MarkItDown is a Microsoft open-source project for converting various file formats to Markdown. It's organized as a monorepo with three packages:

- **markitdown** - Core conversion library (Python)
- **markitdown-mcp** - MCP server wrapper for MarkItDown
- **markitdown-sample-plugin** - Example plugin demonstrating extensibility

## Build & Test Commands

### Core Library (packages/markitdown)

```bash
# Setup
cd packages/markitdown
pip install -e '.[all]'    # Install with all optional deps

# Using Hatch (preferred)
hatch test                    # Run all tests
hatch test -k test_cli_misc   # Run specific test
hatch run types:check         # Type checking
hatch run pre-commit run --all-files  # Format code
```

### MCP Server (packages/markitdown-mcp)

```bash
cd packages/markitdown-mcp
pip install -e .

# Run the server
python -m markitdown_mcp          # STDIO mode (default)
python -m markitdown_mcp --http   # HTTP mode
python -m markitdown_mcp --http --host 127.0.0.1 --port 3001
python -m markitdown_mcp --sse    # SSE mode

# With Anthropic Claude for image descriptions
export ANTHROPIC_API_KEY="your-api-key"
python -m markitdown_mcp --model claude-sonnet-4-20250514
```

### Interactive MCP Testing

```bash
npx @modelcontextprotocol/inspector
# Connect via STDIO: "markitdown-mcp"
# Connect via HTTP: "http://127.0.0.1:3001/mcp"
```

## Architecture

### Converter Pattern (packages/markitdown)

The core uses a **Converter Factory with Plugin System**:

1. **Main Class:** `MarkItDown` in `src/markitdown/_markitdown.py` orchestrates conversion
2. **Base Class:** `DocumentConverter` in `_base_converter.py` - all converters inherit from this
3. **Converters:** Located in `src/markitdown/converters/`, each implements:
   - `accepts(file_stream, stream_info, **kwargs) -> bool`
   - `convert(file_stream, stream_info, **kwargs) -> DocumentConverterResult`
4. **Plugin System:** Loaded via entry point `markitdown.plugin`, plugins implement `register_converters()`

### MCP Server Pattern (packages/markitdown-mcp)

Uses **FastMCP with Multiple Transports**:

- Entry point: `src/markitdown_mcp/__main__.py`
- Single tool exposed: `convert_to_markdown(uri: str)`
- Three transport modes: STDIO (default), HTTP, SSE
- HTTP/SSE modes use uvicorn + starlette

## Supported File Formats

PDF, PPTX, DOCX, XLSX/XLS, HTML, CSV, JSON, XML, Images (with EXIF/OCR), Audio (speech transcription), EPUB, ZIP archives, YouTube URLs, RSS feeds, Outlook MSG, Jupyter notebooks, RTF (via plugin)

## Key Dependencies

- **beautifulsoup4, markdownify** - HTML parsing/conversion
- **magika** - File type detection
- **pdfminer.six** - PDF extraction
- **mammoth** - DOCX parsing
- **python-pptx** - PowerPoint
- **pandas, openpyxl** - Excel
- **mcp** - MCP SDK (for server)

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key for Claude image descriptions
- `ANTHROPIC_API_KEY_FILE` - Path to file containing API key (default: `C:/mcp-servers/Claude_Key.txt`)
- `MARKITDOWN_ENABLE_PLUGINS` - Enable plugin system (true/false)
- `EXIFTOOL_PATH` - Path to exiftool binary
- `FFMPEG_PATH` - Path to ffmpeg binary

## Anthropic Claude Integration

The MCP server integrates Anthropic Claude for intelligent image descriptions. The API key is loaded from:
1. `ANTHROPIC_API_KEY` environment variable (if set)
2. Key file at `C:/mcp-servers/Claude_Key.txt` (fallback)
3. Custom key file path via `ANTHROPIC_API_KEY_FILE` environment variable

**Model Selection:**
- `claude-haiku-4-5-20251001` (default) - Fastest, for high-volume processing
- `claude-sonnet-4-20250514` - Best balance of speed and accuracy
- `claude-opus-4-20250514` - Highest capability for complex technical diagrams

```bash
# Use with specific model
python -m markitdown_mcp --model claude-opus-4-20250514
```

## Adding a New Converter

1. Create `packages/markitdown/src/markitdown/converters/_<format>_converter.py`
2. Inherit from `DocumentConverter`
3. Implement `accepts()` and `convert()` methods
4. Register in `_markitdown.py` converter list
5. Add optional dependencies to `pyproject.toml`
6. Add tests to `packages/markitdown/tests/`

## Creating a Plugin

1. Create package with entry point `markitdown.plugin`
2. Export `__plugin_interface_version__ = 1` and `register_converters(markitdown, **kwargs)`
3. Enable with `--use-plugins` flag or `enable_plugins=True`

See `packages/markitdown-sample-plugin/` for reference implementation.

## Workflow: Save Session to Memory Before Commit

Before committing changes to GitHub, save important session context to memory:

1. **Update the project memory entity** with new observations:
   ```
   mcp__memory-mcp__add_observations({
     observations: [{
       entityName: "markitdown-mcp",
       contents: ["<summary of changes made this session>"]
     }]
   })
   ```

2. **Create entities for significant new components** (converters, features, bugs fixed):
   ```
   mcp__memory-mcp__create_entities({
     entities: [{
       name: "<component-name>",
       entityType: "component|feature|bugfix",
       observations: ["<details>"]
     }]
   })
   ```

3. **Create relations** to link new entities to the project:
   ```
   mcp__memory-mcp__create_relations({
     relations: [{
       from: "<new-entity>",
       to: "markitdown-mcp",
       relationType: "belongs_to"
     }]
   })
   ```

4. **Then proceed with git commit** following standard workflow.
