# MCP Memory

Persistent memory server for AI assistants with semantic search and two-layer context.

Works with any MCP-compatible AI: Claude Code, Cursor, Continue, Cline, and more.

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/TWFBusiness/mpc-memoria/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/TWFBusiness/mpc-memoria.git ~/.mcp-memoria
cd ~/.mcp-memoria
./install.sh
```

## How It Works

### Two Memory Layers

| Layer | Location | Use |
|-------|----------|-----|
| **Global** | `~/.mcp-memoria/data/` | Personal patterns, preferences across all projects |
| **Project** | `.mcp-memoria/` | Project-specific decisions |

### Smart Search

- **FTS5**: Instant text search (always active)
- **Embeddings**: Semantic search in background (optional, +150MB RAM)

With embeddings, searches like "how did I configure auth" find memories about "JWT with refresh token" even without matching words.

### Background Indexing

Embeddings are processed asynchronously:
1. You save a memory → instant response (SQLite)
2. Background worker generates embedding
3. Next searches include new content

No blocking or slowdown when saving.

## Usage

### Save Memories

```
"save that I prefer pytest over unittest"
"remember this project uses PostgreSQL with Tortoise ORM"
"save globally: always use Black for formatting"
```

### Search Memories

```
"what did we decide about tests?"
"how did we configure the database?"
"what are my code patterns?"
```

### Direct Commands (optional)

```
memory_save(
  content="FastAPI always 100% async, never sync",
  type="pattern",
  scope="global",
  tags="python,fastapi,async"
)

memory_search(query="authentication", scope="both")

memory_list(type="decision", scope="project", limit=10)

memory_delete(id="abc123", scope="global")
```

## Memory Types

| Type | When to Use |
|------|-------------|
| `decision` | Technical choices, trade-offs, bug fixes |
| `pattern` | Code preferences, favorite libs, style |
| `architecture` | Project structure, flows, integrations |
| `preference` | General personal settings |
| `todo` | Pending tasks |
| `note` | Miscellaneous notes |

## Configuration

### Claude Code

File: `~/.claude/.mcp.json`

```json
{
  "mcpServers": {
    "memory": {
      "command": "~/.mcp-memoria/.venv/bin/python",
      "args": ["~/.mcp-memoria/server.py"],
      "env": {
        "MCP_MEMORY_EMBEDDING": "true"
      }
    }
  }
}
```

### Cursor

File: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "memory": {
      "command": "~/.mcp-memoria/.venv/bin/python",
      "args": ["~/.mcp-memoria/server.py"],
      "env": {
        "MCP_MEMORY_EMBEDDING": "true"
      }
    }
  }
}
```

### Continue

File: `~/.continue/config.json`

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "~/.mcp-memoria/.venv/bin/python",
          "args": ["~/.mcp-memoria/server.py"]
        }
      }
    ]
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_EMBEDDING` | `true` | Enable semantic search |
| `MCP_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `MCP_PROJECT_DIR` | (auto) | Override project directory |

### Embedding Models

| Model | RAM | Quality | Languages |
|-------|-----|---------|-----------|
| `all-MiniLM-L6-v2` | ~80MB | Good | EN (ok for code) |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~150MB | Good | Multi (better for non-EN) |
| `all-mpnet-base-v2` | ~400MB | Excellent | EN |

## Backup and Restore

### Export

```bash
# Memories
cp ~/.mcp-memoria/data/global.db ~/backup/memory-global.db

# Project memories
cp /path/to/project/.mcp-memoria/project.db ~/backup/project-x.db
```

### Import

```bash
cp ~/backup/memory-global.db ~/.mcp-memoria/data/global.db
```

## File Structure

```
~/.mcp-memoria/
├── server.py          # MCP server
├── data/
│   └── global.db      # SQLite - global memories
└── .venv/             # Python virtual environment

~/your-project/
└── .mcp-memoria/
    └── project.db     # SQLite - project memories
```

## Requirements

- Python 3.10+
- MCP-compatible AI assistant
- ~10MB RAM (FTS only) or ~150MB RAM (with embeddings)

## Uninstall

```bash
rm -rf ~/.mcp-memoria
```

## License

MIT
