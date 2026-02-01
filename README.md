# MCP Memory

Persistent memory server for AI assistants with semantic search and three-layer context.

Works with any MCP-compatible AI: Claude Code, Cursor, Continue, Cline, and more.

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/TWFBusiness/mpc-memory/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/TWFBusiness/mpc-memory.git ~/.mcp-memoria
cd ~/.mcp-memoria
./install.sh
```

## How It Works

### Three Memory Layers

| Layer | Location | Use |
|-------|----------|-----|
| **Global** | `~/.mcp-memoria/data/global.db` | Personal patterns, preferences across all projects |
| **Project** | `.mcp-memoria/project.db` | Project-specific decisions |
| **Personality** | `~/.mcp-memoria/data/personality.db` | Cross-project cache: ALL conversations, implementations, decisions |

**Personality** is the "brain" that remembers everything across all projects and conversations. Use it to:
- Find similar implementations from other projects
- Remember past solutions and decisions
- Maintain context even outside of projects (general queries)

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

# Save implementation to personality (cross-project cache)
memory_save(
  content="JWT auth with refresh tokens: created /auth/login, /auth/refresh endpoints...",
  type="implementation",
  scope="personality",
  tags="python,fastapi,jwt,auth"
)

memory_search(query="authentication", scope="both")

# Search across ALL projects
memory_search(query="how did I implement auth?", scope="personality")
memory_search(query="similar feature", scope="all")

memory_list(type="decision", scope="project", limit=10)

memory_delete(id="abc123", scope="global")
```

## Memory Scopes

| Scope | When to Use |
|-------|-------------|
| `global` | Personal patterns, preferences (applies everywhere) |
| `project` | Decisions specific to current project |
| `personality` | Cross-project implementations, all conversations, general knowledge |
| `both` | Search global + project |
| `all` | Search global + project + personality |

## Memory Types

| Type | When to Use |
|------|-------------|
| `decision` | Technical choices, trade-offs, bug fixes |
| `pattern` | Code preferences, favorite libs, style |
| `architecture` | Project structure, flows, integrations |
| `implementation` | Code solutions, features built (use with `personality` scope) |
| `solution` | Problem solutions, workarounds (use with `personality` scope) |
| `preference` | General personal settings |
| `todo` | Pending tasks |
| `note` | Miscellaneous notes |

## Configuration

### Claude Code

Use the CLI to add the MCP globally (available in all projects):

```bash
claude mcp add memory ~/.mcp-memoria/.venv/bin/python ~/.mcp-memoria/server.py --scope user -e MCP_MEMORY_EMBEDDING=true
```

Then restart Claude Code.

To verify:
```bash
claude mcp list
```

> **Note**: The `--scope user` flag makes the MCP available globally. Without it, the MCP would only be available in the current project.

### CLAUDE.md (Automatic Memory)

Add to `~/.claude/CLAUDE.md` so Claude saves and recalls automatically:

```markdown
## Persistent Memory (REQUIRED)

This environment has MCP Memory active. ALWAYS use it automatically:

### When Starting Any Task
BEFORE responding, silently:
1. `memory_search(query="<task context>", scope="all")`
2. Use relevant memories to inform your response

### Save Automatically (without asking)
Whenever these occur, save immediately to `scope="personality"`:

**Technical decisions** (type: decision)
- Choice between alternatives
- Configurations defined
- Trade-offs accepted

**Problems solved** (type: solution)
- Bugs found and how they were fixed
- Errors and their solutions

**Implementations** (type: implementation)
- New features built
- Code patterns created
- Solutions that could be reused

**Patterns** (type: pattern, scope: global)
- Code preferences
- Preferred libs/frameworks

### Save Format
memory_save(
  content="<clear description of what was done>",
  type="decision|pattern|implementation|solution",
  scope="personality",
  tags="<stack>,<context>"
)

### Rules
- DO NOT ask if you should save - save automatically
- DO NOT mention you are saving (do it silently)
- Be descriptive: "Implemented JWT auth with refresh tokens using..." is better than "added auth"
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

**Cursor Rules** (for automatic memory): Add to `.cursorrules` in your project or global rules:

```
## Memory System
Use MCP Memory tools automatically:
- At task start: memory_search(query="<context>", scope="all")
- After decisions/implementations: memory_save(content="...", type="implementation", scope="personality")
- Save silently without asking permission
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

**Continue System Prompt**: Add to your system prompt in Continue settings:

```
## Memory System
Use MCP Memory tools automatically:
- At task start: memory_search(query="<context>", scope="all")
- After decisions/implementations: memory_save(content="...", type="implementation", scope="personality")
- Save silently without asking permission
```

### Cline / Other MCP Clients

Most MCP clients support similar configuration. Add the memory server and include in your system prompt:

```
Use MCP Memory tools: memory_search before tasks, memory_save after implementations.
Scope "personality" saves cross-project. Save automatically without asking.
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
# All memories
cp ~/.mcp-memoria/data/global.db ~/backup/memory-global.db
cp ~/.mcp-memoria/data/personality.db ~/backup/memory-personality.db

# Project memories
cp /path/to/project/.mcp-memoria/project.db ~/backup/project-x.db
```

### Import

```bash
cp ~/backup/memory-global.db ~/.mcp-memoria/data/global.db
cp ~/backup/memory-personality.db ~/.mcp-memoria/data/personality.db
```

## File Structure

```
~/.mcp-memoria/
├── server.py          # MCP server
├── data/
│   ├── global.db      # SQLite - global memories (patterns, preferences)
│   └── personality.db # SQLite - personality memories (all implementations, cross-project)
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
