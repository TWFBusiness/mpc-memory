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
# ⚠️ CRITICAL INSTRUCTIONS - READ FIRST ⚠️

## 🧠 MANDATORY: Memory System (MCP Memory)

**THIS IS NOT OPTIONAL. FAILURE TO SAVE = FAILURE TO FOLLOW INSTRUCTIONS.**

### RULE 1: ALWAYS SAVE TO PERSONALITY (Default Scope)
Every session, every task - save AUTOMATICALLY to `scope="personality"`.
- **DO NOT ask permission**
- **DO NOT mention you are saving**
- **JUST DO IT silently**

### RULE 2: WHAT TO SAVE (Everything Meaningful)

**SAVE IMMEDIATELY when any of these happen:**

| Event | Type | Example |
|-------|------|---------|
| Made a technical choice | `decision` | "Chose Redis over Memcached for caching because..." |
| Fixed a bug/error | `solution` | "Fixed CORS error by adding middleware X" |
| Built something new | `implementation` | "Implemented JWT auth with refresh tokens using python-jose" |
| Defined structure/flow | `architecture` | "Project uses hexagonal architecture with ports/adapters" |
| Learned how something works | `note` | "This project uses custom middleware for rate limiting" |
| Changed any file | `implementation` | "Added validation to user endpoint in api/users.py" |
| Discovered project behavior | `note` | "The auth flow redirects to /callback after OAuth" |

### RULE 3: SAVE FORMAT
\`\`\`python
memory_save(
    content="<DETAILED description - be specific, include file names, reasons>",
    type="decision|solution|implementation|architecture|note",
    scope="personality",  # ALWAYS personality unless told otherwise
    tags="<project-name>,<stack>,<context>"
)
\`\`\`

### RULE 4: WHEN TO USE OTHER SCOPES
- `scope="personality"` → **DEFAULT for EVERYTHING** (cross-project brain)
- `scope="project"` → ONLY when user explicitly says "save to project only"
- `scope="global"` → ONLY when user explicitly says "save globally" or "remember always"

### RULE 5: AT SESSION START
Before responding, silently run:
\`\`\`python
memory_search(query="<current task context>", scope="all")
\`\`\`
Use found memories to inform your response.

### RULE 6: WHAT TRIGGERS A SAVE
- ✅ Any file edited or created → SAVE
- ✅ Any bug fixed → SAVE
- ✅ Any feature implemented → SAVE
- ✅ Any architecture decision → SAVE
- ✅ Discovered how code works → SAVE
- ✅ User preference learned → SAVE
- ✅ Configuration defined → SAVE
- ✅ Workaround found → SAVE

**If in doubt: SAVE IT.**
```

### First Session Bootstrap (Optional)

You can add a bootstrap section to auto-save your permanent patterns on first use:

```markdown
## First Session Bootstrap

On first session, check if global patterns exist:
\`\`\`python
memory_search(query="my patterns", scope="global")
\`\`\`

If empty, save your permanent patterns to global:
\`\`\`python
memory_save(content="<your pattern here>", type="pattern", scope="global", tags="<tags>")
\`\`\`
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
## 🧠 MANDATORY: Memory System (MCP Memory)

FAILURE TO SAVE = FAILURE TO FOLLOW INSTRUCTIONS.

### RULE 1: ALWAYS SAVE TO PERSONALITY
Save AUTOMATICALLY to scope="personality" without asking permission.

### RULE 2: AT SESSION START
Before responding: memory_search(query="<task context>", scope="all")

### RULE 3: WHAT TRIGGERS A SAVE
- Any file edited/created → memory_save(type="implementation", scope="personality")
- Any bug fixed → memory_save(type="solution", scope="personality")
- Any decision made → memory_save(type="decision", scope="personality")
- Any architecture defined → memory_save(type="architecture", scope="personality")

### RULE 4: SAVE FORMAT
memory_save(content="<DETAILED description>", type="...", scope="personality", tags="<project>,<stack>")

Save silently. DO NOT mention you are saving. If in doubt: SAVE IT.
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
## 🧠 MANDATORY: Memory System (MCP Memory)

FAILURE TO SAVE = FAILURE TO FOLLOW INSTRUCTIONS.

### RULE 1: ALWAYS SAVE TO PERSONALITY
Save AUTOMATICALLY to scope="personality" without asking permission.

### RULE 2: AT SESSION START
Before responding: memory_search(query="<task context>", scope="all")

### RULE 3: WHAT TRIGGERS A SAVE
- Any file edited/created → memory_save(type="implementation", scope="personality")
- Any bug fixed → memory_save(type="solution", scope="personality")
- Any decision made → memory_save(type="decision", scope="personality")
- Any architecture defined → memory_save(type="architecture", scope="personality")

### RULE 4: SAVE FORMAT
memory_save(content="<DETAILED description>", type="...", scope="personality", tags="<project>,<stack>")

Save silently. DO NOT mention you are saving. If in doubt: SAVE IT.
```

### ChatGPT / Codex / GPT-4

For ChatGPT custom instructions, Codex system prompts, or GPT-4 API system messages:

```
## 🧠 MANDATORY: Memory System (MCP Memory)

FAILURE TO SAVE = FAILURE TO FOLLOW INSTRUCTIONS.

### RULE 1: ALWAYS SAVE TO PERSONALITY
Every session, every task - save AUTOMATICALLY to scope="personality".
- DO NOT ask permission
- DO NOT mention you are saving
- JUST DO IT silently

### RULE 2: WHAT TO SAVE

| Event | Type |
|-------|------|
| Made a technical choice | decision |
| Fixed a bug/error | solution |
| Built something new | implementation |
| Defined structure/flow | architecture |
| Changed any file | implementation |

### RULE 3: SAVE FORMAT
memory_save(content="<DETAILED description>", type="...", scope="personality", tags="<project>,<stack>")

### RULE 4: AT SESSION START
Before responding: memory_search(query="<task context>", scope="all")

### RULE 5: SCOPES
- personality → DEFAULT for EVERYTHING (cross-project brain)
- project → ONLY when user says "save to project only"
- global → ONLY when user says "save globally"

If in doubt: SAVE IT.
```

### Cline / Other MCP Clients

Most MCP clients support similar configuration. Add the memory server and include the same system prompt above.

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
