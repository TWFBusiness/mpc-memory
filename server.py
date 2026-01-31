#!/usr/bin/env python3
"""
MCP Memory - Persistent memory server for AI assistants

Two layers:
- Global: ~/.mcp-memoria/data/ (personal patterns, preferences)
- Project: .mcp-memoria/ (project-specific decisions)

Smart architecture:
- Search: FTS5 (instant) + embeddings (semantic, if available)
- Indexing: background thread (non-blocking)
- RAM: ~10MB base, +80MB with embeddings (all-MiniLM-L6-v2)
"""

import os
import sys
import sqlite3
import hashlib
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional
import asyncio

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ============================================================================
# CONFIGURATION
# ============================================================================

GLOBAL_MEMORY_DIR = Path.home() / ".mcp-memoria" / "data"
GLOBAL_DB_PATH = GLOBAL_MEMORY_DIR / "global.db"

# Embedding config
EMBEDDING_ENABLED = os.environ.get("MCP_MEMORY_EMBEDDING", "true").lower() == "true"
EMBEDDING_MODEL = os.environ.get("MCP_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Background queue for indexing
_embedding_queue: queue.Queue = queue.Queue()
_embedding_thread: Optional[threading.Thread] = None
_embedding_model = None
_embedding_model_lock = threading.Lock()


# ============================================================================
# EMBEDDING MANAGER (Background)
# ============================================================================

def get_embedding_model():
    """Load embedding model (lazy, thread-safe)"""
    global _embedding_model

    if not EMBEDDING_ENABLED:
        return None

    with _embedding_model_lock:
        if _embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Memory] Loading embedding model: {EMBEDDING_MODEL}", file=sys.stderr)
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print("[Memory] Embedding model ready", file=sys.stderr)
            except ImportError:
                print("[Memory] sentence-transformers not installed, FTS only", file=sys.stderr)
                return None
            except Exception as e:
                print(f"[Memory] Embedding load error: {e}", file=sys.stderr)
                return None
        return _embedding_model


def embedding_worker():
    """Worker thread for background embedding processing"""
    print("[Memory] Background embedding worker started", file=sys.stderr)

    while True:
        try:
            item = _embedding_queue.get(timeout=5)

            if item is None:  # Poison pill for shutdown
                break

            db_path, record_id, content = item

            model = get_embedding_model()
            if not model:
                continue

            try:
                import numpy as np
                embedding = model.encode(content)
                embedding_blob = embedding.astype(np.float32).tobytes()

                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (embedding_blob, record_id)
                )
                conn.commit()
                conn.close()

            except Exception as e:
                print(f"[Memory] Embedding error for {record_id}: {e}", file=sys.stderr)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Memory] Worker error: {e}", file=sys.stderr)


def start_embedding_worker():
    """Start worker thread if embeddings enabled"""
    global _embedding_thread

    if not EMBEDDING_ENABLED:
        return

    if _embedding_thread is None or not _embedding_thread.is_alive():
        _embedding_thread = threading.Thread(target=embedding_worker, daemon=True)
        _embedding_thread.start()


def queue_embedding(db_path: Path, record_id: str, content: str):
    """Add item to queue for background embedding"""
    if EMBEDDING_ENABLED:
        _embedding_queue.put((db_path, record_id, content))


# ============================================================================
# DATABASE
# ============================================================================

def get_project_dir() -> Optional[Path]:
    """Detect current project directory"""
    cwd = os.environ.get("MCP_PROJECT_DIR") or os.environ.get("CLAUDE_CWD") or os.getcwd()
    return Path(cwd) if cwd else None


def get_project_db_path() -> Optional[Path]:
    """Return project database path"""
    project_dir = get_project_dir()
    if not project_dir:
        return None
    memory_dir = project_dir / ".mcp-memoria"
    return memory_dir / "project.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with FTS5"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Main table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB
        )
    """)

    # FTS5 for full-text search
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                tags,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (NEW.rowid, NEW.content, NEW.tags);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (NEW.rowid, NEW.content, NEW.tags);
            END
        """)
    except Exception as e:
        print(f"[Memory] FTS5 setup warning: {e}", file=sys.stderr)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")

    conn.commit()
    return conn


def generate_id(content: str, mem_type: str) -> str:
    """Generate unique ID based on content"""
    hash_input = f"{mem_type}:{content}:{datetime.now().isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# ============================================================================
# SEARCH
# ============================================================================

def search_fts(conn: sqlite3.Connection, query: str, limit: int = 10) -> list:
    """Search using FTS5 (instant)"""
    cursor = conn.cursor()

    try:
        tokens = query.split()
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in tokens if t)

        cursor.execute("""
            SELECT m.id, m.type, m.content, m.tags, m.created_at,
                   bm25(memories_fts) as relevance
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.rowid
            WHERE memories_fts MATCH ?
            ORDER BY relevance
            LIMIT ?
        """, (fts_query, limit))

        return [
            {
                "id": row[0],
                "type": row[1],
                "content": row[2],
                "tags": row[3],
                "created_at": row[4],
                "relevance": round(abs(row[5]), 3),
                "method": "fts"
            }
            for row in cursor.fetchall()
        ]
    except Exception as e:
        print(f"[Memory] FTS search error: {e}", file=sys.stderr)
        return []


def search_embedding(conn: sqlite3.Connection, query: str, limit: int = 10) -> list:
    """Search using embeddings (semantic)"""
    model = get_embedding_model()
    if not model:
        return []

    try:
        import numpy as np
        query_embedding = model.encode(query)

        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, type, content, tags, created_at, embedding
            FROM memories
            WHERE embedding IS NOT NULL
        """)

        results = []
        for row in cursor.fetchall():
            if row[5]:
                stored_embedding = np.frombuffer(row[5], dtype=np.float32)
                similarity = float(np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding) + 1e-8
                ))
                if similarity > 0.3:  # Minimum threshold
                    results.append({
                        "id": row[0],
                        "type": row[1],
                        "content": row[2],
                        "tags": row[3],
                        "created_at": row[4],
                        "relevance": round(similarity, 3),
                        "method": "embedding"
                    })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
    except Exception as e:
        print(f"[Memory] Embedding search error: {e}", file=sys.stderr)
        return []


def search_hybrid(conn: sqlite3.Connection, query: str, limit: int = 10) -> list:
    """Hybrid search: FTS + embeddings, merge results"""
    fts_results = search_fts(conn, query, limit)
    emb_results = search_embedding(conn, query, limit)

    seen_ids = set()
    merged = []

    # Prioritize embeddings (more semantic)
    for r in emb_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            merged.append(r)

    # Add FTS results not in embeddings
    for r in fts_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            merged.append(r)

    return merged[:limit]


# ============================================================================
# OPERATIONS
# ============================================================================

def save_memory(db_path: Path, mem_type: str, content: str, tags: str = "") -> dict:
    """Save memory (sync SQLite + async embedding)"""
    mem_id = generate_id(content, mem_type)

    conn = init_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO memories (id, type, content, tags, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (mem_id, mem_type, content, tags))

    conn.commit()
    conn.close()

    # Queue embedding in background
    queue_embedding(db_path, mem_id, content)

    return {"id": mem_id, "type": mem_type, "saved": True}


def list_memories(conn: sqlite3.Connection, mem_type: Optional[str] = None, limit: int = 20) -> list:
    """List recent memories"""
    cursor = conn.cursor()

    if mem_type:
        cursor.execute("""
            SELECT id, type, content, tags, created_at
            FROM memories WHERE type = ?
            ORDER BY updated_at DESC LIMIT ?
        """, (mem_type, limit))
    else:
        cursor.execute("""
            SELECT id, type, content, tags, created_at
            FROM memories ORDER BY updated_at DESC LIMIT ?
        """, (limit,))

    return [
        {"id": row[0], "type": row[1], "content": row[2], "tags": row[3], "created_at": row[4]}
        for row in cursor.fetchall()
    ]


def get_stats(db_path: Path) -> dict:
    """Get database statistics"""
    try:
        conn = init_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL")
        indexed = cursor.fetchone()[0]

        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()
        return {"total": total, "indexed": indexed, "by_type": by_type}
    except:
        return {"total": 0, "indexed": 0, "by_type": {}}


# ============================================================================
# MCP SERVER
# ============================================================================

server = Server("memory")


@server.list_tools()
async def list_tools():
    """List available tools"""
    return [
        Tool(
            name="memory_context",
            description="USE AUTOMATICALLY at the start of each conversation. Returns relevant memories for the current context (project + global). Works as an automatic 'recall'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Current context or user question"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_search",
            description="Search specific memories when you need detailed information about past decisions, patterns, or preferences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project", "both"],
                        "default": "both"
                    },
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_save",
            description="Save important decision, pattern, or preference. Use after: (1) making architecture decisions, (2) defining code patterns, (3) learning user preferences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to save"},
                    "type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "preference", "architecture", "todo", "note"]
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project"],
                        "default": "project"
                    },
                    "tags": {"type": "string", "description": "Comma-separated tags"}
                },
                "required": ["content", "type"]
            }
        ),
        Tool(
            name="memory_list",
            description="List recent memories. Useful to review decision history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "preference", "architecture", "todo", "note"]
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project", "both"],
                        "default": "both"
                    },
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="memory_stats",
            description="Show memory statistics (total, indexed, by type).",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="memory_delete",
            description="Remove a memory by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "scope": {"type": "string", "enum": ["global", "project"], "default": "project"}
                },
                "required": ["id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute tool"""

    # ========== CONTEXT (auto-recall) ==========
    if name == "memory_context":
        query = arguments.get("query", "")
        results = []

        for scope, db_path in [("global", GLOBAL_DB_PATH), ("project", get_project_db_path())]:
            if db_path and (db_path.exists() or scope == "global"):
                try:
                    conn = init_db(db_path)
                    for r in search_hybrid(conn, query, limit=5):
                        r["scope"] = scope
                        results.append(r)
                    conn.close()
                except Exception as e:
                    print(f"[Memory] Context search error ({scope}): {e}", file=sys.stderr)

        if not results:
            return [TextContent(type="text", text="[Memory] No relevant context found.")]

        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        results = results[:8]

        output = "## Memory Context\n\n"
        for r in results:
            output += f"**[{r['scope']}:{r['type']}]** {r['content']}\n"
        output += "\n---\n_Use this context to inform your responses._"

        return [TextContent(type="text", text=output)]

    # ========== SEARCH ==========
    elif name == "memory_search":
        query = arguments.get("query", "")
        scope = arguments.get("scope", "both")
        limit = arguments.get("limit", 5)

        results = []

        if scope in ["global", "both"]:
            try:
                conn = init_db(GLOBAL_DB_PATH)
                for r in search_hybrid(conn, query, limit):
                    r["scope"] = "global"
                    results.append(r)
                conn.close()
            except Exception as e:
                print(f"[Memory] Global search error: {e}", file=sys.stderr)

        if scope in ["project", "both"]:
            project_db = get_project_db_path()
            if project_db:
                try:
                    conn = init_db(project_db)
                    for r in search_hybrid(conn, query, limit):
                        r["scope"] = "project"
                        results.append(r)
                    conn.close()
                except Exception as e:
                    print(f"[Memory] Project search error: {e}", file=sys.stderr)

        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        results = results[:limit]

        if not results:
            return [TextContent(type="text", text="No memories found.")]

        output = f"## Memories ({len(results)})\n\n"
        for r in results:
            output += f"**[{r['scope'].upper()}] {r['type']}** (relevance: {r['relevance']}, method: {r.get('method', '?')})\n"
            output += f"{r['content']}\n"
            if r.get('tags'):
                output += f"_Tags: {r['tags']}_\n"
            output += "\n"

        return [TextContent(type="text", text=output)]

    # ========== SAVE ==========
    elif name == "memory_save":
        content = arguments.get("content", "")
        mem_type = arguments.get("type", "note")
        scope = arguments.get("scope", "project")
        tags = arguments.get("tags", "")

        if not content:
            return [TextContent(type="text", text="Error: empty content.")]

        db_path = GLOBAL_DB_PATH if scope == "global" else get_project_db_path()
        if not db_path:
            return [TextContent(type="text", text="Error: project not detected.")]

        try:
            result = save_memory(db_path, mem_type, content, tags)
            return [TextContent(
                type="text",
                text=f"Memory saved ({scope})\n- Type: {mem_type}\n- ID: {result['id']}\n- Embedding: {'queued' if EMBEDDING_ENABLED else 'disabled'}"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ========== LIST ==========
    elif name == "memory_list":
        mem_type = arguments.get("type")
        scope = arguments.get("scope", "both")
        limit = arguments.get("limit", 10)

        results = []

        if scope in ["global", "both"]:
            try:
                conn = init_db(GLOBAL_DB_PATH)
                for r in list_memories(conn, mem_type, limit):
                    r["scope"] = "global"
                    results.append(r)
                conn.close()
            except Exception as e:
                print(f"[Memory] Global list error: {e}", file=sys.stderr)

        if scope in ["project", "both"]:
            project_db = get_project_db_path()
            if project_db:
                try:
                    conn = init_db(project_db)
                    for r in list_memories(conn, mem_type, limit):
                        r["scope"] = "project"
                        results.append(r)
                    conn.close()
                except Exception as e:
                    print(f"[Memory] Project list error: {e}", file=sys.stderr)

        if not results:
            return [TextContent(type="text", text="No memories found.")]

        output = f"## Memories ({len(results)})\n\n"
        for r in results:
            output += f"- **[{r['scope']}] {r['type']}**: {r['content'][:80]}{'...' if len(r['content']) > 80 else ''}\n"
            output += f"  `{r['id']}` | {r['created_at']}\n\n"

        return [TextContent(type="text", text=output)]

    # ========== STATS ==========
    elif name == "memory_stats":
        output = "## Memory Statistics\n\n"

        global_stats = get_stats(GLOBAL_DB_PATH)
        output += f"**Global** ({GLOBAL_DB_PATH}):\n"
        output += f"- Total: {global_stats['total']}\n"
        output += f"- Indexed (embedding): {global_stats['indexed']}\n"
        output += f"- By type: {global_stats['by_type']}\n\n"

        project_db = get_project_db_path()
        if project_db and project_db.exists():
            proj_stats = get_stats(project_db)
            output += f"**Project** ({project_db}):\n"
            output += f"- Total: {proj_stats['total']}\n"
            output += f"- Indexed (embedding): {proj_stats['indexed']}\n"
            output += f"- By type: {proj_stats['by_type']}\n\n"

        output += "**Config**:\n"
        output += f"- Embeddings: {'enabled' if EMBEDDING_ENABLED else 'disabled'}\n"
        output += f"- Model: {EMBEDDING_MODEL}\n"
        output += f"- Queue pending: {_embedding_queue.qsize()}\n"

        return [TextContent(type="text", text=output)]

    # ========== DELETE ==========
    elif name == "memory_delete":
        mem_id = arguments.get("id", "")
        scope = arguments.get("scope", "project")

        if not mem_id:
            return [TextContent(type="text", text="Error: ID required.")]

        db_path = GLOBAL_DB_PATH if scope == "global" else get_project_db_path()
        if not db_path:
            return [TextContent(type="text", text="Error: project not detected.")]

        try:
            conn = init_db(db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted:
                return [TextContent(type="text", text=f"Memory {mem_id} deleted.")]
            return [TextContent(type="text", text=f"Memory {mem_id} not found.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Start MCP server"""
    print("[Memory] ============================================", file=sys.stderr)
    print("[Memory] MCP Memory Server Starting", file=sys.stderr)
    print(f"[Memory] Global DB: {GLOBAL_DB_PATH}", file=sys.stderr)
    print(f"[Memory] Embeddings: {'enabled' if EMBEDDING_ENABLED else 'disabled'}", file=sys.stderr)
    print(f"[Memory] Model: {EMBEDDING_MODEL}", file=sys.stderr)
    print("[Memory] ============================================", file=sys.stderr)

    # Initialize global database
    init_db(GLOBAL_DB_PATH)

    # Start embedding worker in background
    start_embedding_worker()

    # Pre-load embedding model (in background)
    if EMBEDDING_ENABLED:
        threading.Thread(target=get_embedding_model, daemon=True).start()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
