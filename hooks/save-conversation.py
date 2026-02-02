#!/usr/bin/env python3
"""
Hook para Claude Code: salva conversa automaticamente no MCP Memory (personality)

Captura eventos:
- UserPromptSubmit: salva a pergunta do usuário
- Stop: salva resumo do que foi feito (tools usadas)

Os dados são salvos no personality database para persistir entre sessões.
"""

import sys
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import os

# Database path
PERSONALITY_DB = Path.home() / ".mcp-memoria" / "data" / "personality.db"

# Session file to accumulate conversation
SESSION_FILE = Path.home() / ".claude" / "mcp-memoria" / "hooks" / ".current_session.json"


def init_db():
    """Initialize database if needed"""
    PERSONALITY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PERSONALITY_DB))
    cursor = conn.cursor()

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

    # FTS5 for search
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, content='memories', content_rowid='rowid'
            )
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, tags) VALUES (NEW.rowid, NEW.content, NEW.tags);
            END
        """)
    except:
        pass

    conn.commit()
    return conn


def generate_id(content: str) -> str:
    """Generate unique ID"""
    hash_input = f"conversation:{content}:{datetime.now().isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def load_session():
    """Load current session data"""
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"turns": [], "session_id": "", "project": ""}


def save_session(data):
    """Save session data"""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f)


def save_to_memory(content: str, tags: str = ""):
    """Save to personality database"""
    conn = init_db()
    cursor = conn.cursor()

    mem_id = generate_id(content)

    cursor.execute("""
        INSERT OR REPLACE INTO memories (id, type, content, tags, updated_at)
        VALUES (?, 'conversation', ?, ?, CURRENT_TIMESTAMP)
    """, (mem_id, content, tags))

    conn.commit()
    conn.close()

    return mem_id


def handle_user_prompt(hook_data):
    """Handle UserPromptSubmit event - save user's question"""
    session_id = hook_data.get("session_id", "unknown")
    cwd = hook_data.get("cwd", "")
    project_name = Path(cwd).name if cwd else "no-project"

    # Get user prompt from hook data
    prompt = hook_data.get("prompt", "")

    if not prompt:
        return

    # Load or create session
    session = load_session()

    # Reset if new session
    if session.get("session_id") != session_id:
        session = {"turns": [], "session_id": session_id, "project": project_name}

    # Add user turn
    session["turns"].append({
        "role": "user",
        "content": prompt[:1000],  # Truncate if too long
        "timestamp": datetime.now().isoformat()
    })

    save_session(session)
    print(f"[Memory Hook] Captured user prompt ({len(prompt)} chars)", file=sys.stderr)


def handle_stop(hook_data):
    """Handle Stop event - save conversation summary"""
    session_id = hook_data.get("session_id", "unknown")
    cwd = hook_data.get("cwd", "")
    project_name = Path(cwd).name if cwd else "no-project"

    # Get tools used
    tools_used = hook_data.get("stop_hook_active_tools", [])
    tool_names = [t.get("name", "unknown") for t in tools_used] if tools_used else []

    # Load session
    session = load_session()

    # Add assistant turn
    assistant_summary = []
    if tool_names:
        assistant_summary.append(f"Tools: {', '.join(tool_names)}")

    session["turns"].append({
        "role": "assistant",
        "content": " | ".join(assistant_summary) if assistant_summary else "Responded",
        "timestamp": datetime.now().isoformat()
    })

    # Build conversation content
    turns_text = []
    for turn in session.get("turns", [])[-10:]:  # Last 10 turns
        role = turn.get("role", "?")
        content = turn.get("content", "")
        turns_text.append(f"[{role}] {content}")

    if not turns_text:
        return

    content = f"[{project_name}] Session {session_id[:8]}\n" + "\n".join(turns_text)
    tags = f"conversation,claude-code,{project_name},auto-saved"

    mem_id = save_to_memory(content, tags)

    # Save updated session
    save_session(session)

    print(f"[Memory Hook] Saved conversation {mem_id} ({len(session['turns'])} turns)", file=sys.stderr)


def main():
    try:
        # Read hook data from stdin
        input_data = sys.stdin.read()

        if not input_data.strip():
            sys.exit(0)

        hook_data = json.loads(input_data)
        event_name = hook_data.get("hook_event_name", "")

        if event_name == "UserPromptSubmit":
            handle_user_prompt(hook_data)
        elif event_name == "Stop":
            handle_stop(hook_data)
        else:
            print(f"[Memory Hook] Unknown event: {event_name}", file=sys.stderr)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as e:
        print(f"[Memory Hook] Error: {e}", file=sys.stderr)
        sys.exit(0)  # Don't block Claude on errors


if __name__ == "__main__":
    main()
