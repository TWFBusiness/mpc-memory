#!/usr/bin/env python3
"""
Hook para Claude Code: salva conversa automaticamente no MCP Memory (personality)

Melhorias v2:
- Uma memória por sessão (UPSERT, não INSERT duplicado)
- Formato estruturado: extrai ferramentas, arquivos, ações
- Limite de 20 turnos por sessão
- ID determinístico baseado no session_id
- Extração de file paths mencionados

Captura eventos:
- UserPromptSubmit: acumula pergunta do usuário
- Stop: atualiza memória da sessão com resumo
"""

import sys
import json
import re
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

# Database path
PERSONALITY_DB = Path.home() / ".mcp-memoria" / "data" / "personality.db"

# Session file para acumular conversa
SESSION_FILE = (
    Path.home() / ".claude" / "mcp-memoria" / "hooks" / ".current_session.json"
)

# Limite de turnos por sessão
MAX_TURNS = 20


def init_db():
    """Inicializa database se necessário"""
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

    # FTS5
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, content='memories', content_rowid='rowid'
            )
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
            BEGIN
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (NEW.rowid, NEW.content, NEW.tags);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (NEW.rowid, NEW.content, NEW.tags);
            END
        """)
    except Exception:
        pass

    conn.commit()
    return conn


def session_memory_id(session_id: str) -> str:
    """Gera ID determinístico a partir do session_id (uma memória por sessão)"""
    return hashlib.sha256(f"session:{session_id}".encode()).hexdigest()[:16]


def load_session():
    """Carrega dados da sessão atual"""
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "turns": [], "session_id": "", "project": "",
        "tools": [], "files": []
    }


def save_session(data):
    """Salva dados da sessão"""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f)


def extract_files(text: str) -> list:
    """Extrai paths de arquivo do texto"""
    patterns = re.findall(r'(?:/[\w._-]+)+\.\w+', text)
    return list(set(patterns))[:20]


def build_session_content(session: dict) -> str:
    """Monta conteúdo estruturado a partir dos dados da sessão"""
    project = session.get("project", "unknown")
    turns = session.get("turns", [])
    all_tools = sorted(set(session.get("tools", [])))
    all_files = sorted(set(session.get("files", [])))

    # Extrai prompts do usuário (deduplicados)
    user_prompts = []
    seen = set()
    for turn in turns:
        if turn.get("role") == "user":
            content = turn.get("content", "")
            key = content[:50].lower()
            if key not in seen and len(content) > 5:
                seen.add(key)
                user_prompts.append(content[:300])

    # Monta conteúdo
    lines = [f"[{project}] Session ({len(turns)} turns)"]

    if all_tools:
        lines.append(f"Tools: {', '.join(all_tools[:20])}")

    if all_files:
        lines.append(f"Files: {', '.join(all_files[:15])}")

    if user_prompts:
        lines.append("Topics:")
        for p in user_prompts[:10]:
            lines.append(f"  - {p}")

    return "\n".join(lines)


def save_to_db(session: dict):
    """Salva/atualiza memória da sessão no database (UPSERT)"""
    session_id = session.get("session_id", "")
    if not session_id:
        return None

    project = session.get("project", "unknown")
    mem_id = session_memory_id(session_id)
    content = build_session_content(session)
    tags = f"conversation,claude-code,{project},auto-saved"

    conn = init_db()
    cursor = conn.cursor()

    # Verifica se memória desta sessão já existe
    cursor.execute("SELECT id FROM memories WHERE id = ?", (mem_id,))
    exists = cursor.fetchone()

    if exists:
        # Atualiza (UPSERT) - limpa embedding pra ser re-indexado
        cursor.execute(
            "UPDATE memories SET content = ?, tags = ?, "
            "updated_at = CURRENT_TIMESTAMP, embedding = NULL "
            "WHERE id = ?",
            (content, tags, mem_id)
        )
    else:
        cursor.execute(
            "INSERT INTO memories (id, type, content, tags) "
            "VALUES (?, 'conversation', ?, ?)",
            (mem_id, content, tags)
        )

    conn.commit()
    conn.close()
    return mem_id


def handle_user_prompt(hook_data):
    """Trata evento UserPromptSubmit - captura pergunta do usuário"""
    session_id = hook_data.get("session_id", "unknown")
    cwd = hook_data.get("cwd", "")
    project_name = Path(cwd).name if cwd else "no-project"
    prompt = hook_data.get("prompt", "")

    if not prompt:
        return

    session = load_session()

    # Reset se nova sessão
    if session.get("session_id") != session_id:
        session = {
            "turns": [], "session_id": session_id,
            "project": project_name, "tools": [], "files": []
        }

    # Extrai file paths do prompt
    files = extract_files(prompt)
    session.setdefault("files", [])
    for f in files:
        if f not in session["files"]:
            session["files"].append(f)

    # Adiciona turno do usuário
    session["turns"].append({
        "role": "user",
        "content": prompt[:500],
        "timestamp": datetime.now().isoformat()
    })

    # Mantém apenas últimos MAX_TURNS
    if len(session["turns"]) > MAX_TURNS:
        session["turns"] = session["turns"][-MAX_TURNS:]

    save_session(session)
    print(
        f"[Memory Hook] Captured user prompt ({len(prompt)} chars)",
        file=sys.stderr
    )


def handle_stop(hook_data):
    """Trata evento Stop - atualiza memória da sessão"""
    session_id = hook_data.get("session_id", "unknown")
    cwd = hook_data.get("cwd", "")
    project_name = Path(cwd).name if cwd else "no-project"

    # Tools usadas
    tools_used = hook_data.get("stop_hook_active_tools", [])
    tool_names = [
        t.get("name", "unknown") for t in tools_used
    ] if tools_used else []

    session = load_session()

    # Atualiza lista de tools
    session.setdefault("tools", [])
    for t in tool_names:
        if t not in session["tools"]:
            session["tools"].append(t)

    # Adiciona turno do assistente
    assistant_info = []
    if tool_names:
        assistant_info.append(f"Tools: {', '.join(tool_names)}")

    session["turns"].append({
        "role": "assistant",
        "content": " | ".join(assistant_info) if assistant_info else "Responded",
        "timestamp": datetime.now().isoformat()
    })

    # Mantém apenas últimos MAX_TURNS
    if len(session["turns"]) > MAX_TURNS:
        session["turns"] = session["turns"][-MAX_TURNS:]

    # Salva/atualiza no database (uma memória por sessão)
    mem_id = save_to_db(session)
    save_session(session)

    print(
        f"[Memory Hook] Updated session memory {mem_id} "
        f"({len(session['turns'])} turns)",
        file=sys.stderr
    )


def main():
    try:
        input_data = sys.stdin.read()

        if not input_data.strip():
            sys.exit(0)

        hook_data = json.loads(input_data)
        event_name = hook_data.get("hook_event_name", "")

        if event_name == "UserPromptSubmit":
            handle_user_prompt(hook_data)
        elif event_name == "Stop":
            handle_stop(hook_data)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as e:
        print(f"[Memory Hook] Error: {e}", file=sys.stderr)
        sys.exit(0)  # Não bloqueia Claude em caso de erro


if __name__ == "__main__":
    main()
