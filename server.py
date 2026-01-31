#!/usr/bin/env python3
"""
MCP Memoria - Servidor de memória persistente para Claude Code

Duas camadas:
- Global: ~/.claude/memoria/ (padrões pessoais, preferências)
- Projeto: .claude/memoria/ (decisões específicas do projeto)

Arquitetura inteligente:
- Busca: FTS5 (instantâneo) + embeddings (se disponíveis)
- Indexação: background thread (não bloqueia)
- RAM: ~10MB base, +80MB com embeddings (all-MiniLM-L6-v2)
"""

import os
import sys
import json
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
# CONFIGURAÇÃO
# ============================================================================

GLOBAL_MEMORIA_DIR = Path.home() / ".claude" / "memoria"
GLOBAL_DB_PATH = GLOBAL_MEMORIA_DIR / "global.db"

# Embedding config
EMBEDDING_ENABLED = os.environ.get("MEMORIA_EMBEDDING", "true").lower() == "true"
EMBEDDING_MODEL = os.environ.get("MEMORIA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Background queue para indexação
_embedding_queue: queue.Queue = queue.Queue()
_embedding_thread: Optional[threading.Thread] = None
_embedding_model = None
_embedding_model_lock = threading.Lock()


# ============================================================================
# EMBEDDING MANAGER (Background)
# ============================================================================

def get_embedding_model():
    """Carrega modelo de embedding (lazy, thread-safe)"""
    global _embedding_model

    if not EMBEDDING_ENABLED:
        return None

    with _embedding_model_lock:
        if _embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Memoria] Loading embedding model: {EMBEDDING_MODEL}", file=sys.stderr)
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print(f"[Memoria] Embedding model ready", file=sys.stderr)
            except ImportError:
                print("[Memoria] sentence-transformers not installed, FTS only", file=sys.stderr)
                return None
            except Exception as e:
                print(f"[Memoria] Embedding load error: {e}", file=sys.stderr)
                return None
        return _embedding_model


def embedding_worker():
    """Worker thread para processar embeddings em background"""
    print("[Memoria] Background embedding worker started", file=sys.stderr)

    while True:
        try:
            # Espera item na fila (blocking)
            item = _embedding_queue.get(timeout=5)

            if item is None:  # Poison pill para shutdown
                break

            db_path, record_id, conteudo = item

            model = get_embedding_model()
            if not model:
                continue

            try:
                import numpy as np
                embedding = model.encode(conteudo)
                embedding_blob = embedding.astype(np.float32).tobytes()

                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memorias SET embedding = ? WHERE id = ?",
                    (embedding_blob, record_id)
                )
                conn.commit()
                conn.close()

            except Exception as e:
                print(f"[Memoria] Embedding error for {record_id}: {e}", file=sys.stderr)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Memoria] Worker error: {e}", file=sys.stderr)


def start_embedding_worker():
    """Inicia worker thread se embeddings habilitados"""
    global _embedding_thread

    if not EMBEDDING_ENABLED:
        return

    if _embedding_thread is None or not _embedding_thread.is_alive():
        _embedding_thread = threading.Thread(target=embedding_worker, daemon=True)
        _embedding_thread.start()


def queue_embedding(db_path: Path, record_id: str, conteudo: str):
    """Adiciona item na fila para embedding em background"""
    if EMBEDDING_ENABLED:
        _embedding_queue.put((db_path, record_id, conteudo))


# ============================================================================
# DATABASE
# ============================================================================

def get_project_dir() -> Optional[Path]:
    """Detecta diretório do projeto atual"""
    cwd = os.environ.get("CLAUDE_CWD") or os.getcwd()
    return Path(cwd) if cwd else None


def get_project_db_path() -> Optional[Path]:
    """Retorna path do banco do projeto atual"""
    project_dir = get_project_dir()
    if not project_dir:
        return None
    memoria_dir = project_dir / ".claude" / "memoria"
    return memoria_dir / "projeto.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Inicializa banco SQLite com FTS5"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Tabela principal
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memorias (
            id TEXT PRIMARY KEY,
            tipo TEXT NOT NULL,
            conteudo TEXT NOT NULL,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB
        )
    """)

    # FTS5 para busca full-text
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memorias_fts USING fts5(
                conteudo,
                tags,
                content='memorias',
                content_rowid='rowid'
            )
        """)

        # Triggers para manter FTS sincronizado
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memorias_ai AFTER INSERT ON memorias BEGIN
                INSERT INTO memorias_fts(rowid, conteudo, tags)
                VALUES (NEW.rowid, NEW.conteudo, NEW.tags);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memorias_ad AFTER DELETE ON memorias BEGIN
                INSERT INTO memorias_fts(memorias_fts, rowid, conteudo, tags)
                VALUES('delete', OLD.rowid, OLD.conteudo, OLD.tags);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memorias_au AFTER UPDATE ON memorias BEGIN
                INSERT INTO memorias_fts(memorias_fts, rowid, conteudo, tags)
                VALUES('delete', OLD.rowid, OLD.conteudo, OLD.tags);
                INSERT INTO memorias_fts(rowid, conteudo, tags)
                VALUES (NEW.rowid, NEW.conteudo, NEW.tags);
            END
        """)
    except Exception as e:
        print(f"[Memoria] FTS5 setup warning: {e}", file=sys.stderr)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tipo ON memorias(tipo)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memorias(created_at)")

    conn.commit()
    return conn


def generate_id(conteudo: str, tipo: str) -> str:
    """Gera ID único baseado no conteúdo"""
    hash_input = f"{tipo}:{conteudo}:{datetime.now().isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# ============================================================================
# BUSCA
# ============================================================================

def buscar_fts(conn: sqlite3.Connection, query: str, limite: int = 10) -> list:
    """Busca usando FTS5 (instantâneo)"""
    cursor = conn.cursor()

    try:
        # Tokeniza query para FTS5
        tokens = query.split()
        if not tokens:
            return []

        # Busca com OR para ser mais flexível
        fts_query = " OR ".join(f'"{t}"' for t in tokens if t)

        cursor.execute("""
            SELECT m.id, m.tipo, m.conteudo, m.tags, m.created_at,
                   bm25(memorias_fts) as relevancia
            FROM memorias_fts f
            JOIN memorias m ON f.rowid = m.rowid
            WHERE memorias_fts MATCH ?
            ORDER BY relevancia
            LIMIT ?
        """, (fts_query, limite))

        return [
            {
                "id": row[0],
                "tipo": row[1],
                "conteudo": row[2],
                "tags": row[3],
                "created_at": row[4],
                "relevancia": round(abs(row[5]), 3),
                "metodo": "fts"
            }
            for row in cursor.fetchall()
        ]
    except Exception as e:
        print(f"[Memoria] FTS search error: {e}", file=sys.stderr)
        return []


def buscar_embedding(conn: sqlite3.Connection, query: str, limite: int = 10) -> list:
    """Busca usando embeddings (semântica)"""
    model = get_embedding_model()
    if not model:
        return []

    try:
        import numpy as np
        query_embedding = model.encode(query)

        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, tipo, conteudo, tags, created_at, embedding
            FROM memorias
            WHERE embedding IS NOT NULL
        """)

        resultados = []
        for row in cursor.fetchall():
            if row[5]:
                stored_embedding = np.frombuffer(row[5], dtype=np.float32)
                # Similaridade cosseno
                similarity = float(np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding) + 1e-8
                ))
                if similarity > 0.3:  # Threshold mínimo
                    resultados.append({
                        "id": row[0],
                        "tipo": row[1],
                        "conteudo": row[2],
                        "tags": row[3],
                        "created_at": row[4],
                        "relevancia": round(similarity, 3),
                        "metodo": "embedding"
                    })

        resultados.sort(key=lambda x: x["relevancia"], reverse=True)
        return resultados[:limite]
    except Exception as e:
        print(f"[Memoria] Embedding search error: {e}", file=sys.stderr)
        return []


def buscar_hibrido(conn: sqlite3.Connection, query: str, limite: int = 10) -> list:
    """Busca híbrida: FTS + embeddings, mescla resultados"""
    fts_results = buscar_fts(conn, query, limite)
    emb_results = buscar_embedding(conn, query, limite)

    # Mescla resultados, removendo duplicatas
    seen_ids = set()
    merged = []

    # Prioriza embeddings (mais semântico)
    for r in emb_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            merged.append(r)

    # Adiciona FTS que não estão nos embeddings
    for r in fts_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            merged.append(r)

    return merged[:limite]


# ============================================================================
# OPERAÇÕES
# ============================================================================

def salvar_memoria(db_path: Path, tipo: str, conteudo: str, tags: str = "") -> dict:
    """Salva memória (sync SQLite + async embedding)"""
    id = generate_id(conteudo, tipo)

    conn = init_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO memorias (id, tipo, conteudo, tags, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (id, tipo, conteudo, tags))

    conn.commit()
    conn.close()

    # Agenda embedding em background
    queue_embedding(db_path, id, conteudo)

    return {"id": id, "tipo": tipo, "saved": True}


def listar_memorias(conn: sqlite3.Connection, tipo: Optional[str] = None, limite: int = 20) -> list:
    """Lista memórias recentes"""
    cursor = conn.cursor()

    if tipo:
        cursor.execute("""
            SELECT id, tipo, conteudo, tags, created_at
            FROM memorias WHERE tipo = ?
            ORDER BY updated_at DESC LIMIT ?
        """, (tipo, limite))
    else:
        cursor.execute("""
            SELECT id, tipo, conteudo, tags, created_at
            FROM memorias ORDER BY updated_at DESC LIMIT ?
        """, (limite,))

    return [
        {"id": row[0], "tipo": row[1], "conteudo": row[2], "tags": row[3], "created_at": row[4]}
        for row in cursor.fetchall()
    ]


def get_stats(db_path: Path) -> dict:
    """Estatísticas do banco"""
    try:
        conn = init_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memorias")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memorias WHERE embedding IS NOT NULL")
        indexed = cursor.fetchone()[0]

        cursor.execute("SELECT tipo, COUNT(*) FROM memorias GROUP BY tipo")
        por_tipo = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()
        return {"total": total, "indexed": indexed, "por_tipo": por_tipo}
    except:
        return {"total": 0, "indexed": 0, "por_tipo": {}}


# ============================================================================
# MCP SERVER
# ============================================================================

server = Server("memoria")


@server.list_tools()
async def list_tools():
    """Lista ferramentas disponíveis"""
    return [
        Tool(
            name="memoria_contexto",
            description="USAR AUTOMATICAMENTE no início de cada conversa. Retorna memórias relevantes para o contexto atual (projeto + global). Funciona como um 'recall' automático.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Contexto atual ou pergunta do usuário"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memoria_buscar",
            description="Busca memórias específicas quando precisar de informação detalhada sobre decisões passadas, padrões ou preferências.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca"},
                    "escopo": {
                        "type": "string",
                        "enum": ["global", "projeto", "ambos"],
                        "default": "ambos"
                    },
                    "limite": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memoria_salvar",
            description="Salva decisão, padrão ou preferência importante. Use após: (1) tomar decisões de arquitetura, (2) definir padrões de código, (3) aprender preferências do usuário.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conteudo": {"type": "string", "description": "O que salvar"},
                    "tipo": {
                        "type": "string",
                        "enum": ["decisao", "padrao", "preferencia", "arquitetura", "todo", "nota"]
                    },
                    "escopo": {
                        "type": "string",
                        "enum": ["global", "projeto"],
                        "default": "projeto"
                    },
                    "tags": {"type": "string", "description": "Tags separadas por vírgula"}
                },
                "required": ["conteudo", "tipo"]
            }
        ),
        Tool(
            name="memoria_listar",
            description="Lista memórias recentes. Útil para revisar histórico de decisões.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tipo": {
                        "type": "string",
                        "enum": ["decisao", "padrao", "preferencia", "arquitetura", "todo", "nota"]
                    },
                    "escopo": {
                        "type": "string",
                        "enum": ["global", "projeto", "ambos"],
                        "default": "ambos"
                    },
                    "limite": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="memoria_stats",
            description="Mostra estatísticas da memória (total, indexados, por tipo).",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="memoria_deletar",
            description="Remove uma memória pelo ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "escopo": {"type": "string", "enum": ["global", "projeto"], "default": "projeto"}
                },
                "required": ["id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Executa ferramenta"""

    # ========== CONTEXTO (auto-recall) ==========
    if name == "memoria_contexto":
        query = arguments.get("query", "")
        resultados = []

        # Busca em ambos os escopos
        for escopo, db_path in [("global", GLOBAL_DB_PATH), ("projeto", get_project_db_path())]:
            if db_path and (db_path.exists() or escopo == "global"):
                try:
                    conn = init_db(db_path)
                    for r in buscar_hibrido(conn, query, limite=5):
                        r["escopo"] = escopo
                        resultados.append(r)
                    conn.close()
                except Exception as e:
                    print(f"[Memoria] Context search error ({escopo}): {e}", file=sys.stderr)

        if not resultados:
            return [TextContent(type="text", text="[Memoria] Nenhum contexto relevante encontrado.")]

        # Ordena por relevância
        resultados.sort(key=lambda x: x.get("relevancia", 0), reverse=True)
        resultados = resultados[:8]

        output = "## Contexto da Memória\n\n"
        for r in resultados:
            output += f"**[{r['escopo']}:{r['tipo']}]** {r['conteudo']}\n"
        output += "\n---\n_Use este contexto para informar suas respostas._"

        return [TextContent(type="text", text=output)]

    # ========== BUSCAR ==========
    elif name == "memoria_buscar":
        query = arguments.get("query", "")
        escopo = arguments.get("escopo", "ambos")
        limite = arguments.get("limite", 5)

        resultados = []

        if escopo in ["global", "ambos"]:
            try:
                conn = init_db(GLOBAL_DB_PATH)
                for r in buscar_hibrido(conn, query, limite):
                    r["escopo"] = "global"
                    resultados.append(r)
                conn.close()
            except Exception as e:
                print(f"[Memoria] Global search error: {e}", file=sys.stderr)

        if escopo in ["projeto", "ambos"]:
            project_db = get_project_db_path()
            if project_db:
                try:
                    conn = init_db(project_db)
                    for r in buscar_hibrido(conn, query, limite):
                        r["escopo"] = "projeto"
                        resultados.append(r)
                    conn.close()
                except Exception as e:
                    print(f"[Memoria] Project search error: {e}", file=sys.stderr)

        resultados.sort(key=lambda x: x.get("relevancia", 0), reverse=True)
        resultados = resultados[:limite]

        if not resultados:
            return [TextContent(type="text", text="Nenhuma memória encontrada.")]

        output = f"## Memórias ({len(resultados)})\n\n"
        for r in resultados:
            output += f"**[{r['escopo'].upper()}] {r['tipo']}** (relevância: {r['relevancia']}, método: {r.get('metodo', '?')})\n"
            output += f"{r['conteudo']}\n"
            if r.get('tags'):
                output += f"_Tags: {r['tags']}_\n"
            output += "\n"

        return [TextContent(type="text", text=output)]

    # ========== SALVAR ==========
    elif name == "memoria_salvar":
        conteudo = arguments.get("conteudo", "")
        tipo = arguments.get("tipo", "nota")
        escopo = arguments.get("escopo", "projeto")
        tags = arguments.get("tags", "")

        if not conteudo:
            return [TextContent(type="text", text="Erro: conteúdo vazio.")]

        db_path = GLOBAL_DB_PATH if escopo == "global" else get_project_db_path()
        if not db_path:
            return [TextContent(type="text", text="Erro: projeto não detectado.")]

        try:
            result = salvar_memoria(db_path, tipo, conteudo, tags)
            return [TextContent(
                type="text",
                text=f"✓ Memória salva ({escopo})\n- Tipo: {tipo}\n- ID: {result['id']}\n- Embedding: {'em background' if EMBEDDING_ENABLED else 'desabilitado'}"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]

    # ========== LISTAR ==========
    elif name == "memoria_listar":
        tipo = arguments.get("tipo")
        escopo = arguments.get("escopo", "ambos")
        limite = arguments.get("limite", 10)

        resultados = []

        if escopo in ["global", "ambos"]:
            try:
                conn = init_db(GLOBAL_DB_PATH)
                for r in listar_memorias(conn, tipo, limite):
                    r["escopo"] = "global"
                    resultados.append(r)
                conn.close()
            except Exception as e:
                print(f"[Memoria] Global list error: {e}", file=sys.stderr)

        if escopo in ["projeto", "ambos"]:
            project_db = get_project_db_path()
            if project_db:
                try:
                    conn = init_db(project_db)
                    for r in listar_memorias(conn, tipo, limite):
                        r["escopo"] = "projeto"
                        resultados.append(r)
                    conn.close()
                except Exception as e:
                    print(f"[Memoria] Project list error: {e}", file=sys.stderr)

        if not resultados:
            return [TextContent(type="text", text="Nenhuma memória encontrada.")]

        output = f"## Memórias ({len(resultados)})\n\n"
        for r in resultados:
            output += f"- **[{r['escopo']}] {r['tipo']}**: {r['conteudo'][:80]}{'...' if len(r['conteudo']) > 80 else ''}\n"
            output += f"  `{r['id']}` | {r['created_at']}\n\n"

        return [TextContent(type="text", text=output)]

    # ========== STATS ==========
    elif name == "memoria_stats":
        output = "## Estatísticas da Memória\n\n"

        global_stats = get_stats(GLOBAL_DB_PATH)
        output += f"**Global** ({GLOBAL_DB_PATH}):\n"
        output += f"- Total: {global_stats['total']}\n"
        output += f"- Indexados (embedding): {global_stats['indexed']}\n"
        output += f"- Por tipo: {global_stats['por_tipo']}\n\n"

        project_db = get_project_db_path()
        if project_db and project_db.exists():
            proj_stats = get_stats(project_db)
            output += f"**Projeto** ({project_db}):\n"
            output += f"- Total: {proj_stats['total']}\n"
            output += f"- Indexados (embedding): {proj_stats['indexed']}\n"
            output += f"- Por tipo: {proj_stats['por_tipo']}\n\n"

        output += f"**Config**:\n"
        output += f"- Embeddings: {'habilitado' if EMBEDDING_ENABLED else 'desabilitado'}\n"
        output += f"- Modelo: {EMBEDDING_MODEL}\n"
        output += f"- Queue pendente: {_embedding_queue.qsize()}\n"

        return [TextContent(type="text", text=output)]

    # ========== DELETAR ==========
    elif name == "memoria_deletar":
        id = arguments.get("id", "")
        escopo = arguments.get("escopo", "projeto")

        if not id:
            return [TextContent(type="text", text="Erro: ID obrigatório.")]

        db_path = GLOBAL_DB_PATH if escopo == "global" else get_project_db_path()
        if not db_path:
            return [TextContent(type="text", text="Erro: projeto não detectado.")]

        try:
            conn = init_db(db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memorias WHERE id = ?", (id,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted:
                return [TextContent(type="text", text=f"✓ Memória {id} deletada.")]
            return [TextContent(type="text", text=f"Memória {id} não encontrada.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]

    return [TextContent(type="text", text=f"Ferramenta desconhecida: {name}")]


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Inicia servidor MCP"""
    print("[Memoria] ============================================", file=sys.stderr)
    print("[Memoria] MCP Memoria Server Starting", file=sys.stderr)
    print(f"[Memoria] Global DB: {GLOBAL_DB_PATH}", file=sys.stderr)
    print(f"[Memoria] Embeddings: {'enabled' if EMBEDDING_ENABLED else 'disabled'}", file=sys.stderr)
    print(f"[Memoria] Model: {EMBEDDING_MODEL}", file=sys.stderr)
    print("[Memoria] ============================================", file=sys.stderr)

    # Inicializa banco global
    init_db(GLOBAL_DB_PATH)

    # Inicia worker de embedding em background
    start_embedding_worker()

    # Pre-load do modelo de embedding (em background)
    if EMBEDDING_ENABLED:
        threading.Thread(target=get_embedding_model, daemon=True).start()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
