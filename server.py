#!/usr/bin/env python3
"""
MCP Memory v2 - Servidor de memória persistente para assistentes AI

Melhorias sobre v1:
1. Busca híbrida com pesos (0.7 vector + 0.3 BM25)
2. Decay temporal (memórias recentes recebem boost)
3. Deduplicação (Jaccard check antes de salvar)
4. Chunking (memórias longas divididas para melhor embedding)
5. Cache de embeddings (evita recomputar texto idêntico)
6. Tool memory_reindex (processa backlog de embeddings)
7. Tool memory_compact (consolida conversas por projeto)
8. Auto-reindex na inicialização + periódico a cada 5min

Três camadas:
- Global: ~/.mcp-memoria/data/global.db (preferências, padrões)
- Project: .mcp-memoria/project.db (específico do projeto)
- Personality: ~/.mcp-memoria/data/personality.db (cross-project)
"""

import os
import re
import sys
import math
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
PERSONALITY_DB_PATH = GLOBAL_MEMORY_DIR / "personality.db"

# Embedding config
EMBEDDING_ENABLED = os.environ.get("MCP_MEMORY_EMBEDDING", "true").lower() == "true"
EMBEDDING_MODEL = os.environ.get("MCP_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Pesos da busca híbrida
VECTOR_WEIGHT = 0.7
TEXT_WEIGHT = 0.3
MIN_SIMILARITY_THRESHOLD = 0.3

# Decay temporal (0 = sem decay, 1 = decay máximo)
DECAY_STRENGTH = 0.15

# Deduplicação (threshold Jaccard)
DEDUP_THRESHOLD = 0.85

# Chunking
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 80

# Reindex periódico (segundos)
PERIODIC_REINDEX_INTERVAL = 300  # 5 minutos

# Background processing
_embedding_queue: queue.Queue = queue.Queue()
_embedding_thread: Optional[threading.Thread] = None
_embedding_model = None
_embedding_model_lock = threading.Lock()


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Divide texto em chunks com overlap por contagem de palavras"""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


# ============================================================================
# DEDUPLICATION
# ============================================================================

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Similaridade Jaccard rápida por palavras"""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def find_duplicate(conn: sqlite3.Connection, content: str, mem_type: str,
                   threshold: float = DEDUP_THRESHOLD) -> Optional[str]:
    """Verifica se memória similar já existe. Retorna ID existente ou None."""
    cursor = conn.cursor()

    # Passo 1: check de conteúdo exato (instantâneo)
    cursor.execute(
        "SELECT id FROM memories WHERE type = ? AND content = ?",
        (mem_type, content)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Passo 2: FTS rough match + refinamento Jaccard
    tokens = content.split()[:20]
    if not tokens:
        return None

    fts_terms = [t for t in tokens if len(t) > 2]
    if not fts_terms:
        return None

    fts_query = " OR ".join(f'"{t}"' for t in fts_terms)

    try:
        cursor.execute("""
            SELECT m.id, m.content
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.rowid
            WHERE m.type = ? AND memories_fts MATCH ?
            LIMIT 10
        """, (mem_type, fts_query))

        for row in cursor.fetchall():
            similarity = jaccard_similarity(content, row[1])
            if similarity >= threshold:
                return row[0]
    except Exception:
        pass

    return None


# ============================================================================
# EMBEDDING CACHE
# ============================================================================

def get_cached_embedding(conn: sqlite3.Connection, text: str, model: str):
    """Busca embedding no cache por hash do texto"""
    text_hash = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model = ?",
            (text_hash, model)
        )
        row = cursor.fetchone()
        if row and row[0]:
            import numpy as np
            return np.frombuffer(row[0], dtype=np.float32).copy()
    except Exception:
        pass
    return None


def store_cached_embedding(db_path: Path, text: str, model: str,
                           embedding_blob: bytes):
    """Armazena embedding no cache"""
    text_hash = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding, created_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (text_hash, model, embedding_blob)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


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
                print(f"[Memory] Loading embedding model: {EMBEDDING_MODEL}",
                      file=sys.stderr)
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print("[Memory] Embedding model ready", file=sys.stderr)
            except ImportError:
                print("[Memory] sentence-transformers not installed, FTS only",
                      file=sys.stderr)
                return None
            except Exception as e:
                print(f"[Memory] Embedding load error: {e}", file=sys.stderr)
                return None
        return _embedding_model


def compute_embedding(model, text: str):
    """Computa embedding para texto, retorna numpy array float32"""
    import numpy as np
    return model.encode(text).astype(np.float32)


def embedding_worker():
    """Worker com cache + chunking + embedding"""
    print("[Memory] Background embedding worker started", file=sys.stderr)

    while True:
        try:
            item = _embedding_queue.get(timeout=5)

            if item is None:  # Poison pill
                break

            db_path, record_id, content = item

            model = get_embedding_model()
            if not model:
                continue

            try:
                import numpy as np
                conn = sqlite3.connect(str(db_path))
                conn.execute("PRAGMA foreign_keys=ON")

                # Check cache para conteúdo principal
                cached = get_cached_embedding(conn, content, EMBEDDING_MODEL)
                if cached is not None:
                    embedding_blob = cached.tobytes()
                else:
                    embedding = compute_embedding(model, content)
                    embedding_blob = embedding.tobytes()
                    store_cached_embedding(db_path, content, EMBEDDING_MODEL,
                                           embedding_blob)

                # Atualiza embedding da memória principal
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (embedding_blob, record_id)
                )

                # Chunk conteúdos longos
                chunks = chunk_text(content)
                if len(chunks) > 1:
                    # Remove chunks antigos
                    conn.execute(
                        "DELETE FROM memory_chunks WHERE memory_id = ?",
                        (record_id,)
                    )

                    for idx, chunk in enumerate(chunks):
                        chunk_id = f"{record_id}_c{idx}"

                        # Check cache do chunk
                        cached_chunk = get_cached_embedding(
                            conn, chunk, EMBEDDING_MODEL
                        )
                        if cached_chunk is not None:
                            chunk_blob = cached_chunk.tobytes()
                        else:
                            chunk_emb = compute_embedding(model, chunk)
                            chunk_blob = chunk_emb.tobytes()
                            store_cached_embedding(
                                db_path, chunk, EMBEDDING_MODEL, chunk_blob
                            )

                        conn.execute(
                            "INSERT OR REPLACE INTO memory_chunks "
                            "(id, memory_id, chunk_index, chunk_text, embedding) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (chunk_id, record_id, idx, chunk, chunk_blob)
                        )

                conn.commit()
                conn.close()

            except Exception as e:
                print(f"[Memory] Embedding error for {record_id}: {e}",
                      file=sys.stderr)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Memory] Worker error: {e}", file=sys.stderr)


def start_embedding_worker():
    """Inicia worker thread se embeddings habilitados"""
    global _embedding_thread

    if not EMBEDDING_ENABLED:
        return

    if _embedding_thread is None or not _embedding_thread.is_alive():
        _embedding_thread = threading.Thread(
            target=embedding_worker, daemon=True
        )
        _embedding_thread.start()


def queue_embedding(db_path: Path, record_id: str, content: str):
    """Adiciona item na fila para embedding em background"""
    if EMBEDDING_ENABLED:
        _embedding_queue.put((db_path, record_id, content))


# ============================================================================
# DATABASE
# ============================================================================

def get_project_dir() -> Optional[Path]:
    """Detecta diretório do projeto atual"""
    cwd = (os.environ.get("MCP_PROJECT_DIR")
           or os.environ.get("CLAUDE_CWD")
           or os.getcwd())
    return Path(cwd) if cwd else None


def get_project_db_path() -> Optional[Path]:
    """Retorna path do database do projeto"""
    project_dir = get_project_dir()
    if not project_dir:
        return None
    return project_dir / ".mcp-memoria" / "project.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Inicializa SQLite com schema melhorado"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # WAL mode para melhor leitura concorrente
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")

    # Tabela principal
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

    # Tabela de chunks (para memórias longas)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)

    # Cache de embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (text_hash, model)
        )
    """)

    # FTS5 para busca full-text
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, content='memories', content_rowid='rowid'
            )
        """)

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

    # Índices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_memory "
        "ON memory_chunks(memory_id)"
    )

    conn.commit()
    return conn


def generate_id(content: str, mem_type: str) -> str:
    """Gera ID único baseado em conteúdo + timestamp"""
    hash_input = f"{mem_type}:{content}:{datetime.now().isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# ============================================================================
# SEARCH (Enhanced)
# ============================================================================

def apply_temporal_decay(score: float, created_at_str: str) -> float:
    """Aplica boost temporal: memórias recentes recebem score maior"""
    try:
        created_at = datetime.fromisoformat(created_at_str)
        days_old = max(0, (datetime.now() - created_at).days)
        recency = 1.0 / (1.0 + math.log1p(days_old))
        return score * (1.0 - DECAY_STRENGTH + DECAY_STRENGTH * recency)
    except (ValueError, TypeError):
        return score


def search_fts(conn: sqlite3.Connection, query: str,
               limit: int = 10) -> list:
    """Busca FTS5 com scores BM25 normalizados"""
    cursor = conn.cursor()

    try:
        tokens = query.split()
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in tokens if t)

        cursor.execute("""
            SELECT m.id, m.type, m.content, m.tags, m.created_at,
                   bm25(memories_fts) as bm25_score
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.rowid
            WHERE memories_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
        """, (fts_query, limit * 3))  # Busca mais pra merge híbrido

        results = []
        for row in cursor.fetchall():
            bm25_raw = abs(row[5])
            # Normalização sigmoid: mapeia para (0, 1)
            bm25_normalized = bm25_raw / (bm25_raw + 1.0)
            score = apply_temporal_decay(bm25_normalized, row[4])

            results.append({
                "id": row[0], "type": row[1], "content": row[2],
                "tags": row[3], "created_at": row[4],
                "relevance": round(score, 4),
                "bm25_raw": round(bm25_normalized, 4),
                "method": "fts"
            })

        return results
    except Exception as e:
        print(f"[Memory] FTS search error: {e}", file=sys.stderr)
        return []


def search_embedding(conn: sqlite3.Connection, query: str,
                     limit: int = 10) -> list:
    """Busca por embeddings (memórias + chunks) com decay temporal"""
    model = get_embedding_model()
    if not model:
        return []

    try:
        import numpy as np
        query_embedding = compute_embedding(model, query)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-8:
            return []

        cursor = conn.cursor()

        # Busca nos embeddings principais
        cursor.execute("""
            SELECT id, type, content, tags, created_at, embedding
            FROM memories WHERE embedding IS NOT NULL
        """)

        results_map = {}  # memory_id -> melhor resultado

        for row in cursor.fetchall():
            if not row[5]:
                continue
            stored = np.frombuffer(row[5], dtype=np.float32)
            stored_norm = np.linalg.norm(stored)
            if stored_norm < 1e-8:
                continue
            cosine = float(
                np.dot(query_embedding, stored) / (query_norm * stored_norm)
            )

            if cosine > MIN_SIMILARITY_THRESHOLD:
                score = apply_temporal_decay(cosine, row[4])
                mem_id = row[0]
                if (mem_id not in results_map
                        or score > results_map[mem_id]["relevance"]):
                    results_map[mem_id] = {
                        "id": mem_id, "type": row[1], "content": row[2],
                        "tags": row[3], "created_at": row[4],
                        "relevance": round(score, 4),
                        "cosine_raw": round(cosine, 4),
                        "method": "embedding"
                    }

        # Busca nos embeddings de chunks
        try:
            cursor.execute("""
                SELECT c.memory_id, c.embedding,
                       m.type, m.content, m.tags, m.created_at
                FROM memory_chunks c
                JOIN memories m ON c.memory_id = m.id
                WHERE c.embedding IS NOT NULL
            """)

            for row in cursor.fetchall():
                if not row[1]:
                    continue
                stored = np.frombuffer(row[1], dtype=np.float32)
                stored_norm = np.linalg.norm(stored)
                if stored_norm < 1e-8:
                    continue
                cosine = float(
                    np.dot(query_embedding, stored)
                    / (query_norm * stored_norm)
                )

                if cosine > MIN_SIMILARITY_THRESHOLD:
                    score = apply_temporal_decay(cosine, row[5])
                    mem_id = row[0]
                    if (mem_id not in results_map
                            or score > results_map[mem_id]["relevance"]):
                        results_map[mem_id] = {
                            "id": mem_id, "type": row[2], "content": row[3],
                            "tags": row[4], "created_at": row[5],
                            "relevance": round(score, 4),
                            "cosine_raw": round(cosine, 4),
                            "method": "embedding-chunk"
                        }
        except Exception:
            pass  # Tabela chunks pode não existir em DBs antigos

        results = sorted(
            results_map.values(),
            key=lambda x: x["relevance"],
            reverse=True
        )
        return results[:limit]

    except Exception as e:
        print(f"[Memory] Embedding search error: {e}", file=sys.stderr)
        return []


def search_hybrid(conn: sqlite3.Connection, query: str,
                  limit: int = 10) -> list:
    """Busca híbrida ponderada: VECTOR_WEIGHT * embedding + TEXT_WEIGHT * BM25"""
    fts_results = search_fts(conn, query, limit)
    emb_results = search_embedding(conn, query, limit)

    # Mapa de scores: id -> {fts_score, emb_score, data}
    score_map = {}

    for r in fts_results:
        mid = r["id"]
        if mid not in score_map:
            score_map[mid] = {"fts": 0.0, "emb": 0.0, "data": r}
        score_map[mid]["fts"] = max(
            score_map[mid]["fts"], r.get("bm25_raw", r["relevance"])
        )

    for r in emb_results:
        mid = r["id"]
        if mid not in score_map:
            score_map[mid] = {"fts": 0.0, "emb": 0.0, "data": r}
        score_map[mid]["emb"] = max(
            score_map[mid]["emb"], r.get("cosine_raw", r["relevance"])
        )
        # Prefere dados do embedding (mais ricos)
        score_map[mid]["data"] = r

    # Score final ponderado
    merged = []
    for mid, scores in score_map.items():
        raw_hybrid = (VECTOR_WEIGHT * scores["emb"]
                      + TEXT_WEIGHT * scores["fts"])
        final_score = apply_temporal_decay(
            raw_hybrid, scores["data"]["created_at"]
        )

        result = scores["data"].copy()
        result["relevance"] = round(final_score, 4)
        if scores["emb"] > 0 and scores["fts"] > 0:
            result["method"] = "hybrid"
        merged.append(result)

    merged.sort(key=lambda x: x["relevance"], reverse=True)
    return merged[:limit]


# ============================================================================
# OPERATIONS
# ============================================================================

def save_memory(db_path: Path, mem_type: str, content: str,
                tags: str = "", force_id: str = "") -> dict:
    """Salva memória com verificação de deduplicação"""
    conn = init_db(db_path)

    # Dedup check (pula para conversations que têm seu próprio dedup)
    existing_id = None
    if mem_type != "conversation":
        existing_id = find_duplicate(conn, content, mem_type)

    if existing_id:
        # Atualiza memória existente
        conn.execute(
            "UPDATE memories SET content = ?, tags = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (content, tags, existing_id)
        )
        conn.commit()
        conn.close()
        queue_embedding(db_path, existing_id, content)
        return {"id": existing_id, "type": mem_type,
                "saved": True, "dedup": "updated"}

    # Memória nova
    mem_id = force_id or generate_id(content, mem_type)
    conn.execute(
        "INSERT OR REPLACE INTO memories "
        "(id, type, content, tags, updated_at) "
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (mem_id, mem_type, content, tags)
    )
    conn.commit()
    conn.close()

    queue_embedding(db_path, mem_id, content)
    return {"id": mem_id, "type": mem_type, "saved": True, "dedup": "new"}


def reindex_pending(db_path: Path) -> int:
    """Enfileira todas as memórias sem embedding para processamento"""
    try:
        conn = init_db(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM memories WHERE embedding IS NULL")
        count = 0
        for row in cursor.fetchall():
            queue_embedding(db_path, row[0], row[1])
            count += 1
        conn.close()
        return count
    except Exception as e:
        print(f"[Memory] Reindex error: {e}", file=sys.stderr)
        return 0


def compact_conversations(db_path: Path) -> dict:
    """Consolida memórias de conversas por projeto"""
    conn = init_db(db_path)
    cursor = conn.cursor()

    # Busca todas as conversas
    cursor.execute("""
        SELECT id, content, tags, created_at
        FROM memories WHERE type = 'conversation'
        ORDER BY created_at
    """)
    conversations = cursor.fetchall()

    if not conversations:
        conn.close()
        return {"compacted": 0, "deleted": 0, "created": 0, "groups": 0}

    # Agrupa por projeto (extraído de tags ou conteúdo)
    groups = {}
    for row in conversations:
        mem_id, content, tags, created = row

        # Extrai projeto das tags
        project = "unknown"
        if tags:
            for tag in tags.split(","):
                tag = tag.strip()
                if tag and tag not in (
                    "conversation", "claude-code", "auto-saved", "compacted"
                ):
                    project = tag
                    break

        # Ou do conteúdo [project_name]
        match = re.match(r'\[([^\]]+)\]', content)
        if match:
            project = match.group(1)

        if project not in groups:
            groups[project] = []
        groups[project].append({
            "id": mem_id, "content": content,
            "tags": tags, "created": created
        })

    deleted_count = 0
    created_count = 0

    for project, memories in groups.items():
        if len(memories) < 3:
            continue  # Não compacta grupos pequenos

        # Extrai informações de todas as conversas
        all_tools = set()
        user_prompts = []
        date_range_start = memories[0]["created"]
        date_range_end = memories[-1]["created"]

        for mem in memories:
            content = mem["content"]

            # Extrai tools
            tools_match = re.findall(r'Tools?:\s*([^\n]+)', content)
            for t in tools_match:
                for tool in t.split(","):
                    tool = tool.strip()
                    if tool:
                        all_tools.add(tool)

            # Extrai prompts do usuário
            user_matches = re.findall(r'\[user\]\s*(.+)', content)
            for u in user_matches:
                u = u.strip()[:200]
                if u and len(u) > 10:
                    user_prompts.append(u)

        # Deduplica prompts
        seen_prompts = set()
        unique_prompts = []
        for p in user_prompts:
            p_key = p[:50].lower()
            if p_key not in seen_prompts:
                seen_prompts.add(p_key)
                unique_prompts.append(p)

        # Monta conteúdo consolidado
        consolidated = (
            f"[{project}] Consolidated from {len(memories)} sessions "
            f"({date_range_start} to {date_range_end})\n"
        )
        if all_tools:
            consolidated += (
                f"Tools used: {', '.join(sorted(all_tools)[:30])}\n"
            )
        if unique_prompts:
            consolidated += f"Key topics ({len(unique_prompts)}):\n"
            for p in unique_prompts[:50]:
                consolidated += f"  - {p}\n"

        # Salva memória consolidada
        tags = f"conversation,compacted,{project}"
        consolidated_id = generate_id(consolidated, "conversation")
        conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(id, type, content, tags, updated_at) "
            "VALUES (?, 'conversation', ?, ?, CURRENT_TIMESTAMP)",
            (consolidated_id, consolidated, tags)
        )
        created_count += 1

        # Deleta originais
        for mem in memories:
            conn.execute(
                "DELETE FROM memories WHERE id = ?", (mem["id"],)
            )
            deleted_count += 1

        queue_embedding(db_path, consolidated_id, consolidated)

    conn.commit()
    conn.close()

    return {
        "compacted": len([g for g in groups.values() if len(g) >= 3]),
        "deleted": deleted_count,
        "created": created_count,
        "groups": len(groups)
    }


def list_memories(conn: sqlite3.Connection,
                  mem_type: Optional[str] = None,
                  limit: int = 20) -> list:
    """Lista memórias recentes"""
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
        {"id": row[0], "type": row[1], "content": row[2],
         "tags": row[3], "created_at": row[4]}
        for row in cursor.fetchall()
    ]


def get_stats(db_path: Path) -> dict:
    """Estatísticas do database"""
    try:
        conn = init_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        )
        indexed = cursor.fetchone()[0]

        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Stats de chunks
        chunks = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM memory_chunks")
            chunks = cursor.fetchone()[0]
        except Exception:
            pass

        # Stats do cache
        cache = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM embedding_cache")
            cache = cursor.fetchone()[0]
        except Exception:
            pass

        conn.close()
        return {
            "total": total, "indexed": indexed,
            "by_type": by_type, "chunks": chunks,
            "cache_entries": cache
        }
    except Exception:
        return {
            "total": 0, "indexed": 0, "by_type": {},
            "chunks": 0, "cache_entries": 0
        }


# ============================================================================
# MCP SERVER
# ============================================================================

server = Server("memory")


def resolve_scope_dbs(scope: str) -> list:
    """Resolve scope para lista de (nome, path)"""
    if scope == "global":
        return [("global", GLOBAL_DB_PATH)]
    elif scope == "project":
        return [("project", get_project_db_path())]
    elif scope == "personality":
        return [("personality", PERSONALITY_DB_PATH)]
    elif scope == "both":
        return [
            ("global", GLOBAL_DB_PATH),
            ("project", get_project_db_path())
        ]
    elif scope == "all":
        return [
            ("global", GLOBAL_DB_PATH),
            ("project", get_project_db_path()),
            ("personality", PERSONALITY_DB_PATH)
        ]
    return []


@server.list_tools()
async def list_tools():
    """Lista de ferramentas disponíveis"""
    return [
        Tool(
            name="memory_context",
            description=(
                "USE AUTOMATICALLY at the start of each conversation. "
                "Returns relevant memories for the current context "
                "(project + global). Works as an automatic 'recall'."
            ),
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
            description=(
                "Search specific memories when you need detailed information "
                "about past decisions, patterns, or preferences. Use "
                "'personality' scope to find similar implementations "
                "from other projects."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term"
                    },
                    "scope": {
                        "type": "string",
                        "enum": [
                            "global", "project", "personality", "both", "all"
                        ],
                        "description": (
                            "'both'=global+project, "
                            "'all'=global+project+personality, "
                            "'personality'=cross-project implementations only"
                        ),
                        "default": "both"
                    },
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_save",
            description=(
                "Save important decision, pattern, or implementation. "
                "Use after: (1) making architecture decisions, "
                "(2) defining code patterns, (3) learning user preferences, "
                "(4) implementing new features "
                "(use scope='personality' + type='implementation')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "What to save"
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "decision", "pattern", "preference",
                            "architecture", "implementation",
                            "solution", "todo", "note"
                        ],
                        "description": (
                            "'implementation'/'solution' for code/features "
                            "(recommended with scope='personality')"
                        )
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project", "personality"],
                        "description": (
                            "'personality' saves cross-project "
                            "(for implementations/solutions)"
                        ),
                        "default": "project"
                    },
                    "tags": {
                        "type": "string",
                        "description": (
                            "Comma-separated tags "
                            "(e.g., 'python,fastapi,auth')"
                        )
                    },
                    "project_name": {
                        "type": "string",
                        "description": (
                            "Project name (auto-detected if not provided, "
                            "useful for personality scope)"
                        )
                    }
                },
                "required": ["content", "type"]
            }
        ),
        Tool(
            name="memory_list",
            description=(
                "List recent memories. Useful to review decision history "
                "or find past implementations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "decision", "pattern", "preference",
                            "architecture", "implementation",
                            "solution", "todo", "note"
                        ]
                    },
                    "scope": {
                        "type": "string",
                        "enum": [
                            "global", "project", "personality", "both", "all"
                        ],
                        "description": (
                            "'both'=global+project, "
                            "'all'=includes personality"
                        ),
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
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project", "personality"],
                        "default": "project"
                    }
                },
                "required": ["id"]
            }
        ),
        Tool(
            name="memory_reindex",
            description=(
                "Reindex all memories that don't have embeddings yet. "
                "Use when embedding backlog is large."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": [
                            "global", "project", "personality", "all"
                        ],
                        "description": "Which database(s) to reindex",
                        "default": "all"
                    }
                }
            }
        ),
        Tool(
            name="memory_compact",
            description=(
                "Consolidate conversation memories by project. "
                "Reduces noise by merging multiple conversation entries "
                "into summaries. Run periodically for maintenance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["personality", "project", "global"],
                        "description": "Which database to compact",
                        "default": "personality"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Executa ferramenta"""

    # ========== CONTEXT (auto-recall) ==========
    if name == "memory_context":
        query = arguments.get("query", "")
        results = []

        for scope, db_path in resolve_scope_dbs("all"):
            if db_path and (
                db_path.exists() or scope in ("global", "personality")
            ):
                try:
                    conn = init_db(db_path)
                    for r in search_hybrid(conn, query, limit=5):
                        r["scope"] = scope
                        results.append(r)
                    conn.close()
                except Exception as e:
                    print(
                        f"[Memory] Context search error ({scope}): {e}",
                        file=sys.stderr
                    )

        if not results:
            return [TextContent(
                type="text",
                text="[Memory] No relevant context found."
            )]

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

        for scope_name, db_path in resolve_scope_dbs(scope):
            if db_path and (
                db_path.exists() or scope_name in ("global", "personality")
            ):
                try:
                    conn = init_db(db_path)
                    for r in search_hybrid(conn, query, limit):
                        r["scope"] = scope_name
                        results.append(r)
                    conn.close()
                except Exception as e:
                    print(
                        f"[Memory] {scope_name} search error: {e}",
                        file=sys.stderr
                    )

        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        results = results[:limit]

        if not results:
            return [TextContent(type="text", text="No memories found.")]

        output = f"## Memories ({len(results)})\n\n"
        for r in results:
            output += (
                f"**[{r['scope'].upper()}] {r['type']}** "
                f"(relevance: {r['relevance']}, "
                f"method: {r.get('method', '?')})\n"
            )
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
        project_name = arguments.get("project_name", "")

        if not content:
            return [TextContent(type="text", text="Error: empty content.")]

        if scope == "global":
            db_path = GLOBAL_DB_PATH
        elif scope == "personality":
            db_path = PERSONALITY_DB_PATH
            if not project_name:
                project_dir = get_project_dir()
                project_name = (
                    project_dir.name if project_dir else "no-project"
                )
            if project_name and project_name not in tags:
                tags = f"{tags},{project_name}" if tags else project_name
        else:  # project
            db_path = get_project_db_path()
            if not db_path:
                return [TextContent(
                    type="text",
                    text=(
                        "Error: project not detected. "
                        "Use scope='personality' or 'global'."
                    )
                )]

        try:
            result = save_memory(db_path, mem_type, content, tags)
            dedup_info = ""
            if result.get("dedup") == "updated":
                dedup_info = "\n- Dedup: updated existing (similar found)"
            return [TextContent(
                type="text",
                text=(
                    f"Memory saved ({scope})\n"
                    f"- Type: {mem_type}\n"
                    f"- ID: {result['id']}\n"
                    f"- Tags: {tags or 'none'}\n"
                    f"- Embedding: "
                    f"{'queued' if EMBEDDING_ENABLED else 'disabled'}"
                    f"{dedup_info}"
                )
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ========== LIST ==========
    elif name == "memory_list":
        mem_type = arguments.get("type")
        scope = arguments.get("scope", "both")
        limit = arguments.get("limit", 10)

        results = []

        for scope_name, db_path in resolve_scope_dbs(scope):
            if db_path and (
                db_path.exists() or scope_name in ("global", "personality")
            ):
                try:
                    conn = init_db(db_path)
                    for r in list_memories(conn, mem_type, limit):
                        r["scope"] = scope_name
                        results.append(r)
                    conn.close()
                except Exception as e:
                    print(
                        f"[Memory] {scope_name} list error: {e}",
                        file=sys.stderr
                    )

        if not results:
            return [TextContent(type="text", text="No memories found.")]

        output = f"## Memories ({len(results)})\n\n"
        for r in results:
            truncated = r['content'][:80]
            ellipsis = '...' if len(r['content']) > 80 else ''
            output += (
                f"- **[{r['scope']}] {r['type']}**: "
                f"{truncated}{ellipsis}\n"
            )
            if r.get('tags'):
                output += f"  _Tags: {r['tags']}_\n"
            output += f"  `{r['id']}` | {r['created_at']}\n\n"

        return [TextContent(type="text", text=output)]

    # ========== STATS ==========
    elif name == "memory_stats":
        output = "## Memory Statistics\n\n"

        for label, db_path in [
            ("Global", GLOBAL_DB_PATH),
            ("Personality", PERSONALITY_DB_PATH)
        ]:
            stats = get_stats(db_path)
            output += f"**{label}** ({db_path}):\n"
            output += f"- Total: {stats['total']}\n"
            output += f"- Indexed (embedding): {stats['indexed']}\n"
            output += f"- Chunks: {stats['chunks']}\n"
            output += f"- Cache entries: {stats['cache_entries']}\n"
            output += f"- By type: {stats['by_type']}\n\n"

        project_db = get_project_db_path()
        if project_db and project_db.exists():
            stats = get_stats(project_db)
            output += f"**Project** ({project_db}):\n"
            output += f"- Total: {stats['total']}\n"
            output += f"- Indexed (embedding): {stats['indexed']}\n"
            output += f"- Chunks: {stats['chunks']}\n"
            output += f"- Cache entries: {stats['cache_entries']}\n"
            output += f"- By type: {stats['by_type']}\n\n"

        output += "**Config**:\n"
        output += (
            f"- Embeddings: "
            f"{'enabled' if EMBEDDING_ENABLED else 'disabled'}\n"
        )
        output += f"- Model: {EMBEDDING_MODEL}\n"
        output += f"- Queue pending: {_embedding_queue.qsize()}\n"
        output += (
            f"- Search weights: "
            f"vector={VECTOR_WEIGHT}, text={TEXT_WEIGHT}\n"
        )
        output += f"- Temporal decay: {DECAY_STRENGTH}\n"
        output += f"- Dedup threshold: {DEDUP_THRESHOLD}\n"
        output += (
            f"- Chunk size: {CHUNK_SIZE_WORDS} words "
            f"(overlap {CHUNK_OVERLAP_WORDS})\n"
        )

        return [TextContent(type="text", text=output)]

    # ========== DELETE ==========
    elif name == "memory_delete":
        mem_id = arguments.get("id", "")
        scope = arguments.get("scope", "project")

        if not mem_id:
            return [TextContent(type="text", text="Error: ID required.")]

        if scope == "global":
            db_path = GLOBAL_DB_PATH
        elif scope == "personality":
            db_path = PERSONALITY_DB_PATH
        else:
            db_path = get_project_db_path()
            if not db_path:
                return [TextContent(
                    type="text",
                    text="Error: project not detected."
                )]

        try:
            conn = init_db(db_path)
            cursor = conn.cursor()
            # Chunks deletados por CASCADE
            cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted:
                return [TextContent(
                    type="text",
                    text=f"Memory {mem_id} deleted."
                )]
            return [TextContent(
                type="text",
                text=f"Memory {mem_id} not found."
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ========== REINDEX ==========
    elif name == "memory_reindex":
        scope = arguments.get("scope", "all")
        total = 0
        details = []

        dbs = []
        if scope == "all":
            dbs = [
                ("global", GLOBAL_DB_PATH),
                ("personality", PERSONALITY_DB_PATH)
            ]
            project_db = get_project_db_path()
            if project_db and project_db.exists():
                dbs.append(("project", project_db))
        else:
            for scope_name, path in resolve_scope_dbs(scope):
                if path:
                    dbs.append((scope_name, path))

        for db_name, db_path in dbs:
            count = reindex_pending(db_path)
            total += count
            details.append(f"- {db_name}: {count} queued")

        return [TextContent(
            type="text",
            text=(
                f"## Reindex Started\n\n"
                f"Queued {total} memories for embedding.\n"
                + "\n".join(details)
                + f"\n\nWorker processing in background. "
                  f"Queue size: {_embedding_queue.qsize()}"
            )
        )]

    # ========== COMPACT ==========
    elif name == "memory_compact":
        scope = arguments.get("scope", "personality")

        if scope == "personality":
            db_path = PERSONALITY_DB_PATH
        elif scope == "project":
            db_path = get_project_db_path()
            if not db_path:
                return [TextContent(
                    type="text",
                    text="Error: project not detected."
                )]
        else:
            db_path = GLOBAL_DB_PATH

        result = compact_conversations(db_path)

        return [TextContent(
            type="text",
            text=(
                f"## Compaction Complete\n\n"
                f"- Projects processed: {result['groups']}\n"
                f"- Groups compacted: {result['compacted']}\n"
                f"- Memories deleted: {result['deleted']}\n"
                f"- Consolidated created: {result['created']}\n"
            )
        )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def periodic_reindex():
    """Verifica periodicamente e enfileira memórias sem embedding"""
    import time
    while True:
        time.sleep(PERIODIC_REINDEX_INTERVAL)
        for db_path in [GLOBAL_DB_PATH, PERSONALITY_DB_PATH]:
            try:
                count = reindex_pending(db_path)
                if count > 0:
                    print(
                        f"[Memory] Periodic reindex: "
                        f"queued {count} from {db_path.name}",
                        file=sys.stderr
                    )
            except Exception:
                pass


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Inicia servidor MCP"""
    print("[Memory] ============================================",
          file=sys.stderr)
    print("[Memory] MCP Memory Server v2 Starting (Enhanced)",
          file=sys.stderr)
    print(f"[Memory] Global DB: {GLOBAL_DB_PATH}", file=sys.stderr)
    print(f"[Memory] Personality DB: {PERSONALITY_DB_PATH}", file=sys.stderr)
    print(
        f"[Memory] Embeddings: "
        f"{'enabled' if EMBEDDING_ENABLED else 'disabled'}",
        file=sys.stderr
    )
    print(f"[Memory] Model: {EMBEDDING_MODEL}", file=sys.stderr)
    print(
        f"[Memory] Search: hybrid "
        f"(vector={VECTOR_WEIGHT}, text={TEXT_WEIGHT})",
        file=sys.stderr
    )
    print(f"[Memory] Dedup: Jaccard threshold={DEDUP_THRESHOLD}",
          file=sys.stderr)
    print(
        f"[Memory] Chunking: {CHUNK_SIZE_WORDS}w chunks, "
        f"{CHUNK_OVERLAP_WORDS}w overlap",
        file=sys.stderr
    )
    print("[Memory] ============================================",
          file=sys.stderr)

    # Inicializa databases
    init_db(GLOBAL_DB_PATH)
    init_db(PERSONALITY_DB_PATH)

    # Inicia embedding worker
    start_embedding_worker()

    # Pré-carrega modelo de embedding (background)
    if EMBEDDING_ENABLED:
        threading.Thread(target=get_embedding_model, daemon=True).start()

    # Auto-reindex na inicialização + reindex periódico
    if EMBEDDING_ENABLED:
        def startup_reindex():
            import time
            time.sleep(3)  # Espera modelo carregar
            for label, db_path in [
                ("global", GLOBAL_DB_PATH),
                ("personality", PERSONALITY_DB_PATH)
            ]:
                count = reindex_pending(db_path)
                if count > 0:
                    print(
                        f"[Memory] Startup reindex: "
                        f"queued {count} from {label}",
                        file=sys.stderr
                    )

        threading.Thread(target=startup_reindex, daemon=True).start()

        # Reindex periódico a cada 5 minutos
        threading.Thread(target=periodic_reindex, daemon=True).start()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
