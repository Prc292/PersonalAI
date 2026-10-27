
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "memory.db"

CATEGORIES = {
    "general",
    "preferences",
    "personal",
    "projects",
    "people",
    "setup",
    "decisions",
    "lessons",
    "temporary",
}


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(memories)")
        }

        if "category" not in columns:
            conn.execute("""
                ALTER TABLE memories
                ADD COLUMN category TEXT NOT NULL DEFAULT 'general'
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL CHECK (mode IN ('local', 'cloud')),
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                session_id INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL CHECK (mode IN ('local', 'cloud')),
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        chat_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)")
        }
        if "session_id" not in chat_columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN session_id INTEGER")

        # Preserve pre-session histories by moving each mode's old transcript
        # into its own session exactly once.
        legacy_modes = conn.execute("""
            SELECT mode, MIN(created_at) AS created_at, MAX(created_at) AS updated_at
            FROM chat_messages
            WHERE session_id IS NULL
            GROUP BY mode
        """).fetchall()
        for legacy in legacy_modes:
            first_user = conn.execute("""
                SELECT content FROM chat_messages
                WHERE mode = ? AND session_id IS NULL AND role = 'user'
                ORDER BY id LIMIT 1
            """, (legacy["mode"],)).fetchone()
            title = _chat_title(first_user["content"] if first_user else "")
            cursor = conn.execute("""
                INSERT INTO chat_sessions (mode, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (
                legacy["mode"],
                title,
                legacy["created_at"],
                legacy["updated_at"],
            ))
            conn.execute(
                "UPDATE chat_messages SET session_id = ? WHERE mode = ? AND session_id IS NULL",
                (cursor.lastrowid, legacy["mode"]),
            )


def validate_chat_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in {"local", "cloud"}:
        raise ValueError("Chat mode must be local or cloud.")
    return mode


def list_chat_history(mode: str, limit: int = 40):
    mode = validate_chat_mode(mode)
    limit = max(1, min(int(limit), 100))

    with connect() as conn:
        session = conn.execute("""
            SELECT id FROM chat_sessions WHERE mode = ?
            ORDER BY updated_at DESC, id DESC LIMIT 1
        """, (mode,)).fetchone()
        if not session:
            return []
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session["id"], limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def replace_chat_history(mode: str, messages: list[dict]) -> int:
    mode = validate_chat_mode(mode)
    with connect() as conn:
        session = conn.execute("""
            SELECT id FROM chat_sessions WHERE mode = ?
            ORDER BY updated_at DESC, id DESC LIMIT 1
        """, (mode,)).fetchone()
    session_id = (
        session["id"] if session else create_chat_session(mode)
    )
    return replace_session_history(session_id, messages)


def _chat_title(first_message: str) -> str:
    title = " ".join(first_message.split())[:72]
    return title or "New chat"


def list_chat_sessions():
    with connect() as conn:
        rows = conn.execute("""
            SELECT id, mode, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC, id DESC
        """).fetchall()
    return [dict(row) for row in rows]


def get_chat_session(session_id: int):
    with connect() as conn:
        row = conn.execute("""
            SELECT id, mode, title, created_at, updated_at
            FROM chat_sessions WHERE id = ?
        """, (session_id,)).fetchone()
    return dict(row) if row else None


def create_chat_session(mode: str, title: str = "New chat") -> int:
    mode = validate_chat_mode(mode)
    title = _chat_title(title)
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        cursor = conn.execute("""
            INSERT INTO chat_sessions (mode, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (mode, title, now, now))
        return cursor.lastrowid


def list_session_history(session_id: int, limit: int = 40):
    limit = max(1, min(int(limit), 100))
    with connect() as conn:
        rows = conn.execute("""
            SELECT role, content FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit)).fetchall()
    return [dict(row) for row in reversed(rows)]


def replace_session_history(session_id: int, messages: list[dict]) -> int:
    cleaned = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            cleaned.append((role, content[:8000]))

    cleaned = cleaned[-40:]
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        session = conn.execute(
            "SELECT mode FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise ValueError("Chat not found.")
        mode = session["mode"]
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.executemany(
            """
            INSERT INTO chat_messages (mode, role, content, created_at, session_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(mode, role, content, now, session_id) for role, content in cleaned],
        )
        first_user = next(
            (content for role, content in cleaned if role == "user"), ""
        )
        conn.execute("""
            UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?
        """, (_chat_title(first_user), now, session_id))
    return len(cleaned)


def delete_chat_session(session_id: int) -> bool:
    with connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0


def validate_fact(fact: str) -> str:
    fact = fact.strip()
    if not fact:
        raise ValueError("Memory cannot be empty.")
    if len(fact) > 1000:
        raise ValueError("Memory must be 1000 characters or fewer.")
    return fact


def validate_category(category: str) -> str:
    category = category.strip().lower()
    if category not in CATEGORIES:
        allowed = ", ".join(sorted(CATEGORIES))
        raise ValueError(f"Choose a category from: {allowed}")
    return category


def add_memory(fact: str, category: str = "general") -> int:
    fact = validate_fact(fact)
    category = validate_category(category)

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO memories (fact, category, created_at)
            VALUES (?, ?, ?)
            """,
            (fact, category, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.lastrowid


def list_memories():
    with connect() as conn:
        rows = conn.execute("""
            SELECT id, fact, category, created_at
            FROM memories
            ORDER BY id
        """).fetchall()
        return [dict(row) for row in rows]


def update_memory(memory_id: int, fact: str, category: str) -> bool:
    fact = validate_fact(fact)
    category = validate_category(category)

    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE memories
            SET fact = ?, category = ?
            WHERE id = ?
            """,
            (fact, category, memory_id),
        )
        return cursor.rowcount > 0


def delete_memory(memory_id: int) -> bool:
    with connect() as conn:
        cursor = conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,),
        )
        return cursor.rowcount > 0


def clear_memories():
    with connect() as conn:
        conn.execute("DELETE FROM memories")


def search_memories(query: str, limit: int = 8):
    """Return saved memories that best match the query's keywords."""
    stopwords = {
        "a", "an", "and", "are", "do", "does", "for", "how",
        "i", "is", "it", "me", "my", "of", "on", "or", "the",
        "this", "to", "was", "what", "when", "where", "who",
        "why", "with", "you", "your", "about", "tell", "please",
    }

    query_terms = [
        word for word in re.findall(r"[a-z0-9]+", query.casefold())
        if len(word) > 1 and word not in stopwords
    ]
    query_terms = list(dict.fromkeys(query_terms))
    query_words = set(query_terms)

    if not query_words:
        return []

    limit = max(1, min(int(limit), 20))
    memories = list_memories()
    scored = []

    for item in memories:
        fact_terms = [
            word for word in re.findall(r"[a-z0-9]+", item["fact"].casefold())
            if len(word) > 1 and word not in stopwords
        ]
        category_terms = set(
            word for word in re.findall(r"[a-z0-9]+", item["category"].casefold())
            if len(word) > 1 and word not in stopwords
        )

        score = 2 * len(query_words.intersection(fact_terms))
        score += len(query_words.intersection(category_terms))

        if len(query_terms) > 1:
            phrase_length = len(query_terms)
            phrase_matches = any(
                fact_terms[index:index + phrase_length] == query_terms
                for index in range(len(fact_terms) - phrase_length + 1)
            )
            if phrase_matches:
                score += 4

        if score:
            scored.append((score, item["id"], item))

    scored.sort(key=lambda result: (-result[0], -result[1]))
    return [item for _, _, item in scored[:limit]]


init_db()
