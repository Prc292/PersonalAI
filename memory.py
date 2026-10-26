
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

    words = {
        word for word in re.findall(r"[a-z0-9]+", query.lower())
        if len(word) > 1 and word not in stopwords
    }

    if not words:
        return []

    limit = max(1, min(int(limit), 20))
    memories = list_memories()
    scored = []

    for item in memories:
        fact = item["fact"].lower()
        category = item["category"].lower()
        score = sum(
            (2 if word in fact else 0)
            + (1 if word in category else 0)
            for word in words
        )

        if score:
            scored.append((score, item["id"], item))

    scored.sort(key=lambda result: (-result[0], -result[1]))
    return [item for _, _, item in scored[:limit]]


init_db()
