from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from api.models import utc_now

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "would_we_have_met_alpha.db"


def get_db_path() -> Path:
    configured = os.getenv("WWHM_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DB_PATH


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS mood_checkins (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                mood TEXT NOT NULL,
                social_battery TEXT NOT NULL,
                intent TEXT NOT NULL,
                time_horizon TEXT NOT NULL,
                is_open INTEGER NOT NULL,
                online_only INTEGER NOT NULL,
                neighborhood_radius_km REAL,
                interests TEXT NOT NULL DEFAULT '[]',
                conversation_style TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mood_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                mood TEXT NOT NULL,
                intent TEXT NOT NULL,
                social_battery TEXT NOT NULL,
                time_horizon TEXT NOT NULL,
                is_open INTEGER NOT NULL,
                online_only INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES mood_checkins(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS mediator_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_one TEXT NOT NULL,
                participant_two TEXT NOT NULL,
                initiated_by TEXT NOT NULL,
                stage INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                direct_ready INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(participant_one, participant_two),
                FOREIGN KEY (participant_one) REFERENCES mood_checkins(user_id),
                FOREIGN KEY (participant_two) REFERENCES mood_checkins(user_id)
            );

            CREATE TABLE IF NOT EXISTS mediator_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                speaker_role TEXT NOT NULL,
                speaker_user_id TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES mediator_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (speaker_user_id) REFERENCES mood_checkins(user_id)
            );
            """
        )


def save_checkin(payload: dict) -> None:
    existing_checkin = get_checkin(payload["user_id"])

    with get_connection() as connection:
        if existing_checkin is not None:
            connection.execute(
                """
                INSERT INTO mood_history (
                    user_id,
                    mood,
                    intent,
                    social_battery,
                    time_horizon,
                    is_open,
                    online_only,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    existing_checkin["user_id"],
                    existing_checkin["mood"],
                    existing_checkin["intent"],
                    existing_checkin["social_battery"],
                    existing_checkin["time_horizon"],
                    int(existing_checkin["is_open"]),
                    int(existing_checkin["online_only"]),
                    existing_checkin["timestamp"],
                ),
            )

        connection.execute(
            """
            INSERT INTO mood_checkins (
                user_id,
                display_name,
                mood,
                social_battery,
                intent,
                time_horizon,
                is_open,
                online_only,
                neighborhood_radius_km,
                interests,
                conversation_style,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                display_name = excluded.display_name,
                mood = excluded.mood,
                social_battery = excluded.social_battery,
                intent = excluded.intent,
                time_horizon = excluded.time_horizon,
                is_open = excluded.is_open,
                online_only = excluded.online_only,
                neighborhood_radius_km = excluded.neighborhood_radius_km,
                interests = excluded.interests,
                conversation_style = excluded.conversation_style,
                timestamp = excluded.timestamp
            """,
            (
                payload["user_id"],
                payload["display_name"],
                payload["mood"],
                payload["social_battery"],
                payload["intent"],
                payload["time_horizon"],
                int(payload["is_open"]),
                int(payload["online_only"]),
                payload["neighborhood_radius_km"],
                json.dumps(payload["interests"]),
                payload["conversation_style"],
                payload["timestamp"],
            ),
        )


def get_checkin(user_id: str) -> Optional[dict]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM mood_checkins WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return _decode_checkin_row(row)


def get_mood_history(user_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM mood_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            """,
            (user_id,),
        ).fetchall()

    return [_decode_mood_history_row(row) for row in rows]


def get_mood_pattern(user_id: str) -> dict:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT mood, COUNT(*) AS mood_count
            FROM mood_history
            WHERE user_id = ?
            GROUP BY mood
            ORDER BY mood ASC
            """,
            (user_id,),
        ).fetchall()

    return {row["mood"]: row["mood_count"] for row in rows}


def list_open_checkins(exclude_user_id: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM mood_checkins WHERE is_open = 1"
    params: tuple = ()

    if exclude_user_id:
        query += " AND user_id != ?"
        params = (exclude_user_id,)

    query += " ORDER BY timestamp DESC"

    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    return [_decode_checkin_row(row) for row in rows if row is not None]


def get_or_create_session(user_a: str, user_b: str, initiated_by: str) -> tuple[dict, bool]:
    participant_one, participant_two = sorted((user_a, user_b))

    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT * FROM mediator_sessions
            WHERE participant_one = ? AND participant_two = ?
            """,
            (participant_one, participant_two),
        ).fetchone()

        if existing is not None:
            return dict(existing), False

        now = utc_now().isoformat()
        cursor = connection.execute(
            """
            INSERT INTO mediator_sessions (
                participant_one,
                participant_two,
                initiated_by,
                stage,
                status,
                direct_ready,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, 1, 'active', 0, ?, ?)
            """,
            (participant_one, participant_two, initiated_by, now, now),
        )
        created = connection.execute(
            "SELECT * FROM mediator_sessions WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    return dict(created), True


def list_sessions_for_user(user_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM mediator_sessions
            WHERE participant_one = ? OR participant_two = ?
            ORDER BY updated_at DESC
            """,
            (user_id, user_id),
        ).fetchall()

    return [dict(row) for row in rows]


def get_session(session_id: int) -> Optional[dict]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM mediator_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def update_session_state(session_id: int, stage: int, status: str, direct_ready: bool) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE mediator_sessions
            SET stage = ?, status = ?, direct_ready = ?, updated_at = ?
            WHERE id = ?
            """,
            (stage, status, int(direct_ready), utc_now().isoformat(), session_id),
        )


def add_session_message(
    session_id: int,
    speaker_role: str,
    message: str,
    speaker_user_id: Optional[str] = None,
) -> dict:
    created_at = utc_now().isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO mediator_messages (
                session_id,
                speaker_role,
                speaker_user_id,
                message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, speaker_role, speaker_user_id, message, created_at),
        )
        connection.execute(
            "UPDATE mediator_sessions SET updated_at = ? WHERE id = ?",
            (created_at, session_id),
        )
        row = connection.execute(
            "SELECT * FROM mediator_messages WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    return dict(row)


def list_session_messages(session_id: int) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM mediator_messages
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def _decode_checkin_row(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None

    return {
        "user_id": row["user_id"],
        "display_name": row["display_name"],
        "mood": row["mood"],
        "social_battery": row["social_battery"],
        "intent": row["intent"],
        "time_horizon": row["time_horizon"],
        "is_open": bool(row["is_open"]),
        "online_only": bool(row["online_only"]),
        "neighborhood_radius_km": row["neighborhood_radius_km"],
        "interests": json.loads(row["interests"] or "[]"),
        "conversation_style": row["conversation_style"],
        "timestamp": row["timestamp"],
    }


def _decode_mood_history_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "mood": row["mood"],
        "intent": row["intent"],
        "social_battery": row["social_battery"],
        "time_horizon": row["time_horizon"],
        "is_open": bool(row["is_open"]),
        "online_only": bool(row["online_only"]),
        "timestamp": row["timestamp"],
    }
