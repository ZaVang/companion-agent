"""
PostgreSQL 持久化层 for Memory System

提供 NeuronCell 和 Engram 的 PostgreSQL 存储/读取能力。
支持 JSON 序列化、连接序列化、以及表结构自动初始化。
"""

import json
import uuid as uuid_lib
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from memory.neuron import NeuronCell, Connection
from memory.engram import Engram
from utils.common import DEFAULT_AREA


# ── Table creation SQL ──────────────────────────────────────────────────────

INIT_SQL = """
-- Neurons table
CREATE TABLE IF NOT EXISTS neurons (
    event_id   TEXT PRIMARY KEY,   -- UUID string (str(neuron.event_id))
    event_type TEXT NOT NULL,
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    strength    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    decay_rate  DOUBLE PRECISION NOT NULL DEFAULT 0.995,
    impact_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    last_decay_at TIMESTAMP WITH TIME ZONE,
    activation_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.3,
    is_consolidated BOOLEAN NOT NULL DEFAULT FALSE,
    emotional_valence  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    emotional_arousal  DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    emotional_dominance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    actor       TEXT NOT NULL,
    audience    JSONB,               -- list as JSON array
    outgoing_connections JSONB,       -- list of {target_id, create_time}
    incoming_connections JSONB,
    -- Sprint 5/6/7 metadata (stored as JSON string on neuron instance, not Pydantic field)
    _scene_json      JSONB,
    _emotion_json    JSONB,
    _resonance_json  JSONB
);

-- Engrams table
CREATE TABLE IF NOT EXISTS engrams (
    uuid       TEXT PRIMARY KEY,   -- str(engram.uuid)
    scope      TEXT NOT NULL DEFAULT 'partial',
    represent  TEXT NOT NULL,       -- UUID string
    strength   DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    time       TIMESTAMP WITH TIME ZONE NOT NULL,
    summary    TEXT NOT NULL DEFAULT '',
    actor      JSONB NOT NULL,      -- list as JSON array
    audience   JSONB,
    engram_json JSONB NOT NULL       -- full engram dict serialized
);

-- Embeddings table (optional, stores numpy vectors as bytes)
CREATE TABLE IF NOT EXISTS embeddings (
    event_id   TEXT PRIMARY KEY REFERENCES neurons(event_id) ON DELETE CASCADE,
    embedding  BYTEA NOT NULL       -- numpy array as bytes
);
"""


# ── Serialisation helpers ────────────────────────────────────────────────────

def _serialize_connection(conn: Connection) -> Dict[str, Any]:
    return {
        "target_id": str(conn.target_id),
        "create_time": conn.create_time.isoformat() if isinstance(conn.create_time, datetime) else str(conn.create_time),
    }


def _deserialize_connection(data: Dict[str, Any]) -> Connection:
    ts = data["create_time"]
    if isinstance(ts, str):
        naive = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if naive.tzinfo is None:
            naive = naive.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        ts = naive
    return Connection(target_id=uuid_lib.UUID(data["target_id"]), create_time=ts)


def _neuron_to_row(neuron: NeuronCell) -> Dict[str, Any]:
    return {
        "event_id": str(neuron.event_id),
        "event_type": neuron.event_type,
        "create_time": neuron.create_time,
        "strength": neuron.strength,
        "decay_rate": neuron.decay_rate,
        "impact_score": neuron.impact_score,
        "last_decay_at": neuron.last_decay_at,
        "activation_threshold": neuron.activation_threshold,
        "is_consolidated": neuron.is_consolidated,
        "emotional_valence": neuron.emotional_valence,
        "emotional_arousal": neuron.emotional_arousal,
        "emotional_dominance": neuron.emotional_dominance,
        "actor": neuron.actor,
        "audience": json.dumps(neuron.audience) if neuron.audience else None,
        "outgoing_connections": json.dumps([_serialize_connection(c) for c in neuron.outgoing_connections]),
        "incoming_connections": json.dumps([_serialize_connection(c) for c in neuron.incoming_connections]),
        "_scene_json": json.dumps(neuron._scene) if hasattr(neuron, "_scene") and neuron._scene else None,
        "_emotion_json": json.dumps(neuron._emotion) if hasattr(neuron, "_emotion") and neuron._emotion else None,
        "_resonance_json": json.dumps(neuron._resonance) if hasattr(neuron, "_resonance") and neuron._resonance else None,
    }


def _row_to_neuron(row: Dict[str, Any]) -> NeuronCell:
    audience = json.loads(row["audience"]) if row["audience"] else None
    outgoing = [_deserialize_connection(c) for c in (json.loads(row["outgoing_connections"]) or [])]
    incoming = [_deserialize_connection(c) for c in (json.loads(row["incoming_connections"]) or [])]

    kw = {
        "event_id": uuid_lib.UUID(row["event_id"]),
        "event_type": row["event_type"],
        "create_time": row["create_time"],
        "strength": row["strength"],
        "decay_rate": row["decay_rate"],
        "impact_score": row["impact_score"],
        "last_decay_at": row["last_decay_at"],
        "activation_threshold": row["activation_threshold"],
        "is_consolidated": row["is_consolidated"],
        "emotional_valence": row["emotional_valence"],
        "emotional_arousal": row["emotional_arousal"],
        "emotional_dominance": row["emotional_dominance"],
        "actor": row["actor"],
        "audience": audience,
        "outgoing_connections": outgoing,
        "incoming_connections": incoming,
    }
    neuron = NeuronCell(**kw)
    # Restore private metadata
    if row.get("_scene_json"):
        neuron._scene = json.loads(row["_scene_json"])
    if row.get("_emotion_json"):
        neuron._emotion = json.loads(row["_emotion_json"])
    if row.get("_resonance_json"):
        neuron._resonance = json.loads(row["_resonance_json"])
    return neuron


def _engram_to_row(engram: Engram) -> Dict[str, Any]:
    return {
        "uuid": str(engram.uuid),
        "scope": engram.scope,
        "represent": str(engram.represent),
        "strength": engram.strength,
        "time": engram.time,
        "summary": engram.summary,
        "actor": json.dumps(engram.actor),
        "audience": json.dumps(engram.audience) if engram.audience else None,
        "engram_json": json.dumps(engram.engram),
    }


def _row_to_engram(row: Dict[str, Any]) -> Engram:
    # Reconstruct engram dict with NeuronCell objects
    engram_raw = json.loads(row["engram_json"])
    for event_type, neurons in engram_raw.items():
        reconstructed = []
        for n_data in neurons:
            n_data["event_id"] = uuid_lib.UUID(n_data["event_id"])
            n_data["create_time"] = datetime.fromisoformat(
                n_data["create_time"].replace("Z", "+00:00")
            )
            if n_data.get("last_decay_at"):
                n_data["last_decay_at"] = datetime.fromisoformat(
                    n_data["last_decay_at"].replace("Z", "+00:00")
                )
            outgoing = [_deserialize_connection(c) for c in (n_data.pop("outgoing_connections", []) or [])]
            incoming = [_deserialize_connection(c) for c in (n_data.pop("incoming_connections", []) or [])]
            n_data["outgoing_connections"] = outgoing
            n_data["incoming_connections"] = incoming
            try:
                reconstructed.append(NeuronCell(**n_data))
            except Exception:
                # Skip malformed neurons
                pass
        engram_raw[event_type] = reconstructed

    kw = {
        "uuid": uuid_lib.UUID(row["uuid"]),
        "scope": row["scope"],
        "represent": uuid_lib.UUID(row["represent"]),
        "strength": row["strength"],
        "time": row["time"],
        "summary": row["summary"],
        "actor": json.loads(row["actor"]),
        "audience": json.loads(row["audience"]) if row["audience"] else None,
        "engram": engram_raw,
    }
    return Engram(**kw)


# ── PostgresStorage ─────────────────────────────────────────────────────────

class PostgresStorage:
    """
    PostgreSQL-backed storage for NeuronCell and Engram objects.

    Parameters
    ----------
    dsn : str
        PostgreSQL connection string, e.g. ``"postgresql://user:pass@localhost/db"``
        or ``"postgresql:///dbname"`` for unix socket.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn = None
        self._init_db()

    # ── Connection management ────────────────────────────────────────────────

    def _get_conn(self):
        """Lazily acquire a connection, reconnecting if needed."""
        try:
            import psycopg2
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self._dsn)
                self._conn.autocommit = True
            return self._conn
        except ImportError:
            raise ImportError(
                "PostgresStorage requires psycopg2. Install with: pip install psycopg2-binary"
            )

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(INIT_SQL)
        cur.close()

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # ── Neurons ─────────────────────────────────────────────────────────────

    def save_neuron(self, neuron: NeuronCell) -> None:
        """Insert or upsert a NeuronCell."""
        row = _neuron_to_row(neuron)
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO neurons (
                event_id, event_type, create_time, strength, decay_rate,
                impact_score, last_decay_at, activation_threshold,
                is_consolidated, emotional_valence, emotional_arousal,
                emotional_dominance, actor, audience,
                outgoing_connections, incoming_connections,
                _scene_json, _emotion_json, _resonance_json
            ) VALUES (
                %(event_id)s, %(event_type)s, %(create_time)s, %(strength)s,
                %(decay_rate)s, %(impact_score)s, %(last_decay_at)s,
                %(activation_threshold)s, %(is_consolidated)s,
                %(emotional_valence)s, %(emotional_arousal)s,
                %(emotional_dominance)s, %(actor)s, %(audience)s,
                %(outgoing_connections)s, %(incoming_connections)s,
                %(_scene_json)s, %(_emotion_json)s, %(_resonance_json)s
            )
            ON CONFLICT (event_id) DO UPDATE SET
                strength            = EXCLUDED.strength,
                decay_rate         = EXCLUDED.decay_rate,
                impact_score       = EXCLUDED.impact_score,
                last_decay_at      = EXCLUDED.last_decay_at,
                activation_threshold = EXCLUDED.activation_threshold,
                is_consolidated    = EXCLUDED.is_consolidated,
                emotional_valence = EXCLUDED.emotional_valence,
                emotional_arousal  = EXCLUDED.emotional_arousal,
                emotional_dominance = EXCLUDED.emotional_dominance,
                audience           = EXCLUDED.audience,
                outgoing_connections = EXCLUDED.outgoing_connections,
                incoming_connections = EXCLUDED.incoming_connections,
                _scene_json        = EXCLUDED._scene_json,
                _emotion_json      = EXCLUDED._emotion_json,
                _resonance_json    = EXCLUDED._resonance_json
            """,
            row,
        )
        cur.close()

    def load_neuron(self, event_id: str) -> Optional[NeuronCell]:
        """Load a NeuronCell by event_id string. Returns None if not found."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM neurons WHERE event_id = %s", (event_id,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        # Row is (event_id, event_type, create_time, strength, ...)
        col_names = [
            "event_id", "event_type", "create_time", "strength", "decay_rate",
            "impact_score", "last_decay_at", "activation_threshold",
            "is_consolidated", "emotional_valence", "emotional_arousal",
            "emotional_dominance", "actor", "audience",
            "outgoing_connections", "incoming_connections",
            "_scene_json", "_emotion_json", "_resonance_json",
        ]
        row_dict = dict(zip(col_names, row))
        return _row_to_neuron(row_dict)

    def delete_neuron(self, event_id: str) -> bool:
        """Delete a neuron. Returns True if it existed."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM neurons WHERE event_id = %s", (event_id,))
        deleted = cur.rowcount > 0
        cur.close()
        return deleted

    def list_neurons(self, limit: int = 1000) -> List[NeuronCell]:
        """List up to `limit` neurons ordered by create_time desc."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM neurons ORDER BY create_time DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
        cur.close()
        col_names = [
            "event_id", "event_type", "create_time", "strength", "decay_rate",
            "impact_score", "last_decay_at", "activation_threshold",
            "is_consolidated", "emotional_valence", "emotional_arousal",
            "emotional_dominance", "actor", "audience",
            "outgoing_connections", "incoming_connections",
            "_scene_json", "_emotion_json", "_resonance_json",
        ]
        return [_row_to_neuron(dict(zip(col_names, row))) for row in rows]

    # ── Engrams ─────────────────────────────────────────────────────────────

    def save_engram(self, engram: Engram) -> None:
        """Insert or upsert an Engram."""
        row = _engram_to_row(engram)
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO engrams (uuid, scope, represent, strength, time, summary, actor, audience, engram_json)
            VALUES (%(uuid)s, %(scope)s, %(represent)s, %(strength)s, %(time)s, %(summary)s, %(actor)s, %(audience)s, %(engram_json)s)
            ON CONFLICT (uuid) DO UPDATE SET
                scope      = EXCLUDED.scope,
                represent  = EXCLUDED.represent,
                strength   = EXCLUDED.strength,
                time       = EXCLUDED.time,
                summary    = EXCLUDED.summary,
                actor      = EXCLUDED.actor,
                audience   = EXCLUDED.audience,
                engram_json = EXCLUDED.engram_json
            """,
            row,
        )
        cur.close()

    def load_engram(self, uuid: str) -> Optional[Engram]:
        """Load an Engram by UUID string. Returns None if not found."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM engrams WHERE uuid = %s", (uuid,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        col_names = ["uuid", "scope", "represent", "strength", "time", "summary", "actor", "audience", "engram_json"]
        row_dict = dict(zip(col_names, row))
        return _row_to_engram(row_dict)

    def load_engrams(self) -> List[Engram]:
        """Load all Engrams ordered by time desc."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM engrams ORDER BY time DESC")
        rows = cur.fetchall()
        cur.close()
        col_names = ["uuid", "scope", "represent", "strength", "time", "summary", "actor", "audience", "engram_json"]
        return [_row_to_engram(dict(zip(col_names, row))) for row in rows]

    def delete_engram(self, uuid: str) -> bool:
        """Delete an engram. Returns True if it existed."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM engrams WHERE uuid = %s", (uuid,))
        deleted = cur.rowcount > 0
        cur.close()
        return deleted

    # ── Embeddings ───────────────────────────────────────────────────────────

    def save_embedding(self, event_id: str, embedding: np.ndarray) -> None:
        """Store a numpy embedding vector alongside its neuron."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO embeddings (event_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (event_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (event_id, embedding.astype(np.float32).tobytes()),
        )
        cur.close()

    def load_embedding(self, event_id: str) -> Optional[np.ndarray]:
        """Load a stored embedding vector. Returns None if not found."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM embeddings WHERE event_id = %s", (event_id,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)
