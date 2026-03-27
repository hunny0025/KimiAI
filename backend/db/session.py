"""
db/session.py – Database Session Management
SkillGenome X

Provides:
  - engine   : SQLAlchemy engine, configured from DATABASE_URL env var
  - SessionLocal : session factory used throughout the app
  - get_db()    : Flask/context-safe session dependency

Configure via .env:
  DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/skillgenome
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from db.models import Base

# ── Connection URL ──────────────────────────────────────────────────────────
# Falls back to a local default so the app can boot without a full .env.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://skillgenome:skillgenome@localhost:5432/skillgenome"
)

# ── Engine ───────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # drops stale connections from the pool
    pool_size=10,             # keep 10 persistent connections
    max_overflow=20,          # allow 20 extra on burst
    echo=False,               # set True to log all SQL (useful for debugging)
    connect_args={
        "connect_timeout": 10,
        "options": "-c application_name=skillgenome_x"
    } if "postgresql" in DATABASE_URL else {},
)

# ── Session Factory ──────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # keep objects usable after commit (important for APIs)
)


# ── Context-Manager Helper ───────────────────────────────────────────────────
@contextmanager
def get_db() -> Session:
    """
    Yields a SQLAlchemy Session and guarantees rollback on error + close on exit.

    Usage (in any Flask route):
        with get_db() as db:
            profiles = db.query(TalentProfile).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Create all tables if they don't already exist.
    Safe to call on every startup: CREATE TABLE IF NOT EXISTS semantics.
    """
    Base.metadata.create_all(bind=engine)
    print("[db] Tables created / verified.")


def health_check() -> bool:
    """
    Returns True if the database is reachable.
    Used by the /api/health endpoint.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"[db] Health check failed: {exc}")
        return False
