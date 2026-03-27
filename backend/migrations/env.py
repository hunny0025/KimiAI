"""
migrations/env.py – Alembic Environment
SkillGenome X

Reads DATABASE_URL from the environment (or falls back to alembic.ini),
imports the SQLAlchemy metadata from db.models, and wires up both
online (live DB) and offline (SQL script) migration modes.
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Make the backend/ package importable ─────────────────────────────────────
# Alembic runs from within backend/, so one level up is enough.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import ORM Metadata ──────────────────────────────────────────────────────
from db.models import Base   # noqa: E402  (import after sys.path patch)

# ── Alembic Config Object ─────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url from environment variable if set
db_url = os.environ.get("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for 'autogenerate' support
target_metadata = Base.metadata


# ── Offline Mode ─────────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection — emits SQL to stdout/file.
    Useful for generating migration scripts to review before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online Mode ───────────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    """
    Run migrations against a live DB connection.
    Uses NullPool in migration context to avoid connection-pooling side-effects.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,        # detect column type changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
