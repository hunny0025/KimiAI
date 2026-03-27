"""
db/seed.py – One-time CSV → PostgreSQL Data Seeder
SkillGenome X

Reads backend/data/synthetic_talent_data.csv and bulk-inserts every row
into talent_profiles + skill_scores tables.

Usage:
    python -m db.seed                              # seed from default CSV
    python -m db.seed --csv data/my_data.csv       # seed from custom CSV
    python -m db.seed --reset                      # truncate tables first, then seed
    python -m db.seed --limit 5000                 # seed only first N rows (for testing)

Dependencies: pandas (optional – used only here, not in api.py)
"""

import argparse
import json
import os
import sys

# Make backend/ importable when run as `python -m db.seed`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from sqlalchemy import text

from db.session import SessionLocal, init_db
from db.models import TalentProfile, SkillScore

# ── Default CSV path ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "synthetic_talent_data.csv")

BATCH_SIZE = 500   # rows per INSERT batch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(val, default=50):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _derive_skill_level(score: float) -> str:
    if score > 80:
        return "Expert"
    if score > 60:
        return "Advanced"
    return "Intermediate"


def _derive_migration_risk(score: float, opportunity: str) -> str:
    if score > 75 and opportunity == "Low":
        return "High"
    if score > 65 and opportunity == "Moderate":
        return "Medium"
    return "Low"


# ── Core Seeder ───────────────────────────────────────────────────────────────

def seed(csv_path: str, reset: bool = False, limit: int = None):
    print(f"[seed] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)
        print(f"[seed] Limiting to first {limit} rows.")

    total = len(df)
    print(f"[seed] {total} rows loaded.")

    # Ensure tables exist
    init_db()

    db = SessionLocal()
    try:
        if reset:
            print("[seed] --reset flag set. Truncating tables …")
            db.execute(text("TRUNCATE skill_scores, talent_profiles RESTART IDENTITY CASCADE"))
            db.commit()
            print("[seed] Tables truncated.")

        inserted_profiles = 0
        inserted_scores = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch = df.iloc[batch_start : batch_start + BATCH_SIZE]
            profile_objs = []

            for _, row in batch.iterrows():
                profile = TalentProfile(
                    # Identity
                    state=str(row.get("state", "Unknown")),
                    domain=str(row.get("domain", "General")),
                    area_type=str(row.get("area_type", "Urban")),
                    digital_access=str(row.get("digital_access", "Regular")),
                    opportunity_level=str(row.get("opportunity_level", "Moderate")),
                    # Behavioral signals
                    creation_output=_safe_int(row.get("creation_output")),
                    learning_behavior=_safe_int(row.get("learning_behavior")),
                    experience_consistency=_safe_int(row.get("experience_consistency")),
                    economic_activity=_safe_int(row.get("economic_activity")),
                    innovation_problem_solving=_safe_int(row.get("innovation_problem_solving")),
                    collaboration_community=_safe_int(row.get("collaboration_community")),
                    offline_capability=_safe_int(row.get("offline_capability")),
                    digital_presence=_safe_int(row.get("digital_presence")),
                    learning_hours=_safe_int(row.get("learning_hours"), default=20),
                    projects=_safe_int(row.get("projects"), default=3),
                    # Socio-economic
                    internet_penetration=_safe_float(row.get("internet_penetration")),
                    urban_population_percent=_safe_float(row.get("urban_population_percent")),
                    per_capita_income=_safe_float(row.get("per_capita_income")),
                    workforce_participation=_safe_float(row.get("workforce_participation")),
                    literacy_rate=_safe_float(row.get("literacy_rate")),
                    unemployment_rate=_safe_float(row.get("unemployment_rate")),
                    # Temporal
                    skill_history=str(row["skill_history"]) if "skill_history" in row and pd.notna(row.get("skill_history")) else None,
                    created_at=datetime.utcnow(),
                )
                profile_objs.append(profile)

            db.add_all(profile_objs)
            db.flush()   # assigns primary keys without committing

            # Build SkillScore records linked to the flushed profiles
            score_objs = []
            for profile, (_, row) in zip(profile_objs, batch.iterrows()):
                raw_score = _safe_float(row.get("skill_score"), default=50.0)
                score = max(0.0, min(100.0, raw_score))
                opportunity = str(row.get("opportunity_level", "Moderate"))

                is_hidden = (
                    score > 70
                    and str(row.get("area_type", "")) == "Rural"
                    or opportunity == "Low"
                )

                ss = SkillScore(
                    talent_profile_id=profile.id,
                    score=round(score, 1),
                    level=_derive_skill_level(score),
                    confidence=85.0,
                    is_anomaly=False,
                    is_hidden_talent=bool(is_hidden),
                    migration_risk=_derive_migration_risk(score, opportunity),
                    work_capacity="High" if score > 75 else "Moderate" if score > 45 else "Low",
                    growth_potential="High"  if _safe_int(row.get("learning_behavior")) > 60
                                    else "Moderate" if _safe_int(row.get("learning_behavior")) > 30
                                    else "Low",
                    risk_level="Low" if score > 70 else "Moderate" if score > 40 else "High",
                    top_positive_factors=json.dumps([]),
                    top_negative_factors=json.dumps([]),
                    model_version="GradientBoostingRegressor (seed)",
                    computed_at=datetime.utcnow(),
                )
                score_objs.append(ss)

            db.add_all(score_objs)
            db.commit()

            inserted_profiles += len(profile_objs)
            inserted_scores   += len(score_objs)
            print(
                f"[seed] Batch {batch_start}–{batch_start + len(batch) - 1}: "
                f"{len(profile_objs)} profiles, {len(score_objs)} scores inserted "
                f"(total: {inserted_profiles})"
            )

    except Exception as exc:
        db.rollback()
        print(f"[seed] ERROR: {exc}")
        raise
    finally:
        db.close()

    print(
        f"\n[seed] ✅ Done. "
        f"{inserted_profiles} TalentProfile rows, "
        f"{inserted_scores} SkillScore rows inserted."
    )


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SkillGenome X database from CSV")
    parser.add_argument("--csv",   default=DEFAULT_CSV, help="Path to input CSV file")
    parser.add_argument("--reset", action="store_true",  help="Truncate tables before seeding")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to insert")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[seed] CSV file not found: {args.csv}")
        sys.exit(1)

    seed(csv_path=args.csv, reset=args.reset, limit=args.limit)
