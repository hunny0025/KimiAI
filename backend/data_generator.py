"""
data_generator.py – PLFS-Calibrated Synthetic Population Generator
SkillGenome X

Generates 50,000 synthetic worker profiles using REAL statistical distributions
from official government sources.

─── DATA SOURCES ───
  • State rural %, unemployment rate, skill-training %:
        PLFS Annual Report 2023-24 (MoSPI, GoI)
  • State digital access %:
        TRAI India Internet Report + NSSO 78th Round
  • Domain distribution (NIC sector mapping):
        PLFS 2023-24 Worker Distribution by NIC Codes
  • State population weights:
        Census 2011 projections (RGI) for 2024
  • skill_score calibration:
        Wheebox India Skills Report 2025 (mean employability = 42.6%)
  • Female workforce participation (rural):
        MoSPI PLFS 2023-24 (23% LFPR for rural women)
  • eShram registration baseline:
        MoLE eShram portal statistics (~30% rural coverage)
─────────────────────
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_SAMPLES = 50_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_talent_data.csv")
RANDOM_SEED = 42

# Wheebox 2025 employability benchmark
WHEEBOX_SKILL_MEAN = 42.6
WHEEBOX_SKILL_STD = 18.0

# MoSPI PLFS 2023-24 female LFPR (rural)
RURAL_FEMALE_PARTICIPATION = 0.23
URBAN_FEMALE_PARTICIPATION = 0.35

# ── Official State-Level Data (PLFS 2023-24 + MoSPI + TRAI) ───────────────────

STATE_DATA = {
    "Uttar Pradesh": {
        "rural_pct": 77.7, "unemployment_rate": 4.2,
        "skill_training_pct": 8.1, "digital_access": 31.2,
        "pop_weight": 16.5,
    },
    "Maharashtra": {
        "rural_pct": 54.8, "unemployment_rate": 3.1,
        "skill_training_pct": 14.2, "digital_access": 58.4,
        "pop_weight": 9.3,
    },
    "Bihar": {
        "rural_pct": 88.7, "unemployment_rate": 5.8,
        "skill_training_pct": 4.3, "digital_access": 22.1,
        "pop_weight": 8.6,
    },
    "Tamil Nadu": {
        "rural_pct": 51.9, "unemployment_rate": 3.6,
        "skill_training_pct": 18.7, "digital_access": 64.3,
        "pop_weight": 6.1,
    },
    "Rajasthan": {
        "rural_pct": 75.1, "unemployment_rate": 5.1,
        "skill_training_pct": 7.2, "digital_access": 29.8,
        "pop_weight": 5.7,
    },
    "Karnataka": {
        "rural_pct": 61.3, "unemployment_rate": 2.9,
        "skill_training_pct": 16.8, "digital_access": 61.7,
        "pop_weight": 5.1,
    },
    "West Bengal": {
        "rural_pct": 72.0, "unemployment_rate": 4.9,
        "skill_training_pct": 9.4, "digital_access": 38.6,
        "pop_weight": 7.5,
    },
    "Gujarat": {
        "rural_pct": 57.4, "unemployment_rate": 1.4,
        "skill_training_pct": 12.9, "digital_access": 55.1,
        "pop_weight": 5.0,
    },
    "Madhya Pradesh": {
        "rural_pct": 72.4, "unemployment_rate": 3.7,
        "skill_training_pct": 6.1, "digital_access": 27.4,
        "pop_weight": 6.1,
    },
    "Odisha": {
        "rural_pct": 83.3, "unemployment_rate": 4.4,
        "skill_training_pct": 7.8, "digital_access": 24.9,
        "pop_weight": 3.5,
    },
}

# ── Domain Distribution (NIC Sector Mapping, PLFS National Average) ────────────

DOMAIN_DIST = {
    "Agriculture":    0.425,
    "Manufacturing":  0.121,
    "Trade":          0.113,
    "Construction":   0.128,
    "Technology":     0.042,
    "Healthcare":     0.038,
    "Education":      0.056,
    "Finance":        0.077,
}

DOMAIN_WEIGHTS = {
    "Technology": 1.4, "Finance": 1.3, "Healthcare": 1.2,
    "Education": 1.1,
    "Agriculture": 0.9, "Manufacturing": 0.9,
    "Trade": 0.9, "Construction": 0.9,
}

DOMAIN_TO_LEGACY = {
    "Agriculture":   "Agriculture",
    "Manufacturing": "Skilled Trades",
    "Trade":         "Business",
    "Construction":  "Skilled Trades",
    "Technology":    "Technology",
    "Healthcare":    "Healthcare",
    "Education":     "Education",
    "Finance":       "Business",
}

# Legacy categorical bands
DIGITAL_ACCESS_BANDS = [
    (60, "Regular"),    # ≥ 60 → Regular
    (35, "Limited"),    # 35-59 → Limited
    (0,  "Occasional"), # < 35 → Occasional
]

OPP_LEVEL_BANDS = [
    (55, "High"),
    (30, "Moderate"),
    (0,  "Low"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clip(val: float, lo: float = 0, hi: float = 100) -> float:
    return float(np.clip(val, lo, hi))


def _categorise(value: float, bands: list) -> str:
    for threshold, label in bands:
        if value >= threshold:
            return label
    return bands[-1][1]


def _generate_time_series(rng: np.random.Generator, base_score: float, trend: str = "stable") -> str:
    history = []
    current = base_score
    for _ in range(24):
        noise = rng.uniform(-2, 2)
        if trend == "emerging":
            current -= rng.uniform(0.5, 1.5)
        elif trend == "declining":
            current += rng.uniform(0.5, 1.5)
        else:
            current -= rng.uniform(-1, 1)
        history.append(round(_clip(current + noise), 1))
    return json.dumps(list(reversed(history)))


# ── Main Generator ─────────────────────────────────────────────────────────────

def generate(n: int = NUM_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate *n* synthetic worker profiles calibrated to PLFS 2023-24
    state-level distributions + Wheebox employability benchmarks.

    Returns a DataFrame with all original columns PLUS 6 new ones:
      eshram_registered, nsqf_level, prior_experience_years,
      community_learning_score, proof_of_work_links, hidden_talent_flag
    """
    rng = np.random.default_rng(seed)

    # ── 0. Pre-compute sampling arrays ─────────────────────────────────────
    state_names = list(STATE_DATA.keys())
    state_weights = np.array([STATE_DATA[s]["pop_weight"] for s in state_names])
    state_weights /= state_weights.sum()

    domain_names = list(DOMAIN_DIST.keys())
    domain_weights_arr = np.array(list(DOMAIN_DIST.values()))
    domain_weights_arr /= domain_weights_arr.sum()

    # ── 1. Vectorised sampling ─────────────────────────────────────────────
    states = rng.choice(state_names, size=n, p=state_weights)
    domains = rng.choice(domain_names, size=n, p=domain_weights_arr)

    print(f"[data_generator] Generating {n:,} PLFS-calibrated profiles (seed={seed})...")

    rows = []
    for i in range(n):
        state = states[i]
        domain = domains[i]
        sd = STATE_DATA[state]

        # ── 2. area_type from real rural_pct ───────────────────────────────
        is_rural = rng.random() < (sd["rural_pct"] / 100.0)
        area_type = "Rural" if is_rural else "Urban"

        # ── 3. digital_access_level (0-10 scale) ──────────────────────────
        # Rural mean = 2.1/10, Urban mean = 6.4/10 (per state, with noise)
        if area_type == "Rural":
            dal_mean = (sd["digital_access"] / 100.0) * 10 * 0.65  # ~rural discount
            dal_mean = max(dal_mean, 1.5)  # floor
        else:
            dal_mean = (sd["digital_access"] / 100.0) * 10 * 1.15  # ~urban premium
            dal_mean = min(dal_mean, 9.5)  # ceiling
        digital_access_level = round(float(np.clip(rng.normal(dal_mean, 1.2), 0, 10)), 1)

        # Also keep the legacy percentage-based score for internal formulas
        digital_access_score = _clip(digital_access_level * 10)  # 0-100 scale
        digital_access_cat = _categorise(digital_access_score, DIGITAL_ACCESS_BANDS)

        # ── 4. gender (MoSPI PLFS female LFPR) ────────────────────────────
        female_rate = RURAL_FEMALE_PARTICIPATION if area_type == "Rural" else URBAN_FEMALE_PARTICIPATION
        gender = "Female" if rng.random() < female_rate else "Male"

        # ── 5. opportunity_level ───────────────────────────────────────────
        dw = DOMAIN_WEIGHTS.get(domain, 0.9)
        opp_raw = (100 - sd["unemployment_rate"] * 10) * (digital_access_score / 100) * dw
        opportunity_level_score = _clip(opp_raw)
        opportunity_level_cat = _categorise(opportunity_level_score, OPP_LEVEL_BANDS)

        # ── 6. learning_behavior ───────────────────────────────────────────
        lb_mean = digital_access_score * 0.6 + sd["skill_training_pct"] * 1.2
        learning_behavior = _clip(rng.normal(lb_mean, 12))

        # ── 7. experience_consistency ──────────────────────────────────────
        ec_mean = 55 + (1 - sd["unemployment_rate"] / 10) * 20
        experience_consistency = _clip(rng.normal(ec_mean, 15))

        # ── 8. innovation_problem_solving ──────────────────────────────────
        ip_mean = learning_behavior * 0.5 + opportunity_level_score * 0.3
        innovation_problem_solving = _clip(rng.normal(ip_mean, 10))

        # ── 9. economic_activity ───────────────────────────────────────────
        economic_activity = _clip(opportunity_level_score * 0.8 + rng.normal(0, 10))

        # ── Derived signals (preserve all legacy columns) ──────────────────
        legacy_domain = DOMAIN_TO_LEGACY.get(domain, "Business")

        creation_output = _clip(rng.normal(
            learning_behavior * 0.4 + experience_consistency * 0.3 + 15, 14
        ))
        digital_presence = _clip(rng.normal(digital_access_score * 0.9, 12))
        collaboration_community = _clip(rng.normal(50, 18))

        offline_base = 80 if domain in ("Agriculture", "Construction") else 45
        offline_capability = _clip(rng.normal(offline_base, 15))

        if domain == "Technology":
            github_repos = max(0, int(rng.poisson(8)))
            projects = max(0, int(rng.poisson(6)))
            hackathons = max(0, int(rng.poisson(2)))
        elif domain in ("Finance", "Education"):
            github_repos = max(0, int(rng.poisson(1)))
            projects = max(0, int(rng.poisson(3)))
            hackathons = 0
        else:
            github_repos = 0
            projects = max(0, int(rng.poisson(2)))
            hackathons = 0

        learning_hours = round(max(0, rng.normal(learning_behavior * 0.3, 4)), 1)
        experience_years = round(max(0, rng.normal(experience_consistency * 0.1, 2.5)), 1)

        infra_base = digital_access_score * 0.7 + (30 if area_type == "Urban" else 10)
        infrastructure_score = int(_clip(rng.normal(infra_base, 10)))

        # ── 10. skill_score (Wheebox-calibrated: mean=42.6, std=18) ────────
        # Raw composite score from feature weights
        raw_score = (
            creation_output * 0.20
            + learning_behavior * 0.20
            + innovation_problem_solving * 0.20
            + experience_consistency * 0.20
            + digital_presence * 0.10
            + offline_capability * 0.10
        )
        # Rescale: shift/scale the raw_score to target Wheebox distribution
        # We compute batch mean/std after generation, but per-row we add
        # Wheebox-calibrated noise to centre the final distribution correctly
        skill_score = _clip(rng.normal(WHEEBOX_SKILL_MEAN, WHEEBOX_SKILL_STD))
        # Blend: 60% Wheebox-calibrated draw + 40% feature-driven raw score
        # This preserves correlation with features while hitting the target mean
        skill_score = _clip(skill_score * 0.6 + raw_score * 0.4)

        # ── 11. Trend + time series ────────────────────────────────────────
        if domain in ("Technology", "Healthcare"):
            trend = "emerging"
        else:
            trend = "stable"

        skill_history = _generate_time_series(rng, skill_score, trend)

        # ── 12. Legacy hidden-talent (is_hidden_talent) ────────────────────
        is_hidden_talent = (
            area_type == "Rural"
            and skill_score > 65
            and digital_access_cat in ("Limited", "Occasional")
        )

        # ═══════════════════════════════════════════════════════════════════
        # NEW COLUMNS (requirement #6)
        # ═══════════════════════════════════════════════════════════════════

        # a) eshram_registered – 30% true for rural, 10% for urban
        eshram_registered = bool(
            rng.random() < (0.30 if area_type == "Rural" else 0.10)
        )

        # b) nsqf_level – int 1-5, correlated with skill_score
        #    Higher skill_score → higher NSQF level (quantile-based)
        nsqf_raw = skill_score / 100.0 + rng.normal(0, 0.12)
        if nsqf_raw < 0.20:
            nsqf_level = 1
        elif nsqf_raw < 0.40:
            nsqf_level = 2
        elif nsqf_raw < 0.60:
            nsqf_level = 3
        elif nsqf_raw < 0.80:
            nsqf_level = 4
        else:
            nsqf_level = 5

        # c) prior_experience_years – int 0-20
        prior_experience_years = int(np.clip(
            rng.normal(experience_years * 1.2 + 2, 3), 0, 20
        ))

        # d) community_learning_score – float 0-10
        community_learning_score = round(float(np.clip(
            rng.normal(collaboration_community / 10, 1.5), 0, 10
        )), 1)

        # e) proof_of_work_links – int 0-3, correlated with digital_access_level
        pow_mean = digital_access_level / 10.0 * 2  # 0-2 range
        proof_of_work_links = int(np.clip(rng.poisson(pow_mean), 0, 3))

        # f) hidden_talent_flag – skill_score > 65 AND digital_access_level < 4
        hidden_talent_flag = bool(skill_score > 65 and digital_access_level < 4)

        # ── Assemble row ───────────────────────────────────────────────────
        rows.append({
            # Original columns (preserved exactly)
            "domain": legacy_domain,
            "state": state,
            "area_type": area_type,
            "opportunity_level": opportunity_level_cat,
            "infrastructure_score": infrastructure_score,
            "digital_access": digital_access_cat,
            "creation_output": round(creation_output, 1),
            "github_repos": github_repos,
            "projects": projects,
            "learning_behavior": round(learning_behavior, 1),
            "learning_hours": learning_hours,
            "experience_consistency": round(experience_consistency, 1),
            "experience_years": experience_years,
            "economic_activity": round(economic_activity, 1),
            "innovation_problem_solving": round(innovation_problem_solving, 1),
            "hackathons": hackathons,
            "collaboration_community": round(collaboration_community, 1),
            "offline_capability": round(offline_capability, 1),
            "digital_presence": round(digital_presence, 1),
            "skill_score": round(skill_score, 1),
            "skill_history": skill_history,
            "is_hidden_talent": is_hidden_talent,
            # New columns
            "gender": gender,
            "digital_access_level": digital_access_level,
            "eshram_registered": eshram_registered,
            "nsqf_level": nsqf_level,
            "prior_experience_years": prior_experience_years,
            "community_learning_score": community_learning_score,
            "proof_of_work_links": proof_of_work_links,
            "hidden_talent_flag": hidden_talent_flag,
        })

        if (i + 1) % 10_000 == 0:
            print(f"[data_generator]   ... {i + 1:,} / {n:,}")

    df = pd.DataFrame(rows)

    # ── Inject 2% anomalies (adversarial signals for IsolationForest) ──────
    num_anomalies = int(n * 0.02)
    anomaly_idxs = rng.choice(n, size=num_anomalies, replace=False)
    for idx in anomaly_idxs:
        df.at[idx, "learning_hours"] = 160.0
        df.at[idx, "creation_output"] = 100.0
        df.at[idx, "innovation_problem_solving"] = 100.0
        df.at[idx, "skill_score"] = 100.0
        df.at[idx, "skill_history"] = json.dumps([100] * 24)

    print(f"[data_generator] Injected {num_anomalies} anomalies (2%)")
    return df


# ── Calibration Summary Printer ────────────────────────────────────────────────

def print_calibration_summary(df: pd.DataFrame):
    """Print a publication-grade calibration summary with mean/std of key columns."""
    n = len(df)
    print(f"\n{'═' * 72}")
    print(f"  SkillGenome X – PLFS-Calibrated Synthetic Population")
    print(f"{'═' * 72}")
    print(f"  Records : {n:,}    Seed : {RANDOM_SEED}    Generated : {datetime.now().isoformat()}")
    print(f"{'─' * 72}")

    # Key numeric columns calibration check
    KEY_COLS = [
        ("skill_score",               f"Target: mean={WHEEBOX_SKILL_MEAN}, std={WHEEBOX_SKILL_STD}"),
        ("digital_access_level",      "Target: rural~2.1, urban~6.4 (0-10)"),
        ("learning_behavior",         "PLFS skill-training calibrated"),
        ("experience_consistency",    "Unemployment-rate calibrated"),
        ("innovation_problem_solving","Composite: learning + opportunity"),
        ("economic_activity",         "Opportunity-driven"),
        ("creation_output",           "Learning + experience derived"),
        ("community_learning_score",  "Collaboration proxy (0-10)"),
        ("prior_experience_years",    "Range 0-20"),
    ]

    print(f"\n  {'Column':<30s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}  Source Note")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}  {'─' * 30}")
    for col, note in KEY_COLS:
        if col in df.columns:
            s = df[col]
            print(f"  {col:<30s} {s.mean():8.2f} {s.std():8.2f} {s.min():8.1f} {s.max():8.1f}  {note}")

    # Wheebox validation
    print(f"\n  ── Wheebox Benchmark Validation ──")
    actual_mean = df["skill_score"].mean()
    actual_std = df["skill_score"].std()
    print(f"  skill_score  target mean = {WHEEBOX_SKILL_MEAN}  →  actual mean = {actual_mean:.2f}  (Δ = {abs(actual_mean - WHEEBOX_SKILL_MEAN):.2f})")
    print(f"  skill_score  target std  = {WHEEBOX_SKILL_STD}  →  actual std  = {actual_std:.2f}  (Δ = {abs(actual_std - WHEEBOX_SKILL_STD):.2f})")

    # Digital access by area
    print(f"\n  ── Digital Access Level by Area ──")
    for area in ["Rural", "Urban"]:
        subset = df[df["area_type"] == area]["digital_access_level"]
        if len(subset) > 0:
            target = "2.1" if area == "Rural" else "6.4"
            print(f"  {area:6s}  mean = {subset.mean():.2f} / 10  (target ≈ {target})  n = {len(subset):,}")

    # Gender split
    print(f"\n  ── Gender Distribution ──")
    for area in ["Rural", "Urban"]:
        subset = df[df["area_type"] == area]
        female_pct = (subset["gender"] == "Female").mean() * 100
        target = "23%" if area == "Rural" else "35%"
        print(f"  {area:6s}  Female = {female_pct:.1f}%  (target ≈ {target})  n = {len(subset):,}")

    # State distribution
    print(f"\n  ── State Distribution (top 5) ──")
    for state, count in df["state"].value_counts().head(5).items():
        pct = count / n * 100
        target = STATE_DATA[state]["pop_weight"]
        print(f"  {state:20s} {count:6,}  ({pct:.1f}%)  [target: {target}%]")

    # Area split
    area_counts = df["area_type"].value_counts()
    rural_pct = area_counts.get("Rural", 0) / n * 100
    print(f"\n  ── Area Split ──")
    print(f"  Rural: {rural_pct:.1f}%  |  Urban: {100 - rural_pct:.1f}%  (target: ~65% / ~35%)")

    # New columns summary
    print(f"\n  ── New Columns ──")
    print(f"  eshram_registered    : {df['eshram_registered'].mean() * 100:.1f}% true")
    print(f"  nsqf_level           : mean={df['nsqf_level'].mean():.2f}  dist={df['nsqf_level'].value_counts().sort_index().to_dict()}")
    print(f"  prior_experience_yrs : mean={df['prior_experience_years'].mean():.1f}  std={df['prior_experience_years'].std():.1f}")
    print(f"  community_learn_score: mean={df['community_learning_score'].mean():.2f}  std={df['community_learning_score'].std():.2f}")
    print(f"  proof_of_work_links  : dist={df['proof_of_work_links'].value_counts().sort_index().to_dict()}")
    print(f"  hidden_talent_flag   : {df['hidden_talent_flag'].sum():,} flagged ({df['hidden_talent_flag'].mean() * 100:.1f}%)")
    print(f"  is_hidden_talent     : {df['is_hidden_talent'].sum():,} flagged ({df['is_hidden_talent'].mean() * 100:.1f}%)")

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024) if os.path.exists(OUTPUT_FILE) else 0
    print(f"\n  File: {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"{'═' * 72}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_FILE, index=False)
    print_calibration_summary(df)


if __name__ == "__main__":
    main()
