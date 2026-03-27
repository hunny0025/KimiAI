"""
db/repository.py – Data Access Layer (Repository Pattern)
SkillGenome X

All query logic lives here. api.py no longer touches Pandas or CSV files —
it calls these functions instead.  Every function accepts a SQLAlchemy
Session (injected via get_db()) and returns plain Python objects.
"""

from __future__ import annotations

import json
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from db.models import TalentProfile, SkillScore, PolicySimulation


# ════════════════════════════════════════════════════════════════════════════
# TalentProfile Queries
# ════════════════════════════════════════════════════════════════════════════

def count_profiles(db: Session) -> int:
    """Total number of talent profiles in the database."""
    return db.query(func.count(TalentProfile.id)).scalar() or 0


def list_states(db: Session) -> list[str]:
    """Return all unique state names."""
    rows = db.query(TalentProfile.state).distinct().all()
    return [r[0] for r in rows]


def list_domains(db: Session) -> list[str]:
    """Return all unique domain names."""
    rows = db.query(TalentProfile.domain).distinct().all()
    return [r[0] for r in rows]


def get_profiles_by_state(db: Session, state: str) -> list[TalentProfile]:
    """Fetch all profiles for a given state."""
    return db.query(TalentProfile).filter(TalentProfile.state == state).all()


def get_profiles_by_domain(db: Session, domain: str) -> list[TalentProfile]:
    """Fetch all profiles for a given domain."""
    return db.query(TalentProfile).filter(TalentProfile.domain == domain).all()


# ════════════════════════════════════════════════════════════════════════════
# Risk Analysis Queries
# ════════════════════════════════════════════════════════════════════════════

def calculate_risks(db: Session, state_filter: Optional[str] = None) -> list[dict]:
    """
    Replaces the CSV-based calculate_risks() in api.py.

    Returns per-state risk breakdown:
      - digital_divide : % of profiles with Limited/Occasional access
      - skill_deficit  : % with low learning_behavior (<40)
      - migration_risk : % with skill_score>70 AND opportunity_level=Low
    """
    states = [state_filter] if state_filter else list_states(db)
    results = []

    for state in states:
        profiles = get_profiles_by_state(db, state)
        n = len(profiles)
        if n == 0:
            continue

        dig_risk = (
            sum(1 for p in profiles if p.digital_access in ("Limited", "Occasional")) / n * 100
        )
        skill_deficit = (
            sum(1 for p in profiles if p.learning_behavior < 40) / n * 100
        )

        # Migration risk: high score (>70 in SkillScore table) AND low opportunity
        # We join SkillScore on the fly via the ORM relationship
        profile_ids = [p.id for p in profiles]
        high_skill_low_opp = (
            db.query(func.count(SkillScore.id))
            .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(
                TalentProfile.id.in_(profile_ids),
                SkillScore.score > 70,
                TalentProfile.opportunity_level == "Low",
            )
            .scalar() or 0
        )
        mig_risk = high_skill_low_opp / n * 100

        risk_score = (dig_risk * 0.4) + (skill_deficit * 0.4) + (mig_risk * 0.2)
        level = "Critical" if risk_score > 50 else "Moderate" if risk_score > 20 else "Low"

        results.append({
            "state": state,
            "risk_score": round(risk_score, 1),
            "level": level,
            "factors": {
                "digital_divide": round(dig_risk, 1),
                "skill_deficit": round(skill_deficit, 1),
                "migration": round(mig_risk, 1),
            },
        })

    return results


# ════════════════════════════════════════════════════════════════════════════
# Regional Intelligence Queries
# ════════════════════════════════════════════════════════════════════════════

def regional_analysis(db: Session) -> list[dict]:
    """
    Replaces the CSV-based regional_analysis() in api.py.
    Returns per-state talent density, innovation, and domain specialization.
    """
    results = []
    for state in list_states(db):
        profiles = get_profiles_by_state(db, state)
        n = len(profiles)
        if n == 0:
            continue

        innovation = sum(p.innovation_problem_solving for p in profiles) / n

        # Hidden talent: profile with SkillScore>70 AND opportunity_level=Low
        profile_ids = [p.id for p in profiles]
        hidden_count = (
            db.query(func.count(SkillScore.id))
            .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(
                TalentProfile.id.in_(profile_ids),
                SkillScore.score > 70,
                TalentProfile.opportunity_level == "Low",
            )
            .scalar() or 0
        )
        hidden_density = hidden_count / n * 100

        # Domain specialization: most frequent domain by count
        domain_counts: dict[str, int] = {}
        for p in profiles:
            domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1
        specialization = max(domain_counts, key=domain_counts.get) if domain_counts else "General"

        eco_score = sum(
            (p.collaboration_community + p.economic_activity) / 2 for p in profiles
        ) / n

        results.append({
            "state": state,
            "innovation_intensity": round(innovation, 1),
            "hidden_talent_density": round(hidden_density, 1),
            "specialization": specialization,
            "ecosystem_balance_score": round(eco_score, 1),
        })

    return results


# ════════════════════════════════════════════════════════════════════════════
# State Specialization Queries
# ════════════════════════════════════════════════════════════════════════════

def state_specialization(db: Session) -> list[dict]:
    """
    Returns per-state top domain, avg skill score, and hidden talent rate.
    Replaces the state_specs() endpoint's CSV logic.
    """
    results = []
    for state in list_states(db):
        profiles = get_profiles_by_state(db, state)
        if not profiles:
            continue

        # Get avg skill score per domain using joined SkillScore table
        profile_ids = [p.id for p in profiles]
        domain_scores = (
            db.query(TalentProfile.domain, func.avg(SkillScore.score))
            .join(SkillScore, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(TalentProfile.id.in_(profile_ids))
            .group_by(TalentProfile.domain)
            .all()
        )

        if domain_scores:
            top_domain, avg = max(domain_scores, key=lambda x: x[1] or 0)
        else:
            top_domain, avg = "General", 50.0

        # Hidden talent rate: rural + skill > 65
        rural_high_skill = (
            db.query(func.count(SkillScore.id))
            .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(
                TalentProfile.id.in_(profile_ids),
                TalentProfile.area_type == "Rural",
                SkillScore.score > 65,
            )
            .scalar() or 0
        )
        n = len(profiles)
        hidden_rate = rural_high_skill / n * 100 if n > 0 else 0

        results.append({
            "state": state,
            "specialization": top_domain,
            "avg_skill": round(float(avg or 0), 1),
            "hidden_talent_rate": round(hidden_rate, 1),
        })

    return results


# ════════════════════════════════════════════════════════════════════════════
# Market Intelligence Queries
# ════════════════════════════════════════════════════════════════════════════

DEMAND_TABLE = {
    "Retail & Sales": 82,
    "Manufacturing & Operations": 78,
    "Logistics & Delivery": 85,
    "Agriculture & Allied": 75,
    "Construction & Skilled Trades": 80,
    "Education & Training": 72,
    "Business & Administration": 70,
    "Creative & Media": 65,
    "Service Industry": 88,
    "Entrepreneurship": 76,
}


def market_intelligence(db: Session) -> dict:
    """
    Returns supply vs demand for each domain.
    Replaces the CSV-based market_intel() endpoint logic.
    """
    result = {}
    for domain, demand_index in DEMAND_TABLE.items():
        avg_score = (
            db.query(func.avg(SkillScore.score))
            .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(TalentProfile.domain == domain)
            .scalar()
        )
        supply = float(avg_score) if avg_score is not None else 50.0
        gap = demand_index - supply
        if gap > 10:
            status = "Critical Shortage"
        elif gap > 5:
            status = "Shortage"
        elif gap < -5:
            status = "Surplus"
        else:
            status = "Balanced"

        result[domain] = {
            "demand_index": demand_index,
            "supply_index": round(supply, 1),
            "skill_gap": round(gap, 1),
            "status": status,
        }
    return result


# ════════════════════════════════════════════════════════════════════════════
# National Distribution Queries
# ════════════════════════════════════════════════════════════════════════════

def national_distribution(db: Session) -> dict:
    """
    Returns macro-level KPIs: stability index, hidden talent rate, critical zones.
    Replaces nat_stats() CSV logic.
    """
    risks = calculate_risks(db)
    if not risks:
        return {
            "stability_index": 50.0,
            "hidden_talent_rate": 0.0,
            "critical_zones": 0,
            "skill_velocity": 0.0,
        }

    avg_risk = sum(r["risk_score"] for r in risks) / len(risks)
    critical_zones = sum(1 for r in risks if r["risk_score"] > 50)

    total = count_profiles(db)
    hidden = (
        db.query(func.count(SkillScore.id))
        .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
        .filter(SkillScore.score > 70, TalentProfile.opportunity_level == "Low")
        .scalar() or 0
    )
    hidden_talent_rate = round(hidden / total * 100, 1) if total > 0 else 0.0

    return {
        "stability_index": round(100 - avg_risk, 1),
        "hidden_talent_rate": hidden_talent_rate,
        "critical_zones": critical_zones,
        "skill_velocity": 3.4,  # placeholder; replace with time-series computation
    }


# ════════════════════════════════════════════════════════════════════════════
# Economic Impact Queries
# ════════════════════════════════════════════════════════════════════════════

AVG_PRODUCTIVITY_VALUE = 285.4   # ₹ thousands / person / year

def economic_impact(db: Session) -> dict:
    """
    Calculates economic value of the hidden talent pool.
    Replaces economic_impact() CSV logic.
    """
    total = count_profiles(db)

    # Hidden talent: score > 70 AND opportunity_level = Low
    hidden_rows = (
        db.query(TalentProfile.state, func.count(SkillScore.id).label("cnt"))
        .join(SkillScore, SkillScore.talent_profile_id == TalentProfile.id)
        .filter(SkillScore.score > 70, TalentProfile.opportunity_level == "Low")
        .group_by(TalentProfile.state)
        .all()
    )

    hidden_talent_count = sum(r.cnt for r in hidden_rows)
    total_impact = round(hidden_talent_count * AVG_PRODUCTIVITY_VALUE, 1)

    state_breakdown = [
        {
            "state": r.state,
            "hidden_talent_count": r.cnt,
            "impact": round(r.cnt * AVG_PRODUCTIVITY_VALUE, 1),
        }
        for r in sorted(hidden_rows, key=lambda x: x.cnt, reverse=True)[:5]
    ]

    return {
        "hidden_talent_count": hidden_talent_count,
        "economic_impact": total_impact,
        "avg_productivity_value": AVG_PRODUCTIVITY_VALUE,
        "methodology": "Hidden Talent Count × Avg Productivity Value (₹285.4K/person/year)",
        "state_breakdown": state_breakdown,
        "total_profiles": total,
    }


# ════════════════════════════════════════════════════════════════════════════
# PolicySimulation Write / Read
# ════════════════════════════════════════════════════════════════════════════

def save_policy_simulation(db: Session, sim_data: dict) -> PolicySimulation:
    """
    Persist a completed policy simulation run.
    Returns the saved ORM object (committed).
    """
    roi_crore = round(sim_data.get("risk_reduction", 0) * 42.5, 2)
    sim = PolicySimulation(
        state=sim_data["state"],
        policy_type=sim_data["policy_type"],
        original_risk_score=sim_data["original_risk"],
        digital_divide_before=sim_data.get("factors_before", {}).get("digital_divide"),
        skill_deficit_before=sim_data.get("factors_before", {}).get("skill_deficit"),
        migration_risk_before=sim_data.get("factors_before", {}).get("migration"),
        simulated_risk_score=sim_data["simulated_risk"],
        risk_reduction=sim_data["reduction"],
        digital_divide_impact=sim_data.get("factors_impact", {}).get("digital_divide"),
        skill_deficit_impact=sim_data.get("factors_impact", {}).get("skill_deficit"),
        migration_risk_impact=sim_data.get("factors_impact", {}).get("migration"),
        projected_roi_crore=roi_crore,
        economy_roi_label=f"₹{roi_crore} Cr",
        model_confidence=0.78,
    )
    db.add(sim)
    db.flush()   # get the id before context-manager commits
    return sim


def get_simulation_history(db: Session, state: Optional[str] = None) -> list[dict]:
    """Retrieve recent policy simulations, optionally filtered by state."""
    q = db.query(PolicySimulation)
    if state:
        q = q.filter(PolicySimulation.state == state)
    rows = q.order_by(PolicySimulation.simulated_at.desc()).limit(50).all()
    return [r.to_dict() for r in rows]


# ════════════════════════════════════════════════════════════════════════════
# Skill Trend / Forecast  (reads skill_history JSON column)
# ════════════════════════════════════════════════════════════════════════════

def skill_trends(db: Session) -> dict:
    """
    Compute per-domain skill velocity from the skill_history JSON column.
    Replaces the CSV-based get_trends() logic.
    """
    results = {}
    for domain in list_domains(db):
        profiles = (
            db.query(TalentProfile.skill_history)
            .filter(TalentProfile.domain == domain, TalentProfile.skill_history.isnot(None))
            .limit(100)
            .all()
        )
        velocities = []
        for (hist_json,) in profiles:
            try:
                hist = json.loads(hist_json)
                if len(hist) > 1:
                    recent = hist[-6:]
                    slope = (recent[-1] - recent[0]) / len(recent)
                    velocities.append(slope)
            except Exception:
                pass

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        status = "Emerging" if avg_velocity > 0.5 else "Declining" if avg_velocity < -0.5 else "Stable"

        results[domain] = {
            "status": status,
            "growth_rate": round(avg_velocity * 12, 1),
        }
    return results


def skill_forecast(db: Session) -> dict:
    """
    Richer forecast output (trend label + velocity) for the Forecast.jsx component.
    """
    results = {}
    for domain in list_domains(db):
        profiles = (
            db.query(TalentProfile.skill_history)
            .filter(TalentProfile.domain == domain, TalentProfile.skill_history.isnot(None))
            .limit(100)
            .all()
        )
        velocities = []
        for (hist_json,) in profiles:
            try:
                hist = json.loads(hist_json)
                if len(hist) > 1:
                    recent = hist[-6:] if len(hist) >= 6 else hist
                    slope = (recent[-1] - recent[0]) / len(recent)
                    velocities.append(slope)
            except Exception:
                pass

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0

        if avg_velocity > 1.5:
            trend, status = "Exponential", "High Demand"
        elif avg_velocity > 0.3:
            trend, status = "Rising", "Growing"
        elif avg_velocity < -0.3:
            trend, status = "Declining", "Monitor"
        else:
            trend, status = "Stable", "Sustainable"

        results[domain] = {
            "trend": trend,
            "velocity": round(avg_velocity * 12, 2),
            "status": status,
        }
    return results


# ════════════════════════════════════════════════════════════════════════════
# Policy Recommendation Engine
# ════════════════════════════════════════════════════════════════════════════

def generate_policy_recommendations(db: Session, state_filter: Optional[str] = None) -> list[dict]:
    """
    Returns AI/rule-based policy recommendations per state.
    Replaces the CSV-based /api/policy logic.
    """
    states = [state_filter] if state_filter else list_states(db)
    all_policies = []

    for state in states:
        profiles = get_profiles_by_state(db, state)
        n = len(profiles)
        if n == 0:
            continue

        profile_ids = [p.id for p in profiles]

        digital_limited = sum(
            1 for p in profiles if p.digital_access in ("Limited", "Occasional")
        ) / n * 100

        hidden_count = (
            db.query(func.count(SkillScore.id))
            .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(
                TalentProfile.id.in_(profile_ids),
                SkillScore.score > 70,
                TalentProfile.opportunity_level == "Low",
            )
            .scalar() or 0
        )
        hidden_talent = hidden_count / n * 100
        migration_risk = hidden_talent   # same cohort definition

        avg_score_row = (
            db.query(func.avg(SkillScore.score))
            .join(TalentProfile, SkillScore.talent_profile_id == TalentProfile.id)
            .filter(TalentProfile.id.in_(profile_ids))
            .scalar()
        )
        avg_skill = float(avg_score_row or 50)
        skill_gap = 75 - avg_skill

        spec = {
            "state": state,
            "digital_access_level": digital_limited,
            "hidden_talent_rate": hidden_talent,
            "migration_risk": migration_risk,
            "skill_gap": skill_gap,
        }
        all_policies.extend(_rule_engine(spec))

    return all_policies


def _rule_engine(spec: dict) -> list[dict]:
    """Internal rule-based policy generator (unchanged logic, no CSV)."""
    policies = []
    state = spec["state"]
    digital_access = spec["digital_access_level"]
    hidden_talent = spec["hidden_talent_rate"]
    migration_risk = spec["migration_risk"]
    skill_gap = spec["skill_gap"]

    if digital_access > 40:
        policies.append({
            "state": state,
            "recommended_action": "Deploy Rural Broadband Infrastructure",
            "reason": f"Low digital access detected ({digital_access:.1f}% limited connectivity)",
            "impact_estimate": "+12% digital participation",
            "confidence": 0.82,
            "intervention_priority_score": 85,
        })
    if hidden_talent > 15 and migration_risk > 60:
        policies.append({
            "state": state,
            "recommended_action": "Establish Local Employment Hubs",
            "reason": f"High hidden talent ({hidden_talent:.1f}%) with migration risk ({migration_risk:.1f}%)",
            "impact_estimate": "+18% talent retention",
            "confidence": 0.78,
            "intervention_priority_score": 92,
        })
    if skill_gap > 10:
        policies.append({
            "state": state,
            "recommended_action": "Launch State Skilling Programs",
            "reason": f"Skill gap of {skill_gap:.1f} points detected",
            "impact_estimate": "+8-10 pts avg skill score",
            "confidence": 0.75,
            "intervention_priority_score": 70,
        })
    if migration_risk > 50:
        policies.append({
            "state": state,
            "recommended_action": "Industry Partnership Incentives",
            "reason": f"High migration risk ({migration_risk:.1f}%) indicates opportunity gap",
            "impact_estimate": "+22% local employment",
            "confidence": 0.71,
            "intervention_priority_score": 80,
        })

    return policies
