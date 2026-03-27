"""
db/models.py – SQLAlchemy ORM Models
SkillGenome X National Talent Intelligence System

Models:
  - TalentProfile      : Core demographic & behavioral profile per individual
  - SkillScore         : The computed skill score record tied to a profile
  - PolicySimulation   : Saved policy simulation run results per state
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, Text, ForeignKey, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class TalentProfile(Base):
    """
    Represents a single individual's talent profile sourced from
    the synthetic_talent_data.csv (originally) or live ingestion.

    Feature Groups:
      - Identity     : state, domain, area_type
      - Behavioral   : creation_output, learning_behavior, …
      - Socio-economic: digital_access, opportunity_level, internet_penetration, …
      - Temporal     : skill_history (JSON string), created_at
    """
    __tablename__ = "talent_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Identity / Context ---
    state              = Column(String(100), nullable=False, index=True)
    domain             = Column(String(120), nullable=False, index=True)
    area_type          = Column(String(50),  nullable=False)           # Urban / Semi-Urban / Rural
    digital_access     = Column(String(50),  nullable=False)           # High / Regular / Limited / Occasional
    opportunity_level  = Column(String(50),  nullable=False)           # High / Moderate / Low

    # --- Behavioral Signals (0–100 int scale) ---
    creation_output             = Column(Integer, nullable=False, default=50)
    learning_behavior           = Column(Integer, nullable=False, default=50)
    experience_consistency      = Column(Integer, nullable=False, default=50)
    economic_activity           = Column(Integer, nullable=False, default=50)
    innovation_problem_solving  = Column(Integer, nullable=False, default=50)
    collaboration_community     = Column(Integer, nullable=False, default=50)
    offline_capability          = Column(Integer, nullable=False, default=50)
    digital_presence            = Column(Integer, nullable=False, default=50)
    learning_hours              = Column(Integer, nullable=False, default=20)
    projects                    = Column(Integer, nullable=False, default=3)

    # --- Socio-Economic Signals (real-valued) ---
    internet_penetration     = Column(Float, nullable=True)   # %
    urban_population_percent = Column(Float, nullable=True)   # %
    per_capita_income        = Column(Float, nullable=True)   # ₹ per annum
    workforce_participation  = Column(Float, nullable=True)   # %
    literacy_rate            = Column(Float, nullable=True)   # %
    unemployment_rate        = Column(Float, nullable=True)   # %

    # --- Temporal ---
    skill_history = Column(Text, nullable=True)   # JSON-encoded list of monthly scores
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- Relationships ---
    skill_score_record = relationship(
        "SkillScore",
        back_populates="talent_profile",
        uselist=False,
        cascade="all, delete-orphan"
    )

    # --- Indexes for common query patterns ---
    __table_args__ = (
        Index("idx_tp_state_domain", "state", "domain"),
        Index("idx_tp_area_digital", "area_type", "digital_access"),
        Index("idx_tp_opportunity", "opportunity_level"),
    )

    def __repr__(self):
        return (
            f"<TalentProfile id={self.id} state={self.state!r} "
            f"domain={self.domain!r} area={self.area_type!r}>"
        )

    def to_dict(self):
        """Serialize to plain dict for API responses."""
        return {
            "id": self.id,
            "state": self.state,
            "domain": self.domain,
            "area_type": self.area_type,
            "digital_access": self.digital_access,
            "opportunity_level": self.opportunity_level,
            "creation_output": self.creation_output,
            "learning_behavior": self.learning_behavior,
            "experience_consistency": self.experience_consistency,
            "economic_activity": self.economic_activity,
            "innovation_problem_solving": self.innovation_problem_solving,
            "collaboration_community": self.collaboration_community,
            "offline_capability": self.offline_capability,
            "digital_presence": self.digital_presence,
            "learning_hours": self.learning_hours,
            "projects": self.projects,
            "internet_penetration": self.internet_penetration,
            "urban_population_percent": self.urban_population_percent,
            "per_capita_income": self.per_capita_income,
            "workforce_participation": self.workforce_participation,
            "literacy_rate": self.literacy_rate,
            "unemployment_rate": self.unemployment_rate,
            "skill_history": self.skill_history,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SkillScore(Base):
    """
    Records the most recent ML-computed skill score for a TalentProfile.
    Stores the raw score, the explainability breakdown, and metadata
    about the model version that produced it.

    One-to-one with TalentProfile.
    """
    __tablename__ = "skill_scores"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    talent_profile_id  = Column(
        Integer,
        ForeignKey("talent_profiles.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # --- Score Fields ---
    score        = Column(Float, nullable=False)                # 0.0 – 100.0
    level        = Column(String(30), nullable=False)           # Intermediate / Advanced / Expert
    confidence   = Column(Float, nullable=True)                 # 0.0 – 100.0
    is_anomaly   = Column(Boolean, nullable=False, default=False)
    is_hidden_talent = Column(Boolean, nullable=False, default=False)
    migration_risk   = Column(String(20), nullable=False, default="Low")  # Low / Medium / High

    # --- Workforce Assessment ---
    work_capacity    = Column(String(20), nullable=True)
    growth_potential = Column(String(20), nullable=True)
    risk_level       = Column(String(20), nullable=True)

    # --- XAI: Stored as JSON strings ---
    top_positive_factors = Column(Text, nullable=True)  # JSON list
    top_negative_factors = Column(Text, nullable=True)  # JSON list

    # --- Model Metadata ---
    model_version = Column(String(80), nullable=True, default="GradientBoostingRegressor")
    computed_at   = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- Relationship ---
    talent_profile = relationship("TalentProfile", back_populates="skill_score_record")

    __table_args__ = (
        Index("idx_ss_score", "score"),
        Index("idx_ss_hidden", "is_hidden_talent"),
        Index("idx_ss_migration", "migration_risk"),
    )

    def __repr__(self):
        return (
            f"<SkillScore id={self.id} profile_id={self.talent_profile_id} "
            f"score={self.score} level={self.level!r}>"
        )

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "talent_profile_id": self.talent_profile_id,
            "score": self.score,
            "level": self.level,
            "confidence": self.confidence,
            "is_anomaly": self.is_anomaly,
            "is_hidden_talent": self.is_hidden_talent,
            "migration_risk": self.migration_risk,
            "work_capacity": self.work_capacity,
            "growth_potential": self.growth_potential,
            "risk_level": self.risk_level,
            "top_positive_factors": json.loads(self.top_positive_factors) if self.top_positive_factors else [],
            "top_negative_factors": json.loads(self.top_negative_factors) if self.top_negative_factors else [],
            "model_version": self.model_version,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
        }


class PolicySimulation(Base):
    """
    Stores each policy simulation run triggered via /api/policy-simulate.
    Captures the input parameters, the risk deltas, and an economy ROI estimate.

    Each row = one simulation event (auditable log of all What-If runs).
    """
    __tablename__ = "policy_simulations"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    state       = Column(String(100), nullable=False, index=True)
    policy_type = Column(String(80),  nullable=False)    # Broadband / Skilling / Hubs

    # --- Baseline Snapshot ---
    original_risk_score   = Column(Float, nullable=False)
    digital_divide_before = Column(Float, nullable=True)
    skill_deficit_before  = Column(Float, nullable=True)
    migration_risk_before = Column(Float, nullable=True)

    # --- Simulated Outcome ---
    simulated_risk_score   = Column(Float, nullable=False)
    risk_reduction         = Column(Float, nullable=False)
    digital_divide_impact  = Column(Float, nullable=True)
    skill_deficit_impact   = Column(Float, nullable=True)
    migration_risk_impact  = Column(Float, nullable=True)

    # --- Economic Projection ---
    projected_roi_crore    = Column(Float, nullable=True)   # ₹ Crore GDP uplift estimate
    economy_roi_label      = Column(String(50), nullable=True)

    # --- Simulation Metadata ---
    model_confidence = Column(Float, nullable=True, default=0.78)
    simulated_at     = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_ps_state_policy", "state", "policy_type"),
    )

    def __repr__(self):
        return (
            f"<PolicySimulation id={self.id} state={self.state!r} "
            f"policy={self.policy_type!r} reduction={self.risk_reduction}>"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state,
            "policy_type": self.policy_type,
            "original_risk": self.original_risk_score,
            "simulated_risk": self.simulated_risk_score,
            "reduction": self.risk_reduction,
            "digital_divide_before": self.digital_divide_before,
            "skill_deficit_before": self.skill_deficit_before,
            "migration_risk_before": self.migration_risk_before,
            "factors_impact": {
                "digital_divide": self.digital_divide_impact,
                "skill_deficit": self.skill_deficit_impact,
                "migration": self.migration_risk_impact,
            },
            "projected_roi_crore": self.projected_roi_crore,
            "economy_roi_label": self.economy_roi_label,
            "model_confidence": self.model_confidence,
            "simulated_at": self.simulated_at.isoformat() if self.simulated_at else None,
        }
