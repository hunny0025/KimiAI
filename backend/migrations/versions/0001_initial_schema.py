"""Initial migration – Create talent_profiles, skill_scores, policy_simulations

Revision ID: 0001
Revises: None
Create Date: 2026-03-26
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── talent_profiles ───────────────────────────────────────────────────
    op.create_table(
        "talent_profiles",
        sa.Column("id",                          sa.Integer(),    nullable=False),
        sa.Column("state",                       sa.String(100),  nullable=False),
        sa.Column("domain",                      sa.String(120),  nullable=False),
        sa.Column("area_type",                   sa.String(50),   nullable=False),
        sa.Column("digital_access",              sa.String(50),   nullable=False),
        sa.Column("opportunity_level",           sa.String(50),   nullable=False),
        # Behavioral signals
        sa.Column("creation_output",             sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("learning_behavior",           sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("experience_consistency",      sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("economic_activity",           sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("innovation_problem_solving",  sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("collaboration_community",     sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("offline_capability",          sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("digital_presence",            sa.Integer(),    nullable=False, server_default="50"),
        sa.Column("learning_hours",              sa.Integer(),    nullable=False, server_default="20"),
        sa.Column("projects",                    sa.Integer(),    nullable=False, server_default="3"),
        # Socio-economic signals
        sa.Column("internet_penetration",        sa.Float(),      nullable=True),
        sa.Column("urban_population_percent",    sa.Float(),      nullable=True),
        sa.Column("per_capita_income",           sa.Float(),      nullable=True),
        sa.Column("workforce_participation",     sa.Float(),      nullable=True),
        sa.Column("literacy_rate",               sa.Float(),      nullable=True),
        sa.Column("unemployment_rate",           sa.Float(),      nullable=True),
        # Temporal
        sa.Column("skill_history",  sa.Text(),     nullable=True),
        sa.Column("created_at",     sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_tp_state",           "talent_profiles", ["state"])
    op.create_index("idx_tp_domain",          "talent_profiles", ["domain"])
    op.create_index("idx_tp_state_domain",    "talent_profiles", ["state", "domain"])
    op.create_index("idx_tp_area_digital",    "talent_profiles", ["area_type", "digital_access"])
    op.create_index("idx_tp_opportunity",     "talent_profiles", ["opportunity_level"])

    # ── skill_scores ──────────────────────────────────────────────────────
    op.create_table(
        "skill_scores",
        sa.Column("id",                  sa.Integer(),  nullable=False),
        sa.Column("talent_profile_id",   sa.Integer(),  nullable=False),
        sa.Column("score",               sa.Float(),    nullable=False),
        sa.Column("level",               sa.String(30), nullable=False),
        sa.Column("confidence",          sa.Float(),    nullable=True),
        sa.Column("is_anomaly",          sa.Boolean(),  nullable=False, server_default="false"),
        sa.Column("is_hidden_talent",    sa.Boolean(),  nullable=False, server_default="false"),
        sa.Column("migration_risk",      sa.String(20), nullable=False, server_default="Low"),
        sa.Column("work_capacity",       sa.String(20), nullable=True),
        sa.Column("growth_potential",    sa.String(20), nullable=True),
        sa.Column("risk_level",          sa.String(20), nullable=True),
        sa.Column("top_positive_factors", sa.Text(),   nullable=True),
        sa.Column("top_negative_factors", sa.Text(),   nullable=True),
        sa.Column("model_version",       sa.String(80), nullable=True),
        sa.Column("computed_at",         sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["talent_profile_id"], ["talent_profiles.id"],
            name="fk_skill_scores_profile", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("talent_profile_id", name="uq_skill_scores_profile"),
    )
    op.create_index("idx_ss_score",    "skill_scores", ["score"])
    op.create_index("idx_ss_hidden",   "skill_scores", ["is_hidden_talent"])
    op.create_index("idx_ss_migration","skill_scores", ["migration_risk"])

    # ── policy_simulations ────────────────────────────────────────────────
    op.create_table(
        "policy_simulations",
        sa.Column("id",                    sa.Integer(),  nullable=False),
        sa.Column("state",                 sa.String(100), nullable=False),
        sa.Column("policy_type",           sa.String(80),  nullable=False),
        # Before snapshot
        sa.Column("original_risk_score",   sa.Float(),    nullable=False),
        sa.Column("digital_divide_before", sa.Float(),    nullable=True),
        sa.Column("skill_deficit_before",  sa.Float(),    nullable=True),
        sa.Column("migration_risk_before", sa.Float(),    nullable=True),
        # After simulation
        sa.Column("simulated_risk_score",  sa.Float(),    nullable=False),
        sa.Column("risk_reduction",        sa.Float(),    nullable=False),
        sa.Column("digital_divide_impact", sa.Float(),    nullable=True),
        sa.Column("skill_deficit_impact",  sa.Float(),    nullable=True),
        sa.Column("migration_risk_impact", sa.Float(),    nullable=True),
        # Economic
        sa.Column("projected_roi_crore",   sa.Float(),    nullable=True),
        sa.Column("economy_roi_label",     sa.String(50), nullable=True),
        # Meta
        sa.Column("model_confidence",      sa.Float(),    nullable=True, server_default="0.78"),
        sa.Column("simulated_at",          sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_ps_state",        "policy_simulations", ["state"])
    op.create_index("idx_ps_state_policy", "policy_simulations", ["state", "policy_type"])


def downgrade() -> None:
    op.drop_table("policy_simulations")
    op.drop_table("skill_scores")
    op.drop_table("talent_profiles")
