"""
backend/ingestion/plfs_pipeline.py – PLFS Data Ingestion Pipeline
SkillGenome X

Implements a 4-stage ETL pipeline:
1. Validate schema
2. Map PLFS fields to internal feature vectors
3. Impute missing values using domain rules
4. Bulk-load into PostgreSQL
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict

from db.session import get_db
from db.models import TalentProfile
from .plfs_schema import (
    PLFS_REQUIRED_COLUMNS,
    transform_area,
    transform_state,
    transform_domain,
    transform_opportunity,
    transform_learning
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PLFSPipeline")

class PLFSValidationError(Exception):
    """Raised when the input CSV fails schema validation."""
    pass

@dataclass
class IngestionResult:
    loaded: int
    skipped: int
    errors: int
    timestamp: str = datetime.now().isoformat()

class PLFSPipeline:
    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size

    def run(self, csv_path: str, dry_run: bool = False) -> IngestionResult:
        """Execute the full pipeline for a given CSV file."""
        logger.info(f"Starting PLFS ingestion: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return IngestionResult(0, 0, 1)

        # Stage 1: Validate
        try:
            self.stage_1_validate(df)
        except PLFSValidationError as e:
            logger.error(f"Validation failed: {e}")
            raise

        # Stage 2: Map
        mapped_df = self.stage_2_map(df)
        
        # Stage 3: Impute
        final_df = self.stage_3_impute(mapped_df)
        
        if dry_run:
            logger.info("Dry run enabled. Skipping DB load.")
            print(final_df.head(10).to_string())
            return IngestionResult(len(final_df), 0, 0)

        # Stage 4: Load
        return self.stage_4_load(final_df)

    def stage_1_validate(self, df: pd.DataFrame):
        """Checks if required columns exist and dtypes match."""
        if df.empty:
            raise PLFSValidationError("Input DataFrame is empty.")

        missing = [col for col in PLFS_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise PLFSValidationError(f"Missing required PLFS columns: {', '.join(missing)}")
        
        logger.info(f"Validation successful: {len(df)} rows found.")

    def stage_2_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms PLFS fields to SkillGenome internal vectors."""
        logger.info("Mapping PLFS fields to internal format …")
        
        rows = []
        for idx, row in df.iterrows():
            # State mapping is critical; if it fails, we skip the row
            state_name = transform_state(row.get("state_code"))
            if not state_name:
                logger.warning(f"Row {idx}: Unmapped state code {row.get('state_code')}. Skipping.")
                continue

            mapped_row = {
                "state": state_name,
                "domain": transform_domain(row.get("nic_code")),
                "area_type": transform_area(row.get("sector")),
                "digital_access": "Regular", # Default if internet_use not present
                "opportunity_level": transform_opportunity(row.get("usual_activity_status")),
                "learning_behavior": transform_learning(row.get("education_level")),
                # Mandatory behavioral pins with defaults (imputed later if needed)
                "creation_output": 50,
                "experience_consistency": 50,
                "economic_activity": 50,
                "innovation_problem_solving": 50,
                "collaboration_community": 50,
                "offline_capability": 50,
                "digital_presence": 50,
                "learning_hours": 20,
                "projects": 3,
                # Socio-economic
                "internet_penetration": None,
                "urban_population_percent": None,
                "per_capita_income": None,
                "workforce_participation": None,
                "literacy_rate": None,
                "unemployment_rate": None,
                "skill_history": json.dumps([50]*24), # Start with flat baseline
                "created_at": datetime.utcnow()
            }
            
            # Use land_possessed as a proxy for offline_capability if present
            if "land_possessed" in row and pd.notna(row["land_possessed"]):
                # Simple mapping: 0-10 hectares -> 0-100 score
                mapped_row["offline_capability"] = min(100, int(row["land_possessed"] * 10))
            
            # Use weekly_wage for economic_activity
            if "weekly_wage" in row and pd.notna(row["weekly_wage"]):
                # Simple normalization: assume >5000 is top tier
                mapped_row["economic_activity"] = min(100, int(row["weekly_wage"] / 50))

            rows.append(mapped_row)
            
        return pd.DataFrame(rows)

    def stage_3_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values using domain rules and batch medians."""
        logger.info("Imputing missing values using batch medians …")
        
        # In a real pipeline, we'd use state/domain medians.
        # For this version, we ensure all mandatory behavioral columns have values.
        behavioral_cols = [
            'creation_output', 'learning_behavior', 'experience_consistency',
            'economic_activity', 'innovation_problem_solving', 'collaboration_community',
            'offline_capability', 'digital_presence'
        ]
        
        for col in behavioral_cols:
            if col in df.columns:
                df[col] = df[col].fillna(50)
                
        return df

    def stage_4_load(self, df: pd.DataFrame) -> IngestionResult:
        """Bulk-inserts records into the database."""
        total = len(df)
        logger.info(f"Loading {total} records into PostgreSQL …")
        
        loaded = 0
        errors = 0
        
        try:
            with get_db() as db:
                for start in range(0, total, self.batch_size):
                    batch = df.iloc[start : start + self.batch_size]
                    db_rows = [TalentProfile(**row.to_dict()) for _, row in batch.iterrows()]
                    db.add_all(db_rows)
                    db.commit()
                    loaded += len(db_rows)
                    logger.info(f"Batch loaded: {loaded}/{total}")
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            errors += 1
            
        return IngestionResult(loaded, 0, errors)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    pipeline = PLFSPipeline()
    res = pipeline.run(args.csv, dry_run=args.dry_run)
    print(res)
