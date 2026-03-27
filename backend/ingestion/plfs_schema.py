"""
backend/ingestion/plfs_schema.py – PLFS Column Specification & Field Map
SkillGenome X

This module defines the required columns from the MoSPI PLFS microdata exports
and the transformation logic to map them to internal TalentProfile features.
"""

from collections import OrderedDict

# ── MoSPI PLFS Required Columns (based on Annual Report Layout) ──────────────
# We focus on the Employment-Unemployment (Schedule 10.4) key indicators.
PLFS_REQUIRED_COLUMNS = {
    "sector": "int64",             # 1=Rural, 2=Urban
    "state_code": "int64",          # MoSPI 2-digit state code
    "nic_code": "object",           # Industry code (NIC 2008)
    "usual_activity_status": "int64", # UAS code (11-99)
    "education_level": "int64",     # Educational attainment (01-19)
    "age": "int64",                # Age in years
}

PLFS_OPTIONAL_COLUMNS = {
    "weekly_wage": "float64",       # Cash/kind earnings
    "land_possessed": "float64",    # Land area in hectares
    "internet_use": "int64",        # 1=Yes, 2=No
}

# ── MoSPI State Code → Internal State Name ───────────────────────────────────
STATE_CODE_MAP = {
    27: "Maharashtra",
    29: "Karnataka",
    33: "Tamil Nadu",
    "07": "Delhi",
    "09": "Uttar Pradesh",
    36: "Telangana",
    24: "Gujarat",
    19: "West Bengal",
    "08": "Rajasthan",
    10: "Bihar",
    23: "Madhya Pradesh",
    32: "Kerala",
    "03": "Punjab",
    21: "Odisha",
    28: "Andhra Pradesh",
    "06": "Haryana",
    18: "Assam",
    20: "Jharkhand",
    22: "Chhattisgarh",
    "05": "Uttarakhand",
}

# ── NIC-2008 2-digit Industry → SkillGenome Domain ───────────────────────────
NIC_TO_DOMAIN_MAP = {
    "01": "Agriculture",
    "02": "Agriculture",
    "03": "Agriculture",
    "10": "Skilled Trades", # Manufacturing
    "20": "Technology",      # Chemicals/High-tech mfg
    "26": "Technology",      # Electronics
    "62": "Technology",      # IT/Software
    "63": "Data & Research", # Info services
    "72": "Data & Research", # Scientific R&D
    "85": "Education",
    "86": "Healthcare",
    "90": "Creative",       # Arts
    "94": "Social Impact",   # Membership orgs
}

# ── UAS (Usual Activity Status) → Opportunity Level ──────────────────────────
ACTIVITY_TO_OPPORTUNITY = {
    11: "High",      # Self-employed (employer)
    21: "Moderate",  # Self-employed (own account worker)
    31: "Moderate",  # Regular wage/salaried
    41: "Low",       # Casual labor
    51: "Low",       # Casual labor (others)
    91: "Low",       # Attendance in education (unemployed)
}

# ── Education Level (01–19) → Learning Behavior Score (0–100) ────────────────
EDUCATION_TO_LEARNING = {
    1: 10,  # Not literate
    6: 30,  # Primary
    8: 50,  # Middle
    10: 70, # Secondary
    12: 85, # Graduate
    13: 95, # Post Graduate
}

# ── Transformation Functions ──────────────────────────────────────────────────

def transform_area(sector):
    return "Rural" if sector == 1 else "Urban"

def transform_state(code):
    if code is None: return None
    # Try direct map (for ints) or string map (for leading zeros)
    if code in STATE_CODE_MAP:
        return STATE_CODE_MAP[code]
    s_code = str(code).zfill(2)
    return STATE_CODE_MAP.get(s_code) or STATE_CODE_MAP.get(int(code) if str(code).isdigit() else None)

def transform_domain(nic):
    if not nic: return "General"
    prefix = str(nic)[:2]
    return NIC_TO_DOMAIN_MAP.get(prefix, "General")

def transform_opportunity(uas):
    return ACTIVITY_TO_OPPORTUNITY.get(uas, "Moderate")

def transform_learning(edu):
    return EDUCATION_TO_LEARNING.get(edu, 50)
