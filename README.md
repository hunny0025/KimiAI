# SkillGenome X — National Talent Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/India%20Innovates-2026-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Stack-Python%20|%20React%20|%20Flask-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/AI-Explainable%20AI-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Hackathon-Ready-green?style=for-the-badge" />
</p>

> **SkillGenome X** is an AI-powered **National Talent Intelligence System** that maps, predicts, and unlocks India's hidden workforce. Built for high-stakes policy decisions, it uses **Explainable AI (XAI)** to convert raw behavioral signals into actionable national intelligence — empowering planners, investors, and administrators with real-time regional insights.

---

## 🌍 Problem Statement

- **500M+ workers** in India lack verified, credential-backed skill records.
- Rural and peri-urban talent pools remain **invisible to national planners**.
- Brain-drain risk goes **undetected** until economic crises emerge.
- AI-driven policy interventions are deployed **without predictive validation**.

---

## ✅ Solution

SkillGenome X provides a **live, explainable AI dashboard** that:
1. Generates transparent **Skill Risk Scores** using Gradient Boosting & XAI breakdown.
2. Maps **Hidden Talent** in under-served regions through Anomaly Detection (Isolation Forest).
3. Identifies **Digital Divide and Migration Risk** zones on a national heatmap.
4. Simulates the **impact of government policies** before real-world deployment (What-If Engine).

---

## 🧠 Core Features

### 1. 📊 Command Center (Dashboard)
Real-time national KPIs including:
- National Stability Index
- Hidden Talent Discovery Rate
- Critical Risk Zones
- Skill Velocity Metrics

### 2. 🤖 Skill Risk AI Engine
- Accepts behavioral signals: literacy, employment, internet, urbanization, education.
- Predicts a **Skill Risk Score** using Gradient Boosting Regressor.
- Powered by **Explainable AI (XAI)** — every score shows which factors drove it, positively or negatively.
- Live animated **typing effect** for AI-generated explanation paragraphs.

### 3. 🗺️ Regional Intelligence
- Interactive **India choropleth map** (React Simple Maps).
- Click any state to view its talent density, digital divide index, and risk level.
- Side panel dynamically shows state-specific intelligence cards.

### 4. 🧪 Policy Simulator (What-If Engine)
- Select a state and a government policy intervention (Skilling, Broadband, Education).
- Run a simulation to see **Projected Risk Reduction** and **Economy ROI**.
- Results show exact metrics: `₹42.5 Cr` projected ROI, `-7.8 pts` risk reduction.

### 5. ⚠️ Risk Analysis Panel
- State-by-state breakdown of **Structural Risks** (Digital Divide, Skill Deficit, Migration Risk).
- Animated horizontal bar charts with color-coded thresholds.
- Shows critical states like Rajasthan, Bihar, Gujarat at a glance.

---

## 🛠️ Architecture

```
User Dashboard (React 18)
       │
       ▼
REST API Layer (Flask + Gunicorn)
       │
       ▼
AI Engine (Scikit-Learn)
  ├── Gradient Boosting Regressor  → Skill Risk Score
  └── Isolation Forest             → Anomaly/Hidden Talent Detection
       │
       ▼
Explainability Layer (XAI)
  └── Factor Attribution & Human-readable Insight Generation
       │
       ▼
Visualization (Recharts + React Simple Maps)
  └── National Heatmaps, Bar Charts, KPI Gauges
```

---

## 💻 Technology Stack

| Layer        | Technology                                          |
|:-------------|:----------------------------------------------------|
| **Frontend** | React 18, Vite, Tailwind CSS, Framer Motion         |
| **Charts**   | Recharts, React Simple Maps                         |
| **Backend**  | Python 3.12, Flask, Gunicorn                        |
| **AI / ML**  | Scikit-Learn, Pandas, NumPy, Joblib                 |
| **Deploy**   | Vercel (Frontend), Render / Docker (Backend)        |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Clone the Repository
```bash
git clone https://github.com/hunny0025/_skillgenomeX12.git
cd _skillgenomeX12
```

### 2. Run the Backend
```bash
cd backend
pip install -r requirements.txt
python api.py
# Backend starts at http://localhost:5000
```

### 3. Run the Frontend
```bash
cd frontend
npm install
npm run dev
# Frontend starts at http://localhost:5173
```

---

## 🏆 Hackathon Highlights (India Innovates 2026)

| Feature | Description |
|:--------|:------------|
| 🎯 **XAI Transparency** | Every AI prediction auditable by humans — no black box |
| 🔬 **What-If Simulator** | Policy validation before national deployment |
| 🌏 **National Scale** | Calibrated to PLFS / MoSPI government datasets |
| ⚡ **Real-Time Inference** | Sub-200ms predictions from the Flask AI engine |
| 📍 **Hidden Talent Finder** | Actively identifies rural high-potential individuals |
| 💰 **Economic ROI Engine** | Quantifies ₹ GDP uplift potential of interventions |

---

## 📁 Project Structure

```
_skillgenomeX12/
├── backend/
│   ├── api.py                  # Core Flask API with AI endpoints
│   ├── requirements.txt        # Python dependencies
│   └── data/                   # Training & inference data
├── frontend/
│   ├── src/
│   │   ├── components/         # Reusable React UI components
│   │   ├── pages/              # Dashboard, Skill AI, Regional, Policy, Risk
│   │   └── App.jsx             # App routing & layout
│   ├── package.json
│   └── vite.config.js
├── vercel.json                 # Deployment configuration
└── README.md
```

---

## 🔗 Resources
- 📡 **Deployed Dasboard**: https://skillgenome-x12.vercel.app/
- 🎬 **Demo Video**: https://drive.google.com/file/d/1ThKae7l51iJVphI6zgaud8pLxRI8SV5u/view?usp=sharing
- 📄 **Full Documentation**: `PROJECT_DOCUMENTATION.md`
- 📊 **Presentation Summary**: `PRESENTATION_SUMMARY.md`
- 🏛️ **Dataset Sources**: PLFS (Periodic Labour Force Survey), MoSPI

---

## 👥 Team

*Developed for **India Innovates 2026** — Bharat Mandapam, New Delhi.*

---

<p align="center">
  <strong>Building India's intelligent talent graph — one signal at a time.</strong>
</p>
