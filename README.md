# SafeCity Montgomery

**Proactive Environmental Safety Predictor** — A civic-tech Streamlit dashboard that predicts environmental safety risk in Montgomery, Alabama using 311 data, code violations, weather sirens, live weather, and machine learning.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Codebase Analysis](#codebase-analysis)
- [Environment Variables & Secrets](#environment-variables--secrets)
- [Project Structure](#project-structure)

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd SafeCity-Montgomery

# 2. Install dependencies (uses uv)
uv sync

# 3. Configure secrets (see Setup Instructions)
# Copy .env.example to .env and create .streamlit/secrets.toml

# 4. Run the dashboard
uv run streamlit run dashboard.py
```

---

## Prerequisites

- **Python 3.13+**
- **uv** — Fast Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- API keys for optional features (see [Environment Variables](#environment-variables--secrets))

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd SafeCity-Montgomery
```

### Step 2: Create Virtual Environment & Install Dependencies

```bash
# uv reads pyproject.toml and installs all packages into .venv
uv sync
```

> **Note:** `uv sync` creates a virtual environment automatically. No need to run `python -m venv .venv` separately.

### Step 3: Configure Secrets (Required for Dashboard)

The dashboard uses **Streamlit secrets** (`st.secrets`) for API keys. Create the secrets file:

**Option A — From existing `.env` file:**

1. Create a `.env` file in the project root (or workspace root) with your API keys.
2. Create `.streamlit/secrets.toml` with the same keys in TOML format:

```toml
# .streamlit/secrets.toml
WEATHER_API_KEY = "your-weatherapi-key"
GROQ_API_KEY = "your-groq-key"
BRIGHT_DATA_HOST = "brd.superproxy.io"
BRIGHT_DATA_PORT = 33335
BRIGHT_DATA_USERNAME = "your-username"
BRIGHT_DATA_PASSWORD = "your-password"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
```

**Option B — Copy from template:**

```bash
# If .env exists at workspace root, you can manually copy values to:
# SafeCity-Montgomery/.streamlit/secrets.toml
```

> **Important:** Add `.streamlit/secrets.toml` to `.gitignore` — never commit API keys.

### Step 4: Activate Virtual Environment (Optional)

On **Windows:**
```bash
.venv\Scripts\activate
```

On **macOS/Linux:**
```bash
source .venv/bin/activate
```

> With `uv run`, activation is optional — `uv run` uses the project venv automatically.

### Step 5: Run Jupyter Lab (Optional)

Use this if you want to explore or modify the development notebooks:

```bash
uv run jupyter lab
```

---

## Running the Application

### Start the Dashboard

```bash
# From SafeCity-Montgomery directory
uv run streamlit run dashboard.py
```

The app opens in your browser at `http://localhost:8501`.

### Run the Data Pipeline (Optional)

To fetch fresh data from ArcGIS APIs and retrain the model:

```bash
uv run python auto_pipeline.py
```

> **Tip:** You can also trigger this from the dashboard sidebar via **"Retrain Model & Fetch API"**.

### Pipeline Outputs

After running `auto_pipeline.py`, these files are generated/updated:

| File | Description |
|------|-------------|
| `Dataset/311_requests_full.csv` | Raw 311 data from ArcGIS |
| `Dataset/montgomery_code_violations_full.csv` | Raw code violations |
| `Dataset/311_requests_cleaned.csv` | Cleaned 311 data |
| `Dataset/violations_cleaned.csv` | Cleaned violations |
| `Dataset/sirens_cleaned.csv` | Weather siren locations |
| `Dataset/feature_matrix.csv` | ML feature matrix |
| `Dataset/risk_scores.csv` | Risk scores per grid cell |
| `Dataset/feature_importance.csv` | Feature rankings |
| `nuisance_predictor.pkl` | Trained Random Forest model |
| `model_evaluation.png` | ROC, PR curves, confusion matrix |

---

## Codebase Analysis

### 1. Application Type

**Streamlit web app** — A data dashboard for environmental safety risk in Montgomery, Alabama.

- **Primary interface:** `dashboard.py`
- **Purpose:** Proactive environmental safety predictor with real-time risk intelligence
- **Domain:** Civic tech / urban safety analytics

---

### 2. Project Structure

```
SafeCity-Montgomery/
├── dashboard.py              # Main Streamlit app (entry point)
├── auto_pipeline.py          # Data fetch + ML retraining pipeline
├── weather.py                # Live weather API integration
├── generate_report.py        # AI safety report (Groq + Bright Data)
├── pyproject.toml            # Python project config & dependencies
├── uv.lock                   # Locked dependency versions
├── .python-version           # Python 3.13
├── .streamlit/
│   └── secrets.toml          # API keys (create from .env)
├── Dataset/                  # Data files
│   ├── 311_requests_full.csv
│   ├── 311_requests_cleaned.csv
│   ├── montgomery_code_violations_full.csv
│   ├── violations_cleaned.csv
│   ├── sirens_cleaned.csv
│   ├── Weather_Sirens.csv
│   ├── feature_matrix.csv
│   ├── risk_scores.csv
│   ├── feature_importance.csv
│   └── retrain_history.json
├── nuisance_predictor.pkl    # Trained Random Forest model
├── model_evaluation.png      # Evaluation charts
├── development.ipynb         # Data cleaning/EDA notebook
└── EDA.ipynb                 # Exploratory data analysis
```

---

### 3. Data Pipeline (`auto_pipeline.py`)

The pipeline runs in **5 steps** (mirrors `development.ipynb`):

| Step | Description | Output |
|------|-------------|--------|
| **Step 0** | Fetch 311 + Code Violations from ArcGIS REST APIs (incremental) | `311_requests_full.csv`, `montgomery_code_violations_full.csv` |
| **Step 1** | Clean 311 data (drop nulls, parse dates, flag nuisance/chronic) | `311_requests_cleaned.csv` |
| **Step 2** | Clean Code Violations (drop DEBUG_TEST, encode categoricals) | `violations_cleaned.csv` |
| **Step 3** | Clean Weather Sirens (keep matched geocodes, validate coords) | `sirens_cleaned.csv` |
| **Step 4** | Build feature matrix (grid cells, 311/violation/siren features) | `feature_matrix.csv` |
| **Step 5** | Train Random Forest, score grid cells, save model & charts | `nuisance_predictor.pkl`, `risk_scores.csv`, `feature_importance.csv`, `model_evaluation.png` |

---

### 4. Dashboard Flow (`dashboard.py`)

1. **Load data** — `risk_scores.csv`, `feature_importance.csv`, `nuisance_predictor.pkl` (cached)
2. **Weather** — `weather.get_live_weather()` → WeatherAPI.com (cached 10 min)
3. **Risk computation** — `adjusted_score = risk_score × weather_multiplier`
4. **Bins** — Low (<0.33), Medium (0.33–0.66), High (>0.66)
5. **Retrain** — Button triggers `auto_pipeline.main()`
6. **AI report** — Button calls `generate_report.generate_safety_report()` (Groq Llama 3.3 70B)

---

### 5. Weather Integration (`weather.py`)

- **API:** WeatherAPI.com (`forecast.json` with `alerts=yes`)
- **Risk multiplier:** 1.0 base; +0.2 for rain/storm; +0.1 for wind >15 mph; +0.3 for active alerts

---

### 6. AI Report (`generate_report.py`)

- **LLM:** Groq (Llama 3.3 70B)
- **Inputs:** Risk scores, feature importance, 311 data, weather, optional news
- **News:** Bright Data proxy scraping (Montgomery Advertiser, WSFA)

---

### 7. Dependencies

| Category | Packages |
|----------|-----------|
| **Web UI** | Streamlit, streamlit-folium |
| **Maps** | Folium |
| **Visualization** | Plotly, Altair, Matplotlib, Seaborn |
| **Data** | Pandas, NumPy, GeoPandas, Shapely |
| **ML** | scikit-learn, joblib |
| **APIs** | requests, groq |
| **Scraping** | BeautifulSoup4 |
| **Config** | python-dotenv |

---

### 8. Data Flow Summary

```
ArcGIS APIs (311, Violations)  →  raw CSVs
         ↓
   auto_pipeline.py (Steps 1–5)
         ↓
   feature_matrix.csv  →  Random Forest  →  nuisance_predictor.pkl
         ↓                                    risk_scores.csv
   WeatherAPI.com  →  weather_multiplier
         ↓
   dashboard.py  →  adjusted_score = risk_score × weather_multiplier
         ↓
   Heatmap, KPIs, dispatch alerts, AI report
```

---

## Environment Variables & Secrets

| Variable | Purpose | Required |
|----------|---------|----------|
| `WEATHER_API_KEY` | WeatherAPI.com | Yes (dashboard) |
| `GROQ_API_KEY` | Groq (AI briefings) | For AI report |
| `BRIGHT_DATA_HOST`, `BRIGHT_DATA_PORT`, `BRIGHT_DATA_USERNAME`, `BRIGHT_DATA_PASSWORD` | News scraping | For AI report with news |
| `DISCORD_WEBHOOK_URL` | Dispatch alerts to Discord | Optional |

**Where to get keys:**
- Weather: https://www.weatherapi.com/
- Groq: https://console.groq.com/
- Bright Data: https://brightdata.com/
- Discord: Server Settings → Integrations → Webhooks

---

## Project Structure

| File | Purpose |
|------|---------|
| `dashboard.py` | Main Streamlit app — KPIs, heatmap, dispatch alerts, AI report |
| `auto_pipeline.py` | End-to-end data fetch + cleaning + ML pipeline |
| `weather.py` | Live weather from WeatherAPI.com |
| `generate_report.py` | AI safety briefing via Groq + optional news scraping |
| `pyproject.toml` | Dependencies (uv) |
| `development.ipynb` | Original data cleaning/EDA (pipeline mirrors this) |

---

## Deployment

- **Streamlit Cloud:** Uses `st.secrets` — configure in Streamlit Cloud dashboard
- **Local:** Use `.streamlit/secrets.toml` (see Setup Instructions)
- No Docker or CI/CD config in repo

---

## License


