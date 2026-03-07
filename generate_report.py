"""
generate_report.py
==================
Generates an AI-powered city safety report using Groq (free Llama 3.3 70B),
enriched with live data from the SafeCity pipeline and optionally
with local news scraped via Bright Data.

Usage:
    from generate_report import generate_safety_report
    report = generate_safety_report(weather_data, weather_multiplier)
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
BRIGHT_DATA_HOST = st.secrets["BRIGHT_DATA_HOST"]
BRIGHT_DATA_PORT = st.secrets["BRIGHT_DATA_PORT"]
BRIGHT_DATA_USERNAME = st.secrets["BRIGHT_DATA_USERNAME"]
BRIGHT_DATA_PASSWORD = st.secrets["BRIGHT_DATA_PASSWORD"]

#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

#BRIGHT_DATA_HOST     = os.environ.get("BRIGHT_DATA_HOST")
#BRIGHT_DATA_PORT     = os.environ.get("BRIGHT_DATA_PORT" )
#BRIGHT_DATA_USERNAME = os.environ.get("BRIGHT_DATA_USERNAME")
#BRIGHT_DATA_PASSWORD = os.environ.get("BRIGHT_DATA_PASSWORD")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# Local news sources to scrape for context
NEWS_URLS = [
    "https://www.montgomeryadvertiser.com/news/",
    "https://www.wsfa.com/weather/",
]


# ─────────────────────────────────────────
# BRIGHT DATA — LOCAL NEWS SCRAPER
# ─────────────────────────────────────────
def _get_proxy_dict():
    """Build Bright Data proxy config."""
    if BRIGHT_DATA_USERNAME and BRIGHT_DATA_PASSWORD:
        proxy_url = (
            f"http://{BRIGHT_DATA_USERNAME}:{BRIGHT_DATA_PASSWORD}"
            f"@{BRIGHT_DATA_HOST}:{BRIGHT_DATA_PORT}"
        )
        return {"http": proxy_url, "https": proxy_url}
    return None


def scrape_local_news() -> str:
    """
    Scrape Montgomery local news headlines via Bright Data proxy.
    Returns a short summary string of recent headlines.
    Falls back gracefully if proxy is unavailable.
    """
    proxies = _get_proxy_dict()
    if not proxies:
        return "Local news unavailable (no proxy credentials)."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
    }

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    headlines = []
    for url in NEWS_URLS:
        try:
            resp = requests.get(
                url, headers=headers, proxies=proxies,
                timeout=15, verify=False,
            )
            resp.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "a"]):
                text = tag.get_text(strip=True)
                words = text.split()
                if 25 < len(text) < 150 and len(words) >= 5 and re.search(r"[a-zA-Z]{3,}", text):
                    keywords = ["montgomery", "weather", "flood", "storm",
                                "fire", "police", "crime", "road", "water",
                                "emergency", "alert", "city", "county",
                                "shoot", "kill", "arrest", "crash", "danger"]
                    if any(kw in text.lower() for kw in keywords):
                        headlines.append(text)
        except Exception:
            continue

    if headlines:
        seen = set()
        unique = []
        for h in headlines:
            h_lower = h.lower().strip()
            if h_lower not in seen:
                seen.add(h_lower)
                unique.append(h)
        return "\n".join(f"- {h}" for h in unique[:5])

    return "No relevant local news headlines found."


# ─────────────────────────────────────────
# DATA GATHERER — build prompt context
# ─────────────────────────────────────────
def _gather_city_stats(weather_data: dict, weather_multiplier: float) -> str:
    """
    Read risk_scores.csv, feature_importance.csv, and 311 data
    to build a structured data summary for the LLM prompt.
    """
    lines = []
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    lines.append(f"Report generated: {now}")

    # ── Risk Scores ──
    try:
        rs = pd.read_csv(os.path.join(DATASET_DIR, "risk_scores.csv"))
        total = len(rs)
        high  = (rs["risk_label"] == "High").sum()
        med   = (rs["risk_label"] == "Medium").sum()
        low   = (rs["risk_label"] == "Low").sum()
        avg_score = rs["risk_score"].mean()

        lines.append(f"\n## City Risk Overview")
        lines.append(f"- Total grid cells monitored: {total:,}")
        lines.append(f"- High risk zones: {high:,} ({high/total*100:.1f}%)")
        lines.append(f"- Medium risk zones: {med:,} ({med/total*100:.1f}%)")
        lines.append(f"- Low risk zones: {low:,} ({low/total*100:.1f}%)")
        lines.append(f"- Average risk score: {avg_score:.3f}")

        # Top 5 worst zones with context
        top5 = rs.nlargest(5, "risk_score")
        lines.append(f"\n## Top 5 Highest-Risk Zones")
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            nr  = r.get("nuisance_rate", 0) or 0
            ovr = r.get("open_violation_rate", 0) or 0
            cpr = r.get("chronic_parcel_rate", 0) or 0
            sg  = r.get("siren_coverage_gap", 0) or 0
            tc  = r.get("total_complaints", 0) or 0
            lines.append(
                f"  {i}. Zone {r['grid_cell']} (score: {r['risk_score']:.3f}) — "
                f"{int(tc)} total complaints, nuisance rate: {nr:.0%}, "
                f"open violations: {ovr:.0%}, chronic parcels: {cpr:.0%}"
                f"{', NO SIREN COVERAGE' if sg == 1 else ''}"
            )

        siren_gaps = (rs["siren_coverage_gap"] == 1).sum()
        lines.append(f"\n## Infrastructure")
        lines.append(f"- Zones without siren coverage (>3km gap): {siren_gaps}")
    except Exception as e:
        lines.append(f"[Risk data unavailable: {e}]")

    # ── Feature Importance ──
    try:
        imp = pd.read_csv(os.path.join(DATASET_DIR, "feature_importance.csv"))
        top3 = imp.head(3)
        lines.append(f"\n## Top Risk Drivers (ML Model)")
        for _, r in top3.iterrows():
            lines.append(f"- {r['feature']}: importance = {r['importance']:.4f}")
    except Exception:
        pass

    # ── Complaint Type Breakdown ──
    try:
        df311 = pd.read_csv(os.path.join(DATASET_DIR, "311_requests_cleaned.csv"))
        lines.append(f"\n## 311 Complaint Breakdown")
        lines.append(f"- Total complaints on record: {len(df311):,}")
        nuisance_pct = df311["is_nuisance"].mean() * 100
        lines.append(f"- Nuisance-related: {nuisance_pct:.1f}%")
        top_types = df311["Request_Type"].value_counts().head(5)
        lines.append(f"- Top request types:")
        for rt, count in top_types.items():
            lines.append(f"  - {rt}: {count:,}")
    except Exception:
        pass

    # ── Weather ──
    lines.append(f"\n## Current Weather Conditions")
    if weather_data and weather_data.get("success"):
        lines.append(f"- Condition: {weather_data.get('condition', 'Unknown')}")
        if weather_data.get("temp_f"):
            lines.append(f"- Temperature: {weather_data['temp_f']}°F")
        if weather_data.get("humidity"):
            lines.append(f"- Humidity: {weather_data['humidity']}%")
        if weather_data.get("wind_mph"):
            lines.append(f"- Wind: {weather_data['wind_mph']} mph")
        if weather_data.get("alerts"):
            lines.append(f"- Active alerts: {', '.join(weather_data['alerts'])}")
    else:
        lines.append(f"- Live weather: unavailable")
    lines.append(f"- Weather risk multiplier applied: {weather_multiplier}x")

    return "\n".join(lines)


# ─────────────────────────────────────────
# GROQ — REPORT GENERATION (Free Llama 3.3)
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are SafeCity AI, an expert urban safety analyst for Montgomery, Alabama.
You write professional daily briefings for city officials and emergency managers.

Your tone is:
- Professional but accessible
- Data-driven — cite specific numbers and zones
- Action-oriented — always end sections with clear recommendations
- Urgent when appropriate, calm when not

Format the report with these sections:
1. 🔴 Executive Summary (2-3 sentences max)
2. 🗺️ Risk Landscape (key stats, trends)
3. ⚡ Critical Zones (top zones needing immediate attention, with specific actions)
4. 🌦️ Weather Impact (how current weather affects risk)
5. 📰 Local Context (if news data available)
6. ✅ Recommended Actions (prioritized list for today)

Use emojis sparingly for section headers only. Keep the total report under 500 words.
Write in a format that a busy city manager can scan in 60 seconds."""


def generate_safety_report(
    weather_data: dict = None,
    weather_multiplier: float = 1.0,
    include_news: bool = True,
) -> dict:
    """
    Generate an AI-powered city safety report using Groq (Llama 3.3 70B).

    Returns dict with: report, news_headlines, stats_summary, success, error
    """
    if not GROQ_API_KEY:
        return {
            "report": "⚠️ Groq API key not configured. Set GROQ_API_KEY in your .env file.",
            "news_headlines": "",
            "stats_summary": "",
            "success": False,
            "error": "Missing GROQ_API_KEY",
        }

    # 1. Gather city stats
    stats = _gather_city_stats(weather_data or {}, weather_multiplier)

    # 2. Optionally scrape local news
    news = ""
    if include_news:
        try:
            news = scrape_local_news()
        except Exception:
            news = "News scraping unavailable."

    # 3. Build the user prompt
    user_prompt = f"""Generate a SafeCity Montgomery Daily Briefing based on the following live data:

{stats}

## Local News Context (via Bright Data)
{news if news else "No local news data available today."}

Today's date: {datetime.now().strftime('%B %d, %Y')}
Write the report now."""

    # 4. Call Groq (free Llama 3.3 70B)
    try:
        client = Groq(api_key=GROQ_API_KEY)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_completion_tokens=1500,
        )

        report_text = chat_completion.choices[0].message.content

        return {
            "report": report_text,
            "news_headlines": news,
            "stats_summary": stats,
            "success": True,
            "error": None,
            "model_used": "llama-3.3-70b-versatile (Groq)",
        }

    except Exception as e:
        return {
            "report": f"⚠️ Report generation failed: {str(e)}",
            "news_headlines": news,
            "stats_summary": stats,
            "success": False,
            "error": str(e),
        }


# ─────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SafeCity Report Generator — CLI Test (Groq / Llama 3.3)")
    print("=" * 60)

    mock_weather = {
        "condition": "Partly Cloudy",
        "temp_f": 72,
        "humidity": 65,
        "wind_mph": 8,
        "alerts": [],
        "risk_multiplier": 1.0,
        "success": True,
    }

    result = generate_safety_report(
        weather_data=mock_weather,
        weather_multiplier=1.0,
        include_news=True,
    )

    if result["success"]:
        print(f"\n📝 GENERATED REPORT (model: {result.get('model_used', '?')}):")
        print("-" * 60)
        print(result["report"])
        print("-" * 60)
        print(f"\n📰 News used: {result['news_headlines'][:200]}...")
    else:
        print(f"\n❌ Failed: {result['error']}")
