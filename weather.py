"""
weather.py
======================
Scrapes LIVE weather data for Montgomery, AL using WeatherAPI.com.
Replaces the legacy Bright Data web scraper with a highly reliable, 
direct API architecture for improved production readiness.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add this to your .env file or paste it directly here for the hackathon
import streamlit as st

WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]

#WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

def get_live_weather() -> dict:
    # We use forecast.json instead of current.json to get the &alerts=yes parameter,
    # which is crucial for populating the active alerts list on your dashboard.
    url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q=Montgomery&days=1&aqi=no&alerts=yes"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        condition_text = current.get("condition", {}).get("text", "Unknown")
        temp_f = current.get("temp_f")
        humidity = current.get("humidity")
        wind_mph = current.get("wind_mph")
        precip_in = current.get("precip_in", 0.0)

        # Parse National Weather Service alerts if any are active
        alerts_data = data.get("alerts", {}).get("alert", [])
        alerts = [alert.get("event") for alert in alerts_data] if alerts_data else []

        # Calculate risk multiplier for the machine learning model
        risk_multiplier = 1.0
        cond_lower = condition_text.lower()
        if precip_in > 0 or "rain" in cond_lower or "storm" in cond_lower or "shower" in cond_lower:
            risk_multiplier += 0.2
        if wind_mph and wind_mph > 15:
            risk_multiplier += 0.1
        if alerts:
            risk_multiplier += 0.3

        return {
            "condition": condition_text,
            "temp_f": temp_f,
            "humidity": humidity,
            "wind_mph": wind_mph,
            "precip_in": precip_in,
            "alerts": alerts,
            "risk_multiplier": round(risk_multiplier, 2),
            "source": "WeatherAPI.com (Direct API)",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "condition": "Unknown",
            "temp_f": None,
            "humidity": None,
            "wind_mph": None,
            "precip_in": None,
            "alerts": [],
            "risk_multiplier": 1.0,
            "source": "WeatherAPI.com (failed)",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "error": str(e),
        }

def get_weather_summary(weather: dict) -> str:
    """Return a human-readable one-liner for the dashboard."""
    if not weather["success"]:
        return "⚠️ Live weather unavailable — using manual override"

    parts = []
    if weather["condition"] != "Unknown":
        parts.append(weather["condition"])
    if weather["temp_f"] is not None:
        parts.append(f"{weather['temp_f']}°F")
    if weather["humidity"] is not None:
        parts.append(f"Humidity {weather['humidity']}%")
    if weather["wind_mph"] is not None:
        parts.append(f"Wind {weather['wind_mph']} mph")

    summary = " · ".join(parts) if parts else "Conditions unknown"

    if weather["alerts"]:
        summary += f"  🚨 {len(weather['alerts'])} active alert(s)"

    return summary

# ─────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching live weather ...")
    w = get_live_weather()
    print(w)
    print("\nSummary:", get_weather_summary(w))
