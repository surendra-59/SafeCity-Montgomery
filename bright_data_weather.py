"""
bright_data_weather.py
======================
Scrapes LIVE weather data for Montgomery, AL using Bright Data's
Web Scraper proxy infrastructure.

Bright Data proxy is used to reliably access weather data from
weather.com, bypassing potential anti-bot measures.

Usage:
    from bright_data_weather import get_live_weather
    weather = get_live_weather()
    # Returns dict: {
    #   "condition": "Heavy Rain",
    #   "temp_f": 72,
    #   "humidity": 85,
    #   "wind_mph": 12,
    #   "precip_in": 0.5,
    #   "alerts": ["Flood Watch"],
    #   "risk_multiplier": 1.3,
    #   "source": "weather.com via Bright Data",
    #   "fetched_at": "2026-03-04 15:30:00",
    #   "success": True
    # }
"""

import os
import re
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup

# ─────────────────────────────────────────
# BRIGHT DATA PROXY CONFIGURATION
# ─────────────────────────────────────────
# Set these as environment variables for security:
#   BRIGHT_DATA_HOST, BRIGHT_DATA_PORT,
#   BRIGHT_DATA_USERNAME, BRIGHT_DATA_PASSWORD
#
# Or set BRIGHT_DATA_PROXY as a full URL:
#   http://username:password@host:port

BRIGHT_DATA_HOST     = os.environ.get("BRIGHT_DATA_HOST", "brd.superproxy.io")
BRIGHT_DATA_PORT     = os.environ.get("BRIGHT_DATA_PORT", "33335")
BRIGHT_DATA_USERNAME = os.environ.get("BRIGHT_DATA_USERNAME", "")
BRIGHT_DATA_PASSWORD = os.environ.get("BRIGHT_DATA_PASSWORD", "")

# Montgomery, AL weather page
WEATHER_URL = "https://weather.com/weather/today/l/32.3617,-86.2792"
ALERTS_URL  = "https://weather.com/weather/alerts/l/32.3617,-86.2792"

# ─────────────────────────────────────────
# RISK MULTIPLIER MAPPING
# ─────────────────────────────────────────
# Maps weather conditions → risk multiplier for nuisance prediction
CONDITION_MULTIPLIERS = {
    # Severe
    "tornado":            2.0,
    "tropical storm":     2.0,
    "hurricane":          2.0,
    "severe thunderstorm":1.5,
    "flood":              1.7,
    "flash flood":        1.7,

    # Rain
    "heavy rain":         1.3,
    "rain":               1.2,
    "thunderstorm":       1.4,
    "thunder":            1.4,
    "showers":            1.2,
    "drizzle":            1.1,
    "light rain":         1.1,

    # Other elevated
    "fog":                1.05,
    "haze":               1.0,
    "overcast":           1.0,
    "cloudy":             1.0,

    # Baseline
    "sunny":              1.0,
    "clear":              1.0,
    "partly cloudy":      1.0,
    "mostly sunny":       1.0,
    "fair":               1.0,
}


def _get_proxy_dict():
    """Build Bright Data proxy dict for requests library."""
    proxy_url = os.environ.get("BRIGHT_DATA_PROXY", "")

    if not proxy_url and BRIGHT_DATA_USERNAME and BRIGHT_DATA_PASSWORD:
        proxy_url = (
            f"http://{BRIGHT_DATA_USERNAME}:{BRIGHT_DATA_PASSWORD}"
            f"@{BRIGHT_DATA_HOST}:{BRIGHT_DATA_PORT}"
        )

    if proxy_url:
        return {"http": proxy_url, "https": proxy_url}
    return None


def _condition_to_multiplier(condition: str) -> float:
    """Map a weather condition string to a risk multiplier."""
    condition_lower = condition.lower().strip()
    for keyword, mult in sorted(CONDITION_MULTIPLIERS.items(),
                                 key=lambda x: len(x[0]), reverse=True):
        if keyword in condition_lower:
            return mult
    return 1.0


def _parse_weather_html(html: str) -> dict:
    """
    Parse weather.com HTML page to extract current conditions.
    Returns a dict with weather data.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "condition": "Unknown",
        "temp_f": None,
        "humidity": None,
        "wind_mph": None,
        "precip_in": None,
        "alerts": [],
    }

    # --- Current condition phrase ---
    # weather.com uses data-testid attributes
    phrase_el = soup.find(attrs={"data-testid": "wxPhrase"})
    if phrase_el:
        result["condition"] = phrase_el.get_text(strip=True)

    # --- Temperature ---
    temp_el = soup.find(attrs={"data-testid": "TemperatureValue"})
    if temp_el:
        temp_text = temp_el.get_text(strip=True)
        temp_match = re.search(r"(\d+)", temp_text)
        if temp_match:
            result["temp_f"] = int(temp_match.group(1))

    # --- Humidity ---
    # Look for humidity in weather details
    detail_items = soup.find_all(attrs={"data-testid": "WeatherDetailsListItem"})
    for item in detail_items:
        label = item.get_text(" ", strip=True).lower()
        if "humidity" in label:
            pct = re.search(r"(\d+)%", label)
            if pct:
                result["humidity"] = int(pct.group(1))
        if "wind" in label:
            wind = re.search(r"(\d+)\s*mph", label)
            if wind:
                result["wind_mph"] = int(wind.group(1))

    # --- Alerts ---
    alert_els = soup.find_all(attrs={"data-testid": "AlertHeadline"})
    for el in alert_els:
        result["alerts"].append(el.get_text(strip=True))

    # Also scan for severe/watch/warning text
    for tag in soup.find_all(["h2", "h3", "span", "div"]):
        text = tag.get_text(strip=True).lower()
        for keyword in ["watch", "warning", "advisory", "alert"]:
            if keyword in text and len(text) < 100:
                alert_text = tag.get_text(strip=True)
                if alert_text not in result["alerts"]:
                    result["alerts"].append(alert_text)
                break

    return result


def get_live_weather() -> dict:
    """
    Fetch live weather for Montgomery, AL via Bright Data proxy.

    Returns a dict with weather data including:
        - condition, temp_f, humidity, wind_mph, precip_in
        - alerts (list of active weather alerts)
        - risk_multiplier (computed from condition)
        - source, fetched_at, success
    """
    proxies = _get_proxy_dict()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(
            WEATHER_URL,
            headers=headers,
            proxies=proxies,
            timeout=30,
            verify=False,  # Bright Data proxy may use own cert
        )
        resp.raise_for_status()

        weather = _parse_weather_html(resp.text)

        # Compute risk multiplier from condition
        multiplier = _condition_to_multiplier(weather["condition"])

        # Boost multiplier if active alerts
        if weather["alerts"]:
            for alert in weather["alerts"]:
                alert_lower = alert.lower()
                if any(w in alert_lower for w in ["flood", "flash"]):
                    multiplier = max(multiplier, 1.7)
                elif any(w in alert_lower for w in ["tornado", "hurricane"]):
                    multiplier = max(multiplier, 2.0)
                elif any(w in alert_lower for w in ["thunderstorm", "severe"]):
                    multiplier = max(multiplier, 1.5)
                elif any(w in alert_lower for w in ["watch", "warning"]):
                    multiplier = max(multiplier, 1.3)

        return {
            "condition":       weather["condition"],
            "temp_f":          weather["temp_f"],
            "humidity":        weather["humidity"],
            "wind_mph":        weather["wind_mph"],
            "precip_in":       weather["precip_in"],
            "alerts":          weather["alerts"],
            "risk_multiplier": round(multiplier, 2),
            "source":          "weather.com",
            "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success":         True,
        }

    except Exception as e:
        return {
            "condition":       "Unknown",
            "temp_f":          None,
            "humidity":        None,
            "wind_mph":        None,
            "precip_in":       None,
            "alerts":          [],
            "risk_multiplier": 1.0,
            "source":          "Bright Data (failed)",
            "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success":         False,
            "error":           str(e),
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
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("Fetching live weather ...")
    w = get_live_weather()
    print(json.dumps(w, indent=2))
    print(f"\nSummary: {get_weather_summary(w)}")
