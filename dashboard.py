import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
import contextlib
import io
from dotenv import load_dotenv
import auto_pipeline
import weather
import generate_report

load_dotenv()


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SafeCity Montgomery",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Dark industrial/utilitarian theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       #111827;
    --bg-card2:      #1a2235;
    --accent-red:    #ef4444;
    --accent-orange: #f97316;
    --accent-green:  #22c55e;
    --accent-blue:   #3b82f6;
    --accent-yellow: #eab308;
    --text-primary:  #f1f5f9;
    --text-muted:    #94a3b8;
    --border:        #1e293b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

#MainMenu, footer, header { visibility: visible; }

.block-container { padding: 1.5rem 2rem; max-width: 100%; }

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* FIX: Force the folium iframe to fill its container */
iframe {
    width: 100% !important;
    min-height: 480px !important;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.4rem;
}
.metric-red    { color: var(--accent-red); }
.metric-orange { color: var(--accent-orange); }
.metric-green  { color: var(--accent-green); }
.metric-blue   { color: var(--accent-blue); }

.alert-banner {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid var(--accent-red);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
    0%, 100% { border-color: #ef4444; }
    50%       { border-color: #fca5a5; }
}
.alert-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #fca5a5;
}
.alert-text { font-size: 0.9rem; color: #fecaca; margin-top: 0.2rem; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.risk-high   { color: #ef4444; font-weight: 700; }
.risk-medium { color: #f97316; font-weight: 600; }
.risk-low    { color: #22c55e; }

.dispatch-card {
    background: var(--bg-card2);
    border-left: 3px solid var(--accent-red);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}
.dispatch-card.medium { border-left-color: var(--accent-orange); }
.dispatch-card.low    { border-left-color: var(--accent-green); }

.top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.brand-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}
.brand-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}
.live-badge {
    background: #14532d;
    border: 1px solid #22c55e;
    color: #86efac;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/risk_scores.csv")
    df["risk_label"] = df["risk_label"].astype(str)
    return df

@st.cache_data
def load_importance():
    if os.path.exists("Dataset/feature_importance.csv"):
        return pd.read_csv("Dataset/feature_importance.csv")
    return None

@st.cache_resource
def load_model():
    if os.path.exists("nuisance_predictor.pkl"):
        return joblib.load("nuisance_predictor.pkl")
    return None

df        = load_data()
imp_df    = load_importance()
model     = load_model()

# ─────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SafeCity Controls")
    st.markdown("---")

    # ── LIVE WEATHER ────────────────────────
    st.markdown("**🌦️ Live Weather — Montgomery, AL**")

    # Cache weather for 10 minutes to avoid excessive API calls
    @st.cache_data(ttl=600)
    def _fetch_weather():
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return weather.get_live_weather()

    live_weather = _fetch_weather()

    if live_weather["success"]:
        summary = weather.get_weather_summary(live_weather)
        st.success(f"📡 {summary}")
        st.caption(f"Source: {live_weather['source']} · {live_weather['fetched_at']}")
        if live_weather["alerts"]:
            for alert in live_weather["alerts"]:
                st.warning(f"🚨 {alert}")
    else:
        st.warning("⚠️ Live weather unavailable — use manual override below")
    
    # ── Retry Button for Weather ─────────────────────────
    if st.button("🔄 Retry Weather"):
        st.cache_data.clear()  # clear cached weather if using cache
        st.rerun()              # rerun app to fetch weather again

    st.markdown("")
    weather_source = st.radio(
        "Weather source",
        ["🛰️ Live", "🎛️ Manual Override"],
        index=0 if live_weather["success"] else 1,
        horizontal=True,
    )

    if weather_source == "🛰️ Live" and live_weather["success"]:
        weather_event = live_weather["condition"]
        weather_multiplier = live_weather["risk_multiplier"]
        weather_is_live = True
    else:
        weather_event = st.selectbox("Incoming Event", [
            "None (baseline)",
            "Heavy Rain (2in+)",
            "Severe Thunderstorm",
            "Flash Flood Watch",
            "Tropical Storm Warning"
        ])
        weather_multiplier = {
            "None (baseline)":         1.0,
            "Heavy Rain (2in+)":       1.3,
            "Severe Thunderstorm":     1.5,
            "Flash Flood Watch":       1.7,
            "Tropical Storm Warning":  2.0,
        }[weather_event]
        weather_is_live = False

    st.markdown("---")
    st.markdown("**Risk Filter**")
    show_risk = st.multiselect(
        "Show risk levels",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    st.markdown("---")
    st.markdown("**Alert Threshold**")
    threshold = st.slider("Risk Score Threshold", 0.0, 1.0, 0.50, 0.05)

    st.markdown("---")
    st.markdown("**Map Style**")
    map_type = st.radio("View", ["Heatmap", "Markers"], index=0)

    st.markdown("---")
    st.markdown("### 🔄 Model Maintenance")
    if st.button("Retrain Model & Fetch API", width="stretch", type="primary"):
        with st.status("🚀 Running Auto Pipeline...", expanded=True) as status:
            st.write("Initializing incremental fetch and retraining...")
            f = io.StringIO()
            result = None
            with contextlib.redirect_stdout(f):
                try:
                    result = auto_pipeline.main()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

            logs = f.getvalue()
            st.cache_data.clear()
            st.cache_resource.clear()
            status.update(label="✅ Pipeline Complete! Data refreshed.", state="complete", expanded=False)
            st.toast("Model retrained and data updated!")

            # Store results in session state so they persist across reruns
            st.session_state["pipeline_result"] = result
            st.session_state["pipeline_logs"] = logs

    # ── Show persistent pipeline results until dismissed ──────
    if "pipeline_result" in st.session_state and st.session_state["pipeline_result"]:
        result = st.session_state["pipeline_result"]
        st.markdown("---")
        st.markdown("#### 📡 Data Fetch Summary")
        for stat in result["fetch_stats"]:
            label     = stat["label"]
            new_rows  = stat["new_rows"]
            total     = stat["total_rows"]
            fetched   = stat["fetched"]
            if fetched:
                if stat["is_full"]:
                    st.success(f"**{label}**: Full download — **{new_rows:,}** rows saved")
                else:
                    st.success(f"**{label}**: ✅ **{new_rows:,}** new rows fetched (total: {total:,})")
            else:
                st.info(f"**{label}**: ℹ️ No new data — already up to date ({total:,} rows on disk)")

        st.markdown(f"⏱️ **Pipeline completed in {result['elapsed']}s**")

        logs = st.session_state.get("pipeline_logs", "")
        if logs:
            with st.expander("📋 View Pipeline Logs", expanded=False):
                st.text(logs)

        if st.button("Dismiss & Refresh Data", width="stretch"):
            del st.session_state["pipeline_result"]
            if "pipeline_logs" in st.session_state:
                del st.session_state["pipeline_logs"]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.markdown("---")

    st.markdown("---")
    st.markdown("### 📝 AI Safety Report")
    if st.button("Generate AI Briefing", width="stretch", type="secondary"):
        with st.spinner("🤖 Generating report with Grok AI + Bright Data..."):
            result = generate_report.generate_safety_report(
                weather_data=live_weather,
                weather_multiplier=weather_multiplier,
                include_news=True,
            )
            st.session_state["ai_report"] = result

    if "ai_report" in st.session_state:
        rpt = st.session_state["ai_report"]
        if rpt["success"]:
            st.success("Report ready! See below the main dashboard.")
        else:
            st.error(f"Report failed: {rpt.get('error', 'Unknown')}")

    st.markdown("---")
    st.markdown(f"<small style='color:#64748b'>Model: Random Forest<br>Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>Grid cells: {len(df):,}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# APPLY WEATHER MULTIPLIER
# ─────────────────────────────────────────
df["adjusted_score"] = (df["risk_score"] * weather_multiplier).clip(0, 1)
df["adjusted_label"] = pd.cut(
    df["adjusted_score"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
).astype(str)
df["alert"] = (df["adjusted_score"] >= threshold).astype(int)

df_filtered = df[df["adjusted_label"].isin(show_risk)]

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
    <div>
        <br><br>
        <div class="brand-title">🛡️ SafeCity Montgomery</div>
        <div class="brand-sub">Proactive Environmental Safety Predictor — Real-time Risk Intelligence</div>
    </div>
    <div class="live-badge">● LIVE</div>
</div>
""", unsafe_allow_html=True)

if weather_multiplier > 1.0:
    _src_label = "🛰️ LIVE" if weather_is_live else "🎛️ Manual simulation"
    _alert_details = ""
    if weather_is_live and live_weather.get("temp_f"):
        _alert_details = f" | {live_weather['temp_f']}°F, Humidity {live_weather.get('humidity', '?')}%"
    st.markdown(f"""
    <div class="alert-banner">
        <div class="alert-title">⚡ WEATHER TRIGGER ACTIVE — {_src_label}</div>
        <div class="alert-text">{weather_event}{_alert_details} — risk scores boosted ×{weather_multiplier}. Dispatch alerts updated.</div>
    </div>
    """, unsafe_allow_html=True)
elif weather_is_live and live_weather["success"]:
    st.markdown(f"""
    <div style="background: #14532d; border: 1px solid #22c55e; border-radius: 10px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
        <div style="font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em; color: #86efac;">🛰️ LIVE WEATHER</div>
        <div style="font-size: 0.85rem; color: #bbf7d0; margin-top: 0.2rem;">{weather_event} · No elevated risk detected (×{weather_multiplier})</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────
high_count   = (df["adjusted_label"] == "High").sum()
medium_count = (df["adjusted_label"] == "Medium").sum()
low_count    = (df["adjusted_label"] == "Low").sum()
alert_count  = df["alert"].sum()
total_cells  = len(df)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-red">{high_count}</div><div class="metric-label">High Risk Zones</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-orange">{medium_count}</div><div class="metric-label">Medium Risk Zones</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-green">{low_count}</div><div class="metric-label">Low Risk Zones</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-blue">{alert_count}</div><div class="metric-label">Dispatch Alerts</div></div>', unsafe_allow_html=True)
with c5:
    avg_risk = df["adjusted_score"].mean()
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#a78bfa">{avg_risk:.2f}</div><div class="metric-label">Avg Risk Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN LAYOUT — MAP + PANELS
# ─────────────────────────────────────────
map_col, right_col = st.columns([2, 1])

with map_col:
    st.markdown('<div class="section-header">Risk Heatmap — Montgomery, AL</div>', unsafe_allow_html=True)

    # ── FIX: compute center robustly ──────────────────────────────────────────
    valid_coords = df_filtered.dropna(subset=["cell_lat", "cell_lon"])
    valid_coords = valid_coords[valid_coords["adjusted_score"] > 0]

    if len(valid_coords) > 0:
        center_lat = float(valid_coords["cell_lat"].median())
        center_lon = float(valid_coords["cell_lon"].median())
    else:
        # Montgomery, AL fallback coords
        center_lat = 32.3617
        center_lon = -86.2792

    # ── Build Folium map ──────────────────────────────────────────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,       # FIX: canvas renderer is faster & more reliable
    )

    if map_type == "Heatmap":
        # ── FIX: build heat_data only from fully valid rows ───────────────────
        heat_df = valid_coords[["cell_lat", "cell_lon", "adjusted_score"]].dropna().copy()
        
        heat_data = heat_df.values.tolist()
        
        # FIX: Folium HeatMap auto-scales the max value to 1.0, turning low values into Red.
        # We append a dummy anchor coordinate at (0,0) with a 1.0 score to lock the color scale.
        heat_data.append([0.0, 0.0, 1.0])

        if len(heat_data) > 0:
            HeatMap(
                heat_data,
                name="Risk Heatmap",
                radius=15,
                blur=20,
                max_zoom=13,
                min_opacity=0.3,       # FIX: ensures tiles paint even at low scores
                gradient={
                    "0.0": "#22c55e",
                    "0.4": "#eab308",
                    "0.7": "#f97316",
                    "1.0": "#ef4444"
                }
            ).add_to(m)
        else:
            st.warning("No valid coordinate data to display on heatmap.")

    else:  # Markers
        color_map = {"High": "red", "Medium": "orange", "Low": "green"}
        sample = valid_coords.sample(min(500, len(valid_coords)), random_state=42)
        for _, row in sample.iterrows():
            folium.CircleMarker(
                location=[float(row["cell_lat"]), float(row["cell_lon"])],
                radius=6,
                color=color_map.get(str(row["adjusted_label"]), "gray"),
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>Risk: {row['adjusted_label']}</b><br>"
                    f"Score: {row['adjusted_score']:.3f}<br>"
                    f"Cell: {row['grid_cell']}<br>"
                    f"<a href='https://www.google.com/maps?q={row['cell_lat']},{row['cell_lon']}' target='_blank'>View on Google Maps</a>",
                    max_width=250
                )
            ).add_to(m)

    # ── FIX: pass explicit pixel dimensions + unique key ─────────────────────
    # width must be an int (pixels), NOT None — None causes blank render in
    # some versions of streamlit-folium.
    st_folium(
        m,
        width=750,          # explicit px width prevents blank iframe bug
        height=480,
        returned_objects=[],   # FIX: skip return value serialization (faster)
        key=f"folium_map_{map_type}_{weather_event}_{'-'.join(sorted(show_risk))}",
    )

with right_col:
    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)

    risk_counts = df["adjusted_label"].value_counts()
    color_map = {"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"}
    pie_colors = [color_map.get(str(label), "#888888") for label in risk_counts.index]
    fig_donut = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker_colors=pie_colors,
        textinfo="label+percent",
        textfont=dict(family="DM Sans", size=11),
    ))
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        annotations=[dict(
            text=f"<b>{total_cells}</b><br>zones",
            x=0.5, y=0.5,
            font_size=13,
            font_color="#f1f5f9",
            showarrow=False
        )]
    )
    st.plotly_chart(fig_donut, width="stretch")

    st.markdown('<div class="section-header">Top Dispatch Alerts</div>', unsafe_allow_html=True)

    top_alerts = df[df["alert"] == 1].nlargest(6, "adjusted_score")

    def get_dispatch_action(row):
        """Derive a context-aware dispatch recommendation from the zone's risk profile."""
        label = str(row["adjusted_label"])
        if label == "Low":
            return "✅ Monitor", "low"
        if label == "Medium":
            return "⚠️ Schedule inspection", "medium"

        # --- HIGH risk: pick the best action(s) from the data ---
        actions = []
        nr  = row.get("nuisance_rate", 0) or 0
        ovr = row.get("open_violation_rate", 0) or 0
        cpr = row.get("chronic_parcel_rate", 0) or 0
        sg  = row.get("siren_coverage_gap", 0) or 0

        if nr > 0.5:
            actions.append("🦟 Nuisance abatement crew (drainage/mosquito)")
        if ovr > 0.3:
            actions.append("🏚️ Code enforcement — open violations")
        if cpr > 0.3:
            actions.append("📋 Chronic offender — escalate to legal/lien")
        if sg == 1:
            actions.append("📡 Siren gap — alert Emergency Mgmt")

        # Fallback: if none of the specific triggers fire
        if not actions:
            tc = row.get("total_complaints", 0) or 0
            tn = row.get("total_nuisance", 0) or 0
            if tc > 0 and tn / tc < 0.3:
                actions.append("🔧 Maintenance crew (infrastructure/sanitation)")
            else:
                actions.append("🚨 Deploy field inspection team")

        return " · ".join(actions), ""

    if len(top_alerts) == 0:
        st.info("No alerts at current threshold.")
    else:
        for _, row in top_alerts.iterrows():
            label   = str(row["adjusted_label"])
            action, css_cls = get_dispatch_action(row)
            lat = row['cell_lat']
            lon = row['cell_lon']
            maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            
            st.markdown(f"""
            <div class="dispatch-card {css_cls}">
                <b>Zone {row['grid_cell']}</b><br>
                Score: <b>{row['adjusted_score']:.3f}</b> &nbsp;|&nbsp; {label}<br>
                📍 <a href="{maps_url}" target="_blank" style="color:#3b82f6; text-decoration:none;">View on Google Maps</a><br>
                <small style="color:#94a3b8">{action}</small>
            </div>
            """, unsafe_allow_html=True)
            
        # THE MERGE: Your Automated Discord Dispatch Integration
        st.markdown("<br>", unsafe_allow_html=True)
        
        is_severe = (weather_multiplier >= 1.5) or ("severe" in weather_event.lower())
        auto_send = False
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        dispatch_file = "Dataset/.last_dispatch.txt"
        
        if is_severe:
            last_dispatch = ""
            if os.path.exists(dispatch_file):
                with open(dispatch_file, "r") as f:
                    last_dispatch = f.read().strip()
            
            if last_dispatch != today_str:
                auto_send = True

        if st.button("📲 Push Dispatch Orders to Discord", width="stretch", type="primary") or auto_send:
            try:
                WEBHOOK_URL = st.secrets["DISCORD_WEBHOOK_URL"]
            except Exception:
                WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
                
            import requests
            
            if auto_send:
                os.makedirs(os.path.dirname(dispatch_file), exist_ok=True)
                with open(dispatch_file, "w") as f:
                    f.write(today_str)
                st.toast("⚠️ Severe Weather Detected: Automatic Daily Discord Dispatch triggered!")

            with st.spinner("Transmitting automated dispatch orders..."):
                # Send alerts for the top 3 highest risk zones
                for _, row in top_alerts.head(3).iterrows():
                    action, _ = get_dispatch_action(row)
                    prob_score = row['adjusted_score'] * 100
                    historical_count = int(row.get('total_complaints', 0))
                    
                    maps_url = f"https://www.google.com/maps?q={row['cell_lat']},{row['cell_lon']}"
                    
                    payload = {
                        "content": "PROACTIVE CITY ALERT: ENVIRONMENTAL HAZARD PREDICTED",
                        "embeds": [
                            {
                                "title": f"Dispatch Order: Vulnerable Sector {row['grid_cell']}",
                                "description": f"**Location:** Coordinates {row['cell_lat']}, {row['cell_lon']} ([View on Google Maps]({maps_url}))\n**Risk Level:** CRITICAL ({prob_score:.1f}% Probability Score)\n**Historical Baseline:** {historical_count} prior incidents.",
                                "color": 16711680, 
                                "fields": [
                                    {
                                        "name": "Live Environmental Trigger", 
                                        "value": f"Current conditions: {weather_event} (Risk Multiplier: {weather_multiplier}x)"
                                    },
                                    {
                                        "name": "Recommended Municipal Action", 
                                        "value": action
                                    }
                                ],
                                "footer": {
                                    "text": "City of Montgomery - SafeCity AI Command Center"
                                }
                            }
                        ]
                    }
                    if WEBHOOK_URL:
                        try:
                            requests.post(WEBHOOK_URL, json=payload)
                        except Exception as e:
                            st.error(f"Failed to post to Discord: {e}")
            st.success("SUCCESS: Dispatch alerts successfully delivered to the communication channel!")

# ─────────────────────────────────────────
# BOTTOM ROW — CHARTS
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(df, x="adjusted_score", nbins=40, color_discrete_sequence=["#3b82f6"])
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                       annotation_text="Threshold", annotation_font_color="#ef4444")
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9", margin=dict(t=10,b=30,l=30,r=10),
        height=220, xaxis_title="Risk Score", yaxis_title="Grid Cells", showlegend=False
    )
    fig_hist.update_xaxes(gridcolor="#1e293b")
    fig_hist.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig_hist, width="stretch")

with b2:
    st.markdown('<div class="section-header">Top 10 Feature Importances</div>', unsafe_allow_html=True)
    if imp_df is not None:
        top10 = imp_df.head(10)
        fig_imp = px.bar(
            top10[::-1], x="importance", y="feature",
            orientation="h", color="importance",
            color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"]
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9", margin=dict(t=10,b=10,l=10,r=10),
            height=220, showlegend=False, coloraxis_showscale=False,
            yaxis_title="", xaxis_title="Importance"
        )
        fig_imp.update_xaxes(gridcolor="#1e293b")
        fig_imp.update_yaxes(gridcolor="#1e293b", tickfont=dict(size=10))
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("feature_importance.csv not found.")

with b3:
    st.markdown('<div class="section-header">Alerts by Risk Level</div>', unsafe_allow_html=True)
    alert_breakdown = df[df["alert"]==1]["adjusted_label"].value_counts().reset_index()
    alert_breakdown.columns = ["Risk Level", "Count"]
    fig_bar = px.bar(
        alert_breakdown, x="Risk Level", y="Count",
        color="Risk Level",
        color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#22c55e"}
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9", margin=dict(t=10,b=30,l=30,r=10),
        height=220, showlegend=False, xaxis_title="", yaxis_title="Alerts"
    )
    fig_bar.update_xaxes(gridcolor="#1e293b")
    fig_bar.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig_bar, width="stretch")

# ─────────────────────────────────────────
# DATA TABLE
# ─────────────────────────────────────────
with st.expander("📋 View Raw Risk Scores Table"):
    show_cols = ["grid_cell", "cell_lat", "cell_lon", "risk_score", "adjusted_score", "adjusted_label", "alert"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df[show_cols].sort_values("adjusted_score", ascending=False),
        width="stretch",
        height=300
    )
    csv = df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Risk Scores CSV", csv, "Dataset/Export/risk_scores_export.csv", "text/csv")

with st.expander("📈 View Model Training & Evaluation Report"):
    if os.path.exists("model_evaluation.png"):
        st.image("model_evaluation.png", caption="Last Calibration: ROC, PR Curves and Feature Importance", width="stretch")
    else:
        st.info("Evaluation report image not found. Run 'Retrain Model' to generate it.")

# ─────────────────────────────────────────
# AI-GENERATED SAFETY REPORT
# ─────────────────────────────────────────
if "ai_report" in st.session_state and st.session_state["ai_report"]["success"]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 AI-Generated City Safety Briefing</div>', unsafe_allow_html=True)

    rpt = st.session_state["ai_report"]

    # Main report card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #111827, #1a2235);
                border: 1px solid #3b82f6; border-radius: 12px;
                padding: 1.5rem 2rem; margin-bottom: 1rem;">
        <div style="font-family: 'Space Mono', monospace; font-size: 0.7rem;
                    letter-spacing: 0.15em; color: #60a5fa; margin-bottom: 1rem;
                    text-transform: uppercase;">
            SafeCity AI Briefing — {datetime.now().strftime('%B %d, %Y')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(rpt["report"])

    # Expandable sections for transparency
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("📊 Raw Data Fed to AI"):
            st.text(rpt.get("stats_summary", "N/A"))
    with col_b:
        with st.expander("📰 News Headlines Scraped (Bright Data)"):
            st.text(rpt.get("news_headlines", "N/A"))

    if st.button("🗑️ Dismiss Report"):
        del st.session_state["ai_report"]
        st.rerun()