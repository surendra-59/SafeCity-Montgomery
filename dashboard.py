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
import auto_pipeline

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

    st.markdown("**Weather Trigger Simulation**")
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
    if st.button("Retrain Model & Fetch API", use_container_width=True, type="primary"):
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

        if st.button("Dismiss & Refresh Data", use_container_width=True):
            del st.session_state["pipeline_result"]
            if "pipeline_logs" in st.session_state:
                del st.session_state["pipeline_logs"]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.markdown("---")

    st.markdown("---")
    st.markdown(f"<small style='color:#64748b'>Model: Random Forest<br>Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>Grid cells: {len(df):,}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# APPLY WEATHER MULTIPLIER
# ─────────────────────────────────────────
df["adjusted_score"] = (df["risk_score"] * weather_multiplier).clip(0, 1)
df["adjusted_label"] = pd.cut(
    df["adjusted_score"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"]
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

if weather_event != "None (baseline)":
    st.markdown(f"""
    <div class="alert-banner">
        <div class="alert-title">⚡ WEATHER TRIGGER ACTIVE</div>
        <div class="alert-text">{weather_event} detected — risk scores boosted ×{weather_multiplier}. Dispatch alerts updated.</div>
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
        heat_data = (
            valid_coords[["cell_lat", "cell_lon", "adjusted_score"]]
            .dropna()
            .values
            .tolist()
        )

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
                    f"Cell: {row['grid_cell']}",
                    max_width=200
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
    fig_donut = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker_colors=["#ef4444", "#f97316", "#22c55e"],
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
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('<div class="section-header">Top Dispatch Alerts</div>', unsafe_allow_html=True)

    top_alerts = df[df["alert"] == 1].nlargest(6, "adjusted_score")[
        ["grid_cell", "adjusted_score", "adjusted_label"]
    ]

    if len(top_alerts) == 0:
        st.info("No alerts at current threshold.")
    else:
        for _, row in top_alerts.iterrows():
            label   = str(row["adjusted_label"])
            css_cls = "medium" if label == "Medium" else ("low" if label == "Low" else "")
            action  = ("🚨 Deploy mosquito/drainage crew" if label == "High"
                       else "⚠️ Schedule inspection" if label == "Medium"
                       else "✅ Monitor")
            st.markdown(f"""
            <div class="dispatch-card {css_cls}">
                <b>Zone {row['grid_cell']}</b><br>
                Score: <b>{row['adjusted_score']:.3f}</b> &nbsp;|&nbsp; {label}<br>
                <small style="color:#94a3b8">{action}</small>
            </div>
            """, unsafe_allow_html=True)

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
    st.plotly_chart(fig_hist, use_container_width=True)

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
        st.plotly_chart(fig_imp, use_container_width=True)
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
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────
# DATA TABLE
# ─────────────────────────────────────────
with st.expander("📋 View Raw Risk Scores Table"):
    show_cols = ["grid_cell", "cell_lat", "cell_lon", "risk_score", "adjusted_score", "adjusted_label", "alert"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df[show_cols].sort_values("adjusted_score", ascending=False),
        use_container_width=True,
        height=300
    )
    csv = df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Risk Scores CSV", csv, "Dataset/Export/risk_scores_export.csv", "text/csv")

with st.expander("📈 View Model Training & Evaluation Report"):
    if os.path.exists("model_evaluation.png"):
        st.image("model_evaluation.png", caption="Last Calibration: ROC, PR Curves and Feature Importance", use_container_width=True)
    else:
        st.info("Evaluation report image not found. Run 'Retrain Model' to generate it.")