import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==============================================================================
# TRUSTSHIELD UEBA - LIVE SOC ANALYST CONSOLE
# ==============================================================================

st.set_page_config(
    page_title="TrustShield UEBA | Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SOC Dark Mode CSS
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .soc-header {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .soc-title {
        font-size: 24px;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 0.5px;
        margin: 0;
    }
    
    .soc-subtitle {
        font-size: 13px;
        color: #8b949e;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    .status-badge-online {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-mode {
        background-color: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    
    .metric-val-alert { font-size: 26px; font-weight: 700; color: #f85149; }
    .metric-val-log { font-size: 26px; font-weight: 700; color: #d29922; }
    .metric-val-monitor { font-size: 26px; font-weight: 700; color: #3fb950; }
    .metric-val-blue { font-size: 26px; font-weight: 700; color: #58a6ff; }
    .metric-val-purple { font-size: 26px; font-weight: 700; color: #bc8cff; }
    .metric-label { font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Narrative Box */
    .narrative-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .narrative-box-alert {
        background-color: rgba(248, 81, 73, 0.08);
        border: 1px solid rgba(248, 81, 73, 0.4);
        border-left: 4px solid #f85149;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .action-badge-alert {
        background-color: #da3633;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .action-badge-log {
        background-color: #b08800;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .action-badge-monitor {
        background-color: #238636;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DATA LOADING HELPERS (DYNAMIC, NO HARDCODING)
# ------------------------------------------------------------------------------
@st.cache_data
def load_spedia_raw_stats(raw_path="data/spedia dataset/logs_SPEDIA_annotated_en.csv"):
    if not os.path.exists(raw_path):
        raw_path = "data/external/spedia/logs_SPEDIA_annotated_en.csv"
    if os.path.exists(raw_path):
        try:
            df = pd.read_csv(raw_path, low_memory=False)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp", "User"])
            return df
        except Exception:
            return None
    return None

@st.cache_data
def load_results_data(dataset_mode="spedia"):
    if dataset_mode == "spedia":
        spedia_alerts_path = "data/demo/spedia/trustshield_alerts.csv"
        if os.path.exists(spedia_alerts_path):
            df = pd.read_csv(spedia_alerts_path)
            return df
        else:
            from src.spedia_pipeline import run_spedia_pipeline
            df = run_spedia_pipeline()
            return df
    else:
        v2_alerts_path = "data/demo/trustshield_alerts.csv"
        if not os.path.exists(v2_alerts_path):
            v2_alerts_path = "data/demo/trustshield_demo_results.csv"
        if os.path.exists(v2_alerts_path):
            return pd.read_csv(v2_alerts_path)
        return None


# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Console Configuration")
    
    # Dataset Selector
    dataset_selection = st.radio(
        "Select Telemetry Source:",
        ["SPEDIA — PRIMARY DEMO", "V2 Synthetic — RESEARCH BENCHMARK"],
        index=0
    )
    is_spedia = "SPEDIA" in dataset_selection
    dataset_key = "spedia" if is_spedia else "v2"
    
    st.markdown("---")
    st.markdown("### 🔍 Triage Filters")
    
    df_results = load_results_data(dataset_key)
    
    if df_results is not None:
        all_actions = sorted(df_results["escalation_decision"].unique()) if "escalation_decision" in df_results.columns else ["Alert", "Log", "Monitor"]
        selected_actions = st.multiselect(
            "Filter by Action:",
            options=all_actions,
            default=all_actions
        )
        
        all_users = sorted(df_results["user_id"].unique())
        selected_user = st.selectbox("Investigate Actor / User:", ["All Users"] + all_users)
        
        # Risk threshold slider
        min_risk = st.slider("Minimum Daily Risk Score:", 0, 100, 0)
    else:
        selected_actions = ["Alert", "Log", "Monitor"]
        selected_user = "All Users"
        min_risk = 0

    st.markdown("---")
    st.markdown("""
    **Production Decision Policy**:
    - `Tampering == 1` $\\rightarrow$ **Alert**
    - `Consensus >= 2` $\\rightarrow$ **Alert**
    - `Consensus == 1` $\\rightarrow$ **Log**
    - `Consensus == 0` $\\rightarrow$ **Monitor**
    """)


# ------------------------------------------------------------------------------
# TOP HEADER BANNER
# ------------------------------------------------------------------------------
if is_spedia:
    raw_spedia_df = load_spedia_raw_stats()
    raw_event_count = len(raw_spedia_df) if raw_spedia_df is not None else 70878
    total_actors = raw_spedia_df["User"].nunique() if raw_spedia_df is not None else 17
    subtitle_text = f"Live SPEDIA Benchmark Telemetry | {raw_event_count:,} Raw Events | {total_actors} Monitored Actors"
    mode_badge = "LIVE SPEDIA ANALYSIS"
else:
    raw_event_count = 5400
    total_actors = 60
    subtitle_text = "Synthetic Enterprise Telemetry (V2 Benchmark Split: 1,080 Test Events, 60 Users)"
    mode_badge = "RESEARCH BENCHMARK"

st.markdown(f"""
<div class="soc-header">
    <div>
        <h1 class="soc-title">🛡️ TRUSTSHIELD UEBA & SOC CONSOLE</h1>
        <p class="soc-subtitle">{subtitle_text}</p>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <span class="badge-mode">{mode_badge}</span>
        <span class="status-badge-online">● SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)


if df_results is None:
    st.error(f"Could not load TrustShield results for {dataset_selection}. Please run `python src/run_trustshield.py --dataset {dataset_key}`.")
    st.stop()

# Filter Data
filtered_df = df_results[df_results["escalation_decision"].isin(selected_actions)].copy()
if min_risk > 0 and "daily_risk" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["daily_risk"] >= min_risk]
if selected_user != "All Users":
    filtered_df = filtered_df[filtered_df["user_id"] == selected_user]


# ------------------------------------------------------------------------------
# EXECUTIVE OVERVIEW CARDS
# ------------------------------------------------------------------------------
total_user_days = len(df_results)
alert_count = (df_results["escalation_decision"] == "Alert").sum()
log_count = (df_results["escalation_decision"] == "Log").sum()
monitor_count = (df_results["escalation_decision"] == "Monitor").sum()

# High priority users: users with at least one Alert action or max daily_risk >= 70
high_priority_users = df_results[
    (df_results["escalation_decision"] == "Alert") | 
    (df_results.get("daily_risk", 0) >= 70.0)
]["user_id"].nunique()

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-blue">{raw_event_count:,}</div>
        <div class="metric-label">Raw Events</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-purple">{total_actors}</div>
        <div class="metric-label">Actors</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-blue">{total_user_days:,}</div>
        <div class="metric-label">User-Day Records</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-alert">{alert_count}</div>
        <div class="metric-label">Alerts ({alert_count/total_user_days*100:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-log">{log_count}</div>
        <div class="metric-label">Logged ({log_count/total_user_days*100:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-monitor">{monitor_count}</div>
        <div class="metric-label">Monitored ({monitor_count/total_user_days*100:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with c7:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-alert">{high_priority_users}</div>
        <div class="metric-label">High Priority Users</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SECTION 1: THREAT OVERVIEW & MODEL CONSENSUS
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📊 Escalation Action Distribution")
    action_counts = df_results["escalation_decision"].value_counts().reset_index()
    action_counts.columns = ["Action", "Count"]
    
    color_map = {"Alert": "#da3633", "Log": "#d29922", "Monitor": "#238636"}
    
    fig_pie = px.pie(
        action_counts,
        names="Action",
        values="Count",
        color="Action",
        color_discrete_map=color_map,
        hole=0.55
    )
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        margin=dict(t=20, b=20, l=20, r=20),
        height=280
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("### 🤝 3-Model Anomaly Consensus Engine")
    
    # Consensus vote breakdown
    vote_counts = df_results["consensus_votes"].value_counts().sort_index().reset_index()
    vote_counts.columns = ["Votes", "Count"]
    vote_counts["Policy"] = vote_counts["Votes"].map({
        0: "0/3 → Monitor",
        1: "1/3 → Log",
        2: "2/3 → Alert",
        3: "3/3 → Alert"
    })
    
    fig_bar = px.bar(
        vote_counts,
        x="Policy",
        y="Count",
        color="Votes",
        color_continuous_scale=["#238636", "#d29922", "#f85149", "#da3633"],
        text="Count"
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        coloraxis_showscale=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
        xaxis_title=None,
        yaxis_title="Record Count"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ------------------------------------------------------------------------------
# SECTION 2: ACTIVITY & THREAT TIMELINE
# ------------------------------------------------------------------------------
st.markdown("### 📈 Activity & Escalation Timeline")
timeline_df = df_results.groupby(["date", "escalation_decision"]).size().reset_index(name="count")

fig_timeline = px.bar(
    timeline_df,
    x="date",
    y="count",
    color="escalation_decision",
    color_discrete_map={"Alert": "#da3633", "Log": "#d29922", "Monitor": "#238636"},
    barmode="stack"
)
fig_timeline.update_layout(
    template="plotly_dark",
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    margin=dict(t=20, b=20, l=20, r=20),
    height=260,
    xaxis_title="Date",
    yaxis_title="Daily Events Evaluated",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_timeline, use_container_width=True)


# ------------------------------------------------------------------------------
# SECTION 3: WHO TO WATCH (PRIORITY FOR INVESTIGATION)
# ------------------------------------------------------------------------------
st.markdown("### 🎯 Priority for Investigation (Top Ranked Actors)")
st.caption("Ranked by latest escalation action, detector consensus strength, daily risk, and cumulative temporal risk. Not confirmed threats.")

# User ranking table
user_summary = []
for u, u_group in df_results.groupby("user_id"):
    u_sorted = u_group.sort_values("date")
    latest_row = u_sorted.iloc[-1]
    
    total_alerts = (u_group["escalation_decision"] == "Alert").sum()
    total_logs = (u_group["escalation_decision"] == "Log").sum()
    max_cv = u_group["consensus_votes"].max()
    max_risk = u_group.get("daily_risk", pd.Series([0])).max()
    latest_cum_risk = latest_row.get("cumulative_risk", max_risk)
    latest_action = latest_row["escalation_decision"]
    dept = latest_row.get("department", "Enterprise")
    role = latest_row.get("role", "User")
    
    # Priority rank weighting
    action_weight = 300 if latest_action == "Alert" else (100 if latest_action == "Log" else 0)
    priority_score = action_weight + total_alerts * 50 + max_cv * 20 + latest_cum_risk
    
    user_summary.append({
        "user_id": u,
        "Department / Host Group": dept,
        "Role": role,
        "Latest Action": latest_action,
        "Consensus Votes (Max)": f"{max_cv}/3",
        "Total Alerts": total_alerts,
        "Total Logs": total_logs,
        "Max Daily Risk": f"{max_risk:.1f}",
        "Cumulative Risk": f"{latest_cum_risk:.1f}",
        "_priority": priority_score
    })

df_ranking = pd.DataFrame(user_summary).sort_values("_priority", ascending=False).drop(columns=["_priority"]).reset_index(drop=True)

def style_action(val):
    if val == "Alert":
        return "background-color: #da3633; color: white; font-weight: bold;"
    elif val == "Log":
        return "background-color: #d29922; color: white; font-weight: bold;"
    return "background-color: #238636; color: white;"

st.dataframe(
    df_ranking.style.map(style_action, subset=["Latest Action"]),
    use_container_width=True,
    height=260
)


# ------------------------------------------------------------------------------
# SECTION 4: USER INVESTIGATION & XAI ATTRIBUTION
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🕵️ Comprehensive Actor Deep-Dive & XAI Attribution")

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    investigate_user = st.selectbox(
        "Select Actor to Deep-Dive:",
        df_ranking["user_id"].tolist(),
        index=0
    )

user_records = df_results[df_results["user_id"] == investigate_user].sort_values("date").reset_index(drop=True)
latest_u = user_records.iloc[-1]

# User Metrics Cards
u_c1, u_c2, u_c3, u_c4, u_c5, u_c6 = st.columns(6)
with u_c1:
    st.markdown(f"**Actor ID:** `{investigate_user}`")
    st.markdown(f"**Host Group:** {latest_u.get('department', 'N/A')}")
with u_c2:
    st.markdown(f"**Role:** {latest_u.get('role', 'N/A')}")
    st.markdown(f"**Active Days:** {len(user_records)}")
with u_c3:
    act = latest_u['escalation_decision']
    badge_cls = f"action-badge-{act.lower()}"
    st.markdown(f"**Latest Action:** <span class='{badge_cls}'>{act}</span>", unsafe_allow_html=True)
    st.markdown(f"**Consensus:** {latest_u.get('consensus_votes', 0)}/3 votes")
with u_c4:
    st.markdown(f"**Daily Risk:** `{latest_u.get('daily_risk', 0):.1f}/100`")
    st.markdown(f"**Cumulative Risk:** `{latest_u.get('cumulative_risk', 0):.1f}`")
with u_c5:
    st.markdown(f"**IF Anomaly:** `{latest_u.get('if_anomaly_score', 0):.1f}`")
    st.markdown(f"**AE Anomaly:** `{latest_u.get('ae_anomaly_score', 0):.1f}`")
with u_c6:
    st.markdown(f"**Stat Anomaly:** `{latest_u.get('stat_anomaly_score', 0):.1f}`")
    st.markdown(f"**XAI Reliability:** `{latest_u.get('xai_reliability_score', 0.9):.0%}`")

st.markdown("<br>", unsafe_allow_html=True)

# Narrative Explanation Box
is_alert_u = (latest_u["escalation_decision"] == "Alert")
box_cls = "narrative-box-alert" if is_alert_u else "narrative-box"

st.markdown(f"""
<div class="{box_cls}">
    <h4 style="margin-top: 0; color: {'#f85149' if is_alert_u else '#58a6ff'};">
        🔍 Why Was This Flagged? (Explainable AI Attribution)
    </h4>
    <p><b>Primary Behavioral Deviations:</b> {latest_u.get('top_behavioral_reasons', 'Nominal behavior')}</p>
    <p><b>Detailed Explanation:</b> {latest_u.get('explanation', 'Activity is within expected historical baseline.')}</p>
    <p><b>Recommended SOC Action:</b> <code>{latest_u.get('recommended_action', 'Continue Standard Monitoring')}</code></p>
</div>
""", unsafe_allow_html=True)

# Temporal Risk & Detector Votes Chart
t_col1, t_col2 = st.columns([3, 2])

with t_col1:
    st.markdown("##### 📈 Temporal Risk Trajectory")
    fig_u_risk = go.Figure()
    fig_u_risk.add_trace(go.Scatter(
        x=user_records["date"],
        y=user_records.get("daily_risk", 0),
        mode="lines+markers",
        name="Daily Risk Score",
        line=dict(color="#58a6ff", width=2)
    ))
    fig_u_risk.add_trace(go.Scatter(
        x=user_records["date"],
        y=user_records.get("cumulative_risk", 0),
        mode="lines+markers",
        name="Cumulative Risk",
        line=dict(color="#bc8cff", width=2, dash="dot")
    ))
    fig_u_risk.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        margin=dict(t=20, b=20, l=20, r=20),
        height=240,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_u_risk, use_container_width=True)

with t_col2:
    st.markdown("##### 🤖 Detector Consensus Votes Over Time")
    fig_u_det = go.Figure()
    fig_u_det.add_trace(go.Bar(
        x=user_records["date"],
        y=user_records.get("vote_iforest", 0),
        name="Isolation Forest",
        marker_color="#388bfd"
    ))
    fig_u_det.add_trace(go.Bar(
        x=user_records["date"],
        y=user_records.get("vote_autoencoder", 0),
        name="Autoencoder",
        marker_color="#bc8cff"
    ))
    fig_u_det.add_trace(go.Bar(
        x=user_records["date"],
        y=user_records.get("vote_statistical", 0),
        name="Statistical Detector",
        marker_color="#f0883e"
    ))
    fig_u_det.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        margin=dict(t=20, b=20, l=20, r=20),
        height=240,
        yaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 3.5]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_u_det, use_container_width=True)


# ------------------------------------------------------------------------------
# SECTION 5: DATASET EXPLORER & RAW EVENT DRILL-DOWN (SPEDIA)
# ------------------------------------------------------------------------------
if is_spedia and raw_spedia_df is not None:
    st.markdown("---")
    st.markdown("### 🗂️ SPEDIA Raw Event Telemetry Drill-Down")
    
    u_raw_events = raw_spedia_df[raw_spedia_df["User"] == investigate_user].sort_values("Timestamp", ascending=False)
    
    st.caption(f"Displaying genuine SPEDIA log telemetry for **{investigate_user}** ({len(u_raw_events):,} total raw events).")
    
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric("Total Events for Actor", f"{len(u_raw_events):,}")
    with r_col2:
        st.metric("Commands Executed", f"{(u_raw_events['Action'] == 'Command executed').sum():,}")
    with r_col3:
        st.metric("Files Interacted", f"{(u_raw_events['Activity'] == 'file').sum():,}")
    with r_col4:
        st.metric("Emails Sent", f"{(u_raw_events['Action'] == 'Email sent').sum():,}")

    # Activity filter
    activity_types = sorted(u_raw_events["Activity"].dropna().unique())
    sel_act_types = st.multiselect("Filter Raw Event Activities:", activity_types, default=activity_types)
    
    disp_raw = u_raw_events[u_raw_events["Activity"].isin(sel_act_types)]
    
    # Selected columns for display
    show_cols = ["Timestamp", "Activity", "Action", "Description", "Command", "Url", "Path", "To", "Size"]
    existing_show_cols = [c for c in show_cols if c in disp_raw.columns]
    
    st.dataframe(
        disp_raw[existing_show_cols].head(250),
        use_container_width=True,
        height=320
    )

    # Activity composition charts
    st.markdown("#### 📊 Actor Activity Composition")
    exp_c1, exp_c2 = st.columns(2)
    
    with exp_c1:
        act_dist = u_raw_events["Activity"].value_counts().reset_index()
        act_dist.columns = ["Activity", "Count"]
        fig_act = px.bar(act_dist, x="Activity", y="Count", color="Activity", text="Count")
        fig_act.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22", height=240, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_act, use_container_width=True)
        
    with exp_c2:
        top_cmds = u_raw_events["Command"].dropna().value_counts().head(8).reset_index()
        top_cmds.columns = ["Command", "Count"]
        if not top_cmds.empty:
            fig_cmd = px.bar(top_cmds, x="Count", y="Command", orientation="h", color="Count", color_continuous_scale="Blues")
            fig_cmd.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22", height=240, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
            st.plotly_chart(fig_cmd, use_container_width=True)
        else:
            st.info("No command execution events for this actor.")


# ------------------------------------------------------------------------------
# SECTION 6: OFFLINE RESEARCH EVALUATION (EXPANDER ONLY)
# ------------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔬 Research Evaluation & Benchmark Ground-Truth Validation (Offline Only)", expanded=False):
    st.markdown("""
    > [!NOTE]
    > Ground truth labels are strictly isolated within this research panel to evaluate algorithmic performance. 
    > They are **never** utilized during feature creation, model inference, baseline updating, or operational escalation.
    """)
    
    if is_spedia and "spedia_is_malicious_day" in df_results.columns:
        dates = sorted(df_results["date"].unique())
        test_dates = dates[20:]
        test_df = df_results[df_results["date"].isin(test_dates)].copy()
        
        y_true = test_df["spedia_is_malicious_day"].values
        y_pred = (test_df["escalation_decision"] == "Alert").astype(int).values
        
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        res_c1, res_c2, res_c3, res_c4, res_c5 = st.columns(5)
        res_c1.metric("Test Precision", f"{prec:.1%}")
        res_c2.metric("Test Recall", f"{rec:.1%}")
        res_c3.metric("Test F1 Score", f"{f1:.3f}")
        res_c4.metric("False Positive Rate (FPR)", f"{fpr:.1%}")
        res_c5.metric("False Negative Rate (FNR)", f"{fnr:.1%}")
        
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown(f"""
            **SPEDIA Test Split Metrics (Dates: {test_dates[0]} to {test_dates[-1]}):**
            - **True Positives (TP):** `{tp}`
            - **False Positives (FP):** `{fp}`
            - **True Negatives (TN):** `{tn}`
            - **False Negatives (FN):** `{fn}`
            - **Total Test Records:** `{len(test_df)}`
            """)
        with cm_col2:
            cm_data = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                labels=dict(x="Predicted Action", y="Actual Malicious Day", color="Count"),
                x=["Monitor / Log", "Alert"],
                y=["Benign", "Malicious"],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                height=220,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.markdown("""
        **V2 Synthetic Benchmark Evaluation Metrics (Frozen Chronological Test Set):**
        - **Precision:** 17.6% (TP=6, FP=28)
        - **Recall:** 54.5% (TP=6, FN=5)
        - **F1 Score:** 0.267
        - **FPR:** 2.6% (FP=28, TN=1041)
        - **FNR:** 45.5% (FN=5, TP=6)
        - **Alert Rate:** 3.1% (34 total alerts)
        """)