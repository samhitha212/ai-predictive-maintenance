"""Launch with: streamlit run app.py."""

import json

import pandas as pd
import streamlit as st

from blockchain import Blockchain
from model import FEATURES, PredictiveMaintenanceModel

st.set_page_config(
    page_title="Sentinel | Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .risk-card {padding: 1.7rem; border-radius: 18px; margin-bottom: 1rem;}
    .risk-card h2 {margin: 0; color: inherit;}
    .risk-value {font-size: 3.6rem; font-weight: 800; line-height: 1.2;}
    .risk-card p {margin-bottom: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    trained = PredictiveMaintenanceModel()
    trained.train()
    return trained


model = load_model()
if "blockchain" not in st.session_state:
    st.session_state.blockchain = Blockchain()
chain = st.session_state.blockchain

with st.sidebar:
    st.title("Sensor Control Center")
    st.caption("Interactive simulated telemetry; no physical sensor is connected.")
    machine_id = st.text_input("Machine ID", value="SERVER-001", max_chars=80)
    cpu = st.slider("CPU Utilization (%)", 0, 100, 48, 1)
    temperature = st.slider("Core Temperature (°C)", 30, 100, 58, 1)
    disk = st.slider("Disk Health Score", 0, 100, 92, 1)
    uptime = st.slider("Uptime Hours", 0, 20000, 2400, 100)
    st.divider()
    st.caption("Risk bands: Low < 30% · Medium 30–<70% · High ≥ 70%.")
    st.caption("Ledger survives widget reruns, not a new browser session or restart.")

telemetry = dict(zip(FEATURES, [cpu, temperature, disk, uptime]))
risk = model.predict_proba(telemetry)
if risk < 30:
    severity, background, foreground = "LOW", "#dcfce7", "#14532d"
    guidance = "Continue routine monitoring and scheduled maintenance."
elif risk < 70:
    severity, background, foreground = "MEDIUM", "#fef3c7", "#78350f"
    guidance = "Inspect cooling, disk diagnostics, and recent workload changes."
else:
    severity, background, foreground = "HIGH", "#fee2e2", "#7f1d1d"
    guidance = "Prioritize inspection, verify backups, and plan safe maintenance."

st.title("Sentinel: AI-Driven Predictive Maintenance")
st.caption("Telemetry intelligence • Random Forest inference • SHA-256 audit trail")
inference_tab, ledger_tab = st.tabs(
    ["AI Telemetry Inference", "Blockchain Ledger Explorer"]
)

with inference_tab:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            f"""
            <div class="risk-card" style="background:{background};color:{foreground}">
                <h2>{severity} RISK</h2>
                <div class="risk-value">{risk:.2f}%</div>
                <strong>Estimated synthetic failure-class probability</strong>
                <p>{guidance}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, risk / 100)))
        st.caption(
            "The model returns an unrounded percentage; the card shows two decimals. "
            "This is not a calibrated real-world failure probability."
        )
        if st.button(
            "🔒 Commit Alert to Immutable Blockchain Ledger",
            type="primary",
            disabled=not machine_id.strip(),
        ):
            try:
                with st.spinner("Validating the ledger and mining a new block..."):
                    block = chain.add_prediction_block(machine_id, telemetry, risk)
                st.success(f"Block #{block.index} committed for {block.machine_id}.")
                st.code(block.hash, language="text")
            except ValueError as error:
                st.error(str(error))
    with right:
        st.subheader("Current Telemetry Snapshot")
        a, b = st.columns(2)
        a.metric("CPU utilization", f"{cpu}%")
        b.metric("Core temperature", f"{temperature} °C")
        a.metric("Disk health", f"{disk}/100")
        b.metric("Uptime", f"{uptime:,} h")
        st.subheader("Global Feature Importance")
        importance = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": model.classifier.feature_importances_,
            }
        ).set_index("Feature")
        st.bar_chart(importance)
        st.caption("Global impurity-based importance is not a causal explanation.")

    st.divider()
    st.subheader("Held-Out Synthetic Validation")
    metrics = model.evaluate()
    f1, precision, recall = st.columns(3)
    f1.metric("F1-score", f"{metrics['f1_score']:.3f}")
    precision.metric("Precision", f"{metrics['precision']:.3f}")
    recall.metric("Recall", f"{metrics['recall']:.3f}")
    summary = model.training_summary
    st.caption(
        f"{summary['training_samples']} training / {summary['test_samples']} test rows "
        f"• Failure prevalence: {summary['failure_prevalence']:.1%} "
        "• Stratified holdout • Class-weighted Random Forest"
    )

with ledger_tab:
    st.subheader("Cryptographic Prediction Ledger")
    st.info(
        "This local ledger is tamper-evident, not absolutely immutable. "
        "An attacker controlling the process can rewrite and re-mine it."
    )
    count, work, algorithm = st.columns(3)
    count.metric("Prediction blocks", len(chain.ledger) - 1)
    work.metric("PoW leading zeros", chain.DIFFICULTY)
    algorithm.metric("Hash algorithm", "SHA-256")
    records = chain.to_records()
    table = pd.DataFrame(records)
    table["telemetry_snapshot"] = table["telemetry_snapshot"].map(
        lambda snapshot: json.dumps(snapshot, sort_keys=True)
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    if st.button("⚡ Run Ledger Integrity Audit"):
        if chain.is_chain_valid():
            st.success("Integrity verified: genesis, contents, hashes, links, and work match.")
        else:
            st.error("TAMPER ALERT: the ledger is corrupted or has been modified.")
    st.download_button(
        "Download ledger as JSON",
        data=json.dumps(records, indent=2, allow_nan=False),
        file_name="prediction_ledger.json",
        mime="application/json",
    )
    st.caption("UTC timestamps • Genesis is fixed and exempt from proof-of-work.")
