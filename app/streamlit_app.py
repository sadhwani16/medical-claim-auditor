"""
AyushAudit AI — PM-JAY Medical Claim Auditor
Streamlit demo app. Run: streamlit run app/streamlit_app.py
"""

import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.pdf_extractor import extract_from_bytes
from utils.translator import translate_to_english, detect_language
from utils.vector_store import retrieve_relevant_rules, load_index, build_index
from utils.auditor import AuditEngine
from config import VECTOR_STORE_PATH, FRAUD_THRESHOLD

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AyushAudit AI — PM-JAY Claim Auditor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f5f7fb; }
    .stApp header { background-color: #1a1a2e; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;
    }
    .approved-banner {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white; border-radius: 12px; padding: 1.5rem 2rem;
        font-size: 1.6rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 16px rgba(46,125,50,0.35);
    }
    .flagged-banner {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        color: white; border-radius: 12px; padding: 1.5rem 2rem;
        font-size: 1.6rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 16px rgba(198,40,40,0.35);
    }
    .step-done { color: #2e7d32; font-weight: 600; }
    .step-pending { color: #aaa; }
    .rule-card {
        background: #fffde7; border-left: 4px solid #f9a825;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.4rem 0;
        font-size: 0.88rem;
    }
    .violation-item {
        background: #ffebee; border-left: 4px solid #c62828;
        padding: 0.6rem 1rem; border-radius: 6px; margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────
for key, default in {
    "history": [],
    "claims_processed": 0,
    "fraud_count": 0,
    "audit_engine": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 AyushAudit AI")
    st.caption("PM-JAY Agentic Claim Auditor")
    st.divider()

    st.subheader("⚙️ Configuration")
    llm_provider = st.selectbox(
        "LLM Backend",
        ["mock (offline demo)", "sarvam", "anthropic", "openai"],
        index=0,
    )
    api_key_input = st.text_input("API Key (if needed)", type="password")
    fraud_threshold = st.slider(
        "Fraud Flag Threshold", 0.3, 0.9, FRAUD_THRESHOLD, 0.05
    )

    st.divider()
    st.subheader("📊 Session Stats")
    col_a, col_b = st.columns(2)
    col_a.metric("Processed", st.session_state.claims_processed)
    col_b.metric("Flagged", st.session_state.fraud_count)
    if st.session_state.claims_processed > 0:
        rate = st.session_state.fraud_count / st.session_state.claims_processed
        st.metric("Fraud Rate", f"{rate:.0%}")

    st.divider()
    st.subheader("📁 Rules Index")
    rules_file = st.file_uploader(
        "Upload PM-JAY Rulebook (TXT/PDF)", type=["txt", "pdf"], key="rules_upload"
    )
    if rules_file and st.button("Build Vector Index"):
        with st.spinner("Indexing PM-JAY rules…"):
            rules_text = (
                rules_file.read().decode("utf-8")
                if rules_file.name.endswith(".txt")
                else extract_from_bytes(rules_file.read(), rules_file.name)
            )
            build_index(rules_text, VECTOR_STORE_PATH)
            st.success("Index built ✓")

    if st.button("Load Default Index"):
        result = load_index(VECTOR_STORE_PATH)
        st.success("Index loaded ✓" if result else "No saved index found — upload a rulebook first.")

    if st.session_state.history and st.button("🗑 Clear History"):
        st.session_state.history = []
        st.session_state.claims_processed = 0
        st.session_state.fraud_count = 0
        st.rerun()

# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────
st.markdown("# 🏥 AyushAudit AI")
st.markdown("**Agentic Medical Claim Auditor for Ayushman Bharat PM-JAY**")
st.divider()

tabs = st.tabs(["🔍 Audit Claim", "📋 Batch History", "📊 Analytics"])

# ══════════════════════════════════════════════
# TAB 1 — Single Claim Audit
# ══════════════════════════════════════════════
with tabs[0]:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Upload Claim Document")
        uploaded = st.file_uploader(
            "PDF, Image, or TXT discharge summary",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            key="claim_upload",
        )
        paste_text = st.text_area(
            "…or paste claim text directly",
            height=180,
            placeholder="Patient Name: Ramu Kumar\nDiagnosis: Cataract (Right Eye)\n…",
        )

        run_audit = st.button("🚀 Run Audit", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Processing Pipeline")
        step_ph = st.empty()

        def render_steps(steps_done: int):
            labels = [
                "1 · Extract text from document",
                "2 · Detect & translate language",
                "3 · Retrieve PM-JAY rules (RAG)",
                "4 · AI audit decision",
            ]
            lines = []
            for i, lbl in enumerate(labels):
                if i < steps_done:
                    lines.append(f'<div class="step-done">✅ {lbl}</div>')
                elif i == steps_done:
                    lines.append(f'<div style="color:#1565c0;font-weight:600">⏳ {lbl}</div>')
                else:
                    lines.append(f'<div class="step-pending">◦ {lbl}</div>')
            step_ph.markdown("\n".join(lines), unsafe_allow_html=True)

        render_steps(0)

    # ── Run the audit pipeline ──
    if run_audit:
        claim_text = ""
        if uploaded:
            with st.spinner("Extracting text…"):
                claim_text = extract_from_bytes(uploaded.read(), uploaded.name)
        elif paste_text.strip():
            claim_text = paste_text.strip()
        else:
            st.warning("Please upload a document or paste claim text.")
            st.stop()

        render_steps(1)
        time.sleep(0.3)

        # Step 2 — Translate
        with st.spinner("Detecting language…"):
            translation_result = translate_to_english(claim_text)
        translated_text = translation_result["translated_text"]
        render_steps(2)
        time.sleep(0.3)

        # Step 3 — Retrieve rules
        with st.spinner("Retrieving PM-JAY rules…"):
            rules = retrieve_relevant_rules(translated_text, k=5)
        render_steps(3)
        time.sleep(0.3)

        # Step 4 — Audit
        provider = llm_provider.split(" ")[0]
        engine = AuditEngine(
            provider=provider,
            api_key=api_key_input or "",
        )
        with st.spinner("AI is auditing the claim…"):
            audit = engine.audit_claim(translated_text, rules)
        render_steps(4)

        # ── Update stats ──
        st.session_state.claims_processed += 1
        if audit["status"] == "FLAGGED":
            st.session_state.fraud_count += 1
        st.session_state.history.append({
            "claim_snippet": claim_text[:80] + "…",
            "language": translation_result["source_language_name"],
            "status": audit["status"],
            "risk_score": audit.get("risk_score", 0),
            "reason": audit.get("reason", ""),
            "latency": audit.get("latency_seconds", "—"),
        })

        st.divider()

        # ── Verdict banner ──
        if audit["status"] == "APPROVED":
            st.markdown(
                f'<div class="approved-banner">✅ CLAIM APPROVED'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Risk Score: {audit["risk_score"]:.0%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="flagged-banner">🚨 CLAIM FLAGGED FOR FRAUD'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Risk Score: {audit["risk_score"]:.0%}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**Reason:** {audit.get('reason', '')}")
        st.markdown(f"**Cited PM-JAY Rule:** _{audit.get('cited_rule', 'N/A')}_")
        st.markdown(f"**Recommended Action:** {audit.get('recommended_action', '')}")
        st.caption(f"Audited in {audit.get('latency_seconds', '—')}s · Model: {audit.get('llm_model', provider)}")

        if audit.get("violations"):
            st.subheader("⚠️ Violations Detected")
            for v in audit["violations"]:
                st.markdown(f'<div class="violation-item">⛔ {v}</div>', unsafe_allow_html=True)

        # ── Expandable details ──
        if translation_result["translation_needed"]:
            with st.expander(f"🌐 Translation ({translation_result['source_language_name']} → English)"):
                c1, c2 = st.columns(2)
                c1.markdown("**Original**")
                c1.text(translation_result["original_text"][:1000])
                c2.markdown(f"**English** _(via {translation_result['method']})_")
                c2.text(translation_result["translated_text"][:1000])

        with st.expander("📚 Retrieved PM-JAY Rules"):
            for i, rule in enumerate(rules, 1):
                st.markdown(f'<div class="rule-card"><b>Rule {i}</b><br>{rule}</div>', unsafe_allow_html=True)

        with st.expander("📄 Full Audit JSON"):
            st.json(audit)

# ══════════════════════════════════════════════
# TAB 2 — Batch History
# ══════════════════════════════════════════════
with tabs[1]:
    st.subheader("Audit History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df.style.applymap(
                lambda v: "background-color: #ffcdd2" if v == "FLAGGED" else "background-color: #c8e6c9",
                subset=["status"],
            ),
            use_container_width=True,
        )
    else:
        st.info("No claims audited yet. Go to the **Audit Claim** tab to get started.")

# ══════════════════════════════════════════════
# TAB 3 — Analytics
# ══════════════════════════════════════════════
with tabs[2]:
    st.subheader("Analytics")
    if len(st.session_state.history) >= 2:
        df = pd.DataFrame(st.session_state.history)

        c1, c2 = st.columns(2)
        with c1:
            counts = df["status"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker_colors=["#2e7d32", "#c62828"],
                hole=0.45,
            ))
            fig.update_layout(title="Approved vs Flagged", height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure(go.Bar(
                x=df.index.tolist(),
                y=df["risk_score"].tolist(),
                marker_color=["#c62828" if s == "FLAGGED" else "#2e7d32" for s in df["status"]],
            ))
            fig2.update_layout(
                title="Risk Scores per Claim",
                xaxis_title="Claim #",
                yaxis_title="Risk Score",
                height=300,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        lang_counts = df["language"].value_counts()
        st.markdown("**Claims by Language**")
        st.bar_chart(lang_counts)
    else:
        st.info("Audit at least 2 claims to see analytics.")
