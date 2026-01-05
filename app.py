# app.py
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv() 
# st.write("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))

from pipeline.exporter import build_zip_from_bundle
from pipeline.generator import regenerate_with_feedback
import pipeline.generator as gen

# Import individual agent runners (progressive execution)
from pipeline.orchestrator import run_agent
 # 👈 FORCE load .env

# optional debug (remove later)
# st.write("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))


# =========================================================
# Session state initialization
# =========================================================
if "agent_output" not in st.session_state:
    st.session_state.agent_output = None

if "approved" not in st.session_state:
    st.session_state.approved = False

if "feedback" not in st.session_state:
    st.session_state.feedback = ""

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Agentic API Code Generator",
    layout="wide",
)

st.title(" Agentic API Code Generator")
st.caption("Parser → RAG → Planner → Code Generator → Review → Rewrite")

# =========================================================
# User input
# =========================================================
user_prompt = st.text_area(
    "Describe what you want to build",
    height=120,
    placeholder="e.g. Build a dashboard to list AWD inbound shipments with filters by status and date...",
)

col1, col2 = st.columns(2)

with col1:
    frontend_choice = st.selectbox(
        "Frontend framework",
        ["React", "Vue", "Angular", "Svelte", "Next.js"],
        index=0,
    )

with col2:
    backend_choice = st.selectbox(
        "Backend framework",
        ["FastAPI", "Django", "Flask", "Express.js", "Spring Boot"],
        index=0,
    )

run_btn = st.button("🚀 Generate Code")

# =========================================================
# Progressive agent execution
# =========================================================
# =========================================================
# Run agent (two-phase execution)
# =========================================================
if run_btn:
    if not user_prompt.strip():
        st.warning("Please enter a description.")
        st.stop()

    st.session_state.current_prompt = user_prompt
    st.session_state.approved = False
    st.session_state.feedback = ""
    st.session_state.agent_output = None

    # -------- Phase 1: Planning (parser + rag + planner) --------
    with st.spinner("🔍 Understanding your request and planning integration..."):
        try:
            # run everything but codegen
            out = run_agent(
                user_prompt=user_prompt,
                frontend_choice=frontend_choice,
                backend_choice=backend_choice,
            )
        except Exception as e:
            st.error("Failed during planning stage.")
            st.exception(e)
            st.stop()

    st.info(" Plan ready. Generating code next...")

    # -------- Phase 2: Code generation --------
    with st.spinner("💻 Generating code (this may take a moment)..."):
        st.session_state.agent_output = out

    st.success(" Code generation complete!")
    st.rerun()

# =========================================================
# Display results (review + feedback loop)
# =========================================================
# =========================================================
# Display results (ALL agent outputs)
# =========================================================
if st.session_state.agent_output:
    out = st.session_state.agent_output

    # -----------------------------
    # Parsed Intent
    # -----------------------------
    st.subheader(" Parsed Intent")
    st.json(out["parsed"])

    # -----------------------------
    # RAG Results
    # -----------------------------
    st.subheader("📚 Retrieved APIs (RAG)")
    for i, api in enumerate(out["rag"]["results"], 1):
        with st.expander(f"{i}. {api['method']} {api['path']}"):
            st.write("**Score:**", api["score"])
            st.write("**Full URL:**", api["full_url"])
            st.write("**Operation ID:**", api["operationId"])
            st.write("**Parameters:**")
            st.json(api.get("parameters", []))

    # -----------------------------
    # Planner Output
    # -----------------------------
    st.subheader(" Code Plan")
    st.json(out["plan"])

    # -----------------------------
    # Generated Code
    # -----------------------------
    st.subheader(" Generated Code")

    tabs = st.tabs(["Frontend", "Backend", "Notes"])


    with tabs[0]:
        for fname, code in out["bundle"]["frontend_files"].items():
            st.markdown(f"### {fname}")
            st.code(code, language="tsx")

    with tabs[1]:
        for fname, code in out["bundle"]["backend_files"].items():
            st.markdown(f"### {fname}")
            st.code(code)

    with tabs[2]:
        st.write(out["bundle"]["notes"] or "No notes")

    # =========================================================
    # Review controls
    # =========================================================
    st.divider()
    st.subheader(" Review Generated Code")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("✅ Approve"):
            st.session_state.approved = True

            zip_bytes = build_zip_from_bundle(
                st.session_state.agent_output["bundle"]
            )

            st.success("Approved! Download your code below.")

            st.download_button(
                label=" Download Code (ZIP)",
                data=zip_bytes,
                file_name="agentic_output.zip",
                mime="application/zip",
            )

    with col_b:
        if st.button("✍️ Request Changes"):
            st.session_state.approved = False
            st.session_state.feedback = "PENDING"
            st.rerun()

    # =========================================================
    # Feedback + regeneration loop (NO replan)
    # =========================================================
    if st.session_state.feedback == "PENDING":
        feedback_text = st.text_area(
            "Describe what you want to change",
            height=120,
            placeholder=(
                "e.g. Add pagination, improve error handling, "
                "split backend routes into controllers..."
            ),
        )

        if st.button("🔁 Regenerate with Feedback"):
            if not gen.LAST_CODEGEN_PROMPT:
                st.error("No previous codegen prompt found.")
                st.stop()

            with st.spinner("Rewriting prompt and regenerating code..."):
                new_bundle = regenerate_with_feedback(
                    previous_prompt=gen.LAST_CODEGEN_PROMPT,
                    feedback=feedback_text,
                )

            st.session_state.agent_output["bundle"] = new_bundle.model_dump()
            st.session_state.feedback = ""
            st.session_state.approved = False

            st.success("Code regenerated. Please review again.")
            st.rerun()
