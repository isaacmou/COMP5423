"""Simple Streamlit UI for the skill-augmented search agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.agent.loop import SearchAgent  # noqa: E402
from src.config import load_env  # noqa: E402
from src.llm.siliconflow import SiliconFlowClient  # noqa: E402
from src.retrieval.unified import UnifiedRetriever  # noqa: E402

st.set_page_config(page_title="COMP5423 Search Agent", layout="wide")
load_env()

st.title("COMP5423 Skill-Augmented Search Agent")
st.caption("BrowseComp-Plus fixed corpus · DeepSeek-V4-Flash (SiliconFlow)")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "SiliconFlow API Key",
        type="password",
        value=os.environ.get("SILICONFLOW_API_KEY", ""),
        help="Not stored in the repository. Can also be set via SILICONFLOW_API_KEY.",
    )
    bm25_only = st.checkbox(
        "BM25 only (skip dense/instruct indexes)",
        value=False,
        help="Useful when dense indexes are not built yet.",
    )
    force_skill = st.selectbox(
        "Skill selection",
        options=["auto", "none", "multi-clue-decompose", "search-recovery"],
        index=0,
    )
    st.markdown(
        "Tools: `search` (unified), `visit`, `find_in_doc`. "
        "Skills load at runtime from `skills/*/SKILL.md`."
    )


@st.cache_resource(show_spinner=False)
def get_agent(bm25_only_flag: bool, api_key_value: str) -> SearchAgent:
    if api_key_value:
        os.environ["SILICONFLOW_API_KEY"] = api_key_value
    llm = SiliconFlowClient(api_key=api_key_value or None)
    retriever = UnifiedRetriever(
        enable_bm25=True,
        enable_dense=not bm25_only_flag,
        enable_instruct=not bm25_only_flag,
        lazy=True,
    )
    return SearchAgent(llm=llm, retriever=retriever)


question = st.text_area("Question", height=120, placeholder="Enter a question…")
run = st.button("Run agent", type="primary", disabled=not question.strip())

if run:
    if not (api_key or os.environ.get("SILICONFLOW_API_KEY")):
        st.error("Please provide SILICONFLOW_API_KEY in the sidebar or environment.")
        st.stop()

    status = st.empty()
    log_box = st.expander("Live progress", expanded=True)
    progress_lines: list[str] = []

    def on_progress(msg: str) -> None:
        progress_lines.append(msg)
        status.info(msg)
        log_box.code("\n".join(progress_lines[-40:]))

    try:
        agent = get_agent(bm25_only, api_key)
        kwargs = {"on_progress": on_progress}
        if force_skill == "none":
            kwargs["disable_skills"] = True
        elif force_skill != "auto":
            kwargs["force_skill"] = force_skill

        with st.spinner("Running iterative search agent…"):
            result = agent.run(question.strip(), **kwargs)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        st.stop()

    st.subheader("Final answer")
    st.success(result.answer or "(empty)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected skill", result.selected_skill or "none")
    col2.metric("Tool calls", result.tool_calls)
    col3.metric("Total tokens", result.total_tokens)

    st.subheader("Supporting evidence")
    if result.evidence:
        st.json(result.evidence)
    else:
        st.write("No evidence recorded.")

    st.subheader("Tool-use / trajectory")
    st.json(result.trajectory)

    st.download_button(
        "Download this prediction JSON",
        data=json.dumps(
            {
                "id": "ui-query",
                "question": question.strip(),
                "answer": result.answer,
                "trajectory": result.trajectory,
                "tool_calls": result.tool_calls,
                "total_tokens": result.total_tokens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file_name="prediction.json",
        mime="application/json",
    )
