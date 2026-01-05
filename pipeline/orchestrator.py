# pipeline/orchestrator.py
from __future__ import annotations

from typing import Dict, Any, Optional

from pipeline.core import parser_node  # GPT-4o-mini parser runnable
from pipeline.rag import rag_node      # Pinecone RAG runnable
from pipeline.planner import CodeGenInput, codegen_planner_node
from pipeline.generator import codegen_generate_node, CodeBundle


def run_agent(
    user_prompt: str,
    frontend_choice: str = "React",
    backend_choice: str = "FastAPI",
    tag_filter: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates: parser -> RAG -> planner -> generator
    Returns a dict that Streamlit can render easily.
    """

    # 1) Parser
    spec = parser_node.invoke(user_prompt)

    # 2) RAG (optionally filter by tag)
    # default rag_node doesn't accept tag_filter; call rag_node_fn style by re-invoking via rag_node's function
    # simplest: just use rag_node as-is for now
    rag_out = rag_node.invoke(spec)

    # 3) Planner
    codegen_input = CodeGenInput(
        user_prompt=user_prompt,
        rag_results=rag_out["results"],
        frontend_choice=frontend_choice,
        backend_choice=backend_choice,
    )
    plan = codegen_planner_node.invoke(codegen_input)

    # 4) Generator
    bundle: CodeBundle = codegen_generate_node.invoke(plan)

    return {
        "parsed": spec.model_dump(),
        "rag": rag_out,
        "plan": plan.model_dump(),
        "bundle": bundle.model_dump(),
    }

def run_parser(user_prompt: str):
    return parser_node.invoke(user_prompt)

def run_rag(parsed):
    return rag_node.invoke(parsed)

def run_planner(user_prompt, rag_results, frontend_choice, backend_choice):
    return codegen_planner_node.invoke(
        CodeGenInput(
            user_prompt=user_prompt,
            rag_results=rag_results,
            frontend_choice=frontend_choice,
            backend_choice=backend_choice,
        )
    )

def run_codegen(plan):
    return codegen_generate_node.invoke(plan)
