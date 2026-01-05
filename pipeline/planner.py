# pipeline/planner.py
from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# -----------------------------
# Allowed language choices
# -----------------------------
FrontendChoice = Literal["React", "Vue", "Angular", "Svelte", "Next.js"]
BackendChoice  = Literal["Express.js", "Django", "FastAPI", "Spring Boot", "Flask"]

DEFAULT_FRONTEND: FrontendChoice = "React"
DEFAULT_BACKEND:  BackendChoice  = "FastAPI"


# -----------------------------
# Planner input / output models
# -----------------------------
class CodeGenInput(BaseModel):
    user_prompt: str
    rag_results: List[Dict[str, Any]] = Field(
        ..., description="Top API candidates from RAG"
    )
    frontend_choice: Optional[FrontendChoice] = None
    backend_choice: Optional[BackendChoice] = None


class CodePlan(BaseModel):
    frontend: FrontendChoice
    backend: BackendChoice
    apis: List[Dict[str, Any]]
    generator_brief: str


# -----------------------------
# Helpers
# -----------------------------
def _compact_api_payload(r: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields needed by the generator."""
    return {
        "method": r.get("method"),
        "path": r.get("path"),
        "full_url": r.get("full_url"),
        "operationId": r.get("operationId"),
        "parameters": r.get("parameters") or [],
        "responses": r.get("responses") or [],
        "api_name": r.get("api_name"),
        "api_version": r.get("api_version"),
        "tags": r.get("tags") or [],
    }


def _build_generator_brief(user_prompt: str, apis: List[Dict[str, Any]]) -> str:
    """Short, model-friendly instruction for the code generator."""
    lines = [
        f"User intent: {user_prompt}",
        "Selected APIs:",
    ]
    for i, a in enumerate(apis, 1):
        params = ", ".join(
            [p["name"] for p in a.get("parameters", []) if p.get("name")]
        )
        lines.append(f"{i}. {a['method']} {a['path']} (params: {params})")

    lines.append(
        "Generate frontend + backend integration code. "
        "Frontend: forms & fetch calls. "
        "Backend: secure proxy, auth placeholders, basic validation."
    )
    return "\n".join(lines)


# -----------------------------
# Planner function
# -----------------------------
def codegen_planner_fn(payload: CodeGenInput) -> CodePlan:
    frontend = payload.frontend_choice or DEFAULT_FRONTEND
    backend = payload.backend_choice or DEFAULT_BACKEND

    # Take top-3 APIs from RAG
    top = payload.rag_results[:3]
    apis = [_compact_api_payload(r) for r in top]

    brief = _build_generator_brief(payload.user_prompt, apis)

    return CodePlan(
        frontend=frontend,
        backend=backend,
        apis=apis,
        generator_brief=brief,
    )


# -----------------------------
# LangChain runnable
# -----------------------------
codegen_planner_node = RunnableLambda(codegen_planner_fn)
