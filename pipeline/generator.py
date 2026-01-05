from __future__ import annotations

import json
import re
import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from dotenv import load_dotenv

from pipeline.planner import CodePlan
from pipeline.rewriter import prompt_rewriter_node, PromptRewriteInput

# =========================================================
# Environment
# =========================================================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ OPENAI_API_KEY not set — codegen will fail when invoked")

_client = OpenAI()

# =========================================================
# Prompt memory (for feedback loop)
# =========================================================
LAST_CODEGEN_PROMPT: str | None = None

# =========================================================
# Output schema
# =========================================================
class CodeBundle(BaseModel):
    frontend_files: Dict[str, str] = Field(default_factory=dict)
    backend_files: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""

# =========================================================
# LLM call
# =========================================================
def _call_llm(prompt: str) -> str:
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a precise code generator. Return STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""

# =========================================================
# Prompt builder (FRONTEND/BACKEND LOCKED)
# =========================================================
def build_codegen_prompt(plan: CodePlan) -> str:
    apis_lines = []
    checklist = []

    for i, a in enumerate(plan.apis, 1):
        params = ", ".join(
            p.get("name", "") for p in a.get("parameters", []) if p.get("name")
        )
        opid = a.get("operationId") or f"api_{i}"

        apis_lines.append(
            f"{i}. {a['method']} {a['path']} | "
            f"url: {a.get('full_url','')} | "
            f"operationId: {opid} | params: [{params}]"
        )

        checklist.append(
            f"- [ ] Implement backend route `/proxy/{opid}` for {a['method']} {a['path']}."
        )
        checklist.append(
            f"- [ ] Implement frontend form + fetch call for `/proxy/{opid}`."
        )

    return f"""
You are a senior full-stack engineer.

Front-end framework: {plan.frontend}  (FIXED — DO NOT CHANGE)
Back-end framework:  {plan.backend}   (FIXED — DO NOT CHANGE)

Context:
{plan.generator_brief}

APIs to integrate:
{chr(10).join(apis_lines)}

Requirements:
- Implement ALL APIs
- One backend route per API
- One frontend UI per API
- Use env vars for secrets
- Add basic validation
- Do NOT change the selected frontend or backend framework

Return STRICT JSON only with this schema:
{{
  "frontend_files": {{ "...": "..." }},
  "backend_files": {{ "...": "..." }},
  "notes": "..."
}}

Checklist:
{chr(10).join(checklist)}
""".strip()

# =========================================================
# JSON extractor (ROBUST)
# =========================================================
def _extract_json(text: str) -> Dict[str, Any]:
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    # First attempt: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Second attempt: extract last {...}
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        raise ValueError("No JSON block found in LLM output.")

    candidate = m.group(0)

    try:
        return json.loads(candidate)
    except Exception:
        # Final attempt: LLM-based JSON repair
        repair_prompt = f"""
You returned invalid JSON.

TASK:
- Fix the JSON so it is syntactically valid
- Do NOT change any values or structure
- Return ONLY valid JSON
- Do NOT add commentary

Invalid JSON:
{candidate}
""".strip()

        repair_resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You fix JSON formatting errors."},
                {"role": "user", "content": repair_prompt},
            ],
            max_tokens=4000,
        )

        fixed = repair_resp.choices[0].message.content or ""

        try:
            return json.loads(fixed)
        except Exception as e:
            raise ValueError("Failed to repair JSON.") from e

# =========================================================
# Generator (initial)
# =========================================================
def codegen_generate_fn(plan: CodePlan) -> CodeBundle:
    global LAST_CODEGEN_PROMPT

    prompt = build_codegen_prompt(plan)
    LAST_CODEGEN_PROMPT = prompt

    raw = _call_llm(prompt)
    data = _extract_json(raw)

    return CodeBundle(
        frontend_files={str(k): str(v) for k, v in (data.get("frontend_files") or {}).items()},
        backend_files={str(k): str(v) for k, v in (data.get("backend_files") or {}).items()},
        notes=str(data.get("notes", "") or ""),
    )

# =========================================================
# Generator (from rewritten prompt)
# =========================================================
def codegen_from_prompt(raw_prompt: str) -> CodeBundle:
    global LAST_CODEGEN_PROMPT
    LAST_CODEGEN_PROMPT = raw_prompt

    raw = _call_llm(raw_prompt)
    data = _extract_json(raw)

    return CodeBundle(
        frontend_files={str(k): str(v) for k, v in (data.get("frontend_files") or {}).items()},
        backend_files={str(k): str(v) for k, v in (data.get("backend_files") or {}).items()},
        notes=str(data.get("notes", "") or ""),
    )

# =========================================================
# Feedback-based regeneration
# =========================================================
def regenerate_with_feedback(previous_prompt: str, feedback: str) -> CodeBundle:
    rewritten_prompt = prompt_rewriter_node.invoke(
        PromptRewriteInput(
            previous_prompt=previous_prompt,
            feedback_text=feedback,
        )
    )
    return codegen_from_prompt(rewritten_prompt)

# =========================================================
# LangChain runnable
# =========================================================
codegen_generate_node = RunnableLambda(codegen_generate_fn)
