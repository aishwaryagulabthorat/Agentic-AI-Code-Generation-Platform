# pipeline/rewriter.py
from __future__ import annotations

import os
import re
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

_client = OpenAI()


class PromptRewriteInput(BaseModel):
    previous_prompt: str
    feedback_text: str


def build_rewriter_prompt(prev_prompt: str, feedback: str) -> str:
    return f"""
You are a senior engineer acting as a *prompt rewriter*.

TASK:
- Take the previous code-generation PROMPT (between <<<PROMPT and PROMPT>>>).
- Apply ONLY the changes requested in FEEDBACK (between <<<FEEDBACK and FEEDBACK>>>).
- Keep EVERYTHING else identical: headings, sections, wording, ordering, API list, JSON-only instruction, schema, checklists, conventions.
- Do NOT remove or rename these required lines/sections:
  - A line starting with: "Front-end framework:"
  - A line starting with: "Back-end framework:"
  - A section heading that starts with: "APIs to integrate"
- Return the *ENTIRE updated prompt* enclosed EXACTLY between <<<UPDATED_PROMPT and UPDATED_PROMPT>>>.
- Do NOT add commentary outside the fence.

<<<PROMPT
{prev_prompt}
PROMPT>>>

<<<FEEDBACK
{feedback.strip()}
FEEDBACK>>>

Now output:

<<<UPDATED_PROMPT
...the full updated prompt goes here...
UPDATED_PROMPT>>>
""".strip()


def rewrite_prompt_fn(payload: PromptRewriteInput) -> str:
    rewrite_instruction = build_rewriter_prompt(
        prev_prompt=payload.previous_prompt,
        feedback=payload.feedback_text,
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are precise and copy text verbatim unless changes are requested."},
            {"role": "user", "content": rewrite_instruction},
        ],
        max_tokens=3000,
    )
    out = resp.choices[0].message.content or ""

    m = re.search(r"<<<UPDATED_PROMPT\s*(.*?)\s*UPDATED_PROMPT>>>", out, re.S)
    if not m:
        # repair once
        repair = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Return the entire updated prompt between <<<UPDATED_PROMPT and UPDATED_PROMPT>>> only."},
                {"role": "user", "content": f"The previous response lacked a valid UPDATED_PROMPT block. Re-run.\n\n{rewrite_instruction}"},
            ],
            max_tokens=3000,
        )
        out2 = repair.choices[0].message.content or ""
        m = re.search(r"<<<UPDATED_PROMPT\s*(.*?)\s*UPDATED_PROMPT>>>", out2, re.S)
        if not m:
            raise ValueError("Rewriter did not return a valid UPDATED_PROMPT block.")

    rewritten_prompt = m.group(1).strip()

    # basic validation of required lines/sections
    missing = []
    if not re.search(r"^Front-end framework:\s*.+", rewritten_prompt, re.M):
        missing.append("Front-end framework line")
    if not re.search(r"^Back-end framework:\s*.+", rewritten_prompt, re.M):
        missing.append("Back-end framework line")
    if not re.search(r"APIs to integrate[^\n]*:", rewritten_prompt):
        missing.append("APIs to integrate section")

    if missing:
        raise ValueError(f"Rewriter output missing required parts: {missing}")

    return rewritten_prompt


prompt_rewriter_node = RunnableLambda(rewrite_prompt_fn)
