# # pipeline/core.py
# import os
# import json
# import time
# import torch
# from pydantic import BaseModel
# from typing import List
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain_core.runnables import RunnableLambda


# # =====================================
# # 1️⃣ Model Loading (local TinyLlama)
# # =====================================

# # Adjust this path if needed for your setup
# # =====================================
# # 1️⃣ Model Loading (local TinyLlama)
# # =====================================
# import os

# POSSIBLE_PATHS = [
#     "/Users/aishwaryathorat/Desktop/tinyllama-local",  # ✅ YOUR ACTUAL PATH
# ]

# model_path = next((p for p in POSSIBLE_PATHS if os.path.exists(p)), None)

# if not model_path:
#     raise FileNotFoundError("❌ TinyLlama model not found. Check path.")

# print(f"✅ Loading TinyLlama from: {model_path}")

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float32
# ).to("cpu")


# print("✓ TinyLlama model loaded successfully")


# # =====================================
# # 2️⃣ Pydantic Schema for Parser Output
# # =====================================
# class LLMOutput(BaseModel):
#     intent: str
#     entities: List[str]
#     subtasks: List[str]


# # =====================================
# # 3️⃣ Core TinyLlama Functions
# # =====================================
# import time

# def send_query_to_tinyllama(user_query: str, error_context: str = None):
#     start_total = time.time()
#     print("⏳ Building prompt...")

#     prompt = f"""
# Return ONLY valid JSON.

# Format:
# {{"intent": "...", "entities": ["..."], "subtasks": ["..."]}}

# Instruction:
# {user_query}

# JSON:
# """

#     t0 = time.time()
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#         padding=True
#     )
#     print(f"✅ Tokenization done in {time.time() - t0:.2f}s")

#     device = next(model.parameters()).device
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     print("🔥 Starting model.generate() (this is the slow part on CPU)...")
#     t1 = time.time()

#     with torch.inference_mode():
#         outputs = model.generate(
#             inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=40,
#             do_sample=False,
#             use_cache=True,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )

#     print(f"🚀 Generation finished in {time.time() - t1:.2f}s")
#     print(f"🧠 Total time: {time.time() - start_total:.2f}s")

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     generated_text = response[len(prompt):].strip()
#     return generated_text



# def parse_and_validate_json(llm_output: str):
#     """Extract valid JSON and validate structure."""
#     try:
#         start, end = llm_output.find("{"), llm_output.rfind("}") + 1
#         json_str = llm_output[start:end]
#         parsed = json.loads(json_str)
#         return LLMOutput(**parsed)
#     except Exception:
#         return None


# def instruction_parser_agent(user_query: str, max_retries: int = 2):
#     """Main parser agent function."""
#     print(f"Processing with TinyLlama: {user_query}")
#     start_time = time.time()

#     llm_response = send_query_to_tinyllama(user_query)

#     for attempt in range(max_retries + 1):
#         parsed = parse_and_validate_json(llm_response)
#         if parsed:
#             print(f"✓ Success in {time.time() - start_time:.2f} seconds")
#             return parsed.model_dump()  # for Pydantic v2
#         if attempt < max_retries:
#             print(f"Retry {attempt + 1}: Invalid JSON, retrying...")
#             llm_response = send_query_to_tinyllama(user_query, error_context=llm_response)

#     print("All retries failed.")
#     return None


# # =====================================
# # 4️⃣ LangChain-compatible Runnable Node
# # =====================================
# class InstructionSpec(BaseModel):
#     intent: str
#     entities: List[str]
#     subtasks: List[str]


# def run_tiny_llama_parse(user_text: str) -> dict:
#     """Call TinyLlama parser and return dict."""
#     out = instruction_parser_agent(user_text)
#     if not out:
#         raise ValueError("Parser failed to produce valid output.")
#     return out


# def parser_node_fn(user_text: str) -> InstructionSpec:
#     """RunnableLambda-compatible wrapper."""
#     parsed_dict = run_tiny_llama_parse(user_text)
#     return InstructionSpec(**parsed_dict)


# parser_node = RunnableLambda(parser_node_fn)

## END OF TINYLLAMA CODE 

# pipeline/core.py
print("🔥 core.py LOADED")

from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
import json, re

# =====================
# 1️⃣ Parser Output Schema
# =====================
class InstructionSpec(BaseModel):
    intent: str
    entities: List[str]
    subtasks: List[str]


# =====================
# 2️⃣ GPT-4o-mini LLM
# =====================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =====================
# 3️⃣ Parser Function
# =====================
def parser_node_fn(user_text: str) -> InstructionSpec:
    prompt = f"""
You are a software planning assistant.

Extract:
- intent (one sentence)
- entities (key nouns / concepts)
- subtasks (actionable steps)

Return ONLY valid JSON in this format:
{{
  "intent": "...",
  "entities": ["..."],
  "subtasks": ["..."]
}}

Instruction:
{user_text}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Parser did not return JSON")

    data = json.loads(match.group(0))
    return InstructionSpec(**data)

# =====================
# 4️⃣ LangChain Runnable
# =====================
parser_node = RunnableLambda(parser_node_fn)
