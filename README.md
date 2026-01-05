# Agentic AI Code Generation Platform

An end-to-end **Agentic AI system** that transforms a high-level developer prompt into **production-ready full-stack code** using multiple AI agents, Retrieval-Augmented Generation (RAG), and an iterative feedback loop.

---

## What This Project Does

This system allows a user to:

- Describe **what they want to build** in natural language
- Select **frontend and backend frameworks**
- Automatically:
  - Parse intent
  - Retrieve relevant APIs via semantic search
  - Plan integrations
  - Generate full-stack code
- Review generated code
- Provide feedback
- Regenerate code iteratively
- Download the final codebase as a ZIP

All powered by an **agent-orchestrated AI pipeline**.

---

## Architecture Overview

<img width="1024" height="1024" alt="Agentic AI" src="https://github.com/user-attachments/assets/f2973af6-888c-4e11-8a79-171106fcf217" />


High-level flow:

User → Streamlit UI
→ Instruction Parser Agent
→ RAG Agent (Sentence-BERT + Pinecone)
→ Code Planning Agent
→ Code Generator Agent
→ Feedback & Rewrite Loop
→ Final Code Output


---

## Core Agents

### Instruction Parser Agent
- LLM: **GPT-4o-mini**
- Extracts:
  - Intent
  - Entities
  - Subtasks
- Produces structured JSON

---

### RAG Agent (Retrieval-Augmented Generation)
- **Sentence-BERT (e5-base-v2)** for embeddings
- **Pinecone Vector Database**
  - Metric: cosine
  - Dimensions: 768
  - Type: Dense
  - Cloud: AWS (us-east-1)
- Retrieves top-K relevant APIs

**Metadata Source**
- **Couchbase Capella**
- Stores API specifications, parameters, versions, and tags
- Used during embedding creation and indexing

---

### Code Planning Agent
- Consumes:
  - Parsed intent
  - Retrieved APIs
- Locks frontend & backend frameworks
- Produces a structured **Code Plan**

---

### Code Generation Agent
- LLM: **GPT-4o-mini**
- Generates:
  - Frontend source files
  - Backend source files
- Enforces:
  - One backend route per API
  - One frontend UI per API
  - Strict JSON output format

---

### Feedback & Rewrite Agent
- User reviews generated code
- Optional feedback provided
- Prompt is rewritten **without re-running RAG**
- Code regenerated deterministically

---

## Tech Stack

- **Python**
- **Streamlit** — UI
- **LangChain** — Agent orchestration
- **GPT-4o-mini and TinyLlama** — LLM for parsing & codegen
- **Sentence-BERT (e5-base-v2)** — Embeddings
- **Pinecone** — Vector database
- **Couchbase Capella** — API metadata storage
- **Pydantic** — Schema validation

---

## Project Structure

```text
pipeline/
├── core.py          # Parser Agent
├── rag.py           # RAG logic (Sentence-BERT + Pinecone)
├── planner.py       # Code Planning Agent
├── generator.py     # Code Generator Agent
├── rewriter.py      # Feedback / Rewrite Agent
├── orchestrator.py  # LangChain orchestration
└── exporter.py      # ZIP export


