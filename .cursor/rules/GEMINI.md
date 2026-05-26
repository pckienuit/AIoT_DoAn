---
title: "AIoT Face Check-in — Core Rules"
description: "Core rules for the Face Check-in AIoT project. Covers MCP usage, project plan, routing, quality gates, and anti-hallucination discipline for all Cursor Agent sessions."
alwaysApply: true
---

# GEMINI.md — AIoT Face Check-in: Core Rules

> Adapted from `.agent/rules/GEMINI.md` for Cursor IDE. These rules govern every Cursor Agent session.

---

## 0. MCP Codegraph (CRITICAL)

**MANDATORY:** Use the `codegraph` MCP server to read and understand project architecture/context **BEFORE** making ANY codebase changes. Skip only if the codebase is already fully understood.

Run `codegraph_context` or `codegraph_explore` to get symbol-level understanding of the affected files.

---

## 0.1 Project Plan (READ FIRST — SKIP IF ALREADY READ THIS SESSION)

**MANDATORY:** Read the project README at the START of every new session before taking any action.

Skip condition: If the README has already been read in the current conversation, proceed directly.

After reading, briefly announce the system architecture:
> **Project loaded. System: Flight Info Lookup via Face Recognition (Web → FastAPI Server → Qdrant DB → MaixCAM Edge).**

---

## 1. Project Context

This is an **AIoT project** with three components:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Client** | Browser (MediaPipe + ArcFace P3 + AES-GCM-256) | Face registration via browser |
| **FastAPI Server** | Python FastAPI + SQLite + Qdrant | REST API, metadata + vector DB |
| **MaixCAM Edge** | RISC-V edge device (YOLO + V9 Landmarks + ArcFace P3) | Real-time face recognition |

---

## 2. Code Routing & Agent Selection

**Routing by file type:**

| File Pattern | Agent / Area | Notes |
|---|---|---|
| `web/**`, `client/**`, `*.html`, `*.js`, `*.ts` | Frontend / UI | MediaPipe, ArcFace web client |
| `server/**`, `api/**`, `services/**` | Backend | FastAPI, Qdrant, SQLite |
| `edge/**`, `maixcam/**`, `*.py` (embedded) | Edge / Embedded | MaixCAM, YOLO, XTEA-CTR |
| `tests/**`, `*_test.py`, `*_spec.py` | Testing | Pytest, Playwright |
| `scripts/**`, `*.py` (tooling) | DevOps / Scripts | Python tooling |
| `docs/**`, `README*` | Documentation | Docs only |

**Key routing reminders:**
- Mobile web → Frontend (NOT mobile-developer, which targets native apps)
- Edge/embedded Python → Edge area (NOT backend-specialist, which targets the FastAPI server)
- Security/auth → Always invoke security audit first

---

## 3. Global Rules & Modes

**Language:** Reply in the user's language. Code, variables, and comments MUST be English.

**Quality (`@skills/clean-code`):**
- Concise, well-structured code
- Tests follow AAA pyramid pattern
- 2025 Web Vitals standards
- 5-Phase Deploy process

**Simplicity First:** Minimum code that solves the problem. No speculative features, no abstractions for single-use code, no "flexibility" that wasn't requested. If 200 lines could be 50, rewrite it.

**Modes:**

- **`plan` (Plan Mode):** 4-Phase (Analyze → Plan → Solution → Implement). **NO CODE before Phase 4.**
- **`ask` (Ask Mode):** Socratic questioning only.
- **`edit` / default:** Execute changes. Offer a `{task-slug}.md` plan for multi-file changes.

---

## 4. Anti-Hallucination Discipline

- **Admit uncertainty.** Say `"I'm not sure about X — let me verify"` or `"I don't have enough context to answer this confidently."` NEVER fabricate APIs, function signatures, config options, or file paths.
- **Ground in real code.** ALWAYS read the actual file/function BEFORE referencing or modifying it. Never rely on memory.
- **Cite sources.** Provide file path + line numbers when referencing code. Provide URL when referencing external docs.
- **Reason before answering.** For non-trivial questions, explain your reasoning chain before giving a conclusion.
- **Verify MCP tools first.** Check the tool schema/descriptor before calling any MCP tool (`codegraph`, `openfiles`, etc.).

---

## 5. Surgical Changes & Goal-Driven Execution

- **Touch only what you must.** Every changed line must trace directly to the user's request.
- **Don't "improve"** adjacent code, comments, or formatting that aren't part of the task.
- **Don't refactor** things that aren't broken. Match existing style.
- **Orphan cleanup:** Remove imports/variables/functions that YOUR changes made unused.
- **Define success criteria** before implementing:
  - "Add validation" → Write tests for invalid inputs, then make them pass.
  - "Fix the bug" → Write a test that reproduces it, then fix it.
  - "Refactor X" → Ensure tests pass before and after.

---

## 6. Socratic Gate — THINK BEFORE CODING

- **NEVER assume. STOP & ASK** before invoking tools or writing code. Wait for user clearance.
- **State assumptions explicitly.** If uncertain about intent, ask — don't pick silently.
- **New Feature/Build:** Ask ≥3 strategic questions about scope, priority, and constraints.
- **Edit/Fix:** Confirm context & ask impact questions.
- **Vague requests:** Clarify Purpose, Users, Scope.

---

## 7. MCP Configuration

This project uses these MCP servers (configured in `.cursor/mcp_config.json`):

| MCP Server | Purpose |
|---|---|
| `user-codegraph` | Code intelligence, symbol search, call graph |
| `user-openvision` | Vision/UI analysis |
| `user-openfiles` | File operations |
| `user-projectscan` | Project scanning |
| `user-projectedit` | Project editing |

Use these MCP tools when exploring the codebase, especially `codegraph_context` and `codegraph_explore` for architecture understanding before writing code.

---

## 8. Key Reference Files

| File | Purpose |
|------|---------|
| `README.md` | Full system architecture, tech stack, installation |
| `.cursor/rules/GEMINI.md` | These core rules |
| `.cursor/ARCHITECTURE.md` | Detailed AIoT architecture |
| `.cursor/AGENTS.md` | Specialist agent descriptions |

