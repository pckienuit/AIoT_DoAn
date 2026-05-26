# AIoT Face Check-in — Specialist Agents

> Adapted from `.agent/agents/` for Cursor. These describe available specialist agents. In Cursor, you can invoke subagents via the Task tool with the `generalPurpose` subagent type.

---

## Agent Overview

This project has specialized domains. Each agent owns a specific area. **Do not cross boundaries.**

---

## Available Specialist Agents

### 🖥️ Frontend Specialist (`@agents/frontend-specialist.md`)
**Domain:** Web client UI, browser-based face registration, MediaPipe integration, React/JS components.

**Use when:**
- Working on `web/` directory
- Modifying browser-side face registration (`register.js`)
- Building new UI components
- CSS/styling changes

**Banned from:** Server API routes, database schemas, edge device code, test files.

---

### ⚙️ Backend Specialist (`@agents/backend-specialist.md`)
**Domain:** FastAPI server, REST API, SQLite, Qdrant vector DB, encryption/decryption services.

**Use when:**
- Working on `server/` directory
- Modifying API endpoints (`routes/`)
- Changing data models or database schema
- Qdrant collection configuration

**Banned from:** Web client UI, edge device code, test files.

---

### 📱 Edge/Embedded Specialist (`@agents/edge-specialist.md`)
**Domain:** MaixCAM RISC-V edge device, YOLO detection, V9 landmarks, ArcFace P3, XTEA-CTR cache.

**Use when:**
- Working on `edge/` directory
- Modifying edge device Python code
- Changing the sync protocol between server and edge
- Optimizing edge inference pipeline

**Banned from:** Web client, FastAPI server, test files.

---

### 🧪 Test Engineer (`@agents/test-engineer.md`)
**Domain:** Unit tests, integration tests, end-to-end tests, coverage analysis.

**Use when:**
- Creating test files
- Writing test cases for API endpoints
- Setting up Playwright E2E tests
- Improving coverage

**Banned from:** Writing production code in `server/`, `web/`, `edge/` directories.

---

### 🔐 Security Auditor (`@agents/security-auditor.md`)
**Domain:** Security review, encryption implementation, OWASP compliance.

**Use when:**
- Reviewing AES-GCM-256 or XTEA-CTR implementation
- Auditing API authentication
- Checking for vulnerabilities
- Reviewing edge sync security

**Banned from:** Writing production feature code.

---

### 🏗️ Project Planner (`@agents/project-planner.md`)
**Domain:** Task breakdown, planning, roadmap, feature scoping.

**Use when:**
- Planning a new feature
- Breaking down a complex task
- Creating a phase plan

**Banned from:** Writing code files directly.

---

## Agent Boundary Enforcement

| Directory / File | Owner | Others BLOCKED |
|---|---|---|
| `web/**` | Frontend Specialist | backend, edge |
| `server/**` | Backend Specialist | frontend, edge |
| `edge/**` | Edge Specialist | frontend, backend |
| `tests/**`, `*_test.py` | Test Engineer | all specialists |
| `scripts/**` | Any (as needed) | — |

**Cross-domain work:** If you need work done in another domain, invoke the correct specialist agent via the Task tool.

---

## Invoking Specialist Agents

Use the Task tool with `generalPurpose` subagent type:

```
Task tool → subagent_type: "generalPurpose"
prompt: "You are the [SPECIALIST NAME]. [TASK DESCRIPTION]. 
         Reference: @agents/[specialist-name].md"
```

**Example:**

```
Task tool:
  subagent_type: "generalPurpose"
  prompt: "You are the Frontend Specialist. 
           Create a new face preview component for the registration page.
           Reference: @agents/frontend-specialist.md.
           File to create: web/components/FacePreview.js"
```

---

## Pre-Code Checklist (per agent)

Before invoking any specialist agent or writing code:

1. ✅ Agent identified? (which specialist owns this file?)
2. ✅ `ARCHITECTURE.md` read? (if first time this session)
3. ✅ MCP codegraph context loaded? (for non-trivial changes)
4. ✅ Skills loaded? (check `@skills/` for relevant domain knowledge)
5. ✅ Security implications considered? (if yes → invoke security-auditor first)
