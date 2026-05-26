---
description: "Create a new feature or component for the AIoT Face Check-in project. Triggers specialist agents for frontend, backend, edge, and tests."
alwaysApply: false
---

# /create — Feature Creation Workflow

> Adapted from `.agent/workflows/create.md` for Cursor. Orchestrates multi-agent implementation.

---

## When to Use

- User wants to build a new feature
- Multiple components need changes (Web + Server, or Server + Edge, etc.)
- User says "build X", "create Y", "add Z"

## Pre-Check: Is There a Plan?

- ✅ If `docs/PLAN-{slug}.md` exists → proceed to implementation
- ❌ If NOT → create plan first with `/plan` workflow

---

## Workflow Steps

### Step 1: Verify Plan Exists

Check for `docs/PLAN-{task-slug}.md`. If missing, redirect to `/plan`.

---

### Step 2: Analyze Plan

Read the plan file and identify:

| Component | Agent to Invoke | Files to Modify |
|-----------|----------------|-----------------|
| Web client | `@.cursor/agents/frontend-specialist.md` | `web/...` |
| FastAPI server | `@.cursor/agents/backend-specialist.md` | `server/...` |
| MaixCAM edge | `@.cursor/agents/edge-specialist.md` | `edge/...` |
| Tests | `@.cursor/agents/test-engineer.md` | `tests/...` |

---

### Step 3: Implement by Component

**For each component:**

1. Invoke the correct specialist agent via Task tool
2. Pass the plan context and relevant files
3. Wait for completion before moving to next component (unless parallel is safe)

**Safe to parallelize:**
- Frontend Specialist + Backend Specialist (if API contract is agreed)
- Test Engineer (can start writing tests in parallel with implementation)

**Must be sequential:**
- Backend before Frontend (API contract needed)
- Edge after Backend (sync protocol depends on API)

---

### Step 4: Integration Check

After all components implemented:

1. Run `python -m pytest tests/` (if server/edge tests exist)
2. Verify the data flow end-to-end matches the plan
3. Run linting on all changed files
4. Report completion

---

### Step 5: Report

```
[OK] Feature "{name}" implemented.

Components delivered:
✅ Web: [files changed]
✅ Server: [files changed]
✅ Edge: [files changed]
✅ Tests: [files changed]

Next steps:
- Review the implementation
- Run the dev server to test
- Run `python .agent/scripts/checklist.py .` for pre-deploy validation
```

---

## Quality Gates

Before reporting completion:

- [ ] All specialist agents completed successfully
- [ ] Integration test passes (end-to-end)
- [ ] Linting passes on all changed files
- [ ] No hardcoded keys or plaintext embeddings
- [ ] Security implications reviewed
