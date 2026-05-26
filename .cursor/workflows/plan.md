---
description: "Socratic planning workflow for the AIoT Face Check-in project. Use when the user wants to plan a new feature, break down a complex task, or create a roadmap before writing code."
alwaysApply: false
---

# /plan — Project Planning Workflow

> Adapted from `.agent/workflows/plan.md` for Cursor. Creates a plan document before any code is written.

---

## When to Use

- User wants to plan a new feature
- Task is complex and has multiple steps
- User asks "how do we approach X?"
- Before invoking specialist agents for multi-component work

## When NOT to Use

- Single file, trivial fix → just do it
- User explicitly says "just write the code"

---

## Workflow Steps

### Step 1: Clarify (Socratic Gate)

Ask the user:

1. **Scope:** Full feature or specific part?
2. **Priority:** What's most important — security, speed, completeness?
3. **Constraints:** Timeline, tech preferences, existing code to work with?
4. **Success:** How will we know it's done?

> ⚠️ **DO NOT assume.** If unclear, ask. If clear, proceed.

---

### Step 2: Analyze

- Read relevant architecture files (`@.cursor/ARCHITECTURE.md`)
- Identify affected components (Web / Server / Edge)
- Check `docs/` for existing related plans

---

### Step 3: Create Plan File

Create `docs/PLAN-{task-slug}.md`:

```markdown
# Plan: [Feature Name]

## Context
- **Why:** [User's reason]
- **Scope:** [In / Out]
- **Components:** Web / Server / Edge

## Task Breakdown

### Step 1: [Name]
- [ ] Subtask
- → verify: [how to verify]

### Step 2: [Name]
- [ ] Subtask
- → verify: [check]

## Agent Assignments

| Component | Owner | Key Files |
|-----------|-------|-----------|
| Web | Frontend | `web/...` |
| Server | Backend | `server/...` |
| Edge | Edge | `edge/...` |

## Risks

| Risk | Mitigation |
|------|-----------|
| [Risk] | [Plan] |

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
```

---

### Step 4: Report

```
[OK] Plan created: docs/PLAN-{slug}.md

Next steps:
- Review the plan
- Run /create to start implementation
- Or modify the plan manually
```

---

## Naming Examples

| Request | Plan File |
|---------|-----------|
| `/plan add face match page` | `docs/PLAN-face-match.md` |
| `/plan implement edge sync` | `docs/PLAN-edge-sync.md` |
| `/plan improve UI responsiveness` | `docs/PLAN-ui-speed.md` |
