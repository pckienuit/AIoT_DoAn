---
description: "Debugging workflow for the AIoT Face Check-in project. Systematic root cause analysis across web client, FastAPI server, and MaixCAM edge."
alwaysApply: false
---

# /debug — Systematic Debugging Workflow

> Adapted from `.agent/workflows/debug.md` for Cursor.

---

## When to Use

- User reports a bug or unexpected behavior
- Something is failing or returning incorrect results
- Edge case not handled

---

## Workflow Steps

### Step 1: Reproduce

- Gather the exact error message or behavior
- Identify when it started (regression?) or if it never worked
- Reproduce locally if possible

---

### Step 2: Localize

Identify which component is failing:

| Symptom | Likely Component |
|---------|-----------------|
| Camera not working | Web (MediaPipe) |
| Registration fails | Web (encryption) or Server (decryption) |
| Match returns wrong person | Qdrant (embedding/ANN) |
| Edge sync fails | Server (XTEA-CTR) or Edge (decryption) |
| Slow recognition | Edge (inference pipeline) |

---

### Step 3: Investigate

Use MCP codegraph tools to trace the relevant code path:

```
# For web issues:
codegraph_context web/register.js
codegraph_explore web/

# For server issues:
codegraph_context server/routes/
codegraph_explore server/

# For edge issues:
codegraph_context edge/
codegraph_explore edge/
```

---

### Step 4: Identify Root Cause

Find the actual bug (not just the symptom):

- Read the actual code, don't assume
- Add temporary debug logging if needed
- Check inputs and outputs at each step

---

### Step 5: Fix

- Apply surgical fix (change only what's broken)
- Add a test that reproduces the bug
- Ensure the fix doesn't break other components

---

### Step 6: Verify

- Run tests to confirm the fix
- Check the integration still works
- Report: "Fixed: [root cause]. [What was changed]."

---

## Common Bug Patterns in AIoT Face Check-in

| Bug | Cause | Fix |
|-----|-------|-----|
| Registration succeeds but match fails | Embedding dimension mismatch | Check ArcFace P3 = 128D |
| Edge recognition slow | YOLO running every frame | Add frame skip |
| Qdrant search returns empty | Collection not initialized | Run migration |
| AES-GCM decryption fails | Tag mismatch (data tampered) | Check key derivation |
| XTEA-CTR decryption garbled | Nonce reuse | Generate unique nonce per batch |
