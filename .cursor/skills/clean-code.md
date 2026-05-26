---
description: "Clean code principles for the AIoT Face Check-in project. Use globally — applies to all code in web/, server/, edge/, and scripts/ directories."
alwaysApply: false
---

# Clean Code Skill

> References: `@.agent/skills/clean-code/SKILL.md`

---

## Core Principles

### 1. Simplicity First
- Write the minimum code that solves the problem
- If 200 lines could be 50, rewrite it
- No speculative features, no "flexibility" not requested
- Ask: *"Would a senior engineer say this is overcomplicated?"* — If yes, simplify

### 2. Name Things Well
- Variables/functions: `snake_case` in Python, `camelCase` in JS/TS
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Names must reveal intent: `face_embedding` not `v`; `encrypt_for_edge_sync` not `do_encrypt`

### 3. Functions Do One Thing
- Each function has one responsibility
- If you need "and" to describe what it does, split it
- Max ~40 lines per function (hard limit: 100)

### 4. Comments Explain Why, Not What
- ❌ `i += 1  # increment i` — useless
- ✅ `# Nonce must be unique per encryption; GCM tag catches tampering` — explains intent

### 5. No Dead Code
- Remove imports/variables/functions made unused by your changes
- Don't remove pre-existing dead code unless asked — mention it instead

---

## Language-Specific Rules

### Python (server/, edge/, scripts/)
- Type hints required on all public functions
- Docstrings on modules and classes (not on every function)
- No `from module import *`
- Sorted imports (`isort`)

### JavaScript/TypeScript (web/)
- TypeScript strict mode, no `any`
- No `console.log` in production
- Explicit return types on exported functions

---

## Anti-Patterns (Global Ban List)

| ❌ Instead of | ✅ Use |
|--------------|--------|
| `any` type | `unknown` or proper type |
| Magic numbers | Named constants |
| `console.log` (prod) | `logging` / structured logger |
| Deeply nested callbacks | `async/await` |
| Giant files | Split by responsibility |
| Premature abstraction | Wait for the second use |

---

## Success Criteria Before Merging

1. Linting passes (`npm run lint` for web, `ruff`/`flake8` for Python)
2. Type checking passes (`tsc --noEmit`, `mypy`)
3. Tests pass (if tests exist)
4. No new dead code introduced
