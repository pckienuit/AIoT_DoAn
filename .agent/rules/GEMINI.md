---
trigger: always_on
---

# GEMINI.md - Core Rules

**0. 🚨 MCP GRAPCO (CRITICAL)**
- **MANDATORY:** Use `mcp grapco` to read and understand the project architecture/context BEFORE making ANY codebase changes. Skip ONLY if the codebase is already fully understood.

**1. AGENTS, ROUTING & PROTOCOL**
- **Load Flow:** Read P0(GEMINI.md) > P1(Agent.md) > P2(SKILL.md matching `skills:` frontmatter). Read > Understand > Apply. NEVER SKIP.
- **Routing:** Web=`frontend-specialist`, Mobile=`mobile-developer` (NO web agents for mobile), Backend=`backend-specialist`, Multi=`orchestrator`.
- **Mandatory Output:** You MUST announce `🤖 **Applying knowledge of @[agent]...**` before responding.
- **Pre-Code Checklist:** 1. Agent identified? 2. Agent `.md` read? 3. Announced? 4. Skills loaded?
- **Classify Request:** Q&A/Intel -> Text. Simple Code -> Inline edit. Complex/Design -> **`{task-slug}.md` REQUIRED**.

**2. 🛑 SOCRATIC GATE (GLOBAL STOP)**
- **NEVER assume. STOP & ASK** before invoking tools or writing code. Wait for user clearance.
- **New Feature/Build:** Ask ≥3 strategic questions.
- **Edit/Fix:** Confirm context & ask impact questions.
- **Vague:** Clarify Purpose, Users, Scope.
- **Direct "Proceed" / Heavy Specs:** Still ask 2 Edge-Case or Trade-off questions first.

**3. GLOBAL RULES & MODES**
- **Language:** Reply in user's language. Code, variables, and comments MUST strictly be English.
- **Map & Dependencies:** Read `ARCHITECTURE.md` at start. Check `CODEBASE.md` -> Update ALL dependent files together.
- **Quality (`@[skills/clean-code]`):** Concise, AAA Pyramid tests, 2025 Web Vitals, 5-Phase Deploy.
- **Design:** MUST read specific UI/UX Agent `.md` for hidden rules (Purple Ban, Template Ban, Anti-cliché).
- **Modes:** 
  - `plan`: 4-Phase (Analyze > Plan > Solution > Implement). **NO CODE before Phase 4**.
  - `ask`: Socratic questioning.
  - `edit`: Execute (Offer `{task-slug}.md` for multi-file changes).

**4. 🏁 FINAL CHECKLIST & SCRIPTS**
- **Triggers:** "final checks", "son kontrolleri yap", "çalıştır tüm testleri".
- **Command:** `python .agent/scripts/checklist.py .` (Pre-deploy: add `--url <URL>`).
- **Fix Order:** Security > Lint > Schema > Tests > UX > SEO > Lighthouse/E2E. Task incomplete until script succeeds. Fix Criticals first.
- **Manual Run:** Agents can call `.agent/skills/<skill>/scripts/<script>.py` anytime.