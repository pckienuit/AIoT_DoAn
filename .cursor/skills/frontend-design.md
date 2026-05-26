---
description: "Frontend design principles for AIoT Face Check-in web client. Use when designing UI, choosing colors, typography, or layout for the kiosk registration interface."
alwaysApply: false
---

# Frontend Design Skill

> This skill references the full design system from `.agent/skills/frontend-design/`. Read those files for detailed guidance.

---

## Quick Reference

When working on the kiosk registration UI:

### Must Read First
- `@.agent/skills/frontend-design/ux-psychology.md` — ALWAYS before any UI task

### Optional (by need)
- `@.agent/skills/frontend-design/color-system.md` — palette decisions
- `@.agent/skills/frontend-design/typography-system.md` — font selection
- `@.agent/skills/frontend-design/animation-guide.md` — motion design
- `@.agent/skills/frontend-design/decision-trees.md` — context templates

---

## Kiosk-Specific Considerations

This is a **public kiosk** at an airport check-in counter:

| Factor | Implication |
|--------|-------------|
| **Audience** | Travelers of all ages, nationalities, tech levels |
| **Lighting** | Airport ambient (mixed — bright + shadows) |
| **Time pressure** | Users in a hurry — fast, clear feedback |
| **Accessibility** | Must be WCAG AA compliant (public display) |
| **Screen size** | Large kiosk display (typically 1080p portrait) |

---

## Design Priorities for Kiosk

1. **Instant feedback** — Show face detection status in real time
2. **Clarity over beauty** — User must understand what to do without reading instructions
3. **Multilingual-ready** — Text in HTML, ready for i18n
4. **High contrast** — Airport lighting varies; test in bright conditions
5. **Large touch targets** — Kiosk may be touch-enabled (min 48x48px)

---

## Purple Ban (Global Rule)

❌ **NEVER** use purple, violet, indigo, or magenta as the primary/brand color.

For an airport kiosk, consider:
- ✅ Blue (trust, aviation)
- ✅ Teal (modern, calm)
- ✅ Orange (energy, calls to action)
- ✅ Deep red (premium airline feel)

---

## Animation Requirements

- ✅ All elements have entrance animations (scroll-triggered or on-mount)
- ✅ Micro-interactions on all interactive elements (hover, active)
- ✅ Spring physics easing (`cubic-bezier(0.34, 1.56, 0.64, 1)`)
- ✅ `prefers-reduced-motion` respected
- ✅ GPU-accelerated only (`transform`, `opacity`)

---

## Quality Checklist

Before completing a UI task:

- [ ] `ux-psychology.md` read and applied
- [ ] Purple ban respected
- [ ] High contrast tested
- [ ] Touch targets ≥ 48px
- [ ] Animation is GPU-accelerated
- [ ] `prefers-reduced-motion` handled
- [ ] Multilingual-ready text
