---
version: alpha
name: HoloDeck
description: >-
  Dark-first visual identity for HoloDeck, an open-source experimentation
  platform for AI agents. Near-black surfaces, a single plasma-green accent,
  system-pinned Inter + JetBrains Mono. Engineer-facing, CLI-native, confident
  and still. Ships with a companion "Clean Room" light theme.
colors:
  # ----- Dark theme (default) -----
  primary: "#7bff5a" # plasma green — sole brand accent
  primary-bright: "#9bff5f" # lime — gradient pair (lighter stop)
  primary-mint: "#53ff9c" # mint — gradient pair (darker stop)
  primary-soft: "#5ae0a6" # desaturated mint — headline gradient partner
  on-primary: "#041208" # ink for text/icons on a green fill
  surface: "#050b09" # page background (near-black)
  surface-deep: "#030807" # deepest edge
  surface-container: "#0a110f" # card fill
  surface-container-high: "#0c1412" # raised surface
  outline: "#1c2b25" # hairline border
  on-surface: "#e8f5ec" # primary text (cool minty off-white)
  on-surface-variant: "#9bb3a5" # secondary / muted text
  code-on-surface: "#e5e7eb" # text inside code/terminal surfaces
  # ----- Light theme ("Clean Room") -----
  light-primary: "#16a34a" # green shifted down to read on white (AA)
  light-primary-bright: "#22c55e"
  light-primary-deep: "#15803d"
  light-primary-soft: "#2ea043"
  light-on-primary: "#ffffff"
  light-surface: "#f3f6f4" # mint-tinted off-white
  light-surface-deep: "#eef2ee"
  light-surface-container: "#ffffff"
  light-surface-container-high: "#f8faf8"
  light-outline: "#e1e7e3"
  light-on-surface: "#0c1412" # ink (the dark theme's raised surface)
  light-on-surface-variant: "#54645a"
typography:
  hero:
    fontFamily: Inter
    fontSize: 46px
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 26px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.3
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.6
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.55
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: 500
    lineHeight: 1
    letterSpacing: 0.15em
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.55
rounded:
  sm: 6px
  md: 10px
  lg: 16px
  full: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 32px
  xl: 56px
  card-padding: 16px
  section-gap: 56px
  max-width: 1120px
components:
  button-primary:
    backgroundColor: "{colors.primary-mint}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.full}"
    padding: 8px 14px
  button-primary-hover:
    backgroundColor: "{colors.primary-bright}"
  button-ghost:
    backgroundColor: "{colors.surface-container-high}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.full}"
    padding: 8px 14px
  card:
    backgroundColor: "{colors.surface-container}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.md}"
    padding: "{spacing.card-padding}"
  code-card:
    backgroundColor: "{colors.surface-deep}"
    textColor: "{colors.code-on-surface}"
    rounded: "{rounded.lg}"
    padding: 16px 18px
  tag:
    backgroundColor: "{colors.surface-container-high}"
    textColor: "{colors.on-surface-variant}"
    rounded: "{rounded.full}"
    typography: "{typography.label-caps}"
    padding: 4px 10px
---

# HoloDeck DESIGN.md

> Machine-readable tokens live in the front matter above; the prose below explains intent and application. For the long-form rationale, content voice, iconography, and asset inventory, see `README.md`. CSS source of truth: `colors_and_type.css` (dark) + `colors_and_type.light.css` (Clean Room light).

## Overview

HoloDeck is an open-source experimentation platform for AI agents — engineers describe agents in YAML and ship them with a CLI. The brand mirrors that audience: **technical, terse, confident, and still.** The product name references the Star Trek Holodeck, and the visual world leans into it — a near-black starfield with a single plasma-green energy color.

The UI should feel like a developer tool at rest: dark, calm, high-contrast, low-noise. Splashes of green read as energy against the dark, never decoration. No marketing gloss, no motion theatrics, no gradient orbs. When in doubt, remove rather than add — "one thousand no's for every yes." The atmosphere is the brand: a deep space-black canvas lit by two soft green spotlights.

A companion **light theme ("Clean Room")** exists for surfaces that demand a light background (docs, email, print). It is not a naïve invert — it keeps a whisper of the green glow and, critically, **keeps code/terminal surfaces dark** so brand texture survives.

## Colors

The palette is near-monochromatic: one signature green over a ramp of near-black surfaces. There is no secondary brand hue — every colored pixel is a tint, shade, or alpha of the plasma green.

- **Primary — Plasma Green (#7bff5a):** The sole brand accent. Links, focus rings, active states, the energy in the logo. On dark it reaches ~14:1 against text contexts; reserve it for emphasis, never body text.
- **Primary gradient pair (#53ff9c → #9bff5f):** Mint-to-lime sweep used on the primary button fill and other "live" elements.
- **Primary Soft (#5ae0a6):** Partners with the green in the headline text-gradient.
- **Surface ramp (#030807 → #050b09 → #0a110f → #0c1412):** Deepest edge → page → card → raised. Steps are subtle (<2% luminance); hierarchy comes from hairline `outline` borders, not big jumps.
- **Outline (#1c2b25):** Always-visible 1px hairline. Highlighted/selected elements switch to a green-tinted border (`rgba(123,255,90,.18–.25)`).
- **On-surface (#e8f5ec):** Primary text — a cool, minty off-white, never pure white. **On-surface-variant (#9bb3a5):** muted sage for secondary text.

**Light theme ("Clean Room"):** The green drops to **#16a34a** (4.5:1 on the light surface — `#7bff5a` is unreadable on white at ~1.8:1). Surfaces become mint-tinted off-whites (`#f3f6f4` page, `#ffffff` cards); ink is `#0c1412`. Green halos are replaced by neutral ink shadows. Code surfaces stay dark.

## Typography

Two pinned, self-hosted faces (latin subset). The live marketing site ships system fonts; this system pins webfonts for cross-OS consistency, preserving the system stack as fallback.

- **Inter** — all UI and prose. The closest open match to SF Pro Text (what macOS users saw on the original site). Headlines are large and tightly tracked (hero at `-0.04em`); body relaxes to natural spacing at `line-height: 1.6`.
- **JetBrains Mono** — all code, YAML, CLI output, and terminal mocks. Its engineering register is on-brand; the product is CLI-first and mono type carries real semantic weight here.
- **Labels** (`label-caps`) are 11px, uppercase, `0.15em` tracked — used as tiny section eyebrows above headings (`FEATURES`, `FLOW`).

Sentence case everywhere except proper nouns (HoloDeck, YAML, CLI, FastAPI). Never more than two weights on a screen.

## Layout

A **fixed-max-width** model: content centers within `1120px`. The grid is generous — `56px` between sections, `18–32px` between cards. The hero is a two-column grid (3fr / 2.5fr) that collapses to one column below 900px.

Spacing follows a loose 4/8px rhythm (`xs:4 · sm:8 · md:16 · lg:32 · xl:56`). Cards carry `16px` internal padding; code cards get `16px 18px`. **Prefer flex/grid with `gap`** over inline flow or per-element margins for any row of siblings.

## Elevation & Depth

Depth is tonal and atmospheric, not heavy. The signature device is the **page background itself**: `#050b09` lit by two soft green radial spotlights (`rgba(123,255,90,.09)` at 22%/18% and `.06` at 78%/12%, fading to transparent). Never a linear banner gradient.

Two shadow recipes:

- **Green glow** — `0 14px 40px rgba(83,255,156,.35)` (primary button) and `0 0 26px rgba(123,255,90,.35)` (logo halo). Halo-like, brand-energy.
- **Card depth** — `0 8px 20px rgba(0,0,0,.18)`. Neutral, subtle. The hero code card uses the largest soft glow: `0 20px 50px rgba(16,255,122,.22)`.

No inset bevels, no white inner shadows, no `backdrop-filter` blur on the live site. In light mode, green glows vanish (they need black around them) — substitute neutral ink shadows at ~8% black.

## Shapes

A four-step radius scale. Buttons, badges, and the hero tag are **full capsules** (`999px`). Cards, inputs, and code blocks use `10px` (md); large feature panels and the hero code card use `16px` (lg); small internal chips use `6px` (sm). Mixing sharp and round corners in one view is avoided. Corners are the only softening — forms are otherwise rectangular and engineered.

## Components

- **Button — primary:** Mint→lime gradient fill (`#53ff9c → #9bff5f`), `on-primary` ink label, full-capsule, green glow shadow, weight 500. Hover is `filter: brightness(1.05)` only — no scale, no shadow swell.
- **Button — ghost:** Translucent raised surface (`rgba(12,20,18,.75)`) with a green-tinted hairline; hover deepens the border `.18 → .6`. Used as the secondary CTA.
- **Card:** `surface-container` fill (or a subtle 145° dark-green gradient for feature cards), `outline` hairline, `md` radius, `16px` padding.
- **Code / terminal card:** Darkest surface, `lg` radius, JetBrains Mono. Traffic-light header dots (`#ff5f56 / #ffbd2e / #2ecc71`); `$` prompts in plasma green. **Stays dark in both themes** — a deliberate signature.
- **Tag / eyebrow:** Full-capsule, raised surface, green-tinted hairline, `label-caps` type, often prefixed with a small `◆`.
- **Focus state:** `outline: 2px solid {colors.primary}; outline-offset: 2px`.
- **Disabled:** `opacity: 0.5; cursor: not-allowed`.

## Do's and Don'ts

- **Do** keep the plasma green for emphasis only — links, primary actions, focus, the logo's energy. One green moment per view.
- **Do** keep code and terminal surfaces dark in every theme — it's a brand signature, not an oversight.
- **Do** maintain WCAG AA contrast: use `#16a34a` (not `#7bff5a`) for green text/elements on light backgrounds.
- **Do** lean on the dual-spotlight radial background for atmosphere instead of decorative gradients.
- **Don't** use `#7bff5a` for body text or small text on dark — it vibrates; reserve it for accents.
- **Don't** add emoji to marketing or product UI (acceptable only in GitHub READMEs and mocked CLI output). No Lucide/Heroicons on marketing pages — the live site is icon-free.
- **Don't** recolor or place the logo on a light rectangle — its starfield is baked in; request a transparent/inverted variant for light surfaces.
- **Don't** introduce a second accent hue, animation bounces/springs, or full-bleed gradient backgrounds.
- **Don't** use more than two font weights on a single screen, and keep everything sentence case.
