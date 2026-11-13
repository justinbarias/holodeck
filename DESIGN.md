---
name: HoloDeck
description: The theatre prompt book. A house-blue rail over script-white pages, tool calls as numbered cues in a ruled margin, prompt versions as revision-coloured pages.
colors:
  house: "#1d2347"
  house-raised: "#283063"
  house-active: "#323a74"
  house-control: "#7880bb"
  house-text: "#eef0f9"
  page: "#f5f6f9"
  sheet: "#ffffff"
  well: "#f7f8fb"
  rule: "#dde0e8"
  rule-inner: "#eceef3"
  control-chat: "#868c9f"
  control-dashboard: "#8a90a3"
  ink: "#15171e"
  body-ink: "#2b2e38"
  secondary: "#555b6b"
  quiet: "#6b7080"
  go-green: "#17703c"
  go-green-light: "#1f8a4c"
  standby-amber: "#9a6400"
  amber-light: "#d9a227"
  fail-red: "#c2312a"
  fail-ink: "#8f1f19"
  fail-tint: "#f8d5d2"
  pass-tint: "#e4f3ea"
  warn-tint: "#fbf0d6"
  rev-blue: "#cfe0f8"
  rev-blue-ink: "#234f99"
  rev-pink: "#f9d6e4"
  rev-pink-ink: "#93305e"
  rev-yellow: "#f7e8a6"
  rev-yellow-ink: "#6b5200"
  rev-green: "#d2ecd9"
  rev-green-ink: "#1d6337"
  rev-goldenrod: "#f1d7a4"
  rev-goldenrod-ink: "#6e4700"
  rev-buff: "#e8dcc4"
  rev-buff-ink: "#5b4a2a"
  rev-salmon: "#f6d3c6"
  rev-salmon-ink: "#8a3a1c"
  rev-cherry: "#efc6cd"
  rev-cherry-ink: "#85202f"
typography:
  ui:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  answer:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "15.5px"
    fontWeight: 400
    lineHeight: 1.62
  headline:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 600
  label:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
  cue:
    fontFamily: "Courier Prime, ui-monospace, monospace"
    fontSize: "14px"
    fontWeight: 700
rounded:
  sm: "3px"
  base: "6px"
  md: "8px"
  lg: "10px"
spacing:
  gutter: "22px"
  cue-margin: "76px"
  column: "860px"
  page-x: "36px"
  rail-chat: "264px"
  rail-dashboard: "280px"
  target: "44px"
components:
  tool-card-header:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.secondary}"
    rounded: "{rounded.md}"
    height: "44px"
    padding: "8px 14px"
  button-primary:
    backgroundColor: "{colors.house}"
    textColor: "{colors.sheet}"
    rounded: "{rounded.base}"
    height: "44px"
  rail:
    backgroundColor: "{colors.house}"
    textColor: "{colors.house-text}"
  revision-chip:
    rounded: "{rounded.sm}"
    padding: "3px 7px"
---

# Design System: HoloDeck

Visual source of truth: the approved design canvas, https://claude.ai/artifact/Mk2WJnqcRtTGBqxYhKS9i3 (private to its owner until shared). Where the canvas and the shipped code differ, this file records the code and notes the gap.

Two surfaces share one world: the CopilotKit reference chat (`sample/copilotkit`, tokens in `src/app/globals.css`) and the Dash dashboard (`src/holodeck/dashboard/assets/01-tokens.css`, `02-holodeck.css`). The frontmatter lists light-mode values; dark values are in the Colors section and the sidecar.

## Overview

**Creative North Star: "The Prompt Book"**

Every run is a rehearsal and the screen is the stage manager's prompt book. Tool calls are numbered cues in a ruled margin; prompt versions are coloured revision pages; state is a cue light with a word beside it. The house-blue rail is the theatre around the page; the page itself is script-white and quiet so that evidence (answers, facts, scores) is what you read.

It refuses the category default of a neutral sidebar chat with grey tool chips, and the neon-on-black developer dashboard. Density is moderate and tabular; depth is flat, drawn with rules and tint rather than shadow.

**Key Characteristics:**
- House blue (#1d2347) rail and header ink over a #f5f6f9 page with #ffffff sheets.
- Cues: Q-numbers on a 76 px ruled margin, answer brackets, status lights with words.
- Revision colours only for prompt versions and compilations.
- Public Sans for UI; Courier Prime only where the text is a cue or data.
- Saved results and live runs never look alike.
- Light default; the chat has a dark mode ("house lights down"); the dashboard is light only.

## Colors

A blue-ink and paper palette with three cue lights (go green, standby amber, fail red) and an industry-ordered revision set. Colour never carries state alone.

### Primary
- **House Blue** (`house`, #1d2347): rail, primary buttons, user bubble, focus ring, cue ink, chart baseline series. Hover/link-hover #3d4a99 in the chat, #3d4a99 (`house-hover`) in the dashboard.
- **House Raised / Current** (#283063 / #323a74): agent card, active nav row and thread. Dashboard also uses #3a4278 (rule) and #4a5390 (border) inside the rail.

### Neutral
- **Script White** (`sheet`, #ffffff) sheets, cards, composer; **Page** (#f5f6f9) ground; **Well** (#f7f8fb) code and hover; **Rule** (#dde0e8) and **Inner Rule** (#eceef3) dividers.
- **Ink** (#15171e), **Body Ink** (#2b2e38), **Secondary** (#555b6b), **Quiet** (#6b7080): the floor for secondary text is #6b7080.
- **Control border**: chat #868c9f (`control-chat`), dashboard #8a90a3 (`control-dashboard`). The chat value is a deliberate builder deviation from the canvas #8a90a3, darkened to reach 3:1 against the #f5f6f9 page. The dashboard keeps #8a90a3.
- **Rail chip border** (`house-control`, #7880bb): dashboard rail chip and control borders, 3:1 on both house and house-raised. In the chat the equivalent rail line is #7880b8.

### Cue lights (state)
- **Go** text #17703c, light #1f8a4c (Done / Pass). Tint #e4f3ea.
- **Standby** text #9a6400, light #d9a227, ring rgba(217,162,39,.22) (Running, warnings). Tint #fbf0d6 (dashboard), #fbf3df (chat warn bg, ink #4d3a00).
- **Fail** #c2312a, deep ink #8f1f19, chat bg #fdecea with line #f0c4c1 and text #5c1612, dashboard tint #f8d5d2. The stamp is #c2312a.
- Live light in the chat rail #3fbf6f; error #e5534b.

### Revision colours
In industry order, tint with dark same-hue ink (frontmatter `rev-*`): blue, pink, yellow, green, goldenrod, then buff, salmon, cherry (`REVISION_PALETTE` in `revisions.py`, wraps after eight). The nth prompt version in natural version order takes the nth colour, computed from all loaded runs, never a filtered subset, so a version keeps its colour on every chip, band and chart. A version outside the loaded runs gets neutral #eceef3 / #555b6b.

**The Revision-Only Rule.** Revision colours mark prompt versions and compilations and nothing else. Source documents get lettered neutral chips (`.pb-chip` with a Courier key on #eceef3). Metric lines in charts use house blue and neutral data inks (#1d2347, #3d6fc4, #b07300, #2c7a7b, #6a4fb0, #5d6274), each with its own dash pattern.

**The Words Rule.** State is never shown by colour alone: an icon and a word always accompany it. Chat words: Running, Done, Failed, Stopped (also No hits, Degraded, Not found on memory cards). Dashboard words: Pass, Fail, Regression (also Done, Warning, Not run).

### Dark (chat only, `html.dark`)
Page #12162e, sheet #1b2042, well #161a38, house #171b38, rule #2f3768, control #6f78ad, ink #eef0f9, ink-2 #a9b0d4, body #d3d7ee, focus #aab4f5, action #4651a0, user bubble #3a4389. State: done #5fd08b, running #e8b95a, fail #ff8f86 (bg #3a1d2b). Cards drop their shadow; popovers keep a deeper one. Theme colour meta: #1d2347 light, #12162e dark.

## Typography

**UI Font:** Public Sans (system-ui fallback). **Cue/Data Font:** Courier Prime 400 and 700. Both are self-hosted: the dashboard serves woff2 files from `assets/fonts/` (OFL), the chat uses `next/font` which self-hosts at build with no runtime request. The chat loads Public Sans at 400 and 600, so only those weights exist there.

**Character:** a plain, legible sans for reading, a typewriter face for anything a person might type, quote or compare.

### Hierarchy
- **Headline** (600, 20px, 1.35, -0.01em): dashboard verdict sentence.
- **Title** (600, 17px chat header, 18px dashboard page header and rail brand).
- **Answer** (400, 15.5px, 1.62, max 72ch): assistant prose. User bubble 15px; dashboard body 14px / 1.5.
- **Label** (600, 12px): status words, chips, key labels. Never uppercase, never letter-spaced.
- **Cue** (Courier Prime 700, 14px): Q-numbers, tool names, R-numbers, IDs. Data numbers use Courier Prime or tabular numerals; dashboard figures are Public Sans 26px 600 tabular.

**The Cue-Only Mono Rule.** Courier Prime appears only for cues, tool names, IDs and numbers that are data. Prose, headings, buttons and labels never wear it (no mono costume). Tabular numerals are on for the whole chat body.

## Layout

Chat: CSS grid shell, 264 px house-blue rail plus main. Conversation rows sit on a 76 px cue margin (right border 1 px control colour), 22 px gutter, an 860 px content column, 36 px page padding. Below 900 px the rail becomes a drawer. Tool cards use container queries (560 and 400 px) to wrap.

Dashboard: 280 px rail (a compact 44 px icon rail exists), a page header, then sheets. Summary opens with the verdict sentence and a numbered regressions list, then four figures with deltas (two columns on narrow widths), then the pass-rate chart over revision bands, then a runs table. Runs are numbered R1 (oldest) to Rn (newest); the timestamp remains the id.

## Elevation & Depth

Flat. Cards are 1 px #dde0e8 borders on white with a hairline `0 1px 2px rgba(20,24,40,.05)` in light and none in dark. Popovers use `0 12px 30px rgba(20,24,40,.15)`. Selected rail chips add a 2 px double ring, not a shadow. Depth otherwise comes from tint steps (page, sheet, well). Motion is 180 ms ease-out on disclosure height, opacity and chevron, and only under `prefers-reduced-motion: no-preference`.

## Shapes

Small, ruled forms: 3 px chips and revision tags, 6 px controls, 8 px cards and agent card, 10 px panels. Bars are near-square (1 px radius). Lights are 8-9 px circles. User bubble is 12/12/3/12 px. Answer brackets are 1.5 px open rectangles. The red stamp is a 2 px-radius, 1.5 px outlined Courier tag rotated -1.5 degrees.

## Components

### Cue system
- **Q-numbers:** every tool call gets Q1, Q2... in the margin, Courier 700 14px in cue ink, with a 9 px status light (green, amber with ring while running, red on failure, neutral when empty).
- **Answer bracket:** a 1.5 px open bracket on the margin rule spanning the cues an answer drew on. Under the answer: "After Q1 Q2" (quiet word, Courier numbers).
- **Composer cue:** the composer's left margin shows the next cue (Q{n}) in #6b7080 Courier, with the note "The next tool call will be cue Q{n}."
- **Runs:** dashboard runs are R1...Rn in Courier; the same number appears in tables, chips and chart rings.

### Tool card (progressive disclosure)
- Collapsed it is one header row and a real button, min-height 44 px: Courier tool name (max 32ch), a one-line ellipsized headline, status word with icon, an optional duration bar, chevron. `aria-expanded` and `aria-controls` are set; the body is `inert` when closed.
- Cards stay collapsed when they finish; the user opens them. Open state is kept per call in a disclosure store.
- More than three consecutive calls collapse into a group row: Q-range (for example Q2-Q6), the tool names summarised, "in tools" total time, one status word. Opening it lists the individual cards.
- Duration bar: 120 px by 5 px track on one scale per thread, number beside it as text. No duration is shown when timing was not measured (under 16 ms, or history with no timing), never "0 ms".
- Memory timeline: opens to its fact list first; the chart is behind a "Show timeline chart" / "Hide timeline chart" button. Closed facts preview, then "Show n more closed". Retracted or superseded facts are struck and carry the red stamp. Facts and bars use house ink (open) and #a3a9bd (closed), hollow when a bar does not hold at the as-of date.
- Failed cards use a #f0c4c1 border and red headline plus the Failed word. Unknown tools use the generic card (arguments headline, Done / Failed / Stopped).

### Buttons
Primary house blue with white text; secondary and outline with the control border; danger; icon buttons. Radius 6 px. Chat default and composer field are 44 px; see gaps for the 36 and 40 px exceptions. Focus is a 2 px house outline (rail: #eef0f9) with 2 px offset. Icons are inline SVG, 1.5 stroke, `currentColor`, always beside text or with an accessible name.

### Chips
- **Revision chip** (dashboard `.rv`): tinted fill, same-hue ink, 12px 600, 3px radius, "Prompt version <v>" title. Rail version chips keep their tint, pressed adds a ring and a check.
- **Source chip** (chat): #eceef3 fill, 3 px radius, a lettered white Courier key (18 px) plus the source name.

### Dashboard verdict and regressions
- A verdict sentence (20px 600) about the latest run in view against the run before; detail line with cue numbers in 700.
- Regressions: a numbered list, number in red 700, run label, version, "58.3% -> 50.0% · case failed". A regression is a pass-rate drop of more than four points against the previous run in view. A Delta vs previous column accompanies runs.
- Pass-rate chart: translucent revision bands behind contiguous version runs, house-blue line, and hollow red rings (`circle-open`) numbered to match the list.
- Rail: "Saved results · nothing runs here" with a saved icon, so saved never reads as live. The chat rail's live light carries the word (Connecting, error) beside it.
- State badges: icon, word, tint fill (pass #e4f3ea, fail #f8d5d2).

### Rail and navigation
House-blue column; brand, agent card (#283063), views or threads (with cue counts), "new" action. Active row #323a74 with white 600 text. Version filters are revision chips. Chat has a "How to read this" key in the header explaining cues, brackets and lights.

## Do's and Don'ts

### Do:
- **Do** pair every state colour with its icon and word (Running, Done, Failed, Stopped; Pass, Fail, Regression).
- **Do** keep text at WCAG 2.2 AA: status text #17703c, secondary text no lighter than #6b7080, control borders at 3:1 or better (chat #868c9f, dashboard #8a90a3, rail chips #7880bb).
- **Do** give interactive rows a 44 px target and a visible 2 px focus outline.
- **Do** compute revision colours from the full set of loaded runs so a version keeps its colour everywhere.
- **Do** keep saved results visibly saved ("Saved results · nothing runs here") and keep live indicators for the running agent only.
- **Do** omit a duration when it was not measured.
- **Do** draw icons as inline SVG with 1.5 stroke.

### Don't:
- **Don't** use eyebrow labels or small uppercase kickers above headings.
- **Don't** use a coloured border-left wider than 1 px as an accent stripe (the only left-edge rules are a 1.5 px dashed threshold marker in a chart and 1.5 px bracket strokes, which are data marks).
- **Don't** use glyph or emoji icons; use the inline SVG set.
- **Don't** use gradient text.
- **Don't** set prose, buttons or labels in Courier Prime.
- **Don't** use revision colours for source documents, metrics, status or decoration.
- **Don't** let a saved result borrow the live light or running animation.
- **Don't** rely on colour for pass/fail; do not use hard offset shadows.

## Known gaps (code vs contract)

- **Keyboard model (P2, open):** no roving tabindex in Explorer lists and no `/` to search exists in the dashboard or chat sources; only fact bars and hits use `tabIndex`.
- **44 px targets (P2, partial):** tool-card headers, group rows, rail close, rail nav and dashboard buttons reach 44 px. Still under: chat `.pb-btn--small` and danger 36 px, thread rows 40 px, some links 28 and 36 px, dashboard `.rchip` 36 px, timeline lane rows 28 px.
- **Header revision tabs:** not built in the chat (the canvas Main shows compilation tabs). No tool result carries a version field the generic chat could read; revision colours stay reserved for dashboard prompt versions until one does.
- **Dashboard dark mode:** none; chat only.
- **Control border differs by surface:** chat #868c9f versus dashboard #8a90a3.
- Chat Public Sans is loaded at 400 and 600 only; dashboard serves the variable font.
