---
target: Prompt Book canvas mockups
total_score: 29
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 4
target_identity: "file:/Users/justinbarias/Documents/Git/python/agentlab/.claude/worktrees/dependabot-prs-review-670371/.impeccable/mocks/canvas"
timestamp: 2026-11-13T13-30-37Z
slug: impeccable-mocks-canvas
---
Method: dual-agent (A: design review · B: detector + browser)

Heuristics: 1=4 status, 2=3 reviewer jargon, 3=3 no per-cue cancel, 4=2 revision colour has 3 meanings + status vocab, 5=3 no valid_at guard, 6=3 cue bracket recall past 4 cues, 7=2 no keyboard model/permalinks, 8=3 Summary long, config open, 9=4 recovery copy, 10=2 no legend. Total 29/40.

Specificity: chat + Explorer authored (cue margin, next-cue composer, CLOSED stamps, saved-not-live); Summary reverts to BI layout. Detector: 298 findings, 274 design-system drift vs superseded DESIGN.md (false positives); real: low contrast (#1f8a4c 4.4:1, 3.6:1 on selected row; greys #7d8292 #80859a #8d93a6 #7c8193), h1->h3 skips (Summary, Explorer), flat type scale on Summary, cramped segmented padding; browser: text overflow in tool-card headers (div.ch up to 204px) and Compare cells; no viewport meta, 390px untested.

Priority issues:
[P1] Revision colour carries three meanings (versions / compilations / source documents); v1.2.1 buff breaks nth rule. Fix: versions/compilations only; neutral lettered chips for documents; legend in headers.
[P1] Contrast below AA: go green text -> #17703c, placeholder/secondary >= #6b7080, control borders ~#8a90a3.
[P1] Dashboard has no headline; regressions easy to miss; delta vs R15 not previous. Fix: verdict sentence, Regressions list, delta-vs-previous column, numbered rings.
[P1] Reviewer path too technical / trust thin: collapse config, How-to-read disclosure, document name inline in citations, Standby -> Running.
[P2] Keyboard model, 44px targets, fixed widths and header overflow.

Minor: status vocab per surface, unexpected tool call styled as success, icon stroke widths 1.3-1.8, Courier on non-data labels, identical breakdown panels, no dark tokens.
