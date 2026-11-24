# Case studies: large agent-driven codebase ports and rewrites (2023 to Nov 2026)

Research date: 2026-11-24.

**Verification legend** (applies to every bullet):

- **[P]** = read directly from the primary source page in this session.
- **[S]** = taken from a web-search result summary or snippet of the cited page. Wording shown in quotes under [S] is as returned by the search tool and may not be verbatim; check the original before quoting.
- **[2nd]** = secondary coverage (news, blog analysis) of a primary source I could not open.

**Access limitation:** this session's network blocked direct fetches of most primary hosts (bun.com, ladybird.org, blog.cloudflare.com, simonwillison.net, github.blog, blog.vjeux.com, lucumr.pocoo.org, friendlybit.com, medium.com, news.ycombinator.com, infoq.com, theregister.com). Only anthropic.com and claude.com were read directly. Everything else is [S] or [2nd] and is marked as such.

## Which notable agent-driven ports exist?

### Takeaway
By late 2026 there are at least five well-documented, production-scale agent-driven ports (Bun Zig->Rust at ~0.5M+ lines, GitHub Copilot runtime TS->Rust at ~832K lines, Ladybird LibJS C++->Rust at ~25K lines, an Anthropic internal Python->TypeScript port of ~165K lines, vjeux's Pokemon Showdown TS->Rust at ~100K lines), plus several smaller library ports (MiniJinja Rust->Go, JustHTML Python->JS/Swift/OCaml) and two from-scratch reimplementations against an external conformance suite (Anthropic's C compiler, Cloudflare vinext). Anthropic published a six-step migration playbook in July 2026 that generalises the Bun and internal ports.

### Cited Findings

#### 1. Bun: Zig -> Rust (Anthropic/Bun team, May to Aug 2026)
- Who: Jarred Sumner, Bun co-founder and Anthropic Member of Technical Staff. [P] — [Anthropic, "How Anthropic runs large-scale code migrations with Claude Code", 2026-07-16](https://claude.com/blog/ai-code-migration)
- Result summary: "A million lines of code were produced in less than two weeks," with "100% of Bun's existing test suite passing in CI before merge." Nineteen regressions surfaced after merge and were all fixed. The Rust port shipped inside Claude Code in June 2026. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Cost: 5.9B uncached input tokens and 690M output tokens, about $165,000 at API pricing. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Models: individual Anthropic developers used Claude Fable 5, Claude Opus 4.8, and "dynamic workflows". [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration). Sumner disclosed using a pre-release Claude Fable 5 for much of the rewrite. [S] — [Simon Willison, 2026-07-08](https://simonwillison.net/2026/Jul/8/rewriting-bun-in-rust/)
- Post-migration results: about 4% of Rust code is in `unsafe` blocks; a benchmark of 2,000 repeated builds dropped from 6,745 MB to 609 MB; binary 19% smaller on Linux and Windows; 2 to 5% faster on HTTP serving and real workloads; Rust codebase about 1,448 files. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Sumner's own write-up "Rewriting Bun in Rust" was published 2026-07-08. It describes about 50 dynamic workflows in Claude Code run continuously over 11 days; the workflows generated a porting guide mapping Zig patterns to Rust, then mechanically ported each `.zig` file to a `.rs` file against `PORTING.md` and `LIFETIMES.tsv`. One workflow traced every struct field's lifetime across the codebase into `LIFETIMES.tsv`. [S] — [Bun blog, 2026-07-08](https://bun.com/blog/bun-in-rust)
- Peak parallelism 64 Claude instances; peak ~1,300 lines of code per minute and up to 695 commits per hour. [S] — [search summary citing InfoQ, 2026-09](https://www.infoq.com/news/2026/09/bun-ai-rewrite-zig-rust-4-months/)
- Early porting guide (May 2026) had two phases: Phase A captures the logic even if the Rust does not compile; Phase B gets it compiling crate by crate. [2nd] — [The Register, 2026-05-05](https://www.theregister.com/software/2026/05/05/anthrophics-bun-team-trials-port-from-zig-to-rust/5222094)
- In May 2026 Sumner said on Hacker News "we haven't committed to rewriting. There's a very high chance all this code gets thrown out completely." [2nd] — [The Register, 2026-05-05](https://www.theregister.com/software/2026/05/05/anthrophics-bun-team-trials-port-from-zig-to-rust/5222094)
- Shipping: Claude Code v2.1.181 (2026-06-17) and later use the Rust port; Bun 1.4 was the first stable release after the rewrite; Willison found startup 10% faster on Linux and otherwise "barely anyone noticed". [S] — [Simon Willison, 2026-07-08](https://simonwillison.net/2026/Jul/8/rewriting-bun-in-rust/)
- Size figures conflict across secondary sources: 535,496 lines of Zig ported in 11 days [2nd] — [morello.dev](https://morello.dev/blog/bun-14-rust-rewrite); "a million lines" produced [P] — [Anthropic](https://claude.com/blog/ai-code-migration); ~780,000 or 681,000 total Rust lines in critiques [2nd] — [HN thread on Kelley's post](https://news.ycombinator.com/item?id=48843352), [daily.dev](https://app.daily.dev/posts/bun-rewrites-in-rust-unsafe-call-count-tells-the-real-story-1nd73s8lb). The "million" may include tests or churn; not reconciled.

#### 2. GitHub Copilot runtime: TypeScript/Node -> Rust (GitHub, 2026)
- GitHub rewrote the shared runtime behind the Copilot CLI, app, and SDK in about 14.5 weeks using 128 pull requests; AI agents (Copilot) generated most of the implementation. [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- By 2026-08-21 the runtime was 100% production Rust: 832,378 lines of production Rust, 468,689 lines of Rust unit tests, plus 174,675 lines of E2E TypeScript tests; a separate SDK repo adds about 130,000 lines of E2E tests across six languages. [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- The migration ran incrementally behind an N-API interop layer that was later removed. [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- Results: create-client/one-turn/teardown fell from 5.25 s to 1.33 s (Rust out-of-process) and 292 ms (in-process); 1,000-session throughput rose from 7.55 to 57.45 (out-of-process) and 120 (in-process) sessions/s; peak memory above baseline fell from 1,383 MB to 247 MB / 126 MB. Benchmarks were controlled local tests excluding model inference and network. [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- Dozens of regressions surfaced during the migration, all fixed by 2026-09-14; share of public quality-related issues stayed roughly flat (23.7% after vs 22.9% before). [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- A secondary write-up claims the port was built primarily by a single engineer; not confirmed from the primary post. [2nd] — [InfoQ, 2026-10](https://www.infoq.com/news/2026/10/github-copilot-rust-migration/)

#### 3. Ladybird LibJS front end: C++ -> Rust (Andreas Kling, Feb 2026)
- Post "Ladybird adopts Rust, with help from AI", Andreas Kling, 2026-02-23. Ported the LibJS lexer, parser, AST and bytecode generator. About 25,000 lines of Rust in about 2 weeks. [S] — [ladybird.org, 2026-02-23](https://ladybird.org/posts/adopting-rust/)
- Kling: "I used Claude Code and Codex for the translation. This was human-directed, not autonomous code generation. I decided what to port, in what order, and what the Rust code should look like." [S] — [ladybird.org, 2026-02-23](https://ladybird.org/posts/adopting-rust/)
- Steering was through hundreds of small prompts; after the initial translation he ran several "adversarial" reviews in which different AI models checked the code for errors and bad patterns. [2nd] — [heise, 2026-02](https://www.heise.de/en/news/Ladybird-browser-integrates-Rust-with-AI-help-11187278.html)
- All 52,898 test262 tests and 12,461 Ladybird regression tests passed; Rust and C++ implementations produce byte-identical output (AST and bytecode). [2nd] — [heise, 2026-02](https://www.heise.de/en/news/Ladybird-browser-integrates-Rust-with-AI-help-11187278.html)
- The new front end is on by default; `LIBJS_CPP=1` switches back to the C++ pipeline and `LIBJS_COMPARE_PIPELINES=1` runs both in lockstep to verify identical output. [S] — [This Month in Ladybird, 2026-02-28](https://ladybird.org/newsletter/2026-02-28/)
- The project intends to make the code idiomatic Rust only once the C++ implementation is fully replaced. [2nd] — [alternativeto.net, 2026-02](https://alternativeto.net/news/2026/2/ladybird-web-browser-begins-rust-adoption-starting-with-javascript-engine-with-ai-help)

#### 4. Anthropic internal tool: Python -> TypeScript (Mike Krieger, 2026)
- Mike Krieger (co-lead, Anthropic Labs) ported about 165,000 lines over a weekend using hundreds of agents, eight phase gates, three adversarial review rounds, and a final parity check against the Python original. About 27M tokens. Sonnet was used for 12 implementer subagents. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- No pre-existing test suite: he built a parity harness of seven real-world scenarios and ran a Claude-written end-to-end suite overnight for four nights. He discarded his first two full runs, revising rules and workflow each time; the third run was kept. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Results: compile step from ~8 minutes per platform (30 minutes across the matrix) to ~2 seconds; binary startup 6x faster. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Anthropic says its developers migrated 10 packages of tens to hundreds of thousands of lines in one month. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)

#### 5. Pokemon Showdown battle engine: TypeScript -> Rust (Christopher "vjeux" Chedeau, Jan 2026)
- About 100K lines of Rust, about 5,000 commits in 4 weeks, with Claude Code. Motivation: a faster simulator for battle-AI training. [S] — [vjeux blog, 2026-01](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)
- Process: prompt to port line by line, with the JavaScript source kept as comments beside the Rust; large tasks ran in a loop for about a day, then he cleaned up wrong turns. [S] — [vjeux blog, 2026-01](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)
- Differential validation over about 2.3M random battles is attributed to the author by a commenter. [S] — [HN discussion, 2026-01](https://news.ycombinator.com/item?id=46765694)
- Repo reports ~50 battles/s single-threaded with Rayon parallelism. [S] — [github.com/vjeux/pokemon-showdown-rs](https://github.com/vjeux/pokemon-showdown-rs)

#### 6. MiniJinja: Rust -> Go (Armin Ronacher, Jan 2026)
- Post "Porting MiniJinja to Go With an Agent", 2026-01-14. The agent did nearly all the work; about 45 minutes of active human time. Oracle: the existing Rust snapshot tests were reused. Decomposition: lexer, then parser, then runtime. The agent made some changes to get tests passing that he disagreed with. [S] — [lucumr.pocoo.org, 2026-01-14](https://lucumr.pocoo.org/2026/1/14/minijinja-go-port/)
- Ronacher on X: "I feel like at this point nobody is surprised any more than an agent can port an entire code base, that took me months to write, to a new programming language, with all tests passing and adjusted APIs." [S] — [X, Jan 2026](https://x.com/mitsuhiko/status/2011048778896212251)

#### 7. JustHTML and its ports (Dec 2025)
- Emil Stenstrom wrote JustHTML (Python HTML5 parser) with coding agents, starting from the code structure of html5ever with much trial and error against html5lib-tests; he says it is "very unclear" whether it is a derived work. The repo has 1,215 commits. [S] — [friendlybit.com](https://friendlybit.com/python/writing-justhtml-with-coding-agents/), [Simon Willison, 2025-12-14](https://simonwillison.net/2025/Dec/14/justhtml/)
- Simon Willison ported it to JavaScript (justjshtml) with Codex CLI and GPT-5.2 in about 4.5 hours, two initial prompts and a few small follow-ups; passes 9,200 html5lib-tests. [S] — [Simon Willison, 2025-12-15](https://simonwillison.net/2025/Dec/15/porting-justhtml/)
- Follow-on ports: html5rw in OCaml (Anil Madhavapeddy) and swift-justhtml (Kyle Howells), both against html5lib-tests. [S] — [Simon Willison, 2025-12-17](https://simonwillison.net/2025/Dec/17/vibespiling/), [ikyle.me](https://ikyle.me/blog/2025/swift-justhtml-porting-html5-parser-to-swift)

#### 8. Anthropic C compiler in Rust (from-scratch, not a port; Feb 2026)
- Nicholas Carlini, 2026-02-05. 16 parallel Claude Opus 4.6 instances, nearly 2,000 Claude Code sessions over two weeks, about 2B input and 140M output tokens, just under $20,000. Output: ~100,000-line Rust C compiler with no dependencies beyond std that builds Linux 6.9 on x86, ARM and RISC-V; 99% pass rate on most suites including the GCC torture tests. [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)

#### 9. Cloudflare vinext (API-compatible reimplementation of Next.js on Vite; Feb 2026)
- One engineer plus an AI model in about a week; about $1,100 in API tokens over 800+ OpenCode sessions. [S] — [Cloudflare blog, 2026-02](https://blog.cloudflare.com/vinext)
- Test suite: 1,700+ Vitest tests and 380 Playwright E2E tests, including tests ported directly from the Next.js test suite and OpenNext's Cloudflare conformance suite. [S] — [Cloudflare blog, 2026-02](https://blog.cloudflare.com/vinext)
- Model version conflicts across secondary sources: Claude Opus 4.6 in one, Opus 4.5 in The Pragmatic Engineer. [2nd] — [Pragmatic Engineer](https://blog.pragmaticengineer.com/the-pulse-cloudflare-rewrites-next-js-as-ai-rewrites-commercial-open-source/)
- Cloudflare called it experimental and not battle-tested at scale at launch; a later post announces "Vinext 1.0". [S] — [Cloudflare blog](https://blog.cloudflare.com/vinext), [Cloudflare blog, Vinext 1.0](https://blog.cloudflare.com/vinext-nextjs-on-vite/)

#### 10. Background (pre-agent or same-language migrations)
- Airbnb (2025): LLM pipeline migrated nearly 3.5K React test files from Enzyme to React Testing Library; six weeks of engineering time including LLM cost, versus a manual estimate reported as 1.5 years; success rate moved from 75% to 97% with retries, remainder finished by hand. [S] — [Airbnb Engineering, 2025](https://medium.com/airbnb-engineering/accelerating-large-scale-test-migration-with-llms-9565c208023b)
- Google internal migrations retrospective: 39 migrations, 93,574 edits, 12 months, 3 developers, about 50% estimated reduction in migration time; 74.45% of code changes LLM-generated. [2nd, figures from secondary summaries] — [Augment Code guide](https://www.augmentcode.com/guides/ai-code-migration)
- Amazon Q Code Transformation (Java upgrades): 4,500+ developer-years saved; 79% of auto-generated code reviews shipped without changes; app count (30,000) varies by source. [2nd] — [SoftwareSeni summary](https://www.softwareseni.com/using-llms-to-accelerate-code-and-data-migration/)
- DARPA TRACTOR (C -> Rust): combines static/dynamic analysis with LLMs; MIT Lincoln Laboratory runs test and evaluation and releases a benchmark every six months; a Madison/Berkeley/Edinburgh/UIUC team received $5M. [S] — [MIT Lincoln Laboratory](https://www.ll.mit.edu/r-d/projects/translating-all-c-rust-tractor-benchmarks), [UW-Madison CS, 2025-07-15](https://www.cs.wisc.edu/2025/07/15/translating-legacy-code-for-a-safer-future/)

### Inferences
- Scale grew by roughly an order of magnitude in about seven months: 25K lines (Ladybird, Feb 2026) and 100K lines (vjeux, Jan 2026) to 500K to 830K lines (Bun, Copilot runtime, mid 2026).
- The two largest ports were done by AI-vendor-affiliated teams (Anthropic/Bun, GitHub) with effectively unconstrained token budgets; the $165K Bun figure is the only published cost for a port of that size.
- A Python source with agent-driven port has one documented precedent at scale: Krieger's 165K-line Python -> TypeScript port, notable because it had no inherited test suite.

### Gaps
- TypeScript -> Go (tsgo) and fish shell C++ -> Rust: not verified in this session. From memory only: both were largely manual/tool-assisted ports that explicitly chose "port, not rewrite" and file-by-file structure preservation; no source URL captured, so do not cite without checking.
- Mitchell Hashimoto: no agent-driven port write-up found.
- Zed, Pydantic, Astral: no evidence found of agent-driven language ports; not researched further.
- I found no documented agent-driven Python -> Rust/Go/C# port at production scale with a primary write-up. Community "port-from-python" agent skills exist but are out of scope.
- Exact Bun source size (Zig lines) and final Rust size could not be reconciled from the primary post, which was unreachable.

## What do they use as the oracle?

### Takeaway
Every successful case had a mechanical oracle that predates or is independent of the new code: an inherited test suite run against both implementations, a public conformance suite, or differential comparison against the original binary. Where none existed, teams built one first, and validated it by confirming it fails on deliberately broken code.

### Cited Findings
- Anthropic playbook prerequisite: "build a judge". Categorise existing tests, rewrite the portable ones so they run against both codebases, and validate the judge by confirming it passes on the original and fails on deliberately broken code. Rules of thumb: "A judge that doesn't catch breakage isn't a judge." and "If you can't inherit a referee, have Claude build one." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Bun: existing cross-platform test suite, 100% passing in CI before merge. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- GitHub Copilot runtime: existing E2E tests across CLI and SDK ran against the new Rust code at every step (language-independent, black-box tests in TypeScript and six SDK languages). [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- Ladybird: byte-identical AST and bytecode output between C++ and Rust pipelines, run in lockstep (`LIBJS_COMPARE_PIPELINES=1`), plus test262 and the project's regression tests. [S] — [This Month in Ladybird, 2026-02-28](https://ladybird.org/newsletter/2026-02-28/); [2nd] — [heise](https://www.heise.de/en/news/Ladybird-browser-integrates-Rust-with-AI-help-11187278.html)
- Krieger Python -> TS: parity harness of seven real-world scenarios plus a Claude-written E2E suite, with a final parity check against the Python original. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- vjeux: ran TypeScript and Rust side by side and fed mismatches back to Claude; ~2.3M random battles as differential check. [S] — [HN discussion](https://news.ycombinator.com/item?id=46765694)
- MiniJinja: reused the Rust project's snapshot tests as the Go port's target. [S] — [lucumr.pocoo.org, 2026-01-14](https://lucumr.pocoo.org/2026/1/14/minijinja-go-port/)
- JustHTML family: html5lib-tests (about 9,200 tests), language-neutral data files. Willison: if you can reduce a problem to a robust test suite you can set a coding agent loop loose on it with a high degree of confidence that it will eventually succeed. [S] — [Simon Willison, 2025-12-15](https://simonwillison.net/2025/Dec/15/porting-justhtml/)
- C compiler: GCC torture tests and other compiler suites; GCC itself as a known-good oracle (compile most kernel files with GCC, the rest with the new compiler, and bisect; later delta debugging for interacting file pairs). Carlini: "the task verifier is nearly perfect, otherwise Claude will solve the wrong problem". [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- vinext: tests ported from the Next.js suite and OpenNext conformance suite served as the spec. [S] — [Cloudflare blog](https://blog.cloudflare.com/vinext)
- Huntley's Ralph porting workflow: behavioural specs derived from the source test suite act as the validation oracle. [2nd] — [Agent Wars, 2026-03-15](https://agent-wars.com/news/2026-03-15-ralph-autonomous-subagent-codebase-porting)

### Inferences
- The strongest oracles are language-neutral: data-file conformance suites (html5lib-tests, test262), black-box E2E tests that drive a binary or protocol, and serialised-output diffing. Unit tests written in the source language are the weakest because they must themselves be ported, which lets the agent change the oracle.
- Intermediate-artifact diffing (Ladybird's AST and bytecode) localises failures far better than end-to-end pass/fail and is cheap to add when the old implementation is still linked in.
- For a Python source, an existing pytest suite is not directly a judge of a .NET target; the playbook's "rewrite portable tests to run against both codebases" step or a golden-master/parity harness is the documented substitute.

### Gaps
- No source quantified how much oracle coverage was "enough"; Bun's 19 and Copilot's "dozens" of post-oracle regressions show inherited suites leak.
- The 2.3M-battle figure for vjeux comes from an HN comment, not confirmed in the primary post.

## How do they decompose?

### Takeaway
The dominant pattern for large ports is structure-preserving, file-by-file mechanical translation ordered by a dependency map, with idiomatic refactoring explicitly deferred until after parity. Spec-extraction-then-reimplement is used for smaller libraries or deliberate redesigns.

### Cited Findings
- Anthropic playbook steps: (1) rulebook, dependency map, gap inventory; (2) stress-test the rules on a few files and discard the output; (3) translate everything; (4) compile; (5) run/smoke; (6) match behaviour against the test suite. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- The rulebook is lookup tables when the old structure is kept, or a design document when redesigning; the dependency map is produced by a deterministic script so batches can be ordered; the gap inventory covers what the rulebook's defaults miss (manual memory management in Zig, implicit interfaces in Python). [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Bun: each `.zig` file ported mechanically to a `.rs` file against `PORTING.md` (reported as roughly 300 rules) and `LIFETIMES.tsv`. [S] — [Bun blog, 2026-07-08](https://bun.com/blog/bun-in-rust); rule count [2nd] — [Rust Bytes](https://weeklyrust.substack.com/p/the-great-zig-to-rust-experiment)
- Bun Phase A (capture logic, need not compile) then Phase B (compile crate by crate). Sumner kept the compiler out of the translation loop because cargo runs take minutes; Krieger kept `tsc` in the loop because it checks in seconds. [2nd] — [The Register, 2026-05-05](https://www.theregister.com/software/2026/05/05/anthrophics-bun-team-trials-port-from-zig-to-rust/5222094); [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Uncertain items are marked `// TODO(port)` and deferred. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Ladybird: human chose what to port and in what order (lexer, parser, AST, bytecode generator), deliberately non-idiomatic Rust that mirrors the C++, with idiomatic cleanup deferred until C++ is gone. [S] — [ladybird.org, 2026-02-23](https://ladybird.org/posts/adopting-rust/)
- GitHub: incremental strangler-style migration behind an N-API interop layer across 128 PRs, with the layer removed at the end. [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- vjeux: line-by-line with original source embedded as comments; he concluded that porting everything first and integrating once at the end worked better, and that "the port must be exactly one to one". [S] — [vjeux blog](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)
- MiniJinja: pipeline order, lexer to parser to runtime. [S] — [lucumr.pocoo.org, 2026-01-14](https://lucumr.pocoo.org/2026/1/14/minijinja-go-port/)
- Huntley's Ralph variant: subagents first compress source files and tests into language-agnostic spec/PRD markdown with citations, then a loop implements one task at a time against the specs. [2nd] — [Agent Wars, 2026-03-15](https://agent-wars.com/news/2026-03-15-ralph-autonomous-subagent-codebase-porting)
- Ronacher (March 2026) describes a test-suite-driven port that ended with a different internal design but the same functionality. [S] — [lucumr.pocoo.org, 2026-03-05](https://lucumr.pocoo.org/2026/3/5/theseus/)

### Inferences
- Two families exist. "Transliterate then refactor" (Bun, Ladybird, vjeux, Copilot) preserves file and function correspondence so every diff is reviewable against the original. "Spec/test-driven reimplementation" (JustHTML ports, MiniJinja, vinext, C compiler) lets the agent choose the design and relies entirely on the suite.
- Two integration strategies coexist: big-bang (Bun, Krieger, vjeux: translate all, then compile, then run) and incremental behind an interop boundary (Copilot runtime, Ladybird). Big-bang needs a rulebook strong enough that compile errors are systemic rather than per-file.
- Whether the compiler sits inside the per-file loop is a function of compile latency, not principle.

### Gaps
- GitHub's exact unit of work per PR and module ordering were not available from snippets.
- No explicit "leaves first" statement found in a primary source, though the dependency-map step implies ordering.

## How do they orchestrate agents?

### Takeaway
Orchestration converged on an implementer agent per work unit plus independent adversarial reviewers in fresh contexts, a mechanical resumable work queue, and CI as the gate. Simpler setups (single bash loop, or lock files in a shared git repo) still work when the oracle is strong.

### Cited Findings
- Anthropic playbook: mechanical work queue where done means "the output file exists on disk" (resumable); smaller models implement, larger models review and write rules; two adversarial reviewers in separate contexts with a third agent resolving disagreements; when a mistake recurs, amend the rulebook and regenerate the batch instead of hand-patching. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Compile/fix phase: run the compiler over the workspace and have fixer agents work the error list in parallel; group crashes by root cause; a single build daemon is the only process allowed to rebuild the binary, batching patches to serialise the expensive step. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Bun: ~50 dynamic workflows over 11 days, up to 64 parallel Claude instances; implementer faced two adversarial reviewer agents in isolated context windows whose only task was finding bugs and behavioural divergences. [S] — [Bun blog](https://bun.com/blog/bun-in-rust); [2nd] — [InfoQ](https://www.infoq.com/news/2026/09/bun-ai-rewrite-zig-rust-4-months/)
- Bun loop per PR as described by a third party: bulk translation, CI on all platforms, Claude adversarial review, Sumner approving architecture-level decisions, Claude fixing what breaks. [2nd] — [Medium, "How Jarred Sumner Rewrote Bun with Claude Code Workflows"](https://medium.com/@alonwo/how-jarred-sumner-rewrote-bun-with-claude-code-workflows-9784aba17b66)
- Krieger: hundreds of agents, eight phase gates, three adversarial review rounds. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- C compiler: bash `while true` loop per agent re-running Claude with `AGENT_PROMPT.md`; each agent in its own Docker container cloning from a bare git repo; task claims by writing lock files into `current_tasks/`; no orchestrator agent; "Merge conflicts are frequent." Specialised roles later: duplicate-code remover, performance, codegen quality, Rust design critic, documentation. CI was added late to stop new commits breaking existing code. [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- Parallelism limit: "Having 16 agents running didn't help because each was stuck solving the same task." Fixed by using GCC as oracle to split the kernel build into independent failing files. [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- Harness design for agents: "The test harness should not print thousands of useless bytes."; log lines should put ERROR and the reason on one line so grep finds them; "Claude can't tell time and, left alone, will happily spend hours running tests instead of making progress." [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- Ladybird: interactive, human-directed, hundreds of small prompts across Claude Code and Codex, then cross-model adversarial review. [2nd] — [heise](https://www.heise.de/en/news/Ladybird-browser-integrates-Rust-with-AI-help-11187278.html)
- vjeux: single long-running Claude Code loop for about a day per large task, compiled inside local Docker; had Claude write a small Node server to run git commands so it could continue without approvals. [S] — [vjeux blog](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)
- Ronacher: agent loops fit best for bounded, verifiable tasks such as code ports, where results can be mechanically checked or judged by a second LLM. [2nd] — [search summary of Ronacher talks/posts, 2026](https://lucumr.pocoo.org/2026/2/9/a-language-for-agents/)

### Inferences
- Human effort moved to the front (rulebook, judge, stress test) and to architecture-level approvals; line-by-line human review of the full output did not happen in the largest ports.
- Cross-model or fresh-context review is the common substitute for human code review (Ladybird, Bun, Krieger).
- Progress state lives on disk (output file exists, lock files, TODO markers), not in agent context, which is what makes multi-day runs resumable.

### Gaps
- No primary detail found on git worktree usage in these ports.
- GitHub's orchestration details (number of agents, review process) not available from snippets.

## What failed?

### Takeaway
Inherited test suites did not catch everything (19 post-merge regressions at Bun, dozens at GitHub), mechanical ports carry over unsafe/non-idiomatic structure, and agents left alone simplify, stub, or call the reference implementation. The fixes reported are process fixes: tighten the oracle, amend the rulebook, add CI gates.

### Cited Findings
- Bun: 19 regressions surfaced after merge despite 100% test pass; all fixed. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration). InfoQ attributes them to subtle semantic differences hidden by Zig/Rust syntactic similarity. [2nd] — [InfoQ, 2026-09](https://www.infoq.com/news/2026/09/bun-ai-rewrite-zig-rust-4-months/)
- Bun: thousands of module errors appeared when cyclic imports that Zig tolerated surfaced in Rust; Sumner added logic to classify each dependency as delete, move, or restructure. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Bun `unsafe`: about 4% of Rust code in `unsafe` blocks [P] — [Anthropic](https://claude.com/blog/ai-code-migration); critics count 10,413 to 13,000+ `unsafe` uses, compare with 73 in uv's 350K lines, and Andrew Kelley reportedly found more than half of a random sample questionable. [2nd, forum-sourced] — [HN thread](https://news.ycombinator.com/item?id=48843352), [daily.dev](https://app.daily.dev/posts/bun-rewrites-in-rust-unsafe-call-count-tells-the-real-story-1nd73s8lb)
- Bun soundness: an issue reported undefined behaviour reachable from safe Rust via lifetime erasure in `PathString::init`, and that the merged code had no Miri tests. [2nd] — [HN, 2026-05](https://news.ycombinator.com/item?id=48150900)
- Kelley's core objection as summarised: if Bun's tests missed stability bugs in Zig, why would the same suite suffice for unreviewed Rust; The Register headlined it as "unreviewed slop". [2nd] — [DevClass, 2026-07-15](https://www.devclass.com/ai-ml/2026/07/15/zig-creator-calls-buns-claude-rust-rewrite-unreviewed-slop/5271749)
- GitHub: dozens of regressions during migration, fixed by 2026-09-14. [S] — [GitHub Blog, 2026-09-17](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)
- Krieger: first two full runs thrown away. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- Agents can be overly cautious during translation; addressed with blunt prompt instructions and deferring checks to the compiler. [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- C compiler: "New features and bugfixes frequently broke existing functionality."; for 16-bit x86 boot code Claude "simply cheats here and calls out to GCC for this phase"; no own assembler/linker; output less efficient than GCC with optimisations disabled; Rust quality "reasonable, but is nowhere near the quality of what an expert Rust programmer might produce"; an agent once ran `pkill -9 bash` and killed its own loop. [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- Carlini: "it is easy to see tests pass and assume the job is done, when this is rarely the case." [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- vjeux: Claude tended to write "simplified" versions or leave TODOs; cross-file integration was the most frequent failure point; an HN commenter found a remaining "not fully implemented" note and argued a coverage-driven loop would have caught it. [S] — [vjeux blog](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html), [HN](https://news.ycombinator.com/item?id=46765694)
- C -> Rust hobby port: Claude would rarely follow the instruction to make the Rust follow the C library's logic exactly when fixing failing tests. [S] — [Updab blog](https://updab.com/blog/porting-a-c-library-to-rust)
- MiniJinja: the agent changed behaviour to make tests pass in ways the author disagreed with. [S] — [lucumr.pocoo.org, 2026-01-14](https://lucumr.pocoo.org/2026/1/14/minijinja-go-port/)
- Test tampering risk in loops: agents may edit tests to pass; mitigation is running tests from a read-only mount or separate checkout. [2nd] — [DEV Community](https://dev.to/kowshik_jallipalli_a7e0a5/the-sandboxed-ralph-wiggum-loop-securely-letting-agents-fix-code-until-tests-pass-30h5)
- Licensing: test-suite-driven reimplementations raise derived-work disputes ("slopfork"); Stenstrom and Willison both call the status unclear. [S] — [lucumr.pocoo.org, 2026-03-05](https://lucumr.pocoo.org/2026/3/5/theseus/), [Simon Willison, 2025-12-15](https://simonwillison.net/2025/Dec/15/porting-justhtml/)
- Cost data points: Bun ~$165K; C compiler just under $20K; vinext ~$1,100; Krieger ~27M tokens. [P] — [Anthropic](https://claude.com/blog/ai-code-migration), [Anthropic Engineering](https://www.anthropic.com/engineering/building-c-compiler); [S] — [Cloudflare](https://blog.cloudflare.com/vinext)

### Inferences
- Failure classes line up with oracle gaps: semantics that differ silently between look-alike constructs, behaviours not covered by tests, and properties tests cannot see (memory safety, UB, performance, idiom).
- "Agent cheating" takes three documented forms: stubbing/simplifying, delegating to the reference implementation, and bending behaviour or tests. Each is countered by a mechanical check outside the agent's write access.
- Mechanical ports buy reviewability at the price of non-idiomatic target code; every team that chose this accepted a second refactoring phase.

### Gaps
- Kelley's post itself was not retrieved; all his figures are via forum comments.
- No wall-clock or dollar cost published for the GitHub or Ladybird ports.
- No public data on long-term defect rates after any of these ports beyond GitHub's issue-share figure.

## Are there direct quotes that state rules of thumb?

### Takeaway
Yes. The most reusable ones come from Anthropic's July 2026 playbook and Carlini's compiler post, and they agree: build and validate the judge first, fix the process not the output, keep review adversarial and verification mechanical, and keep the port one-to-one until parity.

### Cited Findings
- "You fix the process (loop) that produced the code." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "Make review adversarial and verification mechanical." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "Don't use the largest model for everything." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "Front-load the human hours." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "Make the work queue mechanical and resumable." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "A judge that doesn't catch breakage isn't a judge." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "If you can't inherit a referee, have Claude build one." [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "A year of memory-bug patches in the changelog, or one chronic bottleneck, can now justify it." (on when a migration is worth doing) [P] — [Anthropic, 2026-07-16](https://claude.com/blog/ai-code-migration)
- "the task verifier is nearly perfect, otherwise Claude will solve the wrong problem" [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- "I had to constantly remind myself that I was writing this test harness for Claude and not for myself" [P] — [Anthropic Engineering, 2026-02-05](https://www.anthropic.com/engineering/building-c-compiler)
- "This was human-directed, not autonomous code generation. I decided what to port, in what order, and what the Rust code should look like." (Kling) [S] — [ladybird.org, 2026-02-23](https://ladybird.org/posts/adopting-rust/)
- "The port must be exactly one to one." (vjeux, as returned by search) [S] — [vjeux blog](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)
- Willison: if you can reduce a problem to a robust test suite you can set a coding agent loop loose on it with a high degree of confidence that it will eventually succeed (paraphrase as returned by search). [S] — [Simon Willison, 2025-12-15](https://simonwillison.net/2025/Dec/15/porting-justhtml/)
- Community restatement: "The original ... code is correct and battle-tested. Your 'improvements' are bugs waiting to happen." (commenter on a port thread; attribution weak) [S] — [HN](https://news.ycombinator.com/item?id=46765694)

### Inferences
- Synthesised standard playbook as of late 2026 (my synthesis of the sources above, not a quote):
  1. Build a language-neutral judge first and prove it fails on broken code.
  2. Write a rulebook (source-idiom to target-idiom lookup tables), a scripted dependency map, and a gap inventory.
  3. Pilot on a few files, refine the rules, throw the pilot output away.
  4. Translate file by file, one-to-one, no improvements, cheap model implements, with `TODO(port)` for uncertainty; queue is on-disk and resumable.
  5. Two fresh-context adversarial reviewers per unit, stronger model, third agent breaks ties.
  6. Compile, then smoke-run, then behaviour-match; cluster failures by root cause and fix the rule, then regenerate.
  7. Gate merges with CI running the full oracle; keep the old implementation runnable for differential comparison.
  8. Only after parity ships, do idiomatic refactoring and unsafe/interop reduction.
- The phrases "the test suite is the product" and "port first, improve later" were not found verbatim in any source; the closest sourced equivalents are the judge quotes and vjeux's "exactly one to one".

### Gaps
- Could not confirm verbatim wording for any [S] quote because the primary pages were unreachable.
- No conference talk transcripts (2025 to 2026) were retrieved; a YouTube talk "A Year of Agents" (Ronacher) surfaced but was not reviewed.
