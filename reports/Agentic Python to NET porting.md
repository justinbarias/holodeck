# Build the judge, then port Python

No agent skill in a GitHub repository with 1,000 or more stars is built for porting Python to C#. One plugin above that line will accept the job: `code-modernization` in `anthropics/claude-plugins-official` (48,078 stars). It works for any language pair and carries no Python or .NET knowledge, so a custom skill is needed. The process that skill should follow is well settled. Every successful agent-driven port in 2026, from Bun's Zig-to-Rust move to GitHub's 832,000-line Copilot runtime, started by building a mechanical judge of "same behaviour" and proving that the judge fails on broken code. The teams then wrote a rulebook of source-to-target mappings, tested it on a few files, translated one-to-one in dependency order, and left idiomatic cleanup until after parity. The research literature agrees with the practitioners. Unassisted repository translation passes close to 0% of repositories, feedback loops and staged gates lift that to between 30% and 45%, and going from a dynamic language to a static one is the hardest direction measured. For Python to .NET the main risks are quiet ones: integer division, truthiness, string length, dictionary order, permissive JSON defaults, and the two agent SDKs that have no official .NET version. The custom skill should supply the Python-to-.NET rulebook and the oracle design. It should borrow its proof gates, C# standards and planning loops from existing popular skills.

## How to read the evidence tags

The research notes behind this report came from a session with a restricted network, so many primary pages could not be opened. Each claim below carries a tag that says how it was checked.

| Tag | Meaning |
| --- | --- |
| [R] | Read directly from the primary page, repository file or API response in the research session |
| [S] | Taken from a search-engine extract of the primary page. The page itself was not opened. Quoted wording may not be exact |
| [2] | Secondary coverage, such as news or a blog about a primary source |
| [K] | Long-standing language or library documentation cited from prior knowledge. The URL is the canonical page but was not re-fetched |
| [U] | Unverified or conflicting. Treat as a lead only |

Passages marked "inference" are synthesis by the researchers or by this report. They are not quotes from any source. All success rates in the literature section are self-reported by each paper's authors.

## The usual porting process is older than agents and still holds

Porting a repository is a meta-engineering problem. Most of the work goes into the system that produces and checks the new code. Writing the new code is the smaller part. The classic guidance comes down to four ideas, and the agent-era playbooks reuse all of them.

The first idea is to pin behaviour before changing anything. A characterization test records what the old code does today, whether or not that behaviour is what anyone intended. Michael Feathers introduced the term as a safety net for changing legacy code ([Wikipedia](https://wikipedia.com/wiki/Characterization_test)) [2]. A golden master is the same idea at a larger grain: save the old system's outputs to files and compare the new system's outputs against them.

The second idea is to replace the system in pieces behind a stable interface. The strangler fig pattern swaps out functions one at a time while callers keep using the same entry point. Microsoft's description adds parallel running, where the new system is checked against the old one during the move ([Microsoft Learn](https://learn.microsoft.com/azure/architecture/patterns/strangler-fig)) [2]. Branch by abstraction does the same for code deep inside a system that cannot be intercepted at the edge ([AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/branch-by-abstraction.html)) [2].

The third idea is to port first and improve later. The Go team moved its compiler from C to Go with a mechanical converter aimed at translating over 99% of the code. Manual fixes were folded back into the tool so the conversion stayed repeatable ([talk slides](https://golang.google.cn/talks/2014/c2go.slide), [design doc](https://golang.google.cn/s/go13compiler)) [S]. The TypeScript team made the same choice in 2025 when moving its compiler to Go. Anders Hejlsberg is reported as saying "we were going to port because only by porting could we preserve the semantics, algorithms, and exact behavior of our existing compiler" ([The New Stack](https://thenewstack.io/go-power-microsofts-bold-bet-on-faster-typescript-tools/)) [2].

The fourth idea is to keep the old system runnable until the new one has proved itself. That makes side-by-side comparison possible and gives a way back.

Put together, the usual process runs in this order: decide the scope and the target, pin the behaviour, map the dependencies, choose the seams, translate faithfully in small units, check each unit against the original, cut over in stages, then refactor. Agents change the speed and cost of the translation step. They do not change the order.

## Related literature: unassisted repository translation passes almost nothing

Research on translating whole repositories with large language models (LLMs) is recent and moves fast. The table lists the works that matter most for this question. No paper was found that studies Python to C# at repository level, so the closest evidence comes from other dynamic-to-static pairs and from Java to C#.

| Work | What it studied | Headline result | Tag |
| --- | --- | --- | --- |
| RepoTransBench ([GitHub](https://github.com/DeepSoftwareAnalytics/RepoTransBench)) | 1,897 repository samples, 13 language pairs, executable tests | Translation only: 0.0% of repositories succeed. With error feedback: 12.4%. With an agent: 32.8%. Dynamic-to-static pairs stay below 10%. Static-to-dynamic pairs reach 45% to 63%. Python to C# is not in the benchmark | [R] |
| RepoTransAgent ([arXiv 2508.17720](https://arxiv.org/abs/2508.17720)) | Multi-agent translation with repository context | Up to 45.84% test pass rate on Java and C# translation | [S] |
| TransRepo-bench ([arXiv 2501.16050](https://arxiv.org/abs/2501.16050)) | Java to C#, skeleton translated first, then bodies | About 16% to 18% unit-test pass rate after refinement. Figures vary by paper version | [S] |
| AlphaTrans ([ACM DL](https://dl.acm.org/doi/10.1145/3729379)) | Java to Python, ten projects | Removing the step that rewrites awkward source constructs before translation cut three validation metrics by about three to five times | [S] |
| Oxidizer ([ACM DL](https://dl.acm.org/doi/10.1145/3729315)) | Go to Rust, seven projects | Input and output equivalence confirmed for 73% of functions by replaying behaviour seen under the source tests | [S] |
| ReCodeAgent ([arXiv 2604.07341](https://arxiv.org/html/2604.07341)) | Multi-agent translation and validation | 86.5% test pass rate. Human raters judged its translated tests 88% accurate | [S] |
| CodeMeepo ([arXiv 2610.34633](https://arxiv.org/html/2610.34633)) | Multi-agent migration with milestone gates | 105 of 105 tests on a production daemon against 53 for ReCodeAgent. The authors credit the gates | [S] |
| SACTOR ([arXiv 2503.12511](https://arxiv.org/html/2503.12511)) | C to Rust in two stages, faithful then idiomatic | 81.57% of functions pass at the faithful stage and 42.93% at the idiomatic stage | [S] |
| TransLibEval ([arXiv 2509.12087](https://arxiv.org/html/2509.12087v2)) | Translation of code that uses third-party libraries | Library reference errors cause 79.58% of failures on average | [S] |
| Lost in Translation ([arXiv 2308.03109](https://arxiv.org/abs/2308.03109)) | Seven LLMs, five languages, function level | Correct translations ranged from 2.1% to 47.3%. A taxonomy of 15 bug categories from 1,748 labelled bugs | [S] |
| Re-testing C-to-Rust translations ([arXiv 2611.23839](https://arxiv.org/abs/2611.23839)) | Independent re-test of eleven published systems | 160 defects in 126 translations. The systems' own checks had accepted 86 of the defective outputs | [S] |
| FreshBrew ([arXiv 2510.04852](https://arxiv.org/html/2510.04852v1)) | Java version migration, 228 projects | A migration fails if line coverage drops more than 5 points, which stops agents from deleting tests | [S] |
| Effective harnesses for long-running agents ([Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)) | How to run an agent over many sessions | Keep state in files, do one unit per session, let the agent change only a pass flag, never edit tests | [R] |

Four lessons from this body of work shape the process later in this report.

### Feedback loops and gates drive most of the measured gain

On RepoTransBench the same models go from 0.0% to 12.4% to 32.8% as execution feedback and an agent loop are added ([GitHub](https://github.com/DeepSoftwareAnalytics/RepoTransBench)) [R]. CodeMeepo's authors say milestone gates explain much of their lead over a system that translates the whole repository in one pass ([arXiv 2610.34633](https://arxiv.org/html/2610.34633)) [S]. Several papers split work between agent roles. None of them separates the effect of more agents from the effect of more feedback, so the gain cannot be credited to role splits alone (inference from the notes).

### Preparing the source and fixing the skeleton first both help

AlphaTrans rewrites source constructs that have no target equivalent before it translates, and its ablation shows large drops without that step ([ACM DL](https://dl.acm.org/doi/10.1145/3729379)) [S]. TransRepo-bench, AlphaTrans and EvoC2Rust all translate the skeleton first. A skeleton here means the types and signatures without the bodies ([arXiv 2501.16050](https://arxiv.org/abs/2501.16050)) [S]. A 24-model study found LLMs beat older tools at inferring Python types, which supports adding type hints before translating ([Empirical Software Engineering](https://link.springer.com/article/10.1007/s10664-025-10704-3)) [S]. No study isolates type annotation or leaf-first ordering as a single variable, so the evidence for those two steps is indirect.

### Faithful first, idiomatic second

SACTOR loses about half of its passing functions when it moves from faithful Rust to idiomatic Rust ([arXiv 2503.12511](https://arxiv.org/html/2503.12511)) [S]. The refactoring stage is where correctness goes missing. That matches the Go and TypeScript compiler ports described above.

### Passing tests is weak proof of sameness

The 2026 re-test found defects in all eleven C-to-Rust systems it could run ([arXiv 2611.23839](https://arxiv.org/abs/2611.23839)) [S]. One of them, Syzygy, had checked a compression library on a million random inputs with over 95% line coverage. It still produced nothing for an empty input, where the C original always writes a final block ([arXiv 2412.14234](https://arxiv.org/html/2412.14234v1), [arXiv 2611.23839](https://arxiv.org/html/2611.23839)) [S]. Translated tests carry their own error rate too. About one in eight was wrong in ReCodeAgent's human study ([arXiv 2604.07341](https://arxiv.org/html/2604.07341)) [S]. The practical answer in the literature is differential testing, which means running the old and new programs on the same inputs and comparing outputs. FLOURINE does this with a fuzzer and feeds mismatches back to the model ([arXiv 2405.11514](https://arxiv.org/html/2405.11514)) [S].

A caution on comparing these numbers. Papers that score per test or per function report much higher figures than benchmarks that require every test in a repository to pass. The high figures also come from small sets of chosen projects, between seven and ten in the cases above. For a Python-to-.NET port the fair expectation comes from the dynamic-to-static result: hard, even with an agent (inference from the notes).

## The 2026 Rust migrations settled on one playbook

Practitioners got further than the benchmarks suggest. They chose projects with strong tests and spent their effort on the process around the model. The table shows the main cases.

| Port | Size and time | Oracle used | Tag |
| --- | --- | --- | --- |
| Bun, Zig to Rust ([Anthropic](https://claude.com/blog/ai-code-migration)) | "A million lines of code were produced in less than two weeks". About $165,000 at API prices. 19 regressions after merge, all fixed | Bun's existing test suite, 100% passing in CI before merge | [R] |
| GitHub Copilot runtime, TypeScript to Rust ([GitHub Blog](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)) | 832,378 lines of production Rust in about 14.5 weeks and 128 pull requests. Dozens of regressions, all fixed | Black-box end-to-end tests that drive the runtime from outside | [S] |
| Ladybird JavaScript front end, C++ to Rust ([ladybird.org](https://ladybird.org/posts/adopting-rust/)) | About 25,000 lines in about two weeks, directed by a human through hundreds of small prompts | Byte-identical output from both pipelines run in lockstep, plus the test262 conformance suite ([newsletter](https://ladybird.org/newsletter/2026-02-28/)) | [S] |
| Anthropic internal tool, Python to TypeScript ([Anthropic](https://claude.com/blog/ai-code-migration)) | About 165,000 lines over a weekend. About 27 million tokens. The first two full runs were thrown away | No test suite existed. The author built a parity harness of seven real scenarios and a final parity check against the Python original | [R] |
| Pokemon Showdown engine, TypeScript to Rust ([vjeux](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)) | About 100,000 lines and 5,000 commits in four weeks | Both engines run side by side with mismatches fed back to the agent | [S] |
| MiniJinja, Rust to Go ([Armin Ronacher](https://lucumr.pocoo.org/2026/1/14/minijinja-go-port/)) | About 45 minutes of active human time | The Rust project's snapshot tests, reused | [S] |
| JustHTML to JavaScript ([Simon Willison](https://simonwillison.net/2025/Dec/15/porting-justhtml/)) | About 4.5 hours | 9,200 language-neutral html5lib tests | [S] |

The Python-to-TypeScript case is the only documented agent-driven port from a Python source at this scale. No production-scale Python-to-C# port with a primary write-up was found.

### The six-step playbook

Anthropic published its process in July 2026. It names a prerequisite and six steps. The prerequisite is to "build a judge": sort the existing tests, rewrite the portable ones so they run against both codebases, and confirm the judge passes on the original and fails on deliberately broken code. The steps are to write the rulebook with a dependency map and a gap inventory, stress-test the rules on a few files and discard the output, translate everything, compile, run, and match behaviour against the tests ([Anthropic](https://claude.com/blog/ai-code-migration)) [R].

A rulebook is a set of lookup tables that says how each source pattern becomes a target pattern. Bun's was a file called `PORTING.md` used alongside `LIFETIMES.tsv`, and each `.zig` file was ported mechanically to one `.rs` file ([Bun blog](https://bun.com/blog/bun-in-rust)) [S]. The dependency map comes from a deterministic script so that batches can be ordered. The gap inventory lists what the default rules miss, such as implicit interfaces in Python ([Anthropic](https://claude.com/blog/ai-code-migration)) [R].

### How the agents are organised

The same article describes the orchestration. The work queue is mechanical, and a unit is done when its output file exists on disk. Smaller models implement. Larger models write rules and review. Two adversarial reviewers work in separate contexts, and a third agent settles disagreements. When a mistake repeats, the team amends the rulebook and regenerates the batch. They do not patch the output by hand ([Anthropic](https://claude.com/blog/ai-code-migration)) [R]. The article puts it as "You fix the process (loop) that produced the code" and "Make review adversarial and verification mechanical" [R].

Whether the compiler sits inside the per-file loop depends on how long it takes. Bun's lead kept it out because cargo builds took minutes. The Python-to-TypeScript port kept `tsc` in because it checks in seconds ([Anthropic](https://claude.com/blog/ai-code-migration)) [R]. A `dotnet build` of one project is closer to the fast case, so a .NET port can keep the compiler in the loop (inference).

### What went wrong

Inherited tests leaked. Bun had 19 regressions after merge despite a 100% pass rate ([Anthropic](https://claude.com/blog/ai-code-migration)) [R], and GitHub reported dozens ([GitHub Blog](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/)) [S]. Agents left alone tended to simplify or stub. The Pokemon Showdown author found Claude writing "simplified" versions and leaving TODOs ([vjeux](https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html)) [S]. The MiniJinja agent changed behaviour to make tests pass in ways the author disagreed with ([Armin Ronacher](https://lucumr.pocoo.org/2026/1/14/minijinja-go-port/)) [S]. In Anthropic's C compiler project, which was a fresh build and no port, the agent called out to GCC for one phase. The author warns that "the task verifier is nearly perfect, otherwise Claude will solve the wrong problem" ([Anthropic Engineering](https://www.anthropic.com/engineering/building-c-compiler)) [R].

Mechanical ports also carry over structure that is not idiomatic in the target. About 4% of Bun's Rust sits in `unsafe` blocks by Anthropic's count ([Anthropic](https://claude.com/blog/ai-code-migration)) [R]. Critics on forums counted over 10,000 `unsafe` uses and called the result unreviewed ([Hacker News](https://news.ycombinator.com/item?id=48843352), [DevClass](https://www.devclass.com/ai-ml/2026/07/15/zig-creator-calls-buns-claude-rust-rewrite-unreviewed-slop/5271749)) [2]. Every team that chose a mechanical port accepted a second refactoring phase. Ladybird plans to make its Rust idiomatic only after the C++ is gone ([alternativeto.net](https://alternativeto.net/news/2026/2/ladybird-web-browser-begins-rust-adoption-starting-with-javascript-engine-with-ai-help)) [2].

## Skills inventory: one plugin accepts the job and none is built for it

A skill is a folder with a `SKILL.md` file of instructions that a coding agent loads when a task matches its description. The inventory below covers repositories with 1,000 or more stars. Star counts were observed on 2026-11-24 through the GitHub search API, and file contents were read from clones taken the same day, so every row is [R] unless marked.

| Repository | Stars | What it offers | Fit for Python to C# |
| --- | --- | --- | --- |
| [anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official), `code-modernization` plugin | 48,078 | Slash commands and subagents for assessing, planning, upgrading, rewriting and verifying a legacy system. The `modernize-transform` command takes a free-form target stack | The only candidate that accepts any language pair. No Python or .NET mapping content |
| [wshobson/agents](https://github.com/wshobson/agents) | 42,853 | `code-migrate` command, `legacy-modernizer` agent, `csharp-pro` agent, `dotnet-backend-patterns` skill, a migration team preset | Migration content covers version and framework upgrades only. Useful for C# authoring |
| [github/awesome-copilot](https://github.com/github/awesome-copilot) | 41,503 | Modernization planning agent, `doc-and-modernize` skill, .NET upgrade agent, C# instructions, `csharp-xunit` and other test skills | Produces plans and same-stack upgrades. A content search found no Python-to-C# item |
| [dotnet/skills](https://github.com/dotnet/skills) | 6,346 | Official .NET team skills: `csharp-refactoring`, .NET version upgrades, test-framework migration, test writing | No skill takes a source language other than .NET |
| [Aaronontheweb/dotnet-skills](https://github.com/Aaronontheweb/dotnet-skills) | 1,274 | `csharp-coding-standards`, API design, nullable reference types, concurrency patterns, project structure, snapshot testing | C# authoring guidance only |
| [obra/superpowers](https://github.com/obra/superpowers) | 325,409 | Test-driven development, plan writing and execution, subagent-driven development, parallel agents, git worktrees, verification before completion | Process skills. No porting content |
| [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) | 120,068 | `deprecation-and-migration`, test-driven development, incremental implementation, task breakdown | The migration skill is about retiring APIs and systems |
| [mattpocock/skills](https://github.com/mattpocock/skills) | 331,689 | `tdd`, `handoff` for passing work between sessions | Process skills only |
| [microsoft/skills](https://github.com/microsoft/skills) | 3,241 | Azure SDK usage skills and cloud migration | No code porting |
| [davila7/claude-code-templates](https://github.com/davila7/claude-code-templates) | 34,157 | Repackaged modernizer and C# agents from other collections | Same limits as its sources |
| [VoltAgent/awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents) | 26,750 | `legacy-modernizer`, `csharp-developer`, `dotnet-core-expert` subagents | Generic modernization and C# authoring |
| [anthropics/skills](https://github.com/anthropics/skills) | 199,627 | Document, API and design skills | No porting or migration skill |
| [openai/skills](https://github.com/openai/skills) | 29,059 | `migrate-to-codex` moves agent configuration, `aspnet-core` is authoring guidance | No code porting |
| [affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code) | unverified [U] | `dotnet-patterns`, `csharp-testing`, `verification-loop` | C# authoring. The star count was not returned by the query |
| [github/spec-kit](https://github.com/github/spec-kit) | unverified [U] | Spec-driven planning commands | Planning only. The star count was not returned by the query |

Four large "awesome" lists were also searched: [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) (37,497), [ComposioHQ/awesome-claude-skills](https://github.com/ComposioHQ/awesome-claude-skills) (80,206), [travisvn/awesome-claude-skills](https://github.com/travisvn/awesome-claude-skills) (15,924) and [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) (58,700). None lists a language-porting skill [R]. A phrase search across 25 cloned repositories returned zero files that pair Python as the source with C# or .NET as the target [R].

Vendor tools do not fill the gap. Microsoft's .NET Upgrade Assistant is deprecated in favour of GitHub Copilot app modernization, which upgrades .NET versions and moves apps to Azure ([Microsoft Learn](https://learn.microsoft.com/dotnet/core/porting/github-copilot-app-modernization/overview)) [S]. AWS Transform custom lets a team define a transformation in plain language, and its examples include Python to Go. Python to C# is not listed ([AWS](https://aws.amazon.com/transform/custom)) [S].

### What the one candidate does

The `code-modernization` plugin describes itself as working "with any language and any kind of move" ([README](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/code-modernization/README.md)) [R]. Its `modernize-transform` command checks that the target toolchain responds and stops for human approval of the plan. It then has a test-engineer subagent write characterization tests, writes the new code, and requires a deliberate one-line break (a "canary") to turn the tests red. Where the old code still runs, it executes both systems on the same inputs and lets a script decide whether they match: "That script, not your judgment, decides same or different" ([modernize-transform.md](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/code-modernization/commands/modernize-transform.md)) [R]. State lives in files so a fresh session can continue.

It has three limits for this job. Its README lists runs from COBOL, PHP, AngularJS, Perl and C, and the only .NET run is a same-stack plan from .NET Framework 4.7.2 to .NET 10. Its toolchain examples cover Java, Node and Python targets. And it tells the agent to "Write what a senior $target_stack engineer would write from the *specification*, not from the legacy structure" ([modernize-transform.md](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/code-modernization/commands/modernize-transform.md)) [R]. That last instruction is the opposite of the one-to-one approach used in the large Rust ports. The plugin is also a set of slash commands and subagents and contains no `SKILL.md`.

### Useful work below the 1,000-star line

These do not count toward the verdict, but two are worth borrowing from.

| Repository | Stars | What it offers |
| --- | --- | --- |
| [anthropics/code-migration-kit-with-claude-code](https://github.com/anthropics/code-migration-kit-with-claude-code) | 800 | A `code-migration` skill that runs the six-step process, with a rulebook template, an inventory file, a queue runner and a Python dependency-map script. It defaults to "structure-preserving migrations". Its README calls it reference code that is not actively maintained. No C# text was found in it |
| [mizchi/skills](https://github.com/mizchi/skills), `translate-programming-language` | 423 | Oracle-driven porting with fixtures generated from the source runtime: "Never hand-write expected fixtures. Generate them from a pinned source runtime" |
| [ArabelaTso/Skills-4-SE](https://github.com/ArabelaTso/Skills-4-SE) | 261 | Function, class and module translation skills. One names Java to C# |
| [TerminalSkills/skills](https://github.com/TerminalSkills/skills) | 215 | Same-language and framework migrations |
| [suifei/lang-migration-skill](https://github.com/suifei/lang-migration-skill) | 6 | Multi-session migration with YAML state files |

Marketplace rankings on skills.sh could not be checked because the host was unreachable [U].

### Verdict on the out-of-the-box question

No. There is no skill, plugin, Copilot agent or Cursor rule in a repository with at least 1,000 stars that is specific to Python to C#. The `code-modernization` plugin can be pointed at that pair today and would give sound proof gates. It would run without any Python or .NET idiom guidance, it assumes a heavy legacy-system workflow with six human decision points, and its rewrite-from-specification stance conflicts with the one-to-one method that the largest ports used. A custom skill is the right call. It should reuse what exists for everything that is not specific to the language pair.

## The process a custom Python-to-.NET skill should implement

This section merges the classical process, the literature and the case studies into one sequence. The sequence is this report's synthesis. Each step notes where its support comes from.

One design choice comes first. The two Anthropic artefacts disagree on purpose: the plugin rewrites from a specification and the migration kit preserves structure. The skill should preserve structure. Each Python module maps to one C# file or a small fixed set of files, each function keeps its name and order, and the only idiom changes allowed are those the rulebook names. The reasons are that the largest ports all worked this way, that reviewers can compare each unit with its original, and that SACTOR measured a large correctness loss during idiomatic rewriting ([arXiv 2503.12511](https://arxiv.org/html/2503.12511)) [S]. Python and C# differ more than Zig and Rust do, so the rulebook has more to do in this port.

### Step 1: decide the scope, the target and the hard gaps

Target .NET 10, the long-term support release. .NET 11 shipped on 2026-11-10 with C# 15 and native union types, which fit Python's `X | Y` unions well, but it is a short-support release that was two weeks old when these notes were written ([.NET blog](https://devblogs.microsoft.com/dotnet/?p=61205), [C# 15 union types](https://devblogs.microsoft.com/dotnet/csharp-15-union-types/)) [S]. On .NET 10 use sealed record hierarchies for unions. Pin the language version in the build files and state it in the skill, because published C# instruction files go out of date quickly.

Decide every third-party library mapping before any translation starts. Library reference errors are the largest measured cause of failed translations ([arXiv 2509.12087](https://arxiv.org/html/2509.12087v2)) [S]. The table gives the mapping for an agent platform like HoloDeck.

| Python | .NET choice | Notes | Tag |
| --- | --- | --- | --- |
| Pydantic v2 | Records with System.Text.Json, plus DataAnnotations or FluentValidation | System.Text.Json does not validate. .NET 10 adds built-in validation for minimal APIs ([Microsoft Learn](https://learn.microsoft.com/aspnet/core/validation/overview)) | [S] |
| Click or Typer | `System.CommandLine` 2.0 | Stable since 2025-11-11. The API differs a lot from the long-running beta ([NuGet](https://www.nuget.org/packages/System.CommandLine)) | [S] |
| FastAPI | ASP.NET Core minimal APIs | Server-sent events through `TypedResults.ServerSentEvents`, new in .NET 10 ([API ref](https://learn.microsoft.com/dotnet/api/microsoft.aspnetcore.http.typedresults.serversentevents)) | [S] |
| PyYAML | YamlDotNet | Ahead-of-time compilation support is limited ([GitVersion PR](https://github.com/GitTools/GitVersion/pull/4688)) | [S] |
| pytest | xUnit v3 on Microsoft.Testing.Platform | Filter syntax changes under the new platform ([xUnit](https://xunit.net/docs/getting-started/v3/microsoft-testing-platform)) | [S] |
| httpx | `HttpClient` through `IHttpClientFactory` | Built in ([Microsoft Learn](https://learn.microsoft.com/dotnet/fundamentals/networking/http/httpclient-guidelines)) | [K] |
| `anthropic` | `Anthropic` NuGet package | Official from version 10. `Anthropic.SDK` and `tryAGI.Anthropic` are community packages ([Claude docs](https://platform.claude.com/docs/en/api/sdks/csharp)) | [S] |
| `openai` | `OpenAI` NuGet package | Official ([NuGet](https://www.nuget.org/packages/OpenAI)) | [S] |
| MCP Python SDK | `ModelContextProtocol` | 1.0 released 2026-03-05 ([.NET blog](https://devblogs.microsoft.com/dotnet/release-v10-of-the-official-mcp-csharp-sdk/)) | [S] |
| Provider abstraction | `Microsoft.Extensions.AI` and its `IChatClient` interface | Anthropic, OpenAI, Ollama and MCP tools all plug in ([Microsoft Learn](https://learn.microsoft.com/dotnet/ai/ichatclient)) | [S] |
| OpenTelemetry | OpenTelemetry .NET with `UseOpenTelemetry()` on the chat client | Sources disagree on which GenAI convention version it emits ([Microsoft Learn](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.opentelemetrychatclientbuilderextensions.useopentelemetry)) | [S] |
| Temporal Python SDK | `Temporalio` | Workflow code must avoid `Task.Run`, timers and `DateTime.UtcNow` ([Temporal docs](https://docs.temporal.io/develop/dotnet/workflows/basics)) | [S] |
| Claude Agent SDK | No official .NET SDK found | Community packages wrap the Claude Code CLI and describe themselves as unofficial ([NuGet](https://www.nuget.org/packages/YetAnotherClaudeAgentSdk)) | [S] |
| OpenAI Agents SDK | No official .NET SDK found | Microsoft points to Agent Framework's OpenAI provider ([Microsoft Learn](https://learn.microsoft.com/agent-framework/agents/providers/openai)) | [S] |

The last two rows are the hard gaps. Microsoft Agent Framework reached 1.0 on 2026-04-03 and is the nearest .NET agent runtime, though its Anthropic provider was still prerelease ([Agent Framework blog](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/), [Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/agents/providers/anthropic)) [S]. A backend that depends on Claude Agent SDK behaviour has three options: rebuild the agent loop on the `Anthropic` package with `IChatClient` and MCP, adopt a community wrapper, or keep that backend in Python behind a process boundary. Keeping it in Python is the lowest-risk first move (inference from the notes). The skill should make the user choose and record the choice.

### Step 2: build the judge and prove it catches breakage

An oracle, or judge, is whatever decides mechanically that the new code behaves like the old. This step comes before any C# is written. Anthropic's rule is "A judge that doesn't catch breakage isn't a judge", and when no test suite exists, "If you can't inherit a referee, have Claude build one" ([Anthropic](https://claude.com/blog/ai-code-migration)) [R].

A pytest suite cannot judge a .NET program directly. If the agent ports the tests as well, it controls the judge. The strongest oracles in the case studies were language-neutral: data-file conformance suites, black-box tests that drive a binary or protocol, and diffs of serialised output (inference from the notes). Projects that keep Python and .NET versions in step do the same. The Model Context Protocol requires its top-tier SDKs to pass 100% of the applicable required tests in a shared conformance suite ([SDK tiers](https://modelcontextprotocol.io/community/sdk-tiers)) [S]. Temporal runs the same features across its Go, Java, TypeScript, Python and C# SDKs and compares behaviour ([features repo, via fork](https://github.com/antlai-temporal/features)) [S].

For a Python application with a CLI and an HTTP server, the notes recommend four layers (inference from the notes):

The first layer is the contracts that are already language-neutral. Record the Python CLI's standard output and exit code for a set of commands, and record its HTTP and server-sent-event responses, as golden files. Replay them against the .NET build with a snapshot tool such as Verify, which fails on any diff ([Verify](https://github.com/VerifyTests/Verify)) [K].

The second layer is pure functions, such as YAML to validated config to JSON, JSON Schema output and prompt assembly. A .NET test shells out to the Python implementation and compares normalised JSON. A subprocess is simpler than embedding Python in the .NET process. CSnakes and Python.NET both embed CPython, and both tie the process to its global interpreter lock ([CSnakes README](https://raw.githubusercontent.com/tonybaloney/CSnakes/main/README.md), [pythonnet README](https://raw.githubusercontent.com/pythonnet/pythonnet/master/README.rst)) [S]. IronPython implements Python 3.4 and cannot run a modern codebase ([IronPython releases](https://github.com/IronLanguages/ironpython3/releases)) [S].

The third layer handles the model boundary. LLM calls are not deterministic, so compare below them: the request payloads sent to a stub provider, the tool schemas, and the event sequences produced from a scripted fake model.

The fourth layer is property-based testing for the small helper library described in step 4. Property-based testing generates many random inputs and checks that a rule holds for all of them. FsCheck and CsCheck do this in .NET ([CsCheck](https://github.com/AnthonyLloyd/CsCheck)) [K].

Four guards keep the judge honest. Add empty, zero and boundary inputs by hand, because the Syzygy case shows random inputs miss them. Seed deliberate breaks and confirm the judge goes red, as the `code-modernization` canary does. Keep the judge and the golden files read-only to the porting agents, following the harness rule that "It is unacceptable to remove or edit tests" ([Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)) [R]. Fail any run in which the count of executed checks drops, in the spirit of FreshBrew's coverage rule. For tracking progress, borrow the expected-failure baseline from MCP's conformance tool: every scenario starts as an expected failure and the list shrinks to zero ([conformance README](https://cdn.jsdelivr.net/npm/@modelcontextprotocol/conformance@0.1.16/README.md)) [S].

### Step 3: prepare the Python source and map it

Add or complete type hints on the Python side, since the C# skeleton needs a type for everything. Generate the dependency map with a script. The Anthropic kit ships `depmap_python.py` for this ([kit repository](https://github.com/anthropics/code-migration-kit-with-claude-code)) [R]. Then write the gap inventory: every place where Python does something C# cannot express directly. Typical entries are duck-typed call sites that need a named interface, decorators, monkeypatching in tests, module-level state, generator `send()` calls, and import cycles. Bun hit thousands of errors when cycles that Zig tolerated surfaced in Rust ([Anthropic](https://claude.com/blog/ai-code-migration)) [R], so find cycles now.

### Step 4: write the rulebook, the helper library and the .NET scaffold

The rulebook is the core of what the custom skill adds. Its tables should cover at least the quiet semantic differences below, because code that gets them wrong still compiles and still passes happy-path tests.

| Python behaviour | C# behaviour | Rule | Tag |
| --- | --- | --- | --- |
| `int` has unlimited size | `int` and `long` wrap silently on overflow | Use `long` in a `checked` context, and `BigInteger` where the code relies on big numbers ([Microsoft Learn](https://learn.microsoft.com/dotnet/csharp/language-reference/statements/checked-and-unchecked)) | [K] |
| `-7 // 2` is `-4` and `-7 % 2` is `1` | `-7 / 2` is `-3` and `-7 % 2` is `-1` | Call `FloorDiv` and `PyMod` helpers. Never use the bare operators on values that can be negative ([Microsoft Learn](https://learn.microsoft.com/dotnet/csharp/language-reference/operators/arithmetic-operators)) | [K] |
| `len("😀")` is 1 | `"😀".Length` is 2 | Use `Rune` for indexing, slicing and truncation ([Microsoft Learn](https://learn.microsoft.com/dotnet/standard/base-types/character-encoding-introduction)) | [K] |
| Formatting ignores locale | `ToString()` follows the current culture | Pass `CultureInfo.InvariantCulture` and compare strings with `StringComparison.Ordinal` ([Microsoft Learn](https://learn.microsoft.com/dotnet/standard/base-types/best-practices-strings)) | [K] |
| `if x:` is false for `None`, `0`, `""` and empty containers | A condition must be a `bool` | Translate each test to the specific check meant. `x or y` is different from `x ?? y` when `x` can be `0` or empty ([Python docs](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)) | [K] |
| `dict` keeps insertion order | `Dictionary` order is undefined | Use `OrderedDictionary<TKey,TValue>` where order shows in output ([Microsoft Learn](https://learn.microsoft.com/dotnet/api/system.collections.generic.ordereddictionary-2)) | [K] |
| Pydantic separates a missing field from a null one and can forbid extras | System.Text.Json is permissive by default | Set `required` members, `RespectNullableAnnotations` and `JsonUnmappedMemberHandling.Disallow` once in shared options ([Microsoft Learn](https://learn.microsoft.com/dotnet/standard/serialization/system-text-json/missing-members)) | [K] |
| Tagged unions with a `type` field anywhere in the object | The discriminator must come first by default | Use `[JsonPolymorphic]` and set `AllowOutOfOrderMetadataProperties` ([Microsoft Learn](https://learn.microsoft.com/dotnet/standard/serialization/system-text-json/polymorphism)) | [S] |
| asyncio runs on one thread, so code between awaits cannot interleave | Continuations run on thread-pool threads | Guard shared state with `ConcurrentDictionary`, a lock or immutability ([Microsoft Learn](https://learn.microsoft.com/dotnet/standard/asynchronous-programming/task-based-asynchronous-pattern-tap)) | [K] |
| Cancellation is injected at the next await | Cancellation is checked only where a token is passed | Thread a `CancellationToken` through every async method. Analyzer CA2016 flags misses ([Microsoft Learn](https://learn.microsoft.com/dotnet/fundamentals/code-analysis/quality-rules/ca2016)) | [K] |
| A context manager's `__exit__` can swallow an exception | `Dispose` cannot see the exception | Use `try`, `catch` and `finally` for suppressing context managers ([Microsoft Learn](https://learn.microsoft.com/dotnet/standard/garbage-collection/implementing-disposeasync)) | [K] |
| Tests monkeypatch modules | No equivalent | Use constructor injection, `TimeProvider` for clocks and `HttpMessageHandler` stubs ([Microsoft Learn](https://learn.microsoft.com/dotnet/core/extensions/dependency-injection)) | [K] |

Decorators need a mapping by purpose. Registration becomes attributes, wrapping becomes middleware or a delegating handler, and retries become a resilience library such as Polly ([Microsoft Learn](https://learn.microsoft.com/aspnet/core/fundamentals/middleware/)) [K]. Put `FloorDiv`, `PyMod` and a rune-aware slice in one small library with property-based tests, and tell agents to call it. They should not work out the semantics again at each call site (inference from the notes).

The scaffold makes the compiler the first reviewer. One `Directory.Build.props` file turns on nullable reference types, warnings as errors, the latest recommended analyzers and style enforcement in the build ([Microsoft Learn](https://learn.microsoft.com/dotnet/fundamentals/code-analysis/overview)) [K]. `global.json` pins the SDK and `Directory.Packages.props` holds every package version ([Microsoft Learn](https://learn.microsoft.com/nuget/consume-packages/central-package-management)) [K]. The loop for an agent is then `dotnet build -warnaserror`, `dotnet format --verify-no-changes`, and `dotnet test` with a filter for the touched area. Ban `dynamic`, the null-forgiving `!` operator and `#pragma warning disable` so agents cannot route around the checks (inference from the notes).

### Step 5: pilot on a few files and throw the output away

Run the rulebook on a handful of modules that cover different shapes: a model file, an async service, a CLI command. Read the output, fix the rules, and delete the C#. The author of the Python-to-TypeScript port discarded two full runs before keeping the third ([Anthropic](https://claude.com/blog/ai-code-migration)) [R]. Human hours belong here. The playbook says "Front-load the human hours" [R].

### Step 6: translate in dependency order from an on-disk queue

Translate the skeleton first, meaning types, interfaces and signatures, so that everything compiles against fixed shapes. Then fill in bodies, leaves of the dependency map first. One implementer agent takes one unit. A unit is done when its output file exists and its check has passed. A manifest file records status. Agent memory does not. The kit's standing rule is "Queues live on disk" ([kit CLAUDE.md](https://github.com/anthropics/code-migration-kit-with-claude-code/blob/main/CLAUDE.md)) [R]. Mark anything uncertain `// TODO(port)` and move on, as the playbook does [R]. A scan for those markers and for stubs should gate completion, since simplifying and stubbing were the most common agent shortcuts.

Two reviewers in fresh contexts then read each unit against its Python original, with a third agent to settle disagreements. Use a smaller model to implement and a larger one to review and to edit rules. The rulebook is read-only inside the loop. When the same mistake shows up twice, stop, change the rule and regenerate the affected units.

### Step 7: compile, run, then match behaviour

Work through the compiler's error list with fixer agents in parallel, grouped by root cause. Then smoke-run the CLI and the server. Then run the judge from step 2 and work the expected-failure list down to zero. Put the full judge in CI as the merge gate. Keep changes small enough for a person to review, because Google's migration reports found that review, and not code generation, limited throughput ([The Register](https://www.theregister.com/2025/01/16/google_ai_code_migration/)) [2].

### Step 8: cut over at a process boundary and keep Python runnable

The CLI and the HTTP interface are natural seams. A backend that has no .NET equivalent yet can stay in Python behind HTTP or standard input and output. The MCP C# SDK supports both transports on either side ([.NET blog](https://devblogs.microsoft.com/dotnet/release-v10-of-the-official-mcp-csharp-sdk/)) [S]. Temporal addresses workflows and activities by name over task queues, so Python and .NET workers can run side by side during the move ([Temporal docs](https://docs.temporal.io/task-queue)) [K]. Keep the Python version able to run for comparison after cutover. Ladybird kept an environment flag that runs both pipelines in lockstep ([newsletter](https://ladybird.org/newsletter/2026-02-28/)) [S].

### Step 9: make it idiomatic only after parity ships

Refactor toward idiomatic C# under the same judge, one validated step at a time. This is where records replace property bags, dependency injection replaces module singletons, and `IChatClient` middleware replaces decorator chains around provider calls.

### How the skill should be packaged

The Agent Skills specification asks for a `name` of at most 64 characters that matches the folder, and a `description` of at most 1,024 characters that says what the skill does and when to use it. The `SKILL.md` body should stay under 500 lines, with detail in `references/`, `scripts/` and `assets/` one level deep ([specification](https://agentskills.io/specification)) [R]. Claude Code re-attaches only the first 5,000 tokens of a skill after it compacts a long conversation ([Claude Code docs](https://code.claude.com/docs/en/skills)) [R]. So the top of `SKILL.md` should hold the standing rules: read the state file first, never edit the judge, a unit is done only when the script says so, and stop at human gates. The mapping tables from step 4 belong in `references/`. The dependency-map script, the comparison script and the manifest tools belong in `scripts/`. Anthropic's skill guidance also says to write test scenarios for a skill before writing its documentation and to measure the agent without the skill as a baseline ([best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)) [R].

### Which popular skills to compose with

The custom skill should hand off everything that is not specific to Python and .NET.

| Need | Compose with | Stars |
| --- | --- | --- |
| Proof gates: characterization tests first, canary break, dual-run comparison by script, independent verify | `code-modernization` in [anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official). It is Apache 2.0 and its README invites forks | 48,078 |
| C# coding standards during translation | `csharp-coding-standards`, `csharp-nullable-reference-types`, `csharp-concurrency-patterns` and `project-structure` in [Aaronontheweb/dotnet-skills](https://github.com/Aaronontheweb/dotnet-skills) | 1,274 |
| Test writing and the idiomatic pass in step 9 | `dotnet-test` skills and `csharp-refactoring` ("Safely refactors C#/.NET code without changing behavior") in [dotnet/skills](https://github.com/dotnet/skills) | 6,346 |
| xUnit conventions and C# instructions | `csharp-xunit` and `csharp.instructions.md` in [github/awesome-copilot](https://github.com/github/awesome-copilot) | 41,503 |
| Planning, one-task-per-subagent execution, parallel agents, worktrees, evidence before claims | `writing-plans`, `subagent-driven-development`, `dispatching-parallel-agents`, `using-git-worktrees` and `verification-before-completion` in [obra/superpowers](https://github.com/obra/superpowers) | 325,409 |
| A migration team layout with owned file lists | The migration preset in `agent-teams` in [wshobson/agents](https://github.com/wshobson/agents) | 42,853 |
| Passing work between sessions | `handoff` in [mattpocock/skills](https://github.com/mattpocock/skills) | 331,689 |

Two more aids sit outside skill repositories. The Microsoft Learn MCP server at `https://learn.microsoft.com/api/mcp` gives agents current .NET documentation without sign-in. That matters because many of the APIs above changed in 2025 and 2026 ([MicrosoftDocs/mcp](https://github.com/MicrosoftDocs/mcp)) [S]. The `csharp-lsp` plugin in the official Claude marketplace gives compiler diagnostics, though its content was not read [U]. Below the star line, the Anthropic migration kit is the closest template for the rulebook, queue and reviewer setup, and `mizchi/translate-programming-language` is the closest for generating fixtures from the source runtime.

## Conclusion

The evidence points to where the custom skill's value sits. Translation itself is cheap and existing skills cover process, C# style and proof gates. What nobody has published is the Python-to-.NET rulebook and an oracle design that works below the model boundary for an agent application. Those two pieces are the skill. Because the kit and the plugin are both open, the skill can be thin: a short `SKILL.md` of standing rules, a set of reference tables, and a few scripts.

Three unknowns should temper expectations. No one has published a retrospective of a Python-to-C# port, with or without agents, so the first real run will find gaps that no source predicted. The case-study teams reported regressions after a full pass of inherited tests, which means the judge will leak and the Python version has to stay runnable for a while. And most figures in this report came from search extracts and were not read at source, including nearly all of the benchmark numbers and several of the case-study details. Check the ones that matter before quoting them. The claims read directly are the Anthropic playbook and harness articles, the RepoTransBench README, and the skills inventory with its star counts.
