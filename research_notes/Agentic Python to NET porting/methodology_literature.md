# Methodology literature: LLM/agent repository-level code translation and port validation (2022 to Nov 2026)

Verification note for the report writer. `arxiv.org`, `martinfowler.com`, `go.googlesource.com` and `alirezai.cs.illinois.edu` were blocked by this session's network policy, so paper PDFs could not be opened. Two sources were read directly: the Anthropic harness article and the RepoTransBench GitHub README. Everything else comes from search-engine extracts of the cited pages (mostly arXiv abstract/HTML pages). Each finding is tagged:

- **[D]** read directly from the primary page.
- **[S]** search extract of the primary source (number seen in the extract, paper not opened).
- **[2]** secondary source or second-hand citation.
- **[M]** detail supplied from my own recall (authors, venue), not confirmed this session.

Numbers frequently differ between arXiv versions of the same paper; where I saw a conflict it is stated. All success rates are self-reported by the paper's authors unless stated.

## 1. Key papers and benchmarks for repository-level translation, and reported success rates

### Takeaway
Unassisted LLM translation of whole repositories fails almost completely (0 to 8% of repositories pass their tests), and execution-feedback agents raise this to roughly 20 to 45% on benchmarks with mixed repositories; only heavily staged, validated pipelines on a handful of hand-picked subjects report 85%+ test pass rates. Dynamic-to-static translation is the worst direction measured (below 10% repository success on RepoTransBench versus 45 to 63% for static-to-dynamic), and no paper was found that studies Python to C# at repository level.

### Cited Findings

#### Repository-level benchmarks
- **RepoTransBench** (Wang et al.; arXiv 2412.17744; a later revision is listed as an IEEE TSE 2026 paper on a co-author page [2]). Current version: 1,897 repository samples, 13 translation pairs, executable test suites. Overall: translation only 0.0% success / 26.2% compile; error feedback 12.4% / 30.5%; RepoTransAgent 32.8% / 54.8%. Claude-4 and GPT-4.1 both 32.8%, DeepSeek 22.5%. Static-to-dynamic 45 to 63% success, dynamic-to-static below 10%. Python sources are paired with C++, Go, Java and Rust targets; Python to C# is not in the benchmark. [D] — [RepoTransBench GitHub](https://github.com/DeepSoftwareAnalytics/RepoTransBench)
- RepoTransBench v1 (Dec 2024) reported different numbers: best model 7.33% Success@1 (Claude-3.5-Sonnet best), iterative debugging with error feedback added 7.09 points on average, best still 21%. The two versions use different datasets and methods and are not comparable. [S] — [arXiv 2412.17744v1](https://arxiv.org/html/2412.17744v1)
- **RepoTransAgent** (arXiv 2508.17720): multi-agent, repository-aware; reports up to 55.34% compile rate and 45.84% pass rate on Java/C# translation. An aggregator summary attributes most of the gain to the Context and Refine agents [2]. [S] — [arXiv 2508.17720](https://arxiv.org/abs/2508.17720)
- **TransRepo-bench / Skeleton-Guided-Translation** (Microsoft Research; Findings of EMNLP 2025; arXiv 2501.16050). Java to C#, 13 tasks. Two steps: translate the repository's structural skeletons, then translate full bodies guided by the skeleton. Scores at test-case level instead of all-or-nothing. Reported unit-test pass rates are low and vary by version: DeepSeek-v3 71.14% build rate and 17.56% unit-test pass rate by the third refinement iteration; another version reports 15.7%; GPT-4o ends at 16.03%. [S] — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.986/), [arXiv 2501.16050](https://arxiv.org/abs/2501.16050)
- **RepoMod-Bench** (arXiv 2602.22518): repository modernization scored with implementation-agnostic tests that are hidden from the agent. Pass rates fall with repository size: 91.3% average on small benchmarks, 66.9% on medium. [S] — [arXiv 2602.22518](https://arxiv.org/html/2602.22518v1)
- **RustRepoTrans** (arXiv 2411.13990): repository-level benchmark targeting Rust. Only the title was retrieved. — [arXiv 2411.13990](https://arxiv.org/pdf/2411.13990)
- **f2x50** (arXiv 2611.23268): repository-scale Fortran-to-X migration benchmark, November 2026. Only the title was retrieved. — [arXiv 2611.23268](https://arxiv.org/html/2611.23268)

#### Repository-level translation systems
- **AlphaTrans** (Ibrahimzada, Ke, Pawagi, Abid, Pan, Sinha, Jabbarvand [M]; FSE 2025; arXiv 2410.24117). Java to Python, ten subjects. IBM's summary: 99.1% of translated fragments syntactically correct; runtime behaviour and functional correctness validated for 25.8%. The ACM abstract gives 98.80% of application method fragments and 96.40% of all fragments syntactically correct; an arXiv version gives 27.95% functional equivalence for method fragments. Ablation: without the pre-translation program transformation (for example resolving method overloading), three validation metrics drop from 24.50% to 4.58%, 2.88% to 0.81% and 9.76% to 3.05%. [S] — [IBM Research](https://research.ibm.com/publications/alphatrans-a-neuro-symbolic-compositional-approach-for-repository-level-code-translation-and-validation), [ACM DL](https://dl.acm.org/doi/10.1145/3729379), [arXiv 2410.24117](https://arxiv.org/html/2410.24117v4)
- AlphaTrans uses an LLM for type translation and skeleton construction, and static plus dynamic type information to resolve target types. [S] — [arXiv 2410.24117](https://arxiv.org/html/2410.24117v4)
- **Oxidizer** ("Scalable, Validated Code Translation of Entire Projects using Large Language Models"; arXiv 2412.08035; ACM DOI 10.1145/3729315, which is a PACMPL/PLDI 2025 DOI range [M]). Go to Rust on seven open-source Go projects. Validation is type-compatibility plus I/O equivalence using the source project's tests. I/O equivalence validated for 73% of translated functions on average (63% to 86%). Callees are mocked to isolate each function; nondeterministic functions made validation hard. [S] — [ACM DL](https://dl.acm.org/doi/10.1145/3729315), [arXiv 2412.08035](https://arxiv.org/html/2412.08035)
- **ReCodeAgent** (arXiv 2604.07341, April 2026): multi-agent, language-agnostic translation and validation. Reports 99.4% compilation and 86.5% test pass rate, 2.5 and 60.8 points above alternatives; 88% accuracy in a human study of translated tests. [S] — [arXiv 2604.07341](https://arxiv.org/html/2604.07341)
- **MatchFixAgent** (arXiv 2509.16187): validation and repair. Gives equivalence verdicts for 99.2% of translation pairs versus 71.6% for alternatives; agrees with them on 72.8% of 1,571 shared pairs; repairs 50.6% of translation bugs (32.1 points more than prior approaches). [S] — [arXiv 2509.16187](https://arxiv.org/html/2509.16187.pdf)
- **CodeMeepo** (arXiv 2610.34633, October 2026): multi-agent migration plus optimization, nine subjects, three language pairs. Java to Python 1,166 of 1,180 tests versus 1,029 for ReCodeAgent; C to Rust 63 of 63 versus 47; a production daemon 105 of 105 versus 53. The authors attribute ReCodeAgent's gap largely to translating the whole repository at once without milestone gates. [S] — [arXiv 2610.34633](https://arxiv.org/html/2610.34633)
- **ECAT** (arXiv 2608.09273): adversarial generator/discriminator for repository migration, compared against the RepoTransBench ReAct agent. No numbers retrieved. — [arXiv 2608.09273](https://arxiv.org/html/2608.09273)
- **MockingBird** (arXiv 2511.21878): context-aware repository translation with isolated validation; resolves cross-language type mappings from API documentation and context. No numbers retrieved. — [arXiv 2511.21878](https://arxiv.org/html/2511.21878)
- **DepWareTrans** (arXiv 2608.14128): translates dependency-consistent batches instead of isolated files, with compile- and test-driven validation, for co-executable JVM and .NET language pairs. No numbers retrieved. — [arXiv 2608.14128](https://arxiv.org/html/2608.14128)

#### C to Rust line
- **Syzygy** (arXiv 2412.14234): dual code-and-test translation with dynamic analysis. Translated Zopfli (about 3,000 LoC, 98 functions) to about 4,500 lines of safe Rust; equivalence tested on one million random inputs at the entry point, reaching over 95% line and 83% branch coverage of the C source; about 15 hours and about $2,500 (under $800 with cheaper GPT-4o models); Rust up to 3.67x slower. [S] — [arXiv 2412.14234](https://arxiv.org/html/2412.14234v1)
- **SACTOR** (ACL 2026; arXiv 2503.12511): two stages (unidiomatic then idiomatic), static analysis, FFI-based verification against the C. 93% and 84% end-to-end correctness on two datasets with DeepSeek-R1; on CRust-Bench 788 of 966 functions (81.57%) at the unidiomatic stage and 249 of 580 (42.93%) at the idiomatic stage. Numbers differ slightly across versions. [S] — [arXiv 2503.12511](https://arxiv.org/html/2503.12511), [ACL Anthology](https://aclanthology.org/2026.acl-long.28v2.pdf)
- **C2SaferRust** (Nitin et al.; arXiv 2501.14257): starts from C2Rust transpiler output and uses an LLM to make slices safer, with tests. On seven GNU Coreutils programs: raw pointers down 38%, dereferences down 27% (one version); another version reports unsafe lines, casts and calls down 24%, 8% and 24%. [S] — [arXiv 2501.14257](https://arxiv.org/html/2501.14257v2)
- **VERT** (arXiv 2404.18852): uses WebAssembly compiled from the source as a reference model and checks the LLM's Rust with property-based testing and bounded model checking. [S] — [arXiv 2404.18852](https://arxiv.org/html/2404.18852)
- **FLOURINE** ("Towards Translating Real-World Code with LLMs: A Study of Translating to Rust"; arXiv 2405.11514): cross-language differential fuzzer for C/Go to Rust; counterexamples fed back to the LLM. [S] — [arXiv 2405.11514](https://arxiv.org/html/2405.11514)
- **"From C to Idiomatic Rust: A Ship-of-Theseus Agentic Translation"** (Sartakov; arXiv 2607.28835): generate a semantics-preserving non-idiomatic Rust baseline, then rewrite incrementally to idiomatic Rust, validating every step by compilation and behavioural tests; case study iodine, 12.5k SLOC. [S] — [arXiv 2607.28835](https://arxiv.org/abs/2607.28835)
- **Rustine** (Dehghan et al.; arXiv 2511.20617): automated repository-level C to idiomatic safe Rust. No numbers retrieved. — [arXiv 2511.20617](https://arxiv.org/abs/2511.20617)
- **EvoC2Rust** (arXiv 2508.04295): skeleton-guided project-level C to Rust. Title only. — [arXiv 2508.04295](https://arxiv.org/html/2508.04295v2)

#### Migration (same-language upgrade) benchmarks and industrial reports
- **MigrationBench** (arXiv 2505.09569): Java 8 to 17. SD-Feedback with Claude-3.5-Sonnet-v2 reaches about 62% pass@1 for minimal migration and about 27% for maximal migration on a 300-repository subset; results depend on the iteration budget (up to 80 LLM calls). [S] — [arXiv 2505.09569](https://arxiv.org/abs/2505.09569v2)
- **FreshBrew** (Google; arXiv 2510.04852; ACM DOI 10.1145/3744916.3773214): 228 Java projects, migration to JDK 17 and 21. A migration counts only if the build and tests pass and line coverage does not fall more than 5 percentage points below the Java 8 baseline, a guard against reward hacking. Best JDK 17 result: Gemini 2.5 Flash 52.3% (ACM version; 54.3% in arXiv v1); lowest 10.7%. [S] — [arXiv 2510.04852](https://arxiv.org/html/2510.04852v1), [ACM DL](https://dl.acm.org/doi/full/10.1145/3744916.3773214)
- **"How is Google using AI for internal code migrations?"** (Nikolov et al.; ICSE 2025 SEIP; arXiv 2501.06972). int32 to int64 IDs in Ads: 80% of landed code AI-authored, estimated 50% time saving including review and rollout. JUnit3 to JUnit4: 5,359 files and 149,000+ lines in three months, 87% of AI-generated code committed unmodified; review became the bottleneck. Joda to java.time: estimated 89% time saving, an estimate with no supporting detail. [2 for the figures, via coverage of the paper] — [arXiv 2501.06972](https://arxiv.org/html/2501.06972v1), [The Register](https://www.theregister.com/2025/01/16/google_ai_code_migration/)
- **"Migrating Code At Scale With LLMs At Google"** (Ziftci, Nikolov, Sjövall, Kim, Codecasa, Kim; FSE 2025 Industry; arXiv 2504.09691): the int32 to int64 migration in detail. Pipeline: locate code to change, LLM generates edits, automated validation. 595 code changes submitted; about 50% effort reduction and over 70% of changes produced with the tooling. [S for 595; 2 for the percentages] — [arXiv 2504.09691](https://arxiv.org/abs/2504.09691v1), [FSE 2025 listing](https://conf.researchr.org/details/fse-2025/fse-2025-industry-papers/14/Migrating-Code-At-Scale-With-LLMs-At-Google)

#### Background (pre-2024 and function-level)
- **"Lost in Translation"** (Pan et al.; ICSE 2024; arXiv 2308.03109): C, C++, Go, Java, Python; seven LLMs including GPT-4; correct translations ranged from 2.1% to 47.3%. [S] — [arXiv 2308.03109](https://arxiv.org/abs/2308.03109)
- **CoTran** (arXiv 2306.06755): introduces AVATAR-TC (Java/Python whole programs); Python to Java is the harder direction, 48.68% functional-equivalence accuracy versus 38.26% for the nearest competitor. [S] — [arXiv 2306.06755](https://arxiv.org/pdf/2306.06755)
- **ClassEval-T** (ACM, DOI 10.1145/3728940): class-level Python to Java and Python to C++ benchmark. Its manual rules map dynamically assigned Python variables to `Object` in Java and Python reals to `double`. [S] — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3728940)
- **TransAgent** (arXiv 2409.19894): multi-agent function-level translation with fine-grained execution alignment; outperforms UniTrans. [S] — [arXiv 2409.19894](https://arxiv.org/html/2409.19894v2)
- **Survey**: "Large Language Models for Multilingual Code Intelligence: A Survey" (arXiv 2604.25960, April 2026) covers translation; notes repository-level tasks need reasoning over cross-file dependencies, build systems and configuration. [S] — [arXiv 2604.25960](https://arxiv.org/html/2604.25960v1)
- "An Extensive Empirical Study on Code Translation Techniques" (arXiv 2608.20776) exists; contents not retrieved. — [arXiv 2608.20776](https://arxiv.org/html/2608.20776)

### Inferences
- Reported success depends more on the benchmark's scoring unit than on the method. Per-test or per-function metrics (ReCodeAgent 86.5%, Oxidizer 73%) look far better than per-repository all-tests-pass metrics (RepoTransBench 32.8%), so numbers across papers should not be ranked against each other.
- High numbers come from small, curated subject sets (AlphaTrans 10, Oxidizer 7, CodeMeepo 9). Broad benchmarks stay under 50%.
- For a Python to .NET port, the closest evidence is RepoTransBench's dynamic-to-static result (below 10%) and the Java to C# results (TransRepo-bench about 16 to 18% test pass; RepoTransAgent 45.84%). Expect the Python-to-static direction to be the hard case even with an agent.

### Gaps
- No paper found on repository-level Python to C#. RepoTransBench has no Python to C# pair, and its README gives no per-pair numbers (the Python to Java figure is in the paper, which could not be opened).
- TransCoder / TransCoder-ST, CodeTransOcean and xCodeEval: no numbers retrieved this session; treat as background only.
- No Amazon or Meta peer-reviewed migration paper was found. The search surfaced only a reference to Meta's Java-to-Kotlin engineering blog and an Amazon Science page titled "Agentic translation of C to Rust" whose content was not retrieved ([Amazon Science](https://www.amazon.science/publications/agentic-translation-of-c-to-rust)).
- AlphaTrans, Oxidizer, MigrationBench author lists and the exact meaning of AlphaTrans's three ablation metrics were not confirmed from the primary text.

## 2. Techniques that measurably improve translation correctness

### Takeaway
The best-evidenced gains come from execution feedback loops (0% to 12.4% to 32.8% on RepoTransBench), staged translation with gates between stages, and validating each unit against the source implementation instead of only against translated tests. Evidence for type-inference-before-translation and property-based testing as separate techniques is thin; they appear inside systems without isolated ablations.

### Cited Findings
- **Execution feedback / repair loop.** RepoTransBench: translation only 0.0%, error feedback 12.4%, agent 32.8% repository success. [D] — [RepoTransBench GitHub](https://github.com/DeepSoftwareAnalytics/RepoTransBench). V1: iterative debugging added 7.09 points. [S] — [arXiv 2412.17744v1](https://arxiv.org/html/2412.17744v1)
- **Iteration budget matters.** MigrationBench reports efficacy depends on the number of LLM calls allowed (up to 80). [S] — [arXiv 2505.09569](https://arxiv.org/html/2505.09569v2)
- **Skeleton / interface first.** TransRepo-bench translates structural skeletons first and then bodies guided by them, which also allows per-test scoring because the skeleton fixes the interfaces the tests bind to. [S] — [arXiv 2501.16050](https://arxiv.org/abs/2501.16050). AlphaTrans and EvoC2Rust also build skeletons first. [S] — [arXiv 2410.24117](https://arxiv.org/html/2410.24117v4), [arXiv 2508.04295](https://arxiv.org/html/2508.04295v2)
- **Source-side program transformation before translation.** AlphaTrans normalizes source constructs with no target equivalent (for example method overloading); removing this step cut its three validation metrics by roughly 3 to 5x (24.50% to 4.58%, 2.88% to 0.81%, 9.76% to 3.05%). [S] — [ACM DL](https://dl.acm.org/doi/10.1145/3729379)
- **Dependency-consistent ordering / batching.** DepWareTrans translates dependency-consistent batches and claims this substantially improves repository migration (no number retrieved). [S] — [arXiv 2608.14128](https://arxiv.org/html/2608.14128). "Dependency-Guided Repository-Level C-to-Rust Translation with Reinforcement Alignment" (arXiv 2604.02852) exists; contents not retrieved. — [arXiv 2604.02852](https://arxiv.org/pdf/2604.02852)
- **Milestone gates instead of whole-repository translation.** CodeMeepo attributes its lead over ReCodeAgent (for example 105 of 105 versus 53 tests on a daemon) largely to gating progress at milestones. [S] — [arXiv 2610.34633](https://arxiv.org/html/2610.34633)
- **Two-stage "faithful first, idiomatic second".** SACTOR: 81.57% of functions pass at the unidiomatic stage, 42.93% at the idiomatic stage, so the refactoring stage is where correctness is lost. [S] — [arXiv 2503.12511](https://arxiv.org/html/2503.12511). The Ship-of-Theseus paper and C2SaferRust use the same order, starting from a behaviour-preserving baseline and rewriting in validated steps. [S] — [arXiv 2607.28835](https://arxiv.org/abs/2607.28835), [arXiv 2501.14257](https://arxiv.org/html/2501.14257v2)
- **Differential fuzzing against the source.** FLOURINE generates a fuzz harness that runs source and translation on the same inputs and feeds counterexamples back to the LLM; it needs no target-language unit tests. [S] — [arXiv 2405.11514](https://arxiv.org/html/2405.11514). The same loop with libFuzzer is studied in "Feedback Loops and Code Perturbations in LLM-based Software Engineering: A Case Study on a C-to-Rust Translation System". [S] — [arXiv 2512.02567](https://arxiv.org/html/2512.02567v1)
- **Per-function I/O equivalence with mocked callees (trace capture and replay).** Oxidizer validates 73% of functions (63 to 86%) by replaying I/O observed under the source test suite, with callees mocked. [S] — [ACM DL](https://dl.acm.org/doi/10.1145/3729315)
- **FFI-based verification.** SACTOR links translated Rust against the original C to test each unit in place. [S] — [arXiv 2503.12511](https://arxiv.org/html/2503.12511)
- **Test translation alongside code.** Syzygy translates code and tests together. [S] — [arXiv 2412.14234](https://arxiv.org/html/2412.14234v1). ReCodeAgent's translated tests were judged 88% accurate in a human study, meaning about one in eight translated tests was wrong. [S] — [arXiv 2604.07341](https://arxiv.org/html/2604.07341)
- **Generated specifications in the prompt.** SpecTra (Nitin, Krishna, Ray; arXiv 2405.18574) generates static specifications, test cases and natural-language descriptions, filters them for self-consistency against the source, and adds them to the prompt: up to 26% relative gain across six LLMs on C to Rust, C to Go, JavaScript to TypeScript (v1 reported up to 23% relative, 10% absolute). [S] — [arXiv 2405.18574](https://arxiv.org/abs/2405.18574v3)
- **API mapping retrieval.** APIRAT (arXiv 2504.14852) adds API sequence retrieval, back-translation and API mapping; its ablation says all three help and the combination is best (numbers not retrieved). [S] — [arXiv 2504.14852](https://arxiv.org/abs/2504.14852). A search extract of another study says LLMs score below 20% top-1 on cross-language API mapping; I could not identify which paper it came from. [2]
- **Type resolution.** MockingBird resolves cross-language type mappings from API docs and code context. [S] — [arXiv 2511.21878](https://arxiv.org/html/2511.21878). LLMs outperform HiTyper and HeaderGen at Python type inference in a 24-model study, which supports using an LLM to annotate types before translating. [S] — [Empirical Software Engineering](https://link.springer.com/article/10.1007/s10664-025-10704-3)
- **Multi-agent role splits.** RepoTransAgent (45.84% pass on Java/C#), ReCodeAgent (86.5% test pass) and CodeMeepo all split context gathering, translation and validation/repair into roles. [S] — [arXiv 2508.17720](https://arxiv.org/abs/2508.17720), [arXiv 2604.07341](https://arxiv.org/html/2604.07341), [arXiv 2610.34633](https://arxiv.org/html/2610.34633)
- **Property-based testing.** VERT combines property-based testing with bounded model checking against a Wasm reference. [S] — [arXiv 2404.18852](https://arxiv.org/html/2404.18852)

### Inferences
- A defensible order of operations from this evidence: normalize the source (remove constructs with no target equivalent, add types), fix the target skeleton, translate in dependency order behind milestone gates, validate each unit against the running source, and defer idiomatic rewriting to a second pass with the same checks.
- Role splits are confounded with feedback loops in every paper seen; no ablation isolates "more agents" from "more execution feedback".
- Translated tests are themselves translations and carry an error rate (about 12% in ReCodeAgent's study), so they need an independent check such as differential runs against the source.

### Gaps
- No ablation found that isolates leaf-first ordering, or type annotation before translation, as a single variable with a measured delta.
- No measured result found for golden-master/trace-replay as a named technique outside Oxidizer.
- APIRAT's numeric gains were not retrieved.

## 3. Dominant bug categories in LLM translations

### Takeaway
The measured failure sources are, in order of evidence strength: third-party library/API reference errors, type and data-type errors when moving from dynamic to static typing, compile-level errors, and boundary-input behaviour that tests never exercised. Numeric-semantics bugs are documented by example; no retrieved source quantified overflow, encoding, ordering or concurrency categories.

### Cited Findings
- "Lost in Translation" (Pan et al., ICSE 2024) builds a taxonomy of 15 bug categories in five groups from 1,748 manually labelled bugs (v1 of the paper: 14 categories, four groups, 1,725 bugs). [S] — [arXiv 2308.03109](https://arxiv.org/abs/2308.03109)
- The same study reports Python to Java errors are more frequent than the reverse because models struggle with the move from a dynamic to a static type system, and documents data-type bugs such as Python `//` translated to Java `/`. [S, from the v1-titled copy] — [ResearchGate copy](https://www.researchgate.net/publication/372961610_Understanding_the_Effectiveness_of_Large_Language_Models_in_Code_Translation)
- A later paper cites that 77.8% of large-model translation failures stem from compilation errors. [2] — [arXiv 2410.22818](https://arxiv.org/pdf/2410.22818)
- TransLibEval (FSE 2026; arXiv 2509.12087): third-party reference errors account for 79.58% of failed cases on average, and accuracy drops by over 60% compared with library-free translation. [S] — [arXiv 2509.12087](https://arxiv.org/html/2509.12087v2)
- CoTran attributes Python to Java difficulty to the translator having to infer types Python never declares. [S] — [arXiv 2306.06755](https://arxiv.org/pdf/2306.06755)
- Boundary inputs: an independent re-test found Syzygy's Rust Zopfli emits nothing on empty input where the C always emits a final block; the original test range excluded the empty input. [S] — [arXiv 2611.23839](https://arxiv.org/html/2611.23839)
- Across eleven C-to-Rust systems, the same re-test found 160 defects in 126 translations; 93 are outright errors on ordinary inputs. [S] — [arXiv 2611.23839](https://arxiv.org/abs/2611.23839)
- Nondeterminism defeats equivalence validation for some functions (Oxidizer). [S] — [ACM DL](https://dl.acm.org/doi/10.1145/3729315)
- A 2025 pilot study of agentic Android to iOS translation lists 10 issue types across method, file and package levels. [S] — [arXiv 2507.16037](https://arxiv.org/pdf/2507.16037)

### Inferences
- For a Python to .NET port, library mapping is the largest single risk (TransLibEval) and should be decided per dependency before translation starts, not left to the model.
- Empty, zero and boundary inputs should be mandatory in any differential test corpus; the Syzygy case shows a million random inputs can miss them.

### Gaps
- Per-category percentages of the Pan et al. taxonomy could not be retrieved (PDF hosts blocked). From recall, not verified: an iterative prompt-crafting step improved success by about 5.5% on average.
- No quantified data found for integer overflow, string/encoding, exception semantics, dict/set ordering or concurrency-model bugs specifically.

## 4. Classical migration methods (background, pre-LLM)

### Takeaway
Classical guidance converges on incremental replacement behind a stable interface, behaviour pinned by tests that record what the old code does, and porting faithfully before improving. The two best-documented compiler ports (Go, TypeScript) both chose a mechanical or structure-preserving port explicitly to preserve exact behaviour.

### Cited Findings
- **Characterization tests** (Feathers, *Working Effectively with Legacy Code*): tests that record what existing code actually does, not what it should do, as a safety net for change; they verify observed behaviour, can be generated automatically. [2] — [Wikipedia](https://wikipedia.com/wiki/Characterization_test)
- **Strangler fig**: migrate incrementally by replacing pieces of functionality while consumers keep using the same interface. [2] — [Microsoft Learn](https://learn.microsoft.com/azure/architecture/patterns/strangler-fig)
- **Parallel running during strangler migration**: Microsoft's example has the new system write to both stores and read from the legacy side to validate consistency. [2] — [Microsoft Learn](https://learn.microsoft.com/azure/architecture/patterns/strangler-fig)
- **Branch by abstraction**: for code deep in the stack that cannot be intercepted at the perimeter; create an abstraction, move clients to it, build the new implementation behind it; often combined with feature toggles. [2] — [AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/branch-by-abstraction.html)
- **Parallel change (expand/migrate/contract)** (Fowler bliki): make incompatible interface changes in three phases; skipping the contract phase can leave you worse off than before. [S] — [martinfowler.com](https://www.martinfowler.com/bliki/ParallelChange.html)
- **Transliterate first: Go compiler, C to Go** (Russ Cox, "Go, from C to Go", GopherCon 2014; "Go 1.3+ Compiler Overhaul" design doc): a mechanical conversion with a custom tool aimed at translating 99%+ of the code automatically; manual fixes were folded back into the tool or its configuration as special cases so the conversion stayed repeatable. [S] — [talk slides](https://golang.google.cn/talks/2014/c2go.slide), [design doc](https://golang.google.cn/s/go13compiler), [rsc/c2go](https://github.com/rsc/c2go)
- **Port, not rewrite: TypeScript to Go** (2025). Hejlsberg: "we were not going to rewrite... We were going to port because only by porting could we preserve the semantics, algorithms, and exact behavior of our existing compiler". Go was chosen partly because it allows a structurally similar codebase, so both versions can be maintained side by side. The team first wrote a tool for syntactic translation whose output had no syntax errors but did not compile, and worked from there. [2] — [The New Stack](https://thenewstack.io/go-power-microsofts-bold-bet-on-faster-typescript-tools/), [heise](https://heise.de/en/background/Web-development-The-new-TypeScript-compiler-in-Go-10357839.html?seite=2)
- **Fish shell, C++ to Rust**: a secondary post says the port took almost two years and over 200 contributors for 57,000 lines. [2, weak source] — [DEV Community](https://dev.to/dheerajakula/why-software-is-converging-on-rust-in-the-age-of-ai-18cj)

### Inferences
- The LLM-era "unidiomatic first, idiomatic second" pipelines (SACTOR, C2SaferRust, Ship-of-Theseus) are the same strategy as the Go and TypeScript ports: keep structure parallel so differences are attributable, then refactor under the same tests.
- Keeping manual fixes as replayable rules (Go) corresponds to keeping agent corrections as persistent instructions or scripts so the port can be re-run from a newer source revision.

### Gaps
- Fowler's Strangler Fig and "Patterns of Legacy Displacement" pages, GitHub Scientist, shadow-traffic practice and big-bang versus incremental comparisons could not be read (host blocked or no result). The strangler, branch-by-abstraction and characterization-test entries above are cited to secondary sources.
- The Go design doc's verification method (I recall it compared compiler output of the C and Go versions) and the mechanical-then-refactor sequence were not confirmed from the primary text.
- Fish's incremental, FFI-linked porting method is from recall only and has no reliable source here.

## 5. Equivalence checking limits and test-suite adequacy as an oracle

### Takeaway
Passing the translated or original test suite is weak evidence of equivalence: an independent 2026 re-test found defects in every one of eleven published C-to-Rust systems, and the systems' own checks had accepted most of the defective outputs. Formal equivalence is achieved only for small DSL-style targets or bounded checks; in practice the literature relies on bounded differential fuzzing plus guards against oracle tampering.

### Cited Findings
- "Re-Testing LLM-Based C-to-Rust Translations Against Their Original C" (Hassnain; arXiv 2611.23839, Nov 2026): of 34 surveyed translators only one reported testing its own correctness check; eleven released runnable code. Re-running C and Rust on generated inputs (counting divergences only where the C has no undefined behaviour) found 160 defects in 126 translations across all eleven systems; the systems' own checks had accepted 86 of the defective translations. [S] — [arXiv 2611.23839](https://arxiv.org/abs/2611.23839)
- A separate 2025 comparison found Syzygy had produced no test translation for zopfli. [2] — [arXiv 2611.23839](https://arxiv.org/html/2611.23839)
- Fuzzing equivalence is a bounded-time claim: FLOURINE deems a translation equivalent if 5 minutes of fuzzing finds no counterexample; arXiv 2512.02567 likewise treats no counterexample within budget as an indication only. [S] — [arXiv 2405.11514](https://arxiv.org/html/2405.11514), [arXiv 2512.02567](https://arxiv.org/html/2512.02567v1)
- "A Differential Fuzzing-Based Evaluation of Functional Equivalence in LLM-Generated Code Refactorings" (arXiv 2602.15761): reliance on existing tests may overestimate functional equivalence. [S] — [arXiv 2602.15761](https://arxiv.org/html/2602.15761)
- Coverage as an adequacy measure: Inozemtseva and Holmes (ICSE 2014, background) generated about 31,000 suites for five Java systems and found low-to-moderate correlation between coverage and mutation-measured effectiveness once suite size is controlled; coverage identifies under-tested code but is not a quality target. [S] — [paper PDF](https://cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf)
- A 2026 replicability study on LLM-generated suites finds coverage metrics still correlate with effectiveness but more weakly than the 2014 figures. [S, unusual ID, verify] — [arXiv 2607.22880](https://arxiv.org/pdf/2607.22880)
- Guarding the oracle: FreshBrew fails a migration whose line coverage drops more than 5 points, to stop agents deleting or disabling tests. [S] — [arXiv 2510.04852](https://arxiv.org/html/2510.04852v1). RepoMod-Bench hides its tests from the agent; CodeMeepo cites this as the reason to keep agent-generated tests separate from the oracle. [S] — [arXiv 2602.22518](https://arxiv.org/html/2602.22518v1), [arXiv 2610.34633](https://arxiv.org/html/2610.34633)
- Reference coverage reported by a translation paper: Syzygy's equivalence inputs reached over 95% line and 83% branch coverage of the C source, and still missed the empty-input bug. [S] — [arXiv 2412.14234](https://arxiv.org/html/2412.14234v1), [arXiv 2611.23839](https://arxiv.org/html/2611.23839)
- Formal routes: LLMLift ("Verified Code Transpilation with LLMs", NeurIPS 2024; arXiv 2406.03003) has the LLM produce both translation and proof for DSL targets. [S] — [arXiv 2406.03003](https://arxiv.org/abs/2406.03003). VERT uses bounded model checking against a Wasm reference. [S] — [arXiv 2404.18852](https://arxiv.org/html/2404.18852). IRONPROOF applies SMT equivalence checking to COBOL to Python and notes cross-language validation must bridge a much larger semantic gap than same-IR tools such as Alive2. [S] — [arXiv 2610.11073](https://arxiv.org/pdf/2610.11073)
- Position paper "LLM-Based Code Translation Needs Formal Compositional Reasoning" (Cheung, Seshia et al.; Berkeley tech report EECS-2025-174; a withdrawn ICLR 2026 submission): correctness, not plausibility, must govern translation, via hybrid workflows where formal tools constrain and certify. [S] — [Berkeley EECS](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-174.pdf), [OpenReview](https://openreview.net/forum?id=wGj8LU2EOf)
- MatchFixAgent agrees with other equivalence validators on only 72.8% of pairs, so automated verdicts are themselves noisy. [S] — [arXiv 2509.16187](https://arxiv.org/html/2509.16187.pdf)

### Inferences
- High line coverage of the source by the oracle is necessary but not sufficient (Syzygy). The oracle should run against the original implementation, include boundary inputs explicitly, and be immutable to the porting agent.
- A validation method should itself be tested, for example by seeding known divergences (mutants) into a translated unit and confirming the check catches them; only one of 34 surveyed systems did anything like this.

### Gaps
- No source found that prescribes a specific coverage or mutation-score threshold to reach before starting a port.
- No source retrieved on pseudo-tested methods or on undecidability of equivalence stated in the translation context.
- The full text of the Hassnain re-test (defect categories, which systems) was not read.

## 6. Agent harness design for long-running tasks relevant to ports

### Takeaway
The published guidance is to externalize state (a structured, tamper-resistant work list, a progress log, git history), work one unit per session, start every session by re-establishing a known-good state, and verify end to end; translation papers independently find that milestone gates and human review capacity, not code generation, set the pace. Academic evidence on harness design specifically for ports is sparse.

### Cited Findings
- Anthropic, "Effective harnesses for long-running agents" (Justin Young, 26 Nov 2025): an initializer session creates `init.sh`, a progress log, an initial git commit and a comprehensive feature list with every item marked failing; later sessions each do one feature. [D] — [Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- The feature list is JSON; coding agents may change only the `passes` field. The prompt states: "It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality." [D] — [Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Session start routine: run `pwd`, read the progress file and git log, review the feature list, start the system and run a basic end-to-end test before implementing anything new. [D] — [Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Failure modes named: declaring the project finished too early, leaving bugs or undocumented progress, marking features done prematurely, wasting time working out how to run the app. Unit tests and `curl` checks alone were not enough to show features worked; end-to-end testing as a user was needed. [D] — [Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Milestone gates in a translation agent: CodeMeepo attributes its advantage over whole-repository translation to gating. [S] — [arXiv 2610.34633](https://arxiv.org/html/2610.34633)
- Oracle separation: tests the agent writes are kept apart from the hidden acceptance tests. [S] — [arXiv 2610.34633](https://arxiv.org/html/2610.34633), [arXiv 2602.22518](https://arxiv.org/html/2602.22518v1)
- Anti-reward-hacking guard: coverage must not fall more than 5 points (FreshBrew). [S] — [arXiv 2510.04852](https://arxiv.org/html/2510.04852v1)
- Human review is the bottleneck at scale: in Google's JUnit migration, review rather than generation limited throughput; success needed file discovery, multi-stage validation and human review. [2] — [The Register](https://www.theregister.com/2025/01/16/google_ai_code_migration/), [arXiv 2501.06972](https://arxiv.org/html/2501.06972v1)
- Cost reference point: Syzygy spent about 15 hours and $2,500 on a 3,000-line library. [S] — [arXiv 2412.14234](https://arxiv.org/html/2412.14234v1)
- Compaction versus reset: compaction summarizes within a session and may lose rationale; a reset rebuilds from persisted artifacts and depends on their completeness. [2, community source] — [Seven Mental wiki](https://panqiwei.github.io/advanced-agentic-dev-patterns/en/wiki/concepts/long-running-agents)
- "SDAD: Spec-Driven Agentic Development for the AI-Native SDLC" (arXiv 2608.20341) exists; contents not retrieved. — [arXiv 2608.20341](https://arxiv.org/pdf/2608.20341)

### Inferences
- For a port, the "feature list" maps naturally onto the list of source units and source tests, each with a pass flag the agent can set only by running the differential check, with the check and the source tests read-only.
- The port's per-unit differential test plays the role of the article's end-to-end test: it is the fast, trustworthy signal each session runs first and last.
- Chunking work so that each reviewable change is small matters because review capacity, not generation, is the measured limit.

### Gaps
- No controlled study found comparing harness designs (progress files, context reset versus compaction, spec-driven versus direct) on a translation task.
- No academic source found on context-management strategies specific to repository translation.
