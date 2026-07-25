# Golden Corpus Manifest — Policy Generator (038)

> **Retrieved:** 2026-07-25 · **Domain:** Australian employment services compliance
> (Targeted Compliance Framework, mutual obligations, Points Based Activation System).
>
> Purpose: pin the *exact* policy documents the 038 fidelity evaluation scores
> against. TCF is under active review (see Volatility below), so every entry
> records a retrieval date and a content hash. A fidelity number is meaningless
> without knowing which version of the policy produced it.
>
> **Pinned documents live in `corpus/`** and are committed, because a golden
> corpus is worthless if the artifact it scores against can drift. Verify with
> `shasum -a 256 corpus/*`; a mismatch against the hashes below means the file
> was replaced, not that policy changed.
>
> **Repo weight: ~5.5 MB**, dominated by the 4.6 MB Guidelines PDF. **Decided
> 2026-07-25: accepted, tracked in git.** A pinned corpus that can drift is not
> a corpus, and an external store is one more thing to lose. Do not re-litigate
> this without a concrete reason — rewriting history to extract them later is
> far more expensive than carrying them.

## Status legend

| Status | Meaning |
|---|---|
| `PINNED` | Fetched, content verified as substantive, sha256 recorded |
| `MANUAL` | Publicly readable in a browser, but not machine-fetchable — needs manual capture |

## Sources

### 1. PBAS points values — `PINNED`

- **URL:** https://ministers.dewr.gov.au/sites/default/files/documents/Points%20values%20for%20tasks%20and%20activities%20in%20the%20PBAS.pdf
- **sha256:** `d5d23499883972e4795ca5cf23ba45346cde6ddd5a91f5073b5e8e154b7da8c9`
- **Format:** PDF, 2 pages, 281,887 bytes · text extracts cleanly via `pypdf`
- **Content verified:** yes — full activity/points table extracted

The most decision-table-shaped artifact in the corpus. Contains all four rule
patterns the generator must handle:

| Pattern | Example from the document |
|---|---|
| Flat value | `Completing a job application (job search) — 5 points` |
| Periodic cap | `Creating and updating the profile — 5 points (maximum of 5 points per month)` |
| Annual/count cap | `Attending a job fair — 20 points (one job fair per year)`; `Youth Advisory Sessions — limited to 3 sessions per year` |
| Threshold tier | `Education and training — 20 points/week (contact hours over 15 hours per week); 15 points/week (contact hours up to 15 hours per week)` |

**Deliberate negative case** — the document's own footnote is *not* mechanizable:

> "Providers and the DSCC **may increase** the values of certain tasks or activities
> through an activity bonus **to reflect the individual circumstances** of the
> participant."

A generator that emits a rule for this is wrong in the way that matters. This is
the corpus's canonical unmappable-clause test: the correct output is an explicit
"could not map" marker, not an invented threshold.

Footnote `(1)` ("available to Workforce Australia Services participants only")
is a second test — an eligibility precondition attached to rows, not a rule.

### 2. Reasonable Excuse Determination 2018 — `PINNED`

- **File:** `corpus/reasonable-excuse-determination-2018.pdf`
- **URL:** https://www.legislation.gov.au/Details/F2018L00779
- **sha256:** `a8144e0a6cd46554fcff49464d9f22be39061d1b7dbf8264102cf9c1568068d4`
- **Format:** PDF, 7 pages, 604,438 bytes · Authorised Version, registered 15/06/2018
- **Instrument:** *Social Security (Administration) (Reasonable Excuse – Participation Payments) Determination 2018*, made under ss 42AI(1), 42AI(3), 42U(1), 42U(3) of the *Social Security (Administration) Act 1999*
- **Note:** the URL is a JS-rendered SPA and is **not** machine-fetchable (four
  endpoints probed, all shells). This copy was captured manually from a browser.

**The single most valuable document in the corpus.** Seven pages containing a
complete difficulty gradient — every case the generator must distinguish:

| Provision | Shape | Generator must |
|---|---|---|
| **s5(2)(a)–(j)** | A **closed enumerated list** of ten mandatory factors (housing, literacy, illness, cognitive impairment, drug/alcohol dependency, caring responsibilities, criminal violence, bereavement, paid work, job interview) | Emit a **gate schema**, not a table — ten typed fields |
| **s5(3)** | Sub-tests expanding (a): housing is inadequate if it damages health, threatens safety, lacks amenities, is unaffordable, or there is no right to remain | Nested boolean rules — mechanizable |
| **s6(3)** | *"any matter if the Secretary **is not satisfied** that the matter **directly prevented** the person from meeting the requirement"* | **Refuse to map.** Pure discretion. An invented rule here is the failure mode that matters |
| **s6(4)** | Drug/alcohol exclusion: conditions (a)–(d) **all** required, defeated by **any** of exceptions (e)–(h), where (h) itself requires a qualified medical opinion | A genuinely hard decision table — conjunction plus a disjunction of exceptions |

This is why the sample's schema gate is not a design choice. **s5(2) enumerates
the fields; s6(3) marks where judgement begins.** The statute specifies both the
type of the fact and the boundary of discretion — HoloDeck only has to honour it.

s6(4) is also the corpus's hardest positive case: fully determinable, but a
generator that drops the `unless` limb produces a table that wrongly denies
excuses to people who did engage with treatment. Exactly the class of error the
fidelity SC exists to catch.

### 3. Social Security Guide §3.11.13 (TCF) — `MANUAL`

- **URL:** https://guides.dss.gov.au/social-security-guide/3/11/13
- **Sections:** `3.11.13.10` zones · `3.11.13.30` types of failures ·
  `3.11.13.40` suspensions, demerits & reconnection · `3.11.13.50` financial penalties
- **Why manual:** returns HTTP 403 to all automated clients (WebFetch and curl
  with a browser UA both refused). Browser-readable only.

Best-structured source in the corpus: numbered, individually citable sections —
a natural fit for `provenance.source` (e.g. `"Social Security Guide 3.11.13.50"`).

Rules to capture (per secondary sources; **must be verified against the primary
before use as golden expectations**):

- 1 demerit per mutual obligation failure without a valid reason
- 3 demerits (fast-track) for failure to act on a job referral, or non-attendance/misconduct at a job interview
- A demerit expires after 6 active months
- 3 demerits in 6 active months → capability interview
- 5 demerits in 6 active months → Penalty Zone (subject to a capability assessment finding requirements appropriate)
- Penalty Zone escalation: 1st failure → 1 week's payment; 2nd → 2 weeks'; 3rd → cancellation + 4-week re-application preclusion
- 3 active months fully compliant → return to Green Zone, demerits reset to zero

### 4. Workforce Australia Guidelines Part B v1.24 — `PINNED`

- **File:** `corpus/wa-guidelines-part-b-v1.24.pdf`
- **URL:** https://www.dewr.gov.au/download/13950/workforce-australia-guidelines-part-b-workforce-australia-services/40448/workforce-australia-guidelines-part-b-workforce-australia-services/pdf
- **sha256:** `35cfdef97f7504e6fb0dd03ed3b578e3fbf8b1954b4781742d60f1461f7cdf93`
- **Format:** PDF, **416 pages**, 4,585,415 bytes · **v1.24, effective 1 July 2026**
- **Note:** the `/pdf` URL returns a JS-rendered landing page; captured manually.
  This is a **newer revision** than search metadata suggested (397pp / 1 July 2025) —
  evidence for the Volatility section: this document revises at least annually.

Operational detail behind the Guide's summaries. **Do not feed all 416 pages to a
generator** — extract the relevant spans. Term frequency scan:

| Term | Pages | Notable cluster |
|---|---|---|
| `mutual obligation` | 139 | pervasive — poor selector on its own |
| `Points Target` | 33 | **156–190** (PBAS operational rules) |
| `Targeted Compliance` | 32 | 1, 3, 24, 28, 90, 102–103, 156 |
| `demerit` | 29 | **231–236** |
| `capability interview` | 27 | **231–234** |
| `penalty zone` | 4 | **233–234**, 267 |
| `reasonable excuse` | 4 | 170, 268, **271–273** |

**Start at pp. 231–236** — demerits, penalty zone, and capability interviews all
co-occur there, which is the TCF operational core. pp. 271–273 covers reasonable
excuse in practice and should be cross-read against source 2 (the Determination),
since the Guideline is the *provider-facing* rendering of the same statutory test —
a useful consistency check on whether the generator produces the same table from
a legal instrument and from its operational restatement.

**Caveat carried by the document itself** (p.1): *"This Guideline is not a
stand-alone document and does not contain the entirety of Provider obligations. It
must be read in conjunction with the … Deed."* Golden expectations derived from
this file alone are therefore incomplete by the publisher's own admission — prefer
source 2 or 3 as the authority where they overlap.

## Volatility

TCF is under active remediation. DEWR maintains an
["assuring integrity of the Targeted Compliance Framework"](https://www.dewr.gov.au/assuring-integrity-targeted-compliance-framework)
program, the Commonwealth Ombudsman has published
[*Fairness in the Targeted Compliance Framework*](https://www.ombudsman.gov.au/__data/assets/pdf_file/0015/323205/Fairness-in-the-Targeted-Compliance-Framework.pdf),
and some decisions under the SS(Admin) Act are reported as paused pending review.

Consequences for 039:

1. **Never score against a live URL.** Score against the pinned snapshot only.
2. **Hash mismatch is a signal, not an error** — it means the policy moved and
   the golden expectations need re-derivation.
3. Each generated table records `provenance.source_sha256`, so a determination
   made under a superseded policy version stays identifiable after the fact.
   (This is the same property that Robodebt lacked.)

## Outstanding

- [ ] Manually capture sources 2–4; record sha256 + retrieval date here
- [ ] Verify the §3.11.13 rules above against the primary source before they
      become golden expectations — they are currently sourced from secondary
      summaries and are **not** authoritative
- [ ] Decide whether the fidelity SC scores decomposition (DRD shape) or only
      table content (recommendation: tables numerically, decomposition qualitatively)
