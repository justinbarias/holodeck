"use client";

/**
 * /memory-demo: the memory tool views rendered from captured fixtures, with
 * no backend and no model call. Use it to look at every state of a view.
 */
import { ThemeToggle } from "@/components/ThemeToggle";
import { AsOfResults } from "@/components/memory/AsOfResults";
import { FactTimeline } from "@/components/memory/FactTimeline";
import { ProvenanceCard } from "@/components/memory/ProvenanceCard";
import familyBenefit from "@/components/memory/fixtures/timeline.family-benefit.json";
import unknownEntity from "@/components/memory/fixtures/timeline.unknown.json";
import invalidArgument from "@/components/memory/fixtures/search.invalid.json";
import searchNow from "@/components/memory/fixtures/search.now.json";
import search2021 from "@/components/memory/fixtures/search.2021.json";
import searchDegraded from "@/components/memory/fixtures/search.degraded.synthetic.json";
import provenanceFound from "@/components/memory/fixtures/provenance.rate-150.json";
import provenanceUnknown from "@/components/memory/fixtures/provenance.unknown.json";

const sections = [
  {
    title: "Timeline with eight facts",
    note: "timeline.family-benefit.json. One fact has no known start date.",
    view: (
      <FactTimeline
        name={familyBenefit.name}
        args={familyBenefit.args}
        status="complete"
        result={familyBenefit.result}
      />
    ),
  },
  {
    title: "Same result as a JSON string",
    note: "How the Claude backend can deliver it.",
    view: (
      <FactTimeline
        name="memory_timeline"
        args={familyBenefit.args}
        status="complete"
        result={JSON.stringify(familyBenefit.result)}
      />
    ),
  },
  {
    title: "Empty timeline",
    note: "timeline.unknown.json",
    view: (
      <FactTimeline
        name={unknownEntity.name}
        args={unknownEntity.args}
        status="complete"
        result={unknownEntity.result}
      />
    ),
  },
  {
    title: "Error result",
    note: "The error body from search.invalid.json; every memory tool uses this shape.",
    view: (
      <FactTimeline
        name="memory_timeline"
        args={{ entity: "Family Benefit" }}
        status="complete"
        result={invalidArgument.result}
      />
    ),
  },
  {
    title: "In progress",
    note: "Before the result arrives.",
    view: (
      <FactTimeline
        name="memory_timeline"
        args={{ entity: "Family Benefit" }}
        status="inProgress"
      />
    ),
  },
  {
    title: "Unreadable result",
    note: "Falls back to the generic tool-call card.",
    view: (
      <FactTimeline
        name="memory_timeline"
        args={{ entity: "Family Benefit" }}
        status="complete"
        result="Internal error: <html>502 Bad Gateway</html>"
      />
    ),
  },
  {
    title: "Search, valid now",
    note: "search.now.json. Valid and known on the same day, so the header says so once.",
    view: (
      <AsOfResults
        name={searchNow.name}
        args={searchNow.args}
        status="complete"
        result={searchNow.result}
      />
    ),
  },
  {
    title: "Search, valid in 2021",
    note: "search.2021.json. Valid and known on different days. One fact has no known start date. Delivered as a JSON string.",
    view: (
      <AsOfResults
        name={search2021.name}
        args={search2021.args}
        status="complete"
        result={JSON.stringify(search2021.result)}
      />
    ),
  },
  {
    title: "Search, degraded (synthetic)",
    note: "search.degraded.synthetic.json. SYNTHETIC: invented text about an invented payment. The real fixtures have no passage hits and no degraded result.",
    view: (
      <AsOfResults
        name={searchDegraded.name}
        args={searchDegraded.args}
        status="complete"
        result={searchDegraded.result}
      />
    ),
  },
  {
    title: "Search, no hits",
    note: "Built in the page from the search result shape: valid_at and known_at with an empty hits list.",
    view: (
      <AsOfResults
        name="memory_search"
        args={{ query: "base rate of Family Benefit", valid_at: "1990-01-01" }}
        status="complete"
        result={{
          valid_at: "1990-01-01T00:00:00Z",
          known_at: "2026-11-11T12:00:00Z",
          degraded: false,
          hits: [],
        }}
      />
    ),
  },
  {
    title: "Search, error result",
    note: "search.invalid.json",
    view: (
      <AsOfResults
        name={invalidArgument.name}
        args={invalidArgument.args}
        status="complete"
        result={invalidArgument.result}
      />
    ),
  },
  {
    title: "Search, in progress",
    note: "Before the result arrives.",
    view: (
      <AsOfResults
        name="memory_search"
        args={{ query: "base rate of Family Benefit" }}
        status="inProgress"
      />
    ),
  },
  {
    title: "Search, unreadable result",
    note: "Falls back to the generic tool-call card.",
    view: (
      <AsOfResults
        name="memory_search"
        args={{ query: "base rate of Family Benefit" }}
        status="complete"
        result="Internal error: <html>502 Bad Gateway</html>"
      />
    ),
  },
  {
    title: "Provenance, found",
    note: "provenance.rate-150.json. One source with its passage.",
    view: (
      <ProvenanceCard
        name={provenanceFound.name}
        args={provenanceFound.args}
        status="complete"
        result={provenanceFound.result}
      />
    ),
  },
  {
    title: "Provenance, fact not found",
    note: "provenance.unknown.json",
    view: (
      <ProvenanceCard
        name={provenanceUnknown.name}
        args={provenanceUnknown.args}
        status="complete"
        result={provenanceUnknown.result}
      />
    ),
  },
  {
    title: "Provenance, found with no sources",
    note: "Built in the page from provenance.rate-150.json with the sources removed.",
    view: (
      <ProvenanceCard
        name="memory_provenance"
        args={provenanceFound.args}
        status="complete"
        result={{
          ...provenanceFound.result,
          fact: { ...provenanceFound.result.fact, sources: [] },
          sources: [],
        }}
      />
    ),
  },
  {
    title: "Provenance, in progress",
    note: "Before the result arrives.",
    view: (
      <ProvenanceCard
        name="memory_provenance"
        args={provenanceFound.args}
        status="inProgress"
      />
    ),
  },
  {
    title: "Provenance, error result",
    note: "Same error shape as every memory tool.",
    view: (
      <ProvenanceCard
        name="memory_provenance"
        args={provenanceFound.args}
        status="complete"
        result={{ error: { code: "backend_unavailable", message: "The memory graph is not reachable." } }}
      />
    ),
  },
];

export default function MemoryDemoPage() {
  return (
    <div className="holodeck-demo">
      <header className="holodeck-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1>Memory views</h1>
            <p>Rendered from fixtures. No backend, no model calls.</p>
          </div>
          <ThemeToggle />
        </div>
      </header>
      <main className="holodeck-demo__main">
        {sections.map((section) => (
          <section key={section.title} className="holodeck-demo__section">
            <h2>{section.title}</h2>
            <p>{section.note}</p>
            {section.view}
          </section>
        ))}
      </main>
    </div>
  );
}
