"use client";

/**
 * /memory-demo: the tool views rendered from captured fixtures, with no
 * backend and no model call. Each finished card is shown collapsed and
 * expanded; the conversation at the top shows the cue margin, a bracket and
 * a group of cues. Durations here are fixture values, not measurements.
 */
import type { ReactElement } from "react";
import type { Message } from "@ag-ui/client";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ToolCallCard } from "@/components/ToolCallCard";
import { Conversation } from "@/components/chat/Conversation";
import { AsOfResults } from "@/components/memory/AsOfResults";
import { FactTimeline } from "@/components/memory/FactTimeline";
import { ProvenanceCard } from "@/components/memory/ProvenanceCard";
import { DisclosureProvider, ToolCallInfoProvider } from "@/components/tool-card/context";
import { buildConversation } from "@/lib/conversation";
import type { ToolTiming } from "@/lib/use-run-events";
import familyBenefit from "@/components/memory/fixtures/timeline.family-benefit.json";
import unknownEntity from "@/components/memory/fixtures/timeline.unknown.json";
import invalidArgument from "@/components/memory/fixtures/search.invalid.json";
import searchNow from "@/components/memory/fixtures/search.now.json";
import search2021 from "@/components/memory/fixtures/search.2021.json";
import searchDegraded from "@/components/memory/fixtures/search.degraded.synthetic.json";
import provenanceFound from "@/components/memory/fixtures/provenance.rate-150.json";
import provenanceUnknown from "@/components/memory/fixtures/provenance.unknown.json";

interface DemoSection {
  title: string;
  note: string;
  view: ReactElement;
  /** Still running: shown once, not expandable. */
  running?: boolean;
}

const sections: DemoSection[] = [
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
    running: true,
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
    running: true,
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
    running: true,
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
  {
    title: "Generic card, running",
    note: "Any tool without a dedicated view. Arguments are complete, the result has not arrived; the bar grows.",
    running: true,
    view: <ToolCallCard name="lookup_order" args={{ order_id: "A-8844" }} status="executing" />,
  },
  {
    title: "Generic card, done with a JSON result",
    note: "Arguments table, the result as JSON with its size, and copy actions.",
    view: (
      <ToolCallCard
        name="lookup_order"
        args={{ order_id: "A-8844", include_items: true }}
        status="complete"
        result={{
          id: "A-8844",
          status: "delivered",
          total: 249.0,
          ship_date: "2026-04-03",
          items: [{ sku: "HP-NC-770", name: "NC-770 Headphones" }],
        }}
      />
    ),
  },
  {
    title: "Generic card, failed",
    note: "The tool message carried an error. The headline is the error's first sentence.",
    view: (
      <ToolCallCard
        name="send_email"
        args={{ template: "return_label", to: "customer" }}
        status="complete"
        result=""
        error="Timed out after 30 s. The mail server did not answer; the agent continued without sending the label."
      />
    ),
  },
  {
    title: "Generic card, unknown tool with a plain-text result",
    note: "A long text result starts shortened, with Show full result.",
    view: (
      <ToolCallCard
        name="mcp__filesystem__read_file"
        args={{ path: "./data/cias.md" }}
        status="complete"
        result={"Critical Infrastructure Assurance Standard, section 4. Operators must notify the agency within 72 hours of a reportable incident. ".repeat(6)}
      />
    ),
  },
];


/** A short thread built from the fixtures, to show cues, a bracket and a group. */
function call(id: string, name: string, args: unknown) {
  return { id, type: "function" as const, function: { name, arguments: JSON.stringify(args) } };
}

const THREAD: Message[] = [
  { id: "u1", role: "user", content: "What is the base rate of Family Benefit now, and what was it before?" },
  {
    id: "a1",
    role: "assistant",
    toolCalls: [
      call("t1", "memory_timeline", familyBenefit.args),
      call("t2", "memory_search", searchNow.args),
    ],
  },
  { id: "r1", role: "tool", toolCallId: "t1", content: JSON.stringify(familyBenefit.result) },
  { id: "r2", role: "tool", toolCallId: "t2", content: JSON.stringify(searchNow.result) },
  {
    id: "a2",
    role: "assistant",
    content:
      "The base rate is **150 dollars per fortnight** for each child, from 1 July 2024. Before that it was 120 dollars from 1 July 2022, 100 dollars from 1 July 2020 and 90 dollars from 1 July 2019.",
  },
  { id: "u2", role: "user", content: "Where does the 150 dollar figure come from, and did anything change in 2021?" },
  {
    id: "a3",
    role: "assistant",
    toolCalls: [
      call("t3", "memory_provenance", provenanceFound.args),
      call("t4", "memory_search", search2021.args),
      call("t5", "memory_search", invalidArgument.args),
      call("t6", "lookup_order", { order_id: "A-8844" }),
    ],
  },
  { id: "r3", role: "tool", toolCallId: "t3", content: JSON.stringify(provenanceFound.result) },
  { id: "r4", role: "tool", toolCallId: "t4", content: JSON.stringify(search2021.result) },
  { id: "r5", role: "tool", toolCallId: "t5", content: JSON.stringify(invalidArgument.result) },
  { id: "r6", role: "tool", toolCallId: "t6", content: "", error: "Timed out after 30 s." },
  {
    id: "a4",
    role: "assistant",
    content:
      "The 150 dollar rate comes from `docs/rate-2024.md`, chunk 1, dated 1 July 2024. In 2021 the young child supplement started; the base rate stayed at 100 dollars.",
  },
];

const T0 = Date.UTC(2026, 10, 13, 14, 2);
const THREAD_TIMINGS = new Map<string, ToolTiming>([
  ["t1", { start: T0, end: T0 + 1940 }],
  ["t2", { start: T0 + 2000, end: T0 + 3120 }],
  ["t3", { start: T0 + 9000, end: T0 + 9210 }],
  ["t4", { start: T0 + 9300, end: T0 + 10420 }],
  ["t5", { start: T0 + 10500, end: T0 + 10980 }],
  ["t6", { start: T0 + 11000, end: T0 + 41000 }],
]);

const DEMO_SCALE = 2400;
const DEMO_MS = [1940, 1120, 880, 480, 610, 210, 2310, 740];

function Card({ id, open, running, index, children }: { id: string; open: boolean; running?: boolean; index: number; children: ReactElement }) {
  const elapsedMs = DEMO_MS[index % DEMO_MS.length] * (running ? 0.3 : 1);
  return (
    <ToolCallInfoProvider value={{ key: id, timing: { elapsedMs, scaleMs: DEMO_SCALE, running: !!running } }}>
      <div data-demo-state={open ? "expanded" : "collapsed"}>{children}</div>
    </ToolCallInfoProvider>
  );
}

const initialOpen = Object.fromEntries(sections.map((_, index) => [`demo:${index}:open`, true]));

export default function MemoryDemoPage() {
  const thread = buildConversation(THREAD, false);
  return (
    <DisclosureProvider initial={{ ...initialOpen, "group:t3": true, "demo:0:open:chart": true }}>
      <div className="pb-demo">
        <main className="pb-demo__main">
          <div className="pb-demo__head">
            <div>
              <h1>Tool views</h1>
              <p>Rendered from fixtures. No backend, no model calls; durations are fixture values.</p>
            </div>
            <ThemeToggle />
          </div>

          <section className="pb-demo__section">
            <h2>A thread on the cue margin</h2>
            <p>Two turns. The first answer is bracketed to Q1 and Q2; the second turn used four tools, so they sit in one group.</p>
            <Conversation
              rows={thread.rows}
              timings={THREAD_TIMINGS}
              now={T0 + 41000}
              isRunning={false}
              runError={null}
              onEditResend={() => {}}
            />
          </section>

          {sections.map((section, index) => (
            <section key={section.title} className="pb-demo__section">
              <h2>{section.title}</h2>
              <p>{section.note}</p>
              <div className="pb-demo__pair">
                <Card id={`demo:${index}:closed`} open={false} running={section.running} index={index}>
                  {section.view}
                </Card>
                {section.running ? null : (
                  <Card id={`demo:${index}:open`} open index={index}>
                    {section.view}
                  </Card>
                )}
              </div>
            </section>
          ))}
        </main>
      </div>
    </DisclosureProvider>
  );
}
