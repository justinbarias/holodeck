/**
 * Result shapes of the HoloDeck temporal memory tools.
 *
 * Written from the published JSON schemas in the HoloDeck repository
 * (schemas/memory/*.result.schema.json and error.schema.json). The schemas are
 * the contract; keep these types in step with them.
 *
 * Every timestamp is an ISO 8601 UTC string or null.
 */

/** How the document date of a source was found. */
export type ReferenceTimeSource = "front_matter" | "mtime" | "ingestion";

/** One episode (document passage) that supports a fact. `SourceOut`. */
export interface MemorySource {
  episode_id: string;
  source_tool: string;
  chunk_id: string;
  source_uri: string;
  /** The document date. */
  reference_time: string;
  reference_time_source: ReferenceTimeSource;
  /** The passage text. Present on provenance results; often null elsewhere. */
  text?: string | null;
}

/** One fact about an entity, open or closed. `TimelineEntryOut`. */
export interface TimelineEntry {
  fact_id: string;
  text: string;
  /** null: no known start. */
  valid_from: string | null;
  /** null: the fact is still open (current). */
  valid_to: string | null;
  recorded_at: string | null;
  expired_at: string | null;
  sources: MemorySource[];
}

/** Result of `memory_timeline`. */
export interface MemoryTimelineResult {
  entity: string;
  entries: TimelineEntry[];
}

/** One search hit: a fact or a source passage. `MemoryHitOut`. */
export interface MemorySearchHit {
  kind: "fact" | "chunk";
  text: string;
  score: number;
  source_tool: string;
  chunk_id: string;
  source_uri: string;
  fact_id: string | null;
  valid_from: string | null;
  valid_to: string | null;
  superseded_by: string | null;
}

/** Result of `memory_search`. */
export interface MemorySearchResult {
  valid_at: string;
  known_at: string;
  degraded: boolean;
  hits: MemorySearchHit[];
}

/** Result of `memory_provenance`. */
export interface MemoryProvenanceResult {
  found: boolean;
  fact: TimelineEntry | null;
  sources: MemorySource[];
}

/** Error codes the schema defines. */
export type MemoryErrorCode =
  | "invalid_argument"
  | "not_supported"
  | "backend_unavailable";

/** Result of a failed memory tool call. `MemoryToolError`. */
export interface MemoryToolError {
  error: {
    code: MemoryErrorCode;
    message: string;
  };
}
