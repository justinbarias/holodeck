// Compare view — 3 variants selectable via segmented control.
// All variants share: summary stats, config diff strip, per-case matrix.
// Baseline = first-selected run; others show deltas vs. baseline.

const COMPARE_PALETTE = ['#7bff5a', '#5ae0a6', '#ffcf5a']; // baseline, r1, r2

// --- Helpers ---
const fmtSigned = (v, suffix = '', digits = 2) => {
  if (v === 0 || v === null || v === undefined || Number.isNaN(v)) return '·';
  const s = v > 0 ? '+' : '';
  return s + v.toFixed(digits) + suffix;
};
const fmtPP = (v) => {
  if (v === 0 || v === null || v === undefined) return '·';
  return (v > 0 ? '+' : '') + (v * 100).toFixed(1) + 'pp';
};
const deltaClass = (v, invert = false) => {
  if (v === 0 || v == null) return 'delta-neutral';
  const positive = invert ? v < 0 : v > 0;
  return positive ? 'delta-pos' : 'delta-neg';
};

// Compute aggregate per-run stats from test_results
function runStats(run){
  const cases = run.test_results;
  const gevalScores = cases.flatMap(c => c.metric_results.filter(m => m.kind === 'geval').map(m => m.score));
  const gevalAvg = gevalScores.length ? gevalScores.reduce((s,v)=>s+v,0) / gevalScores.length : 0;
  const ragScores = cases.flatMap(c => c.metric_results.filter(m => m.kind === 'rag').map(m => m.score));
  const ragAvg = ragScores.length ? ragScores.reduce((s,v)=>s+v,0) / ragScores.length : 0;
  // Synthetic cost: duration_ms * rate-by-model
  const ratePerSec = run.metadata.agent_config.model.name.includes('sonnet') ? 0.018 : 0.012;
  const cost = (run.summary.duration_ms / 1000) * ratePerSec;
  return {
    passRate: run.summary.pass_rate,
    passed: run.summary.passed,
    total: run.summary.total,
    duration: run.summary.duration_ms,
    gevalAvg,
    ragAvg,
    cost,
  };
}

// --- Small components ---

function DeltaPill({ value, type = 'raw', invert = false, digits = 2, suffix = '' }){
  if (value == null || value === 0 || Number.isNaN(value)) {
    return <span className="delta delta-neutral">·</span>;
  }
  const cls = deltaClass(value, invert);
  let text = '';
  if (type === 'pp') text = (value > 0 ? '+' : '') + (value * 100).toFixed(1) + 'pp';
  else if (type === 'pct') text = (value > 0 ? '+' : '') + (value * 100).toFixed(1) + '%';
  else text = (value > 0 ? '+' : '') + value.toFixed(digits) + suffix;
  return <span className={`delta ${cls}`}>{text}</span>;
}

function RunSlotHeader({ run, isBaseline, slotIndex, onRemove }){
  const color = COMPARE_PALETTE[slotIndex];
  const cfg = run.metadata.agent_config;
  const pv = run.metadata.prompt_version;
  return (
    <div className="cmp-col-head">
      <div className="cmp-col-head-top">
        <span className="cmp-dot" style={{background: color, boxShadow: `0 0 10px ${color}90`}}/>
        <span className="cmp-slot-label">{isBaseline ? 'BASELINE' : `RUN ${slotIndex + 1}`}</span>
        <button className="cmp-x" onClick={onRemove} title="Remove from comparison">×</button>
      </div>
      <div className="cmp-col-head-ts mono">{fmtDateTime(run.created_at)}</div>
      <div className="cmp-col-head-meta">
        <span className="cmp-ver" style={{color}}>{pv.version}</span>
        <span className="mono">·</span>
        <span className="mono">{cfg.model.name}</span>
      </div>
      <div className="cmp-col-head-commit mono">{run.git_commit}</div>
    </div>
  );
}

// --- Variant 1: Side-by-side columns ---

function CompareV1({ runs }){
  const [baseline, ...rest] = runs;
  const bs = runStats(baseline);
  const statRows = [
    { key: 'pass', label: 'Pass rate', get: (r) => runStats(r).passRate, format: (v) => fmtPct(v), delta: 'pp' },
    { key: 'passed', label: 'Passed', get: (r) => {const s = runStats(r); return s.passed / s.total;}, format: (v, r) => {const s = runStats(r); return `${s.passed}/${s.total}`;}, delta: 'pp' },
    { key: 'geval', label: 'Avg G-Eval', get: (r) => runStats(r).gevalAvg, format: (v) => v.toFixed(2), delta: 'raw' },
    { key: 'rag', label: 'Avg RAG score', get: (r) => runStats(r).ragAvg, format: (v) => v.toFixed(2), delta: 'raw' },
    { key: 'dur', label: 'Duration', get: (r) => runStats(r).duration, format: (v) => fmtDur(v), delta: 'raw', invert: true, deltaDigits: 0, deltaSuffix: 'ms' },
    { key: 'cost', label: 'Est. cost', get: (r) => runStats(r).cost, format: (v) => '$' + v.toFixed(3), delta: 'raw', invert: true, digits: 3, deltaSuffix: '' },
  ];

  const configRows = [
    { label: 'Prompt version', get: (r) => r.metadata.prompt_version.version, mono: true, accent: true },
    { label: 'Model', get: (r) => r.metadata.agent_config.model.name, mono: true },
    { label: 'Temperature', get: (r) => r.metadata.agent_config.model.temperature, mono: true },
    { label: 'Prompt tags', get: (r) => r.metadata.prompt_version.tags.map(t => '#' + t).join(' '), mono: true },
    { label: 'Commit', get: (r) => r.git_commit, mono: true, muted: true },
    { label: 'Extended thinking', get: (r) => String(r.metadata.agent_config.claude.extended_thinking), mono: true },
  ];

  return (
    <div className="cmp-v1">
      {/* Top: column headers */}
      <div className="cmp-cols" style={{gridTemplateColumns: `140px repeat(${runs.length}, minmax(0,1fr))`}}>
        <div/>
        {runs.map((r, i) => (
          <RunSlotHeader key={r.id} run={r} isBaseline={i === 0} slotIndex={i} onRemove={() => window.__cmpRemove && window.__cmpRemove(r.id)}/>
        ))}
      </div>

      {/* Summary stats block */}
      <div className="cmp-block">
        <div className="cmp-block-head">
          <div className="eyebrow">SUMMARY</div>
          <h3>Headline stats</h3>
          <p>Deltas shown against the <span style={{color: COMPARE_PALETTE[0]}}>baseline</span>. Lower-is-better fields (duration, cost) invert delta polarity.</p>
        </div>
        <div className="cmp-rows" style={{gridTemplateColumns: `140px repeat(${runs.length}, minmax(0,1fr))`}}>
          {statRows.map(row => (
            <React.Fragment key={row.key}>
              <div className="cmp-row-label">{row.label}</div>
              {runs.map((r, i) => {
                const v = row.get(r);
                const bV = row.get(baseline);
                const d = i === 0 ? 0 : v - bV;
                return (
                  <div className="cmp-cell" key={r.id}>
                    <div className={`cmp-cell-val mono ${row.key === 'pass' ? 'accent-big' : ''}`}>{row.format(v, r)}</div>
                    {i > 0 && (
                      row.delta === 'pp'
                        ? <DeltaPill value={d} type="pp"/>
                        : <DeltaPill value={d} type="raw" invert={row.invert} digits={row.deltaDigits ?? 2} suffix={row.deltaSuffix ?? ''}/>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Config diff */}
      <div className="cmp-block">
        <div className="cmp-block-head">
          <div className="eyebrow">CONFIG DIFF</div>
          <h3>What's different?</h3>
          <p>Rows where values differ across runs are highlighted.</p>
        </div>
        <div className="cmp-rows" style={{gridTemplateColumns: `140px repeat(${runs.length}, minmax(0,1fr))`}}>
          {configRows.map((row, ri) => {
            const vals = runs.map(row.get);
            const allSame = vals.every(v => v === vals[0]);
            return (
              <React.Fragment key={ri}>
                <div className="cmp-row-label">{row.label}</div>
                {runs.map((r, i) => {
                  const v = row.get(r);
                  const different = !allSame && v !== row.get(baseline) && i > 0;
                  return (
                    <div className={`cmp-cell cmp-cfg-cell ${different ? 'different' : ''} ${allSame ? 'same' : ''}`} key={r.id}>
                      <span className={row.mono ? 'mono fg' : ''} style={row.accent ? {color: COMPARE_PALETTE[i]} : (row.muted ? {color: 'var(--hd-muted)'} : {})}>{v}</span>
                      {different && <span className="cmp-diff-badge">changed</span>}
                    </div>
                  );
                })}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Case matrix */}
      <CaseMatrix runs={runs}/>
    </div>
  );
}

// --- Case matrix (style 0: heatmap cells) — shared across variants ---

function CaseMatrix({ runs }){
  // Union of all case names across selected runs
  const caseNames = [...new Set(runs.flatMap(r => r.test_results.map(c => c.name)))];

  const getCase = (run, name) => run.test_results.find(c => c.name === name);

  // Compute per-case geval score per run (fallback to rag avg if no geval)
  const caseScore = (caseObj) => {
    if (!caseObj) return null;
    const g = caseObj.metric_results.find(m => m.kind === 'geval');
    if (g) return g.score;
    const rag = caseObj.metric_results.filter(m => m.kind === 'rag');
    if (rag.length) return rag.reduce((s,m)=>s+m.score,0) / rag.length;
    return caseObj.passed ? 1 : 0;
  };

  const baseline = runs[0];

  return (
    <div className="cmp-block">
      <div className="cmp-block-head">
        <div className="eyebrow">PER-CASE MATRIX</div>
        <h3>Test-case pass/fail across runs</h3>
        <p>Heatmap of per-case scores. Cells show pass/fail plus the primary metric (geval, else rag avg). Regressions — passing in baseline but failing elsewhere — are outlined.</p>
      </div>
      <div className="cmp-matrix" style={{gridTemplateColumns: `minmax(0, 1fr) repeat(${runs.length}, 88px)`}}>
        <div className="cmp-mh cmp-mh-left">case</div>
        {runs.map((r, i) => (
          <div key={r.id} className="cmp-mh">
            <span className="cmp-mh-dot" style={{background: COMPARE_PALETTE[i]}}/>
            {i === 0 ? 'base' : `r${i}`}
          </div>
        ))}
        {caseNames.map(name => {
          const baseCase = getCase(baseline, name);
          const basePassed = baseCase?.passed;
          return (
            <React.Fragment key={name}>
              <div className="cmp-mc-label" title={name}>{name}</div>
              {runs.map((r, i) => {
                const c = getCase(r, name);
                if (!c) return <div className="cmp-mc cmp-mc-missing" key={r.id}>—</div>;
                const score = caseScore(c);
                const regression = i > 0 && basePassed && !c.passed;
                const improvement = i > 0 && !basePassed && c.passed;
                const cls = c.passed ? 'pass' : 'fail';
                return (
                  <div className={`cmp-mc ${cls} ${regression ? 'regression' : ''} ${improvement ? 'improvement' : ''}`} key={r.id}
                       style={{'--heat': score}}>
                    <span className="cmp-mc-state">{c.passed ? '✓' : '✕'}</span>
                    <span className="cmp-mc-score mono">{score != null ? score.toFixed(2) : '—'}</span>
                  </div>
                );
              })}
            </React.Fragment>
          );
        })}
      </div>

      {/* Legend */}
      <div className="cmp-legend">
        <span className="legend-item"><span className="legend-swatch" style={{background:'rgba(123,255,90,.5)'}}/>pass</span>
        <span className="legend-item"><span className="legend-swatch" style={{background:'rgba(255,120,80,.5)'}}/>fail</span>
        <span className="legend-item"><span className="legend-swatch" style={{background:'transparent', border:'1px dashed #ff9d7e'}}/>regression vs. baseline</span>
        <span className="legend-item"><span className="legend-swatch" style={{background:'transparent', border:'1px dashed #7bff5a'}}/>improvement vs. baseline</span>
      </div>
    </div>
  );
}

// --- Variant 2: Baseline-emphasized ---

function CompareV2({ runs }){
  const [baseline, ...rest] = runs;
  const bs = runStats(baseline);
  const bcfg = baseline.metadata.agent_config;
  const bpv = baseline.metadata.prompt_version;

  return (
    <div className="cmp-v2">
      <div className="cmp-v2-grid" style={{gridTemplateColumns: `minmax(0, 1.4fr) repeat(${rest.length}, minmax(0, 1fr))`}}>
        {/* Baseline card */}
        <div className="cmp-v2-baseline">
          <div className="cmp-v2-label">
            <span className="cmp-dot" style={{background: COMPARE_PALETTE[0]}}/>
            <span className="cmp-slot-label">BASELINE</span>
            <button className="cmp-x" onClick={() => window.__cmpRemove && window.__cmpRemove(baseline.id)}>×</button>
          </div>
          <div className="cmp-v2-title">
            <span style={{color: COMPARE_PALETTE[0]}}>{bpv.version}</span>
            <span className="mono" style={{color:'var(--hd-muted)'}}>{fmtDateTime(baseline.created_at)}</span>
          </div>
          <div className="cmp-v2-model mono">{bcfg.model.name} · T {bcfg.model.temperature} · {baseline.git_commit}</div>
          <div className="cmp-v2-big">
            <div className="cmp-v2-pass">
              <span className="num mono">{fmtPct(bs.passRate)}</span>
              <span className="cmp-sub mono">pass rate · {bs.passed}/{bs.total}</span>
            </div>
            <div className="cmp-v2-stats">
              <div><span className="cmp-sub mono">geval</span><span className="mono fg">{bs.gevalAvg.toFixed(2)}</span></div>
              <div><span className="cmp-sub mono">rag</span><span className="mono fg">{bs.ragAvg.toFixed(2)}</span></div>
              <div><span className="cmp-sub mono">dur</span><span className="mono fg">{fmtDur(bs.duration)}</span></div>
              <div><span className="cmp-sub mono">cost</span><span className="mono fg">${bs.cost.toFixed(3)}</span></div>
            </div>
          </div>
          <div className="cmp-v2-tags chip-row">
            {bpv.tags.map(t => <span key={t} className="chip" style={{cursor:'default'}}>#{t}</span>)}
          </div>
        </div>

        {/* Delta cards */}
        {rest.map((r, i) => {
          const s = runStats(r);
          const cfg = r.metadata.agent_config;
          const pv = r.metadata.prompt_version;
          return (
            <div className="cmp-v2-delta" key={r.id}>
              <div className="cmp-v2-label">
                <span className="cmp-dot" style={{background: COMPARE_PALETTE[i+1]}}/>
                <span className="cmp-slot-label">VS RUN {i+1}</span>
                <button className="cmp-x" onClick={() => window.__cmpRemove && window.__cmpRemove(r.id)}>×</button>
              </div>
              <div className="cmp-v2-title">
                <span style={{color: COMPARE_PALETTE[i+1]}}>{pv.version}</span>
                <span className="mono" style={{color:'var(--hd-muted)'}}>{fmtDateTime(r.created_at)}</span>
              </div>
              <div className="cmp-v2-model mono">{cfg.model.name}{cfg.model.name !== bcfg.model.name && <span className="cmp-diff-dot">changed</span>}</div>

              <div className="cmp-delta-row">
                <span className="cmp-sub mono">pass rate</span>
                <span className="mono fg" style={{fontSize: 15}}>{fmtPct(s.passRate)}</span>
                <DeltaPill value={s.passRate - bs.passRate} type="pp"/>
              </div>
              <div className="cmp-delta-row">
                <span className="cmp-sub mono">geval</span>
                <span className="mono fg">{s.gevalAvg.toFixed(2)}</span>
                <DeltaPill value={s.gevalAvg - bs.gevalAvg}/>
              </div>
              <div className="cmp-delta-row">
                <span className="cmp-sub mono">rag</span>
                <span className="mono fg">{s.ragAvg.toFixed(2)}</span>
                <DeltaPill value={s.ragAvg - bs.ragAvg}/>
              </div>
              <div className="cmp-delta-row">
                <span className="cmp-sub mono">duration</span>
                <span className="mono fg">{fmtDur(s.duration)}</span>
                <DeltaPill value={s.duration - bs.duration} invert={true} digits={0} suffix="ms"/>
              </div>
              <div className="cmp-delta-row">
                <span className="cmp-sub mono">cost</span>
                <span className="mono fg">${s.cost.toFixed(3)}</span>
                <DeltaPill value={s.cost - bs.cost} invert={true} digits={3}/>
              </div>
            </div>
          );
        })}
      </div>

      <CaseMatrix runs={runs}/>
    </div>
  );
}

// --- Variant 3: Matrix-dominant ---

function CompareV3({ runs }){
  const baseline = runs[0];
  const bs = runStats(baseline);

  // Regression & improvement counts
  const caseNames = [...new Set(runs.flatMap(r => r.test_results.map(c => c.name)))];
  const callouts = runs.slice(1).map(r => {
    const regressions = [];
    const improvements = [];
    caseNames.forEach(name => {
      const bc = baseline.test_results.find(c => c.name === name);
      const rc = r.test_results.find(c => c.name === name);
      if (!bc || !rc) return;
      if (bc.passed && !rc.passed) regressions.push(name);
      if (!bc.passed && rc.passed) improvements.push(name);
    });
    return { run: r, regressions, improvements };
  });

  return (
    <div className="cmp-v3">
      {/* Compact strip */}
      <div className="cmp-v3-strip">
        {runs.map((r, i) => {
          const s = runStats(r);
          const isBase = i === 0;
          const diff = isBase ? null : s.passRate - bs.passRate;
          return (
            <div key={r.id} className={`cmp-v3-card ${isBase ? 'baseline' : ''}`}>
              <div className="cmp-v3-card-head">
                <span className="cmp-dot" style={{background: COMPARE_PALETTE[i]}}/>
                <span className="cmp-slot-label">{isBase ? 'BASELINE' : `RUN ${i}`}</span>
                <button className="cmp-x" onClick={() => window.__cmpRemove && window.__cmpRemove(r.id)}>×</button>
              </div>
              <div className="cmp-v3-ver" style={{color: COMPARE_PALETTE[i]}}>{r.metadata.prompt_version.version}</div>
              <div className="cmp-v3-model mono">{r.metadata.agent_config.model.name}</div>
              <div className="cmp-v3-pass">
                <span className="num mono">{fmtPct(s.passRate)}</span>
                {!isBase && <DeltaPill value={diff} type="pp"/>}
              </div>
              <div className="cmp-v3-mini">
                <span className="mono">geval <b>{s.gevalAvg.toFixed(2)}</b></span>
                <span className="mono">rag <b>{s.ragAvg.toFixed(2)}</b></span>
                <span className="mono">{fmtDur(s.duration)}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Callouts */}
      {callouts.some(c => c.regressions.length || c.improvements.length) && (
        <div className="cmp-v3-callouts">
          {callouts.map((c, i) => (
            <div key={c.run.id} className="cmp-v3-callout">
              <div className="cmp-v3-callout-head">
                <span className="cmp-dot" style={{background: COMPARE_PALETTE[i+1]}}/>
                <span className="mono">vs. baseline · {c.run.metadata.prompt_version.version}</span>
              </div>
              <div className="cmp-v3-callout-body">
                {c.regressions.length > 0 && (
                  <div className="cmp-v3-callout-row">
                    <span className="pill pill-fail">{c.regressions.length} regression{c.regressions.length === 1 ? '' : 's'}</span>
                    <span className="mono" style={{color:'var(--hd-muted)', marginLeft: 8}}>{c.regressions.slice(0,3).join(', ')}{c.regressions.length > 3 ? ` +${c.regressions.length - 3}` : ''}</span>
                  </div>
                )}
                {c.improvements.length > 0 && (
                  <div className="cmp-v3-callout-row">
                    <span className="pill pill-pass">{c.improvements.length} improvement{c.improvements.length === 1 ? '' : 's'}</span>
                    <span className="mono" style={{color:'var(--hd-muted)', marginLeft: 8}}>{c.improvements.slice(0,3).join(', ')}{c.improvements.length > 3 ? ` +${c.improvements.length - 3}` : ''}</span>
                  </div>
                )}
                {!c.regressions.length && !c.improvements.length && (
                  <span className="mono" style={{color:'var(--hd-muted)'}}>No case-level changes from baseline.</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <CaseMatrix runs={runs}/>
    </div>
  );
}

// --- Empty state ---

function CompareEmpty({ allRuns, onPickLatest2, onPickLatest3 }){
  return (
    <div className="cmp-empty">
      <div className="cmp-empty-icon">
        <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
          <rect x="4" y="10" width="18" height="44" rx="2" stroke="#7bff5a" strokeWidth="1.2" opacity=".7"/>
          <rect x="24" y="14" width="18" height="40" rx="2" stroke="#5ae0a6" strokeWidth="1.2" opacity=".55"/>
          <rect x="44" y="18" width="18" height="36" rx="2" stroke="#ffcf5a" strokeWidth="1.2" opacity=".4"/>
          <line x1="8" y1="42" x2="18" y2="42" stroke="#7bff5a" strokeWidth="1"/>
          <line x1="28" y1="36" x2="38" y2="36" stroke="#5ae0a6" strokeWidth="1"/>
          <line x1="48" y1="40" x2="58" y2="40" stroke="#ffcf5a" strokeWidth="1"/>
        </svg>
      </div>
      <h2>Pick runs to compare</h2>
      <p>Select up to 3 runs from the Explorer's Runs pane, the Summary table, or quick-pick below. The first-selected run becomes your <span style={{color:'var(--hd-accent)'}}>baseline</span>; others show deltas against it.</p>
      <div className="cmp-empty-actions">
        <button className="hd-btn hd-btn-primary" onClick={onPickLatest2}>Compare latest 2 runs</button>
        <button className="hd-btn hd-btn-ghost" onClick={onPickLatest3}>Compare latest 3 runs</button>
      </div>
      <div className="cmp-empty-hint mono">Tip · click the <span className="cmp-empty-kbd">+</span> icon on any run to add it to the compare queue.</div>
    </div>
  );
}

// --- Root ---

function CompareView({ allRuns, queue, setQueue, variant, setVariant }){
  const runs = queue.map(id => allRuns.find(r => r.id === id)).filter(Boolean);

  // Expose remove-from-queue to inner components
  React.useEffect(() => {
    window.__cmpRemove = (id) => setQueue(queue.filter(q => q !== id));
    return () => { delete window.__cmpRemove; };
  }, [queue, setQueue]);

  const byDate = [...allRuns].sort((a,b) => b.created_at.localeCompare(a.created_at));
  const pickLatest = (n) => setQueue(byDate.slice(0, n).map(r => r.id));

  if (runs.length < 2) {
    return (
      <div className="cmp-wrap">
        <CompareEmpty allRuns={allRuns}
          onPickLatest2={() => pickLatest(2)}
          onPickLatest3={() => pickLatest(3)}/>
      </div>
    );
  }

  return (
    <div className="cmp-wrap">
      <div className="cmp-toolbar">
        <div className="cmp-toolbar-left">
          <div className="eyebrow" style={{color:'var(--hd-accent-soft)'}}>COMPARE</div>
          <h2 style={{margin:'2px 0 0', fontSize: 19, fontWeight: 600, letterSpacing:'-.01em'}}>
            {runs.length} runs · baseline <span style={{color: COMPARE_PALETTE[0]}}>{runs[0].metadata.prompt_version.version}</span>
          </h2>
        </div>
        <div className="cmp-toolbar-right">
          <span className="mono" style={{color:'var(--hd-muted)', fontSize:12}}>layout</span>
          <div className="seg">
            <button className={variant === 1 ? 'on' : ''} onClick={() => setVariant(1)}>side-by-side</button>
            <button className={variant === 2 ? 'on' : ''} onClick={() => setVariant(2)}>baseline + deltas</button>
            <button className={variant === 3 ? 'on' : ''} onClick={() => setVariant(3)}>matrix-first</button>
          </div>
          <button className="reset-btn" onClick={() => setQueue([])}>Clear</button>
        </div>
      </div>

      {variant === 1 && <CompareV1 runs={runs}/>}
      {variant === 2 && <CompareV2 runs={runs}/>}
      {variant === 3 && <CompareV3 runs={runs}/>}
    </div>
  );
}

// --- Floating compare tray (shown across tabs) ---

function CompareTray({ allRuns, queue, setQueue, onOpenCompare }){
  if (!queue.length) return null;
  const items = queue.map(id => allRuns.find(r => r.id === id)).filter(Boolean);
  return (
    <div className="cmp-tray">
      <div className="cmp-tray-label">
        <span className="eyebrow" style={{color:'var(--hd-accent-soft)'}}>COMPARE QUEUE</span>
        <span className="mono" style={{color: 'var(--hd-muted)'}}>{items.length}/3</span>
      </div>
      <div className="cmp-tray-items">
        {items.map((r, i) => (
          <div key={r.id} className="cmp-tray-item">
            <span className="cmp-dot" style={{background: COMPARE_PALETTE[i]}}/>
            {i === 0 && <span className="cmp-tray-base">base</span>}
            <span className="mono fg">{r.metadata.prompt_version.version}</span>
            <span className="mono" style={{color:'var(--hd-muted)'}}>{fmtDate(r.created_at)}</span>
            <button className="cmp-x" onClick={() => setQueue(queue.filter(q => q !== r.id))}>×</button>
          </div>
        ))}
        {Array.from({length: 3 - items.length}).map((_, i) => (
          <div key={'empty-'+i} className="cmp-tray-item cmp-tray-empty">
            <span className="mono">slot {items.length + i + 1}</span>
          </div>
        ))}
      </div>
      <div className="cmp-tray-actions">
        <button className="reset-btn" onClick={() => setQueue([])}>Clear</button>
        <button
          className={`hd-btn ${items.length >= 2 ? 'hd-btn-primary' : 'hd-btn-ghost'}`}
          style={{fontSize:12, padding:'6px 14px', opacity: items.length >= 2 ? 1 : 0.55, cursor: items.length >= 2 ? 'pointer' : 'not-allowed'}}
          disabled={items.length < 2}
          onClick={onOpenCompare}>
          Open Compare →
        </button>
      </div>
    </div>
  );
}

// --- Compare-button (+ icon) for run rows ---

function CompareAddButton({ runId, queue, setQueue, size = 'sm' }){
  const inQueue = queue.includes(runId);
  const idx = queue.indexOf(runId);
  const full = queue.length >= 3;
  const handle = (e) => {
    e.stopPropagation();
    if (inQueue) setQueue(queue.filter(q => q !== runId));
    else if (!full) setQueue([...queue, runId]);
  };
  const title = inQueue ? `In compare queue (slot ${idx+1})` : full ? 'Compare queue is full (3 max)' : 'Add to compare queue';
  return (
    <button
      className={`cmp-add ${inQueue ? 'on' : ''} ${full && !inQueue ? 'disabled' : ''} cmp-add-${size}`}
      onClick={handle}
      title={title}
    >
      {inQueue
        ? <span className="cmp-add-num">{idx+1}</span>
        : <span className="cmp-add-plus">+</span>}
    </button>
  );
}

Object.assign(window, { CompareView, CompareTray, CompareAddButton });
