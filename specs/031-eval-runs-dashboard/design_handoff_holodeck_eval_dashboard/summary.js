// Summary view: KPIs, trend charts, breakdown panels, run table, filter rail.

const fmtPct = (v) => (v * 100).toFixed(1) + '%';
const fmtPct0 = (v) => Math.round(v * 100) + '%';
const fmtDate = (iso) => {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
};
const fmtDateTime = (iso) => {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
};
const fmtDur = (ms) => (ms/1000).toFixed(1) + 's';

function Sparkline({ values, w = 70, h = 28, color = 'var(--hd-accent)' }){
  if (!values.length) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const rng = (max - min) || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / rng) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg className="kpi-spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.9"/>
    </svg>
  );
}

function FilterRail({ filters, setFilters, runs, onReset }){
  const allVersions = [...new Set(runs.map(r => r.metadata.prompt_version.version))];
  const allModels = [...new Set(runs.map(r => r.metadata.agent_config.model.name))];
  const allTags = [...new Set(runs.flatMap(r => r.metadata.prompt_version.tags))];

  const toggle = (key, val) => {
    const cur = filters[key] || [];
    const next = cur.includes(val) ? cur.filter(x => x !== val) : [...cur, val];
    setFilters({ ...filters, [key]: next });
  };

  return (
    <aside className="rail">
      <div className="rail-card">
        <div className="rail-footer" style={{marginBottom:12}}>
          <h4 style={{margin:0}}>Filters</h4>
          <button className="reset-btn" onClick={onReset}>Reset</button>
        </div>

        <div className="rail-group">
          <div className="rail-label">Date range</div>
          <div style={{display:'flex', flexDirection:'column', gap:6}}>
            <div className="date-field">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>
              {filters.from || 'Mar 7, 2026'}
            </div>
            <div className="date-field">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>
              {filters.to || 'Apr 18, 2026'}
            </div>
          </div>
        </div>

        <div className="rail-group">
          <div className="rail-label">Prompt version <span className="value">{(filters.versions||[]).length || 'any'}</span></div>
          <div className="chip-row">
            {allVersions.map(v => (
              <span key={v} className={`chip ${(filters.versions||[]).includes(v) ? 'on' : ''}`} onClick={() => toggle('versions', v)}>
                {v}{(filters.versions||[]).includes(v) ? ' ✕' : ''}
              </span>
            ))}
          </div>
        </div>

        <div className="rail-group">
          <div className="rail-label">Model <span className="value">{(filters.models||[]).length || 'any'}</span></div>
          <div className="chip-row">
            {allModels.map(v => (
              <span key={v} className={`chip ${(filters.models||[]).includes(v) ? 'on' : ''}`} onClick={() => toggle('models', v)}>{v}</span>
            ))}
          </div>
        </div>

        <div className="rail-group">
          <div className="rail-label">Pass-rate threshold <span className="value">≥ {fmtPct0(filters.minPass || 0)}</span></div>
          <div className="slider-track">
            <div className="slider-fill" style={{width: `${(filters.minPass||0)*100}%`}}/>
            <div className="slider-thumb" style={{left: `${(filters.minPass||0)*100}%`}}/>
          </div>
          <div className="mono" style={{display:'flex', justifyContent:'space-between', fontSize:11}}>
            <span>0%</span><span>100%</span>
          </div>
        </div>

        <div className="rail-group">
          <div className="rail-label">Frontmatter tags</div>
          <div className="chip-row">
            {allTags.map(v => (
              <span key={v} className={`chip ${(filters.tags||[]).includes(v) ? 'on' : ''}`} onClick={() => toggle('tags', v)}>#{v}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="rail-card">
        <h4>Share</h4>
        <div className="mono" style={{fontSize:11, color:'var(--hd-muted)', wordBreak:'break-all', lineHeight:1.5}}>
          ?versions=v1.3,v1.4&tags=rag-tuning
        </div>
        <div style={{marginTop:8}}>
          <button className="hd-btn hd-btn-ghost" style={{fontSize:12, padding:'5px 12px'}}>Copy URL</button>
        </div>
      </div>
    </aside>
  );
}

function KpiStrip({ runs }){
  if (!runs.length) return null;
  const latest = runs[runs.length-1];
  const prev = runs[runs.length-2] || latest;
  const passRates = runs.map(r => r.summary.pass_rate);
  const avgDur = runs.reduce((s,r)=>s+r.summary.duration_ms,0) / runs.length;
  const delta = latest.summary.pass_rate - prev.summary.pass_rate;

  // geval avg from latest
  const gevalScores = latest.test_results.flatMap(c => c.metric_results.filter(m=>m.kind==='geval').map(m=>m.score));
  const gevalAvg = gevalScores.reduce((s,v)=>s+v,0) / (gevalScores.length||1);

  const kpis = [
    { label: 'Latest pass rate', value: fmtPct(latest.summary.pass_rate), delta: delta, spark: passRates },
    { label: 'Runs (filtered)', value: runs.length, unit: 'runs', sub: '6 wks' },
    { label: 'Avg G-Eval score', value: gevalAvg.toFixed(2), unit: '/ 1.00', spark: runs.slice(-8).map(r => {
        const s = r.test_results.flatMap(c => c.metric_results.filter(m=>m.kind==='geval').map(m=>m.score));
        return s.reduce((a,b)=>a+b,0)/(s.length||1);
      })
    },
    { label: 'Median duration', value: fmtDur(avgDur), sub: 'per run' },
  ];

  return (
    <div className="kpi-strip">
      {kpis.map((k, i) => (
        <div className="kpi" key={i}>
          <div className="kpi-label">{k.label}</div>
          <div className="kpi-value">{k.value} {k.unit && <span className="kpi-unit">{k.unit}</span>}</div>
          {k.delta !== undefined && (
            <span className={`kpi-delta ${k.delta < 0 ? 'neg' : ''}`}>
              {k.delta >= 0 ? '▲' : '▼'} {(Math.abs(k.delta)*100).toFixed(1)} pp vs prior
            </span>
          )}
          {k.sub && <span className="kpi-delta" style={{background:'rgba(12,20,18,.9)', borderColor:'var(--hd-border)', color:'var(--hd-muted)'}}>{k.sub}</span>}
          {k.spark && <Sparkline values={k.spark} />}
        </div>
      ))}
    </div>
  );
}

function PassRateChart({ runs }){
  const w = 920, h = 230, padL = 44, padR = 16, padT = 16, padB = 28;
  const iw = w - padL - padR, ih = h - padT - padB;
  if (runs.length < 2) return null;

  const xs = runs.map((_, i) => padL + (i / (runs.length - 1)) * iw);
  const yFor = (v) => padT + (1 - v) * ih;

  const areaPts = [`${xs[0]},${padT + ih}`, ...runs.map((r, i) => `${xs[i]},${yFor(r.summary.pass_rate)}`), `${xs[xs.length-1]},${padT+ih}`].join(' ');
  const linePts = runs.map((r, i) => `${xs[i]},${yFor(r.summary.pass_rate)}`).join(' ');

  // detect regressions (drop > 0.05)
  const regressions = [];
  for (let i=1;i<runs.length;i++){
    if (runs[i].summary.pass_rate - runs[i-1].summary.pass_rate < -0.04) regressions.push(i);
  }

  // version boundary markers
  const versionBoundaries = [];
  for (let i=1;i<runs.length;i++){
    if (runs[i].metadata.prompt_version.version !== runs[i-1].metadata.prompt_version.version) {
      versionBoundaries.push({ i, v: runs[i].metadata.prompt_version.version });
    }
  }

  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="accentArea" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgba(123,255,90,.35)"/>
          <stop offset="100%" stopColor="rgba(123,255,90,0)"/>
        </linearGradient>
      </defs>

      {/* gridlines */}
      {[0, 0.25, 0.5, 0.75, 1].map((v, i) => (
        <g key={i}>
          <line className="grid-line" x1={padL} y1={yFor(v)} x2={w-padR} y2={yFor(v)} opacity="0.6"/>
          <text className="axis-label" x={padL-8} y={yFor(v)+3} textAnchor="end">{(v*100).toFixed(0)}%</text>
        </g>
      ))}

      {/* version boundaries */}
      {versionBoundaries.map((b, i) => (
        <g key={i}>
          <line x1={xs[b.i]} y1={padT} x2={xs[b.i]} y2={padT+ih} stroke="rgba(123,255,90,.18)" strokeDasharray="2 3"/>
          <text className="axis-label" x={xs[b.i]+4} y={padT+10} fill="var(--hd-accent-soft)">{b.v}</text>
        </g>
      ))}

      {/* area + line */}
      <polygon points={areaPts} className="area"/>
      <polyline points={linePts} className="main-line"/>

      {/* dots */}
      {runs.map((r, i) => (
        <circle key={i} cx={xs[i]} cy={yFor(r.summary.pass_rate)} r={regressions.includes(i) ? 4.5 : 3}
          className="dot"
          style={regressions.includes(i) ? { fill: '#ff9d7e', stroke: '#050b09' } : {}}/>
      ))}

      {/* x-axis timestamps (every few) */}
      {runs.filter((_,i) => i % Math.ceil(runs.length/8) === 0 || i === runs.length-1).map((r, i) => {
        const idx = runs.indexOf(r);
        return <text key={i} className="axis-label" x={xs[idx]} y={h-10} textAnchor="middle">{fmtDate(r.created_at)}</text>;
      })}
    </svg>
  );
}

function MetricTrendChart({ runs, metricFilter, palette }){
  const w = 920, h = 180, padL = 40, padR = 120, padT = 10, padB = 26;
  const iw = w - padL - padR, ih = h - padT - padB;

  // Collect metric names matching the filter (kind)
  const names = [...new Set(runs.flatMap(r => r.test_results.flatMap(c => c.metric_results.filter(m => m.kind === metricFilter).map(m => m.name))))];

  // per-run average per metric
  const series = names.map((name, idx) => {
    const values = runs.map(r => {
      const ss = r.test_results.flatMap(c => c.metric_results.filter(m => m.kind === metricFilter && m.name === name).map(m => m.score));
      return ss.length ? ss.reduce((a,b)=>a+b,0)/ss.length : null;
    });
    return { name, values, color: palette[idx % palette.length] };
  });

  const xs = runs.map((_, i) => padL + (runs.length === 1 ? iw/2 : (i / (runs.length - 1)) * iw));
  const yFor = (v) => padT + (1 - v) * ih;

  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {[0, 0.5, 1].map((v, i) => (
        <g key={i}>
          <line className="grid-line" x1={padL} y1={yFor(v)} x2={w-padR} y2={yFor(v)} opacity="0.5"/>
          <text className="axis-label" x={padL-6} y={yFor(v)+3} textAnchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      {/* threshold line at 0.7 */}
      <line className="threshold" x1={padL} y1={yFor(0.7)} x2={w-padR} y2={yFor(0.7)}/>
      <text className="axis-label" x={w-padR+4} y={yFor(0.7)+3} fill="rgba(255,120,80,.7)">thresh 0.7</text>

      {series.map((s, si) => {
        const pts = s.values.map((v, i) => v==null ? null : `${xs[i]},${yFor(v)}`).filter(Boolean).join(' ');
        return (
          <g key={si}>
            <polyline points={pts} fill="none" stroke={s.color} strokeWidth="1.8" strokeLinecap="round"/>
            {s.values.map((v, i) => v==null ? null : (
              <circle key={i} cx={xs[i]} cy={yFor(v)} r="2" fill={s.color}/>
            ))}
          </g>
        );
      })}

      {/* legend column */}
      {series.map((s, si) => (
        <g key={si}>
          <line x1={w-padR+10} y1={padT+8+si*16} x2={w-padR+22} y2={padT+8+si*16} stroke={s.color} strokeWidth="2"/>
          <text className="axis-label" x={w-padR+26} y={padT+11+si*16} fill="var(--fg1)">{s.name}</text>
        </g>
      ))}

      {/* x-axis */}
      {runs.filter((_,i) => i % Math.ceil(runs.length/6) === 0 || i === runs.length-1).map((r, i) => {
        const idx = runs.indexOf(r);
        return <text key={i} className="axis-label" x={xs[idx]} y={h-10} textAnchor="middle">{fmtDate(r.created_at)}</text>;
      })}
    </svg>
  );
}

function BreakdownPanel({ title, eyebrow, desc, runs, kind, palette }){
  // Compute aggregate per metric name, across latest 6 runs
  const recent = runs.slice(-6);
  const names = [...new Set(recent.flatMap(r => r.test_results.flatMap(c => c.metric_results.filter(m => m.kind === kind).map(m => m.name))))];
  const rows = names.map(name => {
    const scores = recent.flatMap(r => r.test_results.flatMap(c => c.metric_results.filter(m => m.kind === kind && m.name === name).map(m => m.score)));
    const avg = scores.length ? scores.reduce((a,b)=>a+b,0)/scores.length : 0;
    const passCount = recent.flatMap(r => r.test_results.flatMap(c => c.metric_results.filter(m => m.kind === kind && m.name === name).map(m => m.passed ? 1 : 0))).reduce((a,b)=>a+b,0);
    const total = scores.length;
    return { name, avg, passCount, total };
  });

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <h3>{title}</h3>
          <p>{desc}</p>
        </div>
      </div>
      {rows.length === 0 && <div className="mono" style={{padding:'20px 0', color:'var(--hd-muted)', textAlign:'center'}}>No {kind} metrics in scope</div>}
      {rows.map((r, i) => (
        <div className="metric-row" key={r.name}>
          <span className="metric-name" title={r.name}>{r.name}</span>
          <div className="metric-bar">
            <div className="fill" style={{width: `${r.avg*100}%`, background: `linear-gradient(90deg, ${palette[i%palette.length]}55, ${palette[i%palette.length]})`}}/>
            <div className="thresh" style={{left: '70%'}}/>
          </div>
          <span className={`metric-score ${r.avg < 0.7 ? 'fail' : ''}`}>{r.avg.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function RunTable({ runs, activeId, onSelect, compareQueue, setCompareQueue }){
  const sorted = [...runs].sort((a,b) => b.created_at.localeCompare(a.created_at));
  return (
    <div className="table-card">
      <div className="table-head">
        <div>
          <div className="eyebrow" style={{fontSize:10, letterSpacing:'.15em', textTransform:'uppercase', color:'var(--hd-accent-soft)', marginBottom:4}}>RUNS</div>
          <h3 style={{margin:0, fontSize:15, fontWeight:600}}>All runs in view <span className="mono" style={{color:'var(--hd-muted)', marginLeft:6, fontSize:13}}>{sorted.length}</span></h3>
        </div>
        <div className="gap-tight">
          <div className="table-search">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
            <span>filter runs…</span>
          </div>
          <button className="hd-btn hd-btn-ghost" style={{fontSize:12, padding:'5px 12px'}}>Export CSV</button>
        </div>
      </div>
      <div style={{maxHeight: 360, overflow:'auto'}}>
        <table className="runs">
          <thead>
            <tr>
              <th></th>
              <th>Timestamp</th>
              <th>Pass rate</th>
              <th>Tests</th>
              <th>Prompt</th>
              <th>Model</th>
              <th>Duration</th>
              <th>Commit</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(r => {
              const pass = r.summary.pass_rate;
              const pill = pass >= 0.85 ? 'pill-pass' : pass >= 0.65 ? 'pill-warn' : 'pill-fail';
              return (
                <tr key={r.id} className={r.id === activeId ? 'active' : ''} onClick={() => onSelect(r)}>
                  <td onClick={(e)=>e.stopPropagation()} style={{width:32, paddingRight:0}}>
                    {compareQueue && setCompareQueue && (
                      <CompareAddButton runId={r.id} queue={compareQueue} setQueue={setCompareQueue} size="sm"/>
                    )}
                  </td>
                  <td><span className="ts"><span className="date">{fmtDate(r.created_at)}</span>{new Date(r.created_at).toLocaleTimeString(undefined, { hour:'2-digit', minute:'2-digit'})}</span></td>
                  <td>
                    <div className="bar-inline">
                      <span className={`pill ${pill}`}>{fmtPct(pass)}</span>
                      <div className="bar"><div className="fill" style={{width: `${pass*100}%`}}/></div>
                    </div>
                  </td>
                  <td><span className="mono fg">{r.summary.passed}/{r.summary.total}</span></td>
                  <td><span className="mono" style={{color:'var(--hd-accent)'}}>{r.metadata.prompt_version.version}</span></td>
                  <td><span className="mono fg">{r.metadata.agent_config.model.name}</span></td>
                  <td><span className="mono">{fmtDur(r.summary.duration_ms)}</span></td>
                  <td><span className="mono" style={{color:'var(--hd-muted)'}}>{r.git_commit}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SummaryView({ runs, filters, setFilters, showRail, onOpenRun, compareQueue, setCompareQueue }){
  const palette = ['#7bff5a','#5ae0a6','#53ff9c','#9bff5f','#a7f0ba','#ffcf5a'];
  const gevalPalette = ['#7bff5a','#ffcf5a','#5ae0a6'];
  const stdPalette = ['#7bff5a','#5ae0a6','#9bff5f'];

  if (!runs.length){
    return (
      <div className="empty">
        <div className="icon">∅</div>
        <h3 style={{fontSize:16, margin:'0 0 6px'}}>No runs match your filters</h3>
        <p className="mono" style={{color:'var(--hd-muted)'}}>Try clearing filters, or execute <code>holodeck test agent.yaml</code> to generate one.</p>
      </div>
    );
  }

  return (
    <div className={`layout ${showRail ? '' : 'no-rail'}`}>
      {showRail && <FilterRail runs={runs} filters={filters} setFilters={setFilters} onReset={() => setFilters({ versions: [], models: [], tags: [], minPass: 0 })} />}
      <div className="main">
        <KpiStrip runs={runs}/>

        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">TRENDS</div>
              <h3>Pass rate over time</h3>
              <p>24 runs of <span className="mono" style={{color:'var(--hd-accent)'}}>customer-support</span> · regressions flagged in coral · dashed lines mark prompt-version boundaries</p>
            </div>
            <div className="legend">
              <span className="legend-item"><span className="legend-swatch line" style={{background:'var(--hd-accent)', width:16}}/> pass rate</span>
              <span className="legend-item"><span className="legend-swatch" style={{background:'#ff9d7e'}}/> regression</span>
            </div>
          </div>
          <PassRateChart runs={runs}/>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">METRIC TRENDS</div>
              <h3>Per-metric average scores</h3>
              <p>Mean metric score per run, grouped by kind. Threshold line at <span className="mono" style={{color:'#ff9d7e'}}>0.7</span>.</p>
            </div>
            <div className="seg">
              <button className={filters.metricKind === 'rag' || !filters.metricKind ? 'on' : ''} onClick={() => setFilters({...filters, metricKind:'rag'})}>rag</button>
              <button className={filters.metricKind === 'geval' ? 'on' : ''} onClick={() => setFilters({...filters, metricKind:'geval'})}>geval</button>
              <button className={filters.metricKind === 'standard' ? 'on' : ''} onClick={() => setFilters({...filters, metricKind:'standard'})}>standard</button>
            </div>
          </div>
          <MetricTrendChart runs={runs} metricFilter={filters.metricKind || 'rag'} palette={palette}/>
        </div>

        <div className="breakdowns">
          <BreakdownPanel
            eyebrow="BREAKDOWN · STANDARD"
            title="NLP metrics"
            desc="BLEU / ROUGE / METEOR — last 6 runs, avg across test cases."
            runs={runs} kind="standard" palette={stdPalette}/>
          <BreakdownPanel
            eyebrow="BREAKDOWN · RAG"
            title="Retrieval & grounding"
            desc="Faithfulness, relevancy, precision, recall — averaged across recent runs."
            runs={runs} kind="rag" palette={palette}/>
          <BreakdownPanel
            eyebrow="BREAKDOWN · G-EVAL"
            title="Custom LLM judges"
            desc="Per-name custom G-Eval rubrics defined in agent.yaml."
            runs={runs} kind="geval" palette={gevalPalette}/>
        </div>

        <RunTable runs={runs} activeId={null} onSelect={onOpenRun} compareQueue={compareQueue} setCompareQueue={setCompareQueue}/>
      </div>
    </div>
  );
}

Object.assign(window, { SummaryView });
