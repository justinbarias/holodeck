// Explorer view: run list, case list, case detail panel.

function Collapsible({ eyebrow, title, subtitle, defaultOpen = true, right, children }){
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className="panel">
      <div className="panel-head" style={{cursor:'pointer', marginBottom: open ? 14 : 0}} onClick={() => setOpen(!open)}>
        <div style={{display:'flex', alignItems:'center', gap:14, flex:1, minWidth:0}}>
          <span className="caret" style={{display:'inline-flex', width:32, height:32, alignItems:'center', justifyContent:'center', color:'var(--hd-accent)', transition:'transform 140ms var(--ease-standard)', transform: open ? 'rotate(90deg)' : 'rotate(0deg)', fontSize:30, lineHeight:1, flexShrink:0}}>▸</span>
          <div style={{flex:1, minWidth:0}}>
            {eyebrow && <div className="eyebrow">{eyebrow}</div>}
            <h3>{title}</h3>
            {subtitle && <p>{subtitle}</p>}
          </div>
        </div>
        {right && <div onClick={(e) => e.stopPropagation()}>{right}</div>}
      </div>
      {open && <div>{children}</div>}
    </div>
  );
}

function RunList({ runs, activeId, onSelect, collapsed, onToggle, compareQueue, setCompareQueue }){
  const sorted = [...runs].sort((a,b) => b.created_at.localeCompare(a.created_at));
  if (collapsed) {
    return (
      <div
        onClick={onToggle}
        title="Expand runs"
        style={{
          alignSelf:'stretch',
          display:'flex',
          flexDirection:'column',
          alignItems:'center',
          justifyContent:'flex-start',
          paddingTop:16,
          gap:16,
          cursor:'pointer',
          background:'var(--hd-card)',
          border:'1px solid var(--hd-border)',
          borderRadius:'var(--radius-md)',
          width:48,
          transition:'background 140ms var(--ease-standard), border-color 140ms var(--ease-standard)'
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--hd-accent-glow)'; e.currentTarget.style.background = 'rgba(12,20,18,.9)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--hd-border)'; e.currentTarget.style.background = 'var(--hd-card)'; }}
      >
        <span style={{color:'var(--hd-accent)', fontSize:22, lineHeight:1}}>▸</span>
        <div style={{
          writingMode:'vertical-rl',
          transform:'rotate(180deg)',
          fontSize:11,
          fontFamily:'var(--font-mono)',
          letterSpacing:'.2em',
          textTransform:'uppercase',
          color:'var(--hd-muted)',
          display:'flex',
          alignItems:'center',
          gap:10
        }}>
          <span>Runs</span>
          <span style={{color:'var(--hd-accent)'}}>{sorted.length}</span>
        </div>
      </div>
    );
  }
  return (
    <div className="list-card">
      <div className="list-head" style={{cursor:'pointer'}} onClick={onToggle}>
        <h4 style={{display:'flex', alignItems:'center', gap:12}}><span style={{color:'var(--hd-accent)', transform:'rotate(90deg)', display:'inline-block', fontSize:26, lineHeight:1}}>▸</span>Runs <span className="mono" style={{color:'var(--hd-muted)', fontWeight:400}}>{sorted.length}</span></h4>
        <span className="mono" style={{fontSize:11, color:'var(--hd-muted)'}}>newest first</span>
      </div>
      <div className="list-scroll">
        {sorted.map(r => (
          <div key={r.id} className={`run-item ${r.id === activeId ? 'active' : ''}`} onClick={() => onSelect(r)}>
            <div className="r1">
              <span className="mono fg" style={{fontSize:12}}>{fmtDateTime(r.created_at)}</span>
              <span className="ver">{r.metadata.prompt_version.version}</span>
              {compareQueue && setCompareQueue && (
                <CompareAddButton runId={r.id} queue={compareQueue} setQueue={setCompareQueue} size="sm"/>
              )}
            </div>
            <div className="r2">
              <span className={`pill ${r.summary.pass_rate >= 0.85 ? 'pill-pass' : r.summary.pass_rate >= 0.65 ? 'pill-warn' : 'pill-fail'}`}>
                {fmtPct(r.summary.pass_rate)}
              </span>
              <span>{r.summary.passed}/{r.summary.total}</span>
              <span style={{marginLeft:'auto'}}>{r.metadata.agent_config.model.name.split('-').slice(-2).join('-')}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CaseList({ cases, activeId, onSelect }){
  return (
    <div className="list-card">
      <div className="list-head">
        <h4>Test cases <span className="mono" style={{color:'var(--hd-muted)', marginLeft:6, fontWeight:400}}>{cases.length}</span></h4>
        <span className="mono" style={{fontSize:11, color:'var(--hd-muted)'}}>
          {cases.filter(c=>c.passed).length} pass
        </span>
      </div>
      <div className="list-scroll">
        {cases.map(c => {
          const gev = c.metric_results.find(m => m.kind === 'geval');
          const ragAvg = (() => {
            const r = c.metric_results.filter(m => m.kind === 'rag');
            return r.length ? (r.reduce((s,x)=>s+x.score,0)/r.length).toFixed(2) : null;
          })();
          return (
            <div key={c.name} className={`case-item ${c.name === activeId ? 'active' : ''}`} onClick={() => onSelect(c)}>
              <div className="c1">
                <span className="nm" title={c.name}>{c.name}</span>
                <span className={`pill ${c.passed ? 'pill-pass' : 'pill-fail'}`}>{c.passed ? 'PASS' : 'FAIL'}</span>
              </div>
              <div className="c2">
                {gev && <span className={`mini-metric ${gev.passed ? 'pass' : 'fail'}`}>geval {gev.score.toFixed(2)}</span>}
                {ragAvg && <span className="mini-metric">rag {ragAvg}</span>}
                {c.tools_called.length > 0 && <span className="mini-metric">{c.tools_called.length} tools</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function JsonPretty({ obj, highlight = true }){
  const s = JSON.stringify(obj, null, 2);
  if (!highlight) return <span>{s}</span>;
  // basic syntax highlight
  const parts = [];
  const re = /("[^"\\]*(?:\\.[^"\\]*)*")(\s*:)|("[^"\\]*(?:\\.[^"\\]*)*")|(-?\d+(?:\.\d+)?)|(true|false|null)/g;
  let last = 0, m;
  let i = 0;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) parts.push(<span key={i++}>{s.slice(last, m.index)}</span>);
    if (m[1]) { parts.push(<span key={i++} className="k">{m[1]}</span>); parts.push(<span key={i++}>{m[2]}</span>); }
    else if (m[3]) parts.push(<span key={i++} className="s">{m[3]}</span>);
    else if (m[4]) parts.push(<span key={i++} className="n">{m[4]}</span>);
    else if (m[5]) parts.push(<span key={i++} className="b">{m[5]}</span>);
    last = re.lastIndex;
  }
  if (last < s.length) parts.push(<span key={i++}>{s.slice(last)}</span>);
  return <>{parts}</>;
}

function ToolCall({ call, defaultCollapsed }){
  const [open, setOpen] = React.useState(!defaultCollapsed);
  const [expanded, setExpanded] = React.useState(false);
  const resultSize = JSON.stringify(call.result).length;
  const large = resultSize > 500;
  return (
    <div className="tool-call">
      <div className="th" style={{cursor:'pointer'}} onClick={() => setOpen(!open)}>
        <span style={{color:'var(--hd-accent)', display:'inline-block', transition:'transform 140ms', transform: open ? 'rotate(90deg)' : 'rotate(0deg)', fontSize:22, lineHeight:1, marginRight:4}}>▸</span>
        <span className="badge">TOOL</span>
        <span className="tname">{call.name}()</span>
        <span className="bytes">{resultSize}B</span>
      </div>
      {open && <>
        <div className="kv">
          <span className="label">args</span>
          <pre className="code" style={{margin:0}}><JsonPretty obj={call.args}/></pre>
        </div>
        <div className="kv">
          <span className="label">result</span>
          <div style={{display:'flex', flexDirection:'column', gap:4}}>
            <pre className={`code ${!expanded && large ? 'collapsed' : ''}`} style={{margin:0}}><JsonPretty obj={call.result}/></pre>
            {large && (
              <button className="reset-btn" style={{alignSelf:'flex-start', fontSize:11}} onClick={() => setExpanded(!expanded)}>
                {expanded ? 'Collapse' : `Expand (${resultSize}B)`}
              </button>
            )}
          </div>
        </div>
      </>}
    </div>
  );
}

function CaseDetail({ run, testCase }){
  if (!testCase) {
    return (
      <div className="panel empty" style={{padding:'80px 24px', textAlign:'center'}}>
        <div style={{fontSize:32, marginBottom:8}}>▸</div>
        <h3 style={{fontSize:15, margin:'0 0 6px'}}>Select a test case</h3>
        <p className="mono" style={{color:'var(--hd-muted)'}}>Pick a case on the left to inspect its config, conversation, tool calls, and evaluations.</p>
      </div>
    );
  }

  const convo = window.HD_DATA.sampleConversation[testCase.name] || window.HD_DATA.sampleConversation['refund_eligible_standard'];
  const cfg = run.metadata.agent_config;
  const pv = run.metadata.prompt_version;

  const toolSet = new Set(testCase.tools_called);
  const expected = testCase.expected_tools.map(t => ({ name: t, ok: toolSet.has(t) }));

  const evalsByKind = {
    geval: testCase.metric_results.filter(m => m.kind === 'geval'),
    rag: testCase.metric_results.filter(m => m.kind === 'rag'),
    standard: testCase.metric_results.filter(m => m.kind === 'standard'),
  };

  return (
    <div className="detail">
      <div className="detail-head">
        <div className="row1">
          <span className={`pill ${testCase.passed ? 'pill-pass' : 'pill-fail'}`}>{testCase.passed ? '✓ PASS' : '✕ FAIL'}</span>
          <h2>{testCase.name}</h2>
        </div>
        <div className="row2">
          <span>run <b className="mono">{fmtDateTime(run.created_at)}</b></span>
          <span>prompt <b className="mono" style={{color:'var(--hd-accent)'}}>{pv.version}</b></span>
          <span>model <b className="mono">{cfg.model.name}</b></span>
          <span>temp <b className="mono">{cfg.model.temperature}</b></span>
          <span>commit <b className="mono">{run.git_commit}</b></span>
        </div>
      </div>

      {/* Agent config snapshot */}
      <Collapsible
        eyebrow="AGENT CONFIG SNAPSHOT"
        title="Configuration at run time"
        subtitle={<>Captured from <span className="mono">agent.yaml</span> at run time · secret-bearing fields stripped before persist.</>}
        defaultOpen={false}
        right={<button className="hd-btn hd-btn-ghost" style={{fontSize:12, padding:'5px 12px'}}>View raw JSON</button>}
      >
        <div className="cfg-grid">
          <div className="cfg-item"><span className="k">model.provider</span><span className="v">{cfg.model.provider}</span></div>
          <div className="cfg-item"><span className="k">model.name</span><span className="v">{cfg.model.name}</span></div>
          <div className="cfg-item"><span className="k">model.temperature</span><span className="v">{cfg.model.temperature}</span></div>
          <div className="cfg-item"><span className="k">model.max_tokens</span><span className="v">{cfg.model.max_tokens}</span></div>
          <div className="cfg-item"><span className="k">embedding.provider</span><span className="v">{cfg.embedding.provider}</span></div>
          <div className="cfg-item"><span className="k">embedding.name</span><span className="v">{cfg.embedding.name}</span></div>
          <div className="cfg-item"><span className="k">claude.extended_thinking</span><span className="v">{String(cfg.claude.extended_thinking)}</span></div>
          <div className="cfg-item"><span className="k">prompt.version</span><span className="v" style={{color:'var(--hd-accent)'}}>{pv.version}</span></div>
          <div className="cfg-item"><span className="k">prompt.author</span><span className="v">{pv.author}</span></div>
          <div className="cfg-item"><span className="k">prompt.file_path</span><span className="v">{pv.file_path}</span></div>
          <div className="cfg-item"><span className="k">prompt.source</span><span className="v">{pv.source}</span></div>
          <div className="cfg-item"><span className="k">prompt.tags</span><span className="v">{pv.tags.map(t=>`#${t}`).join(' ')}</span></div>
        </div>
        <div style={{marginTop:12}}>
          <div className="eyebrow" style={{fontSize:10, letterSpacing:'.15em', textTransform:'uppercase', color:'var(--hd-muted)', marginBottom:6}}>Tools ({cfg.tools.length})</div>
          <div className="chip-row">
            {cfg.tools.map(t => (
              <span key={t.name} className="chip" style={{cursor:'default'}}>
                <span style={{color:'var(--hd-accent-soft)', marginRight:6, fontSize:10}}>{t.kind}</span>{t.name}
              </span>
            ))}
          </div>
        </div>
      </Collapsible>

      {/* Conversation + tool calls */}
      <Collapsible
        eyebrow="CONVERSATION"
        title="Thread with tool calls"
        subtitle="User input, agent response, and every tool invocation that happened in between."
        defaultOpen={false}
      >
        <div className="thread">
          <div className="bubble user">
            <div className="who">USER</div>
            <div>{convo.user}</div>
          </div>

          {convo.tool_calls && convo.tool_calls.map((tc, i) => (
            <ToolCall key={i} call={tc} defaultCollapsed={true}/>
          ))}

          <div className="bubble assistant">
            <div className="who">AGENT · {cfg.model.name}</div>
            <div>{convo.assistant}</div>
          </div>
        </div>
      </Collapsible>

      {/* Expected tools */}
      <Collapsible
        eyebrow="EXPECTED TOOLS"
        title="Tool-call coverage"
        subtitle={<>Configured <span className="mono">expected_tools</span> vs. what the agent actually invoked.</>}
        defaultOpen={false}
        right={<span className={`pill ${expected.every(e=>e.ok) ? 'pill-pass' : 'pill-fail'}`}>{expected.filter(e=>e.ok).length}/{expected.length} matched</span>}
      >
        {expected.length === 0 && <div className="mono" style={{padding:'12px 0', color:'var(--hd-muted)'}}>No expected tools configured for this case.</div>}
        <div style={{display:'flex', flexDirection:'column', gap:6}}>
          {expected.map(e => (
            <div key={e.name} className={`expect-row ${e.ok ? 'ok' : 'miss'}`}>
              <span className="ind">{e.ok ? '✓' : '✕'}</span>
              <span className="nm">{e.name}</span>
              <span className="note">{e.ok ? 'called' : 'not invoked'}</span>
            </div>
          ))}
        </div>
      </Collapsible>

      {/* Evaluations */}
      <Collapsible
        eyebrow="EVALUATIONS"
        title="Per-metric results"
        subtitle="Score, threshold, and judge reasoning for every evaluation attached to this case."
        defaultOpen={false}
      >
        {Object.entries(evalsByKind).map(([kind, metrics]) => metrics.length > 0 && (
          <div key={kind} style={{marginBottom: 14}}>
            <div className="eyebrow" style={{fontSize:10, letterSpacing:'.15em', textTransform:'uppercase', color:'var(--hd-accent-soft)', marginBottom:8}}>{kind}</div>
            {metrics.map((m, i) => (
              <div className="eval-row" key={i}>
                <div className="nm">
                  <span className="kind">{kind}</span>
                  <span className="name">{m.name}</span>
                </div>
                <div style={{textAlign:'center'}}>
                  <span className="mono" style={{fontSize:11, color:'var(--hd-muted)'}}>score</span>
                  <div className={`mono ${m.score >= m.threshold ? '' : 'fail'}`} style={{fontSize:14, fontWeight:600, color: m.score >= m.threshold ? 'var(--hd-accent)' : '#ff9d7e'}}>{m.score.toFixed(2)}</div>
                </div>
                <div style={{textAlign:'center'}}>
                  <span className="mono" style={{fontSize:11, color:'var(--hd-muted)'}}>thresh</span>
                  <div className="mono" style={{fontSize:13}}>{m.threshold.toFixed(2)}</div>
                </div>
                <span className={`pill ${m.score >= m.threshold ? 'pill-pass' : 'pill-fail'}`} style={{justifySelf:'end'}}>{m.score >= m.threshold ? 'PASS' : 'FAIL'}</span>
                {m.reasoning && <div className="rsn">{m.reasoning}</div>}
              </div>
            ))}
          </div>
        ))}
      </Collapsible>
    </div>
  );
}

function ExplorerView({ runs, state, setState, compareQueue, setCompareQueue }){
  const [runsCollapsed, setRunsCollapsed] = React.useState(true);
  const selectedRun = state.runId ? runs.find(r => r.id === state.runId) : runs[runs.length-1];
  const cases = selectedRun ? selectedRun.test_results : [];
  const selectedCase = state.caseName ? cases.find(c => c.name === state.caseName) : cases[0];

  if (!runs.length) {
    return (
      <div className="empty">
        <div className="icon">∅</div>
        <h3 style={{fontSize:16, margin:'0 0 6px'}}>No runs found</h3>
        <p className="mono" style={{color:'var(--hd-muted)'}}>Run <code>holodeck test agent.yaml</code> to generate one.</p>
      </div>
    );
  }

  return (
    <div className="explorer" style={runsCollapsed ? {gridTemplateColumns:'48px 340px minmax(0,1fr)'} : undefined}>
      <RunList runs={runs} activeId={selectedRun?.id} collapsed={runsCollapsed} onToggle={() => setRunsCollapsed(!runsCollapsed)} compareQueue={compareQueue} setCompareQueue={setCompareQueue} onSelect={(r) => setState({ runId: r.id, caseName: r.test_results[0]?.name })} />
      <CaseList cases={cases} activeId={selectedCase?.name} onSelect={(c) => setState({ ...state, runId: selectedRun.id, caseName: c.name })}/>
      <CaseDetail run={selectedRun} testCase={selectedCase}/>
    </div>
  );
}

Object.assign(window, { ExplorerView });
