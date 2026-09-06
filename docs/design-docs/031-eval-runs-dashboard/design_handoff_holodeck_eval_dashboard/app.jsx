// App shell — topbar, tabs, view switching, tweaks.

function Topbar({ experiment, runCount }){
  return (
    <header className="topbar">
      <div className="topbar-left">
        <img className="logo-mark" src="assets/holodeck-logo.png" alt="HoloDeck"/>
        <div className="brand">
          <span className="brand-name">HoloDeck</span>
          <span className="brand-sub">Test View</span>
        </div>
        <div className="exp-picker" style={{marginLeft:14}}>
          <span className="dot"/>
          <span className="meta">experiment</span>
          <span className="name">{experiment}</span>
          <span className="chev">▾</span>
        </div>
      </div>
      <div className="topbar-right">
        <span className="mono" style={{fontSize:12}}>{runCount} runs · results/customer-support/</span>
        <span className="warn"><span className="ind"/>streamlit · 0.0.0.0:8501</span>
        <button className="hd-btn hd-btn-ghost" style={{fontSize:12}}>⤓ Download run</button>
        <button className="hd-btn hd-btn-primary" style={{fontSize:12}}>Run test</button>
      </div>
    </header>
  );
}

function Tabs({ tab, setTab, summaryCount, explorerCount, compareCount }){
  return (
    <nav className="tabbar">
      <div className={`tab ${tab==='summary' ? 'active' : ''}`} onClick={() => setTab('summary')}>
        <span>Summary</span><span className="count">{summaryCount}</span>
      </div>
      <div className={`tab ${tab==='explorer' ? 'active' : ''}`} onClick={() => setTab('explorer')}>
        <span>Explorer</span><span className="count">{explorerCount}</span>
      </div>
      <div className={`tab ${tab==='compare' ? 'active' : ''}`} onClick={() => setTab('compare')}>
        <span>Compare</span><span className="count">{compareCount || '—'}</span>
      </div>
      <div className={`tab`} style={{opacity:.7}}>
        <span>Settings</span>
      </div>
      <div className="tabbar-right">
        <span className="mono">launched via</span>
        <span className="cmd-chip"><span className="p">$ </span>holodeck test view agent.yaml</span>
      </div>
    </nav>
  );
}

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "showFilterRail": true,
  "density": "cozy",
  "tabDefault": "summary",
  "compareVariant": 1
}/*EDITMODE-END*/;

function TweaksPanel({ open, tweaks, setTweaks, compareVariant, setCompareVariant }){
  const set = (k, v) => {
    const next = { ...tweaks, [k]: v };
    setTweaks(next);
    try { window.parent.postMessage({ type:'__edit_mode_set_keys', edits: { [k]: v } }, '*'); } catch(e){}
  };
  return (
    <div className={`tweaks ${open ? 'open' : ''}`}>
      <h4>Tweaks</h4>
      <div className="tweak-row">
        <span>Filter rail</span>
        <div className="seg">
          <button className={tweaks.showFilterRail ? 'on' : ''} onClick={() => set('showFilterRail', true)}>shown</button>
          <button className={!tweaks.showFilterRail ? 'on' : ''} onClick={() => set('showFilterRail', false)}>hidden</button>
        </div>
      </div>
      <div className="tweak-row">
        <span>Default tab</span>
        <div className="seg">
          <button className={tweaks.tabDefault==='summary' ? 'on' : ''} onClick={() => set('tabDefault','summary')}>summary</button>
          <button className={tweaks.tabDefault==='explorer' ? 'on' : ''} onClick={() => set('tabDefault','explorer')}>explorer</button>
          <button className={tweaks.tabDefault==='compare' ? 'on' : ''} onClick={() => set('tabDefault','compare')}>compare</button>
        </div>
      </div>
      <div className="tweak-row">
        <span>Density</span>
        <div className="seg">
          <button className={tweaks.density==='cozy' ? 'on' : ''} onClick={() => set('density','cozy')}>cozy</button>
          <button className={tweaks.density==='dense' ? 'on' : ''} onClick={() => set('density','dense')}>dense</button>
        </div>
      </div>
      <div className="tweak-row">
        <span>Compare layout</span>
        <div className="seg">
          <button className={compareVariant===1 ? 'on' : ''} onClick={() => { setCompareVariant(1); set('compareVariant', 1); }}>1</button>
          <button className={compareVariant===2 ? 'on' : ''} onClick={() => { setCompareVariant(2); set('compareVariant', 2); }}>2</button>
          <button className={compareVariant===3 ? 'on' : ''} onClick={() => { setCompareVariant(3); set('compareVariant', 3); }}>3</button>
        </div>
      </div>
    </div>
  );
}

function App(){
  const [tab, setTab] = React.useState(() => localStorage.getItem('hd_tab') || TWEAK_DEFAULTS.tabDefault);
  const [filters, setFilters] = React.useState({ versions: [], models: [], tags: [], minPass: 0, metricKind: 'rag' });
  const [explorerState, setExplorerState] = React.useState({ runId: null, caseName: null });
  const [compareQueue, setCompareQueue] = React.useState(() => {
    try { return JSON.parse(localStorage.getItem('hd_compare_queue') || '[]'); } catch(e) { return []; }
  });
  const [compareVariant, setCompareVariant] = React.useState(() => {
    const v = parseInt(localStorage.getItem('hd_compare_variant') || String(TWEAK_DEFAULTS.compareVariant), 10);
    return [1,2,3].includes(v) ? v : 1;
  });
  const [tweaks, setTweaks] = React.useState(TWEAK_DEFAULTS);
  const [tweaksOpen, setTweaksOpen] = React.useState(false);

  React.useEffect(() => { localStorage.setItem('hd_tab', tab); }, [tab]);
  React.useEffect(() => { localStorage.setItem('hd_compare_queue', JSON.stringify(compareQueue)); }, [compareQueue]);
  React.useEffect(() => { localStorage.setItem('hd_compare_variant', String(compareVariant)); }, [compareVariant]);

  // Edit-mode protocol
  React.useEffect(() => {
    const handler = (e) => {
      const d = e.data || {};
      if (d.type === '__activate_edit_mode') setTweaksOpen(true);
      if (d.type === '__deactivate_edit_mode') setTweaksOpen(false);
    };
    window.addEventListener('message', handler);
    try { window.parent.postMessage({ type:'__edit_mode_available' }, '*'); } catch(e){}
    return () => window.removeEventListener('message', handler);
  }, []);

  const allRuns = window.HD_DATA.runs;

  const filteredRuns = React.useMemo(() => {
    return allRuns.filter(r => {
      if (filters.versions.length && !filters.versions.includes(r.metadata.prompt_version.version)) return false;
      if (filters.models.length && !filters.models.includes(r.metadata.agent_config.model.name)) return false;
      if (filters.tags.length && !filters.tags.some(t => r.metadata.prompt_version.tags.includes(t))) return false;
      if (r.summary.pass_rate < (filters.minPass || 0)) return false;
      return true;
    });
  }, [allRuns, filters]);

  const openRun = (r) => {
    setExplorerState({ runId: r.id, caseName: r.test_results[0]?.name });
    setTab('explorer');
  };

  const setQueueSafe = (next) => setCompareQueue(next.slice(0, 3));

  const density = tweaks.density === 'dense' ? 'dense' : 'cozy';

  const screenLabel = tab === 'summary' ? '01 Summary' : tab === 'explorer' ? '02 Explorer' : '03 Compare';

  return (
    <div className={`app density-${density}`} data-screen-label={screenLabel}>
      <DashStyles/>
      <Topbar experiment="customer-support" runCount={allRuns.length}/>
      <Tabs tab={tab} setTab={setTab} summaryCount={filteredRuns.length} explorerCount={allRuns.length} compareCount={compareQueue.length}/>
      {tab === 'summary' && <SummaryView runs={filteredRuns} filters={filters} setFilters={setFilters} showRail={tweaks.showFilterRail} onOpenRun={openRun} compareQueue={compareQueue} setCompareQueue={setQueueSafe}/>}
      {tab === 'explorer' && <ExplorerView runs={allRuns} state={explorerState} setState={setExplorerState} compareQueue={compareQueue} setCompareQueue={setQueueSafe}/>}
      {tab === 'compare' && <CompareView allRuns={allRuns} queue={compareQueue} setQueue={setQueueSafe} variant={compareVariant} setVariant={setCompareVariant}/>}
      <CompareTray allRuns={allRuns} queue={compareQueue} setQueue={setQueueSafe} onOpenCompare={() => setTab('compare')}/>
      <TweaksPanel open={tweaksOpen} tweaks={tweaks} setTweaks={setTweaks} compareVariant={compareVariant} setCompareVariant={setCompareVariant}/>
    </div>
  );
}

Object.assign(window, { App });
