// Dashboard components — HoloDeck-styled eval dashboard (Streamlit-inspired but richer).

const dashStyles = `
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0; background: var(--hd-page-bg); color: var(--fg1); font-family: var(--font-sans); font-size: 14px; min-height: 100vh; }
  a { color: inherit; text-decoration: none; }
  button { font-family: inherit; }

  /* ---------- App shell ---------- */
  .app { min-height: 100vh; display: grid; grid-template-rows: auto auto 1fr; }
  .topbar { display:flex; align-items:center; justify-content:space-between; padding:14px 28px; border-bottom:1px solid var(--hd-border); background:rgba(5,11,9,.75); backdrop-filter: blur(8px); position:sticky; top:0; z-index:20; }
  .topbar-left { display:flex; align-items:center; gap:14px; }
  .logo-mark { width:30px; height:30px; border-radius:7px; object-fit:cover; box-shadow:0 0 22px rgba(123,255,90,.35); }
  .brand { display:flex; flex-direction:column; line-height:1.1; }
  .brand-name { font-size:15px; font-weight:600; letter-spacing:.03em; }
  .brand-sub { font-size:11px; color:var(--hd-muted); letter-spacing:.12em; text-transform:uppercase; }
  .topbar-center { display:flex; align-items:center; gap:10px; }
  .exp-picker { display:flex; align-items:center; gap:8px; padding:6px 10px 6px 12px; border-radius:999px; background:rgba(12,20,18,.8); border:1px solid var(--hd-accent-glow); font-size:13px; cursor:pointer; transition:border-color var(--dur-quick) var(--ease-standard); }
  .exp-picker:hover { border-color: rgba(123,255,90,.55); }
  .exp-picker .dot { width:6px; height:6px; border-radius:999px; background:var(--hd-accent); box-shadow:0 0 8px var(--hd-accent); }
  .exp-picker .name { font-weight:500; }
  .exp-picker .meta { color:var(--hd-muted); font-size:12px; }
  .exp-picker .chev { color:var(--hd-muted); font-size:11px; }
  .topbar-right { display:flex; align-items:center; gap:10px; font-size:13px; color:var(--hd-muted); }
  .topbar-right .warn { display:flex; align-items:center; gap:6px; font-size:12px; padding:4px 10px; border-radius:999px; background:rgba(123,255,90,.08); border:1px solid var(--hd-accent-glow); color: var(--hd-accent-soft); }
  .topbar-right .warn .ind { width:5px; height:5px; border-radius:999px; background:var(--hd-accent-soft); box-shadow:0 0 6px var(--hd-accent-soft); }

  .tabbar { display:flex; align-items:center; gap:2px; padding:0 24px; border-bottom:1px solid var(--hd-border); background:rgba(5,11,9,.4); }
  .tab { padding:12px 16px; font-size:13px; color:var(--hd-muted); cursor:pointer; border-bottom:2px solid transparent; display:flex; align-items:center; gap:8px; transition: color var(--dur-quick) var(--ease-standard), border-color var(--dur-quick) var(--ease-standard); }
  .tab:hover { color: var(--fg1); }
  .tab.active { color: var(--fg1); border-bottom-color: var(--hd-accent); }
  .tab .count { font-size:11px; color:var(--hd-muted); padding:1px 7px; border-radius:999px; background:rgba(12,20,18,.9); border:1px solid var(--hd-border); font-family:var(--font-mono); }
  .tab.active .count { color: var(--hd-accent); border-color: rgba(123,255,90,.35); }
  .tabbar-right { margin-left:auto; padding:8px 0; display:flex; align-items:center; gap:8px; font-size:12px; color:var(--hd-muted); }
  .cmd-chip { font-family:var(--font-mono); font-size:12px; padding:4px 10px; border-radius:6px; background:#050b09; border:1px solid var(--hd-border); color:#e5e7eb; }
  .cmd-chip .p { color: var(--hd-accent); }

  /* ---------- Summary view ---------- */
  .layout { display:grid; grid-template-columns: 264px minmax(0,1fr); gap:20px; padding:20px 24px 40px; max-width: 1560px; margin: 0 auto; width:100%; }
  .layout.no-rail { grid-template-columns: 1fr; }

  /* Filter rail */
  .rail { display:flex; flex-direction:column; gap:14px; position:sticky; top:120px; max-height: calc(100vh - 140px); overflow:auto; padding-right:4px; }
  .rail-card { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border:1px solid var(--hd-border); border-radius:12px; padding:14px 14px 16px; }
  .rail-card h4 { margin:0 0 10px; font-size:12px; letter-spacing:.15em; text-transform:uppercase; color:var(--hd-muted); font-weight:600; }
  .rail-group { display:flex; flex-direction:column; gap:10px; }
  .rail-group + .rail-group { border-top:1px solid var(--hd-border); padding-top:12px; margin-top:2px; }
  .rail-label { font-size:12px; color:var(--hd-muted); margin-bottom:6px; display:flex; justify-content:space-between; }
  .rail-label .value { color: var(--hd-accent-soft); font-family: var(--font-mono); font-size:11px; }

  .chip-row { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { font-size:12px; padding:4px 10px; border-radius:999px; background:rgba(12,20,18,.7); border:1px solid var(--hd-border); color:var(--hd-muted); cursor:pointer; transition: all var(--dur-quick) var(--ease-standard); font-family: var(--font-mono); }
  .chip:hover { border-color: rgba(123,255,90,.35); color: var(--fg1); }
  .chip.on { background: rgba(123,255,90,.12); border-color: rgba(123,255,90,.5); color: var(--hd-accent); }
  .chip .x { margin-left:5px; opacity:.7; }

  .slider-track { position:relative; height:6px; background:#050b09; border:1px solid var(--hd-border); border-radius:999px; margin:8px 0 4px; }
  .slider-fill { position:absolute; left:0; top:-1px; bottom:-1px; background: linear-gradient(90deg, rgba(123,255,90,.25), var(--hd-accent)); border-radius:999px; }
  .slider-thumb { position:absolute; top:50%; transform: translate(-50%,-50%); width:14px; height:14px; border-radius:999px; background: var(--hd-accent); box-shadow: 0 0 10px rgba(123,255,90,.6); border:2px solid #050b09; }

  .date-field { display:flex; align-items:center; gap:8px; padding:8px 10px; background:#050b09; border:1px solid var(--hd-border); border-radius:8px; font-family:var(--font-mono); font-size:12px; color: var(--fg1); }
  .date-field svg { color:var(--hd-muted); }

  .rail-footer { display:flex; justify-content:space-between; align-items:center; font-size:12px; color:var(--hd-muted); }
  .reset-btn { font-size:12px; color: var(--hd-muted); background:transparent; border:1px solid var(--hd-border); border-radius:999px; padding:4px 10px; cursor:pointer; }
  .reset-btn:hover { color: var(--hd-accent); border-color: rgba(123,255,90,.5); }

  /* Main content */
  .main { display:flex; flex-direction:column; gap:18px; min-width:0; }

  .kpi-strip { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; }
  .kpi { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 55%); border:1px solid var(--hd-border); border-radius:12px; padding:14px 16px; display:flex; flex-direction:column; gap:8px; position:relative; overflow:hidden; }
  .kpi-label { font-size:11px; letter-spacing:.15em; text-transform:uppercase; color: var(--hd-muted); }
  .kpi-value { font-size:28px; font-weight:600; letter-spacing:-.02em; line-height:1; display:flex; align-items:baseline; gap:6px; }
  .kpi-unit { font-size:14px; color:var(--hd-muted); font-weight:400; }
  .kpi-delta { font-size:11px; font-family:var(--font-mono); display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:999px; background: rgba(123,255,90,.1); border:1px solid rgba(123,255,90,.3); color: var(--hd-accent); width: fit-content; }
  .kpi-delta.neg { background: rgba(255, 120, 80, .08); border-color: rgba(255, 120, 80, .3); color: #ff9d7e; }
  .kpi-spark { position:absolute; right:12px; top:50%; transform: translateY(-50%); opacity:.8; }

  .panel { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border:1px solid var(--hd-border); border-radius:14px; padding:18px 20px; }
  .panel-head { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:14px; gap:12px; }
  .panel-head h3 { font-size:16px; margin:0 0 4px; font-weight:600; letter-spacing:-.01em; }
  .panel-head p { margin:0; font-size:12px; color: var(--hd-muted); }
  .panel-head .eyebrow { font-size:10px; letter-spacing:.15em; text-transform:uppercase; color: var(--hd-accent-soft); margin-bottom:4px; }
  .panel-tools { display:flex; gap:6px; }
  .legend { display:flex; flex-wrap:wrap; gap:10px 14px; font-size:12px; color: var(--hd-muted); }
  .legend-item { display:inline-flex; align-items:center; gap:6px; }
  .legend-swatch { width:10px; height:10px; border-radius:2px; }
  .legend-swatch.line { height:3px; border-radius:2px; }

  /* Charts */
  .chart { width:100%; display:block; }
  .chart .grid-line { stroke: var(--hd-border); stroke-width: 1; }
  .chart .axis-label { fill: var(--hd-muted); font-size: 10px; font-family: var(--font-mono); }
  .chart .main-line { fill:none; stroke: var(--hd-accent); stroke-width: 2; }
  .chart .area { fill: url(#accentArea); }
  .chart .dot { fill: var(--hd-accent); stroke: #050b09; stroke-width: 2; }
  .chart .threshold { stroke: rgba(255, 120, 80, .4); stroke-dasharray: 4 4; stroke-width: 1; fill:none; }

  /* breakdown panels row */
  .breakdowns { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; }
  @media (max-width: 1280px) { .breakdowns { grid-template-columns: 1fr; } }

  .metric-row { display:grid; grid-template-columns: 140px 1fr 54px; gap:10px; align-items:center; padding:7px 0; border-bottom:1px dashed rgba(28,43,37,.7); }
  .metric-row:last-child { border-bottom:none; }
  .metric-name { font-size:12px; color:var(--fg1); font-family: var(--font-mono); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .metric-bar { height:8px; background: rgba(5,11,9,.9); border-radius:999px; position:relative; overflow:hidden; border:1px solid var(--hd-border); }
  .metric-bar .fill { position:absolute; inset:0 auto 0 0; background: linear-gradient(90deg, rgba(83,255,156,.5), var(--hd-accent)); border-radius:999px; }
  .metric-bar .thresh { position:absolute; top:-2px; bottom:-2px; width:2px; background: rgba(255,120,80,.7); }
  .metric-score { font-family: var(--font-mono); font-size:12px; text-align:right; color: var(--hd-accent); }
  .metric-score.fail { color: #ff9d7e; }

  /* Run table */
  .table-card { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border:1px solid var(--hd-border); border-radius:14px; overflow:hidden; }
  .table-head { display:flex; align-items:center; justify-content:space-between; padding:16px 20px 10px; }
  .table-search { display:flex; align-items:center; gap:8px; padding:6px 10px; border-radius:8px; background:#050b09; border:1px solid var(--hd-border); color:var(--hd-muted); font-size:12px; font-family:var(--font-mono); min-width:220px; }
  table.runs { width:100%; border-collapse: collapse; font-size:13px; }
  table.runs th { text-align:left; padding:8px 14px; font-size:10px; letter-spacing:.15em; text-transform:uppercase; color: var(--hd-muted); font-weight:600; border-bottom:1px solid var(--hd-border); background: rgba(7,12,10,.9); position:sticky; top:0; }
  table.runs td { padding:10px 14px; border-bottom:1px solid rgba(28,43,37,.5); vertical-align: middle; }
  table.runs tr:last-child td { border-bottom:none; }
  table.runs tbody tr { cursor: pointer; transition: background var(--dur-quick) var(--ease-standard); }
  table.runs tbody tr:hover { background: rgba(123,255,90,.04); }
  table.runs tbody tr.active { background: rgba(123,255,90,.08); box-shadow: inset 3px 0 0 var(--hd-accent); }
  .ts { font-family: var(--font-mono); font-size:12px; color: var(--fg1); }
  .ts .date { color: var(--hd-muted); margin-right:6px; }
  .pill { display:inline-flex; align-items:center; gap:6px; padding:3px 9px; border-radius:999px; font-size:11px; font-family: var(--font-mono); }
  .pill-pass { background: rgba(123,255,90,.1); border:1px solid rgba(123,255,90,.35); color: var(--hd-accent); }
  .pill-warn { background: rgba(255, 200, 90, .08); border:1px solid rgba(255, 200, 90, .3); color: #ffcf5a; }
  .pill-fail { background: rgba(255, 100, 80, .08); border:1px solid rgba(255, 100, 80, .3); color: #ff9d7e; }
  .pill-neutral { background: rgba(12,20,18,.9); border:1px solid var(--hd-border); color: var(--hd-muted); }

  .bar-inline { display:flex; align-items:center; gap:8px; }
  .bar-inline .num { font-family:var(--font-mono); font-size:12px; color: var(--fg1); min-width:48px; }
  .bar-inline .bar { width: 90px; height:6px; background:#050b09; border:1px solid var(--hd-border); border-radius:999px; overflow:hidden; position:relative; }
  .bar-inline .bar .fill { position:absolute; inset:0 auto 0 0; background: linear-gradient(90deg, rgba(83,255,156,.5), var(--hd-accent)); }

  .mono { font-family: var(--font-mono); font-size:12px; color: var(--hd-muted); }
  .mono.fg { color: var(--fg1); }

  /* ---------- Explorer view ---------- */
  .explorer { display:grid; grid-template-columns: 280px 340px minmax(0,1fr); gap:16px; padding:20px 24px 40px; max-width: 1720px; margin: 0 auto; width: 100%; }
  @media (max-width: 1280px) { .explorer { grid-template-columns: 260px 320px minmax(0,1fr); } }

  .list-card { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 55%); border:1px solid var(--hd-border); border-radius:12px; overflow:hidden; display:flex; flex-direction:column; max-height: calc(100vh - 180px); }
  .list-head { padding:14px 14px 10px; border-bottom:1px solid var(--hd-border); display:flex; align-items:center; justify-content:space-between; }
  .list-head h4 { margin:0; font-size:13px; font-weight:600; }
  .list-scroll { overflow:auto; }
  .run-item, .case-item { padding:10px 14px; border-bottom:1px solid rgba(28,43,37,.5); cursor:pointer; display:flex; flex-direction:column; gap:6px; transition: background var(--dur-quick) var(--ease-standard); }
  .run-item:hover, .case-item:hover { background: rgba(123,255,90,.04); }
  .run-item.active, .case-item.active { background: rgba(123,255,90,.08); box-shadow: inset 3px 0 0 var(--hd-accent); }
  .run-item .r1 { display:flex; justify-content:space-between; align-items:center; font-size:12px; }
  .run-item .r1 .ver { color: var(--hd-accent); font-family:var(--font-mono); }
  .run-item .r2 { display:flex; align-items:center; gap:8px; font-size:11px; color: var(--hd-muted); font-family:var(--font-mono); }
  .case-item .c1 { display:flex; justify-content:space-between; align-items:center; gap:8px; }
  .case-item .c1 .nm { font-size:13px; font-family: var(--font-mono); color: var(--fg1); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .case-item .c2 { display:flex; flex-wrap:wrap; gap:6px; }
  .mini-metric { font-size:10px; font-family:var(--font-mono); color: var(--hd-muted); padding:1px 6px; border-radius:4px; background:rgba(5,11,9,.7); border:1px solid var(--hd-border); }
  .mini-metric.pass { color: var(--hd-accent); border-color: rgba(123,255,90,.3); }
  .mini-metric.fail { color: #ff9d7e; border-color: rgba(255,120,80,.3); }

  .detail { display:flex; flex-direction:column; gap:14px; min-width:0; }
  .detail-head { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border:1px solid var(--hd-border); border-radius:14px; padding:16px 20px; }
  .detail-head .row1 { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
  .detail-head h2 { margin:0; font-size:20px; font-weight:600; letter-spacing:-.02em; font-family: var(--font-mono); }
  .detail-head .row2 { font-size:12px; color:var(--hd-muted); display:flex; flex-wrap:wrap; gap:12px; }
  .detail-head .row2 b { color: var(--fg1); font-weight:500; }

  .cfg-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px 18px; margin-top:12px; font-size:12px; }
  .cfg-item { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px dashed rgba(28,43,37,.7); }
  .cfg-item .k { color:var(--hd-muted); }
  .cfg-item .v { font-family:var(--font-mono); color: var(--fg1); max-width:60%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

  .thread { display:flex; flex-direction:column; gap:10px; }
  .bubble { padding:12px 14px; border-radius:12px; max-width: 90%; font-size:13px; line-height: 1.55; }
  .bubble.user { align-self: flex-end; background: rgba(123,255,90,.08); border:1px solid rgba(123,255,90,.25); color: var(--fg1); border-bottom-right-radius:4px; }
  .bubble.assistant { align-self: flex-start; background: #050b09; border:1px solid var(--hd-border); color: var(--fg1); border-bottom-left-radius:4px; }
  .bubble .who { font-size:10px; letter-spacing:.15em; text-transform:uppercase; color: var(--hd-muted); margin-bottom:4px; }
  .bubble.user .who { color: var(--hd-accent-soft); }

  .tool-call { background: linear-gradient(145deg, rgba(12,20,18,.95), rgba(7,12,10,.9)); border:1px solid rgba(123,255,90,.22); border-radius:10px; padding:10px 12px; display:flex; flex-direction:column; gap:8px; }
  .tool-call .th { display:flex; align-items:center; gap:10px; font-size:12px; }
  .tool-call .tname { font-family: var(--font-mono); color: var(--hd-accent); font-weight:500; }
  .tool-call .badge { font-size:10px; padding:2px 7px; border-radius:4px; letter-spacing:.1em; background:rgba(123,255,90,.1); border:1px solid rgba(123,255,90,.3); color: var(--hd-accent); text-transform:uppercase; }
  .tool-call .bytes { font-family:var(--font-mono); font-size:11px; color: var(--hd-muted); margin-left:auto; }
  .kv { display:grid; grid-template-columns: 56px 1fr; gap:8px; align-items:start; font-size:12px; }
  .kv .label { color: var(--hd-muted); font-family: var(--font-mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase; padding-top:2px; }
  .code { background:#050b09; border:1px solid var(--hd-border); border-radius:8px; padding:10px 12px; font-family: var(--font-mono); font-size:12px; color:#e5e7eb; white-space:pre-wrap; word-break: break-word; overflow:auto; max-height: 240px; }
  .code.collapsed { max-height: 64px; position:relative; }
  .code .k { color: #9bff5f; }
  .code .s { color: #e5e7eb; }
  .code .n { color: #5ae0a6; }
  .code .b { color: #ffcf5a; }

  .expect-row { display:flex; align-items:center; gap:10px; padding:7px 10px; border-radius:8px; background:#050b09; border:1px solid var(--hd-border); font-family: var(--font-mono); font-size:12px; }
  .expect-row.ok { border-color: rgba(123,255,90,.3); }
  .expect-row.miss { border-color: rgba(255,120,80,.3); }
  .expect-row .ind { width:18px; height:18px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-size:11px; }
  .expect-row.ok .ind { background: rgba(123,255,90,.15); color: var(--hd-accent); border:1px solid rgba(123,255,90,.35); }
  .expect-row.miss .ind { background: rgba(255,120,80,.1); color: #ff9d7e; border:1px solid rgba(255,120,80,.3); }
  .expect-row .nm { flex:1; }
  .expect-row .note { font-size:11px; color: var(--hd-muted); }

  .eval-row { display:grid; grid-template-columns: 1fr 80px 80px 90px; gap:10px; align-items:center; padding:10px 12px; background:#050b09; border:1px solid var(--hd-border); border-radius:8px; font-size:12px; }
  .eval-row + .eval-row { margin-top:6px; }
  .eval-row .nm { display:flex; flex-direction:column; gap:2px; }
  .eval-row .kind { font-size:10px; letter-spacing:.15em; text-transform:uppercase; color: var(--hd-muted); }
  .eval-row .name { font-family: var(--font-mono); color: var(--fg1); }
  .eval-row .rsn { grid-column: 1 / -1; margin-top:8px; font-size:12px; color: var(--hd-muted); line-height:1.5; background: rgba(7,12,10,.6); border-left:2px solid var(--hd-accent-glow); padding:8px 10px; border-radius:4px; }

  .flex-between { display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .gap-tight { display:flex; align-items:center; gap:8px; }

  /* Tweaks */
  .tweaks { position: fixed; bottom: 20px; right: 20px; width: 280px; background: linear-gradient(145deg, rgba(10,17,15,.97), #070c0a 55%); border:1px solid rgba(123,255,90,.35); border-radius:14px; padding:14px 16px; box-shadow: 0 20px 50px rgba(16,255,122,.22); z-index: 50; display: none; }
  .tweaks.open { display:block; }
  .tweaks h4 { margin:0 0 10px; font-size:12px; letter-spacing:.15em; text-transform:uppercase; color: var(--hd-accent); }
  .tweak-row { display:flex; align-items:center; justify-content:space-between; padding: 6px 0; border-bottom:1px dashed rgba(28,43,37,.6); font-size:13px; }
  .tweak-row:last-child { border-bottom:none; }
  .seg { display:flex; background:#050b09; border:1px solid var(--hd-border); border-radius:999px; padding:2px; }
  .seg button { border: none; background:transparent; color: var(--hd-muted); padding:3px 10px; border-radius:999px; font-size:11px; cursor:pointer; font-family: var(--font-mono); }
  .seg button.on { background: rgba(123,255,90,.15); color: var(--hd-accent); }

  .empty { padding:40px 20px; text-align:center; color: var(--hd-muted); }
  .empty .icon { font-size: 28px; margin-bottom: 8px; }
  .empty code { font-family: var(--font-mono); font-size:12px; background:#050b09; border:1px solid var(--hd-border); padding:3px 8px; border-radius:6px; color: var(--hd-accent); }

  /* ---------- Compare view ---------- */
  .cmp-wrap { padding: 20px 24px 140px; max-width: 1720px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; gap: 16px; min-width: 0; }

  .cmp-toolbar { display: flex; justify-content: space-between; align-items: flex-end; padding: 6px 2px 12px; border-bottom: 1px solid var(--hd-border); }
  .cmp-toolbar-left .eyebrow { font-size: 10px; letter-spacing: .15em; text-transform: uppercase; }
  .cmp-toolbar-right { display: flex; align-items: center; gap: 12px; }

  /* Empty state */
  .cmp-empty { padding: 60px 40px 80px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 18px; background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border: 1px solid var(--hd-border); border-radius: 18px; margin-top: 40px; }
  .cmp-empty-icon { opacity: .9; }
  .cmp-empty h2 { margin: 0; font-size: 28px; font-weight: 600; letter-spacing: -.02em; }
  .cmp-empty p { margin: 0; max-width: 560px; color: var(--hd-muted); font-size: 14px; line-height: 1.6; }
  .cmp-empty-actions { display: flex; gap: 10px; margin-top: 6px; }
  .cmp-empty-hint { margin-top: 12px; color: var(--hd-muted); font-size: 12px; }
  .cmp-empty-kbd { font-family: var(--font-mono); background: rgba(123,255,90,.12); color: var(--hd-accent); padding: 1px 7px; border-radius: 4px; border: 1px solid rgba(123,255,90,.3); margin: 0 2px; }

  /* Column header */
  .cmp-col-head { background: linear-gradient(145deg, rgba(12,20,18,.9), #070c0a 65%); border: 1px solid var(--hd-border); border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .cmp-col-head-top { display: flex; align-items: center; gap: 8px; }
  .cmp-dot { width: 8px; height: 8px; border-radius: 999px; flex-shrink: 0; }
  .cmp-slot-label { font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: var(--hd-muted); font-family: var(--font-mono); white-space: nowrap; }
  .cmp-x { margin-left: auto; background: transparent; border: 1px solid var(--hd-border); color: var(--hd-muted); font-size: 14px; line-height: 1; width: 20px; height: 20px; border-radius: 999px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; padding: 0; transition: all var(--dur-quick); }
  .cmp-x:hover { color: #ff9d7e; border-color: rgba(255,120,80,.5); }
  .cmp-col-head-ts { font-size: 12px; color: var(--fg1); }
  .cmp-col-head-meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; }
  .cmp-ver { font-family: var(--font-mono); font-weight: 500; }
  .cmp-col-head-commit { font-size: 11px; color: var(--hd-muted); }

  /* Block (summary, config, matrix) */
  .cmp-block { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border: 1px solid var(--hd-border); border-radius: 14px; padding: 18px 20px; }
  .cmp-block-head { margin-bottom: 14px; }
  .cmp-block-head .eyebrow { font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--hd-accent-soft); margin-bottom: 4px; }
  .cmp-block-head h3 { font-size: 16px; margin: 0 0 4px; font-weight: 600; letter-spacing: -.01em; }
  .cmp-block-head p { margin: 0; font-size: 12px; color: var(--hd-muted); }

  .cmp-cols { display: grid; gap: 12px; margin-bottom: 12px; }

  .cmp-rows { display: grid; gap: 6px 12px; row-gap: 2px; }
  .cmp-row-label { font-size: 12px; color: var(--hd-muted); font-family: var(--font-mono); padding: 10px 0; border-bottom: 1px dashed rgba(28,43,37,.7); display: flex; align-items: center; }
  .cmp-cell { padding: 10px 12px; border-bottom: 1px dashed rgba(28,43,37,.7); display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px; min-width: 0; }
  .cmp-cell-val { font-size: 14px; color: var(--fg1); font-family: var(--font-mono); min-width: 0; }
  .cmp-cell-val.accent-big { font-size: 18px; font-weight: 600; color: var(--hd-accent); }
  .cmp-cfg-cell { font-size: 12px; }
  .cmp-cfg-cell.different { background: rgba(255, 200, 90, .06); border-left: 2px solid rgba(255, 200, 90, .5); padding-left: 10px; border-radius: 4px; }
  .cmp-cfg-cell.same { opacity: .75; }
  .cmp-diff-badge { font-size: 10px; padding: 1px 7px; border-radius: 999px; background: rgba(255, 200, 90, .12); border: 1px solid rgba(255, 200, 90, .4); color: #ffcf5a; font-family: var(--font-mono); letter-spacing: .1em; text-transform: uppercase; margin-left: auto; }

  /* Delta pill */
  .delta { font-family: var(--font-mono); font-size: 11px; padding: 2px 7px; border-radius: 999px; white-space: nowrap; font-weight: 500; }
  .delta-pos { background: rgba(123,255,90,.1); color: var(--hd-accent); border: 1px solid rgba(123,255,90,.3); }
  .delta-neg { background: rgba(255, 120, 80, .08); color: #ff9d7e; border: 1px solid rgba(255, 120, 80, .3); }
  .delta-neutral { color: var(--hd-muted); padding: 2px 0; font-size: 13px; }

  /* Case matrix */
  .cmp-matrix { display: grid; gap: 3px; align-items: stretch; margin-top: 6px; }
  .cmp-mh { font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--hd-muted); font-family: var(--font-mono); padding: 8px 6px; display: flex; align-items: center; justify-content: center; gap: 5px; background: rgba(7,12,10,.7); border-radius: 6px; }
  .cmp-mh-left { justify-content: flex-start; padding-left: 10px; }
  .cmp-mh-dot { width: 7px; height: 7px; border-radius: 999px; display: inline-block; }
  .cmp-mc-label { font-family: var(--font-mono); font-size: 12px; color: var(--fg1); padding: 10px 10px; background: rgba(5,11,9,.6); border-radius: 6px; display: flex; align-items: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cmp-mc { padding: 10px 8px; border-radius: 6px; display: flex; align-items: center; justify-content: space-between; gap: 6px; font-family: var(--font-mono); font-size: 12px; transition: all var(--dur-quick); cursor: pointer; }
  .cmp-mc:hover { filter: brightness(1.2); }
  .cmp-mc.pass { background: rgba(123,255,90,.12); color: var(--hd-accent); border: 1px solid rgba(123,255,90,.3); }
  .cmp-mc.fail { background: rgba(255, 120, 80, .1); color: #ff9d7e; border: 1px solid rgba(255, 120, 80, .3); }
  .cmp-mc.regression { box-shadow: 0 0 0 1px #ff9d7e, 0 0 0 3px rgba(255,120,80,.15); }
  .cmp-mc.improvement { box-shadow: 0 0 0 1px #7bff5a, 0 0 0 3px rgba(123,255,90,.15); }
  .cmp-mc-missing { background: rgba(5,11,9,.5); color: var(--hd-muted); border: 1px dashed var(--hd-border); justify-content: center; }
  .cmp-mc-state { font-size: 14px; }
  .cmp-mc-score { font-size: 11px; opacity: .85; }

  .cmp-legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--hd-muted); margin-top: 14px; padding-top: 12px; border-top: 1px dashed rgba(28,43,37,.7); }

  /* Variant 2 — baseline card */
  .cmp-v2-grid { display: grid; gap: 14px; align-items: stretch; }
  .cmp-v2-baseline, .cmp-v2-delta { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border: 1px solid var(--hd-border); border-radius: 14px; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; }
  .cmp-v2-baseline { border-color: rgba(123,255,90,.4); box-shadow: 0 0 0 1px rgba(123,255,90,.15); }
  .cmp-v2-label { display: flex; align-items: center; gap: 8px; }
  .cmp-v2-title { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; font-size: 17px; font-weight: 600; }
  .cmp-v2-model { font-size: 12px; color: var(--hd-muted); display: flex; align-items: center; gap: 8px; }
  .cmp-v2-big { display: flex; gap: 24px; align-items: center; margin-top: 4px; }
  .cmp-v2-pass { display: flex; flex-direction: column; gap: 4px; }
  .cmp-v2-pass .num { font-size: 34px; font-weight: 600; letter-spacing: -.02em; color: var(--hd-accent); line-height: 1; }
  .cmp-v2-stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px 18px; flex: 1; }
  .cmp-v2-stats > div { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px dashed rgba(28,43,37,.6); padding: 3px 0; }
  .cmp-sub { font-size: 11px; color: var(--hd-muted); letter-spacing: .08em; text-transform: uppercase; }
  .cmp-delta-row { display: grid; grid-template-columns: 70px 1fr auto; align-items: baseline; gap: 10px; padding: 7px 0; border-bottom: 1px dashed rgba(28,43,37,.6); }
  .cmp-delta-row:last-child { border-bottom: none; }
  .cmp-diff-dot { margin-left: 8px; font-size: 10px; color: #ffcf5a; font-family: var(--font-mono); }

  /* Variant 3 — strip + callouts */
  .cmp-v3-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .cmp-v3-card { background: linear-gradient(145deg, rgba(10,17,15,.95), #070c0a 60%); border: 1px solid var(--hd-border); border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column; gap: 6px; }
  .cmp-v3-card.baseline { border-color: rgba(123,255,90,.4); }
  .cmp-v3-card-head { display: flex; align-items: center; gap: 8px; }
  .cmp-v3-ver { font-family: var(--font-mono); font-size: 17px; font-weight: 600; }
  .cmp-v3-model { font-size: 11px; color: var(--hd-muted); }
  .cmp-v3-pass { display: flex; align-items: baseline; gap: 10px; margin-top: 4px; }
  .cmp-v3-pass .num { font-size: 24px; font-weight: 600; color: var(--hd-accent); font-family: var(--font-mono); }
  .cmp-v3-mini { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--hd-muted); margin-top: 2px; }
  .cmp-v3-mini b { color: var(--fg1); font-weight: 500; }
  .cmp-v3-callouts { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
  .cmp-v3-callout { background: rgba(7,12,10,.7); border: 1px solid var(--hd-border); border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
  .cmp-v3-callout-head { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--hd-muted); }
  .cmp-v3-callout-body { display: flex; flex-direction: column; gap: 6px; }
  .cmp-v3-callout-row { font-size: 12px; display: flex; align-items: center; }

  /* Floating tray */
  .cmp-tray { position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%); background: linear-gradient(145deg, rgba(10,17,15,.97), #070c0a 65%); border: 1px solid rgba(123,255,90,.4); border-radius: 14px; padding: 10px 14px; display: flex; align-items: center; gap: 18px; box-shadow: 0 20px 50px rgba(16,255,122,.2), 0 0 0 1px rgba(123,255,90,.15); z-index: 40; }
  .cmp-tray-label { display: flex; flex-direction: column; gap: 2px; }
  .cmp-tray-items { display: flex; gap: 8px; }
  .cmp-tray-item { display: flex; align-items: center; gap: 6px; padding: 5px 10px; background: #050b09; border: 1px solid var(--hd-border); border-radius: 999px; font-size: 12px; }
  .cmp-tray-empty { opacity: .35; border-style: dashed; font-style: italic; }
  .cmp-tray-base { font-size: 9px; letter-spacing: .15em; text-transform: uppercase; color: var(--hd-accent); background: rgba(123,255,90,.12); border: 1px solid rgba(123,255,90,.3); padding: 1px 5px; border-radius: 4px; font-family: var(--font-mono); }
  .cmp-tray-item .cmp-x { margin-left: 2px; width: 16px; height: 16px; font-size: 12px; border: none; }
  .cmp-tray-actions { display: flex; gap: 8px; align-items: center; }

  /* Compare add button */
  .cmp-add { background: transparent; border: 1px solid var(--hd-border); color: var(--hd-muted); border-radius: 6px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; transition: all var(--dur-quick); font-family: var(--font-mono); padding: 0; flex-shrink: 0; }
  .cmp-add:hover { color: var(--hd-accent); border-color: rgba(123,255,90,.5); background: rgba(123,255,90,.08); }
  .cmp-add.on { background: rgba(123,255,90,.15); border-color: rgba(123,255,90,.6); color: var(--hd-accent); }
  .cmp-add.disabled { opacity: .3; cursor: not-allowed; }
  .cmp-add.disabled:hover { color: var(--hd-muted); border-color: var(--hd-border); background: transparent; }
  .cmp-add-sm { width: 22px; height: 22px; font-size: 14px; }
  .cmp-add-md { width: 26px; height: 26px; font-size: 15px; }
  .cmp-add-plus { line-height: 1; }
  .cmp-add-num { font-size: 11px; font-weight: 600; }
`;

function DashStyles(){ return <style dangerouslySetInnerHTML={{__html: dashStyles}} /> }
Object.assign(window, { DashStyles });
