// Mock EvalRun data — shape mirrors the spec.
window.HD_DATA = (() => {
  const now = new Date('2026-04-18T14:22:09Z');
  const mkDate = (daysAgo, h = 14, m = 22) => {
    const d = new Date(now);
    d.setDate(d.getDate() - daysAgo);
    d.setHours(h, m, 9, 812);
    return d;
  };

  // 24 runs over ~6 weeks for customer-support agent, plus a couple other agents
  const runs = [];
  const promptVersions = [
    { version: 'v1.0', tags: ['baseline'] },
    { version: 'v1.1', tags: ['baseline','rewrite'] },
    { version: 'v1.2', tags: ['rewrite'] },
    { version: 'v1.2.1', tags: ['experimental'] },
    { version: 'v1.3', tags: ['rag-tuning'] },
    { version: 'v1.4', tags: ['rag-tuning','concise'] },
    { version: 'v2.0', tags: ['v2','concise'] },
  ];
  const models = ['claude-sonnet-4.5', 'claude-haiku-4.5', 'gpt-4o'];

  const RAG_METRICS = ['faithfulness','answer_relevancy','contextual_precision','contextual_recall','contextual_relevancy'];
  const GEVAL_METRICS = ['tone_of_voice','policy_compliance','escalation_appropriateness'];
  const STD_METRICS = ['bleu','rouge','meteor'];

  const baseCases = [
    { name: 'refund_eligible_standard', tools: ['lookup_order','issue_refund','send_email'], expected:['lookup_order','issue_refund'] },
    { name: 'refund_outside_window',    tools: ['lookup_order','policy_lookup'], expected:['lookup_order','policy_lookup'] },
    { name: 'escalate_to_human',        tools: ['lookup_order','escalate_to_agent'], expected:['escalate_to_agent'] },
    { name: 'order_status_query',       tools: ['lookup_order'], expected:['lookup_order'] },
    { name: 'cancel_subscription',      tools: ['lookup_account','cancel_subscription'], expected:['cancel_subscription'] },
    { name: 'product_availability',     tools: ['inventory_search'], expected:['inventory_search'] },
    { name: 'password_reset',           tools: ['send_email'], expected:['send_email'] },
    { name: 'shipping_delay_apology',   tools: ['lookup_order'], expected:['lookup_order'] },
    { name: 'multi_item_return',        tools: ['lookup_order','issue_refund'], expected:['lookup_order','issue_refund'] },
    { name: 'billing_dispute',          tools: ['lookup_order','policy_lookup','escalate_to_agent'], expected:['escalate_to_agent'] },
    { name: 'greeting_smalltalk',       tools: [], expected:[] },
    { name: 'out_of_scope_question',    tools: [], expected:[] },
  ];

  // Synthesize a pass-rate trajectory: starts ~0.58, dips at v1.1 regression, recovers to ~0.92 at v2.0
  const trajectory = [
    0.58, 0.60, 0.62, 0.55, 0.52, 0.50,   // v1.0–v1.1 regression
    0.63, 0.68, 0.72, 0.74, 0.76, 0.78,   // v1.2, auto fork
    0.80, 0.81, 0.79, 0.83, 0.85, 0.84,   // v1.3 rag
    0.88, 0.90, 0.89, 0.92, 0.91, 0.93,   // v1.4, v2.0
  ];

  for (let i = 0; i < 24; i++) {
    const daysAgo = 42 - i*1.75;
    const ts = mkDate(daysAgo, 9 + (i % 8), (i*13) % 60);
    const pv = promptVersions[Math.min(Math.floor(i/4), promptVersions.length-1)];
    const model = i < 14 ? 'claude-sonnet-4.5' : (i % 3 === 0 ? 'gpt-4o' : 'claude-sonnet-4.5');
    const passRate = trajectory[i];
    const total = 12;
    const passed = Math.round(passRate * total);

    const cases = baseCases.map((bc, j) => {
      const caseSeed = (i*7 + j*3);
      const caseRoll = (Math.sin(caseSeed) + 1) / 2;
      const passes = caseRoll < passRate;
      const metrics = [];
      // every case has one geval metric
      metrics.push({
        kind: 'geval',
        name: GEVAL_METRICS[j % GEVAL_METRICS.length],
        score: Math.max(0.2, Math.min(1, passRate + (caseRoll - 0.5) * 0.3)),
        threshold: 0.7,
        passed,
        reasoning: passes
          ? 'Response maintains professional tone, acknowledges the customer concern, and follows documented refund policy without overpromising.'
          : 'Response drifted into casual register mid-turn and did not cite the specific policy clause required for refunds outside the 30-day window.'
      });
      // rag metrics on half the cases
      if (j % 2 === 0) {
        RAG_METRICS.forEach((m, k) => {
          metrics.push({
            kind: 'rag',
            name: m,
            score: Math.max(0.3, Math.min(1, passRate + (k-2)*0.04 + (caseRoll-0.5)*0.2)),
            threshold: 0.7,
            passed: passes,
          });
        });
      }
      // standard metrics on a couple
      if (j < 3) {
        STD_METRICS.forEach((m, k) => {
          metrics.push({
            kind: 'standard',
            name: m,
            score: Math.max(0.2, Math.min(0.95, passRate - 0.1 + (caseRoll-0.5)*0.15)),
            threshold: 0.6,
            passed: passes,
          });
        });
      }
      return {
        name: bc.name,
        tools_called: bc.tools,
        expected_tools: bc.expected,
        passed: passes,
        metric_results: metrics,
      };
    });

    runs.push({
      id: `run-${i.toString().padStart(3,'0')}`,
      file: `results/customer-support/${ts.toISOString().replace(/[:.]/g,'-')}.json`,
      created_at: ts.toISOString(),
      holodeck_version: '0.9.2',
      git_commit: 'a3f9' + (caseHash(i) + '').slice(0,4),
      summary: {
        pass_rate: passRate,
        passed,
        total,
        duration_ms: 18000 + (i*500 % 9000),
      },
      metadata: {
        agent_config: {
          name: 'customer-support',
          model: { provider: 'anthropic', name: model, temperature: i > 12 ? 0.2 : 0.7, max_tokens: 1024 },
          embedding: { provider: 'openai', name: 'text-embedding-3-small' },
          tools: [
            { kind: 'function', name: 'lookup_order' },
            { kind: 'function', name: 'issue_refund' },
            { kind: 'function', name: 'cancel_subscription' },
            { kind: 'function', name: 'escalate_to_agent' },
            { kind: 'vectorstore', name: 'policy_docs', provider: 'chroma' },
            { kind: 'function', name: 'send_email' },
            { kind: 'function', name: 'inventory_search' },
            { kind: 'function', name: 'policy_lookup' },
            { kind: 'function', name: 'lookup_account' },
          ],
          claude: { extended_thinking: i > 18, web_search: false },
        },
        prompt_version: {
          version: pv.version,
          author: 'justin@holodeck.ai',
          description: 'Customer support agent — refund, cancellation, escalation flows.',
          tags: pv.tags,
          source: 'file',
          file_path: 'prompts/support.md',
          body_hash: 'sha256:' + (caseHash(i*31) + '').repeat(3).slice(0,32),
          extra: {},
        },
        cli_args: ['holodeck','test','agent.yaml'],
      },
      test_results: cases,
    });
  }

  function caseHash(n) { return Math.abs(Math.floor(Math.sin(n)*1e8)) % 1e6; }

  return {
    runs,
    // Synthetic conversation for detail panel — reused across cases
    sampleConversation: {
      'refund_eligible_standard': {
        user: "Hi, I ordered noise-canceling headphones on April 2 (order #A-8844) and they arrived yesterday but the right cup rattles. I'd like a refund please.",
        assistant: "I'm sorry the headphones arrived faulty — I can take care of that right away. I've pulled up order #A-8844 and confirmed it shipped on April 3, so you're well within the 30-day return window. I've issued a full refund of $249.00 to your original payment method; it should appear in 3–5 business days. I've also emailed you a prepaid return label. Anything else?",
        tool_calls: [
          { name: 'lookup_order', args: { order_id: 'A-8844' }, result: { id: 'A-8844', status: 'delivered', total: 249.00, ship_date: '2026-04-03', items: [{ sku: 'HP-NC-770', name: 'NC-770 Headphones' }] } },
          { name: 'issue_refund', args: { order_id: 'A-8844', amount: 249.00, reason: 'faulty_item' }, result: { refund_id: 'ref_01HX9K2PQ', status: 'processed', eta_business_days: 5 } },
          { name: 'send_email', args: { to: 'dana@example.com', template: 'return_label', order_id: 'A-8844' }, result: { sent: true, message_id: 'msg_a8f83c' } },
        ],
      },
      'refund_outside_window': {
        user: "I bought a desk lamp from you last November. It's now flickering. Can I get a refund?",
        assistant: "Thanks for flagging this. I looked up the order and the purchase was on November 12, 2025, which puts it well outside our 30-day return window. Per our return policy (section 4.2), items purchased more than 30 days ago aren't eligible for a refund, but the lamp carries a 12-month manufacturer warranty which may cover the flickering. I can point you to the warranty claim form — want me to?",
        tool_calls: [
          { name: 'lookup_order', args: { query: 'desk lamp november' }, result: { id: 'B-2210', status: 'delivered', ship_date: '2025-11-12' } },
          { name: 'policy_lookup', args: { topic: 'return_window' }, result: { section: '4.2', window_days: 30, exceptions: ['manufacturer_warranty'] } },
        ],
      },
    },
  };
})();
