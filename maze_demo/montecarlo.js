/* =========================================================================
 * MONTE CARLO CONTROL  (model-free, first-visit, ε-greedy)
 * -------------------------------------------------------------------------
 * Idea: unlike Q-learning, Monte Carlo does NOT update after every move.
 * It plays a WHOLE episode first, records the trajectory, then goes back and
 * computes the actual return G (sum of discounted rewards) that followed each
 * state-action pair. Q is updated toward those real, observed returns:
 *
 *   G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
 *   Q(s,a) ← average of all first-visit returns G seen for (s,a)
 *
 * The unit of work ("step") here is ONE FULL EPISODE, because a return is only
 * known once the episode terminates. This is the key teaching contrast with
 * TD methods: MC waits for the real outcome instead of bootstrapping.
 * ======================================================================= */
window.RLAlgorithms = window.RLAlgorithms || {};

window.RLAlgorithms["montecarlo"] = function (env) {
  const { N, ACTIONS } = env;
  let Q, returnsCount, agent;

  function fresh() {
    Q = Array.from({ length: N }, () =>
        Array.from({ length: N }, () => [0, 0, 0, 0]));
    // how many returns have been averaged into each Q(s,a) so far
    returnsCount = Array.from({ length: N }, () =>
        Array.from({ length: N }, () => [0, 0, 0, 0]));
    agent = { x: env.start.x, y: env.start.y };
  }
  fresh();

  const maxQ = (x, y) => Math.max(...Q[y][x]);
  function argmaxQ(x, y) {
    const q = Q[y][x];
    let best = 0;
    for (let i = 1; i < 4; i++) if (q[i] > q[best]) best = i;
    return best;
  }
  function chooseAction(x, y, eps) {
    if (Math.random() < eps) return Math.floor(Math.random() * 4);
    return argmaxQ(x, y);
  }

  return {
    label: "Monte Carlo",
    tagline: "Plays a full episode, then learns from the actual return.",
    formula:
      'G<sub>t</sub> = Σ γ<sup>k</sup>·r &nbsp;·&nbsp; Q(s,a) ← average of first-visit returns G',
    rewardNote: "r = +1 goal · −0.01 step · −0.20 hit wall",
    stepLabel: "▶ Run episode",
    episodeLabel: "⏭ ×10 episodes",
    usesAlpha: false,
    usesEpsilon: true,
    showsQ: true,
    hasAgent: true,

    reset: fresh,
    value: (x, y) => maxQ(x, y),
    q: (x, y) => Q[y][x],
    policy: (x, y) => (Q[y][x].some(v => Math.abs(v) > 1e-6) ? argmaxQ(x, y) : null),
    get agentPos() { return { x: agent.x, y: agent.y }; },

    /* one full episode: roll out, then back up returns */
    step(params) {
      const { gamma, epsilon } = params;

      // ---- 1. roll out a complete episode (an "experience") ----
      const traj = [];                 // {x,y,a,r}
      let cur = { x: env.start.x, y: env.start.y };
      let guard = 0, totalR = 0, reachedGoal = false;
      while (guard++ < N * N * 6) {
        const a = chooseAction(cur.x, cur.y, epsilon);
        const { nx, ny, reward, done } = env.transition(cur.x, cur.y, a);
        traj.push({ x: cur.x, y: cur.y, a, r: reward });
        totalR += reward;
        cur = { x: nx, y: ny };
        if (done) { reachedGoal = true; break; }
      }
      agent = { x: cur.x, y: cur.y };

      // ---- 2. walk backwards computing the return G, first-visit updates ----
      const seen = new Set();
      let G = 0;
      let updates = 0;
      let sampleLine = null;           // remember one update to show in detail
      for (let t = traj.length - 1; t >= 0; t--) {
        const { x, y, a, r } = traj[t];
        G = r + gamma * G;             // G_t = r_{t+1} + γ·G_{t+1}
        const k = x + "," + y + "," + a;
        if (seen.has(k)) continue;     // first-visit: skip later repeats
        seen.add(k);

        const n = returnsCount[y][x][a] + 1;
        const oldQ = Q[y][x][a];
        const newQ = oldQ + (G - oldQ) / n;   // incremental mean of returns
        Q[y][x][a] = newQ;
        returnsCount[y][x][a] = n;
        updates++;

        if (!sampleLine) {
          sampleLine = { x, y, a, G, oldQ, newQ, n };
        }
      }

      // show the earliest-in-time first-visit update (start of path) in detail
      const d = sampleLine;
      let detail = "";
      if (d) {
        const aName = ACTIONS[d.a].name;
        detail = `
          <div class="eq"><span class="dim">return at</span> s=<span class="b">(${d.x},${d.y})</span>,
            <span class="a">${aName}</span>: &nbsp; G = <b>${d.G.toFixed(3)}</b></div>
          <div class="eq"><span class="dim">visits n =</span> ${d.n}</div>
          <div class="eq result">Q(s,<span class="a">${aName}</span>) ← Q + (G − Q)/n =
            ${d.oldQ.toFixed(3)} + (${d.G.toFixed(3)} − ${d.oldQ.toFixed(3)})/${d.n} =
            <span class="g">${d.newQ.toFixed(3)}</span></div>`;
      }

      const html = `
        <div class="eq"><span class="dim">episode length</span> ${traj.length} steps,
          return <span class="g">${totalR.toFixed(2)}</span>
          ${reachedGoal ? '<span class="g">(reached goal)</span>' : '<span class="dim">(timed out)</span>'}</div>
        <div class="eq"><span class="dim">first-visit updates this episode:</span> ${updates}</div>
        ${detail}
      `;
      return { html, reward: totalR, episodeEnded: true };
    }
  };
};
