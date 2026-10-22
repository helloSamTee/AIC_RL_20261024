/* =========================================================================
 * Q-LEARNING  (model-free, temporal-difference, off-policy control)
 * -------------------------------------------------------------------------
 * Idea: the agent wanders the maze. After every single move it immediately
 * nudges Q(s,a) toward a "target" built from the reward it just got plus the
 * best value it currently believes is reachable from the next state.
 *
 *   Q_new(s,a) = Q_old(s,a) + α · [ r + γ·max_a' Q_old(s',a') − Q_old(s,a) ]
 *
 * The unit of work ("step") is ONE environment transition.
 * ======================================================================= */
window.RLAlgorithms = window.RLAlgorithms || {};

window.RLAlgorithms["qlearning"] = function (env) {
  const { N, ACTIONS } = env;
  let Q, agent;

  function fresh() {
    Q = Array.from({ length: N }, () =>
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
    label: "Q-learning",
    tagline: "Model-free TD control — learns online, one move at a time.",
    formula:
      'Q<sub>new</sub>(s,a) = Q<sub>old</sub>(s,a) + α·[ r + γ·max Q<sub>old</sub>(s′) − Q<sub>old</sub>(s,a) ]',
    rewardNote: "r = +1 goal · −0.01 step · −0.20 hit wall",
    stepLabel: "▶ Train step",
    episodeLabel: "⏭ Episode",
    usesAlpha: true,
    usesEpsilon: true,
    showsQ: true,
    hasAgent: true,

    reset: fresh,
    value: (x, y) => maxQ(x, y),
    q: (x, y) => Q[y][x],
    policy: (x, y) => (Q[y][x].some(v => Math.abs(v) > 1e-6) ? argmaxQ(x, y) : null),
    get agentPos() { return { x: agent.x, y: agent.y }; },

    /* one environment transition + one TD update */
    step(params) {
      const { alpha, gamma, epsilon } = params;
      const s = { x: agent.x, y: agent.y };
      const a = chooseAction(s.x, s.y, epsilon);
      const { nx, ny, reward, done } = env.transition(s.x, s.y, a);

      const oldQ = Q[s.y][s.x][a];
      const futureBest = done ? 0 : maxQ(nx, ny);
      const target = reward + gamma * futureBest;
      const tdError = target - oldQ;
      const newQ = oldQ + alpha * tdError;
      Q[s.y][s.x][a] = newQ;

      agent = { x: nx, y: ny };
      const episodeEnded = done;
      if (done) agent = { x: env.start.x, y: env.start.y };

      const aName = ACTIONS[a].name;
      const html = `
        <div class="eq"><span class="dim">at</span> s=<span class="b">(${s.x},${s.y})</span>
          <span class="dim">take</span> <span class="a">${aName}</span>
          <span class="dim">→</span> s′=<span class="b">(${nx},${ny})</span>,
          r=<span class="g">${reward.toFixed(2)}</span>${done ? ' <span class="g">(goal!)</span>' : ''}</div>
        <div class="eq"><span class="dim">target =</span> r + γ·max Q<sub>old</sub>(s′) = ${reward.toFixed(2)} + ${gamma.toFixed(2)}·${futureBest.toFixed(2)} = <b>${target.toFixed(3)}</b></div>
        <div class="eq"><span class="dim">error&nbsp; =</span> target − Q<sub>old</sub>(s,<span class="a">${aName}</span>) = ${target.toFixed(3)} − ${oldQ.toFixed(3)} = <span class="a">${tdError.toFixed(3)}</span></div>
        <div class="eq result">Q<sub>new</sub>(s,<span class="a">${aName}</span>) = Q<sub>old</sub> + α·error = ${oldQ.toFixed(3)} + ${alpha.toFixed(2)}·(${tdError.toFixed(3)}) = <span class="g">${newQ.toFixed(3)}</span></div>
      `;
      return { html, reward, episodeEnded };
    }
  };
};
