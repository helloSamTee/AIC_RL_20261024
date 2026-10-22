/* =========================================================================
 * DYNAMIC PROGRAMMING  (model-BASED value iteration)
 * -------------------------------------------------------------------------
 * Idea: Dynamic Programming is the "cheat mode" of the three. There is NO
 * wandering agent and NO sampling — because we KNOW the full MDP model (the
 * transitions and rewards), we can compute the optimal value of every state
 * directly with the Bellman optimality equation, sweeping over all states:
 *
 *   V(s) ← max_a [ r(s,a) + γ·V(s') ]
 *
 * The unit of work ("step") is ONE FULL SWEEP over every cell in the grid.
 * The policy is read off afterwards as the action that achieves that max.
 * This contrasts with the model-free methods (MC, Q-learning) that must learn
 * from experience because they do not have the model.
 * ======================================================================= */
window.RLAlgorithms = window.RLAlgorithms || {};

window.RLAlgorithms["dynamicprogramming"] = function (env) {
  const { N, ACTIONS } = env;
  let V, sweeps;

  function fresh() {
    V = Array.from({ length: N }, () => Array.from({ length: N }, () => 0));
    sweeps = 0;
  }
  fresh();

  // value of taking action a from (x,y) under the known model: r + γ·V(s')
  function actionValue(x, y, a, gamma) {
    const { nx, ny, reward } = env.transition(x, y, a);
    return reward + gamma * V[ny][nx];
  }
  function bestAction(x, y, gamma) {
    let best = 0, bestVal = -Infinity;
    for (let a = 0; a < 4; a++) {
      const v = actionValue(x, y, a, gamma);
      if (v > bestVal) { bestVal = v; best = a; }
    }
    return { best, bestVal };
  }

  return {
    label: "Dynamic Programming",
    tagline: "Model is known — compute optimal values directly by sweeping.",
    formula: 'V(s) ← max<sub>a</sub> [ r(s,a) + γ·V(s′) ]',
    rewardNote: "r = +1 goal · −0.01 step · −0.20 hit wall",
    stepLabel: "▶ One sweep",
    episodeLabel: "⏭ ×5 sweeps",
    usesAlpha: false,
    usesEpsilon: false,
    showsQ: false,            // DP tracks V(s), not per-action Q in the UI
    hasAgent: false,

    reset: fresh,
    value: (x, y) => V[y][x],
    q: () => null,
    policy: (x, y) => {
      // only show an arrow once values have started to form
      if (V.flat().every(v => Math.abs(v) < 1e-9)) return null;
      return bestAction(x, y, env.gamma()).best;
    },
    agentPos: null,

    /* one full sweep of the Bellman optimality backup over all states */
    step(params) {
      const { gamma } = params;
      const newV = V.map(row => row.slice());
      let maxDelta = 0;
      let sample = null;

      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (env.isWall(x, y)) continue;
        if (env.isGoal(x, y)) { newV[y][x] = 0; continue; } // terminal: value 0

        const { best, bestVal } = bestAction(x, y, gamma);
        const delta = Math.abs(bestVal - V[y][x]);
        if (delta > maxDelta) maxDelta = delta;
        newV[y][x] = bestVal;

        // capture the cell that changed the most, to display its backup
        if (!sample || delta >= sample.delta) {
          const { nx, ny, reward } = env.transition(x, y, best);
          sample = { x, y, a: best, reward, nextV: V[ny][nx], newVal: bestVal, delta };
        }
      }

      V = newV;
      sweeps++;

      let detail = "";
      if (sample) {
        const aName = ACTIONS[sample.a].name;
        detail = `
          <div class="eq"><span class="dim">biggest change at</span> s=<span class="b">(${sample.x},${sample.y})</span>,
            best action <span class="a">${aName}</span></div>
          <div class="eq result">V(s) ← r + γ·V(s′) =
            ${sample.reward.toFixed(2)} + ${gamma.toFixed(2)}·${sample.nextV.toFixed(3)} =
            <span class="g">${sample.newVal.toFixed(3)}</span></div>`;
      }

      const html = `
        <div class="eq"><span class="dim">sweep #${sweeps}</span> over all ${N * N} cells
          (Bellman optimality backup)</div>
        <div class="eq"><span class="dim">largest value change this sweep:</span>
          <b>${maxDelta.toFixed(4)}</b> ${maxDelta < 1e-3 ? '<span class="g">(converged)</span>' : ''}</div>
        ${detail}
      `;
      // "episodeEnded" lets the shared loop count sweeps like episodes
      return { html, reward: null, episodeEnded: true, converged: maxDelta < 1e-3 };
    }
  };
};
