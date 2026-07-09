const rows = 10;
const cols = 10;
const tile = 44;
const MONSTER_TICK_MS = 350;
const actions = ["up", "down", "left", "right"];
const deltas = {
  up: [-1, 0],
  down: [1, 0],
  left: [0, -1],
  right: [0, 1],
};

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const resetBtn = document.getElementById("resetBtn");
const policySelect = document.getElementById("policySelect");
const policyDescription = document.getElementById("policyDescription");
const debugToggle = document.getElementById("debugToggle");
const debugPanel = document.getElementById("debugPanel");
const epsilonSlider = document.getElementById("epsilonSlider");
const epsilonValue = document.getElementById("epsilonValue");
const dbgState = document.getElementById("dbgState");
const dbgPolicyAction = document.getElementById("dbgPolicyAction");
const dbgExploitAction = document.getElementById("dbgExploitAction");
const dbgChosenAction = document.getElementById("dbgChosenAction");
const dbgDecisionMode = document.getElementById("dbgDecisionMode");
const dbgActionSource = document.getElementById("dbgActionSource");
const dbgValidMoves = document.getElementById("dbgValidMoves");
const statExploit = document.getElementById("statExploit");
const statExplore = document.getElementById("statExplore");
const statPolicyHits = document.getElementById("statPolicyHits");
const statFallback = document.getElementById("statFallback");
const decisionLog = document.getElementById("decisionLog");

function getPolicyBase() {
  const cfg = window.APP_CONFIG ?? {};
  if (cfg.policyBase) {
    return cfg.policyBase.replace(/\/$/, "");
  }
  if (cfg.hfUsername && cfg.hfModelName) {
    return `https://huggingface.co/${cfg.hfUsername}/${cfg.hfModelName}/resolve/main`;
  }
  return "../trained_policies";
}

const POLICY_BASE = getPolicyBase();

const POLICY_SOURCES = {
  online: {
    path: `${POLICY_BASE}/monster_policy_reinforce.json`,
    label: "Online policy gradient",
  },
  td: {
    path: `${POLICY_BASE}/monster_policy_sarsa.json`,
    label: "TD SARSA",
  },
  greedy: { path: null, label: "Greedy chase" },
};

const POLICY_DESCRIPTIONS = {
  online: {
    title: "Online policy gradient",
    body: "A learned lookup table mapping <code>state_key</code> → <code>action</code>. In unseen states it falls back to greedy chase.",
    good_for: "Demonstrating direct policy learning and how sparse state coverage causes policy misses.",
  },
  td: {
    title: "TD SARSA",
    body: "A learned lookup table derived from TD control (SARSA). Chooses the action that maximizes learned action-values for seen states; otherwise falls back to greedy chase.",
    good_for: "Comparing TD bootstrapping behavior vs policy-gradient behavior.",
  },
  greedy: {
    title: "Greedy chase (no policy)",
    body: "Not learned. The monster always picks a move that reduces Manhattan distance to the player (subject to barriers).",
    good_for: "A baseline that is easy to explain and debug.",
  },
};

let policy = {};
let policyMeta = null;
let barriers = new Set();
let player = [0, 0];
let monster = [0, 0];
let goal = [0, 0];
let keyPos = [0, 0];
let hasKey = false;
let done = false;
let lastDecision = null;
let monsterTimer = null;
const episodeStats = {
  exploit: 0,
  explore: 0,
  policyHits: 0,
  fallback: 0,
};
const MAX_LOG_ENTRIES = 12;

canvas.width = cols * tile;
canvas.height = rows * tile;

function setPolicyDescription(sourceKey, meta) {
  const desc = POLICY_DESCRIPTIONS[sourceKey] || POLICY_DESCRIPTIONS.online;
  const metaNotes = meta?.notes ? ` <span style="color:#9aa3b2">(notes: ${meta.notes})</span>` : "";
  policyDescription.innerHTML = `<strong>${desc.title}</strong>: ${desc.body} ${metaNotes}<br/><span style="color:#9aa3b2">Good for:</span> ${desc.good_for}`;
}

function key(pos) {
  return `${pos[0]},${pos[1]}`;
}

function stateKey() {
  return `p=${player[0]},${player[1]}|m=${monster[0]},${monster[1]}|g=${goal[0]},${goal[1]}`;
}

function inBounds([r, c]) {
  return r >= 0 && r < rows && c >= 0 && c < cols;
}

function samePos(a, b) {
  return a[0] === b[0] && a[1] === b[1];
}

function move(pos, action) {
  const [dr, dc] = deltas[action];
  const next = [pos[0] + dr, pos[1] + dc];
  if (!inBounds(next)) return pos;
  if (barriers.has(key(next))) return pos;
  return next;
}

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function manhattan(a, b) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

function validMoves(pos) {
  return actions.filter((a) => {
    const next = move(pos, a);
    return next[0] !== pos[0] || next[1] !== pos[1];
  });
}

function greedyChaseAction() {
  const moves = validMoves(monster);
  if (!moves.length) return randomChoice(actions);

  let bestAction = moves[0];
  let bestDist = manhattan(move(monster, bestAction), player);
  for (const action of moves) {
    const dist = manhattan(move(monster, action), player);
    if (dist < bestDist) {
      bestDist = dist;
      bestAction = action;
    }
  }
  return bestAction;
}

function resetEpisodeStats() {
  episodeStats.exploit = 0;
  episodeStats.explore = 0;
  episodeStats.policyHits = 0;
  episodeStats.fallback = 0;
  lastDecision = null;
  decisionLog.innerHTML = "";
  updateDebugPanel();
}

function getEpsilon() {
  return Number(epsilonSlider.value) / 100;
}

function chooseMonsterAction() {
  const state = stateKey();
  const moves = validMoves(monster);
  const policyAction = policy[state];
  const hasPolicyHit = Boolean(policyAction && actions.includes(policyAction));
  const exploitAction = hasPolicyHit ? policyAction : greedyChaseAction();
  const actionSource = hasPolicyHit ? "policy_table" : "greedy_fallback";
  const exploring = Math.random() < getEpsilon();

  let chosenAction;
  let decisionMode;
  if (exploring) {
    chosenAction = moves.length ? randomChoice(moves) : randomChoice(actions);
    decisionMode = "explore";
    episodeStats.explore += 1;
  } else {
    chosenAction = exploitAction;
    decisionMode = "exploit";
    episodeStats.exploit += 1;
    if (hasPolicyHit) episodeStats.policyHits += 1;
    else episodeStats.fallback += 1;
  }

  return {
    state,
    policyAction: hasPolicyHit ? policyAction : null,
    exploitAction,
    chosenAction,
    decisionMode,
    actionSource,
    validMoves: moves,
    exploring,
  };
}

function updateDebugPanel() {
  if (!debugToggle.checked) return;

  if (lastDecision) {
    dbgState.textContent = lastDecision.state;
    dbgPolicyAction.textContent = lastDecision.policyAction ?? "(missing)";
    dbgExploitAction.textContent = lastDecision.exploitAction;
    dbgChosenAction.textContent = lastDecision.chosenAction;
    dbgDecisionMode.textContent = lastDecision.decisionMode;
    dbgDecisionMode.className = lastDecision.decisionMode === "explore" ? "mode-explore" : "mode-exploit";
    dbgActionSource.textContent = lastDecision.actionSource;
    dbgValidMoves.textContent = lastDecision.validMoves.join(", ") || "(none)";
  } else {
    dbgState.textContent = stateKey();
    dbgPolicyAction.textContent = "—";
    dbgExploitAction.textContent = "—";
    dbgChosenAction.textContent = "—";
    dbgDecisionMode.textContent = "—";
    dbgDecisionMode.className = "";
    dbgActionSource.textContent = "—";
    dbgValidMoves.textContent = validMoves(monster).join(", ") || "(none)";
  }

  statExploit.textContent = String(episodeStats.exploit);
  statExplore.textContent = String(episodeStats.explore);
  statPolicyHits.textContent = String(episodeStats.policyHits);
  statFallback.textContent = String(episodeStats.fallback);
}

function logDecision(decision) {
  if (!debugToggle.checked) return;

  const item = document.createElement("li");
  const modeClass = decision.decisionMode === "explore" ? "mode-explore" : "mode-exploit";
  item.innerHTML = `<span class="${modeClass}">${decision.decisionMode}</span> → ${decision.chosenAction} <span style="color:#9aa3b2">(${decision.actionSource})</span>`;
  decisionLog.prepend(item);

  while (decisionLog.children.length > MAX_LOG_ENTRIES) {
    decisionLog.removeChild(decisionLog.lastChild);
  }
}

function randomizeWorld() {
  barriers = new Set();
  const count = 24;
  while (barriers.size < count) {
    const p = [Math.floor(Math.random() * rows), Math.floor(Math.random() * cols)];
    barriers.add(key(p));
  }

  const free = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!barriers.has(`${r},${c}`)) free.push([r, c]);
    }
  }

  const take = () => free.splice(Math.floor(Math.random() * free.length), 1)[0];
  player = take();
  monster = take();
  goal = take();
  keyPos = take();
  hasKey = false;
  done = false;
  resetEpisodeStats();
}

function stepMonster() {
  const decision = chooseMonsterAction();
  lastDecision = decision;
  monster = move(monster, decision.chosenAction);
  logDecision(decision);
  updateDebugPanel();

  if (samePos(monster, player)) {
    done = true;
    statusEl.textContent = "Game over: monster wins";
  }
}

function checkPlayerGoal() {
  if (samePos(player, goal) && hasKey) {
    done = true;
    statusEl.textContent = "Game over: player wins";
  }
}

function checkKeyPickup() {
  if (!hasKey && samePos(player, keyPos)) {
    hasKey = true;
    statusEl.textContent = "Key collected! Door unlocked.";
  }
}

function drawCell(r, c, color) {
  ctx.fillStyle = color;
  ctx.fillRect(c * tile + 1, r * tile + 1, tile - 2, tile - 2);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      drawCell(r, c, "#2d3340");
    }
  }

  barriers.forEach((p) => {
    const [r, c] = p.split(",").map(Number);
    drawCell(r, c, "#5a6478");
  });

  drawCell(goal[0], goal[1], hasKey ? "#32c766" : "#c9892a");
  if (!hasKey) {
    drawCell(keyPos[0], keyPos[1], "#f4d03f");
  }
  drawCell(player[0], player[1], "#42a5ff");
  drawCell(monster[0], monster[1], "#ff4d5a");
}

function onPlayerMove(action) {
  if (done) return;
  const nextPlayer = move(player, action);
  if (!hasKey && samePos(nextPlayer, goal)) {
    statusEl.textContent = "Door is locked. Find the key first.";
    draw();
    return;
  }

  player = nextPlayer;
  checkKeyPickup();
  checkPlayerGoal();
  updateDebugPanel();
  draw();
}

function startMonsterLoop() {
  if (monsterTimer !== null) {
    clearInterval(monsterTimer);
  }

  monsterTimer = setInterval(() => {
    if (done) return;
    stepMonster();
    draw();
  }, MONSTER_TICK_MS);
}

function mapKeyToAction(evtKey) {
  const k = evtKey.toLowerCase();
  if (k === "arrowup" || k === "w") return "up";
  if (k === "arrowdown" || k === "s") return "down";
  if (k === "arrowleft" || k === "a") return "left";
  if (k === "arrowright" || k === "d") return "right";
  return null;
}

async function loadPolicy(sourceKey) {
  const source = POLICY_SOURCES[sourceKey] || POLICY_SOURCES.online;

  if (!source.path) {
    policy = {};
    policyMeta = null;
    statusEl.textContent = `${source.label} — no policy file.`;
    setPolicyDescription(sourceKey, null);
    return;
  }

  try {
    const res = await fetch(source.path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    policy = data.policy || {};
    policyMeta = data.meta || null;
    const algo = policyMeta?.algo || source.label;
    statusEl.textContent = `Loaded ${algo} (${Object.keys(policy).length} states). Find the key, then open the door.`;
    setPolicyDescription(sourceKey, policyMeta);
  } catch (err) {
    policy = {};
    policyMeta = null;
    statusEl.textContent = `Failed to load ${source.label}. Using greedy chase fallback.`;
    setPolicyDescription(sourceKey, null);
  }
}

window.addEventListener("keydown", (e) => {
  const action = mapKeyToAction(e.key);
  if (action) onPlayerMove(action);
});

resetBtn.addEventListener("click", () => {
  randomizeWorld();
  draw();
});

policySelect.addEventListener("change", async () => {
  await loadPolicy(policySelect.value);
  randomizeWorld();
  draw();
});

debugToggle.addEventListener("change", () => {
  debugPanel.hidden = !debugToggle.checked;
  updateDebugPanel();
});

epsilonSlider.addEventListener("input", () => {
  epsilonValue.textContent = getEpsilon().toFixed(2);
});

async function init() {
  epsilonValue.textContent = getEpsilon().toFixed(2);
  debugPanel.hidden = !debugToggle.checked;
  await loadPolicy(policySelect.value);
  randomizeWorld();
  startMonsterLoop();
  draw();
}

init();
