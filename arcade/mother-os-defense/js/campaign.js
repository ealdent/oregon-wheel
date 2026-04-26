"use strict";

const CAMPAIGN_MAP = {
  nodeWidth: 154,
  nodeHeight: 98,
  gridX: 245,
  gridY: 162,
  originX: BOARD.width / 2,
  originY: BOARD.height / 2
};

const CAMPAIGN_DIRECTIONS = [
  { id: "E", dx: 1, dy: 0 },
  { id: "S", dx: 0, dy: 1 },
  { id: "N", dx: 0, dy: -1 },
  { id: "W", dx: -1, dy: 0 }
];

const facilityNamePools = {
  tokamak: {
    sites: ["Tokamak Facility", "Plasma Intake Plant", "Fusion Valve Station"],
    suffixes: ["B", "K", "R-4", "Delta", "C-11", "Morrow"]
  },
  cargo: {
    sites: ["Orbital Cargo Facility", "Aperture Dockyard", "Freight Relay Yard"],
    suffixes: ["13", "22", "C", "Helix", "Orion", "12"]
  },
  foundry: {
    sites: ["Null Foundry", "Mass Driver Array", "Vehicle Plant"],
    suffixes: ["17", "Gamma", "Kestrel", "F", "9", "V"]
  },
  cryo: {
    sites: ["Cryogenic Vault", "Bio-Research Lab", "Hydro Pump Station"],
    suffixes: ["Sigma", "E-6", "Lumen", "04", "7A", "Pike"]
  },
  radar: {
    sites: ["Deep Radar Annex", "Catenary Relay Station", "Subsurface Data Mine"],
    suffixes: ["3", "Iris", "Q", "88", "V", "31"]
  }
};

function currentState() {
  try {
    return state || null;
  } catch (error) {
    return null;
  }
}

function campaignHash(input) {
  let hash = 2166136261;
  const text = String(input);
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function makeCampaignRng(seed) {
  let value = campaignHash(seed || 1);
  return () => {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function campaignRandomInt(rng, min, max) {
  return min + Math.floor(rng() * (max - min + 1));
}

function campaignPick(rng, list) {
  return list[Math.floor(rng() * list.length)] || list[0];
}

function campaignShuffle(rng, list) {
  const copy = list.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function createCampaign() {
  const campaign = {
    version: 1,
    seed: campaignHash(`mother-os-${Date.now()}-${Math.random()}`),
    nextNodeIndex: 2,
    currentNodeId: "F-001",
    selectedNodeId: "F-001",
    mapUnlocked: false,
    nodes: {},
    edges: [],
    pan: { x: 0, y: 0 },
    stats: {
      facilitiesSecured: 0,
      totalKills: 0,
      totalScore: 0,
      totalCreditsEarned: 0,
      totalBossesDefeated: 0
    }
  };
  const start = {
    id: "F-001",
    index: 1,
    facility: "Tokamak Facility B",
    type: "tokamak",
    gridX: 0,
    gridY: 0,
    seed: campaignHash(`${campaign.seed}:start`),
    sectorCount: 7,
    currentSector: 7,
    plannedExitCount: 3,
    visible: true,
    secured: false,
    exitsGenerated: false,
    parentIds: [],
    childIds: [],
    checkpoint: null,
    completedSummary: null
  };
  start.checkpoint = createDefaultCheckpoint(start);
  campaign.nodes[start.id] = start;
  return campaign;
}

function loadCampaign() {
  try {
    const raw = window.localStorage.getItem(CAMPAIGN_STORAGE_KEY);
    if (!raw) return createCampaign();
    const parsed = JSON.parse(raw);
    return normalizeCampaign(parsed);
  } catch (error) {
    return createCampaign();
  }
}

function normalizeCampaign(campaign) {
  if (!campaign || typeof campaign !== "object" || !campaign.nodes) {
    return createCampaign();
  }
  campaign.version = 1;
  campaign.edges = Array.isArray(campaign.edges) ? campaign.edges : [];
  campaign.pan = campaign.pan || { x: 0, y: 0 };
  campaign.stats = {
    facilitiesSecured: 0,
    totalKills: 0,
    totalScore: 0,
    totalCreditsEarned: 0,
    totalBossesDefeated: 0,
    ...(campaign.stats || {})
  };
  let maxIndex = 1;
  for (const node of Object.values(campaign.nodes)) {
    node.id = node.id || `F-${String(node.index || 1).padStart(3, "0")}`;
    node.type = facilityTypes[node.type] ? node.type : "tokamak";
    node.index = node.index || Number(node.id.replace(/\D/g, "")) || 1;
    node.seed = node.seed || campaignHash(`${campaign.seed}:${node.id}`);
    node.sectorCount = node.sectorCount || 7;
    node.currentSector = clamp(node.currentSector || node.sectorCount, 0, node.sectorCount);
    node.plannedExitCount = Number.isFinite(node.plannedExitCount) ? node.plannedExitCount : 2;
    node.visible = node.visible !== false;
    node.secured = !!node.secured;
    node.exitsGenerated = !!node.exitsGenerated;
    node.parentIds = Array.isArray(node.parentIds) ? node.parentIds : [];
    node.childIds = Array.isArray(node.childIds) ? node.childIds : [];
    if (!node.checkpoint && !node.secured) node.checkpoint = createDefaultCheckpoint(node);
    maxIndex = Math.max(maxIndex, node.index);
  }
  campaign.nextNodeIndex = Math.max(campaign.nextNodeIndex || 2, maxIndex + 1);
  campaign.selectedNodeId = campaign.selectedNodeId && campaign.nodes[campaign.selectedNodeId]
    ? campaign.selectedNodeId
    : Object.keys(campaign.nodes)[0];
  campaign.currentNodeId = campaign.currentNodeId && campaign.nodes[campaign.currentNodeId]
    ? campaign.currentNodeId
    : campaign.selectedNodeId;
  return campaign;
}

function saveCampaign(campaign = currentState() && currentState().campaign) {
  if (!campaign) return;
  try {
    window.localStorage.setItem(CAMPAIGN_STORAGE_KEY, JSON.stringify(campaign));
  } catch (error) {
    showToast("Campaign save unavailable");
  }
}

function createDefaultCheckpoint(node) {
  return {
    wave: Math.max(0, node.sectorCount - node.currentSector),
    operation: campaignOperationForNode(node),
    currentSector: node.currentSector,
    credits: STARTING_CREDITS,
    lives: STARTING_LIVES,
    score: 0,
    kills: 0,
    towerCapacity: 18,
    towers: [],
    mines: [],
    runStats: createRunStats(node),
    updatedAt: Date.now()
  };
}

function createRunStats(node) {
  return {
    facilityId: node.id,
    facility: node.facility,
    type: node.type,
    startedAt: Date.now(),
    sectorsCleared: 0,
    wavesStarted: 0,
    kills: 0,
    enemiesLeaked: 0,
    leakDamage: 0,
    creditsEarned: 0,
    creditsSpent: 0,
    creditsRefunded: 0,
    scoreGained: 0,
    towersBuilt: 0,
    towersSold: 0,
    upgrades: 0,
    bossesDefeated: 0
  };
}

function campaignOperationForNode(node) {
  return {
    index: node.index,
    facility: node.facility,
    sectorCount: node.sectorCount,
    sector: Math.max(1, node.currentSector || 1)
  };
}

function cloneData(value) {
  return JSON.parse(JSON.stringify(value));
}

function selectedCampaignNode(campaign = currentState() && currentState().campaign) {
  return campaign && campaign.nodes[campaign.selectedNodeId] ? campaign.nodes[campaign.selectedNodeId] : null;
}

function canEnterCampaignNode(node, campaign = currentState() && currentState().campaign) {
  if (!node || node.secured || !node.visible) return false;
  if (node.id === "F-001") return true;
  return node.parentIds.some((parentId) => campaign.nodes[parentId] && campaign.nodes[parentId].secured);
}

function visibleCampaignNodes(campaign = currentState() && currentState().campaign) {
  if (!campaign) return [];
  return Object.values(campaign.nodes).filter((node) => node.visible);
}

function campaignGridKey(x, y) {
  return `${x},${y}`;
}

function occupiedCampaignCells(campaign) {
  const occupied = new Set();
  for (const node of Object.values(campaign.nodes)) {
    occupied.add(campaignGridKey(node.gridX, node.gridY));
  }
  return occupied;
}

function weightedExitCount(rng, siblingCount = 3) {
  const options = siblingCount >= 3
    ? [
      { value: 0, weight: 0.08 },
      { value: 1, weight: 0.14 },
      { value: 2, weight: 0.52 },
      { value: 3, weight: 0.22 },
      { value: 4, weight: 0.04 }
    ]
    : [
      { value: 1, weight: 0.18 },
      { value: 2, weight: 0.58 },
      { value: 3, weight: 0.2 },
      { value: 4, weight: 0.04 }
    ];
  const total = options.reduce((sum, item) => sum + item.weight, 0);
  let roll = rng() * total;
  for (const item of options) {
    roll -= item.weight;
    if (roll <= 0) return item.value;
  }
  return 2;
}

function planSiblingExitCounts(rng, count) {
  const exits = Array.from({ length: count }, () => weightedExitCount(rng, count));
  let nonZero = exits.filter(Boolean).length;
  if (count >= 3 && exits.includes(0) && nonZero < 2) {
    for (let i = 0; i < exits.length && nonZero < 2; i += 1) {
      if (exits[i] === 0) {
        exits[i] = 2;
        nonZero += 1;
      }
    }
  }
  if (count < 3) {
    for (let i = 0; i < exits.length; i += 1) {
      if (exits[i] === 0) exits[i] = 2;
    }
  }
  return exits;
}

function createGeneratedCampaignNode(campaign, parent, direction, plannedExitCount) {
  const index = campaign.nextNodeIndex++;
  const id = `F-${String(index).padStart(3, "0")}`;
  const seed = campaignHash(`${campaign.seed}:${parent.id}:${direction.id}:${index}`);
  const rng = makeCampaignRng(seed);
  const type = campaignPick(rng, facilityTypeOrder);
  const namePool = facilityNamePools[type] || facilityNamePools.tokamak;
  const facility = `${campaignPick(rng, namePool.sites)} ${campaignPick(rng, namePool.suffixes)}`;
  const sectorBase = campaignRandomInt(rng, 6, 12 + Math.min(5, Math.floor(index / 5)));
  const sectorCount = clamp(sectorBase + (type === "foundry" ? 1 : type === "cargo" ? -1 : 0), 5, 16);
  const node = {
    id,
    index,
    facility,
    type,
    gridX: parent.gridX + direction.dx,
    gridY: parent.gridY + direction.dy,
    seed,
    sectorCount,
    currentSector: sectorCount,
    plannedExitCount,
    visible: true,
    secured: false,
    exitsGenerated: false,
    parentIds: [parent.id],
    childIds: [],
    checkpoint: null,
    completedSummary: null
  };
  node.checkpoint = createDefaultCheckpoint(node);
  return node;
}

function revealCampaignExits(campaign, nodeId) {
  const node = campaign.nodes[nodeId];
  if (!node || node.exitsGenerated) return;
  const rng = makeCampaignRng(`${node.seed}:exits`);
  const occupied = occupiedCampaignCells(campaign);
  const directions = campaignShuffle(rng, CAMPAIGN_DIRECTIONS)
    .filter((direction) => !occupied.has(campaignGridKey(node.gridX + direction.dx, node.gridY + direction.dy)));
  const exitCount = Math.min(node.plannedExitCount, directions.length);
  const siblingExits = planSiblingExitCounts(rng, exitCount);
  for (let i = 0; i < exitCount; i += 1) {
    const child = createGeneratedCampaignNode(campaign, node, directions[i], siblingExits[i]);
    campaign.nodes[child.id] = child;
    node.childIds.push(child.id);
    campaign.edges.push({ from: node.id, to: child.id });
  }
  node.exitsGenerated = true;
}

function campaignNodePosition(node, campaign = currentState() && currentState().campaign) {
  const pan = campaign.pan || { x: 0, y: 0 };
  return {
    x: CAMPAIGN_MAP.originX + node.gridX * CAMPAIGN_MAP.gridX + pan.x,
    y: CAMPAIGN_MAP.originY + node.gridY * CAMPAIGN_MAP.gridY + pan.y
  };
}

function campaignNodeAt(x, y, campaign = currentState() && currentState().campaign) {
  const nodes = visibleCampaignNodes(campaign).sort((a, b) => b.index - a.index);
  for (const node of nodes) {
    const pos = campaignNodePosition(node, campaign);
    if (
      x >= pos.x - CAMPAIGN_MAP.nodeWidth / 2
      && x <= pos.x + CAMPAIGN_MAP.nodeWidth / 2
      && y >= pos.y - CAMPAIGN_MAP.nodeHeight / 2
      && y <= pos.y + CAMPAIGN_MAP.nodeHeight / 2
    ) {
      return node;
    }
  }
  return null;
}

function campaignUnknownExitDirections(campaign, node) {
  if (!node || node.secured || node.exitsGenerated || node.plannedExitCount <= 0) return [];
  const occupied = occupiedCampaignCells(campaign);
  const rng = makeCampaignRng(`${node.seed}:exits`);
  return campaignShuffle(rng, CAMPAIGN_DIRECTIONS)
    .filter((direction) => !occupied.has(campaignGridKey(node.gridX + direction.dx, node.gridY + direction.dy)))
    .slice(0, node.plannedExitCount);
}

function generateFacilityLayout(node) {
  const type = facilityTypes[node.type] || facilityTypes.tokamak;
  return {
    nodeId: node.id,
    type: node.type,
    typeDef: type,
    pathPoints: generateFacilityPath(node.seed),
    schematicSeed: campaignHash(`${node.seed}:schematic`),
    grimeSeed: campaignHash(`${node.seed}:grime`),
    mapSeed: node.seed
  };
}

function generateFacilityPath(seed) {
  const rng = makeCampaignRng(`${seed}:path`);
  const startX = campaignRandomInt(rng, 360, 640);
  const rows = [
    34,
    campaignRandomInt(rng, 112, 142),
    campaignRandomInt(rng, 206, 246),
    campaignRandomInt(rng, 318, 362),
    campaignRandomInt(rng, 438, 486),
    campaignRandomInt(rng, 548, 590),
    646
  ];
  const points = [{ x: startX, y: rows[0] }];
  let x = startX;
  let direction = rng() > 0.5 ? 1 : -1;
  for (let i = 1; i < rows.length; i += 1) {
    points.push({ x, y: rows[i] });
    const remaining = rows.length - i - 1;
    if (remaining <= 0) break;
    const minTravel = 205 + rng() * 70;
    const nextX = clamp(x + direction * minTravel, 150, 850);
    x = nextX;
    points.push({ x, y: rows[i] });
    if (x < 230) direction = 1;
    else if (x > 770) direction = -1;
    else direction *= -1;
  }
  return points;
}

function buildFacilityRunState(campaign, node) {
  const previousState = currentState();
  const checkpoint = node.checkpoint || createDefaultCheckpoint(node);
  checkpoint.towers = checkpoint.towers || [];
  checkpoint.mines = checkpoint.mines || [];
  checkpoint.credits = Number.isFinite(checkpoint.credits) ? checkpoint.credits : STARTING_CREDITS;
  checkpoint.lives = Number.isFinite(checkpoint.lives) ? checkpoint.lives : STARTING_LIVES;
  checkpoint.score = Number.isFinite(checkpoint.score) ? checkpoint.score : 0;
  node.checkpoint = checkpoint;
  const layout = generateFacilityLayout(node);
  applyFacilityLayout(layout);
  const operation = cloneOperation(checkpoint.operation || campaignOperationForNode(node));
  node.currentSector = checkpoint.currentSector || operation.sector;
  nextTowerId = Math.max(1, ...checkpoint.towers.map((tower) => tower.id + 1), 1);
  nextMineId = Math.max(1, ...checkpoint.mines.map((mine) => mine.id + 1), 1);
  nextEnemyId = 1;
  return {
    mode: "facility",
    campaign,
    facilityNodeId: node.id,
    facilityLayout: layout,
    facilityType: node.type,
    wave: checkpoint.wave || 0,
    phase: "planning",
    credits: checkpoint.credits,
    lives: checkpoint.lives,
    score: checkpoint.score,
    towerCapacity: checkpoint.towerCapacity || 18,
    towers: cloneData(checkpoint.towers || []),
    enemies: [],
    projectiles: [],
    mines: cloneData(checkpoint.mines || []),
    effects: [],
    particles: [],
    selectedTowerId: null,
    placingType: null,
    paused: false,
    speedIndex: previousState ? previousState.speedIndex : 0,
    autoAdvance: false,
    autoStartTimer: 0,
    sound: previousState ? previousState.sound : true,
    operationIndex: node.index,
    operation,
    spawnQueue: [],
    spawnTimer: 0,
    wavePlan: null,
    waveSpawned: 0,
    waveResolved: 0,
    kills: checkpoint.kills || 0,
    runStats: cloneData(checkpoint.runStats || createRunStats(node)),
    summary: null,
    gameOver: false,
    logs: [
      `${operationLabel(operation)} indexed.`,
      `${facilityTypes[node.type].label} modifiers loaded.`,
      "Build protocol active."
    ]
  };
}

function buildCampaignMapState(campaign, selectedNodeId = campaign.selectedNodeId) {
  const previousState = currentState();
  const firstNodeId = Object.keys(campaign.nodes)[0];
  const selectedId = campaign.nodes[selectedNodeId] ? selectedNodeId : campaign.nodes[campaign.selectedNodeId] ? campaign.selectedNodeId : firstNodeId;
  campaign.selectedNodeId = selectedId;
  return {
    mode: "campaign",
    campaign,
    facilityNodeId: null,
    facilityLayout: null,
    facilityType: null,
    wave: 0,
    phase: "campaign",
    credits: 0,
    lives: 0,
    score: campaign.stats.totalScore || 0,
    towerCapacity: 0,
    towers: [],
    enemies: [],
    projectiles: [],
    mines: [],
    effects: [],
    particles: [],
    selectedTowerId: null,
    placingType: null,
    paused: false,
    speedIndex: previousState ? previousState.speedIndex : 0,
    autoAdvance: false,
    autoStartTimer: 0,
    sound: previousState ? previousState.sound : true,
    operationIndex: 0,
    operation: campaignOperationForNode(campaign.nodes[selectedId]),
    spawnQueue: [],
    spawnTimer: 0,
    wavePlan: null,
    waveSpawned: 0,
    waveResolved: 0,
    kills: campaign.stats.totalKills || 0,
    runStats: null,
    summary: previousState ? previousState.summary : null,
    gameOver: false,
    logs: ["Campaign map online.", "Select an available facility."]
  };
}

function enterCampaignMap(selectedNodeId = state && state.campaign ? state.campaign.selectedNodeId : null) {
  const campaign = state.campaign;
  if (!campaign) return;
  if (selectedNodeId && campaign.nodes[selectedNodeId]) campaign.selectedNodeId = selectedNodeId;
  state = buildCampaignMapState(campaign, campaign.selectedNodeId);
  resetRenderKeys();
  saveCampaign(campaign);
}

function selectCampaignNode(nodeId) {
  if (!state.campaign || !state.campaign.nodes[nodeId]) return;
  state.campaign.selectedNodeId = nodeId;
  state.operation = campaignOperationForNode(state.campaign.nodes[nodeId]);
  resetRenderKeys();
  saveCampaign(state.campaign);
  playSound("click");
}

function startSelectedCampaignFacility() {
  const node = selectedCampaignNode();
  if (!node) {
    showToast("Select a facility");
    return;
  }
  if (node.secured) {
    showToast("Facility already secured");
    return;
  }
  if (!canEnterCampaignNode(node)) {
    showToast("Route not connected");
    return;
  }
  state.campaign.currentNodeId = node.id;
  state.campaign.selectedNodeId = node.id;
  state = buildFacilityRunState(state.campaign, node);
  resetRenderKeys();
  saveCampaign(state.campaign);
  showToast(`${node.facility} entered`);
}

function exitFacilityToCampaign() {
  if (!state.campaign) return;
  const nodeId = state.facilityNodeId || state.campaign.selectedNodeId;
  state.campaign.selectedNodeId = nodeId;
  state.campaign.currentNodeId = null;
  enterCampaignMap(nodeId);
  showToast("Facility run suspended");
}

function createCheckpointFromState() {
  return {
    wave: state.wave,
    operation: cloneOperation(state.operation),
    currentSector: state.operation.sector,
    credits: state.credits,
    lives: state.lives,
    score: state.score,
    kills: state.kills,
    towerCapacity: state.towerCapacity,
    towers: state.towers.map((tower) => ({
      id: tower.id,
      type: tower.type,
      x: tower.x,
      y: tower.y,
      level: tower.level,
      cooldown: Math.min(tower.cooldown || 0, 0.25),
      targetMode: tower.targetMode,
      spent: tower.spent,
      refundable: tower.refundable,
      pulse: tower.pulse || 0
    })),
    mines: state.mines.map((mine) => ({ ...mine })),
    runStats: cloneData(state.runStats || createRunStats(state.campaign.nodes[state.facilityNodeId])),
    updatedAt: Date.now()
  };
}

function captureFacilityCheckpoint() {
  if (!state.campaign || !state.facilityNodeId) return;
  const node = state.campaign.nodes[state.facilityNodeId];
  node.currentSector = state.operation.sector;
  node.checkpoint = createCheckpointFromState();
  saveCampaign(state.campaign);
}

function addRunStat(field, amount = 1) {
  if (!state.runStats) return;
  state.runStats[field] = (state.runStats[field] || 0) + amount;
}

function advanceCampaignFacilityAfterClear(clearedPlan) {
  const node = state.campaign.nodes[state.facilityNodeId];
  if (!node) return;
  addRunStat("sectorsCleared", 1);
  if (clearedPlan && clearedPlan.bossWave) {
    addRunStat("bossesDefeated", 1);
    node.currentSector = 0;
    node.secured = true;
    node.completedSummary = buildFacilitySummary(node, state.runStats);
    node.checkpoint = null;
    state.campaign.mapUnlocked = true;
    state.campaign.currentNodeId = null;
    state.campaign.selectedNodeId = node.id;
    state.campaign.stats.facilitiesSecured += 1;
    state.campaign.stats.totalKills += state.runStats.kills || 0;
    state.campaign.stats.totalScore += state.runStats.scoreGained || 0;
    state.campaign.stats.totalCreditsEarned += state.runStats.creditsEarned || 0;
    state.campaign.stats.totalBossesDefeated += state.runStats.bossesDefeated || 0;
    revealCampaignExits(state.campaign, node.id);
    state.autoAdvance = false;
    state.autoStartTimer = 0;
    state.summary = node.completedSummary;
    state.mode = "campaign";
    state.phase = "campaign";
    state.enemies = [];
    state.projectiles = [];
    state.spawnQueue = [];
    state.wavePlan = null;
    saveCampaign(state.campaign);
    resetRenderKeys();
    return;
  }
  state.operation = {
    ...state.operation,
    sector: Math.max(1, state.operation.sector - 1)
  };
  node.currentSector = state.operation.sector;
  captureFacilityCheckpoint();
}

function buildFacilitySummary(node, runStats) {
  const type = facilityTypes[node.type] || facilityTypes.tokamak;
  return {
    title: `${node.facility} Secured`,
    subtitle: `${type.label} / ${node.sectorCount} sectors neutralized`,
    stats: [
      ["Enemies killed", runStats.kills || 0],
      ["Bosses defeated", runStats.bossesDefeated || 0],
      ["Credits earned", `$${fmt(runStats.creditsEarned || 0)}`],
      ["Credits spent", `$${fmt(runStats.creditsSpent || 0)}`],
      ["Credits recovered", `$${fmt(runStats.creditsRefunded || 0)}`],
      ["Score gained", fmt(runStats.scoreGained || 0)],
      ["Enemies leaked", runStats.enemiesLeaked || 0],
      ["Core damage", runStats.leakDamage || 0],
      ["Towers built", runStats.towersBuilt || 0],
      ["Upgrades installed", runStats.upgrades || 0]
    ]
  };
}

function dismissFacilitySummary() {
  if (!state.campaign) return;
  const selectedId = state.facilityNodeId || state.campaign.selectedNodeId;
  state.summary = null;
  enterCampaignMap(selectedId);
}

function resetRenderKeys() {
  sideTacticsRenderKey = "";
  inspectorRenderKey = "";
  logRenderKey = "";
  staticWorldLayer = null;
}

function activeFacilityTypeDef() {
  const key = state && state.facilityType ? state.facilityType : "tokamak";
  return facilityTypes[key] || facilityTypes.tokamak;
}
