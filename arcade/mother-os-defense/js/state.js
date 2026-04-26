"use strict";

const els = {
  field: document.getElementById("field"),
  towerList: document.getElementById("towerList"),
  sideTactics: document.getElementById("sideTactics"),
  toast: document.getElementById("toast"),
  clock: document.getElementById("clock"),
  linkStatus: document.getElementById("linkStatus"),
  startWave: document.getElementById("startWave"),
  pauseBtn: document.getElementById("pauseBtn"),
  speedBtn: document.getElementById("speedBtn"),
  autoBtn: document.getElementById("autoBtn"),
  soundBtn: document.getElementById("soundBtn"),
  mapBtn: document.getElementById("mapBtn"),
  upgradeBtn: document.getElementById("upgradeBtn"),
  targetBtn: document.getElementById("targetBtn"),
  sellBtn: document.getElementById("sellBtn"),
  resourceTitle: document.getElementById("resourceTitle"),
  creditsLabel: document.getElementById("creditsLabel"),
  livesLabel: document.getElementById("livesLabel"),
  scoreLabel: document.getElementById("scoreLabel"),
  inspectorTitle: document.getElementById("inspectorTitle"),
  inspectorBody: document.getElementById("inspectorBody"),
  capacityReadout: document.getElementById("capacityReadout"),
  capacityBar: document.getElementById("capacityBar"),
  creditsReadout: document.getElementById("creditsReadout"),
  livesReadout: document.getElementById("livesReadout"),
  scoreReadout: document.getElementById("scoreReadout"),
  phaseReadout: document.getElementById("phaseReadout"),
  activeReadout: document.getElementById("activeReadout"),
  threatReadout: document.getElementById("threatReadout"),
  waveReadout: document.getElementById("waveReadout"),
  waveBar: document.getElementById("waveBar"),
  logList: document.getElementById("logList"),
  summaryModal: document.getElementById("summaryModal"),
  summaryTitle: document.getElementById("summaryTitle"),
  summarySubtitle: document.getElementById("summarySubtitle"),
  summaryBody: document.getElementById("summaryBody"),
  summaryContinue: document.getElementById("summaryContinue")
};

const ctx = els.field.getContext("2d");
const towerById = Object.fromEntries(towerDefs.map((tower) => [tower.id, tower]));
let nextTowerId = 1;
let nextEnemyId = 1;
let nextMineId = 1;
let audioContext = null;
let lastShotSound = 0;
let view = { scale: 1, offsetX: 0, offsetY: 0, cssWidth: 1000, cssHeight: 680 };
let hover = { x: 455, y: 145, valid: true, reason: "Build ready" };
let toastTimer = 0;
let lastFrame = performance.now();
let uiTimer = 0;
let resizeObserver = null;
let staticWorldLayer = null;

let path = buildPath(pathPoints);
let schematicBlocks = buildSchematicBlocks();
let grimeMarks = buildGrimeMarks();
const STATIC_WORLD_PIXEL_SCALE = 2;
const towerSpriteCache = new Map();
const enemySpriteCache = new Map();
const TOWER_SPRITE_PIXEL_SCALE = 2;
const TOWER_PULSE_BANDS = 16;
const TOWER_SPRITE_CACHE_LIMIT = 128;
const phasedEnemyTypes = new Set(["worm", "wisp", "leech"]);
let sideTacticsRenderKey = "";
let inspectorRenderKey = "";
let logRenderKey = "";

let state = makeInitialState();

function makeInitialState() {
  const campaign = loadCampaign();
  const startNode = campaign.nodes[campaign.currentNodeId] || campaign.nodes[campaign.selectedNodeId] || campaign.nodes["F-001"];
  if (campaign.mapUnlocked) {
    return buildCampaignMapState(campaign, campaign.selectedNodeId || startNode.id);
  }
  return buildFacilityRunState(campaign, startNode);
}

function buildPath(points) {
  const segments = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    segments.push({
      a,
      b,
      dx,
      dy,
      length,
      start: total,
      angle: Math.atan2(dy, dx)
    });
    total += length;
  }
  return { points, segments, total };
}

function buildSchematicBlocks(seed = 76013) {
  let value = seed >>> 0;
  const blocks = [];
  const random = () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
  for (let i = 0; i < 86; i += 1) {
    const w = 18 + random() * 72;
    const h = 8 + random() * 42;
    blocks.push({
      x: 44 + random() * (BOARD.width - 110),
      y: 46 + random() * (BOARD.height - 110),
      w,
      h,
      alpha: 0.035 + random() * 0.055
    });
  }
  return blocks;
}

function buildGrimeMarks(seed = 76077) {
  let value = seed >>> 0;
  const marks = [];
  const random = () => {
    value = (value * 1103515245 + 12345) >>> 0;
    return value / 4294967296;
  };
  for (let i = 0; i < 115; i += 1) {
    marks.push({
      x: random() * BOARD.width,
      y: random() * BOARD.height,
      w: 10 + random() * 86,
      h: 1 + random() * 6,
      a: 0.025 + random() * 0.065,
      rot: (random() - 0.5) * 0.22
    });
  }
  return marks;
}

function applyFacilityLayout(layout) {
  pathPoints = (layout && layout.pathPoints ? layout.pathPoints : DEFAULT_PATH_POINTS).map((point) => ({ ...point }));
  path = buildPath(pathPoints);
  schematicBlocks = buildSchematicBlocks(layout ? layout.schematicSeed : 76013);
  grimeMarks = buildGrimeMarks(layout ? layout.grimeSeed : 76077);
  staticWorldLayer = null;
}

function resizeCanvas() {
  const rect = els.field.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  els.field.width = Math.floor(width * dpr);
  els.field.height = Math.floor(height * dpr);
  view.cssWidth = width;
  view.cssHeight = height;
  view.scale = Math.min(width / BOARD.width, height / BOARD.height);
  view.offsetX = (width - BOARD.width * view.scale) / 2;
  view.offsetY = (height - BOARD.height * view.scale) / 2;
  const fieldPanel = els.field.closest(".field-panel");
  if (fieldPanel) {
    document.documentElement.style.setProperty("--field-panel-height", `${Math.ceil(fieldPanel.getBoundingClientRect().height)}px`);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function pointAtDistance(distance) {
  const d = clamp(distance, 0, path.total);
  for (const segment of path.segments) {
    if (d <= segment.start + segment.length) {
      const t = (d - segment.start) / segment.length;
      return {
        x: segment.a.x + segment.dx * t,
        y: segment.a.y + segment.dy * t,
        angle: segment.angle
      };
    }
  }
  const last = path.segments[path.segments.length - 1];
  return { x: last.b.x, y: last.b.y, angle: last.angle };
}

function closestPathPoint(x, y) {
  let best = { distance: Infinity, progress: 0, x: 0, y: 0 };
  for (const segment of path.segments) {
    const px = x - segment.a.x;
    const py = y - segment.a.y;
    const t = clamp((px * segment.dx + py * segment.dy) / (segment.length * segment.length), 0, 1);
    const cx = segment.a.x + segment.dx * t;
    const cy = segment.a.y + segment.dy * t;
    const distance = Math.hypot(x - cx, y - cy);
    if (distance < best.distance) {
      best = { distance, progress: segment.start + segment.length * t, x: cx, y: cy };
    }
  }
  return best;
}

function screenToWorld(event) {
  const rect = els.field.getBoundingClientRect();
  const sx = event.clientX - rect.left;
  const sy = event.clientY - rect.top;
  return {
    x: (sx - view.offsetX) / view.scale,
    y: (sy - view.offsetY) / view.scale
  };
}

function fmt(value) {
  return Math.floor(value).toLocaleString("en-US");
}

function randomInt(min, max) {
  return min + Math.floor(Math.random() * (max - min + 1));
}

function pick(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function createOperation(index) {
  if (index === 1) {
    return {
      index,
      facility: "Tokamak Facility B",
      sectorCount: 7,
      sector: 7
    };
  }
  const template = pick(operationTemplates);
  const sectorCount = randomInt(6, Math.min(14, 10 + Math.floor(index / 2)));
  return {
    index,
    facility: `${template.site} ${pick(template.suffixes)}`,
    sectorCount,
    sector: sectorCount
  };
}

function cloneOperation(operation) {
  return {
    index: operation.index,
    facility: operation.facility,
    sectorCount: operation.sectorCount,
    sector: operation.sector
  };
}

function operationLabel(operation) {
  return `${operation.facility} Sector ${operation.sector}`;
}

function operationCompactLabel(operation) {
  return `${operation.facility} / S${String(operation.sector).padStart(2, "0")}`;
}

function isBossOperation(operation) {
  return operation.sector <= 1;
}

function bossTypeForOperation(operation) {
  const operationIndex = Math.max(1, operation?.index || 1);
  return bossOrder[(operationIndex - 1) % bossOrder.length];
}

function threatDefinition(signature) {
  return bossDefs[signature] || enemyDefs[signature] || enemyDefs.crawler;
}

function isBossSignature(signature) {
  return !!bossDefs[signature];
}

function activeOperation() {
  return state.phase === "combat" && state.wavePlan ? state.wavePlan.operation : state.operation;
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function log(message) {
  state.logs.unshift(message);
  state.logs = state.logs.slice(0, 5);
}
