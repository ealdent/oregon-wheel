"use strict";

const CAMPAIGN_MAP = {
  nodeWidth: 154,
  nodeHeight: 98,
  gridX: 245,
  gridY: 162
};

const CAMPAIGN_TERRAIN = {
  version: 3,
  chunkSize: 640,
  margin: 260
};

const CAMPAIGN_TERRAIN_FEATURE_KEYS = [
  "rivers",
  "mountains",
  "forests",
  "ridges",
  "plateaus",
  "waters",
  "marshes",
  "rocks",
  "coasts",
  "structures",
  "ticks"
];

const FACILITY_LAYOUT_VERSION = 2;
const FACILITY_PATH_BOUNDS = {
  minX: 82,
  maxX: BOARD.width - 82,
  minY: 34,
  maxY: BOARD.height - 34,
  innerMinX: 118,
  innerMaxX: BOARD.width - 118,
  innerMinY: 82,
  innerMaxY: BOARD.height - 82
};

let campaignTerrainFeatureCacheKey = "";
let campaignTerrainFeatureCache = null;

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
    terrain: createCampaignTerrainStore(),
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
  campaign.terrain = normalizeCampaignTerrain(campaign.terrain);
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
    if (typeof showToast === "function") showToast("Campaign save unavailable");
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

function createCampaignTerrainStore() {
  return {
    version: CAMPAIGN_TERRAIN.version,
    chunkSize: CAMPAIGN_TERRAIN.chunkSize,
    chunks: {}
  };
}

function normalizeCampaignTerrain(terrain) {
  const normalized = createCampaignTerrainStore();
  if (
    !terrain
    || typeof terrain !== "object"
    || terrain.version !== CAMPAIGN_TERRAIN.version
    || terrain.chunkSize !== CAMPAIGN_TERRAIN.chunkSize
    || !terrain.chunks
    || typeof terrain.chunks !== "object"
  ) {
    return normalized;
  }
  for (const [key, chunk] of Object.entries(terrain.chunks)) {
    if (!chunk || typeof chunk !== "object") continue;
    const normalizedChunk = {
      x: Number.isFinite(chunk.x) ? chunk.x : 0,
      y: Number.isFinite(chunk.y) ? chunk.y : 0
    };
    for (const featureKey of CAMPAIGN_TERRAIN_FEATURE_KEYS) {
      normalizedChunk[featureKey] = Array.isArray(chunk[featureKey]) ? chunk[featureKey] : [];
    }
    normalized.chunks[key] = normalizedChunk;
  }
  return normalized;
}

function campaignTerrainChunkKey(x, y) {
  return `${x},${y}`;
}

function invalidateCampaignTerrainFeatureCache() {
  campaignTerrainFeatureCacheKey = "";
  campaignTerrainFeatureCache = null;
}

function campaignTerrainViewportRange(campaign) {
  const pan = campaign.pan || { x: 0, y: 0 };
  const size = CAMPAIGN_TERRAIN.chunkSize;
  const margin = CAMPAIGN_TERRAIN.margin;
  const minX = -pan.x - margin;
  const maxX = -pan.x + campaignViewportWidth() + margin;
  const minY = -pan.y - margin;
  const maxY = -pan.y + campaignViewportHeight() + margin;
  return {
    minChunkX: Math.floor(minX / size),
    maxChunkX: Math.floor(maxX / size),
    minChunkY: Math.floor(minY / size),
    maxChunkY: Math.floor(maxY / size)
  };
}

function ensureCampaignTerrainForViewport(campaign) {
  if (!campaign) return campaignTerrainFeatureCache || emptyCampaignTerrainFeatures();
  if (!campaign.terrain || !campaign.terrain.chunks) {
    campaign.terrain = createCampaignTerrainStore();
    invalidateCampaignTerrainFeatureCache();
  }
  const range = campaignTerrainViewportRange(campaign);
  let generated = false;
  for (let chunkY = range.minChunkY; chunkY <= range.maxChunkY; chunkY += 1) {
    for (let chunkX = range.minChunkX; chunkX <= range.maxChunkX; chunkX += 1) {
      const key = campaignTerrainChunkKey(chunkX, chunkY);
      if (!campaign.terrain.chunks[key]) {
        campaign.terrain.chunks[key] = createCampaignTerrainChunk(campaign, chunkX, chunkY);
        generated = true;
      }
    }
  }
  if (generated) {
    invalidateCampaignTerrainFeatureCache();
    saveCampaign(campaign);
  }
  const cacheKey = [
    campaign.seed,
    range.minChunkX,
    range.maxChunkX,
    range.minChunkY,
    range.maxChunkY,
    Object.keys(campaign.terrain.chunks).length
  ].join(":");
  if (campaignTerrainFeatureCache && campaignTerrainFeatureCacheKey === cacheKey) {
    return campaignTerrainFeatureCache;
  }
  const features = emptyCampaignTerrainFeatures();
  for (let chunkY = range.minChunkY; chunkY <= range.maxChunkY; chunkY += 1) {
    for (let chunkX = range.minChunkX; chunkX <= range.maxChunkX; chunkX += 1) {
      const chunk = campaign.terrain.chunks[campaignTerrainChunkKey(chunkX, chunkY)];
      if (!chunk) continue;
      for (const featureKey of CAMPAIGN_TERRAIN_FEATURE_KEYS) {
        features[featureKey].push(...(chunk[featureKey] || []));
      }
    }
  }
  campaignTerrainFeatureCacheKey = cacheKey;
  campaignTerrainFeatureCache = features;
  return features;
}

function emptyCampaignTerrainFeatures() {
  return Object.fromEntries(CAMPAIGN_TERRAIN_FEATURE_KEYS.map((key) => [key, []]));
}

function createCampaignTerrainChunk(campaign, chunkX, chunkY) {
  const rng = makeCampaignRng(`${campaign.seed}:terrain:${chunkX}:${chunkY}`);
  const size = CAMPAIGN_TERRAIN.chunkSize;
  const originX = chunkX * size;
  const originY = chunkY * size;
  const chunk = {
    x: chunkX,
    y: chunkY
  };
  for (const featureKey of CAMPAIGN_TERRAIN_FEATURE_KEYS) {
    chunk[featureKey] = [];
  }
  if (rng() < 0.54) {
    chunk.rivers.push(createCampaignTerrainRiver(rng, originX, originY, size));
  }
  if (rng() < 0.18) {
    chunk.coasts.push(createCampaignTerrainCoast(rng, originX, originY, size));
  }
  const waterCount = (rng() < 0.42 ? 1 : 0) + (rng() < 0.12 ? 1 : 0);
  for (let i = 0; i < waterCount; i += 1) {
    chunk.waters.push(createCampaignTerrainWater(rng, originX, originY, size));
  }
  const plateauCount = (rng() < 0.7 ? 1 : 0) + (rng() < 0.18 ? 1 : 0);
  for (let i = 0; i < plateauCount; i += 1) {
    chunk.plateaus.push(createCampaignTerrainPlateau(rng, originX, originY, size));
  }
  const mountainCount = campaignRandomInt(rng, 1, 3);
  for (let i = 0; i < mountainCount; i += 1) {
    chunk.mountains.push(createCampaignTerrainMountain(rng, originX, originY, size));
  }
  const forestCount = campaignRandomInt(rng, 2, 4);
  for (let i = 0; i < forestCount; i += 1) {
    chunk.forests.push(createCampaignTerrainForest(rng, originX, originY, size));
  }
  const ridgeCount = campaignRandomInt(rng, 2, 4);
  for (let i = 0; i < ridgeCount; i += 1) {
    chunk.ridges.push(createCampaignTerrainRidge(rng, originX, originY, size));
  }
  const marshCount = rng() < 0.38 ? 1 : 0;
  for (let i = 0; i < marshCount; i += 1) {
    chunk.marshes.push(createCampaignTerrainMarsh(rng, originX, originY, size));
  }
  const rockCount = campaignRandomInt(rng, 1, 3);
  for (let i = 0; i < rockCount; i += 1) {
    chunk.rocks.push(createCampaignTerrainRocks(rng, originX, originY, size));
  }
  if (rng() < 0.5) {
    chunk.structures.push(createCampaignTerrainStructure(rng, originX, originY, size));
  }
  const tickCount = campaignRandomInt(rng, 1, 2);
  for (let i = 0; i < tickCount; i += 1) {
    chunk.ticks.push(createCampaignTerrainTicks(rng, originX, originY, size));
  }
  return chunk;
}

function createTerrainGlyphFeature(rng, originX, originY, size, scaleMin, scaleMax, variants, marginX = 120, marginY = 90) {
  return {
    x: originX + campaignRandomInt(rng, -marginX, size + marginX),
    y: originY + campaignRandomInt(rng, -marginY, size + marginY),
    seed: campaignRandomInt(rng, 100000, 999999999),
    variant: campaignRandomInt(rng, 0, Math.max(0, variants - 1)),
    scale: Math.round((scaleMin + rng() * (scaleMax - scaleMin)) * 100) / 100,
    rotation: Math.round((-0.08 + rng() * 0.16) * 1000) / 1000
  };
}

function createCampaignTerrainRiver(rng, originX, originY, size) {
  const startX = originX - campaignRandomInt(rng, 180, 300);
  const startY = originY + campaignRandomInt(rng, -110, size + 110);
  const pointCount = campaignRandomInt(rng, 6, 8);
  const step = (size + campaignRandomInt(rng, 260, 460)) / (pointCount - 1);
  const drift = campaignRandomInt(rng, -28, 28);
  const amp = campaignRandomInt(rng, 42, 112);
  const phase = rng() * Math.PI * 2;
  const points = [];
  for (let i = 0; i < pointCount; i += 1) {
    points.push({
      x: Math.round(startX + i * step),
      y: Math.round(startY + Math.sin(i * 1.32 + phase) * amp + drift * i + campaignRandomInt(rng, -26, 26))
    });
  }
  const branches = [];
  const branchCount = campaignRandomInt(rng, 0, 2);
  for (let i = 0; i < branchCount; i += 1) {
    const rootIndex = campaignRandomInt(rng, 1, Math.max(1, points.length - 3));
    const root = points[rootIndex];
    const direction = rng() > 0.5 ? 1 : -1;
    const branch = [{ x: root.x, y: root.y }];
    const len = campaignRandomInt(rng, 95, 190);
    for (let j = 1; j <= 3; j += 1) {
      branch.push({
        x: Math.round(root.x + j * len / 3 + campaignRandomInt(rng, -18, 18)),
        y: Math.round(root.y + direction * (j * campaignRandomInt(rng, 22, 48)) + campaignRandomInt(rng, -18, 18))
      });
    }
    branches.push(branch);
  }
  return {
    seed: campaignRandomInt(rng, 100000, 999999999),
    points,
    branches,
    width: Math.round((0.9 + rng() * 0.5) * 100) / 100
  };
}

function createCampaignTerrainMountain(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.72, 1.12, 7, 72, 54);
  feature.width = Math.round(campaignRandomInt(rng, 108, 178) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 54, 96) * feature.scale);
  return feature;
}

function createCampaignTerrainPlateau(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.72, 1.42, 6);
  feature.width = Math.round(campaignRandomInt(rng, 130, 260) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 58, 120) * feature.scale);
  return feature;
}

function createCampaignTerrainForest(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.78, 1.55, 5);
  feature.count = campaignRandomInt(rng, 12, 42);
  feature.width = Math.round(campaignRandomInt(rng, 105, 230) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 64, 128) * feature.scale);
  return feature;
}

function createCampaignTerrainRidge(rng, originX, originY, size) {
  const x = originX + campaignRandomInt(rng, -110, size + 110);
  const y = originY + campaignRandomInt(rng, -90, size + 90);
  const width = campaignRandomInt(rng, 145, 360);
  const segments = campaignRandomInt(rng, 5, 9);
  const points = [];
  for (let i = 0; i <= segments; i += 1) {
    points.push({
      x: Math.round(x - width / 2 + (width / segments) * i),
      y: Math.round(y + Math.sin(i * 1.35 + rng() * 0.4) * campaignRandomInt(rng, 8, 22) + campaignRandomInt(rng, -8, 8))
    });
  }
  return {
    x: Math.round(x),
    y: Math.round(y),
    seed: campaignRandomInt(rng, 100000, 999999999),
    width,
    variant: campaignRandomInt(rng, 0, 3),
    points
  };
}

function createCampaignTerrainWater(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.72, 1.35, 5);
  feature.width = Math.round(campaignRandomInt(rng, 96, 205) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 42, 94) * feature.scale);
  return feature;
}

function createCampaignTerrainMarsh(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.78, 1.44, 4);
  feature.width = Math.round(campaignRandomInt(rng, 142, 275) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 58, 122) * feature.scale);
  return feature;
}

function createCampaignTerrainRocks(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.68, 1.22, 5);
  feature.width = Math.round(campaignRandomInt(rng, 74, 168) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 34, 78) * feature.scale);
  return feature;
}

function createCampaignTerrainCoast(rng, originX, originY, size) {
  const x = originX + campaignRandomInt(rng, -150, size + 150);
  const y = originY + campaignRandomInt(rng, -120, size + 120);
  const width = campaignRandomInt(rng, 210, 420);
  const segments = campaignRandomInt(rng, 8, 13);
  const points = [];
  const vertical = rng() > 0.5;
  for (let i = 0; i <= segments; i += 1) {
    const t = i / segments;
    const wave = Math.sin(t * Math.PI * 3 + rng() * 0.6) * campaignRandomInt(rng, 20, 48);
    points.push(vertical
      ? {
          x: Math.round(x + wave + campaignRandomInt(rng, -18, 18)),
          y: Math.round(y - width / 2 + width * t)
        }
      : {
          x: Math.round(x - width / 2 + width * t),
          y: Math.round(y + wave + campaignRandomInt(rng, -18, 18))
        });
  }
  return {
    x: Math.round(x),
    y: Math.round(y),
    seed: campaignRandomInt(rng, 100000, 999999999),
    width,
    vertical,
    points
  };
}

function createCampaignTerrainStructure(rng, originX, originY, size) {
  const feature = createTerrainGlyphFeature(rng, originX, originY, size, 0.82, 1.36, 4);
  feature.width = Math.round(campaignRandomInt(rng, 120, 250) * feature.scale);
  feature.height = Math.round(campaignRandomInt(rng, 46, 108) * feature.scale);
  return feature;
}

function createCampaignTerrainTicks(rng, originX, originY, size) {
  const x = originX + campaignRandomInt(rng, -90, size + 90);
  const y = originY + campaignRandomInt(rng, -90, size + 90);
  const count = campaignRandomInt(rng, 5, 13);
  const marks = [];
  for (let i = 0; i < count; i += 1) {
    const markX = x + (rng() - 0.5) * 150;
    const markY = y + (rng() - 0.5) * 80;
    const len = 8 + rng() * 24;
    marks.push({
      x: Math.round(markX),
      y: Math.round(markY),
      x2: Math.round(markX + len),
      y2: Math.round(markY + (rng() - 0.5) * 3),
      cross: i % 4 === 0
    });
  }
  return { x: Math.round(x), y: Math.round(y), marks };
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
    x: campaignViewportWidth() / 2 + node.gridX * CAMPAIGN_MAP.gridX + pan.x,
    y: campaignViewportHeight() / 2 + node.gridY * CAMPAIGN_MAP.gridY + pan.y
  };
}

function campaignViewportHeight() {
  return BOARD.height;
}

function campaignViewportWidth() {
  try {
    if (currentState() && currentState().mode === "campaign" && view.cssWidth && view.cssHeight) {
      return Math.max(BOARD.width, BOARD.height * (view.cssWidth / view.cssHeight));
    }
  } catch (error) {
    return BOARD.width;
  }
  return BOARD.width;
}

function campaignRenderScale() {
  try {
    if (view.cssHeight) return view.cssHeight / campaignViewportHeight();
  } catch (error) {
    return 1;
  }
  return 1;
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
  const layout = ensureFacilityLayout(node);
  return {
    nodeId: node.id,
    type: node.type,
    typeDef: type,
    pathPoints: layout.pathPoints.map((point) => ({ ...point })),
    pathProfile: { ...(layout.pathProfile || {}) },
    pathLength: layout.pathLength,
    schematicSeed: campaignHash(`${node.seed}:schematic`),
    grimeSeed: campaignHash(`${node.seed}:grime`),
    mapSeed: node.seed
  };
}

function ensureFacilityLayout(node) {
  if (
    node.layout
    && node.layout.version === FACILITY_LAYOUT_VERSION
    && Array.isArray(node.layout.pathPoints)
    && node.layout.pathPoints.length >= 2
  ) {
    return node.layout;
  }
  const preserveStartedRoute = !!(
    node.checkpoint
    && (
      (node.checkpoint.towers && node.checkpoint.towers.length)
      || (node.checkpoint.mines && node.checkpoint.mines.length)
      || (node.checkpoint.wave && node.checkpoint.wave > 0)
      || (node.checkpoint.currentSector && node.checkpoint.currentSector < node.sectorCount)
    )
  );
  const generated = preserveStartedRoute
    ? createFacilityLayoutFromPath(generateLegacyFacilityPath(node.seed), node, "legacy-preserved")
    : createFacilityLayoutFromPath(generateFacilityPath(node), node, null);
  node.layout = generated;
  return node.layout;
}

function createFacilityLayoutFromPath(result, node, forcedArchetype) {
  const pathPoints = Array.isArray(result) ? result : result.points;
  const normalized = normalizeFacilityPath(pathPoints);
  const length = Math.round(measureFacilityPathLength(normalized));
  return {
    version: FACILITY_LAYOUT_VERSION,
    pathPoints: normalized,
    pathLength: length,
    pathProfile: {
      difficulty: Math.round(facilityPathDifficulty(node) * 1000) / 1000,
      archetype: forcedArchetype || result.archetype || "generated",
      turns: countFacilityPathTurns(normalized),
      targetLength: result.targetLength ? Math.round(result.targetLength) : null
    }
  };
}

function generateLegacyFacilityPath(seed) {
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

function generateFacilityPath(node) {
  const difficulty = facilityPathDifficulty(node);
  const rng = makeCampaignRng(`${node.seed}:path:v${FACILITY_LAYOUT_VERSION}`);
  const targetLength = 2120 - difficulty * 980;
  const archetype = pickFacilityPathArchetype(rng, difficulty);
  let best = null;
  for (let i = 0; i < 14; i += 1) {
    const points = normalizeFacilityPath(generateFacilityPathByArchetype(archetype, rng, difficulty));
    const length = measureFacilityPathLength(points);
    const turns = countFacilityPathTurns(points);
    const score = (
      Math.abs(length - targetLength) / targetLength
      + difficulty * turns * 0.022
      + (1 - difficulty) * Math.max(0, 7 - turns) * 0.052
      + (pathCenterBias(points) * 0.12)
      + rng() * 0.035
    );
    if (!best || score < best.score) {
      best = { points, archetype, length, turns, targetLength, score };
    }
  }
  return best || { points: generateLegacyFacilityPath(node.seed), archetype: "legacy-fallback", targetLength };
}

function facilityPathDifficulty(node) {
  const typeMods = {
    cargo: -0.08,
    cryo: -0.03,
    tokamak: 0.03,
    radar: 0.06,
    foundry: 0.1
  };
  const indexPressure = clamp((node.index - 1) / 18, 0, 1) * 0.5;
  const sectorPressure = clamp((node.sectorCount - 5) / 11, 0, 1) * 0.38;
  return clamp(indexPressure + sectorPressure + (typeMods[node.type] || 0), 0, 1);
}

function pickFacilityPathArchetype(rng, difficulty) {
  const easy = [
    ["wide-serpentine", 3.4],
    ["perimeter-loop", 2.6],
    ["spiral", 2.4]
  ];
  const mid = [
    ["diagonal-weave", 2.4],
    ["offset-switchback", 2.0],
    ["perimeter-loop", 1.1],
    ["direct-dogleg", 1.0],
    ["wide-serpentine", 0.9]
  ];
  const hard = [
    ["direct-dogleg", 3.6],
    ["straight-chicane", 3.0],
    ["diagonal-weave", 1.0]
  ];
  const table = difficulty < 0.35 ? easy : difficulty > 0.68 ? hard : mid;
  const total = table.reduce((sum, [, weight]) => sum + weight, 0);
  let roll = rng() * total;
  for (const [archetype, weight] of table) {
    roll -= weight;
    if (roll <= 0) return archetype;
  }
  return table[0][0];
}

function generateFacilityPathByArchetype(archetype, rng, difficulty) {
  if (archetype === "perimeter-loop") return generatePerimeterLoopPath(rng, difficulty);
  if (archetype === "spiral") return generateSpiralPath(rng, difficulty);
  if (archetype === "diagonal-weave") return generateDiagonalWeavePath(rng, difficulty);
  if (archetype === "offset-switchback") return generateOffsetSwitchbackPath(rng, difficulty);
  if (archetype === "direct-dogleg") return generateDirectDoglegPath(rng, difficulty);
  if (archetype === "straight-chicane") return generateStraightChicanePath(rng, difficulty);
  return generateWideSerpentinePath(rng, difficulty);
}

function edgePoint(edge, rng, variance = 0.72) {
  const b = FACILITY_PATH_BOUNDS;
  const xRange = (b.innerMaxX - b.innerMinX) * variance;
  const yRange = (b.innerMaxY - b.innerMinY) * variance;
  const centerX = BOARD.width / 2;
  const centerY = BOARD.height / 2;
  if (edge === "top") return { x: Math.round(centerX + (rng() - 0.5) * xRange), y: b.minY };
  if (edge === "bottom") return { x: Math.round(centerX + (rng() - 0.5) * xRange), y: b.maxY };
  if (edge === "left") return { x: b.minX, y: Math.round(centerY + (rng() - 0.5) * yRange) };
  return { x: b.maxX, y: Math.round(centerY + (rng() - 0.5) * yRange) };
}

function oppositeEdge(edge) {
  return { top: "bottom", bottom: "top", left: "right", right: "left" }[edge] || "bottom";
}

function perpendicularEdges(edge) {
  return edge === "top" || edge === "bottom" ? ["left", "right"] : ["top", "bottom"];
}

function randomExitEdge(rng, startEdge, difficulty) {
  if (difficulty > 0.64 && rng() < 0.72) return oppositeEdge(startEdge);
  const options = rng() < 0.62 ? perpendicularEdges(startEdge) : [oppositeEdge(startEdge), ...perpendicularEdges(startEdge)];
  return campaignPick(rng, options);
}

function generateWideSerpentinePath(rng, difficulty) {
  const vertical = rng() > 0.38;
  const turns = campaignRandomInt(rng, difficulty < 0.25 ? 5 : 5, difficulty < 0.25 ? 8 : 7);
  const points = [];
  if (vertical) {
    const startEdge = rng() > 0.5 ? "top" : "bottom";
    const endEdge = oppositeEdge(startEdge);
    const start = edgePoint(startEdge, rng);
    const endY = endEdge === "bottom" ? FACILITY_PATH_BOUNDS.maxY : FACILITY_PATH_BOUNDS.minY;
    const dir = endY > start.y ? 1 : -1;
    points.push(start);
    let x = start.x;
    for (let i = 1; i <= turns; i += 1) {
      const t = i / (turns + 1);
      const y = start.y + (endY - start.y) * t + campaignRandomInt(rng, -18, 18);
      points.push({ x, y });
      const targetSide = i % 2 === 1 ? (rng() > 0.5 ? "left" : "right") : (x < BOARD.width / 2 ? "right" : "left");
      const nextX = targetSide === "left"
        ? campaignRandomInt(rng, FACILITY_PATH_BOUNDS.innerMinX, 330)
        : campaignRandomInt(rng, 670, FACILITY_PATH_BOUNDS.innerMaxX);
      x = clamp(nextX + campaignRandomInt(rng, -30, 30), FACILITY_PATH_BOUNDS.innerMinX, FACILITY_PATH_BOUNDS.innerMaxX);
      points.push({ x, y });
    }
    points.push({ x, y: endY });
  } else {
    const startEdge = rng() > 0.5 ? "left" : "right";
    const endEdge = oppositeEdge(startEdge);
    const start = edgePoint(startEdge, rng);
    const endX = endEdge === "right" ? FACILITY_PATH_BOUNDS.maxX : FACILITY_PATH_BOUNDS.minX;
    points.push(start);
    let y = start.y;
    for (let i = 1; i <= turns; i += 1) {
      const t = i / (turns + 1);
      const x = start.x + (endX - start.x) * t + campaignRandomInt(rng, -20, 20);
      points.push({ x, y });
      const targetBand = i % 2 === 1 ? (rng() > 0.5 ? "top" : "bottom") : (y < BOARD.height / 2 ? "bottom" : "top");
      const nextY = targetBand === "top"
        ? campaignRandomInt(rng, FACILITY_PATH_BOUNDS.innerMinY, 230)
        : campaignRandomInt(rng, 450, FACILITY_PATH_BOUNDS.innerMaxY);
      y = clamp(nextY + campaignRandomInt(rng, -22, 22), FACILITY_PATH_BOUNDS.innerMinY, FACILITY_PATH_BOUNDS.innerMaxY);
      points.push({ x, y });
    }
    points.push({ x: endX, y });
  }
  return points;
}

function generatePerimeterLoopPath(rng, difficulty) {
  const inset = campaignRandomInt(rng, 92, 142);
  const startEdge = campaignPick(rng, ["top", "left", "right", "bottom"]);
  const exitEdge = randomExitEdge(rng, startEdge, difficulty * 0.7);
  const start = edgePoint(startEdge, rng, 0.8);
  const end = edgePoint(exitEdge, rng, 0.8);
  const corners = [
    { x: inset, y: inset },
    { x: BOARD.width - inset, y: inset },
    { x: BOARD.width - inset, y: BOARD.height - inset },
    { x: inset, y: BOARD.height - inset }
  ];
  const startInside = projectToInset(start, startEdge, inset);
  const endInside = projectToInset(end, exitEdge, inset);
  const startCorner = perimeterCornerIndex(startEdge, startInside);
  const endCorner = perimeterCornerIndex(exitEdge, endInside);
  const clockwise = perimeterCornerSequence(startCorner, endCorner, 1);
  const counter = perimeterCornerSequence(startCorner, endCorner, -1);
  const cornerIndexes = difficulty < 0.35
    ? (clockwise.length >= counter.length ? clockwise : counter)
    : (rng() > 0.5 ? clockwise : counter);
  const points = [start, startInside];
  for (const index of cornerIndexes) {
    points.push({
      x: corners[index].x + campaignRandomInt(rng, -18, 18),
      y: corners[index].y + campaignRandomInt(rng, -18, 18)
    });
  }
  points.push(endInside, end);
  return points;
}

function projectToInset(point, edge, inset) {
  if (edge === "top") return { x: point.x, y: inset };
  if (edge === "bottom") return { x: point.x, y: BOARD.height - inset };
  if (edge === "left") return { x: inset, y: point.y };
  return { x: BOARD.width - inset, y: point.y };
}

function perimeterCornerIndex(edge, point) {
  if (edge === "top") return point.x < BOARD.width / 2 ? 0 : 1;
  if (edge === "right") return point.y < BOARD.height / 2 ? 1 : 2;
  if (edge === "bottom") return point.x > BOARD.width / 2 ? 2 : 3;
  return point.y > BOARD.height / 2 ? 3 : 0;
}

function perimeterCornerSequence(startIndex, endIndex, step) {
  const indexes = [];
  let index = startIndex;
  let guard = 0;
  while (guard < 6) {
    indexes.push(index);
    if (index === endIndex && indexes.length > 1) break;
    index = (index + step + 4) % 4;
    guard += 1;
  }
  return indexes;
}

function generateSpiralPath(rng, difficulty) {
  const startEdge = campaignPick(rng, ["top", "left"]);
  const endEdge = startEdge === "top" ? "bottom" : "right";
  const start = edgePoint(startEdge, rng, 0.6);
  const end = edgePoint(endEdge, rng, 0.6);
  const outer = campaignRandomInt(rng, 86, 124);
  const inner = campaignRandomInt(rng, 245, 315);
  const cx = BOARD.width / 2 + campaignRandomInt(rng, -80, 80);
  const cy = BOARD.height / 2 + campaignRandomInt(rng, -55, 55);
  const points = [start, projectToInset(start, startEdge, outer)];
  points.push(
    { x: BOARD.width - outer, y: outer },
    { x: BOARD.width - outer, y: BOARD.height - outer },
    { x: outer, y: BOARD.height - outer },
    { x: outer, y: inner },
    { x: cx + campaignRandomInt(rng, 95, 170), y: inner },
    { x: cx + campaignRandomInt(rng, 95, 170), y: cy + campaignRandomInt(rng, 70, 135) },
    { x: cx - campaignRandomInt(rng, 80, 150), y: cy + campaignRandomInt(rng, 70, 135) },
    projectToInset(end, endEdge, outer),
    end
  );
  if (difficulty > 0.45) points.splice(5, 2);
  return points;
}

function generateDiagonalWeavePath(rng, difficulty) {
  const startEdge = campaignPick(rng, ["top", "left", "right", "bottom"]);
  const exitEdge = randomExitEdge(rng, startEdge, difficulty);
  const start = edgePoint(startEdge, rng, 0.82);
  const end = edgePoint(exitEdge, rng, 0.82);
  const count = campaignRandomInt(rng, difficulty > 0.55 ? 2 : 3, difficulty > 0.55 ? 4 : 6);
  const points = [start];
  for (let i = 1; i <= count; i += 1) {
    const t = i / (count + 1);
    const waviness = (1 - difficulty) * 190 + 55;
    points.push({
      x: start.x + (end.x - start.x) * t + Math.sin(t * Math.PI * 2 + rng() * 1.4) * waviness + campaignRandomInt(rng, -45, 45),
      y: start.y + (end.y - start.y) * t + Math.cos(t * Math.PI * 2 + rng() * 1.4) * waviness * 0.62 + campaignRandomInt(rng, -35, 35)
    });
  }
  points.push(end);
  return points;
}

function generateOffsetSwitchbackPath(rng, difficulty) {
  const vertical = rng() > 0.5;
  const rows = campaignRandomInt(rng, 4, difficulty > 0.55 ? 5 : 7);
  const points = [];
  if (vertical) {
    const start = edgePoint("top", rng, 0.82);
    points.push(start);
    let x = start.x;
    for (let i = 1; i <= rows; i += 1) {
      const y = FACILITY_PATH_BOUNDS.minY + (FACILITY_PATH_BOUNDS.maxY - FACILITY_PATH_BOUNDS.minY) * (i / (rows + 1));
      points.push({ x, y });
      x = campaignRandomInt(rng, FACILITY_PATH_BOUNDS.innerMinX, FACILITY_PATH_BOUNDS.innerMaxX);
      points.push({ x, y });
    }
    points.push({ x, y: FACILITY_PATH_BOUNDS.maxY });
  } else {
    const start = edgePoint("left", rng, 0.82);
    points.push(start);
    let y = start.y;
    for (let i = 1; i <= rows; i += 1) {
      const x = FACILITY_PATH_BOUNDS.minX + (FACILITY_PATH_BOUNDS.maxX - FACILITY_PATH_BOUNDS.minX) * (i / (rows + 1));
      points.push({ x, y });
      y = campaignRandomInt(rng, FACILITY_PATH_BOUNDS.innerMinY, FACILITY_PATH_BOUNDS.innerMaxY);
      points.push({ x, y });
    }
    points.push({ x: FACILITY_PATH_BOUNDS.maxX, y });
  }
  return points;
}

function generateDirectDoglegPath(rng, difficulty) {
  const startEdge = campaignPick(rng, ["top", "left", "right", "bottom"]);
  const exitEdge = randomExitEdge(rng, startEdge, Math.max(0.7, difficulty));
  const start = edgePoint(startEdge, rng, 0.66);
  const end = edgePoint(exitEdge, rng, 0.66);
  const bends = campaignRandomInt(rng, 1, difficulty > 0.78 ? 2 : 3);
  const points = [start];
  for (let i = 1; i <= bends; i += 1) {
    const t = i / (bends + 1);
    points.push({
      x: start.x + (end.x - start.x) * t + campaignRandomInt(rng, -80, 80) * (1 - difficulty * 0.45),
      y: start.y + (end.y - start.y) * t + campaignRandomInt(rng, -64, 64) * (1 - difficulty * 0.45)
    });
  }
  points.push(end);
  return points;
}

function generateStraightChicanePath(rng, difficulty) {
  const startEdge = campaignPick(rng, ["top", "left", "right", "bottom"]);
  const exitEdge = oppositeEdge(startEdge);
  const start = edgePoint(startEdge, rng, 0.52);
  const end = edgePoint(exitEdge, rng, 0.52);
  const points = [start];
  const middleCount = difficulty > 0.82 ? 1 : 2;
  for (let i = 1; i <= middleCount; i += 1) {
    const t = i / (middleCount + 1);
    points.push({
      x: start.x + (end.x - start.x) * t + campaignRandomInt(rng, -42, 42),
      y: start.y + (end.y - start.y) * t + campaignRandomInt(rng, -42, 42)
    });
  }
  points.push(end);
  return points;
}

function normalizeFacilityPath(points) {
  const normalized = [];
  for (const point of points) {
    const next = {
      x: Math.round(clamp(point.x, FACILITY_PATH_BOUNDS.minX, FACILITY_PATH_BOUNDS.maxX)),
      y: Math.round(clamp(point.y, FACILITY_PATH_BOUNDS.minY, FACILITY_PATH_BOUNDS.maxY))
    };
    const previous = normalized[normalized.length - 1];
    if (!previous || Math.hypot(previous.x - next.x, previous.y - next.y) >= 34) {
      normalized.push(next);
    }
  }
  return removeFacilityPathCollinear(normalized.length >= 2 ? normalized : DEFAULT_PATH_POINTS);
}

function removeFacilityPathCollinear(points) {
  if (points.length <= 2) return points.map((point) => ({ ...point }));
  const result = [points[0]];
  for (let i = 1; i < points.length - 1; i += 1) {
    const a = result[result.length - 1];
    const b = points[i];
    const c = points[i + 1];
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const bcx = c.x - b.x;
    const bcy = c.y - b.y;
    const cross = Math.abs(abx * bcy - aby * bcx);
    const dot = abx * bcx + aby * bcy;
    if (cross > 220 || dot < 0) result.push(b);
  }
  result.push(points[points.length - 1]);
  return result;
}

function measureFacilityPathLength(points) {
  let length = 0;
  for (let i = 1; i < points.length; i += 1) {
    length += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  }
  return length;
}

function countFacilityPathTurns(points) {
  let turns = 0;
  for (let i = 1; i < points.length - 1; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const c = points[i + 1];
    const angleA = Math.atan2(b.y - a.y, b.x - a.x);
    const angleB = Math.atan2(c.y - b.y, c.x - b.x);
    const delta = Math.abs(Math.atan2(Math.sin(angleB - angleA), Math.cos(angleB - angleA)));
    if (delta > 0.35) turns += 1;
  }
  return turns;
}

function pathCenterBias(points) {
  if (!points.length) return 0;
  const central = points.filter((point) => point.x > 330 && point.x < 670 && point.y > 190 && point.y < 500).length;
  return central / points.length;
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
  saveCampaign(campaign);
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
