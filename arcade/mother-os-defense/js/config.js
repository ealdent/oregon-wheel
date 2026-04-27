"use strict";

const BOARD = { width: 1000, height: 680, pathWidth: 76 };
const MAX_TOWER_LEVEL = 12;
const TARGET_MODES = ["FIRST", "STRONG", "NEAR"];
const SPEEDS = [1, 2, 3];
const CAMPAIGN_STORAGE_KEY = "mother-os-campaign-v1";
const STARTING_CREDITS = 360;
const STARTING_LIVES = 25;
const WAVE_BALANCE = {
  regular: {
    baseCount: 44,
    countPerWave: 6,
    countCap: 138,
    intervalStart: 1.04,
    intervalDrop: 0.01,
    minInterval: 0.44,
    durationBase: 36,
    durationGrowth: 2,
    durationGrowthCap: 24
  },
  boss: {
    baseCount: 34,
    countPerWave: 5,
    countCap: 122,
    intervalStart: 0.96,
    intervalDrop: 0.012,
    minInterval: 0.5,
    durationBase: 64,
    durationGrowth: 2,
    durationGrowthCap: 30
  },
  healthGrowth: 1.145,
  healthLinear: 0.048,
  bossHealthBase: 6.8,
  bossHealthPerWave: 0.42,
  armorLinear: 0.032,
  rewardLinear: 0.055,
  bossRewardMult: 8,
  clearBase: 24,
  clearPerWave: 6,
  bossClearBonus: 90
};
const DEFAULT_PATH_POINTS = [
  { x: 520, y: 34 },
  { x: 520, y: 118 },
  { x: 635, y: 118 },
  { x: 635, y: 212 },
  { x: 350, y: 212 },
  { x: 350, y: 315 },
  { x: 575, y: 315 },
  { x: 575, y: 425 },
  { x: 790, y: 425 },
  { x: 790, y: 530 },
  { x: 245, y: 530 },
  { x: 245, y: 646 }
];
let pathPoints = DEFAULT_PATH_POINTS.map((point) => ({ ...point }));

const facilityTypes = {
  tokamak: {
    label: "Tokamak Reactor",
    desc: "Plasma infrastructure. Fast pressure and volatile returns.",
    glyph: "tokamak",
    color: "#ffcf5a",
    accent: "#85ff91",
    palette: {
      grid: "rgba(255,207,90,0.075)",
      major: "rgba(255,207,90,0.15)",
      pathOuter: "rgba(255,207,90,0.08)",
      pathGlow: "rgba(255,207,90,0.2)",
      pathBody: "rgba(76,74,21,0.62)",
      pathCore: "rgba(32,31,8,0.94)"
    },
    balance: { count: 1.05, health: 1.0, speed: 1.06, reward: 1.04, interval: 0.94, armor: 1.0 },
    enemyBias: { wisp: 1.1, crawler: 0.8 }
  },
  radar: {
    label: "Radar Array",
    desc: "Signal-dense approach. More phasing and burrowing threats.",
    glyph: "radar",
    color: "#d6c3ff",
    accent: "#7ce8ff",
    palette: {
      grid: "rgba(214,195,255,0.058)",
      major: "rgba(214,195,255,0.12)",
      pathOuter: "rgba(214,195,255,0.05)",
      pathGlow: "rgba(214,195,255,0.13)",
      pathBody: "rgba(52,39,76,0.5)",
      pathCore: "rgba(22,13,35,0.9)"
    },
    balance: { count: 1.0, health: 1.02, speed: 1.04, reward: 1.03, interval: 0.98, armor: 1.0 },
    enemyBias: { phantom: 1.2, worm: 1.15, obelisk: 0.7 }
  },
  annex: {
    label: "Annex Complex",
    desc: "Layered civic grid. Balanced pressure from all vectors.",
    glyph: "annex",
    color: "#85ff91",
    accent: "#d2ff78",
    palette: {
      grid: "rgba(133,255,145,0.066)",
      major: "rgba(133,255,145,0.14)",
      pathOuter: "rgba(133,255,145,0.058)",
      pathGlow: "rgba(133,255,145,0.16)",
      pathBody: "rgba(27,70,31,0.54)",
      pathCore: "rgba(8,27,11,0.92)"
    },
    balance: { count: 1.0, health: 1.0, speed: 1.0, reward: 1.0, interval: 1.0, armor: 1.0 },
    enemyBias: { crawler: 1.05, beetle: 1.05 }
  },
  cryo: {
    label: "Bio Research Facility",
    desc: "Containment domes. Slower lanes, resilient anomalies.",
    glyph: "bio",
    color: "#b8f7ff",
    accent: "#7ce8ff",
    palette: {
      grid: "rgba(184,247,255,0.065)",
      major: "rgba(184,247,255,0.13)",
      pathOuter: "rgba(184,247,255,0.055)",
      pathGlow: "rgba(184,247,255,0.15)",
      pathBody: "rgba(25,58,66,0.55)",
      pathCore: "rgba(9,25,31,0.93)"
    },
    balance: { count: 0.96, health: 1.06, speed: 0.9, reward: 1.0, interval: 1.0, armor: 1.04 },
    enemyBias: { slime: 1.2, wisp: 0.9 }
  },
  foundry: {
    label: "Mining Operation",
    desc: "Excavation works. Fewer but harder armored chassis.",
    glyph: "mining",
    color: "#d2ff78",
    accent: "#ffcf5a",
    palette: {
      grid: "rgba(210,255,120,0.07)",
      major: "rgba(210,255,120,0.14)",
      pathOuter: "rgba(210,255,120,0.06)",
      pathGlow: "rgba(210,255,120,0.17)",
      pathBody: "rgba(54,76,20,0.58)",
      pathCore: "rgba(23,31,8,0.92)"
    },
    balance: { count: 0.9, health: 1.14, speed: 0.94, reward: 1.08, interval: 1.08, armor: 1.16 },
    enemyBias: { beetle: 1.3, juggernaut: 1.15, crawler: 0.75 }
  },
  power: {
    label: "Power Plant",
    desc: "High-output stacks. Heavy armor and unstable surges.",
    glyph: "power",
    color: "#ffcf5a",
    accent: "#d2ff78",
    palette: {
      grid: "rgba(255,207,90,0.067)",
      major: "rgba(255,207,90,0.135)",
      pathOuter: "rgba(255,207,90,0.06)",
      pathGlow: "rgba(255,207,90,0.15)",
      pathBody: "rgba(68,58,18,0.58)",
      pathCore: "rgba(31,25,7,0.92)"
    },
    balance: { count: 0.94, health: 1.12, speed: 0.98, reward: 1.08, interval: 1.05, armor: 1.12 },
    enemyBias: { juggernaut: 1.18, wisp: 1.08 }
  },
  comms: {
    label: "Communications Hub",
    desc: "Uplink nests. More mimics and signal anomalies.",
    glyph: "comms",
    color: "#d6c3ff",
    accent: "#85ff91",
    palette: {
      grid: "rgba(214,195,255,0.06)",
      major: "rgba(214,195,255,0.125)",
      pathOuter: "rgba(214,195,255,0.052)",
      pathGlow: "rgba(214,195,255,0.14)",
      pathBody: "rgba(48,40,72,0.52)",
      pathCore: "rgba(20,13,34,0.9)"
    },
    balance: { count: 1.03, health: 0.98, speed: 1.08, reward: 1.02, interval: 0.96, armor: 0.98 },
    enemyBias: { phantom: 1.18, wisp: 1.12, crawler: 0.86 }
  },
  hydro: {
    label: "Hydroponics Farm",
    desc: "Greenhouse corridors. Many light biological swarms.",
    glyph: "hydro",
    color: "#85ff91",
    accent: "#b9ffbd",
    palette: {
      grid: "rgba(133,255,145,0.064)",
      major: "rgba(133,255,145,0.13)",
      pathOuter: "rgba(133,255,145,0.052)",
      pathGlow: "rgba(133,255,145,0.145)",
      pathBody: "rgba(22,70,29,0.52)",
      pathCore: "rgba(7,29,12,0.9)"
    },
    balance: { count: 1.14, health: 0.92, speed: 1.02, reward: 0.98, interval: 0.98, armor: 0.9 },
    enemyBias: { slime: 1.25, mite: 1.16, juggernaut: 0.72 }
  },
  vehicle: {
    label: "Vehicle Depot",
    desc: "Armored bays. Mixed packs with heavier walkers.",
    glyph: "vehicle",
    color: "#7ce8ff",
    accent: "#ffcf5a",
    palette: {
      grid: "rgba(124,232,255,0.064)",
      major: "rgba(124,232,255,0.13)",
      pathOuter: "rgba(124,232,255,0.055)",
      pathGlow: "rgba(124,232,255,0.15)",
      pathBody: "rgba(19,62,70,0.55)",
      pathCore: "rgba(8,29,34,0.91)"
    },
    balance: { count: 0.98, health: 1.08, speed: 1.0, reward: 1.05, interval: 1.03, armor: 1.1 },
    enemyBias: { juggernaut: 1.16, beetle: 1.12 }
  },
  satellite: {
    label: "Satellite Uplink",
    desc: "Orbital relay. Fast phasing threats and weak armor.",
    glyph: "satellite",
    color: "#b8f7ff",
    accent: "#d6c3ff",
    palette: {
      grid: "rgba(184,247,255,0.058)",
      major: "rgba(184,247,255,0.12)",
      pathOuter: "rgba(184,247,255,0.05)",
      pathGlow: "rgba(184,247,255,0.13)",
      pathBody: "rgba(24,54,68,0.5)",
      pathCore: "rgba(8,24,32,0.9)"
    },
    balance: { count: 1.02, health: 0.96, speed: 1.13, reward: 1.02, interval: 0.94, armor: 0.94 },
    enemyBias: { wisp: 1.22, phantom: 1.08, beetle: 0.82 }
  },
  cargo: {
    label: "Storage Facility",
    desc: "Dense storage blocks. Bigger packs, softer armor.",
    glyph: "storage",
    color: "#7ce8ff",
    accent: "#b9ffbd",
    palette: {
      grid: "rgba(124,232,255,0.065)",
      major: "rgba(124,232,255,0.13)",
      pathOuter: "rgba(124,232,255,0.06)",
      pathGlow: "rgba(124,232,255,0.16)",
      pathBody: "rgba(20,68,70,0.56)",
      pathCore: "rgba(8,31,34,0.92)"
    },
    balance: { count: 1.12, health: 0.94, speed: 0.98, reward: 0.98, interval: 1.02, armor: 0.94 },
    enemyBias: { crawler: 1.2, mite: 0.9 }
  },
  command: {
    label: "Command Center",
    desc: "Hardened control nexus. Elite-heavy final approach.",
    glyph: "command",
    color: "#ffcf5a",
    accent: "#d6c3ff",
    palette: {
      grid: "rgba(255,207,90,0.066)",
      major: "rgba(255,207,90,0.13)",
      pathOuter: "rgba(255,207,90,0.055)",
      pathGlow: "rgba(255,207,90,0.15)",
      pathBody: "rgba(70,55,21,0.56)",
      pathCore: "rgba(31,22,8,0.92)"
    },
    balance: { count: 0.92, health: 1.16, speed: 0.98, reward: 1.12, interval: 1.08, armor: 1.14 },
    enemyBias: { obelisk: 1.18, juggernaut: 1.12, phantom: 1.08 }
  }
};

const facilityTypeOrder = Object.keys(facilityTypes);

const towerDefs = [
  {
    id: "pulse",
    code: "01",
    name: "Pulse Needle",
    desc: "High-rate kinetic piercing fire.",
    cost: 75,
    damage: 24,
    range: 138,
    rate: 4.8,
    color: "#85ff91",
    role: "Needle"
  },
  {
    id: "arc",
    code: "02",
    name: "Arc Relay",
    desc: "Chains lightning between targets.",
    cost: 125,
    damage: 34,
    range: 140,
    rate: 1.05,
    color: "#7ce8ff",
    role: "Chain"
  },
  {
    id: "cryo",
    code: "03",
    name: "Cryo Prism",
    desc: "Slows and weakens enemies in range.",
    cost: 105,
    damage: 13,
    range: 132,
    rate: 1.25,
    color: "#b8f7ff",
    role: "Control"
  },
  {
    id: "mine",
    code: "04",
    name: "Mine Layer",
    desc: "Deploys proximity mines on the path.",
    cost: 95,
    damage: 78,
    range: 150,
    rate: 0.36,
    color: "#ffcf5a",
    role: "Burst"
  },
  {
    id: "jammer",
    code: "05",
    name: "Signal Jammer",
    desc: "Disrupts armor, burrow, and phasing.",
    cost: 110,
    damage: 17,
    range: 126,
    rate: 0.82,
    color: "#d2ff78",
    role: "Debuff"
  }
];

const enemyDefs = {
  crawler: {
    code: "01",
    name: "Crawler Drone",
    desc: "Fast, light armor. Swarm executor.",
    hp: 54,
    speed: 68,
    armor: 0,
    reward: 7,
    leak: 1,
    radius: 16,
    unlock: 1,
    weight: 7.8,
    color: "#9fffab"
  },
  beetle: {
    code: "02",
    name: "Shield Beetle",
    desc: "Armored front shell. Jam to crack.",
    hp: 146,
    speed: 39,
    armor: 8,
    reward: 14,
    leak: 2,
    radius: 19,
    unlock: 2,
    weight: 4.6,
    color: "#6dff7d"
  },
  slime: {
    code: "03",
    name: "Split Slime",
    desc: "Divides into smaller slimes on death.",
    hp: 106,
    speed: 47,
    armor: 1.5,
    reward: 11,
    leak: 1,
    radius: 19,
    unlock: 3,
    weight: 4.6,
    color: "#76ffb6"
  },
  worm: {
    code: "04",
    name: "Burrower Worm",
    desc: "Digs under fire. Mines still connect.",
    hp: 192,
    speed: 35,
    armor: 5,
    reward: 17,
    leak: 2,
    radius: 20,
    unlock: 4,
    weight: 3.5,
    color: "#c2ff8a"
  },
  wisp: {
    code: "05",
    name: "Static Wisp",
    desc: "Electrical anomaly. Unpredictable.",
    hp: 118,
    speed: 74,
    armor: 0.5,
    reward: 15,
    leak: 1,
    radius: 18,
    unlock: 5,
    weight: 3.5,
    color: "#7ce8ff"
  },
  juggernaut: {
    code: "06",
    name: "Juggernaut Walker",
    desc: "Slow assault unit. Heavy armor.",
    hp: 468,
    speed: 28,
    armor: 15,
    reward: 34,
    leak: 4,
    radius: 24,
    unlock: 6,
    weight: 2.35,
    color: "#ffcf5a"
  },
  phantom: {
    code: "07",
    name: "Phantom Mimic",
    desc: "Copies defeated units' appearance.",
    hp: 208,
    speed: 52,
    armor: 3,
    reward: 24,
    leak: 2,
    radius: 21,
    unlock: 7,
    weight: 2.25,
    color: "#d6c3ff"
  },
  mite: {
    code: "08",
    name: "Harvester Mite",
    desc: "Low crawler with hooked mandibles.",
    hp: 162,
    speed: 55,
    armor: 4,
    reward: 20,
    leak: 2,
    radius: 18,
    unlock: 8,
    weight: 3.45,
    color: "#86ff8f"
  },
  leech: {
    code: "09",
    name: "Void Leech",
    desc: "Armored tunnel leech with bite armor.",
    hp: 292,
    speed: 33,
    armor: 7.5,
    reward: 29,
    leak: 3,
    radius: 22,
    unlock: 9,
    weight: 2.85,
    color: "#78ff9b"
  },
  obelisk: {
    code: "10",
    name: "Obelisk Floater",
    desc: "Crystal relay drifting over fire lanes.",
    hp: 242,
    speed: 44,
    armor: 5,
    reward: 31,
    leak: 3,
    radius: 21,
    unlock: 10,
    weight: 2.55,
    color: "#b8ffcf"
  }
};

const bossDefs = {
  hive: {
    code: "B-01",
    name: "The Hive Mother",
    desc: "Brood sovereign with layered shell vents and hooked strike legs.",
    hp: 620,
    speed: 24,
    armor: 16,
    reward: 42,
    leak: 9,
    radius: 39,
    color: "#ffcf5a",
    accent: "#8dff53",
    role: "Brood Sovereign"
  },
  conduit: {
    code: "B-02",
    name: "The Conduit",
    desc: "Segmented relay horror with turbine maw and dorsal armor spines.",
    hp: 560,
    speed: 29,
    armor: 14,
    reward: 44,
    leak: 8,
    radius: 38,
    color: "#ffcf5a",
    accent: "#78ff48",
    role: "Relay Horror"
  },
  colossus: {
    code: "B-03",
    name: "The Void Colossus",
    desc: "Massive rupture chassis venting a vertical core breach.",
    hp: 720,
    speed: 18,
    armor: 20,
    reward: 48,
    leak: 11,
    radius: 43,
    color: "#ffcf5a",
    accent: "#70ff43",
    role: "Rupture Chassis"
  },
  harvester: {
    code: "B-04",
    name: "The Harvester",
    desc: "Crystalline reaper chassis with scythe pylons and a live core.",
    hp: 640,
    speed: 25,
    armor: 17,
    reward: 46,
    leak: 10,
    radius: 40,
    color: "#ffcf5a",
    accent: "#86ff57",
    role: "Crystal Reaper"
  }
};

const bossOrder = ["hive", "conduit", "colossus", "harvester"];

const operationTemplates = [
  { site: "Tokamak Reactor", suffixes: ["B", "K", "R-4", "Delta"] },
  { site: "Radar Array", suffixes: ["3", "Iris", "Q", "88"] },
  { site: "Annex Complex", suffixes: ["Aster", "Q-7", "Mosaic", "04"] },
  { site: "Bio Research Facility", suffixes: ["Sigma", "E-6", "Lumen", "04"] },
  { site: "Mining Operation", suffixes: ["17", "Gamma", "Kestrel", "F"] },
  { site: "Power Plant", suffixes: ["Grid-8", "Ion", "Cinder", "V"] },
  { site: "Communications Hub", suffixes: ["V", "31", "Juno", "Beta"] },
  { site: "Hydroponics Farm", suffixes: ["Dawn", "Pike", "Canopy", "04"] },
  { site: "Vehicle Depot", suffixes: ["6", "Orion", "Z", "12"] },
  { site: "Satellite Uplink", suffixes: ["Iris", "Zenith", "31", "Helix"] },
  { site: "Storage Facility", suffixes: ["13", "22", "C", "Helix"] },
  { site: "Command Center", suffixes: ["Prime", "Bastion", "76", "Axis"] }
];
