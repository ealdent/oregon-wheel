"use strict";

const BOARD = { width: 1000, height: 680, pathWidth: 76 };
const MAX_TOWER_LEVEL = 12;
const TARGET_MODES = ["FIRST", "STRONG", "NEAR"];
const SPEEDS = [1, 2, 3];
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
const pathPoints = [
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
  { site: "Tokamak Facility", suffixes: ["B", "K", "R-4", "Delta"] },
  { site: "Orbital Cargo Facility", suffixes: ["13", "22", "C", "Helix"] },
  { site: "Mass Driver Array", suffixes: ["9", "V", "Aster", "D"] },
  { site: "Cryogenic Vault", suffixes: ["Sigma", "E-6", "Lumen", "04"] },
  { site: "Null Foundry", suffixes: ["17", "Gamma", "Kestrel", "F"] },
  { site: "Deep Radar Annex", suffixes: ["3", "Iris", "Q", "88"] },
  { site: "Plasma Intake Plant", suffixes: ["C", "11", "Morrow", "T"] },
  { site: "Catenary Relay Station", suffixes: ["V", "31", "Juno", "Beta"] },
  { site: "Subsurface Data Mine", suffixes: ["L-2", "19", "Nadir", "H"] },
  { site: "Aperture Dockyard", suffixes: ["6", "Orion", "Z", "12"] }
];
