// ============ CONFIGURATION ============
const CFG = {
  PLAYER_SPEED: 220,
  PLAYER_HP: 100,
  PLAYER_RADIUS: 16,
  PLAYER_IFRAMES: 0.4,
  GEM_MAGNET_RANGE: 90,
  GEM_MAGNET_SPEED: 600,
  GEM_BASE_XP: 1,
  BASE_XP_TO_LEVEL: 7,
  XP_LEVEL_SCALE: 1.10,
  SPAWN_INTERVAL_BASE: 0.9,
  SPAWN_INTERVAL_MIN: 0.12,
  SPAWN_RAMP_RATE: 0.03,
  DIFFICULTY_TIMER: 30,
  WORLD_PLANT_COUNT: 200,
  WORLD_ROCK_COUNT: 60,
  DAMAGE_FLASH_DUR: 0.1,
  SCREEN_SHAKE_DUR: 0.15,
  SCREEN_SHAKE_MAG: 6,
};

// ============ COLORS ============
const PAL = {
  bg: '#0b1026',
  ground: '#0d2e2a',
  groundLight: '#134a3f',
  plant1: '#1abc9c',
  plant2: '#8e44ad',
  plant3: '#e67e22',
  glow: '#2ecc71',
  playerBody: '#ff6b35',
  playerVisor: '#74b9ff',
  playerSuit: '#dfe6e9',
  hpBar: '#e74c3c',
  xpBar: '#f1c40f',
  gem: '#5ec4b6',
  gemGlow: 'rgba(94,196,182,0.2)',
  dmgText: '#ff6b6b',
  healText: '#2ecc71',
};

// ============ ENEMY DEFINITIONS ============
const ENEMY_TYPES = {
  spore: {
    name: 'Spore', hp: 15, speed: 100, radius: 10, damage: 8, xp: 1,
    color: '#a855f7', eyeColor: '#fff', pupilColor: '#1a1a2e',
    outline: '#7c3aed',
  },
  beetle: {
    name: 'Beetle', hp: 40, speed: 65, radius: 16, damage: 12, xp: 2,
    color: '#22c55e', eyeColor: '#fff', pupilColor: '#1a1a2e',
    outline: '#16a34a',
  },
  bloomer: {
    name: 'Bloomer', hp: 25, speed: 50, radius: 14, damage: 15, xp: 3,
    color: '#f43f5e', eyeColor: '#fff', pupilColor: '#1a1a2e',
    outline: '#e11d48', ranged: true, fireRate: 3.5, projSpeed: 160,
  },
  brute: {
    name: 'Brute', hp: 120, speed: 40, radius: 24, damage: 20, xp: 5,
    color: '#78716c', eyeColor: '#fbbf24', pupilColor: '#1a1a2e',
    outline: '#57534e',
  },
  stinger: {
    name: 'Stinger', hp: 30, speed: 140, radius: 9, damage: 6, xp: 3,
    color: '#facc15', eyeColor: '#fff', pupilColor: '#1a1a2e',
    outline: '#ca8a04', ranged: true, fireRate: 1.8, projSpeed: 300, projRadius: 3, projColor: '#fde047',
  },
  gloop: {
    name: 'Gloop', hp: 50, speed: 45, radius: 18, damage: 10, xp: 2,
    color: '#65a30d', eyeColor: '#d9f99d', pupilColor: '#1a1a2e',
    outline: '#4d7c0f',
  },
  juggernaut: {
    name: 'Juggernaut', hp: 3000, speed: 50, radius: 48, damage: 25, xp: 10,
    color: '#64748b', eyeColor: '#f87171', pupilColor: '#1a1a2e',
    outline: '#94a3b8', orbiter: true, orbitDist: 380,
  },
  kamikaze: {
    name: 'Kamikaze', hp: 12, speed: 320, radius: 6, damage: 10, xp: 1,
    color: '#fb923c', eyeColor: '#fff', pupilColor: '#1a1a2e',
    outline: '#ea580c',
  },
  queen: {
    name: 'Hive Queen', hp: 18000, speed: 55, radius: 44, damage: 50, xp: 25,
    color: '#c026d3', eyeColor: '#fbbf24', pupilColor: '#1a1a2e',
    outline: '#a21caf', boss: true, orbiter: true, orbitDist: 350,
  },
};

// ============ WEAPON DEFINITIONS ============
const WEAPON_DEFS = {
  plasma: {
    name: 'Plasma Bolt', icon: '\u{1F52B}',
    desc: 'PDC hardpoints fire at nearby enemies',
    baseRate: 0.45, baseDmg: 8, projSpeed: 500, projRadius: 2, projColor: '#74b9ff',
    piercing: 0, hardpoints: 1, burstSizes: [1, 0, 0], range: 250,
    levelUp: (w) => {
      w.damage += 2; w.fireRate *= 0.9;
      if (w.level === 2) w.burstSizes[0] = 2;
      if (w.level === 3) { w.hardpoints = 2; w.burstSizes[1] = 1; }
      if (w.level === 4) w.burstSizes[1] = 2;
      if (w.level === 5) { w.hardpoints = 3; w.burstSizes[2] = 1; }
      if (w.level === 6) w.burstSizes[2] = 2;
    },
    prestige: {
      name: 'Nova Cannon', icon: '\u{1F31F}', desc: 'Triple-mount rapid-fire defense cannons',
      evolve: (w) => {
        w.damage += 4; w.projSpeed *= 1.3; w.fireRate *= 0.85; w.projColor = '#ffd700';
        w.range = Math.round(w.range * 1.25);
        for (let i = 0; i < 3; i++) w.burstSizes[i] *= 2;
      },
      levelUp: (w) => {
        w.damage += 3; w.fireRate *= 0.94;
        if (w.level % 4 === 0) for (let i = 0; i < w.hardpoints; i++) w.burstSizes[i]++;
      },
    },
  },
  drones: {
    name: 'Orbital Drones', icon: '\u{1F6F8}',
    desc: 'Drones orbit you, damaging enemies on contact',
    baseRate: 0.5, baseDmg: 18, orbitRadius: 70, orbitSpeed: 2.4, droneCount: 2, droneRadius: 5,
    levelUp: (w) => { w.damage += 8; w.droneCount++; if (w.level >= 4) w.orbitRadius += 15; },
    prestige: {
      name: 'Sentinel Swarm', icon: '\u{1F300}', desc: 'A devastating swarm of autonomous attack drones',
      evolve: (w) => { w.damage += 25; w.droneCount += 3; w.orbitRadius += 30; w.orbitSpeed += 1; w.droneRadius += 3; },
      levelUp: (w) => { w.damage += 14; w.droneCount += 2; w.orbitRadius += 10; },
    },
  },
  shockwave: {
    name: 'Shockwave', icon: '\u{1F4A5}',
    desc: 'Periodic energy pulse damages nearby enemies',
    baseRate: 2.0, baseDmg: 40, pulseRadius: 100,
    levelUp: (w) => { w.damage += 18; w.pulseRadius += 20; w.fireRate *= 0.88; },
    prestige: {
      name: 'Cataclysm', icon: '\u{1F30B}', desc: 'Massive seismic detonations obliterate everything nearby',
      evolve: (w) => { w.damage += 55; w.pulseRadius += 60; w.fireRate *= 0.7; },
      levelUp: (w) => { w.damage += 30; w.pulseRadius += 25; w.fireRate *= 0.9; },
    },
  },
  seekers: {
    name: 'Seeker Missiles', icon: '\u{1F3AF}',
    desc: 'Homing missiles that track enemies',
    baseRate: 1.2, baseDmg: 25, projSpeed: 420, projRadius: 6, projColor: '#ff6b6b',
    count: 1, homing: true, turnRate: 5,
    levelUp: (w) => { w.damage += 10; if (w.level % 2 === 0) w.count++; w.fireRate *= 0.92; },
    prestige: {
      name: 'Hellfire Barrage', icon: '\u{2604}\u{FE0F}', desc: 'Relentless volleys of hyper-tracking warheads',
      evolve: (w) => { w.damage += 15; w.count += 1; w.fireRate *= 0.75; w.projSpeed += 120; w.turnRate += 3; },
      levelUp: (w) => { w.damage += 10; if (w.level % 3 === 0) w.count++; w.fireRate *= 0.94; },
    },
  },
  lightning: {
    name: 'Chain Lightning', icon: '\u{26A1}',
    desc: 'Lightning arcs between nearby enemies',
    baseRate: 1.5, baseDmg: 30, chainCount: 3, chainRange: 120,
    levelUp: (w) => { w.damage += 10; if (w.level % 2 === 0) w.chainCount++; w.chainRange += 10; w.fireRate *= 0.92; },
    prestige: {
      name: 'Storm Wrath', icon: '\u{1F329}\u{FE0F}', desc: 'Apocalyptic lightning storm that decimates hordes',
      evolve: (w) => { w.damage += 20; w.chainCount += 2; w.chainRange += 30; w.fireRate *= 0.8; },
      levelUp: (w) => { w.damage += 12; if (w.level % 2 === 0) w.chainCount++; w.chainRange += 10; w.fireRate *= 0.92; },
    },
  },
  cryo: {
    name: 'Cryo Field', icon: '\u{2744}\u{FE0F}',
    desc: 'Slows enemies around you',
    baseRate: 2.0, baseDmg: 20, slowRadius: 120, slowFactor: 0.4, slowDuration: 2,
    levelUp: (w) => { w.damage += 12; w.slowRadius += 25; w.slowFactor = Math.max(0.15, w.slowFactor - 0.05); w.fireRate *= 0.88; },
    prestige: {
      name: 'Absolute Zero', icon: '\u{1F9CA}', desc: 'Deep-freeze aura that shatters frozen enemies',
      evolve: (w) => { w.damage += 35; w.slowRadius += 60; w.slowFactor = 0.1; w.slowDuration += 1; w.fireRate *= 0.7; },
      levelUp: (w) => { w.damage += 18; w.slowRadius += 30; w.slowFactor = Math.max(0.05, w.slowFactor - 0.02); w.fireRate *= 0.87; },
    },
  },
};

// ============ PASSIVE UPGRADE DEFINITIONS ============
const PASSIVE_DEFS = {
  armor: { name: 'Colony Armor', icon: '\u{1F6E1}\u{FE0F}', desc: '+20 Max HP', maxLevel: Infinity, apply: (p, lv) => { p.maxHp += 20; p.hp = Math.min(p.hp + 20, p.maxHp); } },
  speed: { name: 'Thruster Boost', icon: '\u{1F680}', desc: '+12% Move Speed', maxLevel: Infinity, apply: (p, lv) => { p.speed *= 1.12; } },
  magnet: { name: 'Bio Magnet', icon: '\u{1F9F2}', desc: '+40% XP Range, +25% Pull Speed', maxLevel: Infinity, apply: (p, lv) => { p.magnetRange *= 1.4; p.magnetSpeedMult *= 1.25; } },
  regen: { name: 'Nano Repair', icon: '\u{1FA79}', desc: '+2 HP/sec Regeneration', maxLevel: Infinity, apply: (p, lv) => { p.regen += 2; } },
  might: { name: 'Overcharge', icon: '\u{1F4AA}', desc: '+15% All Damage', maxLevel: Infinity, apply: (p, lv) => { p.damageMult *= 1.15; } },
  rapid: { name: 'Rapid Deploy', icon: '\u{23E9}', desc: '+12% Attack Speed', maxLevel: Infinity, apply: (p, lv) => { p.attackSpeedMult *= 1.12; } },
};

// ============ WEAPON NAMES (for pause stats) ============
const WEAPON_NAMES = {
  plasma: 'Plasma Bolt', seekers: 'Seeker Missiles', drones: 'Orbital Drones',
  shockwave: 'Shockwave', lightning: 'Chain Lightning', cryo: 'Cryo Field', colony: 'Colony Defense',
};
