// ============ CANVAS ============
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

// ============ GAME STATE ============
let game = null;
let dt = 0;
let lastTime = 0;
let screenShake = { x: 0, y: 0, timer: 0, mag: 0 };
let keys = {};
let joystick = { active: false, dx: 0, dy: 0, touchId: null, ox: 0, oy: 0 };
let paused = false;
let announcement = { text: '', timer: 0, color: '#fff' };
let autoUpgrade = false;
let autoUpgradeTimer = null;

// Stars (fixed screen-space, generated once)
const stars = [];
for (let i = 0; i < 120; i++) {
  stars.push({ x: Math.random(), y: Math.random(), size: Math.random() * 1.8 + 0.5, twinkle: Math.random() * Math.PI * 2, speed: Math.random() * 2 + 1 });
}

function showAnnouncement(text, color, dur) {
  announcement.text = text;
  announcement.color = color || '#fff';
  announcement.timer = dur || 2;
}

function createGameState() {
  return {
    phase: 'playing', // playing, upgrading, dead
    time: 0,
    kills: 0,
    wave: 1,
    waveTimer: 0,
    spawnTimer: 0,
    spawnInterval: CFG.SPAWN_INTERVAL_BASE,
    bossSpawned: {},
    player: {
      x: 0, y: 0, vx: 0, vy: 0,
      hp: CFG.PLAYER_HP, maxHp: CFG.PLAYER_HP,
      speed: CFG.PLAYER_SPEED,
      radius: CFG.PLAYER_RADIUS,
      iframes: 0,
      weapons: [],
      passives: {},
      xp: 0, level: 1, xpToNext: CFG.BASE_XP_TO_LEVEL,
      magnetRange: CFG.GEM_MAGNET_RANGE,
      regen: 0,
      damageMult: 1,
      attackSpeedMult: 1,
      goopSlow: 1,
      magnetSpeedMult: 1,
      facing: 1, // 1 right, -1 left
      walkAnim: 0,
    },
    enemies: [],
    projectiles: [],
    enemyProjectiles: [],
    gems: [],
    particles: [],
    dmgTexts: [],
    lightningBolts: [],
    goopPools: [],
    worldPlants: [],
    worldRocks: [],
    colony: [],
    camera: { x: 0, y: 0 },
    weaponStats: {},
    pendingLevelUps: 0,
    gemSlurp: 0,
    colonyEra: 'station',
    eraTransition: 0,
  };
}

// ============ RESIZE ============
function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();
