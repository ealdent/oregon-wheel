// ============ ENEMY SPAWNING ============
function spawnEnemy(typeKey, px, py, nearRange) {
  const def = ENEMY_TYPES[typeKey];
  if (!def) return;
  const a = rand(0, Math.PI * 2);
  const r = nearRange != null ? rand(nearRange * 0.5, nearRange) : rand(500, 700);
  const x = px + Math.cos(a) * r;
  const y = py + Math.sin(a) * r;
  const hpScale = 1 + game.time * 0.008;
  game.enemies.push({
    type: typeKey,
    x, y, vx: 0, vy: 0,
    hp: Math.floor(def.hp * hpScale),
    maxHp: Math.floor(def.hp * hpScale),
    radius: def.radius,
    damage: def.damage + Math.floor(game.time * 0.03),
    speed: def.speed,
    color: def.color,
    outline: def.outline,
    eyeColor: def.eyeColor,
    pupilColor: def.pupilColor,
    flashTimer: 0,
    fireTimer: def.ranged ? rand(0, def.fireRate) : 0,
    anim: rand(0, Math.PI * 2),
    slowTimer: 0,
    slowFactor: 1,
    boss: def.boss || false,
    orbitDir: def.orbiter ? (Math.random() < 0.5 ? 1 : -1) : 0,
  });
}

function spawnBoss(px, py) {
  const def = ENEMY_TYPES.queen;
  const a = rand(0, Math.PI * 2);
  const r = 600;
  const wave = game.wave;
  const hpScale = 1 + (wave - 1) * 0.8;
  game.enemies.push({
    type: 'queen',
    x: px + Math.cos(a) * r, y: py + Math.sin(a) * r,
    vx: 0, vy: 0,
    hp: Math.floor(def.hp * hpScale),
    maxHp: Math.floor(def.hp * hpScale),
    radius: def.radius,
    damage: def.damage,
    speed: def.speed,
    color: def.color,
    outline: def.outline,
    eyeColor: def.eyeColor,
    pupilColor: def.pupilColor,
    flashTimer: 0,
    fireTimer: 0,
    anim: 0,
    slowTimer: 0,
    slowFactor: 1,
    boss: true,
    railTimer: 0,
    railCharging: false,
    railChargeTime: 0,
    railAngle: 0,
    railBeamTimer: 0,
    railCooldown: 2.5,
    barrageTimer: 0,
    orbitDir: Math.random() < 0.5 ? 1 : -1,
  });
  triggerScreenShake(10, 0.4);
  spawnParticles(px, py, '#c026d3', 30, 200);
}

// ============ PARTICLES ============
function spawnParticles(x, y, color, count, spread, sizeBase) {
  for (let i = 0; i < count; i++) {
    const a = rand(0, Math.PI * 2);
    const s = rand(20, spread || 100);
    game.particles.push({
      x, y,
      vx: Math.cos(a) * s, vy: Math.sin(a) * s,
      color, size: rand(sizeBase || 2, (sizeBase || 2) + 4),
      lifetime: rand(0.3, 0.8),
      maxLife: 0.8,
    });
  }
}

function spawnDmgText(x, y, text, color) {
  game.dmgTexts.push({
    x, y: y - 10, text, color: color || PAL.dmgText,
    lifetime: 0.8, vy: -60,
  });
}

// ============ GEM FACTORY ============
function spawnGem(x, y, xp) {
  game.gems.push({
    x: x + rand(-10, 10), y: y + rand(-10, 10),
    xp, radius: 4 + xp * 0.7,
    vx: rand(-50, 50), vy: rand(-50, 50),
    lifetime: 30,
  });
}

// ============ FIND NEAREST ENEMY ============
function nearestEnemy(x, y, maxDist) {
  let best = null, bestD = maxDist || Infinity;
  for (const e of game.enemies) {
    const d = dist(x, y, e.x, e.y);
    if (d < bestD) { bestD = d; best = e; }
  }
  return best;
}

function nearestEnemies(x, y, maxDist, n) {
  const results = [];
  for (const e of game.enemies) {
    if (e.hp <= 0) continue;
    const d = dist(x, y, e.x, e.y);
    if (d < maxDist) results.push({ enemy: e, dist: d });
  }
  results.sort((a, b) => a.dist - b.dist);
  return results.slice(0, n).map(r => r.enemy);
}
