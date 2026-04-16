// ============ WEAPON INSTANCES ============
function createWeapon(key) {
  const def = WEAPON_DEFS[key];
  return {
    key, level: 1,
    fireRate: def.baseRate,
    damage: def.baseDmg,
    timer: 0,
    projSpeed: def.projSpeed || 0,
    projRadius: def.projRadius || 0,
    projColor: def.projColor || '#fff',
    piercing: def.piercing || 0,
    count: def.count || 1,
    homing: def.homing || false,
    turnRate: def.turnRate || 0,
    orbitRadius: def.orbitRadius || 0,
    orbitSpeed: def.orbitSpeed || 0,
    droneCount: def.droneCount || 0,
    droneRadius: def.droneRadius || 0,
    pulseRadius: def.pulseRadius || 0,
    range: def.range || 0,
    chainCount: def.chainCount || 0,
    chainRange: def.chainRange || 0,
    slowRadius: def.slowRadius || 0,
    slowFactor: def.slowFactor || 1,
    slowDuration: def.slowDuration || 0,
    orbitAngle: 0,
    droneStates: [],
    pulseAnim: 0,
    prestiged: false,
    hardpoints: def.hardpoints || 0,
    burstSizes: def.burstSizes ? [...def.burstSizes] : [0, 0, 0],
    burstQueue: [],
  };
}

// ============ PROJECTILE FACTORY ============
function fireProjectile(x, y, vx, vy, dmg, radius, color, piercing, homing, turnRate, targetEnemy, source) {
  game.projectiles.push({
    x, y, vx, vy, damage: dmg, radius, color,
    piercing, hitEnemies: new Set(),
    lifetime: 3, homing, turnRate: turnRate || 0,
    target: targetEnemy || null,
    source: source || null,
  });
}

function fireEnemyProjectile(x, y, vx, vy, dmg, radius, color) {
  game.enemyProjectiles.push({
    x, y, vx, vy, damage: dmg, radius: radius || 5, color: color || '#f43f5e', lifetime: 2,
  });
}

// ============ WEAPON UPDATE ============
function updateWeapons() {
  const p = game.player;
  for (const w of p.weapons) {
    const def = WEAPON_DEFS[w.key];
    const rate = w.fireRate / p.attackSpeedMult;
    w.timer += dt;

    if (w.key === 'plasma') {
      // Process burst queue
      for (let b = w.burstQueue.length - 1; b >= 0; b--) {
        const burst = w.burstQueue[b];
        burst.delay -= dt;
        if (burst.delay <= 0) {
          const targets = nearestEnemies(p.x, p.y, w.range, w.hardpoints);
          const t = targets[burst.hpIdx] || targets[0];
          if (t) {
            const a = angle(p.x, p.y, t.x, t.y);
            fireProjectile(
              p.x, p.y,
              Math.cos(a) * w.projSpeed + p.vx, Math.sin(a) * w.projSpeed + p.vy,
              Math.floor(w.damage * p.damageMult), w.projRadius, w.projColor,
              w.piercing, false, 0, t, w.key
            );
          }
          w.burstQueue.splice(b, 1);
        }
      }
      // Main fire timer
      if (w.timer >= rate) {
        w.timer = 0;
        const targets = nearestEnemies(p.x, p.y, w.range, w.hardpoints);
        if (targets.length > 0) {
          playSound('shoot');
          for (let hp = 0; hp < w.hardpoints; hp++) {
            const burstSize = w.burstSizes[hp];
            if (burstSize <= 0) continue;
            const t = targets[hp] || targets[0];
            // Fire first shot immediately
            const a = angle(p.x, p.y, t.x, t.y);
            fireProjectile(
              p.x, p.y,
              Math.cos(a) * w.projSpeed + p.vx, Math.sin(a) * w.projSpeed + p.vy,
              Math.floor(w.damage * p.damageMult), w.projRadius, w.projColor,
              w.piercing, false, 0, t, w.key
            );
            // Queue remaining burst shots
            for (let s = 1; s < burstSize; s++) {
              w.burstQueue.push({ delay: s * 0.08, hpIdx: hp });
            }
          }
        }
      }
    }

    if (w.key === 'seekers') {
      if (w.timer >= rate) {
        w.timer = 0;
        const target = nearestEnemy(p.x, p.y, 500);
        if (target) {
          playSound('shoot');
          const spd = w.projSpeed * p.attackSpeedMult;
          for (let i = 0; i < w.count; i++) {
            const spread = (i - (w.count - 1) / 2) * 0.15;
            const a = angle(p.x, p.y, target.x, target.y) + spread;
            fireProjectile(
              p.x, p.y,
              Math.cos(a) * spd + p.vx, Math.sin(a) * spd + p.vy,
              Math.floor(w.damage * p.damageMult), w.projRadius, w.projColor,
              w.piercing, w.homing, w.turnRate, target, w.key
            );
          }
        }
      }
    }

    if (w.key === 'drones') {
      // Sync drone states with current count
      while (w.droneStates.length < w.droneCount) {
        w.droneStates.push({ hp: 4, maxHp: 4, regenTimer: 0 });
      }
      // Regen dead drones
      for (const ds of w.droneStates) {
        if (ds.hp <= 0) {
          ds.regenTimer -= dt;
          if (ds.regenTimer <= 0) { ds.hp = ds.maxHp; ds.regenTimer = 0; }
        }
      }
      // Orbit speed scales with thruster bonus at half rate
      const thrustBonus = p.speed / CFG.PLAYER_SPEED;
      const orbitMult = 1 + (thrustBonus - 1) * 0.5;
      w.angularStep = w.orbitSpeed * orbitMult * (p.goopSlow || 1) * dt;
      w.orbitAngle += w.angularStep;
    }

    if (w.key === 'shockwave') {
      if (w.timer >= rate) {
        w.timer = 0;
        w.pulseAnim = 1;
        playSound('shockwave');
        for (const e of game.enemies) {
          if (dist(p.x, p.y, e.x, e.y) < w.pulseRadius + e.radius) {
            damageEnemy(e, Math.floor(w.damage * p.damageMult), w.key);
          }
        }
      }
      if (w.pulseAnim > 0) w.pulseAnim = Math.max(0, w.pulseAnim - dt * 3);
    }

    if (w.key === 'lightning') {
      if (w.timer >= rate) {
        w.timer = 0;
        const first = nearestEnemy(p.x, p.y, 300);
        if (first) {
          playSound('zap');
          let chain = [first];
          let current = first;
          damageEnemy(current, Math.floor(w.damage * p.damageMult), w.key);
          for (let c = 1; c < w.chainCount; c++) {
            let best = null, bestD = w.chainRange;
            for (const e of game.enemies) {
              if (chain.includes(e)) continue;
              const d = dist(current.x, current.y, e.x, e.y);
              if (d < bestD) { bestD = d; best = e; }
            }
            if (best) {
              spawnLightningBolt(current.x, current.y, best.x, best.y);
              damageEnemy(best, Math.floor(w.damage * p.damageMult * Math.pow(0.65, c)), w.key);
              chain.push(best);
              current = best;
            } else break;
          }
          spawnLightningBolt(p.x, p.y, first.x, first.y);
        }
      }
    }

    if (w.key === 'cryo') {
      if (w.timer >= rate) {
        w.timer = 0;
        w.pulseAnim = 1;
        for (const e of game.enemies) {
          if (dist(p.x, p.y, e.x, e.y) < w.slowRadius + e.radius) {
            e.slowTimer = w.slowDuration;
            e.slowFactor = w.slowFactor;
            damageEnemy(e, Math.floor(w.damage * p.damageMult), w.key);
          }
        }
      }
      if (w.pulseAnim > 0) w.pulseAnim = Math.max(0, w.pulseAnim - dt * 2);
    }
  }
}

function spawnLightningBolt(x1, y1, x2, y2) {
  // Build jagged bolt path
  const segments = 8;
  const points = [{ x: x1, y: y1 }];
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  const nx = -dy / len, ny = dx / len; // perpendicular
  for (let i = 1; i < segments; i++) {
    const t = i / segments;
    const jitter = rand(-len * 0.12, len * 0.12);
    points.push({
      x: lerp(x1, x2, t) + nx * jitter,
      y: lerp(y1, y2, t) + ny * jitter,
    });
  }
  points.push({ x: x2, y: y2 });
  game.lightningBolts.push({ points, lifetime: 0.25, maxLife: 0.25 });
  // Spark particles at endpoints
  for (let i = 0; i < 3; i++) {
    game.particles.push({
      x: x2 + rand(-6, 6), y: y2 + rand(-6, 6),
      vx: rand(-40, 40), vy: rand(-40, 40),
      color: '#e0f7ff', size: rand(2, 4),
      lifetime: 0.2, maxLife: 0.2,
    });
  }
}
