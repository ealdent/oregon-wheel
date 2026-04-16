// ============ UPDATE ============
function update() {
  if (!game || game.phase !== 'playing' || paused) return;
  const g = game;
  const p = g.player;

  // Time & waves
  g.time += dt;
  g.waveTimer += dt;
  if (g.waveTimer >= CFG.DIFFICULTY_TIMER) {
    g.waveTimer = 0;
    g.wave++;
    g.spawnInterval = Math.max(CFG.SPAWN_INTERVAL_MIN, g.spawnInterval - CFG.SPAWN_RAMP_RATE * g.wave);
    showAnnouncement(`WAVE ${g.wave}`, '#4ecdc4', 2.5);
    // Colony era transition
    const newEra = getColonyEra();
    if (newEra !== g.colonyEra) {
      g.colonyEra = newEra;
      g.eraTransition = 3.0;
      const eraNames = { station: 'ORBITAL STATION', airship: 'ATMOSPHERIC DESCENT', fob: 'FOB ESTABLISHED' };
      setTimeout(() => { if (game && game.phase === 'playing') showAnnouncement(eraNames[newEra], '#ffd700', 3); }, 2600);
    }
    // Boss every 5 waves
    if (g.wave % 5 === 0 && !g.bossSpawned[g.wave]) {
      g.bossSpawned[g.wave] = true;
      setTimeout(() => { if (game && game.phase === 'playing') { showAnnouncement('BOSS INCOMING', '#e74c3c', 2); spawnBoss(game.player.x, game.player.y); } }, 1500);
    }
  }

  // Update goop pools
  for (let i = g.goopPools.length - 1; i >= 0; i--) {
    g.goopPools[i].timer -= dt;
    if (g.goopPools[i].timer <= 0) { g.goopPools.splice(i, 1); }
  }

  // Check if player is in goop — slows ship and drone orbit
  let goopSlow = 1;
  for (const pool of g.goopPools) {
    if (dist(p.x, p.y, pool.x, pool.y) < pool.radius) {
      const factor = Math.max(0.2, 1 - pool.intensity * 0.1);
      goopSlow = Math.min(goopSlow, factor);
      pool.timer -= dt * 3;
    }
  }
  p.goopSlow = goopSlow;

  // Player movement
  const input = getInputDir();
  p.vx = input.dx * p.speed * goopSlow;
  p.vy = input.dy * p.speed * goopSlow;
  p.x += p.vx * dt;
  p.y += p.vy * dt;
  if (input.dx !== 0) p.facing = input.dx > 0 ? 1 : -1;
  if (Math.abs(p.vx) > 10 || Math.abs(p.vy) > 10) p.walkAnim += dt * 8;
  p.iframes = Math.max(0, p.iframes - dt);

  // Regen
  if (p.regen > 0 && p.hp < p.maxHp) {
    p.hp = Math.min(p.maxHp, p.hp + p.regen * dt);
  }

  // Camera
  g.camera.x = lerp(g.camera.x, p.x, 5 * dt);
  g.camera.y = lerp(g.camera.y, p.y, 5 * dt);

  // Spawn enemies
  g.spawnTimer += dt;
  if (g.spawnTimer >= g.spawnInterval) {
    g.spawnTimer = 0;
    const types = getSpawnTypes();
    const count = 1 + Math.floor(g.wave * 0.6);
    for (let i = 0; i < count; i++) {
      spawnEnemy(types[randInt(0, types.length - 1)], p.x, p.y);
    }
  }

  // Spawn juggernaut carriers (separate low-rate timer)
  if (g.wave >= 15) {
    g.juggernautTimer = (g.juggernautTimer || 0) + dt;
    const jSpawnRate = Math.max(8, 20 - g.wave * 0.5);
    if (g.juggernautTimer >= jSpawnRate) {
      g.juggernautTimer = 0;
      const jCount = 1 + Math.floor((g.wave - 15) / 5);
      for (let j = 0; j < jCount; j++) spawnEnemy('juggernaut', p.x, p.y);
    }
  }

  // Update enemies
  for (let i = g.enemies.length - 1; i >= 0; i--) {
    const e = g.enemies[i];
    if (e.hp <= 0) { g.enemies.splice(i, 1); continue; }

    e.flashTimer = Math.max(0, e.flashTimer - dt);
    e.slowTimer = Math.max(0, e.slowTimer - dt);
    e.anim += dt * 3;
    const speedMult = e.slowTimer > 0 ? e.slowFactor : 1;

    // Movement
    const eDist = dist(e.x, e.y, p.x, p.y);
    const toPlayer = angle(e.x, e.y, p.x, p.y);
    const eDef2 = ENEMY_TYPES[e.type];
    if (eDef2.orbiter) {
      // Orbital movement: maintain distance, strafe around player
      const orbitDist = eDef2.orbitDist || 350;
      const tangent = toPlayer + (Math.PI / 2) * (e.orbitDir || 1);
      let radial = (eDist - orbitDist) / 200;
      radial = Math.max(-1, Math.min(1, radial));
      const tangWeight = 1 - Math.abs(radial) * 0.3;
      let mx = Math.cos(toPlayer) * radial + Math.cos(tangent) * tangWeight;
      let my = Math.sin(toPlayer) * radial + Math.sin(tangent) * tangWeight;
      const mag = Math.sqrt(mx * mx + my * my) || 1;
      e.vx = (mx / mag) * e.speed * speedMult;
      e.vy = (my / mag) * e.speed * speedMult;
    } else {
      // Beeline toward player
      e.vx = Math.cos(toPlayer) * e.speed * speedMult;
      e.vy = Math.sin(toPlayer) * e.speed * speedMult;
    }
    e.x += e.vx * dt;
    e.y += e.vy * dt;

    // Ranged attack
    const eDef = ENEMY_TYPES[e.type];
    if (eDef.ranged) {
      e.fireTimer += dt;
      if (e.fireTimer >= eDef.fireRate && dist(e.x, e.y, p.x, p.y) < 400) {
        e.fireTimer = 0;
        const pa = angle(e.x, e.y, p.x, p.y);
        fireEnemyProjectile(e.x, e.y, Math.cos(pa) * eDef.projSpeed, Math.sin(pa) * eDef.projSpeed, e.damage, eDef.projRadius, eDef.projColor);
      }
    }

    // Boss spawns minions + plasma barrage
    if (e.boss) {
      e.fireTimer += dt;
      if (e.fireTimer >= 2) {
        e.fireTimer = 0;
        for (let j = 0; j < 5; j++) spawnEnemy('spore', e.x, e.y);
      }
      // Plasma barrage — blockable by drones but overwhelming
      e.barrageTimer = (e.barrageTimer || 0) + dt;
      if (e.barrageTimer >= 3.5) {
        e.barrageTimer = 0;
        const baseAngle = angle(e.x, e.y, p.x, p.y);
        const spread = Math.PI / 3;
        const count = 12;
        for (let j = 0; j < count; j++) {
          const ba = baseAngle - spread / 2 + (spread * j / (count - 1));
          const spd = 220;
          fireEnemyProjectile(e.x, e.y, Math.cos(ba) * spd, Math.sin(ba) * spd, 15, 6, '#c026d3');
        }
        playSound('zap');
      }
    }

    // Juggernaut spawns kamikaze drones — constant bombardment
    if (e.type === 'juggernaut') {
      e.fireTimer += dt;
      if (e.fireTimer >= 0.7) {
        e.fireTimer = 0;
        for (let j = 0; j < 4; j++) spawnEnemy('kamikaze', e.x, e.y, 40);
      }
    }

    // Boss railgun
    if (e.boss) {
      e.railBeamTimer = Math.max(0, e.railBeamTimer - dt);
      if (!e.railCharging) {
        e.railTimer += dt;
        if (e.railTimer >= e.railCooldown) {
          e.railCharging = true;
          e.railChargeTime = 0;
          e.railAngle = angle(e.x, e.y, p.x, p.y);
          playSound('zap');
        }
      } else {
        e.railChargeTime += dt;
        // Track player with limited angular speed — compensates for boss orbit
        // but can't keep up with a fast-moving player strafing transversely
        const targetAngle = angle(e.x, e.y, p.x, p.y);
        let angleDiff = targetAngle - e.railAngle;
        while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
        while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
        const maxTrack = 0.4 * dt; // 0.4 rad/s
        e.railAngle += Math.max(-maxTrack, Math.min(maxTrack, angleDiff));
        if (e.railChargeTime >= 1.5) {
          // Fire the railgun - hitscan beam bypasses drones
          e.railCharging = false;
          e.railTimer = 0;
          e.railBeamTimer = 0.2;
          triggerScreenShake(12, 0.3);
          playSound('shockwave');
          // Point-to-line hit detection
          const beamLen = 900;
          const bx = Math.cos(e.railAngle), by = Math.sin(e.railAngle);
          const dx = p.x - e.x, dy = p.y - e.y;
          const along = dx * bx + dy * by;
          if (along > 0 && along < beamLen) {
            const perpDist = Math.abs(dx * by - dy * bx);
            if (perpDist < p.radius + 20) {
              damagePlayer(40);
            }
          }
        }
      }
    }

    // Contact damage
    if (dist(e.x, e.y, p.x, p.y) < e.radius + p.radius) {
      damagePlayer(e.damage);
    }
  }

  // Update weapons
  updateWeapons();

  // Planetary defenses
  updatePlanetaryDefenses();

  // Drone collision (swept arc to handle high orbit speed)
  for (const w of p.weapons) {
    if (w.key !== 'drones') continue;
    const step = w.angularStep || 0;
    for (let d = 0; d < w.droneCount; d++) {
      const ds = w.droneStates[d];
      if (ds && ds.hp <= 0) continue; // dead drone
      const da = w.orbitAngle + (d / w.droneCount) * Math.PI * 2;
      const dx = p.x + Math.cos(da) * w.orbitRadius;
      const dy = p.y + Math.sin(da) * w.orbitRadius;
      for (const e of g.enemies) {
        if (e.hp <= 0) continue;
        const hitR = w.droneRadius + e.radius;
        // Fast path: point check at current position
        if (dist(dx, dy, e.x, e.y) < hitR) {
          damageEnemy(e, Math.floor(w.damage * p.damageMult * dt * 12), 'drones');
        } else if (step > 0.05) {
          // Swept arc check: is enemy within orbit band and angular sweep?
          const d2e = dist(p.x, p.y, e.x, e.y);
          if (Math.abs(d2e - w.orbitRadius) < hitR) {
            const ea = Math.atan2(e.y - p.y, e.x - p.x);
            let diff = ea - (da - step);
            diff = ((diff % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
            if (diff < step + hitR / w.orbitRadius) {
              damageEnemy(e, Math.floor(w.damage * p.damageMult * dt * 12), 'drones');
            }
          }
        }
      }
    }
  }

  // Update player projectiles
  for (let i = g.projectiles.length - 1; i >= 0; i--) {
    const pr = g.projectiles[i];
    pr.lifetime -= dt;
    if (pr.lifetime <= 0) { g.projectiles.splice(i, 1); continue; }

    // Homing
    if (pr.homing && pr.target) {
      if (pr.target.hp <= 0) pr.target = nearestEnemy(pr.x, pr.y, 400);
      if (pr.target) {
        const desired = angle(pr.x, pr.y, pr.target.x, pr.target.y);
        const current = Math.atan2(pr.vy, pr.vx);
        let diff = desired - current;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        const turn = clamp(diff, -pr.turnRate * dt, pr.turnRate * dt);
        const newAngle = current + turn;
        const spd = Math.sqrt(pr.vx * pr.vx + pr.vy * pr.vy);
        pr.vx = Math.cos(newAngle) * spd;
        pr.vy = Math.sin(newAngle) * spd;
      }
    }

    pr.x += pr.vx * dt;
    pr.y += pr.vy * dt;

    // Hit enemies
    for (const e of g.enemies) {
      if (e.hp <= 0 || pr.hitEnemies.has(e)) continue;
      if (dist(pr.x, pr.y, e.x, e.y) < pr.radius + e.radius) {
        pr.hitEnemies.add(e);
        damageEnemy(e, pr.damage, pr.source);
        spawnParticles(pr.x, pr.y, pr.color, 3, 40, 1);
        if (pr.piercing > 0) {
          pr.piercing--;
        } else {
          pr.lifetime = 0;
        }
        break;
      }
    }
  }

  // Update enemy projectiles
  for (let i = g.enemyProjectiles.length - 1; i >= 0; i--) {
    const pr = g.enemyProjectiles[i];
    pr.lifetime -= dt;
    if (pr.lifetime <= 0) { g.enemyProjectiles.splice(i, 1); continue; }
    pr.x += pr.vx * dt;
    pr.y += pr.vy * dt;
    // Drone bullet absorption — drones intercept enemy projectiles (swept arc aware)
    let absorbed = false;
    for (const w of p.weapons) {
      if (w.key !== 'drones') continue;
      const step = w.angularStep || 0;
      for (let d = 0; d < w.droneCount; d++) {
        const ds = w.droneStates[d];
        if (ds && ds.hp <= 0) continue; // dead drone can't absorb
        const da = w.orbitAngle + (d / w.droneCount) * Math.PI * 2;
        const drx = p.x + Math.cos(da) * w.orbitRadius;
        const dry = p.y + Math.sin(da) * w.orbitRadius;
        const hitR = w.droneRadius + pr.radius;
        if (dist(pr.x, pr.y, drx, dry) < hitR) {
          absorbed = true;
        } else if (step > 0.05) {
          const d2pr = dist(p.x, p.y, pr.x, pr.y);
          if (Math.abs(d2pr - w.orbitRadius) < hitR) {
            const pa = Math.atan2(pr.y - p.y, pr.x - p.x);
            let diff = pa - (da - step);
            diff = ((diff % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
            if (diff < step + hitR / w.orbitRadius) absorbed = true;
          }
        }
        if (absorbed) {
          // Drone takes a hit — dies after 4 hits, 5s regen
          if (ds) { ds.hp--; if (ds.hp <= 0) ds.regenTimer = 5; }
          spawnParticles(pr.x, pr.y, '#74b9ff', 4, 50, 1);
          break;
        }
      }
      if (absorbed) break;
    }
    if (absorbed) { g.enemyProjectiles.splice(i, 1); continue; }
    if (dist(pr.x, pr.y, p.x, p.y) < pr.radius + p.radius) {
      damagePlayer(pr.damage);
      g.enemyProjectiles.splice(i, 1);
    }
  }

  // Update gem slurp timer
  if (g.gemSlurp > 0) g.gemSlurp -= dt;

  // Update gems
  for (let i = g.gems.length - 1; i >= 0; i--) {
    const gem = g.gems[i];
    gem.lifetime -= dt;
    gem.vx *= 0.95;
    gem.vy *= 0.95;
    gem.x += gem.vx * dt;
    gem.y += gem.vy * dt;

    const d = dist(gem.x, gem.y, p.x, p.y);
    if (g.gemSlurp > 0) {
      // Boss kill: slurp ALL gems toward player at high speed
      const a = angle(gem.x, gem.y, p.x, p.y);
      const pull = CFG.GEM_MAGNET_SPEED * 3;
      gem.vx = Math.cos(a) * pull;
      gem.vy = Math.sin(a) * pull;
    } else if (d < p.magnetRange) {
      const a = angle(gem.x, gem.y, p.x, p.y);
      const closeness = 1 - d / p.magnetRange;
      const pull = CFG.GEM_MAGNET_SPEED * p.magnetSpeedMult * (closeness * closeness + 0.3);
      gem.vx = Math.cos(a) * pull;
      gem.vy = Math.sin(a) * pull;
    }

    if (d < p.radius + gem.radius) {
      p.xp += gem.xp;
      playSound('pickup');
      spawnParticles(gem.x, gem.y, PAL.gem, 4, 30, 1);
      g.gems.splice(i, 1);

      // Level up
      while (p.xp >= p.xpToNext) {
        p.xp -= p.xpToNext;
        p.level++;
        p.maxHp += 1;
        p.hp = Math.min(p.hp + 1, p.maxHp);
        if (p.level % 5 === 0) p.regen += 1;
        p.xpToNext = Math.floor(CFG.BASE_XP_TO_LEVEL * Math.pow(CFG.XP_LEVEL_SCALE, p.level - 1));
        if (game.phase === 'upgrading') {
          game.pendingLevelUps++;
        } else {
          showUpgradeScreen();
        }
      }
    } else if (gem.lifetime <= 0) {
      g.gems.splice(i, 1);
    }
  }

  // Update particles
  for (let i = g.particles.length - 1; i >= 0; i--) {
    const pt = g.particles[i];
    pt.lifetime -= dt;
    if (pt.lifetime <= 0) { g.particles.splice(i, 1); continue; }
    pt.x += pt.vx * dt;
    pt.y += pt.vy * dt;
    pt.vx *= 0.96;
    pt.vy *= 0.96;
  }

  // Update lightning bolts
  for (let i = g.lightningBolts.length - 1; i >= 0; i--) {
    g.lightningBolts[i].lifetime -= dt;
    if (g.lightningBolts[i].lifetime <= 0) g.lightningBolts.splice(i, 1);
  }

  // Update damage texts
  for (let i = g.dmgTexts.length - 1; i >= 0; i--) {
    const t = g.dmgTexts[i];
    t.lifetime -= dt;
    if (t.lifetime <= 0) { g.dmgTexts.splice(i, 1); continue; }
    t.y += t.vy * dt;
  }

  // Screen shake
  if (screenShake.timer > 0) {
    screenShake.timer -= dt;
    screenShake.x = rand(-screenShake.mag, screenShake.mag);
    screenShake.y = rand(-screenShake.mag, screenShake.mag);
  } else {
    screenShake.x = 0;
    screenShake.y = 0;
  }

  // Announcement
  if (announcement.timer > 0) announcement.timer -= dt;
  if (g.eraTransition > 0) g.eraTransition -= dt;

  // Update UI
  updateUI();
}

// ============ UPDATE UI ============
function updateUI() {
  const p = game.player;
  document.getElementById('hp-fill').style.width = `${(p.hp / p.maxHp) * 100}%`;
  document.getElementById('hp-label').textContent = `${Math.ceil(p.hp)} / ${p.maxHp}`;
  document.getElementById('xp-fill').style.width = `${(p.xp / p.xpToNext) * 100}%`;
  document.getElementById('xp-label').textContent = `${p.xp} / ${p.xpToNext}`;
  document.getElementById('level-label').textContent = `Lv ${p.level}`;
  const min = Math.floor(game.time / 60);
  const sec = Math.floor(game.time % 60);
  document.getElementById('timer-stat').textContent = `${min}:${sec.toString().padStart(2, '0')}`;
  document.getElementById('kills-stat').textContent = `${game.kills} kills`;
  document.getElementById('wave-stat').textContent = `Wave ${game.wave}`;
}

// ============ PLANETARY DEFENSES ============
function updatePlanetaryDefenses() {
  const g = game;
  const p = g.player;
  const level = p.level;
  if (level < 3) return; // Defenses unlock at level 3

  // Defense tiers based on player level
  const defenseRange = 150 + level * 12;
  const defenseDmg = 3 + level * 1.5;
  const defenseRate = Math.max(0.6, 2.0 - level * 0.05);

  // Initialize defense timer
  if (g.defenseTimer === undefined) g.defenseTimer = 0;
  g.defenseTimer += dt;

  if (g.defenseTimer < defenseRate) return;
  g.defenseTimer = 0;

  // Find enemies near the colony (0,0)
  const targets = [];
  for (const e of g.enemies) {
    if (e.hp <= 0) continue;
    const d = dist(0, 0, e.x, e.y);
    if (d < defenseRange) targets.push({ enemy: e, dist: d });
  }
  if (targets.length === 0) return;

  // Sort by distance, hit closest
  targets.sort((a, b) => a.dist - b.dist);

  // Number of shots scales with level
  const shotCount = Math.min(targets.length, 1 + Math.floor((level - 3) / 6));

  for (let i = 0; i < shotCount; i++) {
    const target = targets[i].enemy;
    // Visual: laser beam from colony to enemy
    spawnDefenseLaser(0, 0, target.x, target.y);
    damageEnemy(target, Math.floor(defenseDmg * p.damageMult), 'colony');
  }
}

function spawnDefenseLaser(x1, y1, x2, y2) {
  const steps = 8;
  for (let i = 0; i < steps; i++) {
    const t = i / steps;
    game.particles.push({
      x: lerp(x1, x2, t) + rand(-3, 3),
      y: lerp(y1, y2, t) + rand(-3, 3),
      vx: rand(-10, 10), vy: rand(-10, 10),
      color: '#4ecdc4', size: rand(1.5, 3.5),
      lifetime: 0.25, maxLife: 0.25,
    });
  }
}
