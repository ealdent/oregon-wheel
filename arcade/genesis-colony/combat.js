// ============ SCREEN SHAKE ============
function triggerScreenShake(mag, dur) {
  screenShake.mag = mag || CFG.SCREEN_SHAKE_MAG;
  screenShake.timer = dur || CFG.SCREEN_SHAKE_DUR;
}

// ============ DAMAGE ENEMY ============
function damageEnemy(enemy, dmg, source) {
  enemy.hp -= dmg;
  enemy.flashTimer = CFG.DAMAGE_FLASH_DUR;
  spawnDmgText(enemy.x, enemy.y - enemy.radius, dmg.toString());

  if (source) {
    if (!game.weaponStats[source]) game.weaponStats[source] = { damage: 0, kills: 0 };
    game.weaponStats[source].damage += dmg;
  }

  if (enemy.hp <= 0) {
    if (source) game.weaponStats[source].kills++;
    game.kills++;
    spawnParticles(enemy.x, enemy.y, enemy.color, enemy.boss ? 30 : 8, enemy.boss ? 200 : 80);
    // Gloop leaves poison pool on death
    if (enemy.type === 'gloop') {
      // Check for nearby existing pool to compound
      let merged = false;
      for (const pool of game.goopPools) {
        if (dist(enemy.x, enemy.y, pool.x, pool.y) < pool.radius + 30) {
          pool.intensity = Math.min(pool.intensity + 1, 8);
          pool.timer = 10;
          pool.radius = Math.min(pool.radius + 10, 120);
          merged = true;
          break;
        }
      }
      if (!merged) {
        game.goopPools.push({ x: enemy.x, y: enemy.y, radius: 40, intensity: 1, timer: 10 });
      }
    }
    const baseXp = ENEMY_TYPES[enemy.type].xp;
    const waveMultiplier = 1 + (game.wave - 1) * 0.075;
    spawnGem(enemy.x, enemy.y, Math.floor(baseXp * waveMultiplier) || 1);
    if (enemy.boss) {
      triggerScreenShake(12, 0.4);
      playSound('explosion');
      game.gemSlurp = 1.5; // Slurp all gems toward player
      // Drop extra gems (also scaled)
      const bossGemXp = Math.floor(3 * waveMultiplier) || 1;
      for (let i = 0; i < 5; i++) spawnGem(enemy.x + rand(-30, 30), enemy.y + rand(-30, 30), bossGemXp);
    } else {
      playSound('hit');
    }
  }
}

// ============ PLAYER DAMAGE ============
function damagePlayer(dmg) {
  const p = game.player;
  if (p.iframes > 0) return;
  p.hp -= dmg;
  p.iframes = CFG.PLAYER_IFRAMES;
  triggerScreenShake(8, 0.2);
  playSound('hurt');
  spawnParticles(p.x, p.y, '#ff6b6b', 6, 60);

  if (p.hp <= 0) {
    p.hp = 0;
    gameOver();
  }
}
