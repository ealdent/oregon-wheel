"use strict";

function waveForecast(wave, operation = state.operation) {
  const zone = cloneOperation(operation);
  const bossWave = isBossOperation(zone);
  const signature = primaryThreatForWave(wave, bossWave, zone);
  const count = waveEnemyCount(wave, bossWave);
  const seconds = waveDurationSeconds(wave, bossWave);
  const threat = bossWave ? "BOSS" : wave < 4 ? "LOW" : wave < 9 ? "MODERATE" : wave < 15 ? "HIGH" : "SEVERE";
  const bossDef = bossDefs[signature] || null;
  return {
    wave,
    operation: zone,
    operationLabel: operationLabel(zone),
    operationCompactLabel: operationCompactLabel(zone),
    facility: zone.facility,
    sector: zone.sector,
    bossWave,
    count,
    threat,
    kind: bossWave ? "BOSS ZONE" : "SECTOR",
    signature,
    duration: formatDuration(seconds),
    note: bossWave
      ? `${zone.facility} final sector. ${bossDef ? bossDef.name : "Boss signature"} is breaching the route.`
      : `Countdown route falls from Sector ${zone.sector} toward the Sector 1 boss zone.`
  };
}

function waveBalanceConfig(bossWave) {
  return bossWave ? WAVE_BALANCE.boss : WAVE_BALANCE.regular;
}

function facilityBalance() {
  const type = activeFacilityTypeDef();
  return type.balance || { count: 1, health: 1, speed: 1, reward: 1, interval: 1, armor: 1 };
}

function waveEnemyCount(wave, bossWave) {
  const balance = waveBalanceConfig(bossWave);
  return Math.round(Math.min(balance.countCap, balance.baseCount + wave * balance.countPerWave) * facilityBalance().count);
}

function waveSpawnInterval(wave, bossWave) {
  const balance = waveBalanceConfig(bossWave);
  return Math.max(balance.minInterval, balance.intervalStart - wave * balance.intervalDrop) * facilityBalance().interval;
}

function waveDurationSeconds(wave, bossWave) {
  const balance = waveBalanceConfig(bossWave);
  return Math.round(
    waveEnemyCount(wave, bossWave) * waveSpawnInterval(wave, bossWave)
    + balance.durationBase
    + Math.min(balance.durationGrowthCap, wave * balance.durationGrowth)
  );
}

function enemyHealthScale(wave) {
  return Math.pow(WAVE_BALANCE.healthGrowth, wave - 1) * (1 + wave * WAVE_BALANCE.healthLinear);
}

function enemyArmorScale(wave) {
  return 1 + wave * WAVE_BALANCE.armorLinear;
}

function enemyRewardScale(wave) {
  return 1 + wave * WAVE_BALANCE.rewardLinear;
}

function waveClearBonus(wave, bossWave) {
  return WAVE_BALANCE.clearBase + wave * WAVE_BALANCE.clearPerWave + (bossWave ? WAVE_BALANCE.bossClearBonus : 0);
}

function primaryThreatForWave(wave, bossWave = false, operation = state.operation) {
  if (bossWave) return bossTypeForOperation(operation);
  const unlocked = Object.entries(enemyDefs)
    .filter(([, enemy]) => enemy.unlock <= wave)
    .sort((a, b) => a[1].unlock - b[1].unlock);
  return unlocked.length ? unlocked[unlocked.length - 1][0] : "crawler";
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = String(seconds % 60).padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function getTowerStats(tower) {
  const def = towerById[tower.type];
  const level = tower.level;
  const over = Math.max(0, level - 5);
  const tier = Math.min(level, 5);
  const damageMult = 1 + (tier - 1) * 0.28 + over * 0.18;
  const rangeMult = 1 + (tier - 1) * 0.045 + over * 0.018;
  const rateMult = 1 + (tier - 1) * 0.09 + over * 0.04;
  const stats = {
    damage: def.damage * damageMult,
    range: def.range * rangeMult,
    rate: def.rate * rateMult,
    color: def.color,
    chain: 2 + Math.floor(level / 2),
    chainRadius: 92 + level * 7,
    slow: Math.min(0.68, 0.32 + level * 0.035),
    slowTime: 1.5 + level * 0.16,
    vulnTime: 1.1 + level * 0.12,
    jamTime: 1.55 + level * 0.2,
    stunTime: level >= 4 ? 0.16 + level * 0.025 : 0,
    mineRadius: 48 + level * 5,
    mineLimit: 3 + level,
    armorPierce: 0.35 + level * 0.035
  };
  if (tower.type === "mine") {
    stats.rate = def.rate * (1 + (level - 1) * 0.12);
  }
  return stats;
}

function upgradeCost(tower) {
  const def = towerById[tower.type];
  const level = tower.level;
  return Math.round(def.cost * (0.58 + level * 0.58) * Math.pow(1.13, Math.max(0, level - 5)));
}

function sellValue(tower) {
  if (isFullRefundEligible(tower)) {
    return towerById[tower.type].cost;
  }
  return Math.floor(tower.spent * 0.7);
}

function isFullRefundEligible(tower) {
  if (!tower) return false;
  const def = towerById[tower.type];
  return state.phase === "planning"
    && tower.refundable
    && tower.level === 1
    && tower.spent === def.cost;
}

function validatePlacement(x, y) {
  if (state.mode === "campaign") {
    return { valid: false, reason: "Campaign map active" };
  }
  if (x < 34 || x > BOARD.width - 34 || y < 34 || y > BOARD.height - 34) {
    return { valid: false, reason: "Outside grid" };
  }
  if (state.towers.length >= state.towerCapacity) {
    return { valid: false, reason: "Tower capacity full" };
  }
  const pathHit = closestPathPoint(x, y);
  if (pathHit.distance < BOARD.pathWidth / 2 + 26) {
    return { valid: false, reason: "Path conflict" };
  }
  for (const tower of state.towers) {
    if (Math.hypot(tower.x - x, tower.y - y) < 58) {
      return { valid: false, reason: "Too close to tower" };
    }
  }
  return { valid: true, reason: "Build ready" };
}

function placeTower(x, y) {
  if (state.mode === "campaign") return;
  const def = towerById[state.placingType];
  if (!def) {
    showToast("Select a tower design");
    return;
  }
  const placement = validatePlacement(x, y);
  if (!placement.valid) {
    showToast(placement.reason);
    return;
  }
  if (state.credits < def.cost) {
    showToast("Insufficient credits");
    playSound("leak");
    return;
  }
  const tower = {
    id: nextTowerId++,
    type: def.id,
    x,
    y,
    level: 1,
    cooldown: 0,
    targetMode: "FIRST",
    spent: def.cost,
    refundable: state.phase === "planning",
    pulse: Math.random() * Math.PI * 2
  };
  state.towers.push(tower);
  state.credits -= def.cost;
  addRunStat("creditsSpent", def.cost);
  addRunStat("towersBuilt", 1);
  state.selectedTowerId = tower.id;
  state.placingType = null;
  playSound("place");
  log(`${def.name} online.`);
}

function findTowerAt(x, y) {
  let best = null;
  let bestDistance = 28;
  for (const tower of state.towers) {
    const d = Math.hypot(tower.x - x, tower.y - y);
    if (d < bestDistance) {
      best = tower;
      bestDistance = d;
    }
  }
  return best;
}

function selectedTower() {
  return state.towers.find((tower) => tower.id === state.selectedTowerId) || null;
}

function clearSelection(showMessage = false) {
  const hadSelection = !!state.selectedTowerId || !!state.placingType;
  state.selectedTowerId = null;
  state.placingType = null;
  hover.reason = "Select a tower design";
  if (showMessage && hadSelection) {
    showToast("Selection cleared");
  }
}

function queueBuild(type) {
  if (state.mode === "campaign") return;
  const def = towerById[type];
  if (!def) return;
  if (!state.selectedTowerId && state.placingType === type) {
    clearSelection(true);
    return;
  }
  state.placingType = type;
  state.selectedTowerId = null;
  hover.reason = "Build ready";
  showToast(`${def.name} queued`);
}

function selectTower(tower) {
  state.selectedTowerId = tower.id;
  state.placingType = null;
  playSound("click");
}

function upgradeSelected() {
  if (state.mode === "campaign") return;
  const tower = selectedTower();
  if (!tower) {
    showToast("No tower selected");
    return;
  }
  if (tower.level >= MAX_TOWER_LEVEL) {
    showToast("Tower fully synchronized");
    return;
  }
  const cost = upgradeCost(tower);
  if (state.credits < cost) {
    showToast("Insufficient credits");
    playSound("leak");
    return;
  }
  state.credits -= cost;
  addRunStat("creditsSpent", cost);
  addRunStat("upgrades", 1);
  tower.spent += cost;
  tower.level += 1;
  tower.refundable = false;
  tower.cooldown = Math.min(tower.cooldown, 0.12);
  playSound("upgrade");
  log(`${towerById[tower.type].name} upgraded to MK-${tower.level}.`);
}

function sellSelected() {
  if (state.mode === "campaign") return;
  const tower = selectedTower();
  if (!tower) return;
  const fullRefund = isFullRefundEligible(tower);
  const refund = sellValue(tower);
  state.credits += refund;
  addRunStat("creditsRefunded", refund);
  addRunStat("towersSold", 1);
  state.towers = state.towers.filter((item) => item.id !== tower.id);
  state.mines = state.mines.filter((mine) => mine.owner !== tower.id);
  state.selectedTowerId = null;
  playSound("click");
  log(fullRefund ? `Tower refunded for $${refund}.` : `Tower recycled for $${refund}.`);
}

function cycleTargetMode() {
  if (state.mode === "campaign") return;
  const tower = selectedTower();
  if (!tower) return;
  const index = TARGET_MODES.indexOf(tower.targetMode);
  tower.targetMode = TARGET_MODES[(index + 1) % TARGET_MODES.length];
  playSound("click");
}

function resetGame() {
  if (state.campaign && state.facilityNodeId) {
    const campaign = state.campaign;
    const node = campaign.nodes[state.facilityNodeId];
    state = buildFacilityRunState(campaign, node);
    resetRenderKeys();
    log("Facility attempt restored to last secured sector.");
    showToast("Facility retry restored");
    return;
  }
  state = makeInitialState();
  nextTowerId = 1;
  nextEnemyId = 1;
  nextMineId = 1;
  sideTacticsRenderKey = "";
  inspectorRenderKey = "";
  logRenderKey = "";
  enemySpriteCache.clear();
  log("Operation rebooted.");
  showToast("Mother OS reboot complete");
}

function generateWave(wave, operation) {
  const forecast = waveForecast(wave, operation);
  const bossWave = forecast.bossWave;
  const unlocked = Object.entries(enemyDefs)
    .filter(([, enemy]) => enemy.unlock <= wave)
    .map(([id]) => id);
  const count = waveEnemyCount(wave, bossWave);
  const interval = waveSpawnInterval(wave, bossWave);
  const queue = [];
  for (let i = 0; i < count; i += 1) {
    const type = chooseEnemyType(unlocked, wave, i);
    queue.push({ type, delay: interval * (0.78 + Math.random() * 0.44) });
  }
  if (bossWave) {
    const bossType = bossTypeForOperation(operation);
    const bossCount = 1 + Math.floor(wave / 20);
    for (let i = 0; i < bossCount; i += 1) {
      const index = Math.min(queue.length - 1, Math.floor(queue.length * (0.45 + i * 0.1)));
      queue.splice(index, 0, { type: bossType, bossType, delay: interval * 1.6, boss: true });
    }
  }
  return {
    ...forecast,
    bossWave,
    queue,
    total: queue.length,
    threat: forecast.threat
  };
}

function chooseEnemyType(unlocked, wave, index) {
  const weighted = [];
  const bias = activeFacilityTypeDef().enemyBias || {};
  for (const id of unlocked) {
    const enemy = enemyDefs[id];
    let weight = enemy.weight + Math.max(0, wave - enemy.unlock) * 0.22;
    if (bias[id]) weight *= bias[id];
    if (wave % 3 === 0 && (id === "beetle" || id === "worm")) weight += 1.7;
    if (wave % 4 === 0 && (id === "slime" || id === "wisp")) weight += 1.8;
    if (wave % 5 === 0 && (id === "mite" || id === "leech")) weight += 1.4;
    if (wave % 6 === 0 && id === "obelisk") weight += 1.6;
    if (index % 11 === 0 && id === "crawler") weight += 2;
    weighted.push({ id, weight });
  }
  const total = weighted.reduce((sum, item) => sum + item.weight, 0);
  let roll = Math.random() * total;
  for (const item of weighted) {
    roll -= item.weight;
    if (roll <= 0) return item.id;
  }
  return weighted[0].id;
}

function startWave() {
  if (state.mode === "campaign") {
    startSelectedCampaignFacility();
    return;
  }
  if (state.summary) return;
  if (state.gameOver) {
    resetGame();
    return;
  }
  if (state.phase === "combat") return;
  if (state.placingType) state.placingType = null;
  for (const tower of state.towers) {
    tower.refundable = false;
  }
  state.wave += 1;
  addRunStat("wavesStarted", 1);
  state.phase = "combat";
  state.autoStartTimer = 0;
  state.wavePlan = generateWave(state.wave, cloneOperation(state.operation));
  state.spawnQueue = state.wavePlan.queue.slice();
  state.spawnTimer = 0;
  state.waveSpawned = 0;
  state.waveResolved = 0;
  log(`${state.wavePlan.operationLabel} inbound.`);
  showToast(state.wavePlan.bossWave ? "Boss sector detected" : `Sector ${state.wavePlan.sector} launched`);
  playSound("wave");
}

function makeEnemy(type, wave, options = {}) {
  const bossType = options.boss ? (options.bossType || type || "hive") : null;
  const bossDef = bossType ? bossDefs[bossType] || bossDefs.hive : null;
  const def = bossDef || enemyDefs[type] || enemyDefs.crawler;
  const balance = facilityBalance();
  const waveScale = enemyHealthScale(wave);
  const bossScale = options.boss ? WAVE_BALANCE.bossHealthBase + wave * WAVE_BALANCE.bossHealthPerWave : 1;
  const childScale = options.child ? 0.42 : 1;
  const hp = def.hp * waveScale * bossScale * childScale * (options.hpMult || 1) * balance.health;
  const speed = def.speed * (options.speedMult || 1) * (options.boss ? 0.78 : 1) * (options.child ? 1.22 : 1) * balance.speed;
  const radius = def.radius * (bossDef ? 1 : options.boss ? 1.85 : options.child ? 0.72 : 1);
  const rewardMult = options.boss ? WAVE_BALANCE.bossRewardMult : options.child ? 0.25 : 1;
  return {
    id: nextEnemyId++,
    type: bossDef ? "boss" : type,
    bossType: bossDef ? bossType : null,
    name: bossDef ? def.name : options.boss ? `Boss ${def.name}` : def.name,
    hp,
    maxHp: hp,
    speed,
    armor: def.armor * enemyArmorScale(wave) * (options.boss ? 1.4 : options.child ? 0.45 : 1) * balance.armor,
    reward: Math.max(1, Math.round(def.reward * enemyRewardScale(wave) * rewardMult * balance.reward)),
    leak: Math.round((def.leak || 1) * (bossDef ? 1 : options.boss ? 4.5 : 1)),
    radius,
    progress: Math.max(0, options.progress || 0),
    status: { slow: 0, slowAmount: 0, jam: 0, vuln: 0, stun: 0 },
    boss: !!options.boss,
    child: !!options.child,
    dead: false,
    remove: false,
    burrowed: false,
    special: Math.random() * 10,
    spritePhase: Math.floor(Math.random() * 8),
    spawnedMimic: false,
    wobble: Math.random() * Math.PI * 2,
    color: def.color
  };
}

function spawnEnemy(entry) {
  const enemy = makeEnemy(entry.type, state.wave, { boss: entry.boss, bossType: entry.bossType });
  state.enemies.push(enemy);
  state.waveSpawned += 1;
  if (entry.boss) {
    addEffect({ type: "ring", x: pathPoints[0].x, y: pathPoints[0].y, radius: 28, maxRadius: 150, color: "#ffcf5a", life: 0.85, maxLife: 0.85 });
  }
}

function update(dt) {
  if (toastTimer > 0) {
    toastTimer -= dt;
    if (toastTimer <= 0) els.toast.classList.remove("show");
  }
  updateEffects(dt);
  if (state.mode === "campaign" || state.summary) return;
  if (state.paused || state.gameOver) return;
  if (state.phase === "planning") {
    if (state.autoStartTimer > 0) {
      state.autoStartTimer -= dt;
      if (state.autoStartTimer <= 0) startWave();
    }
    return;
  }
  updateSpawning(dt);
  updateEnemies(dt);
  updateTowers(dt);
  updateProjectiles(dt);
  updateMines(dt);
  checkWaveComplete();
}

function updateSpawning(dt) {
  state.spawnTimer -= dt;
  while (state.spawnQueue.length > 0 && state.spawnTimer <= 0) {
    const next = state.spawnQueue.shift();
    spawnEnemy(next);
    state.spawnTimer += next.delay;
  }
}

function updateEnemies(dt) {
  for (const enemy of state.enemies) {
    enemy.status.slow = Math.max(0, enemy.status.slow - dt);
    enemy.status.jam = Math.max(0, enemy.status.jam - dt);
    enemy.status.vuln = Math.max(0, enemy.status.vuln - dt);
    enemy.status.stun = Math.max(0, enemy.status.stun - dt);
    enemy.special += dt;

    let speedMult = 1;
    if (enemy.status.stun > 0) speedMult = 0;
    if (enemy.status.slow > 0) {
      speedMult *= 1 - enemy.status.slowAmount * (enemy.boss ? 0.48 : 1);
    }
    if (enemy.status.jam > 0) speedMult *= enemy.boss ? 0.94 : 0.86;

    if (enemy.type === "worm") {
      enemy.burrowed = Math.sin(enemy.special * 1.55) > 0.25 && enemy.status.jam <= 0;
      if (enemy.burrowed) speedMult *= 1.18;
    } else {
      enemy.burrowed = false;
    }

    if (enemy.type === "phantom" && !enemy.child && !enemy.boss && !enemy.spawnedMimic && enemy.progress > path.total * 0.22) {
      enemy.spawnedMimic = true;
      const mimic = makeEnemy("phantom", state.wave, {
        child: true,
        progress: Math.max(0, enemy.progress - 18),
        hpMult: 0.65,
        speedMult: 1.08
      });
      state.enemies.push(mimic);
      addEffect({ type: "ring", x: pointAtDistance(enemy.progress).x, y: pointAtDistance(enemy.progress).y, radius: 12, maxRadius: 70, color: "#d6c3ff", life: 0.5, maxLife: 0.5 });
    }

    enemy.progress += enemy.speed * speedMult * dt;
    if (enemy.progress >= path.total) {
      enemy.remove = true;
      state.waveResolved += 1;
      state.lives -= enemy.leak;
      addRunStat("enemiesLeaked", 1);
      addRunStat("leakDamage", enemy.leak);
      log(`${enemy.name} breached the core.`);
      showToast("Core integrity compromised");
      playSound("leak");
      const end = pointAtDistance(path.total);
      addEffect({ type: "ring", x: end.x, y: end.y, radius: 18, maxRadius: 130, color: "#ff5f61", life: 0.7, maxLife: 0.7 });
      if (state.lives <= 0) {
        state.lives = 0;
        triggerGameOver();
      }
    }
  }
  state.enemies = state.enemies.filter((enemy) => !enemy.remove);
}

function updateTowers(dt) {
  for (const tower of state.towers) {
    const stats = getTowerStats(tower);
    tower.cooldown -= dt;
    tower.pulse += dt;

    if (tower.type === "mine") {
      if (tower.cooldown <= 0) {
        layMine(tower, stats);
        tower.cooldown += 1 / Math.max(0.05, stats.rate);
      }
      continue;
    }

    if (tower.cooldown > 0) continue;
    const target = selectTarget(tower, stats);
    if (!target) continue;
    fireTower(tower, stats, target);
    tower.cooldown += 1 / Math.max(0.05, stats.rate);
  }
}

function selectTarget(tower, stats) {
  const candidates = state.enemies.filter((enemy) => {
    if (enemy.dead || enemy.remove) return false;
    const pos = pointAtDistance(enemy.progress);
    return Math.hypot(pos.x - tower.x, pos.y - tower.y) <= stats.range + enemy.radius;
  });
  if (!candidates.length) return null;
  if (tower.targetMode === "STRONG") {
    candidates.sort((a, b) => b.hp + b.armor * 12 - (a.hp + a.armor * 12));
  } else if (tower.targetMode === "NEAR") {
    candidates.sort((a, b) => {
      const pa = pointAtDistance(a.progress);
      const pb = pointAtDistance(b.progress);
      return Math.hypot(pa.x - tower.x, pa.y - tower.y) - Math.hypot(pb.x - tower.x, pb.y - tower.y);
    });
  } else {
    candidates.sort((a, b) => b.progress - a.progress);
  }
  return candidates[0];
}

function fireTower(tower, stats, target) {
  const targetPos = pointAtDistance(target.progress);
  if (tower.type === "pulse") {
    state.projectiles.push({
      type: "pulse",
      x: tower.x,
      y: tower.y,
      targetId: target.id,
      speed: 720,
      damage: stats.damage,
      color: stats.color,
      armorPierce: stats.armorPierce,
      life: 1.1
    });
    playShotSound();
  } else if (tower.type === "arc") {
    fireArc(tower, stats, target);
  } else if (tower.type === "cryo") {
    fireCryo(tower, stats, target, targetPos);
  } else if (tower.type === "jammer") {
    fireJammer(tower, stats);
  }
}

function playShotSound() {
  const now = performance.now();
  if (now - lastShotSound > 85) {
    lastShotSound = now;
    playSound("hit");
  }
}

function fireArc(tower, stats, firstTarget) {
  const chain = [firstTarget];
  let current = firstTarget;
  for (let i = 1; i < stats.chain; i += 1) {
    const currentPos = pointAtDistance(current.progress);
    const next = state.enemies
      .filter((enemy) => !enemy.dead && !enemy.remove && !chain.includes(enemy))
      .map((enemy) => ({ enemy, pos: pointAtDistance(enemy.progress) }))
      .filter((item) => Math.hypot(item.pos.x - currentPos.x, item.pos.y - currentPos.y) <= stats.chainRadius)
      .sort((a, b) => a.enemy.progress - b.enemy.progress)
      .pop();
    if (!next) break;
    chain.push(next.enemy);
    current = next.enemy;
  }
  let falloff = 1;
  let previous = { x: tower.x, y: tower.y };
  for (const enemy of chain) {
    const pos = pointAtDistance(enemy.progress);
    dealDamage(enemy, stats.damage * falloff, { energy: true, armorPierce: 0.18 });
    addEffect({ type: "line", x: previous.x, y: previous.y, x2: pos.x, y2: pos.y, color: stats.color, life: 0.18, maxLife: 0.18, width: 4 });
    previous = pos;
    falloff *= 0.74;
  }
  playShotSound();
}

function fireCryo(tower, stats, target, targetPos) {
  const radius = 40 + tower.level * 5;
  for (const enemy of state.enemies) {
    if (enemy.dead || enemy.remove) continue;
    const pos = pointAtDistance(enemy.progress);
    if (Math.hypot(pos.x - targetPos.x, pos.y - targetPos.y) <= radius + enemy.radius) {
      enemy.status.slow = Math.max(enemy.status.slow, stats.slowTime);
      enemy.status.slowAmount = Math.max(enemy.status.slowAmount, stats.slow);
      enemy.status.vuln = Math.max(enemy.status.vuln, stats.vulnTime);
      dealDamage(enemy, stats.damage, { energy: true, armorPierce: 0.1 });
    }
  }
  addEffect({ type: "line", x: tower.x, y: tower.y, x2: targetPos.x, y2: targetPos.y, color: stats.color, life: 0.22, maxLife: 0.22, width: 3 });
  addEffect({ type: "ring", x: targetPos.x, y: targetPos.y, radius: 12, maxRadius: radius, color: stats.color, life: 0.34, maxLife: 0.34 });
  playShotSound();
}

function fireJammer(tower, stats) {
  let hits = 0;
  for (const enemy of state.enemies) {
    if (enemy.dead || enemy.remove) continue;
    const pos = pointAtDistance(enemy.progress);
    const d = Math.hypot(pos.x - tower.x, pos.y - tower.y);
    if (d <= stats.range + enemy.radius) {
      enemy.status.jam = Math.max(enemy.status.jam, stats.jamTime);
      enemy.status.vuln = Math.max(enemy.status.vuln, stats.vulnTime * 0.7);
      if (stats.stunTime > 0) enemy.status.stun = Math.max(enemy.status.stun, stats.stunTime);
      dealDamage(enemy, stats.damage * (1 - d / (stats.range + enemy.radius) * 0.35), { signal: true, armorPierce: 0.55, truesight: true });
      hits += 1;
    }
  }
  addEffect({ type: "ring", x: tower.x, y: tower.y, radius: 18, maxRadius: stats.range, color: stats.color, life: 0.42, maxLife: 0.42 });
  if (hits) playShotSound();
}

function layMine(tower, stats) {
  const owned = state.mines.filter((mine) => mine.owner === tower.id);
  if (owned.length >= stats.mineLimit) return;
  const closest = closestPathPoint(tower.x, tower.y);
  if (closest.distance > stats.range + BOARD.pathWidth / 2) return;
  const progress = clamp(closest.progress + (Math.random() - 0.5) * stats.range * 1.1, 40, path.total - 40);
  const pos = pointAtDistance(progress);
  const tooClose = state.mines.some((mine) => Math.hypot(mine.x - pos.x, mine.y - pos.y) < 34 && mine.owner === tower.id);
  if (tooClose) return;
  state.mines.push({
    id: nextMineId++,
    owner: tower.id,
    x: pos.x,
    y: pos.y,
    progress,
    damage: stats.damage,
    radius: stats.mineRadius,
    arm: 0.38,
    color: stats.color,
    level: tower.level
  });
  addEffect({ type: "ring", x: pos.x, y: pos.y, radius: 8, maxRadius: 26, color: stats.color, life: 0.32, maxLife: 0.32 });
}

function updateProjectiles(dt) {
  for (const projectile of state.projectiles) {
    projectile.life -= dt;
    const target = state.enemies.find((enemy) => enemy.id === projectile.targetId);
    if (!target || target.dead || target.remove || projectile.life <= 0) {
      projectile.remove = true;
      continue;
    }
    const targetPos = pointAtDistance(target.progress);
    const dx = targetPos.x - projectile.x;
    const dy = targetPos.y - projectile.y;
    const d = Math.hypot(dx, dy) || 1;
    const step = projectile.speed * dt;
    if (d <= step + target.radius) {
      dealDamage(target, projectile.damage, { kinetic: true, armorPierce: projectile.armorPierce });
      addEffect({ type: "spark", x: targetPos.x, y: targetPos.y, color: projectile.color, life: 0.25, maxLife: 0.25 });
      projectile.remove = true;
    } else {
      projectile.x += (dx / d) * step;
      projectile.y += (dy / d) * step;
    }
  }
  state.projectiles = state.projectiles.filter((projectile) => !projectile.remove);
}

function updateMines(dt) {
  for (const mine of state.mines) {
    mine.arm -= dt;
    if (mine.arm > 0) continue;
    const victims = state.enemies.filter((enemy) => {
      if (enemy.dead || enemy.remove) return false;
      const pos = pointAtDistance(enemy.progress);
      return Math.hypot(pos.x - mine.x, pos.y - mine.y) <= mine.radius + enemy.radius;
    });
    if (!victims.length) continue;
    for (const enemy of victims) {
      dealDamage(enemy, mine.damage, { ground: true, armorPierce: 0.5 });
      enemy.status.stun = Math.max(enemy.status.stun, enemy.boss ? 0.04 : 0.18);
    }
    mine.remove = true;
    playSound("blast");
    addEffect({ type: "ring", x: mine.x, y: mine.y, radius: 12, maxRadius: mine.radius * 1.45, color: mine.color, life: 0.44, maxLife: 0.44 });
    burstParticles(mine.x, mine.y, mine.color, 12);
  }
  state.mines = state.mines.filter((mine) => !mine.remove);
}

function dealDamage(enemy, amount, flags = {}) {
  if (enemy.dead || enemy.remove) return 0;
  if (enemy.type === "wisp" && !flags.truesight && enemy.status.jam <= 0 && Math.random() < 0.16) {
    const pos = pointAtDistance(enemy.progress);
    addEffect({ type: "text", x: pos.x, y: pos.y - 16, text: "PHASE", color: enemy.color, life: 0.42, maxLife: 0.42 });
    return 0;
  }
  let armor = enemy.armor;
  if (enemy.status.jam > 0) armor *= 0.25;
  armor *= 1 - clamp(flags.armorPierce || 0, 0, 0.9);
  let damage = Math.max(amount * 0.16, amount - armor);
  if (enemy.status.vuln > 0) damage *= enemy.boss ? 1.08 : 1.16;
  if (enemy.burrowed && !flags.ground && enemy.status.jam <= 0) damage *= 0.42;
  if (enemy.boss) damage *= 0.92;
  enemy.hp -= damage;
  if (enemy.hp <= 0) killEnemy(enemy);
  return damage;
}

function killEnemy(enemy) {
  if (enemy.dead || enemy.remove) return;
  enemy.dead = true;
  enemy.remove = true;
  state.credits += enemy.reward;
  addRunStat("creditsEarned", enemy.reward);
  const scoreGain = Math.round(enemy.reward * 11 + state.wave * (enemy.boss ? 90 : 8));
  state.score += scoreGain;
  addRunStat("scoreGained", scoreGain);
  state.kills += 1;
  addRunStat("kills", 1);
  state.waveResolved += 1;
  const pos = pointAtDistance(enemy.progress);
  burstParticles(pos.x, pos.y, enemy.color, enemy.boss ? 24 : 8);
  addEffect({ type: "spark", x: pos.x, y: pos.y, color: enemy.color, life: 0.35, maxLife: 0.35 });
  if (enemy.type === "slime" && !enemy.child) {
    const splits = enemy.boss ? 5 : 2;
    for (let i = 0; i < splits; i += 1) {
      const child = makeEnemy("slime", state.wave, {
        child: true,
        progress: Math.max(0, enemy.progress - i * 9),
        hpMult: enemy.boss ? 1.2 : 1
      });
      state.enemies.push(child);
    }
  }
  if (enemy.boss) {
    const bossBonus = 85 + state.wave * 7;
    state.credits += bossBonus;
    addRunStat("creditsEarned", bossBonus);
    log("Boss chassis dismantled.");
    showToast("Boss dismantled");
  }
}

function burstParticles(x, y, color, count) {
  for (let i = 0; i < count; i += 1) {
    const angle = Math.random() * Math.PI * 2;
    const speed = 28 + Math.random() * 96;
    state.particles.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      color,
      life: 0.35 + Math.random() * 0.35,
      maxLife: 0.7
    });
  }
}

function addEffect(effect) {
  state.effects.push(effect);
}

function updateEffects(dt) {
  for (const effect of state.effects) {
    effect.life -= dt;
  }
  state.effects = state.effects.filter((effect) => effect.life > 0);
  for (const particle of state.particles) {
    particle.life -= dt;
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;
    particle.vx *= 0.94;
    particle.vy *= 0.94;
  }
  state.particles = state.particles.filter((particle) => particle.life > 0);
}

function advanceOperationAfterClear(clearedPlan) {
  if (!clearedPlan) return;
  if (state.campaign && state.facilityNodeId) {
    advanceCampaignFacilityAfterClear(clearedPlan);
    return;
  }
  if (clearedPlan.bossWave) {
    const nextIndex = state.operationIndex + 1;
    state.operationIndex = nextIndex;
    state.operation = createOperation(nextIndex);
    log(`Next operation: ${operationLabel(state.operation)}.`);
    return;
  }
  state.operation = {
    ...state.operation,
    sector: Math.max(1, state.operation.sector - 1)
  };
  log(`Countdown advanced: ${operationLabel(state.operation)}.`);
}

function checkWaveComplete() {
  if (state.phase !== "combat") return;
  if (state.spawnQueue.length === 0 && state.enemies.length === 0) {
    const clearedPlan = state.wavePlan;
    const bonus = waveClearBonus(state.wave, clearedPlan && clearedPlan.bossWave);
    state.credits += bonus;
    addRunStat("creditsEarned", bonus);
    state.phase = "planning";
    log(`${clearedPlan ? clearedPlan.operationLabel : `Wave ${state.wave}`} cleared. Buffer +$${bonus}.`);
    showToast(clearedPlan && clearedPlan.bossWave ? "Boss sector cleared" : `Sector ${clearedPlan ? clearedPlan.sector : state.wave} cleared`);
    if (clearedPlan && clearedPlan.bossWave && state.lives < 25) {
      state.lives += 1;
      log("Core integrity patched.");
    }
    advanceOperationAfterClear(clearedPlan);
    if (state.mode !== "campaign") {
      state.wavePlan = null;
    }
    if (state.autoAdvance) {
      state.autoStartTimer = 2.7;
    }
  }
}

function triggerGameOver() {
  state.gameOver = true;
  state.phase = "gameover";
  state.spawnQueue = [];
  log("Mother OS link severed.");
  showToast("Sector lost. Reboot available.");
}
