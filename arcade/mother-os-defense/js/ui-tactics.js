"use strict";

function syncSideTactics() {
  const nextWave = state.gameOver ? state.wave : state.phase === "combat" ? state.wave : state.wave + 1;
  const forecast = waveForecast(nextWave || 1, activeOperation());
  const tower = selectedTower();
  const queued = towerById[state.placingType] || null;
  const selected = tower ? towerById[tower.type] : queued;
  const buildStatus = state.selectedTowerId
    ? `${selected.name} MK-${tower.level} selected`
    : selected ? `${selected.name} queued` : "No tower design queued";
  const signature = threatDefinition(forecast.signature);
  const laneStatus = state.phase === "combat"
    ? `${state.enemies.length} hostiles active`
    : hover.reason || "Build ready";
  const renderKey = [
    nextWave,
    forecast.operationLabel,
    forecast.signature,
    forecast.kind,
    forecast.count,
    forecast.threat,
    forecast.duration,
    state.phase,
    state.enemies.length,
    state.selectedTowerId || "none",
    state.placingType || "none",
    tower ? `${tower.id}:${tower.level}:${tower.targetMode}` : "none",
    state.towers.length,
    Math.floor(state.credits),
    state.autoAdvance,
    state.speedIndex,
    state.sound,
    state.lives,
    laneStatus
  ].join("|");
  if (renderKey === sideTacticsRenderKey) return;
  sideTacticsRenderKey = renderKey;
  if (tower) {
    renderSelectedTowerTactics(tower);
    return;
  }
  els.sideTactics.innerHTML = `
    <div class="tactic-card">
      <div class="tactic-heading"><span>${state.phase === "combat" ? "Active Zone" : "Next Zone"}</span><span>${forecast.kind}</span></div>
      <div class="threat-band">
        <canvas class="threat-preview" width="320" height="168" aria-label="${signature.name} schematic"></canvas>
        <div class="mini-grid">
          <span class="mini-chip">Signature<strong>${signature.name}</strong></span>
          <span class="mini-chip">Armor<strong>${signature.armor}</strong></span>
          <span class="mini-chip">Speed<strong>${signature.speed}</strong></span>
          <span class="mini-chip">Leak<strong>${signature.leak}</strong></span>
        </div>
      </div>
      <div class="mini-grid">
        <span class="mini-chip wide">Facility<strong>${forecast.facility}</strong></span>
        <span class="mini-chip">Sector<strong>${String(forecast.sector).padStart(2, "0")}${forecast.bossWave ? " BOSS" : ""}</strong></span>
        <span class="mini-chip">Wave<strong>${String(nextWave).padStart(2, "0")}</strong></span>
        <span class="mini-chip">Length<strong>${forecast.duration}</strong></span>
        <span class="mini-chip">Threat<strong>${forecast.threat}</strong></span>
        <span class="mini-chip">Spawn Pack<strong>${forecast.count}</strong></span>
      </div>
      <p class="tactic-copy">${forecast.note}</p>
    </div>
    <div class="tactic-card">
      <div class="tactic-heading"><span>Build Doctrine</span><span>${selected ? selected.role : "IDLE"}</span></div>
      <p class="tactic-copy">${buildStatus}. ${laneStatus}. Upgrades remain available during combat and planning windows.</p>
      <div class="mini-grid">
        <span class="mini-chip">Capacity<strong>${state.towers.length}/${state.towerCapacity}</strong></span>
        <span class="mini-chip">Credits<strong>$${fmt(state.credits)}</strong></span>
      </div>
    </div>
    <div class="tactic-card">
      <div class="tactic-heading"><span>Field Systems</span><span>${state.phase.toUpperCase()}</span></div>
      <div class="mini-grid">
        <span class="mini-chip">Auto Flow<strong>${state.autoAdvance ? "ON" : "OFF"}</strong></span>
        <span class="mini-chip">Sim Rate<strong>x${SPEEDS[state.speedIndex]}</strong></span>
        <span class="mini-chip">Sound<strong>${state.sound ? "ON" : "OFF"}</strong></span>
        <span class="mini-chip">Core<strong>${state.lives}/25</strong></span>
      </div>
    </div>
  `;
  renderThreatPreview(els.sideTactics.querySelector(".threat-preview"), forecast.signature);
}

function renderSelectedTowerTactics(tower) {
  const def = towerById[tower.type];
  const stats = getTowerStats(tower);
  const maxed = tower.level >= MAX_TOWER_LEVEL;
  const cost = maxed ? 0 : upgradeCost(tower);
  const canUpgrade = !maxed && state.credits >= cost;
  const refund = sellValue(tower);
  const nextStats = maxed ? null : getTowerStats({ ...tower, level: tower.level + 1 });
  const upgradeState = maxed ? "MAXED" : canUpgrade ? "READY" : "NEED CREDITS";
  els.sideTactics.innerHTML = `
    <div class="tactic-card tower-command-card">
      <div class="tactic-heading"><span>Selected Tower</span><span>${def.role} MK-${tower.level}</span></div>
      <div class="tower-detail-band">
        <canvas class="tower-detail-preview" width="320" height="188" aria-label="${def.name} schematic"></canvas>
        <div class="mini-grid">
          <span class="mini-chip">Design<strong>${def.name}</strong></span>
          <span class="mini-chip">Target<strong>${tower.targetMode}</strong></span>
          <span class="mini-chip">Spent<strong>$${fmt(tower.spent)}</strong></span>
          <span class="mini-chip">Sell<strong>$${fmt(refund)}</strong></span>
        </div>
      </div>
    </div>
    <div class="tactic-card">
      <div class="tactic-heading"><span>Upgrade Path</span><span>${upgradeState}</span></div>
      <div class="upgrade-panel">
        ${towerUpgradePreviewHtml(tower, stats, nextStats)}
      </div>
      <div class="mini-grid">
        <span class="mini-chip ${canUpgrade ? "ready" : maxed ? "maxed" : "blocked"}">Upgrade<strong>${maxed ? "SYNCHED" : `$${fmt(cost)}`}</strong></span>
        <span class="mini-chip">Credits<strong>$${fmt(state.credits)}</strong></span>
        <span class="mini-chip">Command<strong>${maxed ? "MAX SYNC" : canUpgrade ? "UPGRADE" : "HOLD"}</strong></span>
        <span class="mini-chip">Recycle<strong>${isFullRefundEligible(tower) ? "FULL REFUND" : "70% RETURN"}</strong></span>
      </div>
    </div>
    <div class="tactic-card">
      <div class="tactic-heading"><span>Combat Stats</span><span>${towerSpecialLabel(tower.type)}</span></div>
      <div class="mini-grid">
        <span class="mini-chip">Damage<strong>${Math.round(stats.damage)}</strong></span>
        <span class="mini-chip">Range<strong>${Math.round(stats.range)}</strong></span>
        <span class="mini-chip">Rate<strong>${stats.rate.toFixed(2)}/s</strong></span>
        <span class="mini-chip">Level<strong>${tower.level}/${MAX_TOWER_LEVEL}</strong></span>
        ${towerSpecialStatChips(tower, stats)}
      </div>
      <p class="tactic-copy">${towerSpecialCopy(tower, stats)}</p>
    </div>
  `;
  renderTowerDetailPreview(els.sideTactics.querySelector(".tower-detail-preview"), tower);
}

function towerUpgradePreviewHtml(tower, stats, nextStats) {
  if (!nextStats) {
    return `
      <div class="upgrade-row"><span>Damage</span><strong>${Math.round(stats.damage)}</strong><em>MAX</em></div>
      <div class="upgrade-row"><span>Range</span><strong>${Math.round(stats.range)}</strong><em>MAX</em></div>
      <div class="upgrade-row"><span>Rate</span><strong>${stats.rate.toFixed(2)}/s</strong><em>MAX</em></div>
      <div class="upgrade-row"><span>${towerSpecialLabel(tower.type)}</span><strong>${towerSpecialValue(tower, stats)}</strong><em>MAX</em></div>
    `;
  }
  return `
    <div class="upgrade-row"><span>Damage</span><strong>${Math.round(stats.damage)}</strong><em>${Math.round(nextStats.damage)}</em></div>
    <div class="upgrade-row"><span>Range</span><strong>${Math.round(stats.range)}</strong><em>${Math.round(nextStats.range)}</em></div>
    <div class="upgrade-row"><span>Rate</span><strong>${stats.rate.toFixed(2)}/s</strong><em>${nextStats.rate.toFixed(2)}/s</em></div>
    <div class="upgrade-row"><span>${towerSpecialLabel(tower.type)}</span><strong>${towerSpecialValue(tower, stats)}</strong><em>${towerSpecialValue(tower, nextStats)}</em></div>
  `;
}

function towerSpecialLabel(type) {
  if (type === "pulse") return "Pierce";
  if (type === "arc") return "Chain";
  if (type === "cryo") return "Slow";
  if (type === "mine") return "Mines";
  if (type === "jammer") return "Jam";
  return "Special";
}

function towerSpecialValue(tower, stats) {
  if (tower.type === "pulse") return `${Math.round(stats.armorPierce * 100)}%`;
  if (tower.type === "arc") return `${stats.chain}x`;
  if (tower.type === "cryo") return `${Math.round(stats.slow * 100)}%`;
  if (tower.type === "mine") return `${stats.mineLimit}`;
  if (tower.type === "jammer") return `${stats.jamTime.toFixed(1)}s`;
  return "-";
}

function towerSpecialStatChips(tower, stats) {
  if (tower.type === "pulse") {
    return `<span class="mini-chip">Armor Pierce<strong>${Math.round(stats.armorPierce * 100)}%</strong></span><span class="mini-chip">Projectile<strong>KINETIC</strong></span>`;
  }
  if (tower.type === "arc") {
    return `<span class="mini-chip">Chain Count<strong>${stats.chain}</strong></span><span class="mini-chip">Chain Radius<strong>${Math.round(stats.chainRadius)}</strong></span>`;
  }
  if (tower.type === "cryo") {
    return `<span class="mini-chip">Slow<strong>${Math.round(stats.slow * 100)}%</strong></span><span class="mini-chip">Vuln Time<strong>${stats.vulnTime.toFixed(1)}s</strong></span>`;
  }
  if (tower.type === "mine") {
    return `<span class="mini-chip">Mine Limit<strong>${stats.mineLimit}</strong></span><span class="mini-chip">Blast Radius<strong>${Math.round(stats.mineRadius)}</strong></span>`;
  }
  if (tower.type === "jammer") {
    return `<span class="mini-chip">Jam Time<strong>${stats.jamTime.toFixed(1)}s</strong></span><span class="mini-chip">Stun<strong>${stats.stunTime ? `${stats.stunTime.toFixed(1)}s` : "MK-4"}</strong></span>`;
  }
  return "";
}

function towerSpecialCopy(tower, stats) {
  if (tower.type === "pulse") return `Needle fire pierces ${Math.round(stats.armorPierce * 100)}% of armor and scales best into light, fast lanes.`;
  if (tower.type === "arc") return `Relay arcs through ${stats.chain} targets inside ${Math.round(stats.chainRadius)}m chain radius.`;
  if (tower.type === "cryo") return `Prism shots slow enemies by ${Math.round(stats.slow * 100)}% and apply vulnerability for ${stats.vulnTime.toFixed(1)}s.`;
  if (tower.type === "mine") return `Mine layer maintains up to ${stats.mineLimit} armed charges with ${Math.round(stats.mineRadius)}m blast radius.`;
  if (tower.type === "jammer") return `Jammer suppresses armor, burrow, and phasing for ${stats.jamTime.toFixed(1)}s inside its field.`;
  return "Tower systems nominal.";
}
