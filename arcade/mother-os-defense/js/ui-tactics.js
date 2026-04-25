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
