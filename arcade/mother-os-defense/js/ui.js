"use strict";

function syncUI() {
  const now = new Date();
  const displayWave = state.phase === "combat" ? state.wave : state.wave + 1;
  const forecast = waveForecast(displayWave || 1, activeOperation());
  els.clock.textContent = now.toLocaleTimeString("en-US", { hour12: false });
  document.title = `Mother OS: ${forecast.operationLabel}`;
  els.linkStatus.textContent = state.gameOver ? "SYS LINK: SEVERED" : state.paused ? "SYS LINK: HOLD" : "SYS LINK: STABLE";
  els.creditsReadout.textContent = `$${fmt(state.credits)}`;
  els.livesReadout.textContent = `${state.lives}/25`;
  els.scoreReadout.textContent = fmt(state.score).padStart(6, "0");
  els.phaseReadout.textContent = state.phase.toUpperCase();
  els.activeReadout.textContent = String(state.enemies.length).padStart(2, "0");
  els.threatReadout.textContent = forecast.threat;
  els.waveReadout.textContent = operationCompactLabel(forecast.operation);
  els.capacityReadout.textContent = `${String(state.towers.length).padStart(2, "0")} / ${state.towerCapacity}`;
  els.capacityBar.style.width = `${clamp(state.towers.length / state.towerCapacity, 0, 1) * 100}%`;
  const waveTotal = state.wavePlan ? Math.max(1, state.wavePlan.total) : 1;
  const progress = state.phase === "combat"
    ? clamp(state.waveResolved / waveTotal, 0, 1)
    : state.wave > 0 ? 1 : 0;
  els.waveBar.style.width = `${progress * 100}%`;

  if (state.gameOver) {
    els.startWave.textContent = "Reboot Sector";
    els.startWave.disabled = false;
  } else if (state.phase === "combat") {
    els.startWave.textContent = "Sector Active";
    els.startWave.disabled = true;
  } else if (state.autoStartTimer > 0) {
    els.startWave.textContent = `Auto ${state.autoStartTimer.toFixed(1)}s`;
    els.startWave.disabled = true;
  } else {
    els.startWave.textContent = forecast.bossWave ? "Start Boss S1" : `Start Sector ${forecast.sector}`;
    els.startWave.disabled = false;
  }
  els.pauseBtn.textContent = state.paused ? "Resume" : "Pause";
  els.speedBtn.textContent = `Speed x${SPEEDS[state.speedIndex]}`;
  els.autoBtn.textContent = state.autoAdvance ? "Auto On" : "Auto Off";
  els.autoBtn.classList.toggle("on", state.autoAdvance);
  els.autoBtn.setAttribute("aria-checked", String(state.autoAdvance));
  els.soundBtn.textContent = state.sound ? "Sound On" : "Sound Off";
  els.soundBtn.classList.toggle("on", state.sound);
  els.soundBtn.setAttribute("aria-checked", String(state.sound));

  for (const button of els.towerList.querySelectorAll(".tower-card")) {
    const def = towerById[button.dataset.tower];
    button.classList.toggle("active", state.placingType === def.id && !state.selectedTowerId);
    button.disabled = state.credits < def.cost || state.towers.length >= state.towerCapacity;
  }

  const logKey = state.logs.join("\n");
  if (logKey !== logRenderKey) {
    logRenderKey = logKey;
    els.logList.innerHTML = state.logs.map((item) => `<span>&gt; ${item}</span>`).join("");
  }
  syncSideTactics();
  syncInspector();
}

function syncInspector() {
  const tower = selectedTower();
  const inspectorKey = tower
    ? [
      "tower",
      tower.id,
      tower.type,
      tower.level,
      tower.targetMode,
      tower.spent,
      tower.refundable,
      Math.floor(state.credits),
      state.phase
    ].join("|")
    : ["cursor", state.placingType || "none", Math.floor(state.credits), state.phase].join("|");
  if (inspectorKey === inspectorRenderKey) return;
  inspectorRenderKey = inspectorKey;
  if (!tower) {
    const def = towerById[state.placingType];
    els.inspectorTitle.textContent = def ? "Build Cursor" : "No Node Selected";
    els.inspectorBody.innerHTML = def
      ? `
        <div class="statbox"><span class="stat-label">Design</span><span class="value">${def.name}</span></div>
        <div class="statbox"><span class="stat-label">Cost</span><span class="value">$${def.cost}</span></div>
        <div class="statbox"><span class="stat-label">Range</span><span class="value">${Math.round(def.range)}</span></div>
        <div class="statbox"><span class="stat-label">Role</span><span class="value">${def.role}</span></div>
      `
      : `
        <div class="statbox"><span class="stat-label">Status</span><span class="value">Idle</span></div>
        <div class="statbox"><span class="stat-label">Build</span><span class="value">Select Design</span></div>
      `;
    els.upgradeBtn.disabled = true;
    els.targetBtn.disabled = true;
    els.sellBtn.disabled = true;
    els.upgradeBtn.textContent = "Upgrade";
    els.targetBtn.textContent = "Target";
    els.sellBtn.textContent = "Sell";
    return;
  }

  const def = towerById[tower.type];
  const stats = getTowerStats(tower);
  const cost = upgradeCost(tower);
  els.inspectorTitle.textContent = `${def.name} MK-${tower.level}`;
  els.inspectorBody.innerHTML = `
    <div class="statbox"><span class="stat-label">Damage</span><span class="value">${Math.round(stats.damage)}</span></div>
    <div class="statbox"><span class="stat-label">Range</span><span class="value">${Math.round(stats.range)}</span></div>
    <div class="statbox"><span class="stat-label">Rate</span><span class="value">${stats.rate.toFixed(2)}/s</span></div>
    <div class="statbox"><span class="stat-label">Mode</span><span class="value">${tower.targetMode}</span></div>
  `;
  const maxed = tower.level >= MAX_TOWER_LEVEL;
  els.upgradeBtn.disabled = maxed || state.credits < cost;
  els.targetBtn.disabled = false;
  els.sellBtn.disabled = false;
  els.upgradeBtn.textContent = maxed ? "Max Sync" : `Upgrade $${cost}`;
  els.targetBtn.textContent = tower.targetMode;
  els.sellBtn.textContent = isFullRefundEligible(tower) ? `Refund $${sellValue(tower)}` : `Sell $${sellValue(tower)}`;
}
