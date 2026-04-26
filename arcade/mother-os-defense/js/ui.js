"use strict";

function syncUI() {
  const now = new Date();
  els.clock.textContent = now.toLocaleTimeString("en-US", { hour12: false });
  document.body.classList.toggle("campaign-mode", state.mode === "campaign");
  if (state.mode === "campaign") {
    syncCampaignUI(now);
    syncSummaryModal();
    return;
  }
  const displayWave = state.phase === "combat" ? state.wave : state.wave + 1;
  const forecast = waveForecast(displayWave || 1, activeOperation());
  document.title = `Mother OS: ${forecast.operationLabel}`;
  els.resourceTitle.textContent = "Resource Feed";
  els.creditsLabel.textContent = "Credits";
  els.livesLabel.textContent = "Health";
  els.scoreLabel.textContent = "Points";
  els.pauseBtn.disabled = false;
  els.speedBtn.disabled = false;
  els.autoBtn.disabled = false;
  els.soundBtn.disabled = false;
  els.linkStatus.textContent = state.gameOver ? "SYS LINK: SEVERED" : state.paused ? "SYS LINK: HOLD" : "SYS LINK: STABLE";
  els.creditsReadout.textContent = `$${fmt(state.credits)}`;
  els.livesReadout.textContent = `${state.lives}/${STARTING_LIVES}`;
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
  els.mapBtn.textContent = state.campaign && state.campaign.mapUnlocked ? "Exit Facility" : "Map Locked";
  els.mapBtn.disabled = !(state.campaign && state.campaign.mapUnlocked);

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
  syncSummaryModal();
}

function syncCampaignUI() {
  const campaign = state.campaign;
  const node = selectedCampaignNode(campaign);
  const visible = visibleCampaignNodes(campaign);
  const available = visible.filter((item) => canEnterCampaignNode(item, campaign));
  const secured = visible.filter((item) => item.secured);
  const progress = visible.length ? Math.round(secured.length / visible.length * 100) : 0;
  document.title = node ? `Mother OS: Campaign / ${node.facility}` : "Mother OS: Campaign Map";
  els.resourceTitle.textContent = "Campaign Feed";
  els.creditsLabel.textContent = "Secured";
  els.livesLabel.textContent = "Routes";
  els.scoreLabel.textContent = "Score";
  els.creditsReadout.textContent = `${secured.length}/${visible.length}`;
  els.livesReadout.textContent = `${available.length} ready`;
  els.scoreReadout.textContent = fmt(campaign.stats.totalScore || 0).padStart(6, "0");
  els.linkStatus.textContent = "SURVEY MAP: STABLE";
  els.phaseReadout.textContent = "CAMPAIGN";
  els.activeReadout.textContent = String(available.length).padStart(2, "0");
  els.threatReadout.textContent = node && node.secured ? "SECURED" : node && canEnterCampaignNode(node, campaign) ? "AVAILABLE" : "LOCKED";
  els.waveReadout.textContent = node ? `${node.facility} / S${String(node.currentSector).padStart(2, "0")}` : "Select Facility";
  els.capacityReadout.textContent = `${secured.length} / ${visible.length}`;
  els.capacityBar.style.width = `${progress}%`;
  els.waveBar.style.width = `${progress}%`;

  if (node && canEnterCampaignNode(node, campaign)) {
    const resumed = node.checkpoint && node.checkpoint.wave > 0;
    els.startWave.textContent = resumed ? "Resume Facility" : "Start Facility";
    els.startWave.disabled = false;
  } else if (node && node.secured) {
    els.startWave.textContent = "Facility Secured";
    els.startWave.disabled = true;
  } else {
    els.startWave.textContent = "Select Route";
    els.startWave.disabled = true;
  }
  els.upgradeBtn.textContent = "Center Map";
  els.upgradeBtn.disabled = false;
  els.targetBtn.textContent = "Deselect";
  els.targetBtn.disabled = !node;
  els.sellBtn.textContent = node && node.secured ? "Secured" : node && canEnterCampaignNode(node, campaign) ? "Available" : "Route Locked";
  els.sellBtn.disabled = true;
  els.mapBtn.textContent = "Campaign Map";
  els.mapBtn.disabled = true;
  els.pauseBtn.textContent = "Reset View";
  els.pauseBtn.disabled = false;
  els.speedBtn.textContent = `Visible ${visible.length}`;
  els.speedBtn.disabled = true;
  els.autoBtn.textContent = `Survey ${progress}%`;
  els.autoBtn.disabled = true;
  els.autoBtn.classList.remove("on");
  els.autoBtn.setAttribute("aria-checked", "false");
  els.soundBtn.textContent = state.sound ? "Sound On" : "Sound Off";
  els.soundBtn.classList.toggle("on", state.sound);
  els.soundBtn.setAttribute("aria-checked", String(state.sound));

  for (const button of els.towerList.querySelectorAll(".tower-card")) {
    button.classList.remove("active");
    button.disabled = true;
  }
  const logKey = `campaign:${campaign.selectedNodeId}:${visible.length}:${secured.length}:${available.length}`;
  if (logKey !== logRenderKey) {
    logRenderKey = logKey;
    els.logList.innerHTML = [
      `&gt; ${secured.length} facilities secured.`,
      `&gt; ${available.length} routes available.`,
      node ? `&gt; Selected ${node.facility}.` : "&gt; Select a visible facility."
    ].join("");
  }
}

function syncSummaryModal() {
  if (!els.summaryModal) return;
  const summary = state.summary;
  els.summaryModal.hidden = !summary;
  if (!summary) return;
  els.summaryTitle.textContent = summary.title;
  els.summarySubtitle.textContent = summary.subtitle;
  els.summaryBody.innerHTML = summary.stats
    .map(([label, value]) => `<div class="statbox"><span class="stat-label">${label}</span><span class="value">${value}</span></div>`)
    .join("");
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
