"use strict";

let campaignPointer = { active: false, lastX: 0, lastY: 0, moved: false };

function onPointerMove(event) {
  const point = screenToWorld(event);
  if (state.mode === "campaign") {
    if (campaignPointer.active && event.buttons) {
      const dx = point.x - campaignPointer.lastX;
      const dy = point.y - campaignPointer.lastY;
      state.campaign.pan.x += dx;
      state.campaign.pan.y += dy;
      campaignPointer.lastX = point.x;
      campaignPointer.lastY = point.y;
      campaignPointer.moved = true;
    }
    return;
  }
  hover.x = clamp(point.x, 0, BOARD.width);
  hover.y = clamp(point.y, 0, BOARD.height);
}

function onPointerDown(event) {
  const point = screenToWorld(event);
  if (state.mode === "campaign") {
    campaignPointer = { active: true, lastX: point.x, lastY: point.y, moved: false };
    els.field.setPointerCapture(event.pointerId);
    const node = campaignNodeAt(point.x, point.y);
    if (node) selectCampaignNode(node.id);
    return;
  }
  if (point.x < 0 || point.x > BOARD.width || point.y < 0 || point.y > BOARD.height) {
    clearSelection(true);
    return;
  }
  const tower = findTowerAt(point.x, point.y);
  if (tower) {
    if (state.selectedTowerId === tower.id) {
      clearSelection(true);
      playSound("click");
    } else {
      selectTower(tower);
    }
    return;
  }
  if (state.selectedTowerId) {
    clearSelection(true);
    playSound("click");
    return;
  }
  if (state.placingType) {
    placeTower(point.x, point.y);
  }
}

function onPointerUp(event) {
  if (state.mode === "campaign" && campaignPointer.active) {
    campaignPointer.active = false;
    saveCampaign(state.campaign);
    if (els.field.hasPointerCapture(event.pointerId)) {
      els.field.releasePointerCapture(event.pointerId);
    }
  }
}

function centerCampaignMapOnSelection() {
  if (state.mode !== "campaign" || !state.campaign) return;
  const node = selectedCampaignNode(state.campaign);
  if (!node) {
    state.campaign.pan = { x: 0, y: 0 };
  } else {
    const pos = campaignNodePosition(node, state.campaign);
    state.campaign.pan.x += BOARD.width / 2 - pos.x;
    state.campaign.pan.y += BOARD.height / 2 - pos.y;
  }
  saveCampaign(state.campaign);
  playSound("click");
}

function clearCampaignSelection() {
  if (state.mode !== "campaign" || !state.campaign) return;
  state.campaign.selectedNodeId = null;
  resetRenderKeys();
  saveCampaign(state.campaign);
  playSound("click");
}

function onDocumentPointerDown(event) {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (target.closest("#field, button, a, .tower-card")) return;
  if (state.selectedTowerId || state.placingType) {
    clearSelection(true);
    playSound("click");
  }
}

function bindEvents() {
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("load", resizeCanvas);
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(els.field);
  }
  els.field.addEventListener("pointermove", onPointerMove);
  els.field.addEventListener("pointerdown", onPointerDown);
  els.field.addEventListener("pointerup", onPointerUp);
  els.field.addEventListener("pointercancel", onPointerUp);
  document.addEventListener("pointerdown", onDocumentPointerDown);
  els.startWave.addEventListener("click", () => {
    startWave();
  });
  els.pauseBtn.addEventListener("click", () => {
    if (state.mode === "campaign") {
      state.campaign.pan = { x: 0, y: 0 };
      saveCampaign(state.campaign);
      playSound("click");
      return;
    }
    if (state.gameOver) return;
    state.paused = !state.paused;
    playSound("click");
  });
  els.speedBtn.addEventListener("click", () => {
    if (state.mode === "campaign") return;
    state.speedIndex = (state.speedIndex + 1) % SPEEDS.length;
    playSound("click");
  });
  els.autoBtn.addEventListener("click", () => {
    if (state.mode === "campaign") return;
    state.autoAdvance = !state.autoAdvance;
    if (!state.autoAdvance) state.autoStartTimer = 0;
    if (state.autoAdvance && state.phase === "planning" && !state.gameOver && state.wave > 0) {
      state.autoStartTimer = 2.7;
    }
    playSound("click");
  });
  els.soundBtn.addEventListener("click", () => {
    state.sound = !state.sound;
    if (state.sound) playSound("click");
  });
  els.mapBtn.addEventListener("click", () => {
    if (state.mode === "facility" && state.campaign && state.campaign.mapUnlocked) {
      exitFacilityToCampaign();
      playSound("click");
    }
  });
  els.upgradeBtn.addEventListener("click", () => {
    if (state.mode === "campaign") centerCampaignMapOnSelection();
    else upgradeSelected();
  });
  els.sellBtn.addEventListener("click", sellSelected);
  els.targetBtn.addEventListener("click", () => {
    if (state.mode === "campaign") clearCampaignSelection();
    else cycleTargetMode();
  });
  els.summaryContinue.addEventListener("click", () => {
    dismissFacilitySummary();
    playSound("click");
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      event.preventDefault();
      if (state.mode === "campaign" || state.phase === "planning" || state.gameOver) startWave();
    } else if (event.key === "Escape") {
      event.preventDefault();
      if (state.mode === "campaign") clearCampaignSelection();
      else clearSelection(true);
      playSound("click");
    } else if (event.key.toLowerCase() === "p") {
      state.paused = !state.paused;
    } else if (event.key >= "1" && event.key <= "5") {
      const tower = towerDefs[Number(event.key) - 1];
      if (tower) {
        queueBuild(tower.id);
      }
    }
  });
}
