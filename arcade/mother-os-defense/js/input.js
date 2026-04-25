"use strict";

function onPointerMove(event) {
  const point = screenToWorld(event);
  hover.x = clamp(point.x, 0, BOARD.width);
  hover.y = clamp(point.y, 0, BOARD.height);
}

function onPointerDown(event) {
  const point = screenToWorld(event);
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
  document.addEventListener("pointerdown", onDocumentPointerDown);
  els.startWave.addEventListener("click", () => {
    startWave();
  });
  els.pauseBtn.addEventListener("click", () => {
    if (state.gameOver) return;
    state.paused = !state.paused;
    playSound("click");
  });
  els.speedBtn.addEventListener("click", () => {
    state.speedIndex = (state.speedIndex + 1) % SPEEDS.length;
    playSound("click");
  });
  els.autoBtn.addEventListener("click", () => {
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
  els.upgradeBtn.addEventListener("click", upgradeSelected);
  els.sellBtn.addEventListener("click", sellSelected);
  els.targetBtn.addEventListener("click", cycleTargetMode);
  window.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      event.preventDefault();
      if (state.phase === "planning" || state.gameOver) startWave();
    } else if (event.key === "Escape") {
      event.preventDefault();
      clearSelection(true);
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
