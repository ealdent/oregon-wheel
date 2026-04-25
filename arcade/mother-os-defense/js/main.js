"use strict";

function frame(now) {
  const rawDt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const dt = rawDt * SPEEDS[state.speedIndex];
  update(dt);
  render();
  uiTimer -= rawDt;
  if (uiTimer <= 0) {
    syncUI();
    uiTimer = 0.08;
  }
  requestAnimationFrame(frame);
}

buildTowerCards();
bindEvents();
resizeCanvas();
requestAnimationFrame(resizeCanvas);
window.setTimeout(resizeCanvas, 250);
syncUI();
requestAnimationFrame(frame);
