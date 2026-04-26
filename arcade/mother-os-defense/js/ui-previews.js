"use strict";

function buildTowerCards() {
  els.towerList.innerHTML = "";
  for (const tower of towerDefs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tower-card";
    button.dataset.tower = tower.id;
    button.setAttribute("aria-label", `${tower.name}, ${tower.desc}, cost ${tower.cost}`);
    button.innerHTML = `
      <canvas class="tower-preview" width="188" height="168" aria-hidden="true"></canvas>
      <span>
        <span class="tower-name">${tower.name}</span>
        <span class="tower-desc">${tower.desc}</span>
      </span>
      <span class="tower-cost">$${tower.cost}</span>
    `;
    button.addEventListener("click", () => {
      queueBuild(tower.id);
      playSound("click");
    });
    els.towerList.appendChild(button);
    renderTowerPreview(button.querySelector(".tower-preview"), tower);
  }
}

function renderTowerPreview(canvas, tower) {
  const p = canvas.getContext("2d");
  p.clearRect(0, 0, canvas.width, canvas.height);
  p.save();
  p.fillStyle = "rgba(0,10,4,0.9)";
  p.fillRect(0, 0, canvas.width, canvas.height);
  p.strokeStyle = "rgba(97,255,126,0.12)";
  p.lineWidth = 1;
  for (let y = 10; y < canvas.height; y += 12) {
    p.beginPath();
    p.moveTo(6, y);
    p.lineTo(canvas.width - 6, y);
    p.stroke();
  }
  for (let x = 12; x < canvas.width; x += 18) {
    p.beginPath();
    p.moveTo(x, 8);
    p.lineTo(x, canvas.height - 8);
    p.stroke();
  }
  p.restore();
  p.save();
  p.translate(canvas.width / 2, canvas.height / 2 + 24);
  p.scale(1.72, 1.72);
  drawTowerSchematic(p, tower.id, 2, 0.7, tower.color);
  p.restore();
}

function renderThreatPreview(canvas, type) {
  if (!canvas) return;
  const p = canvas.getContext("2d");
  const def = threatDefinition(type);
  const bossPreview = isBossSignature(type);
  p.clearRect(0, 0, canvas.width, canvas.height);
  p.save();
  p.fillStyle = "rgba(0,10,4,0.92)";
  p.fillRect(0, 0, canvas.width, canvas.height);
  p.strokeStyle = "rgba(97,255,126,0.14)";
  p.lineWidth = 1;
  for (let y = 12; y < canvas.height; y += 14) {
    p.beginPath();
    p.moveTo(8, y);
    p.lineTo(canvas.width - 8, y);
    p.stroke();
  }
  for (let x = 16; x < canvas.width; x += 24) {
    p.beginPath();
    p.moveTo(x, 10);
    p.lineTo(x, canvas.height - 10);
    p.stroke();
  }
  p.strokeStyle = "rgba(124,232,255,0.18)";
  for (const [x, y, sx, sy] of [[16, 16, 1, 1], [canvas.width - 16, 16, -1, 1], [16, canvas.height - 16, 1, -1], [canvas.width - 16, canvas.height - 16, -1, -1]]) {
    p.beginPath();
    p.moveTo(x, y);
    p.lineTo(x + sx * 20, y);
    p.moveTo(x, y);
    p.lineTo(x, y + sy * 16);
    p.stroke();
  }
  p.restore();
  p.save();
  p.translate(canvas.width * (bossPreview ? 0.42 : 0.43), canvas.height * (bossPreview ? 0.58 : 0.56));
  const previewScale = {
    crawler: 1.68,
    beetle: 1.55,
    slime: 1.58,
    worm: 1.22,
    wisp: 1.32,
    juggernaut: 1.46,
    phantom: 1.45,
    mite: 1.62,
    leech: 1.2,
    obelisk: 1.42,
    hive: 1.12,
    conduit: 1.0,
    colossus: 1.02,
    harvester: 0.95
  };
  const scale = previewScale[type] || 1.5;
  p.scale(scale, scale);
  if (bossPreview) {
    drawBossSchematic(p, type, def.radius, { preview: true });
  } else {
    drawThreatSchematic(p, type, def);
  }
  p.restore();
  p.save();
  p.font = "700 13px Courier New, monospace";
  p.fillStyle = def.color;
  p.textAlign = "right";
  p.fillText(def.name.toUpperCase(), canvas.width - 12, 25);
  p.font = "10px Courier New, monospace";
  p.fillStyle = "rgba(185,255,189,0.6)";
  p.fillText(bossPreview ? `${def.code} / SECTOR 01` : `SIG ${def.code} / ${def.unlock.toString().padStart(2, "0")}`, canvas.width - 12, 43);
  p.restore();
}
