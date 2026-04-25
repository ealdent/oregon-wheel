"use strict";

function drawTowerSchematic(g, type, level, pulse, color) {
  const ink = color || "#85ff91";
  g.save();
  g.lineCap = "round";
  g.lineJoin = "round";
  g.lineWidth = 1.55;
  g.strokeStyle = ink;
  g.fillStyle = "rgba(5,30,9,0.92)";
  g.shadowColor = ink;
  g.shadowBlur = 9;
  drawSchematicFrame(g, ink);
  drawTowerFoundation(g, level);
  if (type === "pulse") {
    drawPulseTower(g, level);
  } else if (type === "arc") {
    drawArcTower(g, level, pulse);
  } else if (type === "cryo") {
    drawCryoTower(g, level);
  } else if (type === "mine") {
    drawMineTower(g, level);
  } else {
    drawJammerTower(g, level);
  }
  g.shadowBlur = 0;
  g.strokeStyle = "rgba(185,255,189,0.36)";
  for (let i = -1; i <= 1; i += 1) {
    g.beginPath();
    g.moveTo(i * 13 - 4, 20);
    g.lineTo(i * 13 + 4, 20);
    g.stroke();
  }
  g.restore();
}

function drawSchematicFrame(g, color) {
  g.save();
  g.shadowBlur = 0;
  g.globalAlpha = 0.32;
  g.strokeStyle = color;
  g.lineWidth = 0.75;
  for (const [x, y, sx, sy] of [[-36, -46, 1, 1], [36, -46, -1, 1], [-36, 28, 1, -1], [36, 28, -1, -1]]) {
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x + sx * 12, y);
    g.moveTo(x, y);
    g.lineTo(x, y + sy * 10);
    g.stroke();
  }
  g.beginPath();
  g.moveTo(-31, -4);
  g.lineTo(-38, -4);
  g.moveTo(31, -4);
  g.lineTo(38, -4);
  g.moveTo(0, -50);
  g.lineTo(0, -43);
  g.moveTo(0, 29);
  g.lineTo(0, 22);
  g.stroke();
  g.globalAlpha = 0.18;
  g.translate(2.2, -1.4);
  g.beginPath();
  g.ellipse(0, 16, 32, 10, 0, 0, Math.PI * 2);
  g.moveTo(-18, 3);
  g.lineTo(18, 3);
  g.stroke();
  g.restore();
}

function towerBolt(g, x, y, size = 1.5) {
  g.beginPath();
  g.arc(x, y, size, 0, Math.PI * 2);
  g.stroke();
}

function brokenArc(g, x, y, radius, start, end, segments = 7) {
  const span = end - start;
  for (let i = 0; i < segments; i += 1) {
    const a0 = start + span * (i / segments) + span * 0.018;
    const a1 = start + span * ((i + 0.58) / segments);
    g.beginPath();
    g.arc(x, y, radius, a0, a1);
    g.stroke();
  }
}

function drawTowerFoundation(g, level) {
  g.save();
  g.fillStyle = "rgba(5,30,9,0.96)";
  g.beginPath();
  g.ellipse(0, 18, 34, 11, 0, 0, Math.PI * 2);
  g.fill();
  g.stroke();
  g.beginPath();
  g.ellipse(0, 12, 27, 8, 0, 0, Math.PI * 2);
  g.stroke();
  g.beginPath();
  g.ellipse(0, 5, 20, 5.5, 0, 0, Math.PI * 2);
  g.stroke();
  g.strokeRect(-24, 0, 48, 14);
  g.beginPath();
  g.moveTo(-24, 5);
  g.lineTo(-34, 17);
  g.moveTo(24, 5);
  g.lineTo(34, 17);
  g.moveTo(-12, 14);
  g.lineTo(-19, 23);
  g.moveTo(12, 14);
  g.lineTo(19, 23);
  g.stroke();
  for (let i = 0; i < 8; i += 1) {
    const a = i * Math.PI / 4;
    g.beginPath();
    g.moveTo(Math.cos(a) * 22, 15 + Math.sin(a) * 5);
    g.lineTo(Math.cos(a) * 30, 18 + Math.sin(a) * 8);
    g.stroke();
    towerBolt(g, Math.cos(a) * 24, 16 + Math.sin(a) * 5, 1.35);
  }
  g.strokeStyle = "rgba(185,255,189,0.35)";
  for (let x = -16; x <= 16; x += 8) {
    g.beginPath();
    g.moveTo(x, 1);
    g.lineTo(x - 4, 13);
    g.stroke();
  }
  if (level > 3) {
    g.beginPath();
    g.ellipse(0, 24, 38, 5, 0, 0, Math.PI * 2);
    g.stroke();
  }
  g.restore();
}

function drawPulseTower(g, level) {
  g.strokeRect(-17, -3, 34, 15);
  g.strokeRect(-11, -18, 22, 18);
  g.strokeRect(-5, -28, 10, 10);
  g.beginPath();
  g.moveTo(-7, -18);
  g.lineTo(-3, -42 - level * 0.5);
  g.lineTo(3, -42 - level * 0.5);
  g.lineTo(7, -18);
  g.closePath();
  g.stroke();
  g.beginPath();
  g.moveTo(0, -44 - level * 0.5);
  g.lineTo(0, -53 - level * 0.35);
  g.moveTo(-4, -47);
  g.lineTo(4, -47);
  g.stroke();
  for (let y = -36; y <= -19; y += 4.5) {
    g.beginPath();
    g.moveTo(-12, y);
    g.lineTo(12, y);
    g.stroke();
  }
  g.beginPath();
  g.arc(0, -8, 8, 0, Math.PI * 2);
  g.moveTo(-2.8, -18);
  g.lineTo(-2.8, -41);
  g.moveTo(2.8, -18);
  g.lineTo(2.8, -41);
  g.stroke();
  g.beginPath();
  g.moveTo(-20, 4);
  g.lineTo(-32, 13);
  g.lineTo(-25, 18);
  g.moveTo(20, 4);
  g.lineTo(32, 13);
  g.lineTo(25, 18);
  g.stroke();
  towerBolt(g, -9, -10, 1.2);
  towerBolt(g, 9, -10, 1.2);
}

function drawArcTower(g, level, pulse) {
  g.strokeRect(-17, -4, 34, 11);
  g.strokeRect(-27, 5, 10, 13);
  g.strokeRect(17, 5, 10, 13);
  g.beginPath();
  g.moveTo(-9, 7);
  g.lineTo(-9, -32);
  g.moveTo(9, 7);
  g.lineTo(9, -32);
  g.stroke();
  for (let i = 0; i < 6; i += 1) {
    g.beginPath();
    g.ellipse(0, 1 - i * 6.8, 23 - i * 1.6, 4.6, 0, 0, Math.PI * 2);
    g.stroke();
    g.beginPath();
    g.moveTo(-12, 1 - i * 6.8);
    g.lineTo(-18, 1 - i * 6.8 - 2);
    g.moveTo(12, 1 - i * 6.8);
    g.lineTo(18, 1 - i * 6.8 - 2);
    g.stroke();
  }
  g.beginPath();
  g.moveTo(0, -37);
  g.lineTo(Math.sin(pulse * 5) * 11, -27);
  g.lineTo(Math.cos(pulse * 4) * -9, -17);
  g.lineTo(Math.sin(pulse * 3) * 7, -5);
  g.stroke();
  g.beginPath();
  g.arc(0, -36, 5, 0, Math.PI * 2);
  g.moveTo(-26, 11);
  g.lineTo(-18, 11);
  g.moveTo(18, 11);
  g.lineTo(26, 11);
  g.stroke();
  g.save();
  g.strokeStyle = "rgba(185,255,189,0.45)";
  brokenArc(g, 0, -19, 31, Math.PI * 1.12, Math.PI * 1.88, 8);
  g.restore();
}

function drawCryoTower(g, level) {
  g.strokeRect(-20, 3, 40, 13);
  g.strokeRect(-12, 12, 24, 7);
  g.save();
  g.globalAlpha = 0.18;
  g.beginPath();
  g.moveTo(0, -45);
  g.lineTo(18, -11);
  g.lineTo(9, 14);
  g.lineTo(-9, 14);
  g.lineTo(-18, -11);
  g.closePath();
  g.fill();
  g.restore();
  g.beginPath();
  g.moveTo(0, -45);
  g.lineTo(18, -11);
  g.lineTo(9, 14);
  g.lineTo(-9, 14);
  g.lineTo(-18, -11);
  g.closePath();
  g.stroke();
  g.beginPath();
  g.moveTo(0, -45);
  g.lineTo(0, 14);
  g.moveTo(-18, -11);
  g.lineTo(18, -11);
  g.moveTo(-9, 14);
  g.lineTo(0, -45);
  g.moveTo(9, 14);
  g.lineTo(0, -45);
  g.moveTo(-14, -23);
  g.lineTo(0, -32);
  g.lineTo(14, -23);
  g.moveTo(-12, 0);
  g.lineTo(0, -10);
  g.lineTo(12, 0);
  g.stroke();
  g.save();
  g.globalAlpha = 0.58;
  g.beginPath();
  g.moveTo(-23, -18);
  g.lineTo(-29, -24);
  g.moveTo(23, -18);
  g.lineTo(29, -24);
  g.moveTo(-21, -2);
  g.lineTo(-29, 3);
  g.moveTo(21, -2);
  g.lineTo(29, 3);
  g.stroke();
  g.restore();
}

function drawMineTower(g, level) {
  g.strokeRect(-26, 5, 52, 11);
  g.beginPath();
  g.ellipse(0, -3, 25, 21, 0, 0, Math.PI * 2);
  g.fill();
  g.stroke();
  g.beginPath();
  g.ellipse(0, -3, 15, 11, 0, 0, Math.PI * 2);
  g.stroke();
  g.strokeRect(-10, -13, 20, 18);
  g.beginPath();
  g.moveTo(-6, -3);
  g.lineTo(6, -3);
  g.moveTo(0, -9);
  g.lineTo(0, 3);
  g.stroke();
  for (let i = 0; i < 12; i += 1) {
    const a = i * Math.PI / 6;
    g.beginPath();
    g.moveTo(Math.cos(a) * 16, -3 + Math.sin(a) * 13);
    g.lineTo(Math.cos(a) * 28, -3 + Math.sin(a) * 22);
    g.stroke();
  }
  g.strokeStyle = "rgba(255,207,90,0.7)";
  for (let x = -13; x <= 13; x += 13) {
    g.beginPath();
    g.moveTo(x - 3, 12);
    g.lineTo(x, 7);
    g.lineTo(x + 3, 12);
    g.stroke();
  }
  for (let i = -1; i <= 1; i += 1) {
    g.beginPath();
    g.arc(i * 15, 25, 4.2, 0, Math.PI * 2);
    g.stroke();
    g.beginPath();
    g.moveTo(i * 15, 18);
    g.lineTo(i * 15, 31);
    g.moveTo(i * 15 - 6, 25);
    g.lineTo(i * 15 + 6, 25);
    g.stroke();
  }
}

function drawJammerTower(g, level) {
  g.strokeRect(-17, -1, 34, 13);
  g.beginPath();
  g.moveTo(-22, 15);
  g.lineTo(0, -13);
  g.lineTo(22, 15);
  g.stroke();
  for (let y = 8; y >= -31; y -= 8) {
    g.strokeRect(-4.5, y - 3, 9, 5);
  }
  g.beginPath();
  g.moveTo(0, 12);
  g.lineTo(0, -43);
  g.moveTo(-9, -35);
  g.lineTo(0, -43);
  g.lineTo(9, -35);
  g.stroke();
  brokenArc(g, 0, -8, 16, Math.PI * 1.08, Math.PI * 1.92, 5);
  brokenArc(g, 0, -8, 26, Math.PI * 1.12, Math.PI * 1.88, 6);
  brokenArc(g, 0, -8, 36, Math.PI * 1.18, Math.PI * 1.82, 7);
  g.beginPath();
  g.moveTo(-14, -4);
  g.lineTo(-7, -8);
  g.lineTo(0, -4);
  g.lineTo(7, -8);
  g.lineTo(14, -4);
  g.moveTo(-11, -17);
  g.lineTo(11, -17);
  g.stroke();
}
