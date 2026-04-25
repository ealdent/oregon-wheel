"use strict";

function drawThreatSchematic(g, type, def) {
  const r = 22;
  const ink = def.color || "#9fffab";
  const bolt = (x, y, size = 1.35) => {
    g.beginPath();
    g.arc(x, y, size, 0, Math.PI * 2);
    g.stroke();
  };
  const leg = (side, x, y, lift = 1) => {
    g.beginPath();
    g.moveTo(side * x, y);
    g.lineTo(side * (x + 12), y + 7 * lift);
    g.lineTo(side * (x + 20), y + 19 * lift);
    g.stroke();
    bolt(side * (x + 12), y + 7 * lift, 1);
  };
  const segmented = (radius, start = 0, end = Math.PI * 2, pieces = 8) => {
    const span = end - start;
    for (let i = 0; i < pieces; i += 1) {
      g.beginPath();
      g.arc(0, 0, radius, start + span * (i / pieces), start + span * ((i + 0.58) / pieces));
      g.stroke();
    }
  };
  g.save();
  g.lineCap = "round";
  g.lineJoin = "round";
  g.lineWidth = 1.45;
  g.strokeStyle = ink;
  g.fillStyle = "rgba(6,28,10,0.92)";
  g.shadowColor = ink;
  g.shadowBlur = 8;
  g.save();
  g.globalAlpha = 0.18;
  g.beginPath();
  g.ellipse(0, r * 0.7, r * 1.55, r * 0.32, 0, 0, Math.PI * 2);
  g.fill();
  g.restore();
  drawReferenceEnemy(g, type, r, { phase: 0.35, preview: true });
  g.shadowBlur = 0;
  g.strokeStyle = "rgba(185,255,189,0.35)";
  g.lineWidth = 0.9;
  g.beginPath();
  g.moveTo(-38, -24);
  g.lineTo(-25, -24);
  g.moveTo(38, 24);
  g.lineTo(25, 24);
  g.stroke();
  g.restore();
  return;
  if (type === "crawler") {
    for (const y of [-12, 0, 12]) {
      leg(-1, 12, y, y < 0 ? -0.45 : 0.6);
      leg(1, 12, y, y < 0 ? -0.45 : 0.6);
    }
    g.beginPath();
    g.ellipse(0, -1, 25, 18, 0, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.beginPath();
    g.arc(0, -6, 19, Math.PI * 1.04, Math.PI * 1.96);
    g.moveTo(-17, -4);
    g.quadraticCurveTo(0, 5, 17, -4);
    g.moveTo(-12, 6);
    g.quadraticCurveTo(0, 13, 12, 6);
    g.stroke();
    g.strokeRect(-8, -7, 16, 13);
    g.strokeRect(-4, 7, 8, 6);
    g.beginPath();
    g.arc(-7, -3, 2.1, 0, Math.PI * 2);
    g.arc(7, -3, 2.1, 0, Math.PI * 2);
    g.moveTo(-13, -16);
    g.lineTo(-25, -27);
    g.moveTo(13, -16);
    g.lineTo(25, -27);
    g.moveTo(-20, -25);
    g.lineTo(-30, -22);
    g.moveTo(20, -25);
    g.lineTo(30, -22);
    g.stroke();
  } else if (type === "beetle") {
    for (const y of [-14, 0, 14]) {
      leg(-1, 13, y, Math.sign(y || 1) * 0.55);
      leg(1, 13, y, Math.sign(y || 1) * 0.55);
    }
    g.beginPath();
    g.ellipse(-2, 0, 31, 21, 0, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    for (let i = -2; i <= 2; i += 1) {
      g.beginPath();
      g.arc(-8, 0, 8 + i * 3, -Math.PI * 0.5, Math.PI * 0.5);
      g.stroke();
    }
    g.beginPath();
    g.moveTo(-26, 0);
    g.lineTo(22, 0);
    g.moveTo(-18, -14);
    g.quadraticCurveTo(-2, -5, 19, -12);
    g.moveTo(-18, 14);
    g.quadraticCurveTo(-2, 5, 19, 12);
    g.moveTo(21, -10);
    g.lineTo(36, -4);
    g.lineTo(24, 0);
    g.lineTo(36, 4);
    g.lineTo(21, 10);
    g.stroke();
    bolt(7, -8);
    bolt(7, 8);
  } else if (type === "slime") {
    g.beginPath();
    g.moveTo(-26, 5);
    g.bezierCurveTo(-24, -18, -8, -25, 3, -17);
    g.bezierCurveTo(13, -30, 29, -12, 25, 5);
    g.bezierCurveTo(25, 25, 5, 26, -9, 20);
    g.bezierCurveTo(-24, 20, -31, 12, -26, 5);
    g.fill();
    g.stroke();
    g.beginPath();
    g.arc(4, -4, 4, 0, Math.PI * 2);
    g.arc(-10, 4, 2.8, 0, Math.PI * 2);
    g.arc(14, 9, 2.1, 0, Math.PI * 2);
    g.moveTo(-18, 15);
    g.lineTo(-21, 29);
    g.moveTo(1, 19);
    g.lineTo(-1, 31);
    g.moveTo(18, 10);
    g.lineTo(29, 19);
    g.stroke();
  } else if (type === "worm") {
    for (let i = 0; i < 6; i += 1) {
      const x = -29 + i * 10;
      g.beginPath();
      g.ellipse(x, Math.sin(i) * 2, 14, 10, 0, 0, Math.PI * 2);
      g.fill();
      g.stroke();
      g.beginPath();
      g.moveTo(x - 4, -9);
      g.lineTo(x + 4, 9);
      g.stroke();
    }
    g.beginPath();
    g.arc(32, 0, 13, 0, Math.PI * 2);
    g.stroke();
    g.beginPath();
    g.moveTo(42, -7);
    g.lineTo(55, -15);
    g.lineTo(49, -5);
    g.moveTo(42, 7);
    g.lineTo(55, 15);
    g.lineTo(49, 5);
    g.moveTo(45, 0);
    g.lineTo(58, 0);
    g.stroke();
  } else if (type === "wisp") {
    g.beginPath();
    g.arc(0, 0, 8, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    segmented(17, 0, Math.PI * 2, 8);
    segmented(27, Math.PI * 0.1, Math.PI * 1.95, 10);
    for (let i = 0; i < 9; i += 1) {
      const a = i * Math.PI * 2 / 9;
      g.beginPath();
      g.moveTo(Math.cos(a) * 8, Math.sin(a) * 8);
      g.lineTo(Math.cos(a + 0.22) * 22, Math.sin(a + 0.22) * 22);
      g.lineTo(Math.cos(a + 0.46) * 36, Math.sin(a + 0.46) * 36);
      g.stroke();
    }
  } else if (type === "juggernaut") {
    g.beginPath();
    g.ellipse(0, -4, 32, 21, 0, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.beginPath();
    g.arc(0, -10, 22, Math.PI * 1.08, Math.PI * 1.92);
    g.moveTo(-22, 1);
    g.quadraticCurveTo(0, 13, 22, 1);
    g.stroke();
    g.strokeRect(-17, -15, 34, 9);
    g.strokeRect(-8, -4, 16, 10);
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * 18, 8);
      g.lineTo(side * 27, 19);
      g.lineTo(side * 23, 30);
      g.moveTo(side * 8, 12);
      g.lineTo(side * 13, 26);
      g.lineTo(side * 7, 34);
      g.stroke();
      g.strokeRect(side * 21 - (side < 0 ? 12 : 0), 29, 12, 6);
      g.strokeRect(side * 5 - (side < 0 ? 10 : 0), 34, 10, 5);
    }
    for (const x of [-18, -9, 9, 18]) bolt(x, 3, 1.3);
    bolt(0, -10, 2.1);
  } else {
    g.beginPath();
    g.moveTo(20, -2);
    g.bezierCurveTo(12, -30, -12, -30, -20, -2);
    g.lineTo(-14, 28);
    g.lineTo(-4, 18);
    g.lineTo(0, 34);
    g.lineTo(4, 18);
    g.lineTo(14, 28);
    g.closePath();
    g.fill();
    g.stroke();
    g.beginPath();
    g.moveTo(-10, -4);
    g.lineTo(10, -4);
    g.moveTo(0, -18);
    g.lineTo(0, 14);
    g.moveTo(-7, 8);
    g.lineTo(-17, 22);
    g.moveTo(7, 8);
    g.lineTo(17, 22);
    g.stroke();
    g.save();
    g.globalAlpha = 0.46;
    for (let y = -10; y <= 18; y += 7) {
      g.beginPath();
      g.moveTo(-18, y);
      g.lineTo(18, y + Math.sin(y) * 2);
      g.stroke();
    }
    g.restore();
  }
  g.shadowBlur = 0;
  g.strokeStyle = "rgba(185,255,189,0.35)";
  g.lineWidth = 0.9;
  g.beginPath();
  g.moveTo(-38, -24);
  g.lineTo(-25, -24);
  g.moveTo(38, 24);
  g.lineTo(25, 24);
  g.stroke();
  g.restore();
}

function drawReferenceEnemy(g, type, r, options = {}) {
  const phase = options.phase || 0;
  const preview = !!options.preview;
  const lineBase = g.lineWidth || 1.5;
  const dot = (x, y, size = r * 0.07) => {
    g.beginPath();
    g.arc(x, y, Math.max(1, size), 0, Math.PI * 2);
    g.stroke();
  };
  const segmentedArc = (x, y, radius, start, end, pieces = 8) => {
    const span = end - start;
    for (let i = 0; i < pieces; i += 1) {
      g.beginPath();
      g.arc(x, y, radius, start + span * (i / pieces), start + span * ((i + 0.62) / pieces));
      g.stroke();
    }
  };
  const scratch = (points) => {
    g.save();
    g.globalAlpha = preview ? 0.62 : 0.42;
    g.strokeStyle = "rgba(185,255,189,0.58)";
    g.lineWidth = Math.max(0.7, lineBase * 0.58);
    for (const [x1, y1, x2, y2] of points) {
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
    }
    g.restore();
  };
  const grain = (points, alpha = preview ? 0.46 : 0.32) => {
    g.save();
    g.globalAlpha = alpha;
    g.strokeStyle = "rgba(230,255,216,0.64)";
    g.lineWidth = Math.max(0.55, lineBase * 0.36);
    for (const [x, y, w, rot = 0] of points) {
      g.save();
      g.translate(x, y);
      g.rotate(rot);
      g.beginPath();
      g.moveTo(-w * 0.5, 0);
      g.lineTo(w * 0.5, 0);
      g.stroke();
      g.restore();
    }
    g.restore();
  };
  const rivets = (points, size = r * 0.038) => {
    g.save();
    g.globalAlpha = preview ? 0.82 : 0.7;
    g.lineWidth = Math.max(0.65, lineBase * 0.42);
    for (const [x, y] of points) dot(x, y, size);
    g.restore();
  };
  const cable = (points) => {
    g.save();
    g.globalAlpha = preview ? 0.86 : 0.72;
    g.lineWidth = Math.max(0.8, lineBase * 0.5);
    g.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.stroke();
    g.restore();
  };
  const hatchEllipse = (cx, cy, rx, ry, count = 5, slant = 0.4) => {
    g.save();
    g.globalAlpha = preview ? 0.5 : 0.34;
    g.lineWidth = Math.max(0.55, lineBase * 0.36);
    for (let i = -count; i <= count; i += 1) {
      const y = cy + (i / count) * ry * 0.78;
      const span = rx * Math.sqrt(Math.max(0, 1 - Math.pow((y - cy) / ry, 2))) * 0.72;
      g.beginPath();
      g.moveTo(cx - span, y + span * slant * 0.22);
      g.lineTo(cx + span, y - span * slant * 0.22);
      g.stroke();
    }
    g.restore();
  };
  const antenna = (side, x, y, length, bend = -0.35) => {
    g.save();
    g.globalAlpha = preview ? 0.9 : 0.74;
    g.lineWidth = Math.max(0.75, lineBase * 0.48);
    g.beginPath();
    g.moveTo(side * x, y);
    g.quadraticCurveTo(side * (x + length * 0.36), y - length * 0.32, side * (x + length), y + length * bend);
    g.stroke();
    dot(side * (x + length), y + length * bend, r * 0.032);
    g.restore();
  };
  const distress = (kind) => {
    const marks = {
      crawler: [[-0.82, -0.58, 0.32, -0.12], [-0.26, 0.58, 0.2, 0.08], [0.52, -0.5, 0.28, 0.16], [0.9, 0.28, 0.22, -0.2]],
      beetle: [[-0.5, -0.78, 0.36, -0.1], [0.08, -0.68, 0.42, 0.12], [0.7, 0.42, 0.34, -0.18], [-0.78, 0.5, 0.28, 0.14]],
      slime: [[-0.72, -0.18, 0.34, -0.3], [-0.24, 0.64, 0.28, 0.1], [0.42, -0.6, 0.3, 0.22], [0.86, 0.2, 0.24, -0.18]],
      worm: [[-1.18, 0.48, 0.4, 0.14], [-0.48, -0.34, 0.34, 0.2], [0.3, 0.44, 0.32, -0.12], [1.2, -0.38, 0.28, 0.18]],
      wisp: [[-0.92, -0.66, 0.28, -0.12], [-0.52, 0.76, 0.24, 0.2], [0.66, -0.82, 0.22, 0.1], [0.98, 0.48, 0.26, -0.24]],
      juggernaut: [[-0.88, -0.56, 0.42, 0.1], [-0.42, 0.62, 0.32, -0.16], [0.36, -0.66, 0.36, 0.16], [0.92, 0.3, 0.3, -0.22]],
      phantom: [[-0.42, -0.98, 0.28, 0.16], [-0.28, 0.38, 0.24, -0.12], [0.32, -0.18, 0.28, 0.2], [0.5, 0.78, 0.22, -0.16]],
      mite: [[-0.88, -0.44, 0.34, -0.1], [-0.3, 0.52, 0.32, 0.16], [0.42, -0.48, 0.3, 0.12], [0.86, 0.22, 0.24, -0.2]],
      leech: [[-1.24, 0.42, 0.42, -0.08], [-0.58, -0.36, 0.36, 0.14], [0.18, 0.38, 0.34, -0.18], [1.14, -0.3, 0.28, 0.18]],
      obelisk: [[-0.32, -1.18, 0.28, 0.16], [0.22, -0.74, 0.32, -0.2], [-0.18, 0.3, 0.26, 0.12], [0.36, 0.66, 0.24, -0.18]]
    }[kind] || [];
    grain(marks.map(([x, y, w, rot]) => [x * r, y * r, w * r, rot]), preview ? 0.38 : 0.26);
  };
  const bladeLeg = (side, x, y, a = 1, forward = 1) => {
    g.beginPath();
    g.moveTo(side * x, y);
    g.lineTo(side * (x + r * 0.32), y + r * 0.18 * a);
    g.lineTo(side * (x + r * 0.62), y + r * 0.72 * a);
    g.lineTo(side * (x + r * 0.5), y + r * 0.76 * a);
    g.lineTo(side * (x + r * 0.26), y + r * 0.24 * a);
    g.stroke();
    if (forward) dot(side * (x + r * 0.31), y + r * 0.18 * a, r * 0.045);
  };
  const shellPlates = (cx, cy, rx, ry, pieces = 5) => {
    for (let i = 1; i < pieces; i += 1) {
      const t = -1 + (i / pieces) * 2;
      g.beginPath();
      g.moveTo(cx + t * rx, cy - ry * Math.sqrt(Math.max(0, 1 - t * t)));
      g.quadraticCurveTo(cx + t * rx * 0.52, cy, cx + t * rx, cy + ry * Math.sqrt(Math.max(0, 1 - t * t)));
      g.stroke();
    }
    for (let i = -1; i <= 1; i += 1) {
      g.beginPath();
      g.moveTo(cx - rx * 0.72, cy + i * ry * 0.34);
      g.quadraticCurveTo(cx, cy + i * ry * 0.15, cx + rx * 0.72, cy + i * ry * 0.34);
      g.stroke();
    }
  };

  g.save();
  g.lineCap = "round";
  g.lineJoin = "round";
  g.lineWidth = Math.max(1.28, lineBase * 1.08);

  if (type === "crawler") {
    for (const y of [-0.58, -0.08, 0.46]) {
      bladeLeg(-1, r * 0.44, r * y, y < 0 ? -0.7 : 0.8);
      bladeLeg(1, r * 0.44, r * y, y < 0 ? -0.7 : 0.8);
    }
    g.beginPath();
    g.ellipse(0, -r * 0.06, r * 1.1, r * 0.76, 0, Math.PI, Math.PI * 2);
    g.lineTo(r * 0.92, r * 0.16);
    g.quadraticCurveTo(0, r * 0.48, -r * 0.92, r * 0.16);
    g.closePath();
    g.fill();
    g.stroke();
    g.beginPath();
    g.moveTo(-r * 0.96, r * 0.04);
    g.lineTo(r * 0.96, r * 0.04);
    g.moveTo(-r * 0.58, -r * 0.52);
    g.lineTo(-r * 0.22, r * 0.12);
    g.moveTo(r * 0.58, -r * 0.52);
    g.lineTo(r * 0.22, r * 0.12);
    g.moveTo(-r * 0.16, -r * 0.62);
    g.lineTo(r * 0.16, r * 0.1);
    g.stroke();
    dot(-r * 0.38, -r * 0.2);
    dot(r * 0.38, -r * 0.2);
    for (const [side, y, reach] of [[-1, -0.22, 1.14], [1, -0.22, 1.14], [-1, 0.2, 1.36], [1, 0.2, 1.36], [-1, 0.5, 1.12], [1, 0.5, 1.12]]) {
      g.beginPath();
      g.moveTo(side * r * 0.72, r * y);
      g.lineTo(side * r * reach, r * (y + 0.26));
      g.lineTo(side * r * (reach + 0.18), r * (y + 0.76));
      g.lineTo(side * r * (reach + 0.02), r * (y + 0.82));
      g.lineTo(side * r * (reach - 0.16), r * (y + 0.34));
      g.stroke();
    }
    segmentedArc(0, -r * 0.06, r * 0.9, Math.PI * 1.05, Math.PI * 1.95, 6);
    scratch([[-r * 0.78, -r * 0.42, -r * 0.56, -r * 0.55], [r * 0.2, -r * 0.55, r * 0.5, -r * 0.47], [-r * 0.16, r * 0.22, r * 0.18, r * 0.28]]);
    antenna(-1, r * 0.48, -r * 0.5, r * 0.55, -0.42);
    antenna(1, r * 0.48, -r * 0.5, r * 0.55, -0.42);
    hatchEllipse(0, -r * 0.08, r * 0.78, r * 0.54, 5, 0.55);
    cable([[-r * 0.72, r * 0.14], [-r * 0.36, r * 0.36], [0, r * 0.25], [r * 0.36, r * 0.36], [r * 0.72, r * 0.14]]);
    rivets([[-r * 0.7, r * 0.08], [-r * 0.44, r * 0.25], [0, r * 0.28], [r * 0.44, r * 0.25], [r * 0.7, r * 0.08]], r * 0.035);
    distress("crawler");
  } else if (type === "beetle") {
    for (const y of [-0.58, -0.1, 0.42]) {
      bladeLeg(-1, r * 0.72, r * y, y < 0 ? -0.72 : 0.82);
      bladeLeg(1, r * 0.72, r * y, y < 0 ? -0.72 : 0.82);
    }
    g.beginPath();
    g.ellipse(0, -r * 0.03, r * 1.25, r * 0.86, 0, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    shellPlates(0, -r * 0.03, r * 1.18, r * 0.8, 6);
    g.beginPath();
    g.ellipse(-r * 0.82, r * 0.04, r * 0.45, r * 0.38, 0, 0, Math.PI * 2);
    g.stroke();
    g.moveTo(-r * 1.08, -r * 0.05);
    g.lineTo(-r * 1.55, -r * 0.38);
    g.lineTo(-r * 1.38, -r * 0.04);
    g.moveTo(-r * 1.08, r * 0.12);
    g.lineTo(-r * 1.55, r * 0.45);
    g.lineTo(-r * 1.34, r * 0.1);
    g.stroke();
    dot(-r * 0.92, -r * 0.12);
    dot(-r * 0.66, -r * 0.08);
    scratch([[-r * 0.26, -r * 0.68, r * 0.08, -r * 0.52], [r * 0.42, -r * 0.42, r * 0.82, -r * 0.22], [-r * 0.42, r * 0.46, -r * 0.02, r * 0.58]]);
    for (const [x, y, bend] of [[-0.72, 0.46, -1], [-0.28, 0.58, -0.5], [0.26, 0.58, 0.5], [0.72, 0.46, 1]]) {
      g.beginPath();
      g.moveTo(r * x, r * y);
      g.lineTo(r * (x + bend * 0.12), r * 0.92);
      g.lineTo(r * (x + bend * 0.28), r * 1.14);
      g.lineTo(r * (x + bend * 0.08), r * 1.04);
      g.stroke();
    }
    g.beginPath();
    g.moveTo(-r * 1.05, -r * 0.42);
    g.quadraticCurveTo(-r * 0.44, -r * 1.0, r * 0.74, -r * 0.74);
    g.moveTo(-r * 1.12, r * 0.42);
    g.quadraticCurveTo(-r * 0.36, r * 0.88, r * 0.82, r * 0.6);
    g.stroke();
    hatchEllipse(r * 0.16, -r * 0.02, r * 0.86, r * 0.62, 6, -0.35);
    cable([[-r * 1.02, r * 0.18], [-r * 0.54, r * 0.5], [r * 0.08, r * 0.42], [r * 0.72, r * 0.18]]);
    rivets([[-r * 0.48, -r * 0.42], [0, -r * 0.54], [r * 0.5, -r * 0.36], [-r * 0.5, r * 0.36], [0, r * 0.5], [r * 0.54, r * 0.3]], r * 0.035);
    distress("beetle");
  } else if (type === "slime") {
    g.beginPath();
    g.moveTo(-r * 1.28, r * 0.32);
    g.bezierCurveTo(-r * 1.02, -r * 0.52, -r * 0.56, -r * 1.06, r * 0.02, -r * 0.86);
    g.bezierCurveTo(r * 0.38, -r * 1.34, r * 1.02, -r * 0.7, r * 0.98, r * 0.02);
    g.bezierCurveTo(r * 1.3, r * 0.36, r * 0.82, r * 0.94, r * 0.22, r * 0.76);
    g.bezierCurveTo(-r * 0.2, r * 1.12, -r * 0.86, r * 0.92, -r * 1.28, r * 0.32);
    g.fill();
    g.stroke();
    for (const [x, y, rr] of [[0.26, -0.24, 0.14], [-0.34, 0.04, 0.1], [0.58, 0.2, 0.08], [-0.02, 0.34, 0.06]]) {
      dot(r * x, r * y, r * rr);
    }
    g.beginPath();
    g.moveTo(-r * 0.88, r * 0.48);
    g.lineTo(-r * 1.24, r * 0.92);
    g.moveTo(-r * 0.28, r * 0.74);
    g.lineTo(-r * 0.38, r * 1.24);
    g.moveTo(r * 0.68, r * 0.38);
    g.lineTo(r * 1.08, r * 0.68);
    g.stroke();
    g.beginPath();
    g.ellipse(r * 1.42, r * 0.42, r * 0.28, r * 0.18, -0.3, 0, Math.PI * 2);
    g.stroke();
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * r * 0.62, r * 0.16);
      g.bezierCurveTo(side * r * 1.1, r * 0.24, side * r * 1.18, r * 0.82, side * r * 0.76, r * 0.98);
      g.bezierCurveTo(side * r * 0.42, r * 0.8, side * r * 0.34, r * 0.44, side * r * 0.62, r * 0.16);
      g.stroke();
    }
    g.beginPath();
    g.moveTo(-r * 0.4, -r * 0.42);
    g.bezierCurveTo(-r * 0.18, -r * 0.66, r * 0.28, -r * 0.66, r * 0.48, -r * 0.38);
    g.moveTo(-r * 0.72, r * 0.24);
    g.bezierCurveTo(-r * 0.34, r * 0.44, r * 0.26, r * 0.44, r * 0.66, r * 0.18);
    g.stroke();
    scratch([[-r * 0.78, -r * 0.02, -r * 0.48, -r * 0.18], [r * 0.1, -r * 0.62, r * 0.38, -r * 0.48]]);
    for (const [x, y, tx, ty] of [[-0.9, 0.5, -1.35, 1.06], [-0.24, 0.76, -0.28, 1.34], [0.48, 0.58, 0.82, 1.08]]) {
      g.beginPath();
      g.moveTo(r * x, r * y);
      g.bezierCurveTo(r * (x + tx) * 0.5, r * (y + ty) * 0.5, r * tx, r * (ty - 0.12), r * tx, r * ty);
      g.stroke();
      dot(r * tx, r * ty, r * 0.038);
    }
    g.beginPath();
    g.moveTo(-r * 0.7, -r * 0.28);
    g.bezierCurveTo(-r * 0.36, -r * 0.98, r * 0.5, -r * 0.88, r * 0.8, -r * 0.18);
    g.stroke();
    hatchEllipse(-r * 0.04, r * 0.04, r * 0.78, r * 0.58, 5, 0.2);
    cable([[-r * 0.9, r * 0.38], [-r * 0.42, r * 0.62], [r * 0.16, r * 0.55], [r * 0.7, r * 0.28]]);
    rivets([[-r * 0.76, r * 0.28], [-r * 0.32, -r * 0.18], [r * 0.08, -r * 0.42], [r * 0.54, r * 0.02], [r * 0.28, r * 0.48]], r * 0.032);
    distress("slime");
  } else if (type === "worm") {
    const points = [];
    for (let i = 0; i < 7; i += 1) {
      const x = r * (1.28 - i * 0.45);
      const y = Math.sin(phase * 4 + i * 0.72) * r * 0.16 + (i > 3 ? r * 0.14 : 0);
      points.push({ x, y });
    }
    for (let i = points.length - 1; i >= 0; i -= 1) {
      const p = points[i];
      g.beginPath();
      g.ellipse(p.x, p.y, r * 0.42, r * 0.34, -0.16, 0, Math.PI * 2);
      g.fill();
      g.stroke();
      g.beginPath();
      g.moveTo(p.x - r * 0.16, p.y - r * 0.3);
      g.lineTo(p.x + r * 0.12, p.y + r * 0.3);
      g.stroke();
    }
    g.beginPath();
    g.arc(r * 1.55, points[0].y, r * 0.43, 0, Math.PI * 2);
    g.stroke();
    segmentedArc(r * 1.58, points[0].y, r * 0.24, 0, Math.PI * 2, 9);
    g.beginPath();
    g.moveTo(r * 1.9, points[0].y - r * 0.24);
    g.lineTo(r * 2.25, points[0].y - r * 0.5);
    g.moveTo(r * 1.92, points[0].y + r * 0.22);
    g.lineTo(r * 2.25, points[0].y + r * 0.5);
    g.stroke();
    scratch([[r * -1.22, r * 0.55, r * -0.82, r * 0.67], [r * -0.18, r * -0.32, r * 0.14, r * -0.22], [r * 0.52, r * 0.34, r * 0.86, r * 0.42]]);
    for (let i = 0; i < points.length - 1; i += 1) {
      cable([[points[i].x - r * 0.12, points[i].y - r * 0.24], [points[i + 1].x + r * 0.16, points[i + 1].y - r * 0.18]]);
    }
    rivets(points.map((p) => [p.x + r * 0.08, p.y - r * 0.1]), r * 0.032);
    grain(points.map((p, index) => [p.x - r * 0.18, p.y + r * 0.22, r * (0.22 + index * 0.02), 0.32]), preview ? 0.35 : 0.24);
    distress("worm");
  } else if (type === "wisp") {
    g.beginPath();
    g.arc(0, 0, r * 0.28, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    dot(0, 0, r * 0.1);
    for (let i = 0; i < 9; i += 1) {
      const a = i * Math.PI * 2 / 9 + phase * 0.7;
      const p1 = r * 0.35;
      const p2 = r * (0.9 + (i % 2) * 0.18);
      const p3 = r * (1.55 + (i % 3) * 0.24);
      g.beginPath();
      g.moveTo(Math.cos(a) * p1, Math.sin(a) * p1);
      g.lineTo(Math.cos(a + 0.22) * p2, Math.sin(a + 0.22) * p2);
      g.lineTo(Math.cos(a - 0.16) * p3, Math.sin(a - 0.16) * p3);
      g.stroke();
    }
    segmentedArc(0, 0, r * 0.58, 0, Math.PI * 2, 6);
    segmentedArc(0, 0, r * 0.9, Math.PI * 0.08, Math.PI * 1.9, 9);
    segmentedArc(0, 0, r * 1.22, Math.PI * 0.2, Math.PI * 1.72, 10);
    rivets([[0, -r * 0.42], [r * 0.42, 0], [0, r * 0.42], [-r * 0.42, 0]], r * 0.032);
    grain([[-r * 1.28, -r * 0.62, r * 0.42, -0.26], [-r * 0.86, r * 0.95, r * 0.34, 0.3], [r * 0.76, -r * 1.0, r * 0.38, 0.18], [r * 1.22, r * 0.58, r * 0.32, -0.35]], preview ? 0.48 : 0.34);
    distress("wisp");
  } else if (type === "juggernaut") {
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * r * 0.78, r * 0.18);
      g.lineTo(side * r * 1.12, r * 0.66);
      g.lineTo(side * r * 1.02, r * 1.08);
      g.lineTo(side * r * 0.78, r * 1.1);
      g.moveTo(side * r * 0.28, r * 0.38);
      g.lineTo(side * r * 0.46, r * 0.96);
      g.lineTo(side * r * 0.24, r * 1.24);
      g.stroke();
      g.strokeRect(side * r * 0.78 - (side < 0 ? r * 0.38 : 0), r * 1.02, r * 0.38, r * 0.22);
      g.strokeRect(side * r * 0.17 - (side < 0 ? r * 0.34 : 0), r * 1.2, r * 0.34, r * 0.2);
    }
    g.beginPath();
    g.ellipse(0, -r * 0.14, r * 1.25, r * 0.82, 0, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    shellPlates(0, -r * 0.14, r * 1.14, r * 0.72, 5);
    g.strokeRect(-r * 0.58, -r * 0.52, r * 1.16, r * 0.36);
    for (const x of [-0.7, -0.36, 0.36, 0.7]) dot(x * r, r * 0.08, r * 0.055);
    scratch([[-r * 0.86, -r * 0.16, -r * 0.44, -r * 0.26], [r * 0.28, -r * 0.62, r * 0.72, -r * 0.46]]);
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * r * 0.98, -r * 0.18);
      g.lineTo(side * r * 1.34, r * 0.16);
      g.lineTo(side * r * 1.22, r * 0.52);
      g.lineTo(side * r * 1.02, r * 0.24);
      g.moveTo(side * r * 0.7, r * 0.28);
      g.lineTo(side * r * 0.9, r * 0.82);
      g.lineTo(side * r * 0.62, r * 1.12);
      g.stroke();
      g.strokeRect(side * r * 1.1 - (side < 0 ? r * 0.28 : 0), r * 0.5, r * 0.28, r * 0.18);
    }
    g.beginPath();
    g.moveTo(-r * 0.74, -r * 0.58);
    g.lineTo(-r * 0.42, -r * 0.84);
    g.lineTo(r * 0.42, -r * 0.84);
    g.lineTo(r * 0.74, -r * 0.58);
    g.stroke();
    hatchEllipse(0, -r * 0.14, r * 0.92, r * 0.52, 6, -0.25);
    cable([[-r * 0.86, r * 0.22], [-r * 0.34, r * 0.46], [r * 0.3, r * 0.46], [r * 0.86, r * 0.2]]);
    rivets([[-r * 0.86, -r * 0.1], [-r * 0.58, -r * 0.34], [-r * 0.24, r * 0.18], [r * 0.24, r * 0.18], [r * 0.58, -r * 0.34], [r * 0.86, -r * 0.1]], r * 0.04);
    distress("juggernaut");
  } else if (type === "phantom") {
    g.beginPath();
    g.moveTo(0, -r * 1.22);
    g.bezierCurveTo(r * 0.72, -r * 1.12, r * 0.62, r * 0.24, r * 0.46, r * 1.18);
    g.lineTo(r * 0.18, r * 0.78);
    g.lineTo(0, r * 1.42);
    g.lineTo(-r * 0.18, r * 0.78);
    g.lineTo(-r * 0.46, r * 1.18);
    g.bezierCurveTo(-r * 0.62, r * 0.24, -r * 0.72, -r * 1.12, 0, -r * 1.22);
    g.fill();
    g.stroke();
    g.beginPath();
    g.ellipse(0, -r * 0.72, r * 0.34, r * 0.44, 0, 0, Math.PI * 2);
    g.stroke();
    dot(-r * 0.1, -r * 0.76, r * 0.055);
    dot(r * 0.1, -r * 0.76, r * 0.055);
    g.beginPath();
    g.moveTo(-r * 0.22, -r * 0.48);
    g.bezierCurveTo(-r * 0.1, -r * 0.38, r * 0.1, -r * 0.38, r * 0.22, -r * 0.48);
    g.moveTo(-r * 0.3, -r * 0.92);
    g.bezierCurveTo(-r * 0.06, -r * 1.08, r * 0.24, -r * 0.98, r * 0.34, -r * 0.72);
    g.stroke();
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * r * 0.36, -r * 0.1);
      g.bezierCurveTo(side * r * 0.82, r * 0.14, side * r * 0.56, r * 0.72, side * r * 0.94, r * 1.02);
      g.moveTo(side * r * 0.2, r * 0.18);
      g.bezierCurveTo(side * r * 0.42, r * 0.68, side * r * 0.16, r * 0.9, side * r * 0.34, r * 1.34);
      g.stroke();
    }
    for (let x = -0.36; x <= 0.36; x += 0.18) {
      g.beginPath();
      g.moveTo(r * x, -r * 0.1);
      g.bezierCurveTo(r * (x + 0.08), r * 0.36, r * (x - 0.08), r * 0.78, r * x, r * 1.12);
      g.stroke();
    }
    scratch([[-r * 0.22, -r * 0.28, -r * 0.06, r * 0.58], [r * 0.2, -r * 0.2, r * 0.08, r * 0.88]]);
    hatchEllipse(0, -r * 0.14, r * 0.38, r * 0.92, 7, 0.05);
    cable([[-r * 0.36, -r * 0.02], [-r * 0.12, r * 0.4], [0, r * 0.92], [r * 0.12, r * 0.4], [r * 0.36, -r * 0.02]]);
    rivets([[-r * 0.28, -r * 0.62], [r * 0.28, -r * 0.62], [-r * 0.2, r * 0.16], [r * 0.2, r * 0.16], [0, r * 0.74]], r * 0.03);
    distress("phantom");
  } else if (type === "mite") {
    for (const y of [-0.5, -0.05, 0.4]) {
      bladeLeg(-1, r * 0.58, r * y, y < 0 ? -0.7 : 0.78);
      bladeLeg(1, r * 0.58, r * y, y < 0 ? -0.7 : 0.78);
    }
    g.beginPath();
    g.ellipse(r * 0.08, 0, r * 1.0, r * 0.58, 0, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    shellPlates(r * 0.08, 0, r * 0.92, r * 0.52, 5);
    g.beginPath();
    g.ellipse(-r * 0.84, 0, r * 0.42, r * 0.34, 0, 0, Math.PI * 2);
    g.stroke();
    g.moveTo(-r * 1.15, -r * 0.08);
    g.lineTo(-r * 1.72, -r * 0.42);
    g.lineTo(-r * 1.42, -r * 0.02);
    g.moveTo(-r * 1.15, r * 0.08);
    g.lineTo(-r * 1.72, r * 0.42);
    g.lineTo(-r * 1.42, r * 0.02);
    g.stroke();
    g.beginPath();
    g.moveTo(-r * 1.08, -r * 0.34);
    g.quadraticCurveTo(-r * 0.62, -r * 0.56, -r * 0.16, -r * 0.46);
    g.moveTo(-r * 1.08, r * 0.34);
    g.quadraticCurveTo(-r * 0.62, r * 0.56, -r * 0.16, r * 0.46);
    g.stroke();
    dot(-r * 0.92, -r * 0.1, r * 0.05);
    scratch([[-r * 0.24, -r * 0.42, r * 0.14, -r * 0.36], [r * 0.3, r * 0.28, r * 0.66, r * 0.16]]);
    antenna(-1, r * 0.96, -r * 0.18, r * 0.52, -0.18);
    hatchEllipse(r * 0.12, 0, r * 0.72, r * 0.42, 4, 0.32);
    cable([[-r * 0.62, r * 0.2], [-r * 0.16, r * 0.34], [r * 0.38, r * 0.24], [r * 0.86, r * 0.02]]);
    rivets([[-r * 0.52, -r * 0.18], [-r * 0.12, -r * 0.3], [r * 0.28, -r * 0.22], [r * 0.66, -r * 0.02], [r * 0.28, r * 0.26], [-r * 0.18, r * 0.28]], r * 0.033);
    distress("mite");
  } else if (type === "leech") {
    const points = [];
    for (let i = 0; i < 7; i += 1) {
      points.push({ x: r * (1.35 - i * 0.44), y: -r * 0.08 + Math.sin(i * 0.7 + phase) * r * 0.12 });
    }
    for (let i = points.length - 1; i >= 0; i -= 1) {
      const p = points[i];
      g.beginPath();
      g.ellipse(p.x, p.y, r * 0.46, r * 0.38, -0.22, 0, Math.PI * 2);
      g.fill();
      g.stroke();
      segmentedArc(p.x, p.y, r * 0.34, -Math.PI * 0.5, Math.PI * 0.5, 3);
    }
    const hx = r * 1.62;
    g.beginPath();
    g.arc(hx, -r * 0.08, r * 0.52, 0, Math.PI * 2);
    g.stroke();
    g.beginPath();
    g.arc(hx + r * 0.1, -r * 0.08, r * 0.28, 0, Math.PI * 2);
    g.stroke();
    for (let i = 0; i < 12; i += 1) {
      const a = i * Math.PI * 2 / 12;
      g.beginPath();
      g.moveTo(hx + Math.cos(a) * r * 0.18, -r * 0.08 + Math.sin(a) * r * 0.18);
      g.lineTo(hx + Math.cos(a) * r * 0.3, -r * 0.08 + Math.sin(a) * r * 0.3);
      g.stroke();
    }
    scratch([[-r * 1.08, r * 0.38, -r * 0.72, r * 0.52], [-r * 0.18, -r * 0.38, r * 0.1, -r * 0.3], [r * 0.55, r * 0.34, r * 0.9, r * 0.38]]);
    for (let i = 0; i < points.length - 1; i += 1) {
      cable([[points[i].x - r * 0.06, points[i].y + r * 0.24], [points[i + 1].x + r * 0.12, points[i + 1].y + r * 0.18]]);
    }
    grain(points.map((p, index) => [p.x, p.y - r * 0.34, r * (0.24 + index * 0.018), -0.28]), preview ? 0.4 : 0.28);
    rivets(points.map((p) => [p.x + r * 0.12, p.y + r * 0.05]), r * 0.032);
    distress("leech");
  } else if (type === "obelisk") {
    g.beginPath();
    g.moveTo(0, -r * 1.64);
    g.lineTo(r * 0.45, -r * 0.46);
    g.lineTo(r * 0.28, r * 0.58);
    g.lineTo(0, r * 1.16);
    g.lineTo(-r * 0.28, r * 0.58);
    g.lineTo(-r * 0.45, -r * 0.46);
    g.closePath();
    g.fill();
    g.stroke();
    g.beginPath();
    g.moveTo(0, -r * 1.64);
    g.lineTo(0, r * 1.16);
    g.moveTo(-r * 0.45, -r * 0.46);
    g.lineTo(r * 0.45, -r * 0.46);
    g.moveTo(-r * 0.28, r * 0.58);
    g.lineTo(r * 0.28, r * 0.58);
    g.moveTo(-r * 0.24, -r * 0.16);
    g.lineTo(r * 0.22, -r * 0.78);
    g.moveTo(r * 0.18, -r * 0.08);
    g.lineTo(-r * 0.16, r * 0.44);
    g.stroke();
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * r * 0.66, -r * 0.92);
      g.lineTo(side * r * 0.88, -r * 0.56);
      g.lineTo(side * r * 0.68, -r * 0.2);
      g.lineTo(side * r * 0.48, -r * 0.56);
      g.closePath();
      g.stroke();
      g.beginPath();
      g.moveTo(side * r * 0.72, r * 0.1);
      g.lineTo(side * r * 1.04, r * 0.5);
      g.lineTo(side * r * 0.72, r * 0.92);
      g.lineTo(side * r * 0.52, r * 0.48);
      g.closePath();
      g.stroke();
    }
    scratch([[-r * 0.16, -r * 1.1, r * 0.12, -r * 0.86], [-r * 0.12, r * 0.22, r * 0.12, r * 0.42]]);
    hatchEllipse(0, -r * 0.18, r * 0.3, r * 1.08, 8, -0.08);
    cable([[0, -r * 1.34], [r * 0.2, -r * 0.74], [0, -r * 0.12], [-r * 0.18, r * 0.46], [0, r * 0.94]]);
    rivets([[0, -r * 1.22], [-r * 0.2, -r * 0.46], [r * 0.22, -r * 0.46], [-r * 0.14, r * 0.36], [r * 0.14, r * 0.36]], r * 0.03);
    distress("obelisk");
  } else {
    drawReferenceEnemy(g, "phantom", r, options);
  }
  g.restore();
}
