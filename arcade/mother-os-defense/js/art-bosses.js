"use strict";

function drawBossSchematic(g, type, r, options = {}) {
  const def = bossDefs[type] || bossDefs.hive;
  const preview = !!options.preview;
  const phase = options.phase || 0;
  const lineBase = g.lineWidth || 2;
  const ink = def.color || "#ffcf5a";
  const core = def.accent || "#8dff53";
  const glow = preview ? 11 : 14;
  const dot = (x, y, size = r * 0.08, fill = false) => {
    g.beginPath();
    g.arc(x, y, Math.max(1.2, size), 0, Math.PI * 2);
    if (fill) g.fill();
    g.stroke();
  };
  const blade = (points, serrated = true) => {
    g.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.stroke();
    if (!serrated) return;
    g.save();
    g.globalAlpha = preview ? 0.72 : 0.54;
    g.lineWidth = Math.max(0.72, lineBase * 0.34);
    for (let i = 1; i < points.length - 1; i += 1) {
      const [x, y] = points[i];
      const [nx, ny] = points[i + 1];
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + (ny - y) * 0.12, y - (nx - x) * 0.12);
      g.stroke();
    }
    g.restore();
  };
  const scratch = (marks, alpha = preview ? 0.58 : 0.4) => {
    g.save();
    g.globalAlpha = alpha;
    g.strokeStyle = "rgba(255,238,107,0.72)";
    g.lineWidth = Math.max(0.7, lineBase * 0.36);
    for (const [x1, y1, x2, y2] of marks) {
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
    }
    g.restore();
  };
  const greenTrace = (draw, alpha = preview ? 0.86 : 0.76, blur = 5) => {
    g.save();
    g.strokeStyle = core;
    g.fillStyle = "rgba(112,255,67,0.26)";
    g.globalAlpha = alpha;
    g.shadowColor = core;
    g.shadowBlur = blur;
    draw();
    g.restore();
  };
  const brokenArc = (x, y, radius, start, end, pieces = 8) => {
    const span = end - start;
    for (let i = 0; i < pieces; i += 1) {
      g.beginPath();
      g.arc(x, y, radius, start + span * (i / pieces), start + span * ((i + 0.62) / pieces));
      g.stroke();
    }
  };
  const hatchEllipse = (cx, cy, rx, ry, rows = 6, slant = 0.25) => {
    g.save();
    g.globalAlpha = preview ? 0.46 : 0.32;
    g.lineWidth = Math.max(0.6, lineBase * 0.32);
    for (let i = -rows; i <= rows; i += 1) {
      const y = cy + (i / rows) * ry * 0.78;
      const span = rx * Math.sqrt(Math.max(0, 1 - Math.pow((y - cy) / ry, 2))) * 0.72;
      g.beginPath();
      g.moveTo(cx - span, y + span * slant * 0.2);
      g.lineTo(cx + span, y - span * slant * 0.2);
      g.stroke();
    }
    g.restore();
  };
  const shellGrid = (cx, cy, rx, ry, cols = 5, rows = 3) => {
    g.save();
    g.globalAlpha = preview ? 0.64 : 0.46;
    g.lineWidth = Math.max(0.72, lineBase * 0.36);
    for (let i = 1; i < cols; i += 1) {
      const t = -1 + (i / cols) * 2;
      g.beginPath();
      g.moveTo(cx + t * rx, cy - ry * Math.sqrt(Math.max(0, 1 - t * t)));
      g.quadraticCurveTo(cx + t * rx * 0.5, cy, cx + t * rx, cy + ry * Math.sqrt(Math.max(0, 1 - t * t)));
      g.stroke();
    }
    for (let i = -rows; i <= rows; i += 1) {
      g.beginPath();
      g.moveTo(cx - rx * 0.75, cy + (i / rows) * ry * 0.42);
      g.quadraticCurveTo(cx, cy + (i / rows) * ry * 0.2, cx + rx * 0.75, cy + (i / rows) * ry * 0.42);
      g.stroke();
    }
    g.restore();
  };
  const rivetBand = (points, size = r * 0.035) => {
    g.save();
    g.globalAlpha = preview ? 0.8 : 0.62;
    g.lineWidth = Math.max(0.62, lineBase * 0.3);
    for (const [x, y] of points) dot(x, y, size);
    g.restore();
  };
  const grain = (marks, alpha = preview ? 0.36 : 0.25) => {
    g.save();
    g.globalAlpha = alpha;
    g.strokeStyle = "rgba(255,245,155,0.72)";
    g.lineWidth = Math.max(0.52, lineBase * 0.25);
    for (const [x, y, w, rot = 0] of marks) {
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
  const cable = (points, alpha = preview ? 0.62 : 0.5) => {
    g.save();
    g.globalAlpha = alpha;
    g.lineWidth = Math.max(0.8, lineBase * 0.4);
    g.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.stroke();
    g.restore();
  };
  const plate = (points, alpha = 0.1) => {
    g.save();
    g.fillStyle = `rgba(255,207,90,${alpha})`;
    g.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.closePath();
    g.fill();
    g.stroke();
    g.restore();
  };
  const joint = (x, y, size = r * 0.065) => {
    dot(x, y, size);
    g.save();
    g.globalAlpha = preview ? 0.62 : 0.48;
    g.lineWidth = Math.max(0.55, lineBase * 0.26);
    g.beginPath();
    g.moveTo(x - size * 0.78, y);
    g.lineTo(x + size * 0.78, y);
    g.moveTo(x, y - size * 0.78);
    g.lineTo(x, y + size * 0.78);
    g.stroke();
    g.restore();
  };
  const contourRibs = (points, alpha = preview ? 0.58 : 0.42) => {
    g.save();
    g.globalAlpha = alpha;
    g.lineWidth = Math.max(0.7, lineBase * 0.34);
    for (const path of points) {
      g.beginPath();
      path.forEach(([x, y], index) => {
        if (index === 0) g.moveTo(x, y);
        else if (index === 1) g.lineTo(x, y);
        else g.quadraticCurveTo(path[index - 1][0], path[index - 1][1], x, y);
      });
      g.stroke();
    }
    g.restore();
  };
  const crackField = (marks, alpha = preview ? 0.72 : 0.52) => {
    g.save();
    g.globalAlpha = alpha;
    g.strokeStyle = "rgba(255,238,107,0.78)";
    g.lineWidth = Math.max(0.62, lineBase * 0.3);
    for (const mark of marks) {
      g.beginPath();
      mark.forEach(([x, y], index) => {
        if (index === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      });
      g.stroke();
    }
    g.restore();
  };
  const energyStream = (x, top, bottom, width, strands = 4) => {
    greenTrace(() => {
      g.beginPath();
      g.moveTo(x - width * 0.5, top);
      g.bezierCurveTo(x - width * 0.9, top + (bottom - top) * 0.32, x - width * 0.58, top + (bottom - top) * 0.7, x - width * 0.28, bottom);
      g.lineTo(x + width * 0.18, bottom + r * 0.12);
      g.bezierCurveTo(x + width * 0.72, top + (bottom - top) * 0.62, x + width * 0.72, top + (bottom - top) * 0.22, x + width * 0.35, top);
      g.closePath();
      g.fill();
      g.stroke();
      for (let i = 0; i < strands; i += 1) {
        const offset = (-0.34 + i * (0.68 / Math.max(1, strands - 1))) * width;
        g.beginPath();
        g.moveTo(x + offset, top + r * 0.12);
        g.bezierCurveTo(x + offset * 0.35, top + (bottom - top) * 0.36, x + offset * 0.6, top + (bottom - top) * 0.7, x + offset * 0.2, bottom + r * 0.22);
        g.stroke();
      }
    }, preview ? 0.95 : 0.84, 11);
  };
  const scytheLeg = (side, rootX, rootY, kneeX, kneeY, tipX, tipY, hook = 0.16) => {
    blade([
      [side * rootX, rootY],
      [side * kneeX, kneeY],
      [side * tipX, tipY],
      [side * (tipX - r * hook), tipY - r * 0.42],
      [side * (kneeX - r * 0.08), kneeY - r * 0.08]
    ]);
    joint(side * kneeX, kneeY, r * 0.045);
  };

  g.save();
  g.lineCap = "round";
  g.lineJoin = "round";
  g.strokeStyle = ink;
  g.fillStyle = "rgba(19,24,3,0.9)";
  g.lineWidth = Math.max(1.45, lineBase);
  g.shadowColor = ink;
  g.shadowBlur = glow;
  g.save();
  g.globalAlpha = preview ? 0.2 : 0.16;
  g.fillStyle = "rgba(255,207,90,0.22)";
  g.beginPath();
  g.ellipse(0, r * 0.72, r * 1.9, r * 0.4, 0, 0, Math.PI * 2);
  g.fill();
  g.restore();

  if (type === "hive") {
    const bob = Math.sin(phase) * r * 0.03;
    for (const [side, y, reach, drop] of [
      [-1, -0.46, 1.48, 1.04],
      [1, -0.46, 1.48, 1.04],
      [-1, -0.12, 1.76, 1.2],
      [1, -0.12, 1.76, 1.2],
      [-1, 0.28, 1.56, 1.26],
      [1, 0.28, 1.56, 1.26],
      [-1, 0.54, 1.24, 1.1],
      [1, 0.54, 1.24, 1.1]
    ]) {
      blade([
        [side * r * 0.3, r * y + bob],
        [side * r * 0.72, r * (y + 0.22)],
        [side * r * reach, r * drop],
        [side * r * (reach - 0.12), r * (drop - 0.5)],
        [side * r * 0.82, r * (y + 0.12)]
      ]);
    }
    g.beginPath();
    g.ellipse(r * 0.34, -r * 0.4 + bob, r * 0.94, r * 1.02, 0.05, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    shellGrid(r * 0.34, -r * 0.4 + bob, r * 0.86, r * 0.9, 5, 3);
    contourRibs([
      [[-r * 0.18, -r * 0.94 + bob], [r * 0.1, -r * 1.08 + bob], [r * 0.52, -r * 1.03 + bob], [r * 0.9, -r * 0.72 + bob]],
      [[-r * 0.08, -r * 0.58 + bob], [r * 0.24, -r * 0.72 + bob], [r * 0.72, -r * 0.56 + bob], [r * 1.08, -r * 0.26 + bob]],
      [[r * 0.0, -r * 0.16 + bob], [r * 0.36, -r * 0.24 + bob], [r * 0.78, -r * 0.02 + bob], [r * 0.98, r * 0.22 + bob]]
    ], preview ? 0.62 : 0.46);
    g.beginPath();
    g.ellipse(-r * 0.4, -r * 0.05 + bob, r * 0.62, r * 0.52, -0.12, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.beginPath();
    g.ellipse(-r * 0.9, r * 0.02 + bob, r * 0.38, r * 0.35, 0.12, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.beginPath();
    g.moveTo(-r * 1.1, r * 0.22 + bob);
    g.lineTo(-r * 1.34, r * 0.82);
    g.lineTo(-r * 1.0, r * 0.4);
    g.moveTo(-r * 0.72, r * 0.24 + bob);
    g.lineTo(-r * 0.66, r * 0.94);
    g.lineTo(-r * 0.48, r * 0.34);
    g.stroke();
    g.beginPath();
    g.moveTo(-r * 0.86, -r * 0.36 + bob);
    g.quadraticCurveTo(-r * 0.52, -r * 0.6 + bob, -r * 0.18, -r * 0.34 + bob);
    g.moveTo(-r * 1.02, r * 0.02 + bob);
    g.quadraticCurveTo(-r * 0.72, r * 0.22 + bob, -r * 0.38, r * 0.1 + bob);
    g.moveTo(-r * 0.72, -r * 0.2 + bob);
    g.lineTo(-r * 0.38, r * 0.16 + bob);
    g.stroke();
    for (const [x, y] of [[-0.95, 0.0], [-0.76, -0.08], [-0.54, -0.04]]) {
      joint(r * x, r * y + bob, r * 0.04);
    }
    for (const side of [-1, 1]) {
      joint(side * r * 0.72, -r * 0.24 + bob, r * 0.04);
      joint(side * r * 0.86, r * 0.14 + bob, r * 0.04);
      joint(side * r * 0.78, r * 0.48 + bob, r * 0.035);
    }
    greenTrace(() => {
      for (const [x, y, rr] of [[0.3, -0.65, 0.17], [0.58, -0.2, 0.13], [0.02, -0.18, 0.14], [0.72, -0.62, 0.08], [-0.78, -0.08, 0.06], [-0.5, 0.04, 0.05]]) {
        dot(r * x, r * y + bob, r * rr, rr > 0.1);
      }
    }, preview ? 0.9 : 0.78, 7);
    scratch([
      [r * 0.02, -r * 1.1, r * 0.36, -r * 0.94],
      [r * 0.55, -r * 0.92, r * 0.84, -r * 0.72],
      [r * 0.02, -r * 0.02, r * 0.42, r * 0.12],
      [-r * 0.88, -r * 0.22, -r * 0.56, -r * 0.36],
      [-r * 0.52, r * 0.22, -r * 0.22, r * 0.1]
    ]);
    grain([
      [-r * 0.92, -r * 0.18, r * 0.28, -0.2],
      [-r * 0.42, -r * 0.42, r * 0.24, 0.12],
      [r * 0.2, -r * 0.82, r * 0.34, -0.28],
      [r * 0.62, -r * 0.46, r * 0.28, 0.2],
      [r * 0.58, r * 0.1, r * 0.24, -0.12],
      [-r * 1.16, r * 0.76, r * 0.28, -0.72],
      [r * 1.18, r * 0.74, r * 0.28, 0.72]
    ]);
    rivetBand([[-r * 0.8, r * 0.18], [-r * 0.58, r * 0.26], [-r * 0.32, r * 0.22], [r * 0.05, r * 0.28], [r * 0.5, r * 0.25]], r * 0.032);
  } else if (type === "conduit") {
    const segments = [
      [-r * 0.82, -r * 0.04, r * 0.58, r * 0.58],
      [-r * 0.22, -r * 0.08, r * 0.72, r * 0.62],
      [r * 0.42, -r * 0.04, r * 0.72, r * 0.56],
      [r * 1.02, r * 0.02, r * 0.66, r * 0.48],
      [r * 1.52, r * 0.02, r * 0.46, r * 0.36]
    ];
    for (const [cx, cy, rx, ry] of segments) {
      g.beginPath();
      g.ellipse(cx, cy, rx, ry, 0.08, 0, Math.PI * 2);
      g.fill();
      g.stroke();
      hatchEllipse(cx, cy, rx * 0.75, ry * 0.84, 5, -0.2);
      contourRibs([
        [[cx - rx * 0.64, cy - ry * 0.24], [cx - rx * 0.18, cy - ry * 0.38], [cx + rx * 0.48, cy - ry * 0.22]],
        [[cx - rx * 0.52, cy + ry * 0.2], [cx + rx * 0.02, cy + ry * 0.34], [cx + rx * 0.58, cy + ry * 0.14]]
      ], preview ? 0.46 : 0.34);
    }
    for (let i = 0; i < segments.length - 1; i += 1) {
      const [cx, cy, rx, ry] = segments[i];
      brokenArc(cx + rx * 0.55, cy, ry * 0.95, -Math.PI * 0.5, Math.PI * 0.5, 6);
    }
    g.beginPath();
    g.arc(-r * 1.34, -r * 0.03, r * 0.5, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.beginPath();
    g.arc(-r * 1.34, -r * 0.03, r * 0.34, 0, Math.PI * 2);
    g.stroke();
    for (let i = 0; i < 16; i += 1) {
      const a = i * Math.PI * 2 / 16;
      g.beginPath();
      g.moveTo(-r * 1.34 + Math.cos(a) * r * 0.24, -r * 0.03 + Math.sin(a) * r * 0.24);
      g.lineTo(-r * 1.34 + Math.cos(a) * r * 0.42, -r * 0.03 + Math.sin(a) * r * 0.42);
      g.stroke();
    }
    g.beginPath();
    g.arc(-r * 1.34, -r * 0.03, r * 0.18, 0, Math.PI * 2);
    g.stroke();
    for (const [x, y] of [[-0.68, -0.1], [-0.04, -0.12], [0.58, -0.07], [1.1, 0]]) {
      joint(r * x, r * y, r * 0.045);
    }
    for (const [x, h] of [[-0.38, 0.76], [0.22, 0.86], [0.82, 0.72], [1.28, 0.54]]) {
      blade([
        [r * x, -r * 0.55],
        [r * (x + 0.18), -r * h],
        [r * (x + 0.33), -r * 1.38],
        [r * (x + 0.14), -r * 0.92]
      ], false);
    }
    for (const [x, drop, foot] of [[-0.76, 0.5, 1.0], [-0.28, 0.55, 1.1], [0.32, 0.52, 1.02], [0.82, 0.48, 0.9]]) {
      blade([
        [r * x, r * 0.42],
        [r * (x - 0.14), r * 0.78],
        [r * (x - 0.38), r * foot],
        [r * (x - 0.08), r * (drop + 0.34)]
      ]);
    }
    for (const [x, y, tip] of [[-0.9, 0.46, -1.12], [-0.36, 0.52, -0.52], [0.22, 0.5, 0.02], [0.78, 0.42, 0.62]]) {
      g.beginPath();
      g.moveTo(r * x, r * y);
      g.quadraticCurveTo(r * (x - 0.08), r * 0.8, r * tip, r * 1.02);
      g.stroke();
      joint(r * x, r * y, r * 0.035);
    }
    g.beginPath();
    g.moveTo(r * 1.62, -r * 0.32);
    g.quadraticCurveTo(r * 2.1, -r * 0.44, r * 2.18, -r * 0.92);
    g.lineTo(r * 2.34, -r * 0.24);
    g.quadraticCurveTo(r * 2.08, r * 0.38, r * 1.64, r * 0.32);
    g.stroke();
    greenTrace(() => {
      for (const [x, y, rr] of [[-0.18, -0.06, 0.14], [0.5, -0.03, 0.12], [1.03, -0.02, 0.1], [-1.34, -0.03, 0.15]]) {
        dot(r * x, r * y, r * rr, rr > 0.12);
      }
    }, preview ? 0.88 : 0.74, 7);
    scratch([
      [-r * 0.65, -r * 0.54, -r * 0.32, -r * 0.42],
      [-r * 0.08, r * 0.34, r * 0.28, r * 0.42],
      [r * 0.55, -r * 0.45, r * 0.98, -r * 0.28],
      [r * 1.22, r * 0.24, r * 1.52, r * 0.18],
      [-r * 1.52, r * 0.1, -r * 1.2, r * 0.2]
    ]);
    grain([
      [-r * 1.22, -r * 0.32, r * 0.22, 0.12],
      [-r * 0.58, r * 0.22, r * 0.24, -0.18],
      [r * 0.14, -r * 0.4, r * 0.3, 0.22],
      [r * 0.72, r * 0.31, r * 0.25, -0.2],
      [r * 1.28, -r * 0.26, r * 0.23, 0.2],
      [r * 1.84, -r * 0.24, r * 0.24, -0.54]
    ]);
    cable([[-r * 0.92, r * 0.22], [-r * 0.24, r * 0.46], [r * 0.5, r * 0.34], [r * 1.1, r * 0.16]]);
  } else if (type === "colossus") {
    g.save();
    g.globalAlpha = preview ? 0.74 : 0.56;
    g.lineWidth = Math.max(1.1, lineBase * 0.62);
    for (const side of [-1, 1]) {
      g.beginPath();
      g.moveTo(side * r * 0.88, r * 0.44);
      g.bezierCurveTo(side * r * 1.22, r * 0.58, side * r * 1.52, r * 0.84, side * r * 1.76, r * 1.36);
      g.moveTo(side * r * 0.72, r * 0.78);
      g.bezierCurveTo(side * r * 1.02, r * 0.88, side * r * 1.16, r * 1.1, side * r * 1.18, r * 1.48);
      g.moveTo(side * r * 1.04, r * 0.62);
      g.bezierCurveTo(side * r * 1.48, r * 0.4, side * r * 1.76, r * 0.66, side * r * 1.92, r * 1.22);
      g.stroke();
      joint(side * r * 1.04, r * 0.84, r * 0.045);
    }
    g.restore();

    g.beginPath();
    g.moveTo(-r * 1.32, r * 1.32);
    g.lineTo(-r * 1.18, r * 0.78);
    g.bezierCurveTo(-r * 1.14, r * 0.12, -r * 0.92, -r * 0.78, -r * 0.34, -r * 1.34);
    g.quadraticCurveTo(0, -r * 1.68, r * 0.36, -r * 1.32);
    g.bezierCurveTo(r * 0.94, -r * 0.74, r * 1.18, r * 0.12, r * 1.28, r * 0.78);
    g.lineTo(r * 1.38, r * 1.32);
    g.lineTo(r * 0.74, r * 1.32);
    g.bezierCurveTo(r * 0.66, r * 0.52, r * 0.42, -r * 0.28, r * 0.08, -r * 0.72);
    g.quadraticCurveTo(0, -r * 0.86, -r * 0.1, -r * 0.72);
    g.bezierCurveTo(-r * 0.44, -r * 0.26, -r * 0.62, r * 0.54, -r * 0.68, r * 1.32);
    g.closePath();
    g.fill();
    g.stroke();

    g.beginPath();
    g.moveTo(-r * 1.2, r * 1.18);
    g.lineTo(-r * 1.38, r * 0.82);
    g.lineTo(-r * 1.18, r * 0.4);
    g.lineTo(-r * 1.34, r * 0.12);
    g.lineTo(-r * 0.98, -r * 0.18);
    g.lineTo(-r * 0.86, -r * 0.54);
    g.lineTo(-r * 0.56, -r * 0.78);
    g.lineTo(-r * 0.42, -r * 1.04);
    g.lineTo(-r * 0.08, -r * 1.46);
    g.lineTo(r * 0.24, -r * 1.38);
    g.lineTo(r * 0.5, -r * 1.04);
    g.lineTo(r * 0.66, -r * 0.78);
    g.lineTo(r * 0.96, -r * 0.5);
    g.lineTo(r * 1.1, -r * 0.16);
    g.lineTo(r * 1.34, r * 0.2);
    g.lineTo(r * 1.18, r * 0.52);
    g.lineTo(r * 1.42, r * 0.86);
    g.lineTo(r * 1.22, r * 1.18);
    g.stroke();

    plate([
      [-r * 1.2, r * 0.94],
      [-r * 0.98, r * 0.18],
      [-r * 0.68, -r * 0.52],
      [-r * 0.24, -r * 0.82],
      [-r * 0.56, r * 1.18]
    ], 0.08);
    plate([
      [r * 1.18, r * 0.92],
      [r * 0.96, r * 0.16],
      [r * 0.66, -r * 0.48],
      [r * 0.22, -r * 0.8],
      [r * 0.52, r * 1.18]
    ], 0.08);
    plate([
      [-r * 1.36, r * 0.96],
      [-r * 1.16, r * 0.36],
      [-r * 0.94, -r * 0.2],
      [-r * 0.64, -r * 0.56],
      [-r * 0.8, r * 1.2]
    ], 0.055);
    plate([
      [r * 1.36, r * 0.96],
      [r * 1.14, r * 0.34],
      [r * 0.92, -r * 0.18],
      [r * 0.62, -r * 0.54],
      [r * 0.78, r * 1.2]
    ], 0.055);
    plate([
      [-r * 0.46, -r * 0.96],
      [-r * 0.12, -r * 1.38],
      [r * 0.2, -r * 1.34],
      [r * 0.5, -r * 0.92],
      [r * 0.16, -r * 0.62],
      [-r * 0.16, -r * 0.62]
    ], 0.07);
    plate([
      [-r * 0.28, -r * 0.98],
      [0, -r * 1.24],
      [r * 0.28, -r * 0.98],
      [r * 0.18, r * 1.18],
      [0, r * 1.34],
      [-r * 0.18, r * 1.18]
    ], 0.035);

    g.beginPath();
    g.moveTo(-r * 0.84, r * 1.26);
    g.bezierCurveTo(-r * 1.1, r * 0.78, -r * 1.38, r * 0.76, -r * 1.64, r * 1.3);
    g.moveTo(r * 0.84, r * 1.26);
    g.bezierCurveTo(r * 1.12, r * 0.78, r * 1.42, r * 0.78, r * 1.68, r * 1.3);
    g.moveTo(-r * 0.54, r * 1.28);
    g.bezierCurveTo(-r * 0.64, r * 0.88, -r * 0.78, r * 0.72, -r * 1.0, r * 0.72);
    g.moveTo(r * 0.54, r * 1.28);
    g.bezierCurveTo(r * 0.66, r * 0.88, r * 0.8, r * 0.72, r * 1.02, r * 0.72);
    g.stroke();

    contourRibs([
      [[-r * 0.76, -r * 0.76], [-r * 0.42, -r * 1.12], [0, -r * 1.24], [r * 0.44, -r * 1.1], [r * 0.78, -r * 0.72]],
      [[-r * 0.92, -r * 0.28], [-r * 0.5, -r * 0.54], [-r * 0.14, -r * 0.42]],
      [[r * 0.92, -r * 0.24], [r * 0.5, -r * 0.52], [r * 0.14, -r * 0.4]],
      [[-r * 1.02, r * 0.18], [-r * 0.72, r * 0.0], [-r * 0.36, r * 0.12]],
      [[r * 1.04, r * 0.16], [r * 0.72, -r * 0.02], [r * 0.36, r * 0.1]],
      [[-r * 0.98, r * 0.66], [-r * 0.62, r * 0.5], [-r * 0.32, r * 0.62]],
      [[r * 0.98, r * 0.66], [r * 0.62, r * 0.5], [r * 0.32, r * 0.62]],
      [[-r * 0.32, -r * 0.84], [-r * 0.48, -r * 0.08], [-r * 0.38, r * 0.96]],
      [[r * 0.32, -r * 0.82], [r * 0.48, -r * 0.06], [r * 0.38, r * 0.96]]
    ], preview ? 0.66 : 0.5);

    greenTrace(() => {
      dot(-r * 0.78, r * 0.05, r * 0.12, true);
      dot(-r * 0.58, r * 0.08, r * 0.07, false);
      dot(r * 0.72, r * 0.02, r * 0.11, true);
      dot(r * 0.94, r * 0.06, r * 0.07, false);
      g.beginPath();
      g.arc(0, -r * 0.84, r * 0.14, Math.PI * 0.12, Math.PI * 0.88);
      g.stroke();
    }, preview ? 0.88 : 0.76, 8);
    energyStream(0, -r * 0.86, r * 1.2, r * 0.28, 6);
    greenTrace(() => {
      for (const side of [-1, 1]) {
        g.beginPath();
        g.moveTo(side * r * 0.2, -r * 0.72);
        g.bezierCurveTo(side * r * 0.36, -r * 0.18, side * r * 0.32, r * 0.62, side * r * 0.16, r * 1.16);
        g.stroke();
      }
    }, preview ? 0.58 : 0.48, 5);
    greenTrace(() => {
      g.beginPath();
      g.ellipse(0, r * 1.34, r * 0.46, r * 0.12, 0, 0, Math.PI * 2);
      g.fill();
      g.stroke();
      for (const x of [-0.24, 0, 0.24]) {
        g.beginPath();
        g.moveTo(r * x, r * 1.22);
        g.lineTo(r * (x * 0.55), r * 1.52);
        g.stroke();
      }
    }, preview ? 0.84 : 0.72, 9);

    crackField([
      [[-r * 0.7, -r * 1.0], [-r * 0.52, -r * 0.82], [-r * 0.62, -r * 0.62]],
      [[r * 0.42, -r * 1.04], [r * 0.24, -r * 0.82], [r * 0.34, -r * 0.64]],
      [[-r * 1.04, -r * 0.1], [-r * 0.78, -r * 0.02], [-r * 0.86, r * 0.18]],
      [[r * 1.04, -r * 0.02], [r * 0.8, r * 0.1], [r * 0.9, r * 0.3]],
      [[-r * 0.84, r * 0.56], [-r * 0.58, r * 0.72], [-r * 0.64, r * 0.98]],
      [[r * 0.82, r * 0.54], [r * 0.54, r * 0.72], [r * 0.6, r * 0.98]]
    ], preview ? 0.78 : 0.58);
    grain([
      [-r * 0.46, -r * 1.2, r * 0.3, -0.26],
      [r * 0.08, -r * 1.18, r * 0.34, 0.16],
      [r * 0.6, -r * 0.62, r * 0.28, -0.1],
      [-r * 0.88, r * 0.2, r * 0.24, 0.22],
      [r * 0.84, r * 0.28, r * 0.24, -0.22],
      [-r * 0.42, r * 0.86, r * 0.3, -0.08],
      [r * 0.28, r * 0.96, r * 0.28, 0.12],
      [-r * 1.34, r * 1.08, r * 0.2, -0.58],
      [r * 1.36, r * 1.08, r * 0.2, 0.58]
    ], preview ? 0.42 : 0.3);
  } else {
    for (const side of [-1, 1]) {
      scytheLeg(side, r * 0.34, -r * 0.1, r * 0.9, -r * 0.78, r * 1.42, -r * 1.28, 0.28);
      scytheLeg(side, r * 0.42, r * 0.16, r * 1.18, r * 0.52, r * 1.7, r * 1.22, 0.24);
      scytheLeg(side, r * 0.22, r * 0.72, r * 0.42, r * 1.08, r * 0.34, r * 1.56, 0.08);
    }
    plate([
      [0, -r * 1.7],
      [r * 0.58, -r * 0.5],
      [r * 0.34, r * 0.26],
      [r * 0.18, r * 1.1],
      [0, r * 1.32],
      [-r * 0.18, r * 1.1],
      [-r * 0.34, r * 0.26],
      [-r * 0.58, -r * 0.5]
    ], 0.11);
    plate([
      [0, -r * 1.7],
      [r * 0.28, -r * 0.56],
      [0, -r * 0.28],
      [-r * 0.28, -r * 0.56]
    ], 0.06);
    plate([
      [-r * 0.34, r * 0.3],
      [0, r * 0.64],
      [r * 0.34, r * 0.3],
      [r * 0.2, r * 0.94],
      [0, r * 1.18],
      [-r * 0.2, r * 0.94]
    ], 0.06);
    contourRibs([
      [[0, -r * 1.48], [-r * 0.18, -r * 1.02], [-r * 0.4, -r * 0.54]],
      [[0, -r * 1.48], [r * 0.18, -r * 1.02], [r * 0.4, -r * 0.54]],
      [[-r * 0.46, -r * 0.46], [-r * 0.12, -r * 0.18], [r * 0.46, -r * 0.46]],
      [[-r * 0.28, r * 0.18], [0, r * 0.42], [r * 0.28, r * 0.18]],
      [[-r * 0.22, r * 0.72], [0, r * 0.92], [r * 0.22, r * 0.72]]
    ], preview ? 0.68 : 0.5);
    g.beginPath();
    g.moveTo(0, -r * 1.62);
    g.lineTo(0, r * 1.18);
    g.moveTo(-r * 0.54, -r * 0.48);
    g.lineTo(r * 0.54, -r * 0.48);
    g.moveTo(-r * 0.34, r * 0.28);
    g.lineTo(r * 0.34, r * 0.28);
    g.moveTo(-r * 0.18, r * 0.84);
    g.lineTo(r * 0.18, r * 0.84);
    g.stroke();
    greenTrace(() => {
      dot(0, -r * 0.34, r * 0.13, true);
      dot(0, r * 0.72, r * 0.12, true);
      g.beginPath();
      g.moveTo(-r * 0.14, r * 0.0);
      g.lineTo(0, r * 0.58);
      g.lineTo(r * 0.14, r * 0.0);
      g.stroke();
      for (let y = 0.02; y <= 0.52; y += 0.16) {
        g.beginPath();
        g.moveTo(-r * 0.18, r * y);
        g.lineTo(r * 0.18, r * y);
        g.stroke();
      }
    }, preview ? 0.92 : 0.8, 8);
    for (const side of [-1, 1]) {
      cable([[side * r * 0.34, -r * 0.1], [side * r * 0.48, r * 0.26], [side * r * 0.42, r * 0.74]], preview ? 0.72 : 0.56);
      joint(side * r * 0.34, -r * 0.1, r * 0.04);
      joint(side * r * 0.42, r * 0.16, r * 0.04);
      joint(side * r * 0.22, r * 0.72, r * 0.035);
    }
    crackField([
      [[-r * 0.18, -r * 1.18], [r * 0.06, -r * 0.98], [-r * 0.08, -r * 0.72]],
      [[r * 0.28, -r * 0.7], [r * 0.08, -r * 0.54], [r * 0.18, -r * 0.34]],
      [[-r * 0.2, r * 0.1], [r * 0.08, -r * 0.02], [r * 0.24, r * 0.1]],
      [[-r * 0.22, r * 0.62], [r * 0.06, r * 0.48], [r * 0.18, r * 0.62]],
      [[-r * 1.02, -r * 0.68], [-r * 0.78, -r * 0.46], [-r * 0.86, -r * 0.22]],
      [[r * 1.02, r * 0.54], [r * 1.28, r * 0.74], [r * 1.16, r * 0.96]]
    ], preview ? 0.7 : 0.5);
    grain([
      [0, -r * 1.3, r * 0.34, 0.32],
      [r * 0.32, -r * 0.76, r * 0.26, -0.2],
      [-r * 0.22, r * 0.22, r * 0.24, 0.14],
      [r * 0.22, r * 0.72, r * 0.22, -0.18],
      [-r * 1.16, -r * 0.72, r * 0.24, -0.62],
      [r * 1.28, -r * 0.56, r * 0.24, 0.62],
      [-r * 1.28, r * 0.88, r * 0.22, 0.72],
      [r * 1.38, r * 0.88, r * 0.22, -0.72]
    ], preview ? 0.4 : 0.28);
  }

  g.save();
  g.shadowBlur = 0;
  g.globalAlpha = preview ? 0.34 : 0.22;
  g.strokeStyle = "rgba(255,238,107,0.65)";
  g.lineWidth = Math.max(0.55, lineBase * 0.24);
  for (let i = -2; i <= 2; i += 1) {
    g.beginPath();
    g.moveTo(-r * 1.52, r * (i * 0.22));
    g.lineTo(-r * 1.38, r * (i * 0.22 + 0.02));
    g.moveTo(r * 1.52, r * (i * 0.22));
    g.lineTo(r * 1.38, r * (i * 0.22 + 0.02));
    g.stroke();
  }
  g.restore();
  g.restore();
}
