// ============ COLONY RENDERING ============
function drawColony() {
  const s = worldToScreen(0, 0);
  const era = getColonyEra();

  // Era transition flash
  if (game.eraTransition > 0) {
    const progress = 1 - (game.eraTransition / 3.0);
    const alpha = (1 - progress) * 0.4;
    ctx.strokeStyle = `rgba(255,215,0,${alpha})`;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(s.x, s.y, 40 + progress * 120, 0, Math.PI * 2); ctx.stroke();
  }

  // Draw central hub
  if (era === 'station') drawHubStation(s);
  else if (era === 'airship') drawHubAirship(s);
  else drawHubFOB(s);

  // Draw peripheral structures
  for (const c of game.colony) {
    if (c.type === 'pad') continue;
    const cs = worldToScreen(c.x, c.y);
    if (cs.x < -80 || cs.x > canvas.width + 80 || cs.y < -80 || cs.y > canvas.height + 80) continue;
    drawColonyStructure(c, cs, era);
  }
}

function drawHubStation(s) {
  const t = game.time;
  const rot = t * 0.1;

  // Core module
  ctx.fillStyle = '#2d3436';
  ctx.strokeStyle = '#556';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(s.x, s.y, 30, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  // Inner ring highlight
  ctx.fillStyle = '#3d4a4f';
  ctx.beginPath(); ctx.arc(s.x, s.y, 18, 0, Math.PI * 2); ctx.fill();

  // Docking ring
  ctx.strokeStyle = '#4ecdc4';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.arc(s.x, s.y, 38, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);

  // 4 radial arms with solar panels
  for (let i = 0; i < 4; i++) {
    const a = rot + i * Math.PI / 2;
    const ax = Math.cos(a), ay = Math.sin(a);
    const armEnd = 35;

    // Arm strut
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(s.x + ax * 22, s.y + ay * 22);
    ctx.lineTo(s.x + ax * (22 + armEnd), s.y + ay * (22 + armEnd));
    ctx.stroke();

    // Solar panel at tip
    const px = s.x + ax * (22 + armEnd);
    const py = s.y + ay * (22 + armEnd);
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(a);
    ctx.fillStyle = '#2d3f6e';
    ctx.strokeStyle = '#556';
    ctx.lineWidth = 1;
    ctx.fillRect(-6, -12, 12, 24);
    ctx.strokeRect(-6, -12, 12, 24);
    // Panel grid lines
    ctx.strokeStyle = '#74b9ff33';
    ctx.beginPath(); ctx.moveTo(0, -12); ctx.lineTo(0, 12); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-6, 0); ctx.lineTo(6, 0); ctx.stroke();
    ctx.restore();

    // Joint blink light
    const jx = s.x + ax * 22;
    const jy = s.y + ay * 22;
    ctx.fillStyle = (Math.sin(t * 4 + i * 1.5) > 0.3) ? 'rgba(78,205,196,0.9)' : 'rgba(78,205,196,0.3)';
    ctx.beginPath(); ctx.arc(jx, jy, 2, 0, Math.PI * 2); ctx.fill();
  }

  // Center glow
  const glowAlpha = 0.4 + Math.sin(t * 3) * 0.2;
  ctx.fillStyle = `rgba(78,205,196,${glowAlpha})`;
  ctx.beginPath(); ctx.arc(s.x, s.y, 6, 0, Math.PI * 2); ctx.fill();

  // Center dot
  ctx.fillStyle = '#4ecdc4';
  ctx.beginPath(); ctx.arc(s.x, s.y, 3, 0, Math.PI * 2); ctx.fill();
}

function drawHubAirship(s) {
  const t = game.time;
  const bob = Math.sin(t * 1.5) * 3;
  const sy = s.y + bob;

  // Balloon envelope
  ctx.fillStyle = '#3d5a80';
  ctx.strokeStyle = '#4a7c9b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(s.x, sy - 20, 48, 24, 0, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  // Envelope highlight
  ctx.fillStyle = 'rgba(100,160,200,0.2)';
  ctx.beginPath();
  ctx.ellipse(s.x, sy - 26, 30, 10, 0, 0, Math.PI * 2);
  ctx.fill();
  // Envelope ribs
  ctx.strokeStyle = 'rgba(74,124,155,0.4)';
  ctx.lineWidth = 1;
  for (let i = -2; i <= 2; i++) {
    ctx.beginPath();
    ctx.ellipse(s.x + i * 12, sy - 20, 3, 22, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Rigging lines
  ctx.strokeStyle = '#6a8a9e';
  ctx.lineWidth = 1;
  const rigX = [-20, -8, 8, 20];
  for (const rx of rigX) {
    ctx.beginPath();
    ctx.moveTo(s.x + rx, sy - 2);
    ctx.lineTo(s.x + rx * 0.8, sy + 10);
    ctx.stroke();
  }

  // Gondola hull
  ctx.fillStyle = '#2d3436';
  ctx.strokeStyle = '#556';
  ctx.lineWidth = 2;
  roundRect(ctx, s.x - 28, sy + 10, 56, 18, 5);
  ctx.fill(); ctx.stroke();
  // Gondola keel
  ctx.fillStyle = '#222';
  ctx.beginPath();
  ctx.moveTo(s.x - 20, sy + 28);
  ctx.lineTo(s.x + 20, sy + 28);
  ctx.lineTo(s.x + 15, sy + 33);
  ctx.lineTo(s.x - 15, sy + 33);
  ctx.closePath();
  ctx.fill();

  // Windows (amber glow)
  const windowY = sy + 16;
  for (let i = -1; i <= 1; i++) {
    ctx.fillStyle = '#f0a500';
    ctx.fillRect(s.x + i * 14 - 4, windowY, 8, 6);
    // Window glow
    ctx.fillStyle = 'rgba(240,165,0,0.15)';
    ctx.beginPath(); ctx.arc(s.x + i * 14, windowY + 3, 8, 0, Math.PI * 2); ctx.fill();
  }

  // Side thrusters
  for (const side of [-1, 1]) {
    const tx = s.x + side * 32;
    const ty = sy + 18;
    // Thruster housing
    ctx.fillStyle = '#444';
    ctx.fillRect(tx - 4, ty - 3, 8, 8);
    // Exhaust glow
    const exAlpha = 0.3 + Math.sin(t * 6 + side) * 0.15;
    ctx.fillStyle = `rgba(78,205,196,${exAlpha})`;
    ctx.beginPath(); ctx.arc(tx + side * 6, ty + 1, 4, 0, Math.PI * 2); ctx.fill();
  }

  // Rear propeller
  const propX = s.x + 36;
  const propY = sy + 19;
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 2;
  const propAngle = t * 12;
  for (let i = 0; i < 3; i++) {
    const pa = propAngle + i * Math.PI * 2 / 3;
    ctx.beginPath();
    ctx.moveTo(propX, propY);
    ctx.lineTo(propX + Math.cos(pa) * 8, propY + Math.sin(pa) * 8);
    ctx.stroke();
  }
  // Prop hub
  ctx.fillStyle = '#666';
  ctx.beginPath(); ctx.arc(propX, propY, 2, 0, Math.PI * 2); ctx.fill();

  // Running lights
  ctx.fillStyle = '#e74c3c'; // port (left)
  ctx.beginPath(); ctx.arc(s.x - 30, sy + 14, 2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#2ecc71'; // starboard (right)
  ctx.beginPath(); ctx.arc(s.x + 30, sy + 14, 2, 0, Math.PI * 2); ctx.fill();

  // Antenna mast
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(s.x, sy - 44); ctx.lineTo(s.x, sy - 54); ctx.stroke();
  ctx.fillStyle = (Math.sin(t * 4) > 0) ? '#e74c3c' : '#c0392b';
  ctx.beginPath(); ctx.arc(s.x, sy - 54, 2, 0, Math.PI * 2); ctx.fill();
}

function drawHubFOB(s) {
  const t = game.time;
  const hw = 50, hh = 38; // half width/height of compound

  // Cleared ground zone
  ctx.fillStyle = 'rgba(60,60,40,0.12)';
  ctx.beginPath(); ctx.arc(s.x, s.y, 65, 0, Math.PI * 2); ctx.fill();

  // Perimeter walls
  ctx.fillStyle = '#2a2a1e';
  ctx.strokeStyle = '#5a5a3e';
  ctx.lineWidth = 3;
  roundRect(ctx, s.x - hw, s.y - hh, hw * 2, hh * 2, 4);
  ctx.fill(); ctx.stroke();

  // Wall detail lines (horizontal planks)
  ctx.strokeStyle = 'rgba(90,90,62,0.3)';
  ctx.lineWidth = 1;
  for (let i = -1; i <= 1; i++) {
    ctx.beginPath();
    ctx.moveTo(s.x - hw + 4, s.y + i * 14);
    ctx.lineTo(s.x + hw - 4, s.y + i * 14);
    ctx.stroke();
  }

  // Entrance gap (south wall)
  ctx.fillStyle = '#1a1a12';
  ctx.fillRect(s.x - 10, s.y + hh - 2, 20, 6);

  // Corner watchtowers
  const corners = [[-1,-1],[1,-1],[1,1],[-1,1]];
  for (let i = 0; i < 4; i++) {
    const cx = s.x + corners[i][0] * (hw - 2);
    const cy = s.y + corners[i][1] * (hh - 2);
    // Tower base
    ctx.fillStyle = '#3a3a2e';
    ctx.strokeStyle = '#5a5a3e';
    ctx.lineWidth = 1.5;
    ctx.fillRect(cx - 5, cy - 5, 10, 10);
    ctx.strokeRect(cx - 5, cy - 5, 10, 10);
    // Sensor light
    const blink = Math.sin(t * 3 + i * 1.2) > 0.2;
    ctx.fillStyle = blink ? '#4ecdc4' : 'rgba(78,205,196,0.3)';
    ctx.beginPath(); ctx.arc(cx, cy, 2, 0, Math.PI * 2); ctx.fill();
  }

  // Central landing pad
  ctx.fillStyle = '#1a1a15';
  ctx.beginPath(); ctx.arc(s.x, s.y, 18, 0, Math.PI * 2); ctx.fill();
  // Crosshair
  ctx.strokeStyle = '#4ecdc4';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.arc(s.x, s.y, 14, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(s.x - 10, s.y); ctx.lineTo(s.x + 10, s.y);
  ctx.moveTo(s.x, s.y - 10); ctx.lineTo(s.x, s.y + 10);
  ctx.stroke();

  // Sandbag barriers (outside walls)
  const sandbags = [
    { x: s.x - hw - 10, y: s.y - 12, w: 8, h: 24 },
    { x: s.x + hw + 2, y: s.y + 5, w: 8, h: 20 },
    { x: s.x - 20, y: s.y - hh - 8, w: 25, h: 6 },
    { x: s.x + 10, y: s.y + hh + 2, w: 20, h: 6 },
  ];
  ctx.fillStyle = '#8b7d5b';
  ctx.strokeStyle = '#6b5d3b';
  ctx.lineWidth = 1;
  for (const sb of sandbags) {
    roundRect(ctx, sb.x, sb.y, sb.w, sb.h, 2);
    ctx.fill(); ctx.stroke();
  }

  // Fuel tanks (right side interior)
  ctx.fillStyle = '#555';
  ctx.strokeStyle = '#666';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(s.x + 30, s.y - 15, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.arc(s.x + 30, s.y - 2, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

  // Antenna mast
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(s.x - 30, s.y - 10); ctx.lineTo(s.x - 30, s.y - 35); ctx.stroke();
  ctx.fillStyle = (Math.sin(t * 4) > 0) ? '#e74c3c' : '#c0392b';
  ctx.beginPath(); ctx.arc(s.x - 30, s.y - 35, 2.5, 0, Math.PI * 2); ctx.fill();
  // Dish
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(s.x - 30, s.y - 30, 6, -Math.PI * 0.7, -Math.PI * 0.3);
  ctx.stroke();

  // FOB label on ground
  ctx.fillStyle = 'rgba(78,205,196,0.15)';
  ctx.font = '8px "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('FOB', s.x, s.y + hh + 16);
}

function drawColonyStructure(c, s, era) {
  ctx.strokeStyle = '#556';
  ctx.lineWidth = 2;

  // Era color tinting
  const windowColor = era === 'fob' ? '#c8a960' : era === 'airship' ? '#a8d8ea' : '#74b9ff';
  const habFill = era === 'fob' ? '#4a4a3a' : '#636e72';
  const blinkColor = era === 'fob' ? '#f0a500' : '#e74c3c';
  const blinkDim = era === 'fob' ? '#c08000' : '#c0392b';

  if (c.type === 'hab') {
    ctx.fillStyle = habFill;
    roundRect(ctx, s.x - 16, s.y - 12, 32, 24, 6);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = windowColor;
    ctx.fillRect(s.x - 5, s.y - 5, 10, 8);
  } else if (c.type === 'antenna') {
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(s.x, s.y - 30); ctx.stroke();
    ctx.fillStyle = (Math.sin(game.time * 4) > 0) ? blinkColor : blinkDim;
    ctx.beginPath(); ctx.arc(s.x, s.y - 30, 3, 0, Math.PI * 2); ctx.fill();
  } else if (c.type === 'solar') {
    ctx.fillStyle = '#2d3f6e';
    ctx.fillRect(s.x - 14, s.y - 10, 28, 20);
    ctx.strokeRect(s.x - 14, s.y - 10, 28, 20);
    ctx.strokeStyle = '#74b9ff33';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(s.x, s.y - 10); ctx.lineTo(s.x, s.y + 10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(s.x - 14, s.y); ctx.lineTo(s.x + 14, s.y); ctx.stroke();
  } else if (c.type === 'dome') {
    ctx.fillStyle = 'rgba(116,185,255,0.2)';
    ctx.strokeStyle = '#74b9ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(s.x, s.y, 20, Math.PI, 0); ctx.lineTo(s.x + 20, s.y); ctx.lineTo(s.x - 20, s.y); ctx.closePath();
    ctx.fill(); ctx.stroke();
  } else if (c.type === 'engine') {
    // Airship-era: small thruster nozzle
    ctx.fillStyle = '#444';
    ctx.strokeStyle = '#556';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(s.x - 6, s.y - 8);
    ctx.lineTo(s.x + 6, s.y - 8);
    ctx.lineTo(s.x + 9, s.y + 8);
    ctx.lineTo(s.x - 9, s.y + 8);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    // Exhaust glow
    const ea = 0.2 + Math.sin(game.time * 5) * 0.15;
    ctx.fillStyle = `rgba(78,205,196,${ea})`;
    ctx.beginPath(); ctx.arc(s.x, s.y + 10, 5, 0, Math.PI * 2); ctx.fill();
  } else if (c.type === 'balloon') {
    // Airship-era: small tethered balloon
    ctx.strokeStyle = '#6a8a9e';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(s.x, s.y - 20); ctx.stroke();
    ctx.fillStyle = '#3d5a80';
    ctx.strokeStyle = '#4a7c9b';
    ctx.beginPath(); ctx.ellipse(s.x, s.y - 26, 8, 6, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } else if (c.type === 'bunker') {
    // FOB-era: squat bunker with slit
    ctx.fillStyle = '#3a3a2e';
    ctx.strokeStyle = '#5a5a3e';
    ctx.lineWidth = 1.5;
    roundRect(ctx, s.x - 14, s.y - 8, 28, 16, 3);
    ctx.fill(); ctx.stroke();
    // Slit window
    ctx.fillStyle = '#1a1a12';
    ctx.fillRect(s.x - 8, s.y - 2, 16, 3);
  } else if (c.type === 'turret_base') {
    // FOB-era: ground turret mount
    ctx.fillStyle = '#4a4a3a';
    ctx.strokeStyle = '#5a5a3e';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(s.x, s.y, 8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // Cross mount
    ctx.strokeStyle = '#6b6b5b';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(s.x - 6, s.y); ctx.lineTo(s.x + 6, s.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(s.x, s.y - 6); ctx.lineTo(s.x, s.y + 6); ctx.stroke();
  }
}
