// ============ DRAW PLAYER ============
function drawPlayer() {
  const p = game.player;
  const s = worldToScreen(p.x, p.y);

  // Blink when iframes
  if (p.iframes > 0 && Math.sin(p.iframes * 30) > 0) return;

  const bob = Math.sin(p.walkAnim) * 2;

  ctx.save();
  ctx.translate(s.x, s.y + bob);

  // Jetpack trail when moving
  if (Math.abs(p.vx) > 10 || Math.abs(p.vy) > 10) {
    ctx.fillStyle = 'rgba(255,107,53,0.4)';
    const trailA = Math.atan2(-p.vy, -p.vx);
    for (let i = 0; i < 3; i++) {
      const off = rand(-4, 4);
      ctx.beginPath();
      ctx.arc(
        Math.cos(trailA) * (14 + i * 6) + off,
        Math.sin(trailA) * (14 + i * 6) + off,
        rand(2, 5), 0, Math.PI * 2
      );
      ctx.fill();
    }
  }

  // Body
  ctx.fillStyle = PAL.playerSuit;
  ctx.strokeStyle = '#636e72';
  ctx.lineWidth = 2;
  roundRect(ctx, -8, -2, 16, 16, 4);
  ctx.fill(); ctx.stroke();

  // Backpack
  ctx.fillStyle = '#636e72';
  ctx.fillRect(-p.facing * 8 - 3, 0, 6, 12);

  // Helmet
  ctx.fillStyle = PAL.playerBody;
  ctx.strokeStyle = '#d35400';
  ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.arc(0, -6, 12, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

  // Visor
  ctx.fillStyle = PAL.playerVisor;
  ctx.beginPath(); ctx.arc(p.facing * 3, -6, 7, 0, Math.PI * 2); ctx.fill();
  // Visor shine
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.beginPath(); ctx.arc(p.facing * 5, -9, 2.5, 0, Math.PI * 2); ctx.fill();

  // Feet
  const footOffset = Math.sin(p.walkAnim) * 3;
  ctx.fillStyle = '#2d3436';
  ctx.beginPath(); ctx.arc(-4, 16 + footOffset, 4, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(4, 16 - footOffset, 4, 0, Math.PI * 2); ctx.fill();

  ctx.restore();
}

// ============ DRAW ENEMY ============
function drawEnemy(e) {
  const s = worldToScreen(e.x, e.y);
  if (s.x < -60 || s.x > canvas.width + 60 || s.y < -60 || s.y > canvas.height + 60) return;

  const wobble = Math.sin(e.anim) * 0.08;
  const squash = 1 + wobble;
  const stretch = 1 - wobble;

  ctx.save();
  ctx.translate(s.x, s.y);
  ctx.scale(squash, stretch);

  // Flash white on damage
  const baseColor = e.flashTimer > 0 ? '#fff' : e.color;
  const outlineColor = e.flashTimer > 0 ? '#ddd' : e.outline;

  // Slow indicator
  if (e.slowTimer > 0) {
    ctx.fillStyle = 'rgba(100,200,255,0.2)';
    ctx.beginPath(); ctx.arc(0, 0, e.radius + 6, 0, Math.PI * 2); ctx.fill();
  }

  // Body
  ctx.fillStyle = baseColor;
  ctx.strokeStyle = outlineColor;
  ctx.lineWidth = 2.5;

  if (e.type === 'spore') {
    // Blobby circle
    ctx.beginPath(); ctx.arc(0, 0, e.radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } else if (e.type === 'beetle') {
    // Oval body
    ctx.beginPath(); ctx.ellipse(0, 0, e.radius, e.radius * 0.75, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // Shell line
    ctx.strokeStyle = outlineColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0, -e.radius * 0.75); ctx.lineTo(0, e.radius * 0.75); ctx.stroke();
    // Antennae
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 2;
    const dir = Math.atan2(game.player.y - e.y, game.player.x - e.x);
    ctx.beginPath(); ctx.moveTo(Math.cos(dir - 0.4) * e.radius * 0.5, Math.sin(dir - 0.4) * e.radius * 0.5);
    ctx.lineTo(Math.cos(dir - 0.3) * e.radius * 1.5, Math.sin(dir - 0.3) * e.radius * 1.5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(Math.cos(dir + 0.4) * e.radius * 0.5, Math.sin(dir + 0.4) * e.radius * 0.5);
    ctx.lineTo(Math.cos(dir + 0.3) * e.radius * 1.5, Math.sin(dir + 0.3) * e.radius * 1.5); ctx.stroke();
  } else if (e.type === 'bloomer') {
    // Petals
    for (let i = 0; i < 6; i++) {
      const pa = (i / 6) * Math.PI * 2 + e.anim * 0.3;
      ctx.fillStyle = e.flashTimer > 0 ? '#fcc' : '#fb7185';
      ctx.beginPath(); ctx.arc(Math.cos(pa) * e.radius * 0.7, Math.sin(pa) * e.radius * 0.7, e.radius * 0.45, 0, Math.PI * 2); ctx.fill();
    }
    // Center
    ctx.fillStyle = baseColor;
    ctx.beginPath(); ctx.arc(0, 0, e.radius * 0.6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } else if (e.type === 'brute') {
    // Chunky square-ish
    roundRect(ctx, -e.radius, -e.radius * 0.8, e.radius * 2, e.radius * 1.6, 8);
    ctx.fill(); ctx.stroke();
    // Brow ridge
    ctx.fillStyle = outlineColor;
    ctx.fillRect(-e.radius * 0.8, -e.radius * 0.8, e.radius * 1.6, e.radius * 0.3);
  } else if (e.type === 'gloop') {
    // Blobby oozing shape
    for (let i = 0; i < 5; i++) {
      const ba = (i / 5) * Math.PI * 2 + e.anim * 0.2;
      const br = e.radius * (0.6 + Math.sin(e.anim * 1.5 + i * 1.3) * 0.15);
      ctx.fillStyle = e.flashTimer > 0 ? '#cfc' : '#84cc16';
      ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.arc(Math.cos(ba) * e.radius * 0.3, Math.sin(ba) * e.radius * 0.3, br, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = baseColor;
    ctx.beginPath(); ctx.arc(0, 0, e.radius * 0.7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } else if (e.type === 'stinger') {
    // Wasp-like pointed body
    ctx.beginPath();
    ctx.moveTo(e.radius * 1.2, 0);
    ctx.lineTo(-e.radius * 0.5, -e.radius * 0.6);
    ctx.lineTo(-e.radius, 0);
    ctx.lineTo(-e.radius * 0.5, e.radius * 0.6);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    // Wings
    ctx.fillStyle = e.flashTimer > 0 ? '#eee' : 'rgba(255,255,200,0.4)';
    const wingFlap = Math.sin(e.anim * 3) * 0.3;
    ctx.beginPath(); ctx.ellipse(0, -e.radius * 0.5, e.radius * 0.7, e.radius * 0.25, wingFlap, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(0, e.radius * 0.5, e.radius * 0.7, e.radius * 0.25, -wingFlap, 0, Math.PI * 2); ctx.fill();
  } else if (e.type === 'queen') {
    // Crown/wings
    for (let i = 0; i < 4; i++) {
      const wa = (i / 4) * Math.PI * 2 + e.anim * 0.5;
      ctx.fillStyle = e.flashTimer > 0 ? '#eee' : '#d946ef';
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.ellipse(Math.cos(wa) * e.radius * 0.5, Math.sin(wa) * e.radius * 0.5, e.radius * 0.8, e.radius * 0.4, wa, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    // Body
    ctx.fillStyle = baseColor;
    ctx.beginPath(); ctx.arc(0, 0, e.radius * 0.7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // Crown spikes
    for (let i = 0; i < 5; i++) {
      const ca = (i / 5) * Math.PI - Math.PI * 0.5;
      ctx.fillStyle = '#fbbf24';
      ctx.beginPath();
      ctx.moveTo(Math.cos(ca - 0.1) * e.radius * 0.65, Math.sin(ca - 0.1) * e.radius * 0.65);
      ctx.lineTo(Math.cos(ca) * e.radius, Math.sin(ca) * e.radius);
      ctx.lineTo(Math.cos(ca + 0.1) * e.radius * 0.65, Math.sin(ca + 0.1) * e.radius * 0.65);
      ctx.fill();
    }
  }

  // Eyes (for non-bloomer, non-queen at small scale)
  if (e.type !== 'bloomer' || e.flashTimer <= 0) {
    const dir = Math.atan2(game.player.y - e.y, game.player.x - e.x);
    const eyeOff = e.radius * 0.3;
    const eyeR = e.radius * (e.boss ? 0.15 : 0.25);
    const pupilR = eyeR * 0.5;
    const lookD = eyeR * 0.3;

    const ex1 = Math.cos(dir - 0.4) * eyeOff;
    const ey1 = Math.sin(dir - 0.4) * eyeOff - (e.type === 'brute' ? e.radius * 0.2 : 0);
    const ex2 = Math.cos(dir + 0.4) * eyeOff;
    const ey2 = Math.sin(dir + 0.4) * eyeOff - (e.type === 'brute' ? e.radius * 0.2 : 0);

    ctx.fillStyle = e.eyeColor;
    ctx.beginPath(); ctx.arc(ex1, ey1, eyeR, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(ex2, ey2, eyeR, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = e.pupilColor;
    ctx.beginPath(); ctx.arc(ex1 + Math.cos(dir) * lookD, ey1 + Math.sin(dir) * lookD, pupilR, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(ex2 + Math.cos(dir) * lookD, ey2 + Math.sin(dir) * lookD, pupilR, 0, Math.PI * 2); ctx.fill();
  }

  ctx.restore();

  // HP bar for bosses or damaged enemies
  if (e.boss || e.hp < e.maxHp) {
    const barW = e.radius * 2;
    const barH = 4;
    const bx = s.x - barW / 2;
    const by = s.y - e.radius - 10;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(bx, by, barW, barH);
    ctx.fillStyle = e.boss ? '#c026d3' : '#e74c3c';
    ctx.fillRect(bx, by, barW * (e.hp / e.maxHp), barH);
  }
}
