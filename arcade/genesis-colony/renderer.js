// ============ RENDERING ============
function worldToScreen(wx, wy) {
  return {
    x: (wx - game.camera.x) + canvas.width / 2 + screenShake.x,
    y: (wy - game.camera.y) + canvas.height / 2 + screenShake.y,
  };
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Title screen stars
  if (!game) {
    ctx.fillStyle = PAL.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const tt = performance.now() / 1000;
    for (const star of stars) {
      const twinkle = 0.4 + 0.6 * Math.sin(tt * star.speed + star.twinkle);
      ctx.globalAlpha = twinkle * 0.5;
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(star.x * canvas.width, star.y * canvas.height, star.size, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    return;
  }

  // Background - phase-based rendering
  const wave = game.wave;
  const t = game.time;

  // Smooth transition factor between phases (0-1 over 2 waves)
  const atmosBlend = clamp((wave - 8) / 2, 0, 1);   // 0 at wave 8, 1 at wave 10+
  const groundBlend = clamp((wave - 18) / 2, 0, 1);  // 0 at wave 18, 1 at wave 20+

  if (groundBlend > 0) {
    // Phase 3: Alien forest ground (wave 20+)
    const skyGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    // Lerp from atmosphere to forest sky
    const r1 = lerp(12, 25, groundBlend), g1 = lerp(20, 10, groundBlend), b1 = lerp(60, 45, groundBlend);
    const r2 = lerp(30, 15, groundBlend), g2 = lerp(60, 50, groundBlend), b2 = lerp(80, 35, groundBlend);
    const r3 = lerp(50, 20, groundBlend), g3 = lerp(90, 65, groundBlend), b3 = lerp(70, 30, groundBlend);
    skyGrad.addColorStop(0, `rgb(${r1|0},${g1|0},${b1|0})`);
    skyGrad.addColorStop(0.5, `rgb(${r2|0},${g2|0},${b2|0})`);
    skyGrad.addColorStop(1, `rgb(${r3|0},${g3|0},${b3|0})`);
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Background canopy silhouettes
    if (groundBlend > 0.3) {
      const canopyAlpha = (groundBlend - 0.3) * 0.4;
      ctx.fillStyle = `rgba(8, 30, 15, ${canopyAlpha})`;
      for (let i = 0; i < 12; i++) {
        const bx = ((i * 137 + game.camera.x * 0.01) % (canvas.width + 200)) - 100;
        const bh = 80 + (i * 47 % 60);
        const bw = 60 + (i * 31 % 50);
        ctx.beginPath();
        ctx.ellipse(bx, canvas.height * 0.05, bw, bh, 0, 0, Math.PI * 2);
        ctx.fill();
        // Bottom canopy
        const bx2 = ((i * 113 + 50 + game.camera.x * 0.015) % (canvas.width + 200)) - 100;
        ctx.beginPath();
        ctx.ellipse(bx2, canvas.height * 0.95, bw * 1.2, bh * 0.7, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Faint remaining stars at top (fade out with groundBlend)
    if (groundBlend < 1) {
      const starFade = 1 - groundBlend;
      for (const star of stars) {
        if (star.y > 0.3) continue; // Only top stars
        const sx = ((star.x * canvas.width - game.camera.x * 0.02) % canvas.width + canvas.width) % canvas.width;
        const sy = star.y * canvas.height * 0.3;
        const twinkle = 0.4 + 0.6 * Math.sin(t * star.speed + star.twinkle);
        ctx.globalAlpha = twinkle * 0.3 * starFade;
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(sx, sy, star.size * 0.7, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // Organic ground texture
    ctx.strokeStyle = `rgba(20, 70, 40, ${0.08 + groundBlend * 0.07})`;
    ctx.lineWidth = 1;
    const gGridSize = 60;
    const gOffX = (-game.camera.x % gGridSize + gGridSize) % gGridSize + screenShake.x;
    const gOffY = (-game.camera.y % gGridSize + gGridSize) % gGridSize + screenShake.y;
    // Wavy organic lines instead of straight grid
    for (let x = gOffX - gGridSize; x < canvas.width + gGridSize; x += gGridSize) {
      ctx.beginPath();
      for (let y = 0; y <= canvas.height; y += 20) {
        const wx = x + Math.sin(y * 0.02 + t * 0.3 + x * 0.01) * 8;
        if (y === 0) ctx.moveTo(wx, y); else ctx.lineTo(wx, y);
      }
      ctx.stroke();
    }

  } else if (atmosBlend > 0) {
    // Phase 2: Upper atmosphere (waves 10-19)
    const skyGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    // Lerp from space to atmosphere
    const r1 = lerp(16, 12, atmosBlend), g1 = lerp(32, 20, atmosBlend), b1 = lerp(40, 60, atmosBlend);
    const r2 = lerp(11, 30, atmosBlend), g2 = lerp(16, 60, atmosBlend), b2 = lerp(38, 80, atmosBlend);
    const r3 = lerp(11, 50, atmosBlend), g3 = lerp(16, 90, atmosBlend), b3 = lerp(38, 70, atmosBlend);
    skyGrad.addColorStop(0, `rgb(${r1|0},${g1|0},${b1|0})`);
    skyGrad.addColorStop(0.6, `rgb(${r2|0},${g2|0},${b2|0})`);
    skyGrad.addColorStop(1, `rgb(${r3|0},${g3|0},${b3|0})`);
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Fading stars (visible at top, fade with atmosBlend)
    const starAlpha = 0.7 * (1 - atmosBlend * 0.7);
    for (const star of stars) {
      const sx = ((star.x * canvas.width - game.camera.x * 0.02) % canvas.width + canvas.width) % canvas.width;
      const sy = ((star.y * canvas.height - game.camera.y * 0.02) % canvas.height + canvas.height) % canvas.height;
      const vertFade = 1 - (sy / canvas.height) * atmosBlend; // Stars fade toward bottom
      const twinkle = 0.4 + 0.6 * Math.sin(t * star.speed + star.twinkle);
      ctx.globalAlpha = twinkle * starAlpha * Math.max(0, vertFade);
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(sx, sy, star.size, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Wispy clouds drifting across
    ctx.fillStyle = `rgba(100, 160, 180, ${atmosBlend * 0.08})`;
    for (let i = 0; i < 8; i++) {
      const cx = ((i * 200 + t * (15 + i * 5) + game.camera.x * 0.03) % (canvas.width + 400)) - 200;
      const cy = canvas.height * (0.2 + (i * 0.09));
      const cw = 150 + (i * 37 % 100);
      const ch = 20 + (i * 13 % 15);
      ctx.beginPath();
      ctx.ellipse(cx, cy, cw, ch, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Atmospheric grid (wind-like horizontal lines)
    const gridAlpha = lerp(0.15, 0.06, atmosBlend);
    ctx.strokeStyle = `rgba(30, 80, 70, ${gridAlpha})`;
    ctx.lineWidth = 1;
    const gridSize = 80;
    const offsetX = (-game.camera.x % gridSize + gridSize) % gridSize + screenShake.x;
    const offsetY = (-game.camera.y % gridSize + gridSize) % gridSize + screenShake.y;
    // Vertical lines fade out with atmosphere
    if (atmosBlend < 0.8) {
      ctx.globalAlpha = 1 - atmosBlend;
      for (let x = offsetX - gridSize; x < canvas.width + gridSize; x += gridSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    for (let y = offsetY - gridSize; y < canvas.height + gridSize; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

  } else {
    // Phase 1: Space (waves 1-9)
    const grad = ctx.createRadialGradient(canvas.width / 2, canvas.height / 2, 0, canvas.width / 2, canvas.height / 2, canvas.width * 0.7);
    grad.addColorStop(0, '#102028');
    grad.addColorStop(1, PAL.bg);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Stars
    for (const star of stars) {
      const sx = ((star.x * canvas.width - game.camera.x * 0.02) % canvas.width + canvas.width) % canvas.width;
      const sy = ((star.y * canvas.height - game.camera.y * 0.02) % canvas.height + canvas.height) % canvas.height;
      const twinkle = 0.4 + 0.6 * Math.sin(t * star.speed + star.twinkle);
      ctx.globalAlpha = twinkle * 0.7;
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(sx, sy, star.size, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Planet approaching (grows each wave)
    if (wave >= 1 && wave < 10) {
      const planetRadius = 20 + wave * 12;
      const planetX = canvas.width * 0.5;
      const planetY = -planetRadius * 0.4 + wave * 8; // Rises into view
      // Atmospheric glow
      const glowGrad = ctx.createRadialGradient(planetX, planetY, planetRadius * 0.8, planetX, planetY, planetRadius * 1.6);
      glowGrad.addColorStop(0, 'rgba(26, 188, 156, 0.15)');
      glowGrad.addColorStop(0.5, 'rgba(26, 188, 156, 0.06)');
      glowGrad.addColorStop(1, 'rgba(26, 188, 156, 0)');
      ctx.fillStyle = glowGrad;
      ctx.beginPath(); ctx.arc(planetX, planetY, planetRadius * 1.6, 0, Math.PI * 2); ctx.fill();
      // Planet body
      const pGrad = ctx.createRadialGradient(planetX - planetRadius * 0.3, planetY - planetRadius * 0.3, 0, planetX, planetY, planetRadius);
      pGrad.addColorStop(0, '#1abc9c');
      pGrad.addColorStop(0.5, '#0d7a5f');
      pGrad.addColorStop(1, '#063b2e');
      ctx.fillStyle = pGrad;
      ctx.beginPath(); ctx.arc(planetX, planetY, planetRadius, 0, Math.PI * 2); ctx.fill();
      // Surface detail bands
      ctx.strokeStyle = 'rgba(46, 204, 113, 0.2)';
      ctx.lineWidth = 2;
      for (let b = 0; b < 3; b++) {
        const by = planetY - planetRadius * 0.4 + b * planetRadius * 0.4;
        const bw = Math.sqrt(Math.max(0, planetRadius * planetRadius - (by - planetY) * (by - planetY)));
        if (bw > 5) {
          ctx.beginPath();
          ctx.ellipse(planetX, by, bw, planetRadius * 0.05, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      // Atmosphere rim
      ctx.strokeStyle = 'rgba(26, 188, 156, 0.3)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(planetX, planetY, planetRadius + 3, 0, Math.PI * 2); ctx.stroke();
    }

    // Grid
    ctx.strokeStyle = 'rgba(30, 80, 70, 0.15)';
    ctx.lineWidth = 1;
    const gridSize = 80;
    const offsetX = (-game.camera.x % gridSize + gridSize) % gridSize + screenShake.x;
    const offsetY = (-game.camera.y % gridSize + gridSize) % gridSize + screenShake.y;
    for (let x = offsetX - gridSize; x < canvas.width + gridSize; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = offsetY - gridSize; y < canvas.height + gridSize; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }
  }

  // World decorations (rocks, plants)
  drawWorldDecor();

  // Colony structures
  drawColony();

  // Planetary defenses
  drawPlanetaryDefenses();

  // Weapon effects (shockwave, cryo field)
  drawWeaponEffects();

  // Goop pools
  for (const pool of game.goopPools) {
    const s = worldToScreen(pool.x, pool.y);
    const fadeAlpha = Math.min(1, pool.timer / 3);
    const pulse = 1 + Math.sin(game.time * 2) * 0.05;
    // Outer glow
    const glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, pool.radius * pulse);
    glow.addColorStop(0, `rgba(101, 163, 13, ${0.35 * fadeAlpha * Math.min(pool.intensity * 0.3, 1)})`);
    glow.addColorStop(0.7, `rgba(77, 124, 15, ${0.2 * fadeAlpha * Math.min(pool.intensity * 0.25, 1)})`);
    glow.addColorStop(1, 'rgba(77, 124, 15, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(s.x, s.y, pool.radius * pulse, 0, Math.PI * 2); ctx.fill();
    // Inner bubbles
    for (let b = 0; b < Math.min(pool.intensity, 5); b++) {
      const bx = s.x + Math.sin(game.time * 1.5 + b * 2.1) * pool.radius * 0.4;
      const by = s.y + Math.cos(game.time * 1.2 + b * 1.7) * pool.radius * 0.4;
      ctx.fillStyle = `rgba(132, 204, 22, ${0.3 * fadeAlpha})`;
      ctx.beginPath(); ctx.arc(bx, by, 3 + b, 0, Math.PI * 2); ctx.fill();
    }
  }

  // Gems
  for (const gem of game.gems) {
    const s = worldToScreen(gem.x, gem.y);
    const pulse = 1 + Math.sin(game.time * 6 + gem.x) * 0.15;
    const r = gem.radius * pulse;
    // Outer bloom (subtle)
    const bloom = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, r * 2.5);
    bloom.addColorStop(0, 'rgba(94, 196, 182, 0.22)');
    bloom.addColorStop(0.5, 'rgba(94, 196, 182, 0.06)');
    bloom.addColorStop(1, 'rgba(94, 196, 182, 0)');
    ctx.fillStyle = bloom;
    ctx.beginPath(); ctx.arc(s.x, s.y, r * 2.5, 0, Math.PI * 2); ctx.fill();
    // Diamond shape
    ctx.fillStyle = PAL.gem;
    ctx.strokeStyle = '#3d9486';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(s.x, s.y - r);
    ctx.lineTo(s.x + r * 0.7, s.y);
    ctx.lineTo(s.x, s.y + r);
    ctx.lineTo(s.x - r * 0.7, s.y);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
  }

  // Enemy projectiles
  for (const pr of game.enemyProjectiles) {
    const s = worldToScreen(pr.x, pr.y);
    ctx.fillStyle = pr.color;
    ctx.beginPath(); ctx.arc(s.x, s.y, pr.radius, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(244,63,94,0.3)';
    ctx.beginPath(); ctx.arc(s.x, s.y, pr.radius * 2, 0, Math.PI * 2); ctx.fill();
  }

  // Enemies
  for (const e of game.enemies) {
    drawEnemy(e);
  }

  // Boss railgun telegraph & beam
  for (const e of game.enemies) {
    if (!e.boss) continue;
    const es = worldToScreen(e.x, e.y);
    const beamLen = 900;

    // Telegraph during charge
    if (e.railCharging) {
      const progress = e.railChargeTime / 1.5;
      const pulse = 0.3 + Math.sin(e.railChargeTime * 12) * 0.2;
      const alpha = (0.15 + progress * 0.6) * (0.8 + pulse * 0.2);
      const ex = es.x + Math.cos(e.railAngle) * beamLen;
      const ey = es.y + Math.sin(e.railAngle) * beamLen;

      // Warning line
      ctx.strokeStyle = `rgba(255, 40, 40, ${alpha})`;
      ctx.lineWidth = 2 + progress * 4;
      ctx.setLineDash([8, 12 - progress * 8]);
      ctx.beginPath(); ctx.moveTo(es.x, es.y); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);

      // Charge glow at boss
      const glowR = 10 + progress * 20;
      const glow = ctx.createRadialGradient(es.x, es.y, 0, es.x, es.y, glowR);
      glow.addColorStop(0, `rgba(255, 60, 60, ${0.6 * progress})`);
      glow.addColorStop(1, 'rgba(255, 60, 60, 0)');
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(es.x, es.y, glowR, 0, Math.PI * 2); ctx.fill();
    }

    // Fired beam flash
    if (e.railBeamTimer > 0) {
      const alpha = e.railBeamTimer / 0.2;
      const ex = es.x + Math.cos(e.railAngle) * beamLen;
      const ey = es.y + Math.sin(e.railAngle) * beamLen;

      // Outer glow
      ctx.strokeStyle = `rgba(255, 80, 80, ${alpha * 0.5})`;
      ctx.lineWidth = 16;
      ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(es.x, es.y); ctx.lineTo(ex, ey); ctx.stroke();

      // Core beam
      ctx.strokeStyle = `rgba(255, 220, 220, ${alpha * 0.9})`;
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(es.x, es.y); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.lineCap = 'butt';
    }
  }

  // Player projectiles
  for (const pr of game.projectiles) {
    const s = worldToScreen(pr.x, pr.y);
    if (pr.homing) {
      // Missile shape - pointed nose, fins, exhaust trail
      const a = Math.atan2(pr.vy, pr.vx);
      const r = pr.radius;
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(a);
      // Exhaust glow
      const eg = ctx.createRadialGradient(-r * 2, 0, 0, -r * 2, 0, r * 3);
      eg.addColorStop(0, 'rgba(255,200,50,0.6)');
      eg.addColorStop(1, 'rgba(255,100,50,0)');
      ctx.fillStyle = eg;
      ctx.beginPath(); ctx.arc(-r * 2, 0, r * 3, 0, Math.PI * 2); ctx.fill();
      // Body
      ctx.fillStyle = pr.color;
      ctx.beginPath();
      ctx.moveTo(r * 2.5, 0);           // nose
      ctx.lineTo(-r * 1.2, -r * 0.8);   // top
      ctx.lineTo(-r * 1.8, -r * 1.5);   // top fin
      ctx.lineTo(-r * 1.2, -r * 0.5);
      ctx.lineTo(-r * 1.5, 0);          // tail
      ctx.lineTo(-r * 1.2, r * 0.5);
      ctx.lineTo(-r * 1.8, r * 1.5);    // bottom fin
      ctx.lineTo(-r * 1.2, r * 0.8);
      ctx.closePath();
      ctx.fill();
      // Bright nose tip
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(r * 1.5, 0, r * 0.35, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    } else {
      ctx.fillStyle = pr.color;
      ctx.beginPath(); ctx.arc(s.x, s.y, pr.radius, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = `${pr.color}66`;
      ctx.beginPath(); ctx.arc(s.x, s.y, pr.radius * 2, 0, Math.PI * 2); ctx.fill();
    }
  }

  // Player
  drawPlayer();

  // Drone weapons
  drawDrones();

  // Lightning bolts
  for (const bolt of game.lightningBolts) {
    const alpha = bolt.lifetime / bolt.maxLife;
    const pts = bolt.points;
    // Outer glow
    ctx.strokeStyle = `rgba(165, 243, 252, ${alpha * 0.4})`;
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const s = worldToScreen(pts[i].x, pts[i].y);
      if (i === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
    }
    ctx.stroke();
    // Bright core
    ctx.strokeStyle = `rgba(224, 247, 255, ${alpha * 0.9})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const s = worldToScreen(pts[i].x, pts[i].y);
      if (i === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
    }
    ctx.stroke();
  }

  // Particles
  for (const pt of game.particles) {
    const s = worldToScreen(pt.x, pt.y);
    const alpha = pt.lifetime / pt.maxLife;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = pt.color;
    ctx.beginPath(); ctx.arc(s.x, s.y, pt.size * alpha, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Damage texts
  for (const t of game.dmgTexts) {
    const s = worldToScreen(t.x, t.y);
    const alpha = t.lifetime / 0.8;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = t.color;
    ctx.font = 'bold 16px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(t.text, s.x, s.y);
  }
  ctx.globalAlpha = 1;

  // Virtual joystick indicator on mobile
  if (joystick.active) {
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath(); ctx.arc(joystick.ox, joystick.oy, 60, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.beginPath(); ctx.arc(joystick.ox + joystick.dx, joystick.oy + joystick.dy, 24, 0, Math.PI * 2); ctx.fill();
  }

  // Compass pointing to colony
  drawCompass();

  // Announcement overlay
  if (announcement.timer > 0) {
    const alpha = Math.min(1, announcement.timer, announcement.timer > 1 ? 1 : announcement.timer);
    ctx.globalAlpha = alpha;
    ctx.font = `bold ${Math.min(canvas.width * 0.08, 64)}px "Segoe UI", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    // Shadow
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillText(announcement.text, canvas.width / 2 + 2, canvas.height * 0.35 + 2);
    // Text
    ctx.fillStyle = announcement.color;
    ctx.fillText(announcement.text, canvas.width / 2, canvas.height * 0.35);
    ctx.globalAlpha = 1;
  }
}

function drawWorldDecor() {
  // Rocks
  for (const r of game.worldRocks) {
    const s = worldToScreen(r.x, r.y);
    if (s.x < -50 || s.x > canvas.width + 50 || s.y < -50 || s.y > canvas.height + 50) continue;
    ctx.fillStyle = r.color;
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < r.verts; i++) {
      const a = (i / r.verts) * Math.PI * 2 + r.rotation;
      const rad = r.size * (0.7 + 0.3 * Math.sin(i * 2.5));
      const x = s.x + Math.cos(a) * rad;
      const y = s.y + Math.sin(a) * rad;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }

  // Plants
  for (const p of game.worldPlants) {
    const s = worldToScreen(p.x, p.y);
    if (s.x < -30 || s.x > canvas.width + 30 || s.y < -30 || s.y > canvas.height + 30) continue;
    const sway = Math.sin(game.time * 1.5 + p.phase) * 3;
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.6;
    if (p.type === 0) {
      // Stem + circle top
      ctx.fillRect(s.x - 1, s.y, 2, p.size);
      ctx.beginPath(); ctx.arc(s.x + sway, s.y, p.size * 0.4, 0, Math.PI * 2); ctx.fill();
    } else if (p.type === 1) {
      // Triangle plant
      ctx.beginPath();
      ctx.moveTo(s.x + sway, s.y - p.size);
      ctx.lineTo(s.x + p.size * 0.5, s.y + p.size * 0.3);
      ctx.lineTo(s.x - p.size * 0.5, s.y + p.size * 0.3);
      ctx.closePath(); ctx.fill();
    } else {
      // Glowing dot
      ctx.beginPath(); ctx.arc(s.x, s.y, Math.max(0.1, p.size * 0.3 + Math.sin(game.time * 2 + p.phase) * 1.5), 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(46,204,113,0.2)';
      ctx.beginPath(); ctx.arc(s.x, s.y, p.size, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

function drawWeaponEffects() {
  const p = game.player;
  const ps = worldToScreen(p.x, p.y);

  for (const w of p.weapons) {
    if (w.key === 'shockwave' && w.pulseAnim > 0) {
      ctx.strokeStyle = `rgba(255,200,50,${w.pulseAnim * 0.6})`;
      ctx.lineWidth = 3;
      const r = w.pulseRadius * (1 - w.pulseAnim * 0.3);
      ctx.beginPath(); ctx.arc(ps.x, ps.y, r, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = `rgba(255,200,50,${w.pulseAnim * 0.3})`;
      ctx.lineWidth = 6;
      ctx.beginPath(); ctx.arc(ps.x, ps.y, r * 0.8, 0, Math.PI * 2); ctx.stroke();
    }
    if (w.key === 'cryo' && w.pulseAnim > 0) {
      ctx.fillStyle = `rgba(100,200,255,${w.pulseAnim * 0.15})`;
      ctx.beginPath(); ctx.arc(ps.x, ps.y, w.slowRadius, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = `rgba(100,200,255,${w.pulseAnim * 0.5})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(ps.x, ps.y, w.slowRadius, 0, Math.PI * 2); ctx.stroke();
    }
  }
}

function drawDrones() {
  const p = game.player;
  for (const w of p.weapons) {
    if (w.key !== 'drones') continue;
    for (let d = 0; d < w.droneCount; d++) {
      const ds = w.droneStates[d];
      const da = w.orbitAngle + (d / w.droneCount) * Math.PI * 2;
      const dx = p.x + Math.cos(da) * w.orbitRadius;
      const dy = p.y + Math.sin(da) * w.orbitRadius;
      const s = worldToScreen(dx, dy);
      if (ds && ds.hp <= 0) {
        // Dead drone — faint regen ghost
        const regenProg = 1 - ds.regenTimer / 5;
        ctx.globalAlpha = 0.15 + regenProg * 0.15;
        ctx.fillStyle = '#74b9ff';
        ctx.beginPath(); ctx.arc(s.x, s.y, w.droneRadius, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
        continue;
      }
      // Glow
      ctx.fillStyle = 'rgba(100,200,255,0.2)';
      ctx.beginPath(); ctx.arc(s.x, s.y, w.droneRadius * 2, 0, Math.PI * 2); ctx.fill();
      // Body
      ctx.fillStyle = '#74b9ff';
      ctx.strokeStyle = '#2980b9';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(s.x, s.y, w.droneRadius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      // Core
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(s.x, s.y, 2, 0, Math.PI * 2); ctx.fill();
      // HP indicator — dim pips when damaged
      if (ds && ds.hp < ds.maxHp) {
        for (let h = 0; h < ds.maxHp; h++) {
          const ha = da + (h - (ds.maxHp - 1) / 2) * 0.3;
          const hx = s.x + Math.cos(ha) * (w.droneRadius + 5);
          const hy = s.y + Math.sin(ha) * (w.droneRadius + 5);
          ctx.fillStyle = h < ds.hp ? 'rgba(100,200,255,0.8)' : 'rgba(100,200,255,0.2)';
          ctx.beginPath(); ctx.arc(hx, hy, 1.5, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  }
}

function drawCompass() {
  const p = game.player;
  const d = dist(p.x, p.y, 0, 0);
  if (d < 200) return; // Don't show when near colony

  const cx = canvas.width - 50;
  const cy = canvas.height - 50;
  const a = angle(p.x, p.y, 0, 0);

  // Outer ring
  ctx.strokeStyle = 'rgba(78,205,196,0.3)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, 24, 0, Math.PI * 2); ctx.stroke();

  // Colony icon in center (small dot)
  ctx.fillStyle = 'rgba(78,205,196,0.4)';
  ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();

  // Arrow pointing toward colony
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(a);
  ctx.fillStyle = '#4ecdc4';
  ctx.beginPath();
  ctx.moveTo(22, 0);
  ctx.lineTo(12, -6);
  ctx.lineTo(14, 0);
  ctx.lineTo(12, 6);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Distance label
  const distLabel = d >= 1000 ? `${(d / 1000).toFixed(1)}k` : Math.floor(d).toString();
  ctx.font = '10px "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(78,205,196,0.7)';
  ctx.fillText(distLabel, cx, cy + 38);
}

function drawPlanetaryDefenses() {
  if (!game || game.player.level < 3) return;

  const level = game.player.level;
  const defenseRange = 150 + level * 12;
  const s = worldToScreen(0, 0);

  // Defense range indicator
  ctx.strokeStyle = `rgba(78,205,196,${0.08 + Math.sin(game.time * 2) * 0.04})`;
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 6]);
  ctx.beginPath(); ctx.arc(s.x, s.y, defenseRange, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);

  // Defense turrets around colony (visual only)
  const turretCount = Math.min(6, 1 + Math.floor((level - 3) / 3));
  for (let i = 0; i < turretCount; i++) {
    const ta = (i / turretCount) * Math.PI * 2 + game.time * 0.2;
    const tr = getColonyEra() === 'fob' ? 68 : 50;
    const tx = s.x + Math.cos(ta) * tr;
    const ty = s.y + Math.sin(ta) * tr;

    // Turret base
    ctx.fillStyle = '#2d3436';
    ctx.strokeStyle = '#4ecdc4';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(tx, ty, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

    // Turret barrel (point at nearest enemy if any)
    let barrelAngle = ta + Math.PI;
    for (const e of game.enemies) {
      if (dist(0, 0, e.x, e.y) < defenseRange) {
        barrelAngle = angle(tx, ty, worldToScreen(e.x, e.y).x, worldToScreen(e.x, e.y).y);
        break;
      }
    }
    ctx.strokeStyle = '#4ecdc4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx + Math.cos(barrelAngle) * 10, ty + Math.sin(barrelAngle) * 10);
    ctx.stroke();

    // Glow
    ctx.fillStyle = `rgba(78,205,196,${0.15 + Math.sin(game.time * 4 + i) * 0.1})`;
    ctx.beginPath(); ctx.arc(tx, ty, 10, 0, Math.PI * 2); ctx.fill();
  }
}
