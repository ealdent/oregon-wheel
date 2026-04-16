// ============ INPUT ============
document.addEventListener('keydown', e => {
  keys[e.key.toLowerCase()] = true;
  if (e.key === 'Escape' || e.key === 'p' || e.key === ' ') {
    if (game && game.phase === 'playing') {
      paused = !paused;
      document.getElementById('pause-hint').style.display = paused ? 'block' : 'none';
      if (paused) updatePauseStats();
    }
  }
});
document.addEventListener('keyup', e => { keys[e.key.toLowerCase()] = false; });

// Touch joystick
const joystickZone = document.getElementById('joystick-zone');
joystickZone.addEventListener('touchstart', e => {
  e.preventDefault();
  const t = e.changedTouches[0];
  joystick.active = true;
  joystick.touchId = t.identifier;
  joystick.ox = t.clientX;
  joystick.oy = t.clientY;
  joystick.dx = 0;
  joystick.dy = 0;
}, { passive: false });
joystickZone.addEventListener('touchmove', e => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    if (t.identifier === joystick.touchId) {
      const dx = t.clientX - joystick.ox;
      const dy = t.clientY - joystick.oy;
      const d = Math.sqrt(dx * dx + dy * dy);
      const maxR = 60;
      if (d > maxR) {
        joystick.dx = (dx / d) * maxR;
        joystick.dy = (dy / d) * maxR;
      } else {
        joystick.dx = dx;
        joystick.dy = dy;
      }
    }
  }
}, { passive: false });
joystickZone.addEventListener('touchend', e => {
  for (const t of e.changedTouches) {
    if (t.identifier === joystick.touchId) {
      joystick.active = false;
      joystick.dx = 0;
      joystick.dy = 0;
    }
  }
});
joystickZone.addEventListener('touchcancel', e => {
  joystick.active = false;
  joystick.dx = 0;
  joystick.dy = 0;
});

function getInputDir() {
  let dx = 0, dy = 0;
  if (keys['w'] || keys['arrowup']) dy -= 1;
  if (keys['s'] || keys['arrowdown']) dy += 1;
  if (keys['a'] || keys['arrowleft']) dx -= 1;
  if (keys['d'] || keys['arrowright']) dx += 1;
  if (joystick.active) {
    const maxR = 60;
    dx += joystick.dx / maxR;
    dy += joystick.dy / maxR;
  }
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len > 1) { dx /= len; dy /= len; }
  return { dx, dy };
}
