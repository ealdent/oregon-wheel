// ============ GAME FLOW ============
function startGame() {
  initAudio();
  game = createGameState();
  game.player.weapons.push(createWeapon('plasma'));
  generateWorld();
  updateWeaponsDisplay();

  document.getElementById('title-screen').classList.add('hidden');
  document.getElementById('gameover-screen').classList.add('hidden');
  document.getElementById('upgrade-screen').classList.add('hidden');
  document.getElementById('ui').classList.remove('hidden');
  document.getElementById('weapons-display').classList.remove('hidden');
  document.getElementById('pause-hint').style.display = 'none';
  paused = false;
  keys = {};
  joystick.active = false;
  joystick.dx = 0;
  joystick.dy = 0;
  lastTime = performance.now();

  updateUI();
  showAnnouncement('COLONY ESTABLISHED', '#4ecdc4', 2.5);
}

function gameOver() {
  game.phase = 'dead';
  playSound('explosion');

  const min = Math.floor(game.time / 60);
  const sec = Math.floor(game.time % 60);
  document.getElementById('gameover-stats').innerHTML = `
    <div>Survived: <span class="stat-val">${min}:${sec.toString().padStart(2, '0')}</span></div>
    <div>Level: <span class="stat-val">${game.player.level}</span></div>
    <div>Kills: <span class="stat-val">${game.kills}</span></div>
    <div>Wave: <span class="stat-val">${game.wave}</span></div>
    <div>Weapons: <span class="stat-val">${game.player.weapons.map(w => WEAPON_DEFS[w.key].name).join(', ')}</span></div>
  `;
  document.getElementById('gameover-screen').classList.remove('hidden');
  document.getElementById('ui').classList.add('hidden');
  document.getElementById('weapons-display').classList.add('hidden');
}

// ============ MAIN LOOP ============
function gameLoop(timestamp) {
  dt = Math.min((timestamp - lastTime) / 1000, 0.05);
  lastTime = timestamp;

  update();
  render();
  requestAnimationFrame(gameLoop);
}

// ============ START ============
lastTime = performance.now();
requestAnimationFrame(gameLoop);
