// ============ PAUSE STATS ============
function updatePauseStats() {
  const el = document.getElementById('pause-stats');
  const stats = game ? game.weaponStats : {};
  const keys = Object.keys(stats).filter(k => stats[k].damage > 0 || stats[k].kills > 0);
  if (keys.length === 0) { el.innerHTML = '<div class="no-stats">No weapon stats yet</div>'; return; }
  // Check for prestige names
  const getName = (key) => {
    if (WEAPON_NAMES[key]) return WEAPON_NAMES[key];
    const w = game.player.weapons.find(w => w.key === key);
    return w ? w.name : key;
  };
  const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : n.toString();
  let html = '<table><tr><th>Weapon</th><th>Kills</th><th>Damage</th></tr>';
  keys.sort((a, b) => stats[b].damage - stats[a].damage);
  for (const k of keys) {
    html += `<tr><td>${getName(k)}</td><td>${fmt(stats[k].kills)}</td><td>${fmt(stats[k].damage)}</td></tr>`;
  }
  html += '</table>';
  el.innerHTML = html;
}

// ============ UPGRADE SYSTEM ============
function getUpgradeChoices() {
  const p = game.player;
  const pool = [];

  // Weapons player already has (can level up or prestige)
  for (const w of p.weapons) {
    const def = WEAPON_DEFS[w.key];
    if (w.level === 6 && !w.prestiged && def.prestige) {
      // Offer prestige evolution at level 6
      const pr = def.prestige;
      pool.push({ type: 'weapon', key: w.key, name: pr.name, icon: pr.icon, desc: pr.desc, level: 'PRESTIGE', isNew: false, isPrestige: true });
    } else if (w.level < 6 || w.prestiged) {
      // Normal level up (pre-prestige caps at 6, post-prestige keeps going)
      if (w.level < 6 || w.prestiged) {
        const name = w.prestiged ? def.prestige.name : def.name;
        const icon = w.prestiged ? def.prestige.icon : def.icon;
        const desc = w.prestiged ? def.prestige.desc : def.desc;
        if (!w.prestiged && w.level >= 6) continue; // shouldn't happen, but guard
        pool.push({ type: 'weapon', key: w.key, name, icon, desc, level: w.level + 1, isNew: false, isPrestige: false });
      }
    }
  }

  // New weapons player doesn't have (max 6 weapons)
  if (p.weapons.length < 6) {
    for (const key in WEAPON_DEFS) {
      if (!p.weapons.find(w => w.key === key)) {
        const def = WEAPON_DEFS[key];
        pool.push({ type: 'weapon', key, name: def.name, icon: def.icon, desc: def.desc, level: 1, isNew: true });
      }
    }
  }

  // Passives
  for (const key in PASSIVE_DEFS) {
    const currentLv = p.passives[key] || 0;
    const def = PASSIVE_DEFS[key];
    if (currentLv < def.maxLevel) {
      pool.push({ type: 'passive', key, name: def.name, icon: def.icon, desc: def.desc, level: currentLv + 1, isNew: currentLv === 0 });
    }
  }

  shuffle(pool);
  return pool.slice(0, 3);
}

function applyUpgrade(choice) {
  const p = game.player;
  if (choice.type === 'weapon') {
    let w = p.weapons.find(w => w.key === choice.key);
    if (w) {
      const def = WEAPON_DEFS[w.key];
      if (choice.isPrestige) {
        // Prestige evolution!
        w.prestiged = true;
        w.level = 7;
        def.prestige.evolve(w);
        showAnnouncement(`${def.prestige.name} AWAKENED!`, '#ffd700', 3);
        playSound('levelup');
      } else {
        w.level++;
        if (w.prestiged && def.prestige) {
          def.prestige.levelUp(w);
        } else {
          def.levelUp(w);
        }
      }
    } else {
      w = createWeapon(choice.key);
      p.weapons.push(w);
    }
  } else if (choice.type === 'passive') {
    const lv = (p.passives[choice.key] || 0) + 1;
    p.passives[choice.key] = lv;
    PASSIVE_DEFS[choice.key].apply(p, lv);
  }
  addColonyStructure();
  updateWeaponsDisplay();
}

function rankUpgradeChoice(c) {
  if (c.isPrestige) return 0;
  if (c.type === 'weapon' && !c.isNew) return 1;
  if (c.type === 'weapon' && c.isNew) return 2;
  return 3; // passive
}

function cancelAutoUpgrade() {
  if (autoUpgradeTimer) {
    clearTimeout(autoUpgradeTimer);
    autoUpgradeTimer = null;
  }
}

function selectUpgrade(choice) {
  cancelAutoUpgrade();
  applyUpgrade(choice);
  document.getElementById('upgrade-screen').classList.add('hidden');
  game.phase = 'playing';
  checkPendingLevelUps();
}

function showUpgradeScreen() {
  game.phase = 'upgrading';
  playSound('levelup');
  cancelAutoUpgrade();
  const choices = getUpgradeChoices();
  const container = document.getElementById('upgrade-cards');
  container.innerHTML = '';
  document.getElementById('upgrade-level-label').textContent = `Level ${game.player.level}`;

  if (choices.length === 0) {
    // All maxed out - give HP instead
    game.player.hp = Math.min(game.player.hp + 20, game.player.maxHp);
    spawnDmgText(game.player.x, game.player.y, '+20 HP', PAL.healText);
    game.phase = 'playing';
    checkPendingLevelUps();
    return;
  }

  // Find best choice for auto-upgrade
  let bestIdx = 0;
  let bestRank = rankUpgradeChoice(choices[0]);
  for (let i = 1; i < choices.length; i++) {
    const r = rankUpgradeChoice(choices[i]);
    if (r < bestRank) { bestRank = r; bestIdx = i; }
  }

  choices.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'upgrade-card' + (c.isNew ? ' new-weapon' : '') + (c.isPrestige ? ' prestige-weapon' : '');
    const levelText = c.isNew ? 'NEW!' : c.isPrestige ? '\u{2B50} PRESTIGE \u{2B50}' : 'Lv ' + c.level;
    card.innerHTML = `
      <span class="card-icon">${c.icon}</span>
      <div class="card-name">${c.name}</div>
      <div class="card-level">${levelText}</div>
      <div class="card-desc">${c.desc}</div>
    `;
    if (autoUpgrade && i === bestIdx) {
      card.classList.add('auto-selected');
      card.insertAdjacentHTML('beforeend', '<div class="auto-countdown">3</div>');
    }
    card.onclick = () => selectUpgrade(c);
    container.appendChild(card);
  });

  document.getElementById('upgrade-screen').classList.remove('hidden');

  // Auto-upgrade countdown
  if (autoUpgrade) {
    const bestChoice = choices[bestIdx];
    const countdownEl = container.querySelector('.auto-countdown');
    let remaining = 3;
    const tick = () => {
      remaining--;
      if (remaining <= 0) {
        selectUpgrade(bestChoice);
      } else {
        if (countdownEl) countdownEl.textContent = remaining;
        autoUpgradeTimer = setTimeout(tick, 1000);
      }
    };
    autoUpgradeTimer = setTimeout(tick, 1000);
  }
}

function checkPendingLevelUps() {
  if (game.pendingLevelUps > 0) {
    game.pendingLevelUps--;
    showUpgradeScreen();
  }
}

// ============ UPDATE WEAPONS DISPLAY ============
function updateWeaponsDisplay() {
  const el = document.getElementById('weapons-display');
  el.innerHTML = game.player.weapons.map(w => {
    const def = WEAPON_DEFS[w.key];
    const name = w.prestiged && def.prestige ? def.prestige.name : def.name;
    const icon = w.prestiged && def.prestige ? def.prestige.icon : def.icon;
    const lvClass = w.prestiged ? 'wlv prestige' : 'wlv';
    return `<div class="weapon-badge">${icon} ${name} <span class="${lvClass}">Lv${w.level}</span></div>`;
  }).join('');
}
