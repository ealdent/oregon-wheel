// ============ WORLD GENERATION ============
function generateWorld() {
  const g = game;
  g.worldPlants = [];
  g.worldRocks = [];
  for (let i = 0; i < CFG.WORLD_PLANT_COUNT; i++) {
    g.worldPlants.push({
      x: rand(-3000, 3000), y: rand(-3000, 3000),
      type: randInt(0, 2),
      size: rand(4, 14),
      color: [PAL.plant1, PAL.plant2, PAL.plant3][randInt(0, 2)],
      phase: rand(0, Math.PI * 2),
    });
  }
  for (let i = 0; i < CFG.WORLD_ROCK_COUNT; i++) {
    g.worldRocks.push({
      x: rand(-3000, 3000), y: rand(-3000, 3000),
      size: rand(8, 25),
      color: `hsl(${randInt(160, 200)}, ${randInt(15, 30)}%, ${randInt(12, 22)}%)`,
      verts: randInt(4, 7),
      rotation: rand(0, Math.PI * 2),
    });
  }
  // Colony landing pad at origin
  g.colony = [{ type: 'pad', x: 0, y: 0 }];
}

function getColonyEra() {
  const w = game.wave;
  if (w <= 9) return 'station';
  if (w <= 19) return 'airship';
  return 'fob';
}

function addColonyStructure() {
  const g = game;
  const era = getColonyEra();
  const typesByEra = {
    station: ['hab', 'antenna', 'solar', 'dome'],
    airship: ['hab', 'antenna', 'solar', 'engine', 'balloon'],
    fob: ['hab', 'antenna', 'solar', 'bunker', 'turret_base'],
  };
  const types = typesByEra[era];
  const a = rand(0, Math.PI * 2);
  const r = rand(60, 150);
  g.colony.push({
    type: types[randInt(0, types.length - 1)],
    x: Math.cos(a) * r,
    y: Math.sin(a) * r,
  });
}

// ============ SPAWN SYSTEM ============
function getSpawnTypes() {
  const t = game.time;
  const types = ['spore'];
  if (t > 30) types.push('beetle');
  if (t > 75) types.push('bloomer');
  if (t > 100) types.push('stinger');
  if (t > 150) types.push('brute');
  if (game.wave >= 20) types.push('gloop');
  return types;
}
