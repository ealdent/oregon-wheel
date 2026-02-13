# Vector Defense

A retro tactical missile command game with a CRT military aesthetic. Defend your cities from waves of incoming missiles, bombers, and increasingly dangerous enemies by building turrets and shield generators across a two-tier terrain.

**[Play Now](https://ealdent.github.io/oregon-wheel/vector-defense/vector-defense.html)**

## How to Play

1. **Placement Phase** — Before each wave, spend funds to place turrets on the upper terrain and shield generators over cities. Click a turret type from the armory, then click the terrain to place it. Click an existing turret or shield generator to upgrade it.
2. **Combat Phase** — Press **ENGAGE** to start the wave. Turrets fire automatically at enemies within range. Shield generators intercept missiles before they reach your cities.
3. **Survive** — Protect your cities. If all cities are destroyed, it's game over.

## Turrets

| Type | Cost | Description |
|------|------|-------------|
| FLAK | $800 | Dual-barrel suppression fire with medium range |
| SAM | $1,800 | Homing missiles with high single-target impact |
| ARTILLERY | $2,800 | Heavy cannon that targets the strongest enemy. Massive single-hit damage with high velocity shells, but slow tracking makes it ineffective against fast targets |
| LASER | $4,500 | Focused beam that grows stronger the longer it dwells on a target. Overheats after 1s of continuous fire, forcing a 3s cooldown |
| SWARM | $3,500 | Launches a barrage of homing missiles that prioritize high-value targets and evade point defense fire |

Turrets can be upgraded by clicking them during the placement phase. Each upgrade improves fire rate, range, and damage (cost doubles per level).

## Shield Generators

Shield generators project a floating energy arc above city buildings, intercepting enemy missiles before they reach the ground. Place them via the SHIELD button in the armory — they auto-position over the widest uncovered section of a city.

- Click a generator's underground bunker to upgrade it (max level 3)
- Each level increases shield HP (5 / 10 / 15 / 20)
- Destroyed shields go offline for one wave, then auto-repair at full HP
- Shields merge when cities merge

## Cities & Population

Cities grow organically between waves — undamaged cities sprout new buildings and expand outward. Cities that grow into each other automatically merge, potentially triggering a tier upgrade.

- **Villages** spawn in gaps between cities. After surviving 4 waves undamaged, they promote to tier-1 towns.
- **Tiers** (0–5) affect building style, height, and glow color, progressing from teal to purple to gold.
- **Population** is tracked in the HUD, calculated from building size and tier. Damage reduces population.

## Enemies

- **Normal** (red) — Standard missiles
- **Fast** (purple) — High speed, appears from wave 3
- **Armored** (white) — Heavy HP, appears from wave 4
- **Boss** (red, large) — Massive HP with ember trail, appears from wave 5
- **MIRV** (orange) — Splits into 3 bomblets mid-flight, appears from wave 7
- **Bomber** (yellow) — Flies across the screen dropping ordnance, appears from wave 3
- **Boss Ship** (every 5th wave) — Heavily-armored gunship with multiple weapon systems:
  - Bombing runs at 3x speed
  - EMP pulses that disable nearby turrets
  - Cluster bomb salvos
  - Rapid-fire point defense tracers that intercept incoming player missiles
  - Mega-boss variant (every 10th wave) adds energy shields and escort bombers

Difficulty scales in tiers: after every boss wave (every 5 levels), all enemies gain +50% HP. Between bosses, waves ramp up with more enemies and a harder mix of enemy types.

## Economy

- Destroying enemies earns funds and score (rewards scale with wave number)
- Completing a wave awards a base bonus plus a per-city survival bonus
- Losing a city costs $300
- Damaged cities heal HP and repair buildings between waves
