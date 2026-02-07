# Vector Defense

A retro-styled missile command game built as a single self-contained HTML file using React and Canvas.

Defend your cities from waves of incoming missiles, bombers, and increasingly dangerous enemies by building and upgrading turrets on the mountainous terrain.

## How to Play

1. **Placement Phase** — Before each wave, spend funds to place turrets on the terrain and upgrade existing ones. Click a turret type from the armory, then click the map to place it. Click an existing turret to upgrade it.
2. **Combat Phase** — Press **ENGAGE** to start the wave. Turrets fire automatically at enemies within range.
3. **Survive** — Protect your cities. If all cities are destroyed, it's game over.

## Turrets

| Type | Cost | Description |
|------|------|-------------|
| FLAK | $800 | Area damage with medium range |
| SAM | $1,800 | Homing missiles with high single-target impact |
| HEAVY | $2,800 | Massive explosion radius |
| LASER | $4,500 | Long-range rapid fire beam |

Turrets can be upgraded by clicking them during the placement phase. Each upgrade improves fire rate, range, and damage (cost doubles per level).

## Enemies

- **Normal** (red) — Standard missiles
- **Fast** (purple) — High speed, appears from wave 3
- **Armored** (white) — Heavy HP, appears from wave 4
- **Boss** (red, large) — Massive HP pool, appears from wave 5
- **MIRV** (orange) — Splits into 3 bomblets mid-flight, appears from wave 7
- **Bomber** (yellow) — Flies across the screen dropping ordnance, appears from wave 3

## Economy

- Destroying enemies earns funds and score
- Completing a wave awards a base bonus plus a per-city survival bonus
- Losing a city costs $300
- Cities can be rebuilt for $2,000 during the placement phase
