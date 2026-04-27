# Mother OS Defense Design Catalog

Last verified against implementation: 2026-04-27.

This is the running record for Mother OS Defense. Update it whenever a rule, asset family, balance decision, campaign mechanic, or UI requirement changes. The implemented source of truth lives in `js/config.js`, `js/campaign.js`, `js/gameplay.js`, `js/input.js`, `js/ui.js`, `js/render.js`, and the procedural art files.

## Core Requirements

- The game must remain a static browser game. It should load from `arcade/mother-os-defense/index.html` with no install step, build step, package manager, or local server requirement.
- The arcade index should expose the game as a normal arcade entry.
- The visual theme is Mother OS v1.7.3: green CRT phosphor, tactical schematic panels, grungy scanlines, high-fidelity neon line art, and industrial alien containment language.
- The UI should be responsive, professional, and playable on desktop and mobile.
- Expensive procedural art should be cached on offscreen canvases or equivalent sprites so high-fidelity linework does not make panning, placing towers, or combat feel sluggish.
- Gameplay should support unlimited sector progression inside the non-campaign fallback loop, while the campaign mode treats each facility as a finite level that ends at its boss sector.

## Runtime And Persistence

- `BOARD` is `1000 x 680`, with `pathWidth` set to `76`.
- Starting credits are `360`.
- Starting system health is `25`.
- Maximum tower level is `12`.
- Speed controls are `x1`, `x2`, and `x3`.
- Campaign data persists in `localStorage` under `mother-os-campaign-v1`.
- Campaign graph placement is versioned separately and currently uses graph layout version `2`.
- Campaign terrain is versioned separately and currently uses terrain version `3`.
- Facility path layouts are versioned separately and currently use layout version `2`.

## Campaign Loop

- The first facility starts as the entry point. The campaign map unlocks after the first facility is secured.
- Facilities are individual levels. Money, towers, mines, score, and lives do not carry into a brand-new facility.
- A facility can be abandoned between sectors. Completed sector progress is checkpointed; an in-progress sector resets to the last completed-sector checkpoint.
- Uncleared facilities can be entered only when visible and connected to a secured facility. The starting facility is the only exception.
- Securing a facility reveals its outgoing routes and connected unknown nodes.
- Campaign facilities use continuous world coordinates, not an orthogonal grid. Branches should fan outward like a computer-science tree: child exits point generally away from the route that led into the parent facility.
- All planned exits should survive placement. Collision avoidance may push a branch farther out or rotate it slightly, but should not silently drop an exit because another facility occupies a cardinal slot.
- Completing a facility shows a summary dialog with run stats, then returns control to the campaign map.
- Auto advance is disabled/reset when a facility is completed. Starting or resuming a facility begins with auto advance off.
- Secured facilities are marked secured. Campaign reward systems are intentionally deferred.

## Facility And Sector Rules

- Every facility has a countdown of sectors. Sector `1` is the boss sector.
- Non-boss sectors use regular waves. Boss sectors still include regular enemies but insert one or more boss chassis into the queue.
- Facility names combine a type/site and suffix, for example `Tokamak Reactor B` or `Orbital Cargo Facility 13`.
- Facility type affects visuals, route palette, enemy balance modifiers, and enemy weighting.
- Facility routes are generated per node and cached. Earlier/easier facilities bias toward longer, winding paths; harder facilities bias toward shorter, straighter paths.
- Current route archetypes include wide serpentine, perimeter loop, spiral, diagonal weave, offset switchback, direct dogleg, and straight chicane.

## Facility Classifications

These 12 classifications currently have unique campaign-map facility graphics.

| Key | Display label | Glyph | Gameplay identity |
| --- | --- | --- | --- |
| `tokamak` | Tokamak Reactor | `tokamak` | Faster pressure and volatile returns. Count `1.05`, speed `1.06`, reward `1.04`, interval `0.94`; wisp up, crawler down. |
| `radar` | Radar Array | `radar` | Signal-heavy phasing and burrowing threats. Health `1.02`, speed `1.04`, reward `1.03`; phantom/worm up, obelisk down. |
| `annex` | Annex Complex | `annex` | Balanced civic grid. Neutral modifiers; crawler/beetle slightly up. |
| `cryo` | Bio Research Facility | `bio` | Containment domes with slower, resilient anomalies. Count `0.96`, health `1.06`, speed `0.90`, armor `1.04`; slime up, wisp down. Key retained for save compatibility. |
| `foundry` | Mining Operation | `mining` | Fewer but harder armored chassis. Count `0.90`, health `1.14`, reward `1.08`, interval `1.08`, armor `1.16`; beetle/juggernaut up, crawler down. Key retained for save compatibility. |
| `power` | Power Plant | `power` | Heavy armor and unstable surges. Count `0.94`, health `1.12`, reward `1.08`, interval `1.05`, armor `1.12`; juggernaut/wisp up. |
| `comms` | Communications Hub | `comms` | More mimics and signal anomalies. Count `1.03`, speed `1.08`, interval `0.96`; phantom/wisp up, crawler down. |
| `hydro` | Hydroponics Farm | `hydro` | Many light biological swarms. Count `1.14`, health `0.92`, reward `0.98`, armor `0.90`; slime/mite up, juggernaut down. |
| `vehicle` | Vehicle Depot | `vehicle` | Mixed packs with heavier walkers. Health `1.08`, reward `1.05`, interval `1.03`, armor `1.10`; juggernaut/beetle up. |
| `satellite` | Satellite Uplink | `satellite` | Fast phasing threats and weak armor. Count `1.02`, health `0.96`, speed `1.13`, interval `0.94`, armor `0.94`; wisp/phantom up, beetle down. |
| `cargo` | Storage Facility | `storage` | Bigger packs with softer armor. Count `1.12`, health `0.94`, reward `0.98`, interval `1.02`, armor `0.94`; crawler up, mite down. Key retained for save compatibility. |
| `command` | Command Center | `command` | Elite-heavy hardened control nexus. Count `0.92`, health `1.16`, reward `1.12`, interval `1.08`, armor `1.14`; obelisk/juggernaut/phantom up. |

Facility node art rules:

- Nodes should feel like bracketed schematic modules, not filled plaques.
- Side brackets should be stronger than the top and bottom rails.
- Top and bottom borders should be thinner/partial lines.
- Facility buildings should fit inside the node frame with breathing room and should not protrude past the bracket.
- Facility glyphs are cached so the map can pan smoothly.

## Towers

These 5 towers currently have unique design-panel, placement-preview, and board graphics.

| Key | Code | Name | Role | Cost | Base damage | Range | Rate | Graphic identity | Mechanics |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `pulse` | `01` | Pulse Needle | Needle | 75 | 24 | 138 | 4.80 | Needle tower with stacked base and tall emitter. | High-rate kinetic projectile with armor pierce. |
| `arc` | `02` | Arc Relay | Chain | 125 | 34 | 140 | 1.05 | Coil relay with circular platform and stacked rings. | Chains lightning between targets. |
| `cryo` | `03` | Cryo Prism | Control | 105 | 13 | 132 | 1.25 | Crystal prism on a reinforced pedestal. | Slows and applies vulnerability. |
| `mine` | `04` | Mine Layer | Burst | 95 | 78 | 150 | 0.36 | Low mine platform / disc mechanism. | Lays proximity mines on the path. |
| `jammer` | `05` | Signal Jammer | Debuff | 110 | 17 | 126 | 0.82 | Signal mast with antenna dish and base. | Jams armor, burrow, and phasing; can stun at higher levels. |

Tower economy and upgrade rules:

- Upgrade cost is `round(baseCost * (0.58 + level * 0.58) * 1.13^(max(0, level - 5)))`.
- Damage scales by tier through level 5, then with a smaller over-level multiplier.
- Range and fire rate also scale with level.
- Arc chain count is `2 + floor(level / 2)`.
- Cryo slow caps at `0.68`.
- Mine radius is `48 + level * 5`; mine limit is `3 + level`.
- Pulse armor pierce starts at `0.35 + level * 0.035`.
- A tower placed during planning can be refunded at full cost only if it is still level 1, was just placed, and has not started a wave.
- Other tower sales return `70%` of total spent.

## Non-Boss Enemies

These 10 non-boss enemies currently have unique cached graphics inspired by the non-boss enemy design sheet.

| Key | Code | Name | Unlock | HP | Speed | Armor | Reward | Leak | Graphic identity | Special behavior |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `crawler` | `01` | Crawler Drone | 1 | 54 | 68 | 0 | 7 | 1 | Light spider drone with hooked legs. | Fast swarm unit. |
| `beetle` | `02` | Shield Beetle | 2 | 146 | 39 | 8 | 14 | 2 | Armored beetle shell with frontal plating. | Armor-heavy target; jam helps crack armor. |
| `slime` | `03` | Split Slime | 3 | 106 | 47 | 1.5 | 11 | 1 | Organic blob with tendrils and smaller lobes. | Splits into two child slimes on death. |
| `worm` | `04` | Burrower Worm | 4 | 192 | 35 | 5 | 17 | 2 | Segmented worm with biting head and shell rings. | Periodically burrows when not jammed; non-ground damage is reduced while burrowed. |
| `wisp` | `05` | Static Wisp | 5 | 118 | 74 | 0.5 | 15 | 1 | Electrical anomaly with branching tendrils. | Can phase through non-truesight damage unless jammed. |
| `juggernaut` | `06` | Juggernaut Walker | 6 | 468 | 28 | 15 | 34 | 4 | Heavy quadruped assault shell. | Slow, high-health, high-armor pressure unit. |
| `phantom` | `07` | Phantom Mimic | 7 | 208 | 52 | 3 | 24 | 2 | Cloaked humanoid mimic silhouette. | Spawns one weaker mimic child after entering the route. |
| `mite` | `08` | Harvester Mite | 8 | 162 | 55 | 4 | 20 | 2 | Low armored crawler with hooked mandibles. | Medium swarm pressure. |
| `leech` | `09` | Void Leech | 9 | 292 | 33 | 7.5 | 29 | 3 | Armored tunnel leech with toothed maw. | Durable lane pressure. |
| `obelisk` | `10` | Obelisk Floater | 10 | 242 | 44 | 5 | 31 | 3 | Floating crystal relay with shard supports. | Higher-value crystal threat. |

Enemy scaling rules:

- Health scale is `1.145^(wave - 1) * (1 + wave * 0.048)`.
- Armor scale is `1 + wave * 0.032`.
- Reward scale is `1 + wave * 0.055`.
- Bosses use the same base scaling with additional boss multipliers.

## Bosses

These 4 boss enemies currently have unique cached graphics inspired by the boss design sheet. They appear on Sector `1` of a facility.

| Key | Code | Name | Role | HP | Speed | Armor | Reward | Leak | Graphic identity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hive` | `B-01` | The Hive Mother | Brood Sovereign | 620 | 24 | 16 | 42 | 9 | Massive spider brood body with shell vents and hooked legs. |
| `conduit` | `B-02` | The Conduit | Relay Horror | 560 | 29 | 14 | 44 | 8 | Segmented turbine-maw relay horror with dorsal spines. |
| `colossus` | `B-03` | The Void Colossus | Rupture Chassis | 720 | 18 | 20 | 48 | 11 | Huge rupture body venting a vertical core breach. |
| `harvester` | `B-04` | The Harvester | Crystal Reaper | 640 | 25 | 17 | 46 | 10 | Crystalline reaper chassis with scythe pylons and live core. |

Boss wave rules:

- A boss sector inserts `1 + floor(wave / 20)` bosses into the enemy queue.
- Boss HP multiplier is `6.8 + wave * 0.42`.
- Boss rewards use an `8x` reward multiplier plus an additional boss kill bonus of `85 + wave * 7`.
- Bosses take slightly reduced damage and receive reduced slow/vulnerability impact compared with regular enemies.

## Wave Balance

Regular sector waves:

- Base count `44`.
- Count growth `+6` per wave.
- Count cap `138`.
- Spawn interval starts at `1.04`, drops by `0.01` per wave, and cannot go below `0.44`.
- Duration base is `36` seconds, with up to `24` seconds of growth.

Boss sector waves:

- Base count `34`.
- Count growth `+5` per wave.
- Count cap `122`.
- Spawn interval starts at `0.96`, drops by `0.012` per wave, and cannot go below `0.50`.
- Duration base is `64` seconds, with up to `30` seconds of growth.

Clear rewards:

- Sector clear bonus is `24 + wave * 6`.
- Boss sector clear adds `90`.
- Clearing a boss sector repairs `+1` system health if below `25`.

## Controls And UX Rules

- Desktop: number keys `1` through `5` select tower designs.
- Desktop: `Space` starts/resumes a wave when allowed.
- Desktop: `Escape` deselects the selected tower/campaign node and clears placement intent.
- Mobile and pointer UX: tapping outside the map area or tapping the same tower again should deselect to prevent accidental placements.
- Campaign map supports drag-to-pan and node selection.
- Facility controls in the top center should show relevant gameplay controls: start/pause, upgrade, target mode, sell/refund, speed, auto advance, and sound.
- Campaign map controls in the top center and top right should switch to campaign-relevant information instead of facility combat controls.
- The right Tactical Control panel should show enemy/wave intel when no tower is selected, and switch to selected-tower command intel when a tower is selected.
- Credits, system health, points, and time should be visible in the resource feed during play.
- Sound effects must have an on/off toggle.

## Procedural Campaign Terrain

Terrain is generated in persistent world-space chunks, not per viewport. When the player pans away and returns, rivers, forests, mountains, ridges, and other features must remain fixed.

Terrain generation is capped by the explored campaign frontier. Visible facilities, visible routes, and question-mark unknown exits define the current exploration geometry. The map renders a muted orange/red fog-of-war haze beyond that frontier: light haze just past the question marks, then progressively denser cover until distant terrain is fully hidden and no new terrain chunks are generated there.

Campaign map connections should read as schematic roads, not abstract graph lines. Visible and unknown routes use layered road strokes, side rails, center dashes, and small survey ticks inspired by the reference sheet's manmade terrain connectors.

Current terrain feature families:

- Rivers and streams.
- Mountain clusters.
- Forest clusters.
- Ridgelines and cliffs.
- Hills and plateaus.
- Water bodies.
- Swamps and marshes.
- Rocks and boulders.
- Coastlines.
- Terrain-context structures and survey ticks.

Terrain art rules:

- Topography should be visible enough to create an immersive world map, but it must remain behind facilities and routes.
- Mountains should use high-fidelity linework, facets, ridges, and base strokes inspired by the reference sheet.
- Forests should read as clusters of individual tree silhouettes, not generic noise.
- Rivers should be present and legible without overpowering routes or facility nodes.
- Avoid random circular squiggles or marks that do not read as intentional topography.
- Fog should obscure unexplored terrain before it can become a performance problem, while keeping frontier question marks readable.

## Graphics And Performance Rules

- Towers, non-boss enemies, bosses, facility glyphs, terrain glyphs, and preview art should keep the high-detail CRT schematic style.
- Repeated high-fidelity graphics should be cached and reused.
- Runtime drawing should avoid rebuilding complex line art every frame when a cached sprite can be used.
- Tower placement previews must remain responsive during combat.
- Health bars should align directly over their enemies and should not drift away from cached enemy sprites.
- Map panning must stay stable and should not regenerate visible terrain or facility art.

## Current Source Map

- `js/config.js`: core constants, wave balance, facility type definitions, tower definitions, enemy definitions, boss definitions, and legacy operation templates.
- `js/campaign.js`: campaign generation, localStorage persistence, terrain chunks, node graph expansion, facility path generation, checkpoints, and facility completion.
- `js/gameplay.js`: tower stats, upgrades, selling, wave generation, enemy spawning, combat, damage rules, special enemy behavior, rewards, and sector advancement.
- `js/input.js`: pointer, keyboard, mobile deselection, start/pause, speed, auto advance, sound, upgrade, target mode, and sell/refund controls.
- `js/ui.js`, `js/ui-tactics.js`, `js/ui-previews.js`: HUD, campaign/facility control states, tower cards, preview panels, and tactical summaries.
- `js/render.js`: main canvas rendering, campaign map rendering, cached facility and terrain glyphs, combat field drawing, HUD overlays, effects, and range previews.
- `js/art-towers.js`: unique tower line-art graphics.
- `js/art-enemies.js`: unique non-boss enemy line-art graphics.
- `js/art-bosses.js`: unique boss line-art graphics.

## Open Design Notes

- Campaign meta rewards are intentionally not implemented yet. Secured facilities only mark progression and reveal routes.
- A future campaign reward layer should build on secured facilities without breaking the current fresh-start-per-facility rule unless explicitly redesigned.
- Balance should continue to discourage a single fully upgraded tower from solving every facility. Enemy HP, armor, mix, and facility pressure should require tower variety.
- Art quality targets are set by the uploaded reference sheets: high-fidelity green/yellow CRT line art, crisp silhouette readability, and grungy tactical scan texture.
- This document should stay concise enough to maintain, but complete enough that future art, balance, and mechanics changes can be reviewed against it.
