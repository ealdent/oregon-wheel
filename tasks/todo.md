# Mother OS Tower Defense

## Spec

- [x] Create a new no-build browser game under `arcade/mother-os-defense/` that loads directly from `index.html`.
- [x] Use the supplied image as theme direction: green CRT tactical schematic, industrial sector map, Mother OS interface, tower dossiers, and enemy intel.
- [x] Implement a substantial tower-defense loop with unlimited waves, regular waves around 1-2 minutes, boss waves closer to 2-3 minutes, and scaling difficulty.
- [x] Allow tower placement, selling, targeting, mid-wave upgrades, and between-wave upgrades.
- [x] Include multiple tower designs inspired by the image: pulse needle, arc relay, cryo prism, mine layer, and signal jammer.
- [x] Include varied enemies inspired by the image: crawler drone, shield beetle, split slime, burrower worm, static wisp, juggernaut walker, phantom mimic, and boss variants.
- [x] Add player controls for pause, speed, auto-continue between waves, and sound effects on/off.
- [x] Make the UI responsive, polished, and playable on desktop and mobile without external dependencies.
- [x] Add the game to `arcade/index.html`.
- [x] Verify static self-containment, JavaScript parsing, browser boot, core interactions, wave progression, and responsive layout.

## Plan

- [x] Read project instructions, arcade hub conventions, and browser-game workflow guidance.
- [x] Build a single-file `index.html` with separated simulation, rendering, input, audio, wave, and UI sections.
- [x] Create the CRT tactical visual system with responsive canvas layout, side/bottom controls, readable HUD, and reduced-motion support.
- [x] Implement path traversal, enemy behaviors, tower targeting/effects, projectiles, mines, upgrades, economy, lives, and infinite scaling.
- [x] Update the arcade hub card and styles for the new game.
- [x] Run verification and record results in this file.

## Progress

- [x] Context gathered
- [x] Task plan written
- [x] Game implemented
- [x] Arcade index updated
- [x] Verification completed

## Review

- Created `arcade/mother-os-defense/index.html`, a self-contained canvas + DOM tower defense game with no external dependencies.
- Added `Mother OS Defense` to `arcade/index.html`.
- Implemented the CRT tactical Mother OS theme, unlimited generated waves, boss waves, auto-continue toggle, sound toggle, pause/speed controls, five tower designs, seven enemy classes, mid-wave and between-wave upgrades, selling, target modes, mines, projectiles, status effects, and responsive desktop/mobile layouts.
- Verified inline JavaScript parsing for the new game and arcade index.
- Verified the new game file contains no `http`, `https`, `@import`, external stylesheet, external script, or `url(...)` references.
- Verified in headless Chrome from `file://`:
  - title loads as `Mother OS: Sector-76 Defense`
  - five tower cards and seven enemy intel cards render
  - tower placement works
  - upgrading works (`Pulse Needle MK-1` to `MK-2`)
  - wave 1 starts and resolves back to planning
  - auto-continue toggle activates
  - mobile layout puts the playfield first and sizes the canvas to `350x238` at a 390px viewport
  - no console errors or warnings were emitted
