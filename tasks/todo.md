# Single-Page Tower Defense

## Spec

- [x] Create a green-field, single-page HTML tower defense game that can be opened directly in a browser with no installs, build step, or external dependencies.
- [x] Use a fresh unexpected theme, not based on existing local tower-defense files.
- [x] Generate a random playable track on each new game.
- [x] Include five distinct tower types, each with four upgrade levels.
- [x] Include enemies with varied speed, health, armor, behavior, and rewards.
- [x] Include at least 20 varied waves with between-wave upgrade time.
- [x] Build a polished, cute, cartoony, responsive UI with a professional feel.
- [x] Verify the game renders, starts, upgrades towers, advances waves, and remains self-contained.

## Plan

- [x] Define the fantasy, controls, game loop, wave schedule, tower data, enemy data, and random track generator.
- [x] Implement simulation state separately from canvas rendering inside one HTML file.
- [x] Build the DOM HUD, tower shop, inspector, upgrade controls, pause/wave controls, and responsive layout.
- [x] Add cartoon rendering, particles, path variation, tower effects, projectiles, status effects, and wave intermission behavior.
- [x] Run browser-level smoke tests and static checks for external dependencies.

## Progress

- [x] Context gathered
- [x] Task plan written
- [x] Game implemented
- [x] Verification completed

## Review

- Created `tower-def/spin-cycle-skies.html`, a standalone canvas + DOM tower defense game with the unexpected dream laundromat theme.
- Added `Spin Cycle Skies` to `tower-def/index.html` as a launchable mission card.
- Implemented a random stitched track generator, 24 varied waves, five enemy types plus boss variants, five tower types, four upgrade purchases per tower, between-wave upgrade windows, selling, speed control, pause, and a charged Spin Cycle field ability.
- Verified the file is self-contained with no `http`, `https`, `@import`, external stylesheet, or external script references.
- Verified the inline JavaScript parses with Node.
- Verified in the browser from `file://`:
  - page title loads as `Spin Cycle Skies`
  - no console errors after reload
  - tower cards and wave controls are visible
  - tower placement works
  - planning-phase upgrade works
  - wave 1 starts
  - wave 1 resolves back to the planning window for wave 2
- Checked desktop and narrow mobile screenshots with headless Chrome; fixed the mobile overflow found during verification.
