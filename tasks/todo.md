# Mother OS Campaign Map Planning

## Campaign Map Bracket and Terrain Cleanup

### Spec

- [x] Convert facility nodes away from filled plaques and heavy top rails into low-fill bracketed schematic frames.
- [x] Keep every facility schematic fully contained inside the node frame with connected, intentional linework.
- [x] Remove terrain contour marks that read as circular squiggles and replace them with subtler cartographic detail.
- [x] Preserve the full-width, non-stretched campaign viewport and static no-build architecture.
- [x] Verify with static JS checks and a campaign screenshot before committing.

### Plan

- [x] Reduce node fill, remove the dominant upper border, and draw corner/side brackets plus subtle scan hatching.
- [x] Scale and constrain campaign facility icons, rebuilding any overshooting icon details inside a fixed schematic viewport.
- [x] Replace circular contour clusters with faint ridge lines, forest pockets, and sparse survey ticks.
- [x] Run `node --check`, `git diff --check`, and capture a campaign map screenshot.
- [ ] Commit and push the completed checkpoint.

### Review

- Facility nodes now use a dark route mask, very faint type tint, corner brackets, side brackets, and sparse scan hatching instead of a colored plaque or dominant top rail.
- Facility schematics are clipped inside a fixed viewport and redrawn with connected bases, supports, masts, gantries, and equipment details so they do not exceed the node frame.
- Circular contour clusters were replaced with subtle ridge fields and survey ticks; mountains and forests were lowered in density/opacity.
- Verified with `node --check`, `git diff --check`, and a forced campaign-state Chrome screenshot at 2048px width.

## Campaign Map Fidelity Pass

### Spec

- [x] Remove campaign-map visual stretching while keeping the grid full-width.
- [x] Rework facility nodes from filled badges into lighter schematic frames with top/bottom rails and side/corner structure like the reference.
- [x] Increase facility schematic fidelity with more internal construction detail and deterministic variation.
- [x] Make landscape/topography subtler but more intricate: finer mountain, forest, river, contour, and scanline detail.
- [x] Preserve the static no-build/no-server architecture and verify with screenshot-based browser QA.

### Plan

- [x] Replace campaign non-uniform scaling with a wider logical viewport and matching pointer coordinate mapping.
- [x] Update campaign render bounds helpers so grid, terrain, frame, legend, routes, and selection panel use the dynamic map width.
- [x] Refactor node rendering into transparent framed schematic modules with richer icons and reduced fill opacity.
- [x] Tune terrain alpha/detail toward the uploaded reference image.
- [x] Run JS/static checks and capture campaign screenshots before committing.

### Review

- Campaign rendering now uses uniform scaling plus a dynamic logical map width, so the grid fills the horizontal canvas without stretching facility art or text.
- Campaign input, map centering, legend, selection panel, frame, and terrain now use the same dynamic campaign viewport.
- Facility nodes now use lower-opacity fills, top/bottom rails, side rails, internal hatching, and richer schematic details.
- Terrain now uses subtler rivers plus denser mountains, forests, contours, and scanline grime.
- Verified with `node --check`, `git diff --check`, and a forced campaign-state Chrome screenshot at 2048px width.

## Current Visual Pass

### Spec

- [x] Make the campaign grid fill the full horizontal canvas instead of sitting inside a letterboxed board area.
- [x] Add facility schematic graphics inside campaign nodes, with type-specific silhouettes and deterministic per-facility variation.
- [x] Add localized topography to the campaign map: mountain ranges, forest clusters, rivers, and terrain contour detail.
- [x] Keep the static no-build/no-server architecture and verify with browser screenshots.

### Plan

- [x] Split campaign rendering/input scaling from combat rendering so campaign mode can fill the whole canvas.
- [x] Add deterministic terrain rendering anchored to campaign world coordinates and responsive to map pan.
- [x] Replace plain campaign node boxes with larger nodes that include facility art, status text, and existing route states.
- [x] Verify desktop campaign screenshot and static script checks.
- [x] Commit and push the completed checkpoint.

### Review

- Campaign mode now renders with full-canvas scaling instead of combat-board letterboxing, with pointer mapping adjusted to match.
- Facility nodes now include deterministic schematic artwork by type: tokamak, cargo, foundry, cryo, and radar variants.
- The campaign map background now includes seeded rivers, mountain ranges, forest clusters, and contour rings anchored to campaign pan.
- Verified with `node --check`, `git diff --check`, and a 2048px-wide headless Chrome campaign screenshot.

## Spec

- [x] Add a persistent `localStorage` campaign map made of facility nodes connected in a fog-of-war graph.
- [x] Keep the first facility as the starting run; show the campaign map only after the first facility is secured, then make it available between zones/facilities.
- [x] Treat every facility as a fresh self-contained run with its own randomized map/path, sector countdown, and boss sector.
- [x] Allow players to enter any visible uncleared facility connected to a secured facility; the starting facility is the only exception.
- [x] Allow exiting a facility mid-run; completed sectors are retained, but partial-sector progress resets to the last completed sector state.
- [x] On failure or abandoned partial progress, retry the same facility with the same generated map and sector count, resetting to the saved facility checkpoint.
- [x] At facility completion, mark it secured, turn auto-advance off, reveal connected next options, and show a dismissible summary dialog before returning to the map.
- [x] Generate 0-4 exits per secured facility, heavily weighting 2, then 3, with 0 and 4 rare; only allow 0 when at least two sibling nodes have non-zero exits.
- [x] Make facility type affect both visual treatment and balance.
- [x] At the campaign map level, replace the top-center gameplay controls and top-right resource feed with campaign-relevant controls and status instead of leaving wave/tower controls visible.
- [x] Preserve the static no-build/no-server architecture.

## Plan

- [x] Define campaign data model: nodes, edges, visibility, secured state, facility seed, facility type, sector count, current sector, and checkpoint stats.
- [x] Add deterministic seeded generation for facility graph expansion, facility names/types, exit counts, procedural TD paths, build pads, and visual variants.
- [x] Split current game state into campaign state and facility-run state while keeping the current renderer modules static and script-order based.
- [x] Implement campaign map screen with pan/scroll, completed/current/available/unknown styling, connector lines, legend, and node selection.
- [x] Add campaign-specific top chrome showing map actions, selected facility status, secured count, visible threats, campaign score/progress, and session controls.
- [x] Add facility start/resume/exit flow, including checkpoint restore and partial-sector reset.
- [x] Generate per-facility TD maps and visuals, then adapt path/build validation/rendering to use the active facility layout instead of global fixed path constants.
- [x] Add facility balance modifiers by facility type and apply them to enemy composition, health, rewards, timing, or sector count.
- [x] Add facility completion summary modal with kills, enemies leaked, credits earned/spent, score gained, towers built/sold/upgraded, sectors cleared, and boss defeated.
- [x] Ensure auto-advance turns off when a facility completes and starts off for every new/resumed facility.
- [x] Verify localStorage persistence, first-run behavior, map reveal rules, facility retry/reset behavior, procedural map rendering, boss completion summary, and static file loading.
- [ ] Commit and push after implementation checkpoints.

## Progress

- [x] Clarifying questions answered
- [x] Planning artifact written
- [x] Plan confirmed
- [x] Implementation started
- [x] Verification completed

## Review

- Added static `campaign.js` with localStorage persistence, seeded graph generation, facility checkpoints, summaries, and facility entry/exit helpers.
- Updated the existing static modules to support campaign vs facility modes, procedural facility paths, campaign-specific top chrome, summary modal, and facility type balance/visual modifiers.
- Verified with `node --check` across every game script.
- Verified in headless Chrome from `file://` that the game loads and renders the facility battlefield.
- Verified via Chrome DevTools runtime flow that first-facility completion unlocks the map, reveals three connected facilities and unknown routes, displays a summary, starts new facilities with fresh resources and auto off, resets abandoned partial progress, and restores completed-sector checkpoints.
