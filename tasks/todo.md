# Mother OS Campaign Map Planning

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
