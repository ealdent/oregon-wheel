# Mother OS Non-Boss Enemy Art Pass

## Spec

- [x] Rework all non-boss enemy visuals to match the new enemy design reference.
- [x] Preserve the first seven existing enemy identities while improving their silhouettes and line detail.
- [x] Add the three referenced non-boss designs that are not currently in the game: Harvester Mite, Void Leech, and Obelisk Floater.
- [x] Use the same high-fidelity schematic language in the right-side threat preview and live path rendering.
- [x] Keep boss logic intact; this pass is about the non-boss visual catalog.
- [x] Preserve single-file/no-dependency structure.
- [x] Verify with static checks and browser screenshots.

## Plan

- [x] Record the enemy-art correction pattern in `tasks/lessons.md`.
- [x] Inspect existing enemy definitions, threat preview drawing, and live enemy drawing.
- [x] Expand enemy definitions to all ten reference enemies.
- [x] Build shared schematic enemy drawing primitives and route preview/live render through them.
- [x] Run parse/self-containment checks and browser visual verification.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Enemy catalog expanded
- [x] Enemy rendering reworked
- [x] Verification completed

## Review

- The enemy catalog now contains all ten reference non-boss designs: Crawler Drone, Shield Beetle, Split Slime, Burrower Worm, Static Wisp, Juggernaut Walker, Phantom Mimic, Harvester Mite, Void Leech, and Obelisk Floater.
- Added Harvester Mite, Void Leech, and Obelisk Floater as unlockable non-boss enemies after the original seven.
- Preview and live path rendering now route through one shared reference-style schematic drawing function so the side panel and active enemies stay visually consistent.
- Browser verification generated a 10-enemy lineup screenshot at `/tmp/mother-os-enemy-sheet.png` and a live game screenshot at `/tmp/mother-os-enemies-game.png`.
