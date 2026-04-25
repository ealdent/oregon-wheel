# Mother OS Boss Asset Pass

## Spec

- [x] Add four high-fidelity boss assets inspired by the new reference: Hive Mother, Conduit, Void Colossus, and Harvester.
- [x] Use these new bosses instead of scaled regular enemies on Sector 1 boss waves.
- [x] Preserve the countdown structure where Sector 1 is the boss before moving to the next facility.
- [x] Cache boss line art separately from regular enemies so high detail does not hurt runtime performance.
- [x] Show boss schematics in the tactical-control preview and render them clearly in-map.
- [x] Keep the game single-file and dependency-free.
- [ ] Commit and push the completed checkpoint.

## Plan

- [x] Inspect current boss wave, preview, and enemy sprite cache paths.
- [x] Add boss definitions and deterministic boss selection for Sector 1.
- [x] Implement high-fidelity boss drawing helpers and cached boss sprites.
- [x] Wire boss previews and live boss rendering.
- [x] Run JavaScript parse, dependency scan, whitespace check, and browser screenshot verification.
- [ ] Commit and push to `main`.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Boss definitions wired
- [x] Boss art implemented
- [x] Verification completed

## Review

- Added four dedicated cached boss schematics: The Hive Mother, The Conduit, The Void Colossus, and The Harvester.
- Replaced scaled regular-enemy bosses with operation-rotated Sector 1 boss signatures and live boss spawn data.
- Verified inline JavaScript parsing, whitespace, dependency-free game page scan, forced Sector 1 live boss rendering, and all four boss preview rotations.
- Browser verification screenshots: `/tmp/mother-os-boss-sector-check.png` and `/tmp/mother-os-boss-rotation-check.png`.
