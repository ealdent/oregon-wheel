# Mother OS Difficulty Balance Pass

## Spec

- [x] Compare possible tower DPS/cost against current enemy health and wave sizes.
- [x] Increase enemy HP and/or enemy counts enough that one fully upgraded tower is not a dominant strategy.
- [x] Preserve substantial wave length, readable pacing, and the current single-file/no-dependency setup.
- [x] Keep rewards and boss-sector cadence coherent after the difficulty increase.
- [x] Verify with static checks and balance simulation.
- [x] Commit and push the completed checkpoint.

## Plan

- [x] Inspect tower stat, upgrade, enemy scaling, and wave generation formulas.
- [x] Build a lightweight balance model from the in-game formulas.
- [x] Adjust enemy HP/count/reward pacing to increase pressure against single-tower play.
- [x] Run JavaScript parse, dependency scan, whitespace check, and balance model verification.
- [x] Commit and push to `main`.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Balance model completed
- [x] Difficulty tuning implemented
- [x] Verification completed

## Review

- Increased enemy base HP/armor and wave counts, then tightened spawn intervals so added enemies create density rather than bloated wave length.
- Centralized balance formulas for enemy counts, spawn intervals, duration estimates, health scaling, armor scaling, rewards, and clear bonuses.
- Balance model shows regular wave 10 pressure rising from about 488 HP/sec to about 900 HP/sec while rewards only rise about 1.04x; this should break the single fully upgraded tower path without starving mixed builds.
- Browser smoke test loaded the game, confirmed Sector 7 shows Spawn Pack 50 and Length 1:30, started combat, and captured `/tmp/mother-os-balance-smoke.png`.
