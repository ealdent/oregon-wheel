# Mother OS Operation Countdown Pass

## Spec

- [x] Replace the fixed Sector-76 label with facility names that count down through sectors.
- [x] Start from a readable example like `Tokamak Facility B Sector 7`.
- [x] Make Sector 1 the boss zone for the current facility.
- [x] After the boss sector clears, roll into a new randomized facility with a new sector countdown.
- [x] Remove the old every-five-waves boss cadence.
- [x] Surface the operation name in the canvas, side tactics panel, start button, logs, and wave summary.
- [x] Preserve unlimited waves, existing scaling, and single-file/no-dependency structure.
- [x] Verify with static checks and browser screenshots.

## Plan

- [x] Record the level-naming correction pattern in `tasks/lessons.md`.
- [x] Inspect current wave, boss, forecast, and Sector-76 rendering code.
- [x] Add operation generation, sector countdown state, and boss-sector detection.
- [x] Update forecast, wave generation, completion, logs, and UI rendering to use operation labels.
- [x] Run parse/self-containment checks and browser layout verification.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Countdown logic implemented
- [x] UI labels updated
- [x] Verification completed

## Review

- Initial operation is `Tokamak Facility B Sector 7`.
- Boss waves now trigger when the current operation reaches Sector 1, not every five waves.
- Clearing a non-boss sector decrements the current operation sector; clearing a boss sector creates a new randomized facility and sector countdown.
- The operation label appears in the canvas overlay, right tactics panel, start button, browser title, logs, and wave summary.
- Browser verification passed at desktop and mobile widths with screenshots at `/tmp/mother-os-zones-desktop.png` and `/tmp/mother-os-zones-mobile.png`.
