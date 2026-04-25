# Mother OS Enemy Render Performance Pass

## Spec

- [x] Explain whether the detailed enemy line art is causing sluggishness.
- [x] Optimize the expensive enemy rendering path without reducing visual fidelity.
- [x] Preserve status overlays, health bars, boss scaling, and the 10-enemy catalog.
- [x] Keep the game single-file and dependency-free.
- [x] Verify with static checks and browser performance smoke tests.

## Plan

- [x] Record the performance correction pattern in `tasks/lessons.md`.
- [x] Inspect the render loop and enemy drawing path.
- [x] Cache detailed enemy silhouettes into offscreen canvases and draw cached sprites in the live frame loop.
- [x] Keep dynamic overlays separate from cached art.
- [x] Run parse/self-containment checks and browser performance verification.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Sprite caching implemented
- [x] Verification completed

## Review

- The detailed enemy line art was a likely cause of sluggishness because every live enemy replayed the full canvas path construction every frame.
- Live enemy silhouettes now render once into high-resolution cached canvases and are reused with `drawImage`.
- Slow/jam overlays, health bars, burrow offsets, boss sizing, child sizing, and the 10-enemy catalog remain dynamic.
- HUD-heavy panels now avoid repeated DOM rebuilds when their content has not changed.
- Browser benchmark: cached enemy drawing was about 2.06x faster than raw path drawing in the micro-benchmark.
- Verification screenshot: `/tmp/mother-os-perf-game.png`.
