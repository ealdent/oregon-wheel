# Mother OS Art Fidelity Pass

## Spec

- [x] Increase tower design fidelity so shop previews and placed towers feel closer to the reference image's detailed schematic drawings.
- [x] Increase enemy silhouette fidelity so units look more like distinct dossier creatures, not simple placeholders.
- [x] Add a grungier CRT/industrial feel without obscuring gameplay readability.
- [x] Preserve the single-file, no-build, no-dependency requirement.
- [x] Verify the game still boots, places/upgrades towers, starts a wave, and has no console issues.

## Plan

- [x] Record the correction pattern in `tasks/lessons.md`.
- [x] Add reusable schematic drawing detail for towers and tower previews.
- [x] Add layered enemy body details, appendages, armor plates, and phase/electrical marks.
- [x] Add canvas-level grit, scratches, and terminal wear that stays subtle over the path.
- [x] Run static parse/self-containment checks and a headless Chrome smoke test with screenshots.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Art pass implemented
- [x] Verification completed

## Review

- Rebuilt tower rendering around shared high-detail schematic routines used by both placed towers and shop previews.
- Added more reference-like tower details: layered bases, bolts, antennae, coil stacks, faceted prism lines, mine disc spokes, and jammer wave arcs.
- Increased enemy visual scale and added creature-specific linework: crawler legs/sensors, beetle shell plates/tusks, slime bubbles/drips, worm segments/mandibles, wisp electrical arcs, juggernaut armor plates, and phantom cloak/rib lines.
- Added subtle CRT grime, scratches, terminal wear, and field noise without hiding the path.
- Verified inline JavaScript parsing and self-containment.
- Verified in headless Chrome from `file://`:
  - enemy art capture shows active enemies on wave 1
  - five tower previews render
  - tower placement and upgrade still work
  - wave 1 starts and resolves back to planning
  - mobile layout still puts the playfield first
  - no console errors or warnings were emitted
