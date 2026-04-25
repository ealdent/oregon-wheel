# Mother OS Reference Fidelity Pass

## Spec

- [x] Make tower previews and placed towers closer to the reference's large, detailed dossier art.
- [x] Make enemies closer to the reference's distinct creature silhouettes and schematic anatomy.
- [x] Keep the right rail tactical, not a full enemy list.
- [x] Preserve gameplay readability and the single-file/no-dependency requirement.
- [x] Verify in browser with screenshots.

## Plan

- [x] Record the second correction pattern in `tasks/lessons.md`.
- [x] Enlarge tower card previews and add reference-specific tower machinery detail.
- [x] Add underglow/outline passes and stronger enemy silhouette detail.
- [x] Run static parse/self-containment checks and browser smoke screenshots.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Art pass implemented
- [x] Verification completed

## Review

- Enlarged the left-rail tower design canvases and added more reference-like machinery: layered bases, rivets, braces, segmented coils, crystal facets, mine hatches, jammer arcs, and schematic registration marks.
- Reworked enemy rendering with stronger silhouettes, underglow, shell plates, antennae, legs, armor seams, broken rings, scan bars, and high-fidelity live crawler readability.
- Added a single large tactical threat-signature canvas to the right rail so enemy art is inspectable without restoring the rejected full enemy list.
- Verified `arcade/mother-os-defense/index.html` parses, has no external dependency references, places/upgrades a tower, starts wave 1, shows active hostiles, and has no mobile horizontal overflow.
- Screenshots captured: `/tmp/mother-os-final-fidelity-desktop.png`, `/tmp/mother-os-final-fidelity-combat.png`, `/tmp/mother-os-final-fidelity-mobile.png`.
