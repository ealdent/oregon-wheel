# Mother OS Control UX Pass

## Spec

- [x] Let Escape deselect any selected tower or queued build design.
- [x] Let mobile users deselect safely by tapping the same tower again or tapping outside the map.
- [x] Prevent accidental mobile tower placement from a permanently queued tower.
- [x] Let a just-placed, not-upgraded tower sell for full cost between waves.
- [x] Move wave and tower action controls higher into the top middle header.
- [x] Increase the Mother OS title phosphor glow intensity.
- [x] Preserve gameplay, tower/enemy art, and the single-file/no-dependency requirement.
- [x] Verify with browser interaction checks and screenshots.

## Plan

- [x] Record the UX correction pattern in `tasks/lessons.md`.
- [x] Inspect current selection, placement, sell, and control rendering code.
- [x] Implement nullable build selection, deselect gestures, and full refund eligibility.
- [x] Move command controls into the top center header and strengthen the title style.
- [x] Run parse/self-containment checks and browser smoke tests.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] UX changes implemented
- [x] Verification completed

## Review

- Escape now clears any selected tower or queued build cursor.
- Clicking the same tower again, clicking empty map space while a tower is selected, or tapping outside the map clears selection instead of leaving the user in an accidental placement state.
- New tower placements are no longer sticky: after placement, the build cursor clears and the placed tower is selected.
- Planning-phase towers that have not been upgraded are refundable for their full placement cost; starting a wave or upgrading the tower removes that full-refund eligibility.
- Wave, upgrade, targeting, sell, pause, speed, auto, and sound controls now live in the top center header; the old schematic label was removed.
- Browser verification passed with desktop and mobile screenshots at `/tmp/mother-os-controls-desktop.png` and `/tmp/mother-os-controls-mobile.png`.
