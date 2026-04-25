# Mother OS Map Spacing Cleanup

## Spec

- [x] Remove the wasted vertical space above and below the map.
- [x] Preserve the tower and enemy art changes already in progress.
- [x] Keep the controls and tactical panels usable on desktop and mobile.
- [x] Preserve the single-file/no-dependency requirement.
- [x] Verify with browser screenshots.

## Plan

- [x] Record the layout correction pattern in `tasks/lessons.md`.
- [x] Change the field canvas/shell to honor the board aspect ratio instead of stretching vertically.
- [x] Run parse/self-containment checks and browser screenshots for desktop and mobile.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Layout implemented
- [x] Verification completed

## Review

- Changed the field panel and canvas shell to stay at the board's 1000:680 aspect ratio on desktop instead of stretching vertically.
- Capped the side panels to the field panel height on desktop so the status strip moves up directly below the map row.
- Added a resize-time CSS variable to keep side panel height synchronized with the actual field panel height.
- Verified inline JavaScript parsing, no external dependency references, desktop field ratio `1.4706`, status gap `10px`, wave 1 combat start with active enemies, and no mobile horizontal overflow.
- Screenshots captured: `/tmp/mother-os-spacing2-desktop.png`, `/tmp/mother-os-spacing-final-combat.png`, `/tmp/mother-os-spacing2-mobile.png`.
