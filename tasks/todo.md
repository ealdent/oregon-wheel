# Mother OS UI Refinement

## Spec

- [ ] Make start/end path markers much less intrusive or move them off the active track so spawning enemies remain visible.
- [ ] Replace the right-side enemy list with more useful gameplay information.
- [ ] Update left-side tower type cards to show tower silhouettes instead of numeric listing codes.
- [ ] Preserve the no-build, no-dependency single-file game requirement.
- [ ] Verify the game still boots, renders, places/upgrades towers, starts a wave, and has no console issues.
- [ ] Commit and push the fix to `main`.

## Plan

- [x] Confirm branch/worktree state and review current UI implementation.
- [x] Patch `arcade/mother-os-defense/index.html`:
  - move gate markers away from the path and reduce their opacity
  - replace enemy dossier list with wave/tactics/modules panels
  - add canvas-rendered mini tower previews to shop cards
- [x] Run static parse and self-containment checks.
- [x] Run headless Chrome smoke test with screenshots.
- [x] Commit and push `main`.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] UI patched
- [x] Verification completed
- [x] Committed and pushed

## Review

- Moved entry/core gate labels off the active path and reduced their visual weight so enemies remain visible at spawn and exit.
- Replaced the right-side enemy dossier list with tactical control modules for next wave forecast, build doctrine, field systems, and selected tower inspection.
- Reworked tower shop cards to use rendered tower silhouettes instead of numeric codes.
- Verified the updated file still parses as inline JavaScript and remains self-contained with no external dependencies.
- Verified in headless Chrome from `file://`:
  - five tower cards render with five previews
  - enemy dossier text is absent from the right rail
  - right rail title is `Tactical Control`
  - tower placement and upgrade still work
  - wave 1 starts and resolves back to planning
  - auto-continue still activates
  - mobile layout still puts the playfield first and keeps five tower previews
  - no console errors or warnings were emitted
