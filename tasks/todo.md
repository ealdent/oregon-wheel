# Mother OS Enemy Healthbar Alignment Fix

## Spec

- [x] Fix health bars appearing offset from cached enemy art.
- [x] Preserve enemy sprite caching and line-art quality.
- [x] Verify active enemies align visually with their health bars.
- [x] Commit and push the completed checkpoint.

## Plan

- [x] Identify the cache transform issue.
- [x] Center high-DPI cached enemy sprites correctly in their offscreen canvases.
- [x] Run static checks and browser screenshot verification.
- [x] Commit and push to `main`.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Alignment fixed
- [x] Verification completed

## Review

- Fixed the cached enemy sprite transform so the offscreen canvas center translation is scaled with the sprite pixel ratio.
- Verified with JavaScript parse check, dependency scan, whitespace check, and a browser screenshot pass at `/tmp/mother-os-healthbar-align.png`.
