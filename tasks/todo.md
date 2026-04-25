# Mother OS Static Modular Refactor

## Spec

- [ ] Split the single `index.html` into static CSS and ordered classic JavaScript files.
- [ ] Avoid build tools, ES modules, package dependencies, or server requirements.
- [ ] Keep the game loadable from a static website and from direct file access where browser policy allows classic scripts.
- [ ] Preserve existing gameplay, boss/enemy/tower rendering, input, audio, UI, and wave behavior.
- [ ] Keep script ordering explicit and understandable for future changes.
- [ ] Verify parsing, dependency scan, file load, gameplay boot, and boss sector rendering.
- [x] Commit and push the completed checkpoint.

## Plan

- [x] Confirm current working tree and relevant game architecture.
- [x] Extract CSS to `arcade/mother-os-defense/styles.css`.
- [x] Split JavaScript into static domain files under `arcade/mother-os-defense/js/`.
- [x] Replace inline `<style>` and `<script>` blocks with static includes.
- [x] Run syntax, dependency, file-load, and browser smoke checks.
- [x] Commit and push to `main`.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] CSS extracted
- [x] JavaScript split
- [x] Verification completed

## Review

- Extracted the Mother OS CSS into `arcade/mother-os-defense/styles.css`.
- Split the previous inline JavaScript into ordered classic scripts under `arcade/mother-os-defense/js/` with no modules, imports, build steps, or server requirements.
- Verified each JavaScript file parses, the game directory has no external dependency URLs or module imports, whitespace is clean, and the extracted CSS/JS reconstructs the previous inline implementation.
- Browser verification loaded `arcade/mother-os-defense/index.html` directly through `file://`, rendered the game, populated tower cards, and started combat.
- Boss-sector browser verification used a temporary static copy to force Sector 1 and confirmed the Hive Mother preview and live cached boss sprite still render.
- Browser verification screenshots: `/tmp/mother-os-static-refactor-file-load.png` and `/tmp/mother-os-static-refactor-boss-sector.png`.
