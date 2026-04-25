# Mother OS Enemy Fidelity Pass

## Spec

- [x] Rework enemy line art toward the attached design image's stronger schematic fidelity.
- [x] Improve silhouettes, internal plating, legs/tendrils, grunge scratches, and phosphor line layering.
- [x] Preserve the cached enemy sprite pipeline and runtime performance.
- [x] Verify both the large tactical-control preview and in-map enemy sprites.
- [x] Keep the game single-file and dependency-free.
- [x] Commit and push the completed checkpoint.

## Plan

- [x] Inspect current enemy drawing helpers and cache path.
- [x] Add shared line-art helpers for double strokes, hatch marks, seams, and antenna/leg structure.
- [x] Enhance all non-boss enemy silhouettes with reference-specific detail.
- [x] Run JavaScript parse, dependency scan, whitespace check, and browser screenshot verification.
- [x] Commit and push to `main`.

## Progress

- [x] Context gathered
- [x] Plan written
- [x] Enemy art enhanced
- [x] Verification completed

## Review

- Added shared enemy line-art helpers for hatch shading, rivets, cables, antennae, grunge scratches, and per-model distress marks.
- Enhanced all ten non-boss enemies through the shared cached `drawReferenceEnemy` path, including stronger beetle, slime, and juggernaut silhouettes after reviewing the gallery.
- Increased cached enemy sprite pixel density and stroke/glow strength so the added detail remains readable in the live map.
- Browser verification screenshots: `/tmp/mother-os-enemy-fidelity-game-2.png` for gameplay scale and `/tmp/mother-os-enemy-gallery-2.png` for the full enemy catalog.
