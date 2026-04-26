# Lessons

- For image-inspired game art, do not stop at matching palette and layout. Reproduce the reference's fidelity signals too: silhouette complexity, internal linework, panel seams, grunge/noise, layered glows, and distinct readable shapes for each unit/tower before calling the UI polished.
- If a user says the assets still do not match the reference, scale the artwork up where it is meant to be inspected and add reference-specific silhouette traits, not just extra generic linework.
- When a user calls generated game art "messed up," reduce clever linework and rebuild from clean, recognizable silhouettes first; add detail only after the base shape reads correctly at live gameplay scale.
- For canvas games, do not let the canvas element stretch away from the world aspect ratio; that creates letterboxing inside the render surface even when the board itself is correctly scaled.
- For tower defense controls, never leave a build tool permanently armed after placement; explicit selection, easy deselection, and refundable misclick recovery are core mobile UX safeguards.
- In resource-driven games, current currency, health, and score must live near primary controls with stronger visual hierarchy than decorative telemetry such as clocks.
- For wave-based games, avoid mechanical boss intervals when the player wants world flavor; express cadence through diegetic countdown labels and boss-zone anticipation.
- When reworking enemies from a lineup reference, mirror the full catalog and distinctive silhouette role of each unit, not just the enemies already present in the current implementation.
- High-detail canvas line art should be cached into reusable sprites before it appears in quantity; preserving quality means avoiding repeated path construction, not simplifying silhouettes.
- After each significant completed checkpoint, commit and push the verified changes instead of leaving them uncommitted for a later prompt.
- When caching high-DPI canvas sprites, scale the offscreen center translation by the same pixel ratio; otherwise the cached art shifts away from simulation-space overlays like health bars.
- Tower placement previews share the same performance budget as live combat; cache high-fidelity tower schematic art for both placed towers and ghost previews so pointer movement only redraws lightweight overlays.
- For tower-defense balance passes, compare enemy effective health, spawn density, rewards, and upgrade efficiency together; raising enemy count without flattening reward growth can accidentally preserve the same dominant upgrade path.
- When a user says enemy art is still "bleh" compared to a reference, another pass should change the drawing vocabulary, not just add more lines: stronger outer silhouettes, layered phosphor strokes, unique anatomy, interior panel logic, and deliberate grunge must all improve together.
- Boss assets should not be implemented as scaled regular enemies when the reference provides distinct boss silhouettes; give bosses their own definitions, selection cadence, preview path, cache key, and high-detail drawing function.
- Boss quality passes should make every interior stroke explain anatomy or construction; when lines feel arbitrary, rebuild the silhouette with named structures first, then add cracks, grime, and glow as secondary detail.
