# Lessons

- For image-inspired game art, do not stop at matching palette and layout. Reproduce the reference's fidelity signals too: silhouette complexity, internal linework, panel seams, grunge/noise, layered glows, and distinct readable shapes for each unit/tower before calling the UI polished.
- If a user says the assets still do not match the reference, scale the artwork up where it is meant to be inspected and add reference-specific silhouette traits, not just extra generic linework.
- When a user calls generated game art "messed up," reduce clever linework and rebuild from clean, recognizable silhouettes first; add detail only after the base shape reads correctly at live gameplay scale.
- For canvas games, do not let the canvas element stretch away from the world aspect ratio; that creates letterboxing inside the render surface even when the board itself is correctly scaled.
- For tower defense controls, never leave a build tool permanently armed after placement; explicit selection, easy deselection, and refundable misclick recovery are core mobile UX safeguards.
- In resource-driven games, current currency, health, and score must live near primary controls with stronger visual hierarchy than decorative telemetry such as clocks.
- For wave-based games, avoid mechanical boss intervals when the player wants world flavor; express cadence through diegetic countdown labels and boss-zone anticipation.
