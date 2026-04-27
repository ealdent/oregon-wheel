# Mother OS Campaign Map Planning

## Campaign Facility Frame Proportion Pass

### Spec

- [x] Reduce facility building glyph scale so art has breathing room inside each campaign node.
- [x] Make facility node frames taller and less squat, with a more vertical rectangular bracket aspect.
- [x] Strengthen the side bracket effect while keeping the top and bottom as thinner, partial rails.
- [x] Preserve cached facility glyph rendering and static no-build loading.
- [x] Verify with static checks and browser campaign-map screenshots before committing and pushing.

### Plan

- [x] Adjust campaign node dimensions and grid spacing to support taller facility frames without crowding.
- [x] Reduce cached facility glyph canvas size and reposition glyphs away from the text block.
- [x] Rework `drawCampaignNodeBracketFrame` so side brackets dominate and top/bottom lines are partial and subtle.
- [x] Run `node --check`, `git diff --check`, browser screenshot QA, then commit and push.

### Review

- Campaign facility nodes are now taller and less squat: `166x128`, with increased vertical grid spacing.
- Facility glyphs render smaller in-frame with more top/text breathing room while preserving cached sprite reuse.
- The node frame now emphasizes heavier vertical side brackets and diagonal corners, with only short partial top/bottom rails and a much fainter continuous outline.
- Browser QA verified all 12 facility types still render clearly and facility glyph cache size stays stable while panning.
- Verified with `node --check` across all Mother OS scripts and `git diff --check`.

## Campaign Facility Building Fidelity Pass

### Spec

- [x] Add 12 campaign facility classifications inspired by the uploaded building sheet, with labels that fit the Mother OS theme.
- [x] Make campaign node facility art substantially higher fidelity, with distinct silhouettes for every classification.
- [x] Cache high-detail facility building glyphs on offscreen canvases so map panning stays responsive.
- [x] Preserve existing campaign saves by keeping old type keys valid and deterministic.
- [x] Preserve static no-build/no-server loading.
- [x] Verify with static checks and browser campaign-map screenshots before committing and pushing.

### Plan

- [x] Expand `facilityTypes` and campaign name pools to 12 classifications while keeping existing keys compatible.
- [x] Add a compact facility glyph cache modeled after the terrain glyph cache.
- [x] Replace campaign node schematic rendering with cached high-detail line-art building glyphs.
- [x] Tune node spacing/label placement only as needed to make the richer art readable.
- [x] Run `node --check`, `git diff --check`, screenshot QA, then commit and push.

### Review

- Expanded campaign facility classifications to 12: Tokamak Reactor, Radar Array, Annex Complex, Bio Research Facility, Mining Operation, Power Plant, Communications Hub, Hydroponics Farm, Vehicle Depot, Satellite Uplink, Storage Facility, and Command Center.
- Existing save keys remain valid; older `cargo`, `foundry`, and `cryo` nodes now map to the new storage, mining, and bio-research visual classifications.
- Campaign nodes now render cached high-detail building glyph sprites with distinct silhouettes, linework, windows, domes, towers, antennas, silos, doors, and grime.
- Node art viewport and label placement were adjusted so the richer facility drawings stay readable at map scale.
- Verified 12 visible facility types in a browser screenshot, confirmed cache reuse during panning, and passed `node --check` plus `git diff --check`.

## Campaign Topography Fidelity Pass

### Spec

- [x] Brighten campaign topography slightly without letting terrain overpower routes or facilities.
- [x] Replace weak mountain symbols with higher-fidelity mountain and plateau glyphs inspired by the reference sheet.
- [x] Add stronger forests, rivers/streams, ridgelines, water, marsh, boulder, coastline, and terrain-context symbols from the reference vocabulary.
- [x] Keep terrain generated in persistent world space so panning away and back preserves the same symbols.
- [x] Preserve static browser loading and avoid per-frame procedural drawing work where possible.
- [x] Verify with static checks and browser screenshots before committing and pushing.

### Plan

- [x] Bump the campaign terrain art version so old low-fidelity terrain chunks regenerate while campaign progress remains intact.
- [x] Expand terrain chunk generation to store compact seeded glyph metadata for multiple topography families.
- [x] Replace immediate generic mountain/forest/ridge drawing with reusable high-detail canvas glyph renderers and careful culling.
- [x] Tune topography brightness and layer ordering against the campaign map.
- [x] Run `node --check`, `git diff --check`, browser screenshot QA, then commit and push.

### Review

- Terrain art version is now `3`, so previous low-fidelity chunks regenerate while campaign graph progress remains intact.
- Terrain chunks now persist compact world-space glyph metadata for mountains, plateaus, forests, rivers, waters, marshes, rocks, coasts, ridgelines, structures, and survey ticks.
- Campaign rendering now caches high-detail terrain glyph canvases, keeping symbol quality higher without rebuilding every line on each frame.
- Browser QA verified richer topography visibility and compact mountain silhouettes, plus panning away and back preserved original terrain chunk data.
- Verified with `node --check` across all Mother OS scripts and `git diff --check`.

## Facility Path Variation

### Spec

- [x] Replace the repeated central switchback generator with multiple seeded route archetypes.
- [x] Make easier/earlier facilities bias toward longer, more winding tracks.
- [x] Make harder/later facilities bias toward shorter, straighter tracks with fewer turns.
- [x] Persist generated facility layouts on campaign nodes so a facility keeps the same route when revisited.
- [x] Verify route length/turn metrics across generated facilities plus browser screenshot QA.

### Plan

- [x] Add a facility layout version and node-level layout cache.
- [x] Add difficulty scoring from facility index, sector count, and facility type.
- [x] Implement varied path archetypes: wide serpentine, perimeter loop, spiral, diagonal weave, offset switchback, dogleg, and straight chicane.
- [x] Select candidate paths by target length and turn count so easier routes are long/winding and harder routes are short/direct.
- [x] Run static checks, browser metric checks, and screenshot verification.
- [x] Commit and push the checkpoint.

### Review

- Added versioned facility layout caching on campaign nodes so newly generated routes persist when a facility is revisited.
- Added seeded path archetypes: wide serpentine, perimeter loop, spiral, diagonal weave, offset switchback, direct dogleg, and straight chicane.
- Path selection now uses facility index, sector count, and facility type to bias easier facilities toward longer winding routes and harder facilities toward shorter straighter routes.
- Browser metric sampling showed easy routes averaging about 2423m with 7 turns and hard routes averaging about 874m with 1 turn.
- Verified with `node --check`, `git diff --check`, and a campaign facility screenshot.

## Campaign Terrain Persistence

### Spec

- [x] Generate campaign terrain once into persistent world-space data instead of re-rolling it from the current viewport.
- [x] Keep rivers, forests, mountains, ridges, and survey ticks fixed as the player pans away and comes back.
- [x] Generate additional terrain only when newly visible world chunks are encountered, then persist those chunks in `localStorage`.
- [x] Avoid per-frame random terrain construction and keep the existing static browser/no-build architecture.
- [x] Verify by panning the campaign map and comparing before/after terrain positions, plus static checks and screenshot QA.

### Plan

- [x] Add a normalized `campaign.terrain` model with versioned chunk storage.
- [x] Implement deterministic chunk generation keyed by campaign seed and chunk coordinates, storing all feature geometry.
- [x] Update campaign terrain rendering to draw stored features from visible chunks through `campaignWorldToScreen`.
- [x] Add a lightweight visible-chunk cache so unchanged pans do not repeatedly flatten feature lists.
- [x] Run `node --check`, `git diff --check`, and browser screenshot/pan stability verification.
- [x] Commit and push the checkpoint.

### Review

- Campaign saves now include `terrain` chunk storage, normalized for existing localStorage saves.
- Terrain chunks are generated deterministically from campaign seed plus chunk coordinates, then persisted; rivers, mountains, forests, ridges, and ticks store fixed world-space geometry.
- Rendering now draws persisted features through the current pan and uses a visible-chunk feature cache instead of rebuilding random terrain every frame.
- Verified that panning away generated new chunks, returning to the original pan produced an identical canvas image, and the originally visible chunks remained unchanged.
- Verified with `node --check`, `git diff --check`, and a campaign screenshot.

## Campaign Terrain Visibility Balance

### Spec

- [x] Increase mountain and forest visibility enough to restore the immersive world-map feel.
- [x] Reduce river dominance so rivers remain present without overpowering facilities and topography.
- [x] Keep terrain marks intentional and cartographic, avoiding the previous circular squiggle issue.
- [x] Preserve full-width campaign map scaling and static browser loading.
- [x] Verify with static checks and a campaign screenshot before committing.

### Plan

- [x] Tune terrain layer alpha, stroke widths, density, and drawing order in the campaign renderer.
- [x] Add slightly stronger mountain ridge details and forest silhouettes without making them noisy.
- [x] Lower river glow/width while keeping a readable blue route through the landscape.
- [x] Run `node --check`, `git diff --check`, and capture a campaign-map screenshot.
- [x] Commit and push the checkpoint.

### Review

- Mountains now render with stronger crest lines, secondary facet strokes, and brighter ridge contours.
- Forest clusters have higher outline/fill contrast and a faint ground contour so they read as localized terrain features.
- Rivers now use a narrower, lower-alpha glow and thinner core so they support the map without dominating it.
- Verified with `node --check`, `git diff --check`, and a forced campaign-state Chrome screenshot.

## Campaign Map Bracket and Terrain Cleanup

### Spec

- [x] Convert facility nodes away from filled plaques and heavy top rails into low-fill bracketed schematic frames.
- [x] Keep every facility schematic fully contained inside the node frame with connected, intentional linework.
- [x] Remove terrain contour marks that read as circular squiggles and replace them with subtler cartographic detail.
- [x] Preserve the full-width, non-stretched campaign viewport and static no-build architecture.
- [x] Verify with static JS checks and a campaign screenshot before committing.

### Plan

- [x] Reduce node fill, remove the dominant upper border, and draw corner/side brackets plus subtle scan hatching.
- [x] Scale and constrain campaign facility icons, rebuilding any overshooting icon details inside a fixed schematic viewport.
- [x] Replace circular contour clusters with faint ridge lines, forest pockets, and sparse survey ticks.
- [x] Run `node --check`, `git diff --check`, and capture a campaign map screenshot.
- [x] Commit and push the completed checkpoint.

### Review

- Facility nodes now use a dark route mask, very faint type tint, corner brackets, side brackets, and sparse scan hatching instead of a colored plaque or dominant top rail.
- Facility schematics are clipped inside a fixed viewport and redrawn with connected bases, supports, masts, gantries, and equipment details so they do not exceed the node frame.
- Circular contour clusters were replaced with subtle ridge fields and survey ticks; mountains and forests were lowered in density/opacity.
- Verified with `node --check`, `git diff --check`, and a forced campaign-state Chrome screenshot at 2048px width.

## Campaign Map Fidelity Pass

### Spec

- [x] Remove campaign-map visual stretching while keeping the grid full-width.
- [x] Rework facility nodes from filled badges into lighter schematic frames with top/bottom rails and side/corner structure like the reference.
- [x] Increase facility schematic fidelity with more internal construction detail and deterministic variation.
- [x] Make landscape/topography subtler but more intricate: finer mountain, forest, river, contour, and scanline detail.
- [x] Preserve the static no-build/no-server architecture and verify with screenshot-based browser QA.

### Plan

- [x] Replace campaign non-uniform scaling with a wider logical viewport and matching pointer coordinate mapping.
- [x] Update campaign render bounds helpers so grid, terrain, frame, legend, routes, and selection panel use the dynamic map width.
- [x] Refactor node rendering into transparent framed schematic modules with richer icons and reduced fill opacity.
- [x] Tune terrain alpha/detail toward the uploaded reference image.
- [x] Run JS/static checks and capture campaign screenshots before committing.

### Review

- Campaign rendering now uses uniform scaling plus a dynamic logical map width, so the grid fills the horizontal canvas without stretching facility art or text.
- Campaign input, map centering, legend, selection panel, frame, and terrain now use the same dynamic campaign viewport.
- Facility nodes now use lower-opacity fills, top/bottom rails, side rails, internal hatching, and richer schematic details.
- Terrain now uses subtler rivers plus denser mountains, forests, contours, and scanline grime.
- Verified with `node --check`, `git diff --check`, and a forced campaign-state Chrome screenshot at 2048px width.

## Current Visual Pass

### Spec

- [x] Make the campaign grid fill the full horizontal canvas instead of sitting inside a letterboxed board area.
- [x] Add facility schematic graphics inside campaign nodes, with type-specific silhouettes and deterministic per-facility variation.
- [x] Add localized topography to the campaign map: mountain ranges, forest clusters, rivers, and terrain contour detail.
- [x] Keep the static no-build/no-server architecture and verify with browser screenshots.

### Plan

- [x] Split campaign rendering/input scaling from combat rendering so campaign mode can fill the whole canvas.
- [x] Add deterministic terrain rendering anchored to campaign world coordinates and responsive to map pan.
- [x] Replace plain campaign node boxes with larger nodes that include facility art, status text, and existing route states.
- [x] Verify desktop campaign screenshot and static script checks.
- [x] Commit and push the completed checkpoint.

### Review

- Campaign mode now renders with full-canvas scaling instead of combat-board letterboxing, with pointer mapping adjusted to match.
- Facility nodes now include deterministic schematic artwork by type: tokamak, cargo, foundry, cryo, and radar variants.
- The campaign map background now includes seeded rivers, mountain ranges, forest clusters, and contour rings anchored to campaign pan.
- Verified with `node --check`, `git diff --check`, and a 2048px-wide headless Chrome campaign screenshot.

## Spec

- [x] Add a persistent `localStorage` campaign map made of facility nodes connected in a fog-of-war graph.
- [x] Keep the first facility as the starting run; show the campaign map only after the first facility is secured, then make it available between zones/facilities.
- [x] Treat every facility as a fresh self-contained run with its own randomized map/path, sector countdown, and boss sector.
- [x] Allow players to enter any visible uncleared facility connected to a secured facility; the starting facility is the only exception.
- [x] Allow exiting a facility mid-run; completed sectors are retained, but partial-sector progress resets to the last completed sector state.
- [x] On failure or abandoned partial progress, retry the same facility with the same generated map and sector count, resetting to the saved facility checkpoint.
- [x] At facility completion, mark it secured, turn auto-advance off, reveal connected next options, and show a dismissible summary dialog before returning to the map.
- [x] Generate 0-4 exits per secured facility, heavily weighting 2, then 3, with 0 and 4 rare; only allow 0 when at least two sibling nodes have non-zero exits.
- [x] Make facility type affect both visual treatment and balance.
- [x] At the campaign map level, replace the top-center gameplay controls and top-right resource feed with campaign-relevant controls and status instead of leaving wave/tower controls visible.
- [x] Preserve the static no-build/no-server architecture.

## Plan

- [x] Define campaign data model: nodes, edges, visibility, secured state, facility seed, facility type, sector count, current sector, and checkpoint stats.
- [x] Add deterministic seeded generation for facility graph expansion, facility names/types, exit counts, procedural TD paths, build pads, and visual variants.
- [x] Split current game state into campaign state and facility-run state while keeping the current renderer modules static and script-order based.
- [x] Implement campaign map screen with pan/scroll, completed/current/available/unknown styling, connector lines, legend, and node selection.
- [x] Add campaign-specific top chrome showing map actions, selected facility status, secured count, visible threats, campaign score/progress, and session controls.
- [x] Add facility start/resume/exit flow, including checkpoint restore and partial-sector reset.
- [x] Generate per-facility TD maps and visuals, then adapt path/build validation/rendering to use the active facility layout instead of global fixed path constants.
- [x] Add facility balance modifiers by facility type and apply them to enemy composition, health, rewards, timing, or sector count.
- [x] Add facility completion summary modal with kills, enemies leaked, credits earned/spent, score gained, towers built/sold/upgraded, sectors cleared, and boss defeated.
- [x] Ensure auto-advance turns off when a facility completes and starts off for every new/resumed facility.
- [x] Verify localStorage persistence, first-run behavior, map reveal rules, facility retry/reset behavior, procedural map rendering, boss completion summary, and static file loading.
- [ ] Commit and push after implementation checkpoints.

## Progress

- [x] Clarifying questions answered
- [x] Planning artifact written
- [x] Plan confirmed
- [x] Implementation started
- [x] Verification completed

## Review

- Added static `campaign.js` with localStorage persistence, seeded graph generation, facility checkpoints, summaries, and facility entry/exit helpers.
- Updated the existing static modules to support campaign vs facility modes, procedural facility paths, campaign-specific top chrome, summary modal, and facility type balance/visual modifiers.
- Verified with `node --check` across every game script.
- Verified in headless Chrome from `file://` that the game loads and renders the facility battlefield.
- Verified via Chrome DevTools runtime flow that first-facility completion unlocks the map, reveals three connected facilities and unknown routes, displays a summary, starts new facilities with fresh resources and auto off, resets abandoned partial progress, and restores completed-sector checkpoints.
