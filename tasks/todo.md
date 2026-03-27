# Oregon Wheel Revamp

## Spec

- [x] Audit the current `depths/oregon-wheel.html` implementation and preserve only the parts worth keeping.
- [x] Replace the old journey list with the full current roster from the provided screenshots.
- [x] Redesign the page to better match the visual language of `depths/` while making the wheel feel more polished and intentional.
- [x] Improve the spin experience with stronger motion, clearer feedback, and a more prominent winner state.
- [x] Keep the page as a self-contained HTML file with responsive behavior on desktop and mobile.
- [x] Verify the result with a browser-level render check and document outcomes here.

## Progress

- [x] Context gathered
- [x] Task plan written
- [x] UI and animation revamp implemented
- [x] Verification completed

## Review

- Rebuilt `depths/oregon-wheel.html` as a more dramatic single-page experience with a bioluminescent stage, donut-style wheel, roster control deck, and stronger winner presentation.
- Expanded the wheel data to the full 27-journey roster from the supplied screenshots, including `Planet Penguin Episode 2`, `Potato's Day Out`, `The Strange Case`, `Voyage to Tobago`, `The Suburbs`, and `Trail Of Horror`.
- Verified desktop and mobile rendering with headless Chromium screenshots.
- Verified spin behavior with a headless Chromium automation pass that clicked `#spinBtn` and confirmed:
  - `#activeState` becomes `Winner locked`
  - `#statSpins` increments to `1`
  - `#winnerLabel` becomes `Wheel selected`
  - `#spinBtn` returns to `Spin the wheel`
