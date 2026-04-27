"use strict";

const terrainGlyphCache = new Map();
const TERRAIN_GLYPH_PIXEL_SCALE = 2;
const TERRAIN_GLYPH_CACHE_LIMIT = 420;
const facilityGlyphCache = new Map();
const FACILITY_GLYPH_PIXEL_SCALE = 2;
const FACILITY_GLYPH_CACHE_LIMIT = 192;

function render() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, view.cssWidth, view.cssHeight);
  ctx.fillStyle = "#000400";
  ctx.fillRect(0, 0, view.cssWidth, view.cssHeight);
  if (state.mode === "campaign") {
    ctx.save();
    const scale = campaignRenderScale();
    ctx.scale(scale, scale);
    drawWorld();
    ctx.restore();
    return;
  }
  ctx.save();
  ctx.translate(view.offsetX, view.offsetY);
  ctx.scale(view.scale, view.scale);
  drawWorld();
  ctx.restore();
}

function drawWorld() {
  if (state.mode === "campaign") {
    drawCampaignMap();
    return;
  }
  drawStaticWorld();
  drawMines();
  drawTowers();
  drawEnemies();
  drawProjectiles();
  drawEffects();
  drawPlacement();
  drawPhaseOverlay();
}

function drawStaticWorld() {
  const layer = getStaticWorldLayer();
  ctx.drawImage(layer, 0, 0, BOARD.width, BOARD.height);
}

function getStaticWorldLayer() {
  if (staticWorldLayer) return staticWorldLayer;
  const canvas = document.createElement("canvas");
  canvas.width = BOARD.width * STATIC_WORLD_PIXEL_SCALE;
  canvas.height = BOARD.height * STATIC_WORLD_PIXEL_SCALE;
  const g = canvas.getContext("2d");
  g.setTransform(STATIC_WORLD_PIXEL_SCALE, 0, 0, STATIC_WORLD_PIXEL_SCALE, 0, 0);
  drawBackground(g);
  drawPath(g);
  staticWorldLayer = canvas;
  return staticWorldLayer;
}

function drawBackground(g = ctx) {
  const type = activeFacilityTypeDef();
  const palette = type.palette || facilityTypes.tokamak.palette;
  g.fillStyle = "#010801";
  g.fillRect(0, 0, BOARD.width, BOARD.height);
  g.save();
  g.lineWidth = 1;
  for (let x = 0; x <= BOARD.width; x += 24) {
    g.strokeStyle = x % 96 === 0 ? palette.major : palette.grid;
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, BOARD.height);
    g.stroke();
  }
  for (let y = 0; y <= BOARD.height; y += 24) {
    g.strokeStyle = y % 96 === 0 ? palette.major : palette.grid;
    g.beginPath();
    g.moveTo(0, y);
    g.lineTo(BOARD.width, y);
    g.stroke();
  }
  for (const block of schematicBlocks) {
    g.strokeStyle = `rgba(89,255,115,${block.alpha})`;
    g.strokeRect(block.x, block.y, block.w, block.h);
    if (block.w > 34) {
      g.beginPath();
      g.moveTo(block.x + 6, block.y + block.h / 2);
      g.lineTo(block.x + block.w - 6, block.y + block.h / 2);
      g.stroke();
    }
  }
  drawGrimeLayer(g);
  g.strokeStyle = "rgba(124,232,255,0.18)";
  g.strokeRect(24, 24, BOARD.width - 48, BOARD.height - 48);
  g.restore();
}

function drawGrimeLayer(g = ctx, width = BOARD.width, height = BOARD.height) {
  g.save();
  g.lineWidth = 1;
  for (const mark of grimeMarks) {
    g.save();
    g.translate(mark.x, mark.y);
    g.rotate(mark.rot);
    g.strokeStyle = `rgba(185,255,189,${mark.a})`;
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(mark.w, mark.h);
    g.stroke();
    if (mark.w > 46) {
      g.strokeStyle = `rgba(0,0,0,${mark.a * 2.2})`;
      g.beginPath();
      g.moveTo(mark.w * 0.18, 2);
      g.lineTo(mark.w * 0.82, 2 + mark.h * 0.45);
      g.stroke();
    }
    g.restore();
  }
  g.fillStyle = "rgba(97,255,126,0.028)";
  for (let y = 3; y < height; y += 9) {
    g.fillRect(0, y, width, 1);
  }
  g.restore();
}

function drawPath(g = ctx) {
  const palette = activeFacilityTypeDef().palette || facilityTypes.tokamak.palette;
  g.save();
  g.lineCap = "round";
  g.lineJoin = "round";
  drawPathStroke(g, BOARD.pathWidth + 24, palette.pathOuter);
  drawPathStroke(g, BOARD.pathWidth + 8, palette.pathGlow);
  drawPathStroke(g, BOARD.pathWidth, palette.pathBody);
  drawPathStroke(g, BOARD.pathWidth - 18, palette.pathCore);
  g.setLineDash([16, 18]);
  drawPathStroke(g, 2, "rgba(185,255,189,0.42)");
  g.setLineDash([]);
  for (const point of pathPoints) {
    g.strokeStyle = "rgba(124,232,255,0.28)";
    g.strokeRect(point.x - 9, point.y - 9, 18, 18);
  }
  drawGate(g, pathPoints[0], true);
  drawGate(g, pathPoints[pathPoints.length - 1], false);
  g.restore();
}

function drawCampaignMap() {
  const campaign = state.campaign;
  if (!campaign) return;
  drawCampaignBackground();
  drawCampaignEdges(campaign);
  drawCampaignUnknownRoutes(campaign);
  for (const node of visibleCampaignNodes(campaign).sort((a, b) => a.index - b.index)) {
    drawCampaignNode(campaign, node);
  }
  drawCampaignLegend(campaign);
  drawCampaignSelectionPanel(campaign);
}

function drawCampaignBackground() {
  const campaign = state.campaign;
  const pan = campaign.pan || { x: 0, y: 0 };
  const mapWidth = campaignViewportWidth();
  const mapHeight = campaignViewportHeight();
  ctx.fillStyle = "#010801";
  ctx.fillRect(0, 0, mapWidth, mapHeight);
  ctx.save();
  ctx.lineWidth = 1;
  const gridStep = 24;
  const gridStartX = -gridStep + ((pan.x % gridStep) + gridStep) % gridStep;
  const gridStartY = -gridStep + ((pan.y % gridStep) + gridStep) % gridStep;
  for (let x = gridStartX; x <= mapWidth + gridStep; x += gridStep) {
    const worldX = Math.round((x - pan.x) / gridStep);
    ctx.strokeStyle = worldX % 4 === 0 ? "rgba(97,255,126,0.14)" : "rgba(97,255,126,0.05)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, mapHeight);
    ctx.stroke();
  }
  for (let y = gridStartY; y <= mapHeight + gridStep; y += gridStep) {
    const worldY = Math.round((y - pan.y) / gridStep);
    ctx.strokeStyle = worldY % 4 === 0 ? "rgba(97,255,126,0.14)" : "rgba(97,255,126,0.05)";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(mapWidth, y);
    ctx.stroke();
  }
  drawCampaignTerrain(campaign);
  drawGrimeLayer(ctx, mapWidth, mapHeight);
  drawCampaignMapFrame(mapWidth, mapHeight);
  ctx.restore();
}

function drawCampaignMapFrame(width, height) {
  ctx.save();
  ctx.strokeStyle = "rgba(124,232,255,0.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, width - 16, height - 16);
  ctx.strokeStyle = "rgba(97,255,126,0.2)";
  ctx.beginPath();
  ctx.moveTo(18, 22);
  ctx.lineTo(width - 18, 22);
  ctx.moveTo(18, height - 22);
  ctx.lineTo(width - 18, height - 22);
  ctx.stroke();
  ctx.strokeStyle = "rgba(97,255,126,0.1)";
  for (const x of [30, width - 30]) {
    ctx.beginPath();
    ctx.moveTo(x, 34);
    ctx.lineTo(x, height - 34);
    ctx.stroke();
  }
  ctx.restore();
}

function campaignWorldToScreen(campaign, x, y) {
  const pan = campaign.pan || { x: 0, y: 0 };
  return { x: x + pan.x, y: y + pan.y };
}

function drawCampaignTerrain(campaign) {
  const terrain = ensureCampaignTerrainForViewport(campaign);
  terrain.coasts.forEach((coast) => drawCampaignCoastFeature(campaign, coast));
  terrain.rivers.forEach((river) => drawCampaignRiverFeature(campaign, river));
  terrain.waters.forEach((water) => drawWaterFeature(campaign, water));
  terrain.plateaus.forEach((plateau) => drawPlateauFeature(campaign, plateau));
  terrain.ridges.forEach((ridge) => drawSurveyRidgeFeature(campaign, ridge));
  terrain.mountains.forEach((mountain) => drawMountainFeature(campaign, mountain));
  terrain.forests.forEach((forest) => drawForestFeature(campaign, forest));
  terrain.rocks.forEach((rocks) => drawRocksFeature(campaign, rocks));
  terrain.marshes.forEach((marsh) => drawMarshFeature(campaign, marsh));
  terrain.structures.forEach((structure) => drawTerrainStructureFeature(campaign, structure));
  terrain.ticks.forEach((ticks) => drawSurveyTickFeature(campaign, ticks));
}

function drawCampaignRiverFeature(campaign, river) {
  const points = (river.points || []).map((point) => campaignWorldToScreen(campaign, point.x, point.y));
  if (points.length < 2 || !screenPolylineNearView(points, 80)) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(124,232,255,0.05)";
  ctx.lineWidth = 7 * (river.width || 1);
  drawTerrainPolyline(points);
  ctx.strokeStyle = "rgba(124,232,255,0.18)";
  ctx.lineWidth = 1.45 * (river.width || 1);
  drawTerrainPolyline(points);
  ctx.strokeStyle = "rgba(181,255,111,0.115)";
  ctx.lineWidth = 0.9;
  ctx.setLineDash([8, 11]);
  drawTerrainPolyline(points);
  ctx.setLineDash([]);
  for (const branch of river.branches || []) {
    const branchPoints = branch.map((point) => campaignWorldToScreen(campaign, point.x, point.y));
    if (branchPoints.length < 2) continue;
    ctx.strokeStyle = "rgba(124,232,255,0.1)";
    ctx.lineWidth = 0.9;
    drawTerrainPolyline(branchPoints);
    ctx.strokeStyle = "rgba(181,255,111,0.075)";
    ctx.lineWidth = 0.65;
    ctx.setLineDash([5, 8]);
    drawTerrainPolyline(branchPoints);
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawTerrainPolyline(points) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else {
      const previous = points[index - 1];
      ctx.quadraticCurveTo((previous.x + point.x) / 2, previous.y, point.x, point.y);
    }
  });
  ctx.stroke();
}

function screenPolylineNearView(points, margin) {
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  return maxX >= -margin && minX <= campaignViewportWidth() + margin && maxY >= -margin && minY <= campaignViewportHeight() + margin;
}

function worldFeatureNearView(campaign, x, y, radius) {
  const point = campaignWorldToScreen(campaign, x, y);
  return point.x >= -radius && point.x <= campaignViewportWidth() + radius && point.y >= -radius && point.y <= campaignViewportHeight() + radius;
}

function drawMountainFeature(campaign, mountain) {
  drawCachedTerrainGlyph(campaign, mountain, "mountain", 190, 116, drawMountainGlyph, 1);
}

function drawPlateauFeature(campaign, plateau) {
  drawCachedTerrainGlyph(campaign, plateau, "plateau", 210, 92, drawPlateauGlyph, 0.92);
}

function drawForestFeature(campaign, forest) {
  drawCachedTerrainGlyph(campaign, forest, "forest", 190, 116, drawForestGlyph, 0.96);
}

function drawWaterFeature(campaign, water) {
  drawCachedTerrainGlyph(campaign, water, "water", 180, 86, drawWaterGlyph, 0.9);
}

function drawMarshFeature(campaign, marsh) {
  drawCachedTerrainGlyph(campaign, marsh, "marsh", 210, 100, drawMarshGlyph, 0.86);
}

function drawRocksFeature(campaign, rocks) {
  drawCachedTerrainGlyph(campaign, rocks, "rocks", 140, 68, drawRocksGlyph, 0.9);
}

function drawTerrainStructureFeature(campaign, structure) {
  drawCachedTerrainGlyph(campaign, structure, "structure", 190, 82, drawStructureGlyph, 0.72);
}

function drawSurveyRidgeFeature(campaign, ridge) {
  if (!worldFeatureNearView(campaign, ridge.x, ridge.y, (ridge.width || 180) + 90)) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (ridge.points) {
    const points = ridge.points.map((point) => campaignWorldToScreen(campaign, point.x, point.y));
    ctx.strokeStyle = "rgba(181,255,111,0.18)";
    ctx.lineWidth = 1.15;
    drawTerrainPolyline(points);
    ctx.strokeStyle = "rgba(97,255,126,0.09)";
    ctx.lineWidth = 0.8;
    ctx.setLineDash([8, 9]);
    drawTerrainPolyline(points.map((point) => ({ x: point.x, y: point.y + 10 })));
    ctx.setLineDash([]);
    const rng = makeCampaignRng(`ridge:${ridge.seed || 1}`);
    for (let i = 0; i < points.length - 1; i += 1) {
      const a = points[i];
      const b = points[i + 1];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.hypot(dx, dy) || 1;
      const nx = -dy / length;
      const ny = dx / length;
      const hatchCount = Math.max(2, Math.floor(length / 26));
      for (let h = 0; h < hatchCount; h += 1) {
        if (rng() < 0.18) continue;
        const t = (h + 0.22 + rng() * 0.34) / hatchCount;
        const x = a.x + dx * t;
        const y = a.y + dy * t;
        const hatch = 7 + rng() * 10;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + nx * hatch, y + ny * hatch);
        ctx.stroke();
      }
    }
  } else {
    for (const lane of ridge.lanes || []) {
      ctx.strokeStyle = "rgba(185,255,189,0.095)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      lane.forEach((lanePoint, index) => {
        const point = campaignWorldToScreen(campaign, lanePoint.x, lanePoint.y);
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawCampaignCoastFeature(campaign, coast) {
  const points = (coast.points || []).map((point) => campaignWorldToScreen(campaign, point.x, point.y));
  if (points.length < 2 || !screenPolylineNearView(points, 120)) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(181,255,111,0.13)";
  ctx.lineWidth = 1.1;
  drawTerrainPolyline(points);
  ctx.strokeStyle = "rgba(124,232,255,0.06)";
  ctx.lineWidth = 4.5;
  drawTerrainPolyline(points.map((point, index) => ({
    x: point.x + (coast.vertical ? 10 + index % 2 * 3 : 0),
    y: point.y + (coast.vertical ? 0 : 10 + index % 2 * 3)
  })));
  ctx.strokeStyle = "rgba(181,255,111,0.09)";
  ctx.lineWidth = 0.7;
  ctx.setLineDash([6, 9]);
  drawTerrainPolyline(points.map((point) => ({ x: point.x + 6, y: point.y + 6 })));
  ctx.restore();
}

function drawSurveyTickFeature(campaign, ticks) {
  if (!worldFeatureNearView(campaign, ticks.x, ticks.y, 120)) return;
  ctx.save();
  ctx.strokeStyle = "rgba(124,232,255,0.065)";
  ctx.lineWidth = 1;
  for (const mark of ticks.marks || []) {
    const a = campaignWorldToScreen(campaign, mark.x, mark.y);
    const b = campaignWorldToScreen(campaign, mark.x2, mark.y2);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    if (mark.cross) {
      const crossX = a.x + (b.x - a.x) * 0.55;
      const crossY = a.y + (b.y - a.y) * 0.55;
      ctx.moveTo(crossX, crossY - 3);
      ctx.lineTo(crossX, crossY + 3);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawCachedTerrainGlyph(campaign, feature, type, baseWidth, baseHeight, drawGlyph, alpha = 1) {
  if (!feature || !Number.isFinite(feature.x) || !Number.isFinite(feature.y)) return;
  const width = Math.max(32, Math.round(feature.width || baseWidth * (feature.scale || 1)));
  const height = Math.max(28, Math.round(feature.height || baseHeight * (feature.scale || 1)));
  const radius = Math.max(width, height) * 0.72 + 70;
  if (!worldFeatureNearView(campaign, feature.x, feature.y, radius)) return;
  const screen = campaignWorldToScreen(campaign, feature.x, feature.y);
  const cacheKey = [
    "topo",
    type,
    feature.seed || 0,
    feature.variant || 0,
    width,
    height,
    feature.count || 0
  ].join(":");
  let sprite = terrainGlyphCache.get(cacheKey);
  if (sprite) {
    terrainGlyphCache.delete(cacheKey);
    terrainGlyphCache.set(cacheKey, sprite);
  } else {
    sprite = createTerrainGlyphSprite(type, feature, width, height, drawGlyph);
    terrainGlyphCache.set(cacheKey, sprite);
    while (terrainGlyphCache.size > TERRAIN_GLYPH_CACHE_LIMIT) {
      terrainGlyphCache.delete(terrainGlyphCache.keys().next().value);
    }
  }
  ctx.save();
  ctx.translate(screen.x, screen.y);
  ctx.rotate(feature.rotation || 0);
  ctx.globalAlpha = alpha;
  ctx.drawImage(sprite.canvas, -sprite.width / 2, -sprite.height / 2, sprite.width, sprite.height);
  ctx.restore();
}

function createTerrainGlyphSprite(type, feature, width, height, drawGlyph) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(width * TERRAIN_GLYPH_PIXEL_SCALE);
  canvas.height = Math.ceil(height * TERRAIN_GLYPH_PIXEL_SCALE);
  const g = canvas.getContext("2d");
  g.setTransform(
    TERRAIN_GLYPH_PIXEL_SCALE,
    0,
    0,
    TERRAIN_GLYPH_PIXEL_SCALE,
    width * TERRAIN_GLYPH_PIXEL_SCALE / 2,
    height * TERRAIN_GLYPH_PIXEL_SCALE / 2
  );
  g.lineCap = "round";
  g.lineJoin = "round";
  g.shadowColor = "rgba(148,255,88,0.23)";
  g.shadowBlur = type === "structure" ? 2 : 5;
  const rng = makeCampaignRng(`${type}:${feature.seed || 1}:${feature.variant || 0}`);
  drawGlyph(g, width, height, feature, rng);
  return { canvas, width, height };
}

function topoStroke(g, alpha, width = 1, color = "181,255,111") {
  g.strokeStyle = `rgba(${color},${alpha})`;
  g.lineWidth = width;
}

function topoFill(g, alpha, color = "97,255,126") {
  g.fillStyle = `rgba(${color},${alpha})`;
}

function drawLocalPolyline(g, points, close = false) {
  if (!points.length) return;
  g.beginPath();
  points.forEach((point, index) => {
    if (index === 0) g.moveTo(point.x, point.y);
    else g.lineTo(point.x, point.y);
  });
  if (close) g.closePath();
  g.stroke();
}

function drawLocalIrregularRing(g, rng, rx, ry, jitter, steps = 34) {
  const points = [];
  const phaseA = rng() * Math.PI * 2;
  const phaseB = rng() * Math.PI * 2;
  for (let i = 0; i < steps; i += 1) {
    const t = i / steps * Math.PI * 2;
    const wobble = 1 + Math.sin(t * 3 + phaseA) * jitter + Math.cos(t * 5 + phaseB) * jitter * 0.55;
    points.push({ x: Math.cos(t) * rx * wobble, y: Math.sin(t) * ry * wobble });
  }
  drawLocalPolyline(g, points, true);
}

function drawMountainGlyph(g, width, height, feature, rng) {
  const variant = feature.variant || 0;
  if (variant === 1 || variant === 5) {
    drawVolcanicMountainGlyph(g, width, height, rng);
    return;
  }
  const baseY = height * 0.28;
  const left = -width * 0.41;
  const right = width * 0.41;
  const peakCount = 2 + (variant % 3);
  const peaks = [];
  for (let i = 0; i < peakCount; i += 1) {
    const t = peakCount === 1 ? 0.5 : i / (peakCount - 1);
    const centerBias = 1 - Math.abs(t - 0.5) * 0.9;
    const x = left + (right - left) * t + (rng() - 0.5) * width * 0.08;
    const y = -height * (0.28 + centerBias * 0.2 + rng() * 0.12);
    peaks.push({ x, y });
  }
  topoFill(g, 0.018);
  g.beginPath();
  g.moveTo(left, baseY);
  peaks.forEach((peak, index) => {
    const shoulder = {
      x: peak.x + (index % 2 === 0 ? -1 : 1) * width * (0.08 + rng() * 0.04),
      y: -height * (0.06 + rng() * 0.08)
    };
    g.lineTo(shoulder.x, shoulder.y);
    g.lineTo(peak.x, peak.y);
    g.lineTo(peak.x + (rng() - 0.5) * width * 0.08, -height * (0.07 + rng() * 0.08));
  });
  g.lineTo(right, baseY);
  g.closePath();
  g.fill();

  for (const [index, peak] of peaks.entries()) {
    const span = width * (0.14 + rng() * 0.08);
    const leftBase = { x: Math.max(left, peak.x - span * (1.1 + rng() * 0.5)), y: baseY + (rng() - 0.5) * height * 0.06 };
    const rightBase = { x: Math.min(right, peak.x + span * (1.1 + rng() * 0.5)), y: baseY + (rng() - 0.5) * height * 0.06 };
    topoStroke(g, 0.34, 1.02);
    drawLocalPolyline(g, [leftBase, peak, rightBase]);
    topoStroke(g, index === 1 ? 0.48 : 0.36, 0.62, "216,255,127");
    drawLocalPolyline(g, [
      { x: peak.x - span * 0.18, y: peak.y + height * 0.06 },
      peak,
      { x: peak.x + span * 0.22, y: peak.y + height * 0.08 }
    ]);
    const baseA = { x: peak.x - span * (0.45 + rng() * 0.38), y: baseY - height * (0.02 + rng() * 0.08) };
    const baseB = { x: peak.x + span * (0.38 + rng() * 0.42), y: baseY - height * (0.03 + rng() * 0.07) };
    topoStroke(g, 0.25, 0.88);
    drawLocalPolyline(g, [peak, baseA]);
    topoStroke(g, 0.18, 0.72);
    drawLocalPolyline(g, [peak, baseB]);
    for (let rib = 0; rib < 3; rib += 1) {
      const side = rib % 2 === 0 ? -1 : 1;
      const startY = peak.y + height * (0.2 + rng() * 0.22);
      const start = { x: peak.x + side * (6 + rng() * span * 0.45), y: startY };
      const end = { x: start.x + side * span * (0.32 + rng() * 0.36), y: startY + height * (0.08 + rng() * 0.12) };
      topoStroke(g, 0.16, 0.58);
      drawLocalPolyline(g, [start, end]);
    }
  }

  topoStroke(g, 0.18, 0.72);
  for (let lane = 0; lane < 2; lane += 1) {
    const y = baseY + lane * height * 0.08;
    const points = [];
    for (let i = 0; i <= 12; i += 1) {
      const t = i / 12;
      points.push({
        x: left + (right - left) * t,
        y: y + Math.sin(t * Math.PI * 3 + lane) * height * 0.02 + (rng() - 0.5) * 1.5
      });
    }
    drawLocalPolyline(g, points);
  }
}

function drawVolcanicMountainGlyph(g, width, height, rng) {
  const baseY = height * 0.24;
  const rx = width * 0.39;
  const ry = height * 0.12;
  topoFill(g, 0.016);
  g.beginPath();
  g.ellipse(0, baseY, rx, ry, 0, 0, Math.PI * 2);
  g.fill();
  g.save();
  g.translate(0, baseY);
  for (let ring = 0; ring < 4; ring += 1) {
    topoStroke(g, ring === 0 ? 0.34 : 0.17, ring === 0 ? 1.08 : 0.7);
    drawLocalIrregularRing(g, rng, rx * (1 - ring * 0.17), ry * (1 - ring * 0.14), 0.045, 34);
  }
  g.restore();
  topoStroke(g, 0.22, 0.72);
  for (let i = 0; i < 18; i += 1) {
    const a = i / 18 * Math.PI * 2;
    const inner = { x: Math.cos(a) * rx * 0.18, y: Math.sin(a) * ry * 0.17 - height * 0.11 };
    const outer = { x: Math.cos(a) * rx * (0.58 + rng() * 0.25), y: baseY + Math.sin(a) * ry * (0.58 + rng() * 0.2) };
    drawLocalPolyline(g, [inner, outer]);
  }
  topoStroke(g, 0.45, 1, "216,255,127");
  g.beginPath();
  g.ellipse(0, -height * 0.12, rx * 0.16, ry * 0.36, 0, 0, Math.PI * 2);
  g.stroke();
}

function drawPlateauGlyph(g, width, height, feature, rng) {
  const variant = feature.variant || 0;
  const rings = 4 + (variant % 3);
  for (let ring = 0; ring < rings; ring += 1) {
    const scale = 1 - ring * 0.15;
    topoStroke(g, ring === 0 ? 0.32 : 0.18, ring === 0 ? 1.05 : 0.72);
    drawLocalIrregularRing(g, rng, width * 0.42 * scale, height * 0.3 * scale, 0.055, 36);
  }
  topoStroke(g, 0.13, 0.7);
  for (let i = 0; i < 16; i += 1) {
    if (rng() < 0.25) continue;
    const x = -width * 0.39 + i * width * 0.052 + (rng() - 0.5) * 5;
    const y = height * 0.14 + rng() * height * 0.18;
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x + (rng() - 0.5) * 3, y + 7 + rng() * 9);
    g.stroke();
  }
  topoStroke(g, 0.23, 0.8, "216,255,127");
  g.beginPath();
  g.ellipse(0, -height * 0.02, width * 0.11, height * 0.045, 0, 0, Math.PI * 2);
  g.stroke();
}

function drawForestGlyph(g, width, height, feature, rng) {
  const count = feature.count || 24;
  const trees = [];
  for (let i = 0; i < count; i += 1) {
    const angle = rng() * Math.PI * 2;
    const radius = Math.sqrt(rng());
    trees.push({
      x: Math.cos(angle) * radius * width * 0.42 + (rng() - 0.5) * 8,
      y: Math.sin(angle) * radius * height * 0.32 + (rng() - 0.5) * 6,
      size: 5.5 + rng() * 8.5
    });
  }
  trees.sort((a, b) => a.y - b.y);
  topoStroke(g, 0.11, 0.7);
  drawLocalIrregularRing(g, rng, width * 0.42, height * 0.3, 0.08, 28);
  for (const tree of trees) {
    drawPineSymbol(g, tree.x, tree.y, tree.size, rng);
  }
  topoStroke(g, 0.18, 0.68, "216,255,127");
  for (let i = 0; i < Math.min(8, count / 4); i += 1) {
    const tree = trees[Math.floor(rng() * trees.length)];
    drawPineSymbol(g, tree.x, tree.y, tree.size * 0.78, rng, true);
  }
}

function drawPineSymbol(g, x, y, size, rng, bright = false) {
  topoStroke(g, bright ? 0.34 : 0.22, bright ? 0.82 : 0.66, bright ? "216,255,127" : "181,255,111");
  for (let tier = 0; tier < 3; tier += 1) {
    const yTop = y - size + tier * size * 0.32;
    const half = size * (0.35 + tier * 0.22);
    g.beginPath();
    g.moveTo(x, yTop);
    g.lineTo(x - half, yTop + size * 0.48);
    g.lineTo(x + half, yTop + size * 0.48);
    g.closePath();
    g.stroke();
  }
  g.beginPath();
  g.moveTo(x, y + size * 0.2);
  g.lineTo(x, y + size * (0.72 + rng() * 0.18));
  g.stroke();
}

function drawWaterGlyph(g, width, height, feature, rng) {
  topoFill(g, 0.022, "124,232,255");
  g.beginPath();
  drawLocalIrregularRing(g, rng, width * 0.39, height * 0.31, 0.08, 36);
  g.fill();
  for (let ring = 0; ring < 3; ring += 1) {
    topoStroke(g, ring === 0 ? 0.18 : 0.1, ring === 0 ? 0.95 : 0.65, ring === 0 ? "124,232,255" : "181,255,111");
    drawLocalIrregularRing(g, rng, width * (0.4 - ring * 0.06), height * (0.31 - ring * 0.04), 0.075, 34);
  }
  topoStroke(g, 0.13, 0.72, "124,232,255");
  for (let i = -3; i <= 3; i += 1) {
    const y = i * height * 0.075 + (rng() - 0.5) * 3;
    const span = width * (0.24 + rng() * 0.16) * (1 - Math.abs(i) * 0.11);
    drawLocalPolyline(g, [
      { x: -span, y },
      { x: -span * 0.3, y: y + (rng() - 0.5) * 3 },
      { x: span * 0.42, y: y + (rng() - 0.5) * 3 },
      { x: span, y: y + (rng() - 0.5) * 2 }
    ]);
  }
}

function drawMarshGlyph(g, width, height, feature, rng) {
  topoStroke(g, 0.14, 0.76, "124,232,255");
  for (let lane = 0; lane < 7; lane += 1) {
    const y = -height * 0.27 + lane * height * 0.09 + (rng() - 0.5) * 4;
    const span = width * (0.24 + rng() * 0.2);
    drawLocalPolyline(g, [
      { x: -span, y },
      { x: -span * 0.35, y: y + Math.sin(lane) * 4 },
      { x: span * 0.35, y: y + Math.cos(lane) * 3 },
      { x: span, y: y + (rng() - 0.5) * 3 }
    ]);
  }
  for (let i = 0; i < 28; i += 1) {
    const x = (rng() - 0.5) * width * 0.72;
    const y = (rng() - 0.5) * height * 0.52;
    drawGrassTuft(g, x, y, 4 + rng() * 7, rng);
  }
  topoStroke(g, 0.11, 0.62);
  drawLocalIrregularRing(g, rng, width * 0.39, height * 0.3, 0.12, 30);
}

function drawGrassTuft(g, x, y, size, rng) {
  topoStroke(g, 0.2, 0.58, "181,255,111");
  g.beginPath();
  g.moveTo(x, y);
  g.lineTo(x - size * 0.38, y - size * 0.72);
  g.moveTo(x, y);
  g.lineTo(x, y - size);
  g.moveTo(x, y);
  g.lineTo(x + size * 0.42, y - size * (0.62 + rng() * 0.25));
  g.stroke();
}

function drawRocksGlyph(g, width, height, feature, rng) {
  const count = 4 + (feature.variant || 0);
  for (let i = 0; i < count; i += 1) {
    const x = (rng() - 0.5) * width * 0.72;
    const y = (rng() - 0.3) * height * 0.48;
    const rx = width * (0.055 + rng() * 0.075);
    const ry = height * (0.08 + rng() * 0.13);
    drawBoulderSymbol(g, x, y, rx, ry, rng);
  }
}

function drawBoulderSymbol(g, x, y, rx, ry, rng) {
  const points = [];
  for (let i = 0; i < 8; i += 1) {
    const t = i / 8 * Math.PI * 2;
    const wobble = 0.82 + rng() * 0.34;
    points.push({ x: x + Math.cos(t) * rx * wobble, y: y + Math.sin(t) * ry * wobble });
  }
  topoFill(g, 0.015);
  g.beginPath();
  points.forEach((point, index) => {
    if (index === 0) g.moveTo(point.x, point.y);
    else g.lineTo(point.x, point.y);
  });
  g.closePath();
  g.fill();
  topoStroke(g, 0.28, 0.82);
  drawLocalPolyline(g, points, true);
  topoStroke(g, 0.15, 0.58, "216,255,127");
  drawLocalPolyline(g, [
    { x: x - rx * 0.42, y: y - ry * 0.12 },
    { x: x + rx * 0.08, y: y - ry * 0.46 },
    { x: x + rx * 0.46, y: y - ry * 0.12 }
  ]);
}

function drawStructureGlyph(g, width, height, feature, rng) {
  const variant = feature.variant || 0;
  topoStroke(g, 0.11, 0.72, "124,232,255");
  for (let row = 0; row < 4; row += 1) {
    const y = -height * 0.28 + row * height * 0.18;
    drawLocalPolyline(g, [
      { x: -width * 0.42, y },
      { x: -width * 0.19, y },
      { x: -width * 0.08, y: y - height * 0.07 },
      { x: width * 0.42, y: y - height * 0.07 }
    ]);
  }
  topoStroke(g, 0.1, 0.68);
  for (let i = 0; i < 24; i += 1) {
    const x = -width * 0.36 + i % 8 * width * 0.09;
    const y = -height * 0.1 + Math.floor(i / 8) * height * 0.14;
    g.beginPath();
    g.moveTo(x - 2, y);
    g.lineTo(x + 2, y);
    g.moveTo(x, y - 2);
    g.lineTo(x, y + 2);
    g.stroke();
  }
  if (variant > 1) {
    topoStroke(g, 0.18, 0.85, "181,255,111");
    g.strokeRect(width * 0.1, -height * 0.28, width * 0.18, height * 0.24);
    g.strokeRect(width * 0.28, -height * 0.08, width * 0.16, height * 0.26);
    g.strokeRect(width * 0.02, height * 0.04, width * 0.2, height * 0.2);
  }
}

function drawRoundedOctagon(w, h, cut) {
  ctx.beginPath();
  ctx.moveTo(-w / 2 + cut, -h / 2);
  ctx.lineTo(w / 2 - cut, -h / 2);
  ctx.lineTo(w / 2, -h / 2 + cut);
  ctx.lineTo(w / 2, h / 2 - cut);
  ctx.lineTo(w / 2 - cut, h / 2);
  ctx.lineTo(-w / 2 + cut, h / 2);
  ctx.lineTo(-w / 2, h / 2 - cut);
  ctx.lineTo(-w / 2, -h / 2 + cut);
  ctx.closePath();
}

function drawCampaignEdges(campaign) {
  ctx.save();
  ctx.lineWidth = 2;
  for (const edge of campaign.edges) {
    const from = campaign.nodes[edge.from];
    const to = campaign.nodes[edge.to];
    if (!from || !to || !from.visible || !to.visible) continue;
    const a = campaignNodePosition(from, campaign);
    const b = campaignNodePosition(to, campaign);
    const secured = from.secured && to.secured;
    const available = canEnterCampaignNode(to, campaign);
    ctx.strokeStyle = secured ? "rgba(97,255,126,0.72)" : available ? "rgba(255,207,90,0.72)" : "rgba(97,255,126,0.28)";
    ctx.setLineDash(secured ? [] : [8, 9]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    ctx.quadraticCurveTo(midX + (b.y - a.y) * 0.08, midY - (b.x - a.x) * 0.08, b.x, b.y);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function drawCampaignUnknownRoutes(campaign) {
  ctx.save();
  ctx.lineWidth = 1.6;
  ctx.setLineDash([5, 8]);
  for (const node of visibleCampaignNodes(campaign)) {
    const directions = campaignUnknownExitDirections(campaign, node);
    if (!directions.length) continue;
    const from = campaignNodePosition(node, campaign);
    for (const direction of directions) {
      const to = {
        x: from.x + direction.dx * CAMPAIGN_MAP.gridX * 0.72,
        y: from.y + direction.dy * CAMPAIGN_MAP.gridY * 0.72
      };
      ctx.strokeStyle = "rgba(97,255,126,0.35)";
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      drawUnknownCampaignNode(to.x, to.y);
    }
  }
  ctx.restore();
}

function drawUnknownCampaignNode(x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.strokeStyle = "rgba(97,255,126,0.28)";
  ctx.fillStyle = "rgba(5,24,8,0.085)";
  ctx.lineWidth = 1.3;
  drawRoundedOctagon(78, 50, 8);
  ctx.fill();
  ctx.globalAlpha = 0.72;
  drawCampaignNodeBracketFrame(78, 50, "#85ff91", false);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(97,255,126,0.16)";
  ctx.beginPath();
  ctx.moveTo(-18, 16);
  ctx.lineTo(18, 16);
  ctx.stroke();
  ctx.fillStyle = "rgba(185,255,189,0.52)";
  ctx.font = "700 20px Courier New, monospace";
  ctx.textAlign = "center";
  ctx.fillText("?", 0, 0);
  ctx.font = "700 8px Courier New, monospace";
  ctx.fillText("UNKNOWN", 0, 15);
  ctx.restore();
}

function drawCampaignNode(campaign, node) {
  const pos = campaignNodePosition(node, campaign);
  const selected = campaign.selectedNodeId === node.id;
  const available = canEnterCampaignNode(node, campaign);
  const type = facilityTypes[node.type] || facilityTypes.tokamak;
  const color = node.secured ? "#85ff91" : available ? "#ffcf5a" : type.color;
  ctx.save();
  ctx.translate(pos.x, pos.y);
  ctx.shadowColor = color;
  const frameGlow = selected ? 8 : node.secured ? 4 : available ? 5 : 2;
  ctx.shadowBlur = 0;
  const nodeTint = node.secured ? "rgba(25,86,34,0.055)" : available ? "rgba(106,82,18,0.07)" : "rgba(7,28,10,0.07)";
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 1.4 : 1;
  const w = CAMPAIGN_MAP.nodeWidth;
  const h = CAMPAIGN_MAP.nodeHeight;
  ctx.fillStyle = "rgba(0,7,2,0.58)";
  drawRoundedOctagon(w, h, 13);
  ctx.fill();
  ctx.fillStyle = nodeTint;
  drawRoundedOctagon(w, h, 13);
  ctx.fill();
  ctx.save();
  ctx.globalAlpha = selected ? 0.16 : 0.07;
  ctx.stroke();
  ctx.restore();
  ctx.shadowColor = color;
  ctx.shadowBlur = frameGlow;
  drawCampaignNodeBracketFrame(w, h, color, selected);
  ctx.shadowBlur = 0;
  drawCampaignFacilitySchematic(node, type, color, available || node.secured);
  ctx.fillStyle = color;
  ctx.font = "700 12px Courier New, monospace";
  ctx.textAlign = "center";
  ctx.fillText(String(node.index).padStart(2, "0"), 0, -51);
  ctx.font = "700 8px Courier New, monospace";
  const name = node.facility.toUpperCase();
  ctx.fillText(name.length > 22 ? `${name.slice(0, 21)}.` : name, 0, 35);
  ctx.fillStyle = "rgba(185,255,189,0.72)";
  ctx.font = "7px Courier New, monospace";
  ctx.fillText(type.label.toUpperCase().slice(0, 26), 0, 49);
  ctx.fillStyle = node.secured ? "#85ff91" : available ? "#ffcf5a" : "rgba(185,255,189,0.58)";
  ctx.fillText(node.secured ? "SECURED" : `SECTOR ${String(node.currentSector).padStart(2, "0")}`, 0, 60);
  ctx.restore();
}

function drawCampaignNodeBracketFrame(w, h, color, selected) {
  ctx.save();
  const left = -w / 2;
  const right = w / 2;
  const top = -h / 2;
  const bottom = h / 2;
  const cut = Math.min(14, w * 0.11, h * 0.16);
  const sideInset = 5;
  const sideDepth = Math.min(44, h * 0.36);
  const rail = Math.min(26, w * 0.17);
  ctx.lineWidth = selected ? 1.9 : 1.25;
  ctx.strokeStyle = color;
  ctx.globalAlpha = selected ? 0.98 : 0.78;
  ctx.beginPath();
  ctx.moveTo(left + cut, top + 6);
  ctx.lineTo(left + cut + rail, top + 6);
  ctx.moveTo(left + sideInset, top + cut);
  ctx.lineTo(left + sideInset, top + cut + sideDepth);
  ctx.moveTo(left + sideInset, top + cut);
  ctx.lineTo(left + cut, top + 6);
  ctx.moveTo(left + sideInset, bottom - cut);
  ctx.lineTo(left + sideInset, bottom - cut - sideDepth);
  ctx.moveTo(left + sideInset, bottom - cut);
  ctx.lineTo(left + cut, bottom - 6);
  ctx.moveTo(left + cut, bottom - 6);
  ctx.lineTo(left + cut + rail, bottom - 6);
  ctx.moveTo(right - cut, top + 6);
  ctx.lineTo(right - cut - rail, top + 6);
  ctx.moveTo(right - sideInset, top + cut);
  ctx.lineTo(right - sideInset, top + cut + sideDepth);
  ctx.moveTo(right - sideInset, top + cut);
  ctx.lineTo(right - cut, top + 6);
  ctx.moveTo(right - sideInset, bottom - cut);
  ctx.lineTo(right - sideInset, bottom - cut - sideDepth);
  ctx.moveTo(right - sideInset, bottom - cut);
  ctx.lineTo(right - cut, bottom - 6);
  ctx.moveTo(right - cut, bottom - 6);
  ctx.lineTo(right - cut - rail, bottom - 6);
  ctx.stroke();
  ctx.lineWidth = selected ? 1.2 : 0.8;
  ctx.globalAlpha = selected ? 0.32 : 0.2;
  ctx.strokeStyle = "rgba(185,255,189,0.75)";
  ctx.beginPath();
  ctx.moveTo(left + 17, top + 28);
  ctx.lineTo(left + 17, bottom - 28);
  ctx.moveTo(right - 17, top + 28);
  ctx.lineTo(right - 17, bottom - 28);
  ctx.moveTo(-22, top + 10);
  ctx.lineTo(22, top + 10);
  ctx.moveTo(-22, bottom - 10);
  ctx.lineTo(22, bottom - 10);
  ctx.stroke();
  ctx.globalAlpha = 0.11;
  for (let y = top + 22; y < bottom - 17; y += 5) {
    ctx.beginPath();
    ctx.moveTo(left + 24, y);
    ctx.lineTo(right - 24, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawCampaignFacilitySchematic(node, type, color, bright) {
  const glyph = type.glyph || node.type || "annex";
  const variant = campaignHash(`${node.seed}:facility-icon`) % 4;
  const sprite = getFacilityGlyphSprite(glyph, variant, color, bright);
  ctx.save();
  ctx.translate(0, -20);
  ctx.globalAlpha = bright ? 1 : 0.64;
  const displayWidth = 128;
  const displayHeight = 62;
  ctx.drawImage(sprite.canvas, -displayWidth / 2, -displayHeight / 2, displayWidth, displayHeight);
  ctx.restore();
}

function getFacilityGlyphSprite(glyph, variant, color, bright) {
  const key = ["facility", glyph, variant, color, bright ? "b" : "d"].join(":");
  const cached = facilityGlyphCache.get(key);
  if (cached) {
    facilityGlyphCache.delete(key);
    facilityGlyphCache.set(key, cached);
    return cached;
  }
  const width = 148;
  const height = 76;
  const canvas = document.createElement("canvas");
  canvas.width = width * FACILITY_GLYPH_PIXEL_SCALE;
  canvas.height = height * FACILITY_GLYPH_PIXEL_SCALE;
  const g = canvas.getContext("2d");
  g.setTransform(
    FACILITY_GLYPH_PIXEL_SCALE,
    0,
    0,
    FACILITY_GLYPH_PIXEL_SCALE,
    width * FACILITY_GLYPH_PIXEL_SCALE / 2,
    height * FACILITY_GLYPH_PIXEL_SCALE / 2
  );
  g.lineCap = "round";
  g.lineJoin = "round";
  g.shadowColor = color;
  g.shadowBlur = bright ? 5 : 2;
  const rng = makeCampaignRng(`facility-glyph:${glyph}:${variant}`);
  drawFacilityGlyph(g, glyph, variant, color, bright, rng);
  const sprite = { canvas, width, height };
  facilityGlyphCache.set(key, sprite);
  while (facilityGlyphCache.size > FACILITY_GLYPH_CACHE_LIMIT) {
    facilityGlyphCache.delete(facilityGlyphCache.keys().next().value);
  }
  return sprite;
}

function facilityStroke(g, color, bright, alpha = 1, width = 1) {
  g.strokeStyle = bright ? color : `rgba(185,255,189,${0.4 * alpha})`;
  g.globalAlpha = bright ? alpha : Math.min(0.72, alpha * 0.82);
  g.lineWidth = width;
}

function facilityFill(g, bright, alpha = 0.035) {
  g.fillStyle = bright ? `rgba(97,255,126,${alpha})` : `rgba(97,255,126,${alpha * 0.52})`;
}

function drawFacilityGlyph(g, glyph, variant, color, bright, rng) {
  facilityStroke(g, color, bright, 0.6, 0.75);
  drawFacilityGround(g);
  facilityStroke(g, color, bright, 1, 1.05);
  facilityFill(g, bright);
  if (glyph === "tokamak") drawFacilityTokamak(g, variant, color, bright, rng);
  else if (glyph === "radar") drawFacilityRadar(g, variant, color, bright, rng);
  else if (glyph === "annex") drawFacilityAnnex(g, variant, color, bright, rng);
  else if (glyph === "bio") drawFacilityBio(g, variant, color, bright, rng);
  else if (glyph === "mining") drawFacilityMining(g, variant, color, bright, rng);
  else if (glyph === "power") drawFacilityPower(g, variant, color, bright, rng);
  else if (glyph === "comms") drawFacilityComms(g, variant, color, bright, rng);
  else if (glyph === "hydro") drawFacilityHydro(g, variant, color, bright, rng);
  else if (glyph === "vehicle") drawFacilityVehicle(g, variant, color, bright, rng);
  else if (glyph === "satellite") drawFacilitySatellite(g, variant, color, bright, rng);
  else if (glyph === "storage") drawFacilityStorage(g, variant, color, bright, rng);
  else drawFacilityCommand(g, variant, color, bright, rng);
  drawFacilityGrime(g, variant, color, bright, rng);
  g.globalAlpha = 1;
}

function drawFacilityGround(g) {
  g.beginPath();
  g.moveTo(-66, 27);
  g.lineTo(66, 27);
  g.moveTo(-58, 30);
  g.lineTo(58, 30);
  g.stroke();
}

function drawFacilityRect(g, x, y, w, h, fill = false) {
  if (fill) g.fillRect(x, y, w, h);
  g.strokeRect(x, y, w, h);
}

function drawFacilityPoly(g, points, fill = false) {
  g.beginPath();
  points.forEach((point, index) => {
    if (index === 0) g.moveTo(point.x, point.y);
    else g.lineTo(point.x, point.y);
  });
  g.closePath();
  if (fill) g.fill();
  g.stroke();
}

function drawFacilityWindows(g, x, y, cols, rows, stepX = 6, stepY = 6) {
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      g.strokeRect(x + col * stepX, y + row * stepY, 2.2, 2.6);
    }
  }
}

function drawFacilityDoor(g, x, y, w, h, rows = 3) {
  g.strokeRect(x, y, w, h);
  for (let i = 1; i < rows; i += 1) {
    g.beginPath();
    g.moveTo(x + 2, y + h * i / rows);
    g.lineTo(x + w - 2, y + h * i / rows);
    g.stroke();
  }
}

function drawFacilityAntenna(g, x, top, bottom, radius = 2) {
  g.beginPath();
  g.moveTo(x, bottom);
  g.lineTo(x, top);
  g.moveTo(x - 4, top + 8);
  g.lineTo(x + 4, top + 8);
  g.stroke();
  g.beginPath();
  g.arc(x, top - radius, radius, 0, Math.PI * 2);
  g.stroke();
}

function drawFacilityDome(g, x, y, rx, ry, ribs = 4) {
  g.beginPath();
  g.arc(x, y, rx, Math.PI, 0);
  g.lineTo(x + rx, y + ry * 0.58);
  g.lineTo(x - rx, y + ry * 0.58);
  g.closePath();
  g.fill();
  g.stroke();
  for (let i = -ribs; i <= ribs; i += 1) {
    const t = i / ribs;
    g.beginPath();
    g.ellipse(x + t * rx * 0.68, y + ry * 0.03, rx * 0.12, ry * 0.98, 0, Math.PI, 0);
    g.stroke();
  }
  for (let i = 1; i <= 2; i += 1) {
    g.beginPath();
    g.ellipse(x, y + i * ry * 0.2, rx * (0.86 - i * 0.13), ry * 0.16, 0, Math.PI, 0);
    g.stroke();
  }
}

function drawFacilityDish(g, x, y, scale, tilt = -0.38) {
  g.save();
  g.translate(x, y);
  g.rotate(tilt);
  g.beginPath();
  g.ellipse(0, 0, 21 * scale, 9 * scale, 0, Math.PI * 0.1, Math.PI * 1.92);
  g.stroke();
  g.beginPath();
  g.ellipse(0, 0, 14 * scale, 5.7 * scale, 0, Math.PI * 0.14, Math.PI * 1.86);
  g.stroke();
  for (const spoke of [-0.55, 0, 0.55]) {
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(Math.cos(spoke) * 18 * scale, Math.sin(spoke) * 7 * scale);
    g.stroke();
  }
  g.restore();
  g.beginPath();
  g.moveTo(x, y + 5 * scale);
  g.lineTo(x - 7 * scale, y + 21 * scale);
  g.lineTo(x + 8 * scale, y + 21 * scale);
  g.closePath();
  g.stroke();
}

function drawFacilitySilo(g, x, y, w, h) {
  g.beginPath();
  g.ellipse(x + w / 2, y, w / 2, 3, 0, 0, Math.PI * 2);
  g.moveTo(x, y);
  g.lineTo(x, y + h);
  g.moveTo(x + w, y);
  g.lineTo(x + w, y + h);
  g.ellipse(x + w / 2, y + h, w / 2, 3, 0, 0, Math.PI * 2);
  g.stroke();
}

function drawFacilityTokamak(g, variant, color, bright, rng) {
  drawFacilityRect(g, -54, 9, 24, 18, true);
  drawFacilityRect(g, 34, 6, 22, 21, true);
  drawFacilityRect(g, -15, 13, 30, 14, true);
  drawFacilityDoor(g, -7, 18, 14, 9, 2);
  drawFacilityDome(g, 0, 2, 29, 18, 5);
  for (let i = 0; i < 5; i += 1) {
    g.beginPath();
    g.ellipse(0, 3 + i * 1.7, 26 + i * 1.6, 5 + i * 0.25, 0, 0, Math.PI * 2);
    g.stroke();
  }
  drawFacilityRect(g, -61, -8, 10, 35, false);
  drawFacilityAntenna(g, -56, -25, -8, 1.8);
  drawFacilityWindows(g, 41, 13, 2, 2);
  g.beginPath();
  g.moveTo(-48, 9);
  g.lineTo(-36, 20);
  g.moveTo(-44, 1);
  g.lineTo(-32, 13);
  g.moveTo(37, 6);
  g.lineTo(49, 6);
  g.moveTo(37, 17);
  g.lineTo(52, 17);
  g.stroke();
}

function drawFacilityRadar(g, variant, color, bright, rng) {
  drawFacilityRect(g, -48, 19, 22, 8, true);
  drawFacilityRect(g, -22, 15, 28, 12, true);
  drawFacilityRect(g, 34, 8, 18, 19, true);
  drawFacilityDish(g, -1, -6, 1.15 + variant * 0.04, -0.48);
  drawFacilityRect(g, -10, 14, 20, 13, true);
  g.beginPath();
  g.moveTo(-7, 14);
  g.lineTo(-1, 4);
  g.lineTo(6, 14);
  g.moveTo(-38, 19);
  g.lineTo(-38, 12);
  g.lineTo(-34, 12);
  g.moveTo(43, 8);
  g.lineTo(43, -7);
  g.stroke();
  drawFacilityAntenna(g, 58, -22, 27, 1.8);
  drawFacilityWindows(g, 39, 14, 2, 2);
}

function drawFacilityAnnex(g, variant, color, bright, rng) {
  for (let i = 0; i < 5; i += 1) {
    const x = -52 + i * 22;
    const h = 15 + (i % 3) * 7 + (i === 2 ? 9 : 0);
    drawFacilityRect(g, x, 27 - h, 20, h, true);
    drawFacilityWindows(g, x + 5, 30 - h, 2, Math.max(1, Math.floor(h / 10)), 7, 7);
  }
  drawFacilityRect(g, -19, -5, 38, 32, true);
  drawFacilityDoor(g, -7, 12, 14, 15, 2);
  drawFacilityPoly(g, [{ x: -15, y: -5 }, { x: 0, y: -25 }, { x: 15, y: -5 }], false);
  g.beginPath();
  g.arc(0, -27, 6, 0, Math.PI * 2);
  g.moveTo(0, -21);
  g.lineTo(0, -5);
  g.stroke();
  drawFacilityAntenna(g, 45, -16, 3, 1.4);
}

function drawFacilityBio(g, variant, color, bright, rng) {
  drawFacilityDome(g, -28, 5, 24, 19, 4);
  drawFacilityDome(g, 20, -1, 33, 23, 5);
  drawFacilityRect(g, -48, 15, 34, 12, true);
  drawFacilityRect(g, 0, 10, 46, 17, true);
  drawFacilityDoor(g, 15, 15, 14, 12, 2);
  drawFacilitySilo(g, 55, -5, 7, 32);
  drawFacilityAntenna(g, 60, -18, 27, 1.4);
  g.beginPath();
  for (let i = 0; i < 3; i += 1) {
    g.moveTo(-58, -11 + i * 6);
    g.quadraticCurveTo(-48, -13 + i * 4, -42, -9 + i * 6);
  }
  g.stroke();
}

function drawFacilityMining(g, variant, color, bright, rng) {
  drawFacilityRect(g, -39, -17, 18, 44, false);
  g.beginPath();
  g.moveTo(-39, 27);
  g.lineTo(-30, -17);
  g.lineTo(-21, 27);
  g.moveTo(-39, 5);
  g.lineTo(-21, 5);
  g.moveTo(-37, -7);
  g.lineTo(-23, 17);
  g.moveTo(-23, -7);
  g.lineTo(-37, 17);
  g.stroke();
  drawFacilityRect(g, -43, -23, 26, 6, false);
  drawFacilityRect(g, -24, 14, 29, 13, true);
  drawFacilityRect(g, 14, 8, 30, 19, true);
  g.beginPath();
  g.moveTo(-19, -7);
  g.lineTo(28, 12);
  g.lineTo(26, 16);
  g.lineTo(-20, -2);
  g.stroke();
  drawFacilityDoor(g, 22, 18, 12, 9, 2);
  drawOrePile(g, -58, 22, 14);
  drawOrePile(g, 55, 22, 13);
}

function drawOrePile(g, x, y, r) {
  g.beginPath();
  g.arc(x, y, r, Math.PI, 0);
  g.arc(x + r * 0.55, y + 1, r * 0.65, Math.PI, 0);
  g.arc(x - r * 0.62, y + 2, r * 0.55, Math.PI, 0);
  g.stroke();
}

function drawFacilityPower(g, variant, color, bright, rng) {
  drawFacilityRect(g, -38, 7, 36, 20, true);
  drawFacilityRect(g, 4, 2, 34, 25, true);
  drawFacilityRect(g, 37, 15, 20, 12, true);
  for (const x of [-28, -10]) {
    g.beginPath();
    g.moveTo(x, 7);
    g.lineTo(x + 3, -27);
    g.lineTo(x + 11, -27);
    g.lineTo(x + 14, 7);
    g.closePath();
    g.stroke();
    for (let y = -17; y < 3; y += 8) {
      g.beginPath();
      g.moveTo(x + 2.5, y);
      g.lineTo(x + 11.5, y);
      g.stroke();
    }
  }
  drawFacilitySilo(g, 41, 9, 9, 18);
  drawFacilitySilo(g, 53, 9, 9, 18);
  g.beginPath();
  g.moveTo(-15, 14);
  g.lineTo(-6, 3);
  g.lineTo(-10, 14);
  g.lineTo(-2, 14);
  g.lineTo(-13, 27);
  g.stroke();
  drawFacilityWindows(g, 12, 8, 3, 2);
}

function drawFacilityComms(g, variant, color, bright, rng) {
  drawFacilityRect(g, -50, 13, 34, 14, true);
  drawFacilityRect(g, -15, 8, 39, 19, true);
  drawFacilityRect(g, 25, 15, 26, 12, true);
  drawFacilityDish(g, -3, -10, 0.9, -0.42);
  drawFacilityAntenna(g, 34, -25, 27, 1.5);
  g.beginPath();
  g.moveTo(28, 27);
  g.lineTo(34, -2);
  g.lineTo(41, 27);
  g.moveTo(30, 12);
  g.lineTo(39, 12);
  g.moveTo(31, 4);
  g.lineTo(38, 4);
  g.stroke();
  drawFacilityDoor(g, -5, 16, 15, 11, 2);
  drawFacilityWindows(g, -43, 18, 3, 1);
  drawFacilityWindows(g, 30, 19, 3, 1);
}

function drawFacilityHydro(g, variant, color, bright, rng) {
  drawFacilityPoly(g, [{ x: -54, y: 5 }, { x: -45, y: -7 }, { x: 18, y: -7 }, { x: 28, y: 5 }, { x: 28, y: 27 }, { x: -54, y: 27 }], true);
  for (let i = 0; i < 6; i += 1) {
    const x = -43 + i * 11;
    g.beginPath();
    g.moveTo(x, -5);
    g.lineTo(x - 5, 26);
    g.moveTo(x + 8, -5);
    g.lineTo(x + 3, 26);
    g.stroke();
  }
  for (let i = 0; i < 5; i += 1) {
    g.beginPath();
    g.moveTo(-47 + i * 14, 5);
    g.lineTo(-36 + i * 14, 5);
    g.stroke();
  }
  drawFacilitySilo(g, 39, -16, 9, 43);
  drawFacilitySilo(g, 52, -15, 9, 42);
  drawFacilityRect(g, 31, 12, 36, 15, true);
}

function drawFacilityVehicle(g, variant, color, bright, rng) {
  drawFacilityPoly(g, [{ x: -59, y: 12 }, { x: -43, y: -14 }, { x: 39, y: -14 }, { x: 58, y: 4 }, { x: 58, y: 27 }, { x: -59, y: 27 }], true);
  drawFacilityDoor(g, -35, 3, 42, 24, 4);
  drawFacilityRect(g, -21, -8, 31, 10, false);
  drawFacilityRect(g, 22, -4, 24, 31, false);
  drawFacilityAntenna(g, 37, -31, -4, 1.6);
  g.beginPath();
  g.moveTo(-13, 21);
  g.lineTo(-2, 14);
  g.lineTo(16, 14);
  g.lineTo(28, 21);
  g.lineTo(-13, 21);
  g.moveTo(-4, 21);
  g.arc(-4, 21, 3, 0, Math.PI * 2);
  g.moveTo(19, 21);
  g.arc(19, 21, 3, 0, Math.PI * 2);
  g.stroke();
}

function drawFacilitySatellite(g, variant, color, bright, rng) {
  drawFacilityRect(g, -51, 6, 42, 21, true);
  drawFacilityRect(g, -36, -5, 32, 11, false);
  drawFacilityDoor(g, -39, 14, 24, 13, 3);
  g.beginPath();
  g.arc(-23, -19, 15, 0, Math.PI * 2);
  g.stroke();
  for (let i = -2; i <= 2; i += 1) {
    g.beginPath();
    g.moveTo(-23 + i * 6, -32 + Math.abs(i) * 3);
    g.lineTo(-23 - i * 5, -7 - Math.abs(i) * 3);
    g.stroke();
  }
  g.beginPath();
  g.ellipse(-23, -19, 15, 5, 0, 0, Math.PI * 2);
  g.moveTo(-38, -19);
  g.lineTo(-8, -19);
  g.stroke();
  drawFacilityRect(g, 13, 9, 27, 18, true);
  drawFacilityDish(g, 45, -1, 0.52, -0.22);
  drawFacilityAntenna(g, 27, -22, 9, 1.4);
}

function drawFacilityStorage(g, variant, color, bright, rng) {
  drawFacilityPoly(g, [{ x: -50, y: -8 }, { x: -34, y: -19 }, { x: 32, y: -19 }, { x: 48, y: -8 }, { x: 48, y: 27 }, { x: -50, y: 27 }], true);
  drawFacilityDoor(g, -39, 2, 21, 25, 4);
  drawFacilityDoor(g, -9, 4, 24, 23, 4);
  drawFacilityRect(g, 24, -3, 22, 30, true);
  drawFacilityRect(g, 51, 10, 13, 17, true);
  drawFacilityWindows(g, -32, -12, 5, 1, 12, 5);
  drawFacilityWindows(g, 26, 6, 2, 3, 7, 6);
  for (let i = 0; i < 4; i += 1) {
    drawFacilityRect(g, -58 + i * 10, 18, 8, 9, false);
  }
}

function drawFacilityCommand(g, variant, color, bright, rng) {
  drawFacilityRect(g, -58, 12, 22, 15, true);
  drawFacilityRect(g, 36, 12, 22, 15, true);
  drawFacilityRect(g, -34, 2, 68, 25, true);
  drawFacilityRect(g, -22, -10, 44, 12, true);
  drawFacilityRect(g, -12, -21, 24, 11, true);
  drawFacilityDoor(g, -9, 13, 18, 14, 2);
  drawFacilityWindows(g, -27, 8, 4, 1, 12, 5);
  drawFacilityWindows(g, -15, -5, 4, 1, 9, 5);
  drawFacilityWindows(g, -52, 18, 2, 1, 9, 5);
  drawFacilityWindows(g, 43, 18, 2, 1, 9, 5);
  drawFacilityAntenna(g, 0, -37, -21, 1.8);
  g.beginPath();
  g.moveTo(-16, -21);
  g.lineTo(0, -31);
  g.lineTo(16, -21);
  g.moveTo(-4, -26);
  g.lineTo(4, -26);
  g.stroke();
}

function drawFacilityGrime(g, variant, color, bright, rng) {
  facilityStroke(g, color, bright, 0.24, 0.55);
  for (let i = 0; i < 16; i += 1) {
    const x = -58 + rng() * 116;
    const y = -24 + rng() * 48;
    if (rng() < 0.46) {
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + 3 + rng() * 10, y + (rng() - 0.5) * 2);
      g.stroke();
    }
  }
}

function drawFacilityMicroDetail(variant) {
  ctx.save();
  ctx.globalAlpha = 0.38;
  ctx.lineWidth = 0.75;
  for (let i = 0; i < 4; i += 1) {
    const y = 16 + i * 2;
    ctx.beginPath();
    ctx.moveTo(-31 + i * 4, y);
    ctx.lineTo(-18 + i * 4, y);
    ctx.moveTo(18 - i * 4, y);
    ctx.lineTo(31 - i * 4, y);
    ctx.stroke();
  }
  for (let i = 0; i < 3 + variant; i += 1) {
    const x = -21 + i * 12;
    ctx.strokeRect(x, 21, 3, 2.5);
  }
  ctx.restore();
}

function drawTokamakFacilityIcon(variant) {
  ctx.beginPath();
  ctx.ellipse(0, 10, 30, 7, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-28, 10);
  ctx.lineTo(-22, 3);
  ctx.lineTo(22, 3);
  ctx.lineTo(28, 10);
  ctx.lineTo(22, 16);
  ctx.lineTo(-22, 16);
  ctx.closePath();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(0, 2, 17, Math.PI, 0);
  ctx.lineTo(17, 9);
  ctx.moveTo(-17, 9);
  ctx.lineTo(-17, 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(0, -24);
  ctx.lineTo(0, 3);
  ctx.moveTo(-8, -10);
  ctx.lineTo(8, -10);
  ctx.moveTo(-5, -18);
  ctx.lineTo(5, -18);
  ctx.stroke();
  for (let i = 0; i < 3 + variant; i += 1) {
    ctx.beginPath();
    ctx.ellipse(0, -1 + i * 2, 16 + i * 2, 3.4 + i * 0.35, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.arc(0, 2, 5 + variant * 0.7, 0, Math.PI * 2);
  ctx.moveTo(-24, 12);
  ctx.lineTo(24, 12);
  ctx.moveTo(-15, 5);
  ctx.lineTo(-15, 16);
  ctx.moveTo(15, 5);
  ctx.lineTo(15, 16);
  ctx.stroke();
}

function drawCargoFacilityIcon(variant) {
  ctx.beginPath();
  ctx.moveTo(-35, 17);
  ctx.lineTo(35, 17);
  ctx.moveTo(-31, 22);
  ctx.lineTo(31, 22);
  ctx.stroke();
  for (let i = 0; i < 3; i += 1) {
    const x = -29 + i * 20;
    ctx.strokeRect(x, 2, 17, 13);
    ctx.beginPath();
    ctx.moveTo(x + 3, 7);
    ctx.lineTo(x + 14, 7);
    ctx.moveTo(x + 8.5, 2);
    ctx.lineTo(x + 8.5, 15);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(-28, 2);
  ctx.lineTo(-28, -23);
  ctx.lineTo(19 + variant * 2, -23);
  ctx.lineTo(19 + variant * 2, -16);
  ctx.moveTo(-30, -14);
  ctx.lineTo(8, -23);
  ctx.moveTo(-20, -23);
  ctx.lineTo(-5, -4);
  ctx.stroke();
  ctx.strokeRect(16 + variant * 2, -16, 8, 7);
  ctx.beginPath();
  for (let i = 0; i < 4; i += 1) {
    const x = -30 + i * 17;
    ctx.moveTo(x, 22);
    ctx.lineTo(x + 11, 22);
  }
  ctx.stroke();
}

function drawFoundryFacilityIcon(variant) {
  ctx.beginPath();
  ctx.moveTo(-34, 15);
  ctx.lineTo(-34, -1);
  ctx.lineTo(-21, -14);
  ctx.lineTo(-8, -1);
  ctx.lineTo(7, -15);
  ctx.lineTo(34, -1);
  ctx.lineTo(34, 15);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  for (let i = 0; i < 3; i += 1) {
    const x = -23 + i * 22;
    const lift = i === variant % 3 ? 4 : 0;
    ctx.strokeRect(x, -25 - lift, 8, 24 + lift);
    ctx.beginPath();
    ctx.moveTo(x - 2, -26 - lift);
    ctx.lineTo(x + 10, -26 - lift);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.arc(0, 7, 6, 0, Math.PI * 2);
  ctx.moveTo(-26, 20);
  ctx.lineTo(26, 20);
  ctx.moveTo(-29, 7);
  ctx.lineTo(-12, 7);
  ctx.moveTo(12, 7);
  ctx.lineTo(29, 7);
  for (let i = 0; i < 5; i += 1) {
    ctx.moveTo(-24 + i * 12, 20);
    ctx.lineTo(-19 + i * 12, 13);
  }
  ctx.stroke();
}

function drawCryoFacilityIcon(variant) {
  for (const x of [-14, 14]) {
    ctx.beginPath();
    ctx.ellipse(x, -14, 8, 4, 0, 0, Math.PI * 2);
    ctx.moveTo(x - 8, -14);
    ctx.lineTo(x - 8, 11);
    ctx.moveTo(x + 8, -14);
    ctx.lineTo(x + 8, 11);
    ctx.ellipse(x, 11, 8, 4, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(-34, 15);
  ctx.lineTo(34, 15);
  ctx.moveTo(-29, 22);
  ctx.lineTo(29, 22);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-14, -3);
  ctx.lineTo(14, -3);
  ctx.moveTo(0, -25);
  ctx.lineTo(0, 20);
  ctx.moveTo(-6 - variant, -17);
  ctx.lineTo(6 + variant, -9);
  ctx.moveTo(6 + variant, -17);
  ctx.lineTo(-6 - variant, -9);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(0, -3, 5, 0, Math.PI * 2);
  ctx.moveTo(-22, 18);
  ctx.lineTo(22, 18);
  ctx.moveTo(-25, 12);
  ctx.lineTo(-25, 22);
  ctx.moveTo(25, 12);
  ctx.lineTo(25, 22);
  ctx.stroke();
}

function drawRadarFacilityIcon(variant) {
  ctx.beginPath();
  ctx.moveTo(-31, 10);
  ctx.lineTo(31, 10);
  ctx.lineTo(31, 21);
  ctx.lineTo(-31, 21);
  ctx.closePath();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(0, 10);
  ctx.lineTo(0, -21);
  ctx.arc(0, -12, 14, Math.PI * 1.1, Math.PI * 1.9);
  ctx.moveTo(0, -12);
  ctx.lineTo(-16, -21);
  ctx.moveTo(0, -12);
  ctx.lineTo(16, -21);
  ctx.stroke();
  for (let i = 0; i < 3; i += 1) {
    ctx.beginPath();
    ctx.arc(0, -12, 18 + i * 6 + variant * 0.5, Math.PI * 1.18, Math.PI * 1.82);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(-20, 10);
  ctx.lineTo(0, -3);
  ctx.lineTo(20, 10);
  ctx.moveTo(-12, 21);
  ctx.lineTo(-12, 10);
  ctx.moveTo(12, 21);
  ctx.lineTo(12, 10);
  ctx.moveTo(-26, 16);
  ctx.lineTo(26, 16);
  ctx.stroke();
}

function drawCampaignLegend(campaign) {
  ctx.save();
  ctx.translate(34, campaignViewportHeight() - 124);
  ctx.fillStyle = "rgba(0,12,4,0.76)";
  ctx.strokeStyle = "rgba(97,255,126,0.28)";
  ctx.lineWidth = 1;
  ctx.fillRect(0, 0, 174, 94);
  ctx.strokeRect(0, 0, 174, 94);
  ctx.fillStyle = "#b9ffbd";
  ctx.font = "700 12px Courier New, monospace";
  ctx.fillText("LEGEND", 16, 24);
  const rows = [
    ["#85ff91", "SECURED"],
    ["#ffcf5a", "AVAILABLE"],
    ["#7ce8ff", "VISIBLE"],
    ["rgba(185,255,189,0.45)", "UNKNOWN"]
  ];
  ctx.font = "10px Courier New, monospace";
  rows.forEach((row, index) => {
    const y = 42 + index * 12;
    ctx.strokeStyle = row[0];
    ctx.beginPath();
    ctx.moveTo(16, y);
    ctx.lineTo(54, y);
    ctx.stroke();
    ctx.fillStyle = "#b9ffbd";
    ctx.fillText(row[1], 66, y + 3);
  });
  ctx.restore();
}

function drawCampaignSelectionPanel(campaign) {
  const node = selectedCampaignNode(campaign);
  if (!node) return;
  const type = facilityTypes[node.type] || facilityTypes.tokamak;
  ctx.save();
  ctx.translate(campaignViewportWidth() - 256, campaignViewportHeight() - 116);
  ctx.fillStyle = "rgba(0,12,4,0.8)";
  ctx.strokeStyle = node.secured ? "rgba(133,255,145,0.42)" : canEnterCampaignNode(node, campaign) ? "rgba(255,207,90,0.52)" : "rgba(97,255,126,0.24)";
  ctx.fillRect(0, 0, 222, 86);
  ctx.strokeRect(0, 0, 222, 86);
  ctx.fillStyle = type.color;
  ctx.font = "700 12px Courier New, monospace";
  ctx.fillText(node.facility.toUpperCase(), 14, 22);
  ctx.fillStyle = "rgba(185,255,189,0.66)";
  ctx.font = "10px Courier New, monospace";
  ctx.fillText(type.desc.toUpperCase().slice(0, 31), 14, 40);
  ctx.fillStyle = "#b9ffbd";
  ctx.fillText(node.secured ? "STATUS: SECURED" : canEnterCampaignNode(node, campaign) ? `READY: SECTOR ${node.currentSector}` : "STATUS: ROUTE LOCKED", 14, 58);
  ctx.fillText(`EXITS: ${node.plannedExitCount}  FACILITIES: ${campaign.stats.facilitiesSecured}`, 14, 73);
  ctx.restore();
}

function drawPathStroke(g, width, strokeStyle) {
  g.lineWidth = width;
  g.strokeStyle = strokeStyle;
  g.beginPath();
  g.moveTo(pathPoints[0].x, pathPoints[0].y);
  for (let i = 1; i < pathPoints.length; i += 1) {
    g.lineTo(pathPoints[i].x, pathPoints[i].y);
  }
  g.stroke();
}

function drawGate(g, point, entry) {
  g.save();
  const offset = gateLabelOffset(point, entry);
  g.translate(point.x + offset.x, point.y + offset.y);
  g.globalAlpha = 0.48;
  g.strokeStyle = entry ? "rgba(124,232,255,0.72)" : "rgba(255,95,97,0.62)";
  g.fillStyle = entry ? "rgba(124,232,255,0.045)" : "rgba(255,95,97,0.04)";
  g.lineWidth = 1.5;
  g.beginPath();
  g.rect(-34, -15, 68, 30);
  g.fill();
  g.stroke();
  g.font = "700 10px Courier New, monospace";
  g.textAlign = "center";
  g.fillStyle = entry ? "rgba(196,247,255,0.7)" : "rgba(255,176,176,0.62)";
  g.fillText(entry ? "ENTRY" : "CORE", 0, 4);
  for (let i = -1; i <= 1; i += 1) {
    g.beginPath();
    if (offset.edge === "top") {
      g.moveTo(i * 9 - 4, -24);
      g.lineTo(i * 9, -17);
      g.lineTo(i * 9 + 4, -24);
    } else if (offset.edge === "bottom") {
      g.moveTo(i * 9 - 4, 24);
      g.lineTo(i * 9, 17);
      g.lineTo(i * 9 + 4, 24);
    } else if (offset.edge === "right") {
      g.moveTo(-48, i * 7 - 4);
      g.lineTo(-39, i * 7);
      g.lineTo(-48, i * 7 + 4);
    } else {
      g.moveTo(48, i * 7 - 4);
      g.lineTo(39, i * 7);
      g.lineTo(48, i * 7 + 4);
    }
    g.stroke();
  }
  g.restore();
}

function gateLabelOffset(point, entry) {
  const margin = 92;
  if (point.y <= FACILITY_PATH_BOUNDS.minY + 18) return { x: 0, y: 34, edge: "top" };
  if (point.y >= FACILITY_PATH_BOUNDS.maxY - 18) return { x: 0, y: -34, edge: "bottom" };
  if (point.x <= FACILITY_PATH_BOUNDS.minX + 18) return { x: 88, y: 0, edge: "left" };
  if (point.x >= FACILITY_PATH_BOUNDS.maxX - 18) return { x: -88, y: 0, edge: "right" };
  return entry ? { x: margin, y: 0, edge: "left" } : { x: -margin, y: 0, edge: "right" };
}

function drawMines() {
  for (const mine of state.mines) {
    ctx.save();
    ctx.translate(mine.x, mine.y);
    const armed = mine.arm <= 0;
    ctx.strokeStyle = armed ? mine.color : "rgba(185,255,189,0.35)";
    ctx.fillStyle = armed ? "rgba(255,207,90,0.14)" : "rgba(185,255,189,0.06)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-16, 0);
    ctx.lineTo(16, 0);
    ctx.moveTo(0, -16);
    ctx.lineTo(0, 16);
    ctx.stroke();
    ctx.restore();
  }
}

function drawTowers() {
  const selected = selectedTower();
  for (const tower of state.towers) {
    if (selected && selected.id === tower.id) {
      const stats = getTowerStats(tower);
      drawRange(tower.x, tower.y, stats.range, "rgba(124,232,255,0.16)");
    }
    drawTower(tower);
  }
}

function drawTower(tower) {
  const def = towerById[tower.type];
  const stats = getTowerStats(tower);
  ctx.save();
  ctx.translate(tower.x, tower.y);
  drawTowerShape(tower.type, tower.level, tower.pulse);
  ctx.fillStyle = def.color;
  ctx.font = "700 11px Courier New, monospace";
  ctx.textAlign = "center";
  ctx.fillText(`MK${tower.level}`, 0, 34);
  const cooldownRatio = clamp(tower.cooldown * stats.rate, 0, 1);
  if (cooldownRatio > 0.03) {
    ctx.strokeStyle = "rgba(185,255,189,0.38)";
    ctx.beginPath();
    ctx.arc(0, 0, 27, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * (1 - cooldownRatio));
    ctx.stroke();
  }
  ctx.restore();
}

function drawTowerShape(type, level, pulse) {
  const def = towerById[type] || { color: "#85ff91" };
  drawTowerSprite(type, level, pulse, def.color);
}

function drawTowerSprite(type, level, pulse, color) {
  const sprite = getTowerSprite(type, level, pulse, color);
  ctx.drawImage(sprite.canvas, -sprite.half, -sprite.half, sprite.size, sprite.size);
}

function towerPulseBand(type, pulse) {
  if (type !== "arc") return 0;
  const period = Math.PI * 2;
  const normalized = ((pulse % period) + period) % period;
  return Math.round(normalized / period * (TOWER_PULSE_BANDS - 1));
}

function towerPulseFromBand(type, band) {
  if (type !== "arc") return 0;
  return band / TOWER_PULSE_BANDS * Math.PI * 2;
}

function getTowerSprite(type, level, pulse, color) {
  const band = towerPulseBand(type, pulse || 0);
  const key = [type, level, band, color].join("|");
  const cached = towerSpriteCache.get(key);
  if (cached) {
    towerSpriteCache.delete(key);
    towerSpriteCache.set(key, cached);
    return cached;
  }

  const half = 76;
  const canvas = document.createElement("canvas");
  canvas.width = half * 2 * TOWER_SPRITE_PIXEL_SCALE;
  canvas.height = half * 2 * TOWER_SPRITE_PIXEL_SCALE;
  const g = canvas.getContext("2d");
  g.setTransform(
    TOWER_SPRITE_PIXEL_SCALE,
    0,
    0,
    TOWER_SPRITE_PIXEL_SCALE,
    half * TOWER_SPRITE_PIXEL_SCALE,
    half * TOWER_SPRITE_PIXEL_SCALE
  );
  drawTowerSchematic(g, type, level, towerPulseFromBand(type, band), color);
  const sprite = { canvas, size: half * 2, half };
  towerSpriteCache.set(key, sprite);
  if (towerSpriteCache.size > TOWER_SPRITE_CACHE_LIMIT) {
    const oldestKey = towerSpriteCache.keys().next().value;
    towerSpriteCache.delete(oldestKey);
  }
  return sprite;
}

function enemySpriteHalfExtent(type, radius, boss) {
  const reach = {
    crawler: 2.25,
    beetle: 2.15,
    slime: 2.0,
    worm: 2.7,
    wisp: 2.15,
    juggernaut: 1.95,
    phantom: 1.75,
    mite: 2.35,
    leech: 2.75,
    obelisk: 1.95
  }[type] || 2.1;
  return Math.ceil(radius * reach + (boss ? 34 : 18));
}

function bossSpriteHalfExtent(bossType, radius) {
  const reach = {
    hive: 2.2,
    conduit: 2.72,
    colossus: 2.12,
    harvester: 2.0
  }[bossType] || 2.2;
  return Math.ceil(radius * reach + 34);
}

function enemySpriteCacheKey(enemy) {
  if (enemy.boss && enemy.bossType) {
    return [
      "boss",
      enemy.bossType,
      Math.round(enemy.radius * 10),
      enemy.color
    ].join("|");
  }
  const phase = phasedEnemyTypes.has(enemy.type) ? enemy.spritePhase || 0 : 0;
  return [
    enemy.type,
    Math.round(enemy.radius * 10),
    enemy.boss ? "boss" : "unit",
    enemy.child ? "child" : "main",
    enemy.burrowed ? "burrowed" : "surface",
    phase,
    enemy.color
  ].join("|");
}

function getEnemySprite(enemy) {
  const key = enemySpriteCacheKey(enemy);
  const cached = enemySpriteCache.get(key);
  if (cached) return cached;

  const bossDef = enemy.boss && enemy.bossType ? bossDefs[enemy.bossType] || bossDefs.hive : null;
  const pixelScale = 4;
  const half = bossDef ? bossSpriteHalfExtent(enemy.bossType, enemy.radius) : enemySpriteHalfExtent(enemy.type, enemy.radius, enemy.boss);
  const canvas = document.createElement("canvas");
  canvas.width = half * 2 * pixelScale;
  canvas.height = half * 2 * pixelScale;
  const g = canvas.getContext("2d");
  g.setTransform(pixelScale, 0, 0, pixelScale, half * pixelScale, half * pixelScale);
  g.strokeStyle = bossDef ? bossDef.color : enemy.boss ? "#ffcf5a" : enemy.color;
  g.fillStyle = enemy.burrowed ? "rgba(97,255,126,0.06)" : "rgba(6,28,10,0.92)";
  g.lineWidth = bossDef ? 3.1 : enemy.boss ? 3.35 : 2.35;
  g.shadowColor = bossDef ? bossDef.color : enemy.boss ? "#ffcf5a" : enemy.color;
  g.shadowBlur = bossDef ? 16 : enemy.boss ? 14 : 7;
  if (bossDef) {
    drawBossSchematic(g, enemy.bossType, enemy.radius, { phase: 0 });
  } else {
    drawReferenceEnemy(g, enemy.type, enemy.radius, {
      phase: (enemy.spritePhase || 0) * 0.78,
      child: enemy.child,
      boss: enemy.boss
    });
  }
  const sprite = { canvas, size: half * 2, half };
  enemySpriteCache.set(key, sprite);
  if (enemySpriteCache.size > 220) {
    enemySpriteCache.clear();
    enemySpriteCache.set(key, sprite);
  }
  return sprite;
}

function drawEnemies() {
  for (const enemy of state.enemies) {
    const pos = pointAtDistance(enemy.progress);
    drawEnemy(enemy, pos);
  }
}

function drawEnemy(enemy, pos) {
  const alpha = enemy.type === "phantom" && enemy.status.jam <= 0 ? 0.58 : 1;
  const jitter = enemy.type === "wisp" ? Math.sin(enemy.special * 18) * 3 : 0;
  ctx.save();
  ctx.translate(pos.x + jitter, pos.y - (enemy.burrowed ? 5 : 0));
  ctx.globalAlpha = alpha;
  ctx.save();
  ctx.shadowBlur = 0;
  ctx.fillStyle = enemy.boss ? "rgba(255,207,90,0.13)" : "rgba(97,255,126,0.08)";
  ctx.beginPath();
  ctx.ellipse(0, enemy.radius * 0.62, enemy.radius * 1.48, enemy.radius * 0.38, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  const sprite = getEnemySprite(enemy);
  ctx.drawImage(sprite.canvas, -sprite.half, -sprite.half, sprite.size, sprite.size);
  if (enemy.status.slow > 0) {
    ctx.strokeStyle = "rgba(184,247,255,0.8)";
    ctx.beginPath();
    ctx.arc(0, 0, enemy.radius + 6, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (enemy.status.jam > 0) {
    ctx.strokeStyle = "rgba(210,255,120,0.8)";
    ctx.beginPath();
    ctx.moveTo(-enemy.radius - 5, -enemy.radius - 5);
    ctx.lineTo(enemy.radius + 5, enemy.radius + 5);
    ctx.moveTo(enemy.radius + 5, -enemy.radius - 5);
    ctx.lineTo(-enemy.radius - 5, enemy.radius + 5);
    ctx.stroke();
  }
  ctx.restore();
  drawEnemyHealth(enemy, pos);
}

function drawEnemyShape(enemy) {
  const r = enemy.radius;
  drawReferenceEnemy(ctx, enemy.type, r, { phase: enemy.special, child: enemy.child, boss: enemy.boss });
  return;
  const bolt = (x, y, size = 1.6) => {
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.stroke();
  };
  const jointedLeg = (side, y, length = 1) => {
    ctx.beginPath();
    ctx.moveTo(side * r * 0.42, y);
    ctx.lineTo(side * r * 0.95, y + r * 0.16 * Math.sign(y || 1));
    ctx.lineTo(side * r * 1.45 * length, y + r * 0.55 * Math.sign(y || 1));
    ctx.stroke();
    bolt(side * r * 0.92, y + r * 0.16 * Math.sign(y || 1), 1.2);
  };
  const segmentedArc = (x, y, radius, start, end, pieces = 7) => {
    const span = end - start;
    for (let i = 0; i < pieces; i += 1) {
      ctx.beginPath();
      ctx.arc(x, y, radius, start + span * (i / pieces), start + span * ((i + 0.55) / pieces));
      ctx.stroke();
    }
  };
  const ghostStroke = (draw) => {
    ctx.save();
    ctx.globalAlpha = 0.34;
    ctx.strokeStyle = "rgba(185,255,189,0.45)";
    ctx.lineWidth = Math.max(1, r * 0.045);
    draw();
    ctx.restore();
  };
  if (enemy.type === "crawler") {
    for (const y of [-r * 0.48, 0, r * 0.48]) {
      jointedLeg(-1, y, 0.82);
      jointedLeg(1, y, 0.82);
    }
    ctx.beginPath();
    ctx.ellipse(0, -r * 0.05, r * 1.02, r * 0.78, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(0, -r * 0.26, r * 0.74, Math.PI * 1.05, Math.PI * 1.95);
    ctx.moveTo(-r * 0.72, -r * 0.08);
    ctx.quadraticCurveTo(0, r * 0.26, r * 0.72, -r * 0.08);
    ctx.moveTo(-r * 0.5, r * 0.26);
    ctx.quadraticCurveTo(0, r * 0.52, r * 0.5, r * 0.26);
    ctx.stroke();
    ctx.strokeRect(-r * 0.33, -r * 0.28, r * 0.66, r * 0.52);
    ctx.strokeRect(-r * 0.16, r * 0.28, r * 0.32, r * 0.2);
    ctx.beginPath();
    ctx.arc(-r * 0.3, -r * 0.12, Math.max(1.5, r * 0.09), 0, Math.PI * 2);
    ctx.arc(r * 0.3, -r * 0.12, Math.max(1.5, r * 0.09), 0, Math.PI * 2);
    ctx.moveTo(-r * 0.55, -r * 0.58);
    ctx.lineTo(-r * 1.05, -r * 1.04);
    ctx.moveTo(r * 0.55, -r * 0.58);
    ctx.lineTo(r * 1.05, -r * 1.04);
    ctx.moveTo(-r * 0.92, -r * 0.96);
    ctx.lineTo(-r * 1.28, -r * 0.82);
    ctx.moveTo(r * 0.92, -r * 0.96);
    ctx.lineTo(r * 1.28, -r * 0.82);
    ctx.stroke();
    ghostStroke(() => {
      ctx.beginPath();
      ctx.ellipse(0, -r * 0.04, r * 1.12, r * 0.88, 0, 0, Math.PI * 2);
      ctx.stroke();
    });
  } else if (enemy.type === "beetle") {
    for (const y of [-r * 0.58, 0, r * 0.58]) {
      jointedLeg(-1, y, 1);
      jointedLeg(1, y, 1);
    }
    ctx.beginPath();
    ctx.ellipse(-r * 0.08, 0, r * 1.25, r * 0.85, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-r * 0.95, -r * 0.1);
    ctx.bezierCurveTo(-r * 0.72, -r * 0.8, r * 0.46, -r * 0.76, r * 0.94, -r * 0.2);
    ctx.moveTo(-r * 0.95, r * 0.1);
    ctx.bezierCurveTo(-r * 0.72, r * 0.8, r * 0.46, r * 0.76, r * 0.94, r * 0.2);
    ctx.stroke();
    for (let i = -2; i <= 2; i += 1) {
      ctx.beginPath();
      ctx.arc(-r * 0.18, 0, r * (0.34 + i * 0.1), -Math.PI * 0.5, Math.PI * 0.5);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(-r * 1.0, 0);
    ctx.lineTo(r * 0.92, 0);
    ctx.moveTo(-r * 0.72, -r * 0.58);
    ctx.quadraticCurveTo(-r * 0.04, -r * 0.2, r * 0.82, -r * 0.44);
    ctx.moveTo(-r * 0.72, r * 0.58);
    ctx.quadraticCurveTo(-r * 0.04, r * 0.2, r * 0.82, r * 0.44);
    ctx.moveTo(r * 1.1, -r * 0.45);
    ctx.lineTo(r * 1.65, -r * 0.12);
    ctx.lineTo(r * 1.16, 0);
    ctx.lineTo(r * 1.65, r * 0.12);
    ctx.lineTo(r * 1.1, r * 0.45);
    ctx.stroke();
    bolt(r * 0.48, -r * 0.28);
    bolt(r * 0.48, r * 0.28);
    bolt(-r * 0.32, -r * 0.42, 1.2);
    bolt(-r * 0.32, r * 0.42, 1.2);
    ghostStroke(() => {
      ctx.beginPath();
      ctx.moveTo(-r * 1.06, -r * 0.55);
      ctx.lineTo(r * 0.85, -r * 0.55);
      ctx.moveTo(-r * 1.06, r * 0.55);
      ctx.lineTo(r * 0.85, r * 0.55);
      ctx.stroke();
    });
  } else if (enemy.type === "slime") {
    ctx.beginPath();
    ctx.moveTo(-r * 1.05, 4);
    ctx.bezierCurveTo(-r * 0.98, -r * 0.82, -r * 0.4, -r * 1.05, r * 0.08, -r * 0.82);
    ctx.bezierCurveTo(r * 0.45, -r * 1.3, r * 1.12, -r * 0.58, r * 1.02, r * 0.1);
    ctx.bezierCurveTo(r * 1.04, r * 0.9, r * 0.32, r * 1.08, -r * 0.3, r * 0.82);
    ctx.bezierCurveTo(-r * 0.94, r * 0.72, -r * 1.22, r * 0.3, -r * 1.05, 4);
    ctx.fill();
    ctx.stroke();
    ghostStroke(() => {
      ctx.beginPath();
      ctx.moveTo(-r * 0.72, r * 0.18);
      ctx.bezierCurveTo(-r * 0.44, -r * 0.62, r * 0.36, -r * 0.64, r * 0.62, r * 0.06);
      ctx.bezierCurveTo(r * 0.38, r * 0.56, -r * 0.28, r * 0.62, -r * 0.72, r * 0.18);
      ctx.stroke();
    });
    ctx.beginPath();
    ctx.arc(r * 0.18, -r * 0.18, r * 0.16, 0, Math.PI * 2);
    ctx.arc(-r * 0.38, r * 0.12, r * 0.1, 0, Math.PI * 2);
    ctx.arc(r * 0.52, r * 0.34, r * 0.08, 0, Math.PI * 2);
    ctx.arc(-r * 0.02, r * 0.34, r * 0.06, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-r * 0.72, r * 0.55);
    ctx.lineTo(-r * 0.6, r * 1.05);
    ctx.lineTo(-r * 0.74, r * 1.17);
    ctx.moveTo(r * 0.08, r * 0.76);
    ctx.lineTo(r * 0.02, r * 1.18);
    ctx.lineTo(r * 0.16, r * 1.28);
    ctx.moveTo(r * 0.72, r * 0.34);
    ctx.lineTo(r * 1.12, r * 0.62);
    ctx.stroke();
  } else if (enemy.type === "worm") {
    for (let i = 0; i < 6; i += 1) {
      ctx.beginPath();
      ctx.ellipse(-i * r * 0.36, Math.sin(enemy.special * 4 + i) * 2, r * 0.54, r * 0.42, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-i * r * 0.36 - r * 0.2, -r * 0.36);
      ctx.lineTo(-i * r * 0.36 + r * 0.16, r * 0.36);
      ctx.stroke();
      if (i < 5) {
        bolt(-i * r * 0.36 - r * 0.18, -r * 0.1, 1.1);
      }
    }
    ctx.beginPath();
    ctx.arc(r * 0.62, 0, r * 0.54, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(r * 0.78, 0, r * 0.26, -Math.PI * 0.45, Math.PI * 0.45);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(r * 0.96, -r * 0.28);
    ctx.lineTo(r * 1.45, -r * 0.6);
    ctx.lineTo(r * 1.24, -r * 0.22);
    ctx.moveTo(r * 0.98, r * 0.28);
    ctx.lineTo(r * 1.45, r * 0.6);
    ctx.lineTo(r * 1.24, r * 0.22);
    ctx.moveTo(r * 1.1, 0);
    ctx.lineTo(r * 1.55, 0);
    ctx.stroke();
    ghostStroke(() => {
      ctx.beginPath();
      ctx.moveTo(-r * 2.25, r * 0.58);
      ctx.bezierCurveTo(-r * 1.3, r * 1.0, -r * 0.12, r * 0.96, r * 0.96, r * 0.5);
      ctx.stroke();
    });
  } else if (enemy.type === "wisp") {
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.36, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    segmentedArc(0, 0, r * 0.72, 0, Math.PI * 2, 8);
    segmentedArc(0, 0, r * 1.08, Math.PI * 0.15, Math.PI * 1.9, 10);
    for (let i = 0; i < 9; i += 1) {
      const a = i * Math.PI * 2 / 9 + enemy.special;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * r * 0.35, Math.sin(a) * r * 0.35);
      ctx.lineTo(Math.cos(a + 0.2) * r * 0.92, Math.sin(a + 0.2) * r * 0.92);
      ctx.lineTo(Math.cos(a + 0.44) * r * 1.55, Math.sin(a + 0.44) * r * 1.55);
      ctx.stroke();
      if (i % 3 === 0) {
        bolt(Math.cos(a + 0.44) * r * 1.22, Math.sin(a + 0.44) * r * 1.22, 1.4);
      }
    }
    ghostStroke(() => {
      ctx.beginPath();
      ctx.moveTo(-r * 1.2, -r * 0.68);
      ctx.lineTo(-r * 0.7, -r * 0.68);
      ctx.lineTo(-r * 0.7, -r * 1.03);
      ctx.moveTo(r * 1.2, r * 0.68);
      ctx.lineTo(r * 0.68, r * 0.68);
      ctx.lineTo(r * 0.68, r * 1.02);
      ctx.stroke();
    });
  } else if (enemy.type === "juggernaut") {
    ctx.beginPath();
    ctx.ellipse(0, -r * 0.18, r * 1.22, r * 0.84, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(0, -r * 0.42, r * 0.82, Math.PI * 1.08, Math.PI * 1.92);
    ctx.moveTo(-r * 0.82, r * 0.06);
    ctx.quadraticCurveTo(0, r * 0.48, r * 0.82, r * 0.06);
    ctx.stroke();
    ctx.strokeRect(-r * 0.62, -r * 0.64, r * 1.24, r * 0.34);
    ctx.strokeRect(-r * 0.3, -r * 0.1, r * 0.6, r * 0.38);
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(side * r * 0.68, r * 0.22);
      ctx.lineTo(side * r * 1.04, r * 0.68);
      ctx.lineTo(side * r * 0.9, r * 1.08);
      ctx.moveTo(side * r * 0.3, r * 0.34);
      ctx.lineTo(side * r * 0.52, r * 0.92);
      ctx.lineTo(side * r * 0.26, r * 1.2);
      ctx.stroke();
      ctx.strokeRect(side * r * 0.78 - (side < 0 ? r * 0.44 : 0), r * 1.04, r * 0.44, r * 0.2);
      ctx.strokeRect(side * r * 0.2 - (side < 0 ? r * 0.36 : 0), r * 1.2, r * 0.36, r * 0.18);
    }
    for (const x of [-0.7, -0.35, 0.35, 0.7]) bolt(x * r, r * 0.1, 1.3);
    bolt(0, -r * 0.46, 2);
    ghostStroke(() => {
      ctx.beginPath();
      ctx.ellipse(0, -r * 0.18, r * 1.34, r * 0.95, 0, 0, Math.PI * 2);
      ctx.stroke();
    });
  } else {
    ghostStroke(() => {
      ctx.beginPath();
      ctx.moveTo(r * 0.9, -r * 0.08);
      ctx.bezierCurveTo(r * 0.55, -r * 1.3, -r * 0.56, -r * 1.3, -r * 0.9, -r * 0.08);
      ctx.lineTo(-r * 0.62, r * 1.2);
      ctx.stroke();
    });
    ctx.beginPath();
    ctx.moveTo(r * 0.82, -r * 0.06);
    ctx.bezierCurveTo(r * 0.5, -r * 1.24, -r * 0.48, -r * 1.24, -r * 0.82, -r * 0.06);
    ctx.lineTo(-r * 0.56, r * 1.18);
    ctx.lineTo(-r * 0.16, r * 0.74);
    ctx.lineTo(0, r * 1.42);
    ctx.lineTo(r * 0.16, r * 0.74);
    ctx.lineTo(r * 0.56, r * 1.18);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-r * 0.42, -r * 0.18);
    ctx.lineTo(r * 0.42, -r * 0.18);
    ctx.moveTo(-r * 0.24, r * 0.2);
    ctx.lineTo(-r * 0.62, r * 0.72);
    ctx.moveTo(r * 0.24, r * 0.2);
    ctx.lineTo(r * 0.62, r * 0.72);
    ctx.moveTo(0, -r * 0.74);
    ctx.lineTo(0, r * 0.56);
    ctx.stroke();
    ctx.save();
    ctx.globalAlpha = 0.45;
    for (let y = -0.42; y <= 0.72; y += 0.28) {
      ctx.beginPath();
      ctx.moveTo(-r * 0.7, r * y);
      ctx.lineTo(r * 0.7, r * y + Math.sin(enemy.special * 8 + y) * 2);
      ctx.stroke();
    }
    ctx.restore();
  }
  ctx.strokeStyle = "rgba(185,255,189,0.28)";
  ctx.lineWidth = Math.max(1, r * 0.055);
  ctx.beginPath();
  ctx.moveTo(-r * 0.72, -r * 0.08);
  ctx.lineTo(-r * 0.16, -r * 0.2);
  ctx.moveTo(r * 0.18, r * 0.22);
  ctx.lineTo(r * 0.74, r * 0.1);
  ctx.stroke();
}

function drawEnemyHealth(enemy, pos) {
  const width = enemy.boss ? 58 : 34;
  const y = pos.y - enemy.radius - 15;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.62)";
  ctx.fillRect(pos.x - width / 2, y, width, 5);
  ctx.fillStyle = enemy.hp / enemy.maxHp < 0.28 ? "#ff5f61" : enemy.boss ? "#ffcf5a" : "#85ff91";
  ctx.fillRect(pos.x - width / 2, y, width * clamp(enemy.hp / enemy.maxHp, 0, 1), 5);
  ctx.strokeStyle = "rgba(185,255,189,0.28)";
  ctx.strokeRect(pos.x - width / 2, y, width, 5);
  ctx.restore();
}

function drawProjectiles() {
  for (const projectile of state.projectiles) {
    ctx.save();
    ctx.strokeStyle = projectile.color;
    ctx.fillStyle = projectile.color;
    ctx.shadowColor = projectile.color;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(projectile.x, projectile.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

function drawEffects() {
  for (const effect of state.effects) {
    const t = clamp(effect.life / effect.maxLife, 0, 1);
    ctx.save();
    ctx.globalAlpha = t;
    ctx.strokeStyle = effect.color;
    ctx.fillStyle = effect.color;
    ctx.lineWidth = effect.width || 2;
    ctx.shadowColor = effect.color;
    ctx.shadowBlur = 14;
    if (effect.type === "line") {
      ctx.beginPath();
      ctx.moveTo(effect.x, effect.y);
      const midX = (effect.x + effect.x2) / 2 + (Math.random() - 0.5) * 10;
      const midY = (effect.y + effect.y2) / 2 + (Math.random() - 0.5) * 10;
      ctx.lineTo(midX, midY);
      ctx.lineTo(effect.x2, effect.y2);
      ctx.stroke();
    } else if (effect.type === "ring") {
      const radius = effect.radius + (effect.maxRadius - effect.radius) * (1 - t);
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, radius, 0, Math.PI * 2);
      ctx.stroke();
    } else if (effect.type === "spark") {
      for (let i = 0; i < 8; i += 1) {
        const a = i * Math.PI / 4;
        ctx.beginPath();
        ctx.moveTo(effect.x, effect.y);
        ctx.lineTo(effect.x + Math.cos(a) * 18 * (1 - t), effect.y + Math.sin(a) * 18 * (1 - t));
        ctx.stroke();
      }
    } else if (effect.type === "text") {
      ctx.font = "700 12px Courier New, monospace";
      ctx.textAlign = "center";
      ctx.fillText(effect.text, effect.x, effect.y - (1 - t) * 16);
    }
    ctx.restore();
  }
  for (const particle of state.particles) {
    const t = clamp(particle.life / particle.maxLife, 0, 1);
    ctx.save();
    ctx.globalAlpha = t;
    ctx.fillStyle = particle.color;
    ctx.fillRect(particle.x - 1.5, particle.y - 1.5, 3, 3);
    ctx.restore();
  }
}

function drawPlacement() {
  if (state.mode === "campaign") return;
  const def = towerById[state.placingType];
  if (!def || state.selectedTowerId) return;
  const placement = validatePlacement(hover.x, hover.y);
  hover.valid = placement.valid;
  hover.reason = placement.reason;
  const tempTower = { type: def.id, level: 1, x: hover.x, y: hover.y, pulse: performance.now() / 900 };
  const stats = getTowerStats(tempTower);
  drawRange(hover.x, hover.y, stats.range, placement.valid ? "rgba(133,255,145,0.12)" : "rgba(255,95,97,0.12)");
  ctx.save();
  ctx.globalAlpha = placement.valid ? 0.72 : 0.42;
  ctx.translate(hover.x, hover.y);
  drawTowerSprite(def.id, 1, performance.now() / 600, placement.valid ? def.color : "#ff5f61");
  ctx.restore();
  if (!placement.valid) {
    ctx.save();
    ctx.fillStyle = "#ff5f61";
    ctx.font = "700 12px Courier New, monospace";
    ctx.textAlign = "center";
    ctx.fillText(placement.reason.toUpperCase(), hover.x, hover.y + 50);
    ctx.restore();
  }
}

function drawRange(x, y, range, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, range, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawPhaseOverlay() {
  if (state.mode === "campaign") return;
  const displayWave = state.phase === "combat" ? state.wave : state.wave + 1;
  const forecast = waveForecast(displayWave || 1, activeOperation());
  ctx.save();
  ctx.fillStyle = "rgba(185,255,189,0.72)";
  ctx.font = "700 14px Courier New, monospace";
  ctx.textAlign = "left";
  ctx.fillText(forecast.operationLabel.toUpperCase(), 42, 58);
  ctx.fillText(`PATH LENGTH ${Math.round(path.total)}m`, BOARD.width - 214, 58);
  ctx.font = "12px Courier New, monospace";
  ctx.fillStyle = "rgba(185,255,189,0.42)";
  ctx.fillText(`${forecast.kind} / THREAT ${forecast.threat}`, 42, 76);
  if (state.paused || state.gameOver) {
    ctx.fillStyle = "rgba(0,8,2,0.62)";
    ctx.fillRect(0, 0, BOARD.width, BOARD.height);
    ctx.strokeStyle = state.gameOver ? "rgba(255,95,97,0.8)" : "rgba(124,232,255,0.8)";
    ctx.strokeRect(BOARD.width / 2 - 170, BOARD.height / 2 - 42, 340, 84);
    ctx.fillStyle = state.gameOver ? "#ffb0b0" : "#c4f7ff";
    ctx.font = "700 24px Courier New, monospace";
    ctx.textAlign = "center";
    ctx.fillText(state.gameOver ? "SECTOR LOST" : "SIMULATION PAUSED", BOARD.width / 2, BOARD.height / 2 + 8);
  }
  ctx.restore();
}
