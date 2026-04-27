"use strict";

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
  terrain.rivers.forEach((river) => drawCampaignRiverFeature(campaign, river));
  terrain.ridges.forEach((ridge) => drawSurveyRidgeFeature(campaign, ridge));
  terrain.mountains.forEach((mountain) => drawMountainFeature(campaign, mountain));
  terrain.forests.forEach((forest) => drawForestFeature(campaign, forest));
  terrain.ticks.forEach((ticks) => drawSurveyTickFeature(campaign, ticks));
}

function drawCampaignRiverFeature(campaign, river) {
  const points = (river.points || []).map((point) => campaignWorldToScreen(campaign, point.x, point.y));
  if (points.length < 2 || !screenPolylineNearView(points, 80)) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(124,232,255,0.028)";
  ctx.lineWidth = 5.5;
  drawTerrainPolyline(points);
  ctx.strokeStyle = "rgba(124,232,255,0.095)";
  ctx.lineWidth = 1.15;
  drawTerrainPolyline(points);
  ctx.strokeStyle = "rgba(185,255,189,0.05)";
  ctx.lineWidth = 1;
  ctx.setLineDash([10, 13]);
  drawTerrainPolyline(points);
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
  if (!worldFeatureNearView(campaign, mountain.x, mountain.y, Math.max(mountain.width || 180, mountain.height || 80) + 80)) return;
  ctx.save();
  const crest = mountain.crest || [];
  ctx.strokeStyle = "rgba(97,255,126,0.145)";
  ctx.lineWidth = 1.05;
  ctx.beginPath();
  crest.forEach((crestPoint, index) => {
    const point = campaignWorldToScreen(campaign, crestPoint.x, crestPoint.y);
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();
  ctx.strokeStyle = "rgba(185,255,189,0.08)";
  for (const crestPoint of crest.filter((point) => point.high)) {
    const peak = campaignWorldToScreen(campaign, crestPoint.x, crestPoint.y);
    const base = campaignWorldToScreen(campaign, crestPoint.baseX, crestPoint.baseY);
    ctx.beginPath();
    ctx.moveTo(peak.x, peak.y);
    ctx.lineTo(base.x, base.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 0.76;
  ctx.strokeStyle = "rgba(97,255,126,0.108)";
  for (const contourPoints of mountain.contours || []) {
    ctx.beginPath();
    contourPoints.forEach((contourPoint, index) => {
      const point = campaignWorldToScreen(campaign, contourPoint.x, contourPoint.y);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }
  ctx.restore();
}

function drawForestFeature(campaign, forest) {
  if (!worldFeatureNearView(campaign, forest.x, forest.y, 110)) return;
  const center = campaignWorldToScreen(campaign, forest.x, forest.y);
  ctx.save();
  ctx.strokeStyle = "rgba(133,255,145,0.135)";
  ctx.fillStyle = "rgba(97,255,126,0.028)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(center.x, center.y + 8, 48, 18, forest.rotation || 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(97,255,126,0.05)";
  ctx.stroke();
  ctx.strokeStyle = "rgba(133,255,145,0.135)";
  for (const tree of forest.trees || []) {
    const point = campaignWorldToScreen(campaign, tree.x, tree.y);
    const size = tree.size || 7;
    ctx.beginPath();
    ctx.moveTo(point.x, point.y - size);
    ctx.lineTo(point.x - size * 0.72, point.y + size * 0.38);
    ctx.lineTo(point.x + size * 0.72, point.y + size * 0.38);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(point.x, point.y + size * 0.36);
    ctx.lineTo(point.x, point.y + size * 0.9);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSurveyRidgeFeature(campaign, ridge) {
  if (!worldFeatureNearView(campaign, ridge.x, ridge.y, (ridge.width || 180) + 90)) return;
  ctx.save();
  ctx.strokeStyle = "rgba(185,255,189,0.062)";
  ctx.lineWidth = 1;
  for (const lane of ridge.lanes || []) {
    ctx.beginPath();
    lane.forEach((lanePoint, index) => {
      const point = campaignWorldToScreen(campaign, lanePoint.x, lanePoint.y);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }
  ctx.restore();
}

function drawSurveyTickFeature(campaign, ticks) {
  if (!worldFeatureNearView(campaign, ticks.x, ticks.y, 120)) return;
  ctx.save();
  ctx.strokeStyle = "rgba(124,232,255,0.045)";
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
  ctx.globalAlpha = selected ? 0.28 : 0.14;
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
  ctx.fillText(String(node.index).padStart(2, "0"), 0, -36);
  ctx.font = "700 8px Courier New, monospace";
  const name = node.facility.toUpperCase();
  ctx.fillText(name.length > 20 ? `${name.slice(0, 19)}.` : name, 0, 22);
  ctx.fillStyle = "rgba(185,255,189,0.72)";
  ctx.font = "7px Courier New, monospace";
  ctx.fillText(type.label.toUpperCase(), 0, 34);
  ctx.fillStyle = node.secured ? "#85ff91" : available ? "#ffcf5a" : "rgba(185,255,189,0.58)";
  ctx.fillText(node.secured ? "SECURED" : `SECTOR ${String(node.currentSector).padStart(2, "0")}`, 0, 44);
  ctx.restore();
}

function drawCampaignNodeBracketFrame(w, h, color, selected) {
  ctx.save();
  const left = -w / 2;
  const right = w / 2;
  const top = -h / 2;
  const bottom = h / 2;
  const cut = Math.min(13, w * 0.1, h * 0.18);
  const bracketX = Math.min(34, w * 0.22);
  const bracketY = Math.min(25, h * 0.28);
  ctx.lineWidth = selected ? 1.65 : 1.05;
  ctx.strokeStyle = color;
  ctx.globalAlpha = selected ? 0.92 : 0.66;
  ctx.beginPath();
  ctx.moveTo(left + cut, top + 5);
  ctx.lineTo(left + cut + bracketX, top + 5);
  ctx.moveTo(left + 5, top + cut);
  ctx.lineTo(left + 5, top + cut + bracketY);
  ctx.moveTo(left + 5, top + cut);
  ctx.lineTo(left + cut, top + 5);
  ctx.moveTo(right - cut, top + 5);
  ctx.lineTo(right - cut - bracketX, top + 5);
  ctx.moveTo(right - 5, top + cut);
  ctx.lineTo(right - 5, top + cut + bracketY);
  ctx.moveTo(right - 5, top + cut);
  ctx.lineTo(right - cut, top + 5);
  ctx.moveTo(left + cut, bottom - 5);
  ctx.lineTo(left + cut + bracketX, bottom - 5);
  ctx.moveTo(left + 5, bottom - cut);
  ctx.lineTo(left + 5, bottom - cut - bracketY);
  ctx.moveTo(left + 5, bottom - cut);
  ctx.lineTo(left + cut, bottom - 5);
  ctx.moveTo(right - cut, bottom - 5);
  ctx.lineTo(right - cut - bracketX, bottom - 5);
  ctx.moveTo(right - 5, bottom - cut);
  ctx.lineTo(right - 5, bottom - cut - bracketY);
  ctx.moveTo(right - 5, bottom - cut);
  ctx.lineTo(right - cut, bottom - 5);
  ctx.stroke();
  ctx.globalAlpha = selected ? 0.42 : 0.28;
  ctx.strokeStyle = "rgba(185,255,189,0.75)";
  ctx.beginPath();
  ctx.moveTo(left + 16, -8);
  ctx.lineTo(left + 16, 8);
  ctx.moveTo(right - 16, -8);
  ctx.lineTo(right - 16, 8);
  ctx.moveTo(-18, top + 9);
  ctx.lineTo(18, top + 9);
  ctx.moveTo(-18, bottom - 9);
  ctx.lineTo(18, bottom - 9);
  ctx.stroke();
  ctx.globalAlpha = 0.13;
  for (let y = top + 17; y < bottom - 14; y += 5) {
    ctx.beginPath();
    ctx.moveTo(left + 22, y);
    ctx.lineTo(right - 22, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawCampaignFacilitySchematic(node, type, color, bright) {
  const variant = campaignHash(`${node.seed}:facility-icon`) % 4;
  ctx.save();
  ctx.beginPath();
  ctx.rect(-49, -41, 98, 57);
  ctx.clip();
  ctx.translate(0, -12);
  ctx.scale(0.86, 0.86);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = bright ? color : "rgba(185,255,189,0.46)";
  ctx.fillStyle = bright ? "rgba(97,255,126,0.042)" : "rgba(97,255,126,0.022)";
  ctx.lineWidth = 1.3;
  ctx.shadowColor = color;
  ctx.shadowBlur = bright ? 4 : 1;
  if (node.type === "tokamak") {
    drawTokamakFacilityIcon(variant);
  } else if (node.type === "cargo") {
    drawCargoFacilityIcon(variant);
  } else if (node.type === "foundry") {
    drawFoundryFacilityIcon(variant);
  } else if (node.type === "cryo") {
    drawCryoFacilityIcon(variant);
  } else {
    drawRadarFacilityIcon(variant);
  }
  drawFacilityMicroDetail(variant);
  ctx.restore();
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
  const offset = entry ? { x: 88, y: 34 } : { x: -86, y: -42 };
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
    if (entry) {
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
