import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

THREE.Cache.enabled = true;

let camera, scene, renderer, controls;
let raycaster, mouse;
let composer; // post-processing composer
let sharedLeafMat; // shared leaf material across plants
const textureLoader = new THREE.TextureLoader();
const sharedAssets = {}; // shared textures / materials

// Sun + lighting (updated each tick)
let sky, sunLight, skyFill, warmFill;
const bulbLights = []; // PointLights inside Edison bulbs
const bulbMeshes = []; // bulb glass meshes for emissive control
const SUN_LOCATION = { lat: 40.7128, lng: -74.0060 }; // NYC — Eastern Time
let lastSunUpdate = 0;

// Instanced empty pots — one InstancedMesh per pot piece, hidden per-slot when planted
let emptyPotInstances = null;
const emptyPotOccupied = [];

// Forest + atmosphere
let treeMaterial = null;     // tree billboard material with onBeforeCompile-injected wind
let currentDayness = 1;      // 1 = full day, 0 = full night (set by updateSunAndLighting)
const eyePairs = [];         // glowing-red eye pair state machines

const objects = []; // Interactable objects (plants)
let todos = []; // Data for todos

// Time tracking
let simulatedTimeOffset = 0; // Fast forward offset in ms

// Local Storage keys
const STORAGE_KEY = 'greenhouse-todos-data';

// Touch device detection (coarse pointer = primary input is touch)
const isTouchDevice = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
let mobileActive = false; // True when exploring on a touch device

function saveTodosToLocal() {
    // Only save the data, not the THREE.js meshes
    const dataToSave = todos.map(t => {
        const { mesh, ...rest } = t;
        return rest;
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
        todos: dataToSave,
        simulatedTimeOffset: simulatedTimeOffset
    }));
}

function loadTodosFromLocal() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
        try {
            const data = JSON.parse(saved);
            todos = data.todos || [];
            simulatedTimeOffset = data.simulatedTimeOffset || 0;

            // Rebuild plants for each loaded todo
            todos.forEach(todo => {
                createPlant(todo, true);
            });
        } catch (e) {
            console.error("Failed to parse saved data:", e);
        }
    }
}

// Movement state
let moveForward = false;
let moveBackward = false;
let moveLeft = false;
let moveRight = false;
let prevTime = performance.now();
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();

// DOM Elements
const blocker = document.getElementById('blocker');
const instructions = document.getElementById('instructions');
const uiContainer = document.getElementById('ui-container');
const resumeBtn = document.getElementById('resume-btn');

function init() {
    // 1. Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0xc8dfee, 0.005); // soft humid haze

    // 2. Camera
    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.05, 2000);
    camera.position.y = 1.6;

    // 3. Renderer (PBR pipeline)
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.body.appendChild(renderer.domElement);

    // 4. Image-based lighting via procedural room environment (gives nice IBL on PBR materials)
    const pmrem = new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    // 5. Sky shader for outdoor backdrop
    sky = new Sky();
    sky.scale.setScalar(1500);
    sky.material.uniforms.mieCoefficient.value = 0.005;
    sky.material.uniforms.mieDirectionalG.value = 0.85;
    scene.add(sky);

    // 6. Lighting — sun (directional) + soft fill from sky
    sunLight = new THREE.DirectionalLight(0xfff0d6, 0); // intensity set by updateSunAndLighting
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(1024, 1024);
    sunLight.shadow.camera.left = -25;
    sunLight.shadow.camera.right = 25;
    sunLight.shadow.camera.top = 15;
    sunLight.shadow.camera.bottom = -55;
    sunLight.shadow.camera.near = 1;
    sunLight.shadow.camera.far = 120;
    sunLight.shadow.bias = -0.0005;
    sunLight.shadow.normalBias = 0.04;
    sunLight.shadow.radius = 4;
    scene.add(sunLight);

    skyFill = new THREE.HemisphereLight(0xb6dbff, 0x4a3a2a, 0);
    scene.add(skyFill);

    // Warm interior fill — mimics light bouncing off the wood and floor
    warmFill = new THREE.PointLight(0xffd9a0, 0, 30, 1.6);
    warmFill.position.set(0, 3.5, -20);
    scene.add(warmFill);

    // 5. Controls
    controls = new PointerLockControls(camera, document.body);

    instructions.addEventListener('click', startExploring);
    resumeBtn.addEventListener('click', startExploring);

    controls.addEventListener('lock', function () {
        blocker.style.display = 'none';
        uiContainer.style.display = 'none';
    });

    controls.addEventListener('unlock', function () {
        // Only show main UI if modal isn't open
        if (document.getElementById('todo-modal').style.display === 'none') {
            uiContainer.style.display = 'block';
            blocker.style.display = 'none'; // Keep blocker hidden when UI is open, rely on UI bg
        }
    });

    scene.add(controls.getObject());

    // 6. Movement Event Listeners
    const onKeyDown = function (event) {
        switch (event.code) {
            case 'ArrowUp':
            case 'KeyW':
                moveForward = true;
                break;
            case 'ArrowLeft':
            case 'KeyA':
                moveLeft = true;
                break;
            case 'ArrowDown':
            case 'KeyS':
                moveBackward = true;
                break;
            case 'ArrowRight':
            case 'KeyD':
                moveRight = true;
                break;
        }
    };

    const onKeyUp = function (event) {
        switch (event.code) {
            case 'ArrowUp':
            case 'KeyW':
                moveForward = false;
                break;
            case 'ArrowLeft':
            case 'KeyA':
                moveLeft = false;
                break;
            case 'ArrowDown':
            case 'KeyS':
                moveBackward = false;
                break;
            case 'ArrowRight':
            case 'KeyD':
                moveRight = false;
                break;
        }
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    // 7. Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2(0, 0); // Always center for crosshair

    // 8. Build Greenhouse Environment
    buildGreenhouse();

    // 8b. Haunted forest backdrop + glowing eyes
    buildHauntedForest();
    buildHauntedEyes();

    // 9. Initial sun + lighting (uses real Eastern Time)
    updateSunAndLighting();

    // 10. Load Saved Data
    loadTodosFromLocal();

    // 10. Post-processing — bloom for soft highlights through the glass
    composer = new EffectComposer(renderer);
    composer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    composer.setSize(window.innerWidth, window.innerHeight);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.15, // strength — light bloom for highlights only
        0.6,  // radius
        0.96  // threshold (only true highlights bloom)
    );
    composer.addPass(bloom);
    composer.addPass(new OutputPass());

    // Window resize handler
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('orientationchange', onWindowResize);

    // Mobile touch controls
    if (isTouchDevice) {
        document.body.classList.add('touch');
        setupTouchControls();
    }
}

// --- PBR Texture / Material Helpers ---

// Convert a heightmap canvas (grayscale brightness = height) into a normal map texture.
// Runs once per texture at startup.
function makeNormalMapFromCanvas(srcCanvas, scale = 1.0) {
    const w = srcCanvas.width, h = srcCanvas.height;
    const src = srcCanvas.getContext('2d').getImageData(0, 0, w, h).data;
    const out = new Uint8ClampedArray(w * h * 4);
    const heightAt = (x, y) => {
        const xi = ((x % w) + w) % w;
        const yi = ((y % h) + h) % h;
        const i = (yi * w + xi) * 4;
        return (src[i] + src[i + 1] + src[i + 2]) / (3 * 255);
    };
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const dx = (heightAt(x + 1, y) - heightAt(x - 1, y)) * scale;
            const dy = (heightAt(x, y + 1) - heightAt(x, y - 1)) * scale;
            const nx = -dx, ny = -dy, nz = 1.0;
            const inv = 1 / Math.hypot(nx, ny, nz);
            const i = (y * w + x) * 4;
            out[i + 0] = (nx * inv * 0.5 + 0.5) * 255;
            out[i + 1] = (ny * inv * 0.5 + 0.5) * 255;
            out[i + 2] = (nz * inv * 0.5 + 0.5) * 255;
            out[i + 3] = 255;
        }
    }
    const normCanvas = document.createElement('canvas');
    normCanvas.width = w;
    normCanvas.height = h;
    normCanvas.getContext('2d').putImageData(new ImageData(out, w, h), 0, 0);
    const tex = new THREE.CanvasTexture(normCanvas);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function configureRepeat(tex, repeat, srgb) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeat[0], repeat[1]);
    tex.anisotropy = 8;
    if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
}

// Real PBR wood from threejs.org's CDN. Each call returns a material with cloned-textures
// so different surfaces can have their own UV repeats.
function makeWoodMaterial({ repeat = [1, 1], roughness = 0.85, color = 0xffffff } = {}) {
    const colorMap = textureLoader.load('https://threejs.org/examples/textures/hardwood2_diffuse.jpg');
    const bumpMap = textureLoader.load('https://threejs.org/examples/textures/hardwood2_bump.jpg');
    const roughMap = textureLoader.load('https://threejs.org/examples/textures/hardwood2_roughness.jpg');
    configureRepeat(colorMap, repeat, true);
    configureRepeat(bumpMap, repeat, false);
    configureRepeat(roughMap, repeat, false);
    return new THREE.MeshPhysicalMaterial({
        color,
        map: colorMap,
        bumpMap: bumpMap,
        bumpScale: 0.0015,
        roughnessMap: roughMap,
        roughness,
        metalness: 0,
        envMapIntensity: 0.7
    });
}

function getGlassMaterial() {
    if (sharedAssets.glass) return sharedAssets.glass;
    // MeshStandardMaterial (no transmission render pass) — much cheaper than Physical+transmission.
    sharedAssets.glass = new THREE.MeshStandardMaterial({
        color: 0x88c890,
        metalness: 0,
        roughness: 0.22,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
        envMapIntensity: 0.7,
        depthWrite: false
    });
    return sharedAssets.glass;
}

// Verdigris (oxidized) copper — patinated greenish-blue with mottled texture
function getCopperMaterial() {
    if (sharedAssets.copper) return sharedAssets.copper;
    const SIZE = 256;

    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = colorCanvas.height = SIZE;
    const ctx = colorCanvas.getContext('2d');
    // Base verdigris green
    ctx.fillStyle = '#5a9b80';
    ctx.fillRect(0, 0, SIZE, SIZE);
    // Patina patches — varying greens and blues
    for (let i = 0; i < 350; i++) {
        const r = 60 + Math.random() * 50;
        const g = 130 + Math.random() * 70;
        const b = 100 + Math.random() * 60;
        ctx.fillStyle = `rgba(${r|0},${g|0},${b|0},${0.25 + Math.random() * 0.45})`;
        ctx.beginPath();
        ctx.arc(Math.random() * SIZE, Math.random() * SIZE, 3 + Math.random() * 12, 0, Math.PI * 2);
        ctx.fill();
    }
    // Exposed copper streaks (warmer reddish-brown)
    for (let i = 0; i < 40; i++) {
        const r = 140 + Math.random() * 60;
        const g = 80 + Math.random() * 30;
        const b = 40 + Math.random() * 20;
        ctx.fillStyle = `rgba(${r|0},${g|0},${b|0},${0.35 + Math.random() * 0.35})`;
        ctx.beginPath();
        ctx.arc(Math.random() * SIZE, Math.random() * SIZE, 1.5 + Math.random() * 4, 0, Math.PI * 2);
        ctx.fill();
    }
    // Dark crevices
    for (let i = 0; i < 200; i++) {
        ctx.fillStyle = `rgba(20,40,30,${0.15 + Math.random() * 0.25})`;
        ctx.fillRect(Math.random() * SIZE, Math.random() * SIZE, 1 + Math.random() * 2, 1 + Math.random() * 2);
    }
    const colorTex = new THREE.CanvasTexture(colorCanvas);
    colorTex.colorSpace = THREE.SRGBColorSpace;
    colorTex.wrapS = colorTex.wrapT = THREE.RepeatWrapping;

    // Bumpy patina surface
    const heightCanvas = document.createElement('canvas');
    heightCanvas.width = heightCanvas.height = SIZE;
    const hctx = heightCanvas.getContext('2d');
    hctx.fillStyle = '#808080';
    hctx.fillRect(0, 0, SIZE, SIZE);
    for (let i = 0; i < 300; i++) {
        const grey = 80 + Math.random() * 130;
        hctx.fillStyle = `rgb(${grey|0},${grey|0},${grey|0})`;
        hctx.beginPath();
        hctx.arc(Math.random() * SIZE, Math.random() * SIZE, 2 + Math.random() * 6, 0, Math.PI * 2);
        hctx.fill();
    }
    const normalTex = makeNormalMapFromCanvas(heightCanvas, 5);
    normalTex.wrapS = normalTex.wrapT = THREE.RepeatWrapping;

    sharedAssets.copper = new THREE.MeshPhysicalMaterial({
        map: colorTex,
        normalMap: normalTex,
        normalScale: new THREE.Vector2(1, 1),
        color: 0xb0d8c0,
        roughness: 0.55,
        metalness: 0.35,
        envMapIntensity: 0.85,
        sheen: 0.25,
        sheenColor: new THREE.Color(0x9adfba),
        sheenRoughness: 0.7
    });
    return sharedAssets.copper;
}

// Edison-style bulb glass with controllable emissive (off during day, glowing at night)
function makeBulbMaterial() {
    return new THREE.MeshStandardMaterial({
        color: 0xffd29a,
        emissive: 0xffaa55,
        emissiveIntensity: 0,
        roughness: 0.25,
        metalness: 0,
        transparent: true,
        opacity: 0.85
    });
}

function getMetalFrameMaterial() {
    if (sharedAssets.frame) return sharedAssets.frame;
    sharedAssets.frame = new THREE.MeshPhysicalMaterial({
        color: 0x2c3a30,
        metalness: 0.85,
        roughness: 0.45,
        envMapIntensity: 0.9
    });
    return sharedAssets.frame;
}

function getDoorHandleMaterial() {
    if (sharedAssets.handle) return sharedAssets.handle;
    sharedAssets.handle = new THREE.MeshPhysicalMaterial({
        color: 0xc8b870,
        metalness: 0.95,
        roughness: 0.18,
        envMapIntensity: 1.2
    });
    return sharedAssets.handle;
}

function getDirtFloorMaterial() {
    if (sharedAssets.floor) return sharedAssets.floor;
    const SIZE = 512;

    // Color
    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = colorCanvas.height = SIZE;
    const cctx = colorCanvas.getContext('2d');
    cctx.fillStyle = '#3a2718';
    cctx.fillRect(0, 0, SIZE, SIZE);
    for (let i = 0; i < 5000; i++) {
        const r = 30 + Math.random() * 35;
        const g = 18 + Math.random() * 25;
        const b = 8 + Math.random() * 18;
        cctx.fillStyle = `rgba(${r|0},${g|0},${b|0},${0.15 + Math.random() * 0.45})`;
        cctx.beginPath();
        cctx.arc(Math.random() * SIZE, Math.random() * SIZE, 1 + Math.random() * 3, 0, Math.PI * 2);
        cctx.fill();
    }
    // Pebbles
    for (let i = 0; i < 220; i++) {
        const shade = 70 + Math.random() * 70;
        cctx.fillStyle = `rgb(${shade|0},${(shade-12)|0},${(shade-22)|0})`;
        cctx.beginPath();
        cctx.arc(Math.random() * SIZE, Math.random() * SIZE, 1.5 + Math.random() * 4.5, 0, Math.PI * 2);
        cctx.fill();
    }
    const colorTex = new THREE.CanvasTexture(colorCanvas);
    configureRepeat(colorTex, [12, 12], true);

    // Height
    const heightCanvas = document.createElement('canvas');
    heightCanvas.width = heightCanvas.height = SIZE;
    const hctx = heightCanvas.getContext('2d');
    hctx.fillStyle = '#3a3a3a';
    hctx.fillRect(0, 0, SIZE, SIZE);
    for (let i = 0; i < 1200; i++) {
        const grey = 50 + Math.random() * 130;
        hctx.fillStyle = `rgb(${grey|0},${grey|0},${grey|0})`;
        hctx.beginPath();
        hctx.arc(Math.random() * SIZE, Math.random() * SIZE, 1 + Math.random() * 6, 0, Math.PI * 2);
        hctx.fill();
    }
    const normalTex = makeNormalMapFromCanvas(heightCanvas, 8);
    configureRepeat(normalTex, [12, 12], false);

    sharedAssets.floor = new THREE.MeshPhysicalMaterial({
        map: colorTex,
        normalMap: normalTex,
        normalScale: new THREE.Vector2(1.4, 1.4),
        roughness: 0.95,
        metalness: 0
    });
    return sharedAssets.floor;
}

function getPotMaterial() {
    if (sharedAssets.pot) return sharedAssets.pot;
    const SIZE = 256;
    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = colorCanvas.height = SIZE;
    const ctx = colorCanvas.getContext('2d');
    // Terracotta base
    const grad = ctx.createLinearGradient(0, 0, 0, SIZE);
    grad.addColorStop(0, '#b06138');
    grad.addColorStop(0.5, '#9a5331');
    grad.addColorStop(1, '#7a3f24');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, SIZE, SIZE);
    // Variation specks
    for (let i = 0; i < 600; i++) {
        const r = 130 + Math.random() * 80;
        const g = 60 + Math.random() * 40;
        const b = 35 + Math.random() * 25;
        ctx.fillStyle = `rgba(${r|0},${g|0},${b|0},${Math.random() * 0.45})`;
        ctx.fillRect(Math.random() * SIZE, Math.random() * SIZE, 1 + Math.random() * 4, 1 + Math.random() * 3);
    }
    // Subtle horizontal bands (throwing rings)
    for (let y = 0; y < SIZE; y += 6 + Math.random() * 4) {
        ctx.fillStyle = `rgba(40,20,10,${0.04 + Math.random() * 0.08})`;
        ctx.fillRect(0, y, SIZE, 1);
    }
    const colorTex = new THREE.CanvasTexture(colorCanvas);
    colorTex.colorSpace = THREE.SRGBColorSpace;

    const heightCanvas = document.createElement('canvas');
    heightCanvas.width = heightCanvas.height = SIZE;
    const hctx = heightCanvas.getContext('2d');
    hctx.fillStyle = '#808080';
    hctx.fillRect(0, 0, SIZE, SIZE);
    for (let y = 0; y < SIZE; y += 6 + Math.random() * 4) {
        hctx.fillStyle = `rgba(40,40,40,0.6)`;
        hctx.fillRect(0, y, SIZE, 1);
    }
    for (let i = 0; i < 240; i++) {
        const grey = 110 + Math.random() * 100;
        hctx.fillStyle = `rgb(${grey|0},${grey|0},${grey|0})`;
        hctx.beginPath();
        hctx.arc(Math.random() * SIZE, Math.random() * SIZE, 1 + Math.random() * 2.5, 0, Math.PI * 2);
        hctx.fill();
    }
    const normalTex = makeNormalMapFromCanvas(heightCanvas, 4);

    sharedAssets.pot = new THREE.MeshStandardMaterial({
        map: colorTex,
        normalMap: normalTex,
        normalScale: new THREE.Vector2(0.6, 0.6),
        roughness: 0.82,
        metalness: 0
    });
    return sharedAssets.pot;
}

function getSoilMaterial() {
    if (sharedAssets.soil) return sharedAssets.soil;
    const SIZE = 128;
    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = colorCanvas.height = SIZE;
    const ctx = colorCanvas.getContext('2d');
    ctx.fillStyle = '#1f1108';
    ctx.fillRect(0, 0, SIZE, SIZE);
    for (let i = 0; i < 800; i++) {
        const r = 30 + Math.random() * 40;
        const g = 18 + Math.random() * 25;
        const b = 8 + Math.random() * 18;
        ctx.fillStyle = `rgba(${r|0},${g|0},${b|0},${0.4 + Math.random() * 0.4})`;
        ctx.beginPath();
        ctx.arc(Math.random() * SIZE, Math.random() * SIZE, 0.8 + Math.random() * 2.5, 0, Math.PI * 2);
        ctx.fill();
    }
    const colorTex = new THREE.CanvasTexture(colorCanvas);
    colorTex.colorSpace = THREE.SRGBColorSpace;

    const heightCanvas = document.createElement('canvas');
    heightCanvas.width = heightCanvas.height = SIZE;
    const hctx = heightCanvas.getContext('2d');
    hctx.fillStyle = '#606060';
    hctx.fillRect(0, 0, SIZE, SIZE);
    for (let i = 0; i < 600; i++) {
        const grey = 80 + Math.random() * 120;
        hctx.fillStyle = `rgb(${grey|0},${grey|0},${grey|0})`;
        hctx.beginPath();
        hctx.arc(Math.random() * SIZE, Math.random() * SIZE, 0.8 + Math.random() * 2, 0, Math.PI * 2);
        hctx.fill();
    }
    const normalTex = makeNormalMapFromCanvas(heightCanvas, 5);

    sharedAssets.soil = new THREE.MeshPhysicalMaterial({
        map: colorTex,
        normalMap: normalTex,
        normalScale: new THREE.Vector2(1, 1),
        roughness: 1,
        metalness: 0
    });
    return sharedAssets.soil;
}

function createLeafMaterial() {
    const SIZE = 256;

    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = colorCanvas.height = SIZE;
    const cctx = colorCanvas.getContext('2d');
    cctx.clearRect(0, 0, SIZE, SIZE);
    const grad = cctx.createRadialGradient(SIZE / 2, SIZE * 0.55, 8, SIZE / 2, SIZE / 2, SIZE / 2);
    grad.addColorStop(0, '#7ed47a');
    grad.addColorStop(0.55, '#3da650');
    grad.addColorStop(1, '#1f5a2c');
    cctx.fillStyle = grad;
    cctx.beginPath();
    cctx.ellipse(SIZE / 2, SIZE / 2, SIZE / 2.45, SIZE / 2.05, 0, 0, Math.PI * 2);
    cctx.fill();

    cctx.strokeStyle = 'rgba(20, 60, 25, 0.5)';
    cctx.lineWidth = 1.6;
    cctx.beginPath();
    cctx.moveTo(SIZE / 2, 8);
    cctx.lineTo(SIZE / 2, SIZE - 8);
    for (let i = 1; i < 9; i++) {
        const t = i / 9;
        const yPos = 12 + t * (SIZE - 24);
        const len = (1 - Math.abs(t - 0.5) * 1.5) * SIZE / 2.6;
        cctx.moveTo(SIZE / 2, yPos);
        cctx.lineTo(SIZE / 2 + len, yPos - len * 0.4);
        cctx.moveTo(SIZE / 2, yPos);
        cctx.lineTo(SIZE / 2 - len, yPos - len * 0.4);
    }
    cctx.stroke();
    const colorTex = new THREE.CanvasTexture(colorCanvas);
    colorTex.colorSpace = THREE.SRGBColorSpace;

    const alphaCanvas = document.createElement('canvas');
    alphaCanvas.width = alphaCanvas.height = SIZE;
    const actx = alphaCanvas.getContext('2d');
    actx.fillStyle = '#000';
    actx.fillRect(0, 0, SIZE, SIZE);
    actx.fillStyle = '#fff';
    actx.beginPath();
    actx.ellipse(SIZE / 2, SIZE / 2, SIZE / 2.45, SIZE / 2.05, 0, 0, Math.PI * 2);
    actx.fill();
    const alphaTex = new THREE.CanvasTexture(alphaCanvas);

    const heightCanvas = document.createElement('canvas');
    heightCanvas.width = heightCanvas.height = SIZE;
    const hctx = heightCanvas.getContext('2d');
    hctx.fillStyle = '#7a7a7a';
    hctx.fillRect(0, 0, SIZE, SIZE);
    hctx.strokeStyle = '#a8a8a8';
    hctx.lineWidth = 4;
    hctx.beginPath();
    hctx.moveTo(SIZE / 2, 8);
    hctx.lineTo(SIZE / 2, SIZE - 8);
    hctx.stroke();
    hctx.strokeStyle = '#909090';
    hctx.lineWidth = 2;
    hctx.beginPath();
    for (let i = 1; i < 9; i++) {
        const t = i / 9;
        const yPos = 12 + t * (SIZE - 24);
        const len = (1 - Math.abs(t - 0.5) * 1.5) * SIZE / 2.6;
        hctx.moveTo(SIZE / 2, yPos);
        hctx.lineTo(SIZE / 2 + len, yPos - len * 0.4);
        hctx.moveTo(SIZE / 2, yPos);
        hctx.lineTo(SIZE / 2 - len, yPos - len * 0.4);
    }
    hctx.stroke();
    const normalTex = makeNormalMapFromCanvas(heightCanvas, 6);

    return new THREE.MeshStandardMaterial({
        map: colorTex,
        alphaMap: alphaTex,
        normalMap: normalTex,
        normalScale: new THREE.Vector2(0.6, 0.6),
        roughness: 0.65,
        metalness: 0,
        side: THREE.DoubleSide,
        transparent: true,
        alphaTest: 0.45
    });
}

// --- Haunted forest backdrop ---

function createDeadTreeTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 512);
    ctx.strokeStyle = '#0a0604';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Recursive gnarled branch generator
    function drawBranch(x1, y1, x2, y2, w, depth) {
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        const cx = (x1 + x2) / 2 + (Math.random() - 0.5) * w * 4;
        const cy = (y1 + y2) / 2 + (Math.random() - 0.5) * 8;
        ctx.quadraticCurveTo(cx, cy, x2, y2);
        ctx.stroke();
        if (depth <= 0 || w < 1.6) return;
        const len = Math.hypot(x2 - x1, y2 - y1);
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const branches = 2 + Math.floor(Math.random() * 2);
        for (let i = 0; i < branches; i++) {
            const ba = angle + (Math.random() - 0.5) * 1.4;
            const bl = len * (0.45 + Math.random() * 0.35);
            drawBranch(
                x2, y2,
                x2 + Math.cos(ba) * bl,
                y2 + Math.sin(ba) * bl,
                w * 0.55, depth - 1
            );
        }
    }

    // Trunk (roots near bottom, taper toward top)
    drawBranch(128, 510, 128 + (Math.random() - 0.5) * 20, 80, 18, 4);

    // Side branches off the trunk
    const sideCount = 4 + Math.floor(Math.random() * 3);
    for (let i = 0; i < sideCount; i++) {
        const yStart = 380 - i * (260 / sideCount);
        const isLeft = Math.random() > 0.5;
        const dx = (isLeft ? -1 : 1) * (40 + Math.random() * 70);
        const dy = -20 - Math.random() * 60;
        drawBranch(128, yStart, 128 + dx, yStart + dy, 5 + Math.random() * 4, 3);
    }

    return new THREE.CanvasTexture(canvas);
}

function buildHauntedForest() {
    if (sharedAssets.treeTex === undefined) {
        sharedAssets.treeTex = createDeadTreeTexture();
        sharedAssets.treeTex.colorSpace = THREE.SRGBColorSpace;
    }

    // Crossed planes per tree → merged into one BufferGeometry, instanced.
    const treeWidth = 5;
    const treeHeight = 10;
    const plane1 = new THREE.PlaneGeometry(treeWidth, treeHeight);
    plane1.translate(0, treeHeight / 2, 0);
    const plane2 = plane1.clone();
    plane2.rotateY(Math.PI / 2);
    const treeGeom = mergeGeometries([plane1, plane2]);

    treeMaterial = new THREE.MeshStandardMaterial({
        map: sharedAssets.treeTex,
        color: 0x1c1814,
        roughness: 1,
        metalness: 0,
        side: THREE.DoubleSide,
        transparent: true,
        alphaTest: 0.5
    });

    // Inject GPU-only wind sway. Top of tree displaces in local x/z based on time
    // and per-instance origin so trees don't all sway in unison.
    treeMaterial.onBeforeCompile = (shader) => {
        shader.uniforms.uTime = { value: 0 };
        shader.uniforms.uWindStrength = { value: 0 };
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', `
                #include <common>
                uniform float uTime;
                uniform float uWindStrength;
            `)
            .replace('#include <begin_vertex>', `
                #include <begin_vertex>
                #ifdef USE_INSTANCING
                    vec4 _instOrigin = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
                    float _phase = uTime + _instOrigin.x * 0.3 + _instOrigin.z * 0.21;
                #else
                    float _phase = uTime;
                #endif
                float _swayFactor = smoothstep(0.0, 7.0, position.y);
                float _swayX = sin(_phase) * uWindStrength * _swayFactor;
                float _swayZ = sin(_phase * 0.8 + 1.2) * uWindStrength * 0.5 * _swayFactor;
                transformed.x += _swayX;
                transformed.z += _swayZ;
            `);
        treeMaterial.userData.shader = shader;
    };

    // Place ~80 trees in a strip 10–28 m beyond the greenhouse glass
    const trees = [];
    const treeCount = 80;
    for (let i = 0; i < treeCount; i++) {
        const side = i % 4;
        const dist = 10 + Math.random() * 18;
        let x, z;
        if (side === 0)      { x = -8 - dist; z = -55 + Math.random() * 70; }
        else if (side === 1) { x =  8 + dist; z = -55 + Math.random() * 70; }
        else if (side === 2) { x = -28 + Math.random() * 56; z = -45 - dist; }
        else                 { x = -28 + Math.random() * 56; z =   5 + dist; }
        trees.push({
            x, z,
            scale: 0.7 + Math.random() * 0.7,
            rotY: Math.random() * Math.PI * 2
        });
    }

    const treesMesh = new THREE.InstancedMesh(treeGeom, treeMaterial, treeCount);
    const _m = new THREE.Matrix4();
    const _q = new THREE.Quaternion();
    const _yAxis = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < treeCount; i++) {
        const t = trees[i];
        _q.setFromAxisAngle(_yAxis, t.rotY);
        _m.compose(new THREE.Vector3(t.x, 0, t.z), _q, new THREE.Vector3(t.scale, t.scale, t.scale));
        treesMesh.setMatrixAt(i, _m);
    }
    treesMesh.instanceMatrix.needsUpdate = true;
    // Trees are far away — no shadow casting, no frustum culling on the
    // (incorrectly tiny) per-instance bounds.
    treesMesh.castShadow = false;
    treesMesh.receiveShadow = false;
    treesMesh.frustumCulled = false;
    scene.add(treesMesh);
}

// --- Glowing red eyes at the forest edge (night only) ---

function createEyeTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
    grad.addColorStop(0,   'rgba(255, 120, 60, 1)');
    grad.addColorStop(0.3, 'rgba(255, 30, 0, 0.85)');
    grad.addColorStop(0.7, 'rgba(180, 0, 0, 0.25)');
    grad.addColorStop(1,   'rgba(120, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
}

function buildHauntedEyes() {
    const tex = createEyeTexture();
    const PAIRS = 6;
    for (let i = 0; i < PAIRS; i++) {
        const mat = new THREE.SpriteMaterial({
            map: tex,
            color: 0xff2200,
            transparent: true,
            opacity: 0,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            fog: false
        });
        const left = new THREE.Sprite(mat);
        const right = new THREE.Sprite(mat);
        left.scale.set(0.16, 0.16, 1);
        right.scale.set(0.16, 0.16, 1);
        left.visible = false;
        right.visible = false;
        scene.add(left);
        scene.add(right);
        eyePairs.push({
            material: mat,
            left, right,
            state: 'off',
            // Stagger initial appearance so they don't all spawn together
            nextEvent: performance.now() + 4000 + Math.random() * 30000,
            transitionStart: 0,
            transitionEnd: 0
        });
    }
}

function pickEyeForestPoint() {
    const side = Math.floor(Math.random() * 4);
    const dist = 11 + Math.random() * 9;
    const y = 1.3 + Math.random() * 0.5;
    if (side === 0) return new THREE.Vector3(-8 - dist, y, -50 + Math.random() * 60);
    if (side === 1) return new THREE.Vector3( 8 + dist, y, -50 + Math.random() * 60);
    if (side === 2) return new THREE.Vector3(-25 + Math.random() * 50, y, -45 - dist);
    return                new THREE.Vector3(-25 + Math.random() * 50, y,   5 + dist);
}

function updateHauntedEyes(now) {
    const isNight = currentDayness < 0.4;
    for (const pair of eyePairs) {
        if (!isNight) {
            if (pair.state !== 'off') {
                pair.state = 'off';
                pair.left.visible = false;
                pair.right.visible = false;
                pair.material.opacity = 0;
                pair.nextEvent = now + 5000 + Math.random() * 10000;
            }
            continue;
        }
        switch (pair.state) {
            case 'off':
                if (now >= pair.nextEvent) {
                    const c = pickEyeForestPoint();
                    pair.left.position.set(c.x - 0.05, c.y, c.z);
                    pair.right.position.set(c.x + 0.05, c.y, c.z);
                    pair.left.visible = true;
                    pair.right.visible = true;
                    pair.state = 'fadeIn';
                    pair.transitionStart = now;
                    pair.transitionEnd = now + 1000 + Math.random() * 1200;
                }
                break;
            case 'fadeIn': {
                const t = (now - pair.transitionStart) / (pair.transitionEnd - pair.transitionStart);
                pair.material.opacity = Math.min(t, 1);
                if (t >= 1) {
                    pair.state = 'hold';
                    pair.nextEvent = now + 2000 + Math.random() * 4500;
                }
                break;
            }
            case 'hold':
                if (now >= pair.nextEvent) {
                    pair.state = 'fadeOut';
                    pair.transitionStart = now;
                    pair.transitionEnd = now + 700 + Math.random() * 700;
                }
                break;
            case 'fadeOut': {
                const t = (now - pair.transitionStart) / (pair.transitionEnd - pair.transitionStart);
                pair.material.opacity = Math.max(1 - t, 0);
                if (t >= 1) {
                    pair.state = 'off';
                    pair.left.visible = false;
                    pair.right.visible = false;
                    pair.nextEvent = now + 5000 + Math.random() * 18000;
                }
                break;
            }
        }
    }
}

function updateTreeWind(now) {
    if (!treeMaterial || !treeMaterial.userData.shader) return;
    const u = treeMaterial.userData.shader.uniforms;
    u.uTime.value = now / 1000;

    if (currentDayness < 0.5) {
        u.uWindStrength.value = 0;
        return;
    }
    // Roughly every 22 s, a 5 s gust during the day
    const cycle = (now / 1000) % 22;
    if (cycle < 5) {
        const t = cycle / 5;
        u.uWindStrength.value = Math.sin(t * Math.PI) * 0.09 + 0.004;
    } else {
        u.uWindStrength.value = 0.004;
    }
}

function buildFlower() {
    const group = new THREE.Group();

    if (!sharedAssets.flowerCenterMat) {
        sharedAssets.flowerCenterMat = new THREE.MeshStandardMaterial({
            color: 0xffd54f, roughness: 0.55, metalness: 0
        });
        sharedAssets.flowerPetalMat = new THREE.MeshStandardMaterial({
            color: 0xff7eb6, roughness: 0.55, metalness: 0
        });
        sharedAssets.flowerCenterGeom = new THREE.SphereGeometry(0.03, 12, 10);
        sharedAssets.flowerPetalGeom = new THREE.SphereGeometry(0.045, 12, 10);
        sharedAssets.flowerPetalGeom.scale(1, 0.32, 1.3);
    }

    const center = new THREE.Mesh(sharedAssets.flowerCenterGeom, sharedAssets.flowerCenterMat);
    group.add(center);

    const petalCount = 6;
    for (let i = 0; i < petalCount; i++) {
        const petal = new THREE.Mesh(sharedAssets.flowerPetalGeom, sharedAssets.flowerPetalMat);
        const a = (i / petalCount) * Math.PI * 2;
        const r = 0.05;
        petal.position.set(Math.cos(a) * r, -0.005, Math.sin(a) * r);
        petal.rotation.y = a;
        petal.rotation.x = -Math.PI / 9;
        group.add(petal);
    }

    return group;
}

function createLeafGeometry() {
    if (sharedAssets.leafGeom) return sharedAssets.leafGeom;
    const geom = new THREE.PlaneGeometry(0.18, 0.22, 6, 8);
    const pos = geom.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const z = -Math.pow(x / 0.09, 2) * 0.03 + Math.pow((y + 0.11) / 0.22, 1.5) * 0.022;
        pos.setZ(i, z);
    }
    geom.computeVertexNormals();
    sharedAssets.leafGeom = geom;
    return geom;
}

// --- Plant Generation Logic ---
const tablePositions = []; // To track where to put next plant

function buildGreenhouse() {
    // Collect table positions for plant placement
    for (let i = 0; i < 10; i++) {
        const zPos = -i * 4;

        // Define grid points on left table (2x3 grid)
        for(let xOffset = -0.5; xOffset <= 0.5; xOffset += 1.0) {
            for(let zOffset = -1.0; zOffset <= 1.0; zOffset += 1.0) {
                tablePositions.push(new THREE.Vector3(-3 + xOffset, 1.05, zPos + zOffset));
            }
        }

        // Define grid points on right table (2x3 grid)
        for(let xOffset = -0.5; xOffset <= 0.5; xOffset += 1.0) {
            for(let zOffset = -1.0; zOffset <= 1.0; zOffset += 1.0) {
                tablePositions.push(new THREE.Vector3(3 + xOffset, 1.05, zPos + zOffset));
            }
        }
    }

    // Empty pots — InstancedMesh (one per pot piece) for all 120 positions
    createEmptyPotsInstanced();

    // Floor
    const floorGeometry = new THREE.PlaneGeometry(200, 200);
    const floor = new THREE.Mesh(floorGeometry, getDirtFloorMaterial());
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Tables — InstancedMesh for tops + legs across all 20 tables
    const tableMaterial = makeWoodMaterial({ repeat: [2, 3], roughness: 0.78 });
    const numTables = 10;
    const tableSpacing = 4;
    const totalTables = numTables * 2;

    const topGeom = new THREE.BoxGeometry(2, 0.1, 3);
    const legGeom = new THREE.BoxGeometry(0.1, 1.0, 0.1);
    const topsMesh = new THREE.InstancedMesh(topGeom, tableMaterial, totalTables);
    const legsMesh = new THREE.InstancedMesh(legGeom, tableMaterial, totalTables * 4);
    topsMesh.castShadow = topsMesh.receiveShadow = true;
    legsMesh.castShadow = legsMesh.receiveShadow = true;

    const _m = new THREE.Matrix4();
    const _q = new THREE.Quaternion();
    const _s = new THREE.Vector3(1, 1, 1);
    const legOffsets = [
        [-0.9, -1.4], [0.9, -1.4],
        [-0.9, 1.4], [0.9, 1.4]
    ];
    let topIdx = 0;
    let legIdx = 0;
    for (let i = 0; i < numTables; i++) {
        const zPos = -i * tableSpacing;
        for (const x of [-3, 3]) {
            _m.compose(new THREE.Vector3(x, 1.0, zPos), _q, _s);
            topsMesh.setMatrixAt(topIdx++, _m);
            for (const [lx, lz] of legOffsets) {
                _m.compose(new THREE.Vector3(x + lx, 0.5, zPos + lz), _q, _s);
                legsMesh.setMatrixAt(legIdx++, _m);
            }
        }
    }
    topsMesh.instanceMatrix.needsUpdate = true;
    legsMesh.instanceMatrix.needsUpdate = true;
    scene.add(topsMesh);
    scene.add(legsMesh);

    // Greenhouse Structure
    const glassMat = getGlassMaterial();
    const copperMat = getCopperMaterial();

    const ghGroup = new THREE.Group();

    const woodMat = makeWoodMaterial({ repeat: [1, 8], roughness: 0.82, color: 0xc8a070 });
    // Weathered wood for the rafters/trusses — darker, more saturated
    const rafterMat = makeWoodMaterial({ repeat: [4, 1], roughness: 0.92, color: 0x8a6a48 });

    // Waist-level Wood Bases
    const baseHeight = 1.2;
    const wallHeight = 4.8; // glass section height (top of glass at y=6)
    const totalLength = 50;
    const totalWidth = 16;
    const zCenter = -20;
    const wallTopY = baseHeight + wallHeight; // 6
    const ridgeY = wallTopY + 5;               // 11 — peak of gable roof
    const halfWidth = totalWidth / 2;          // 8
    const slopeRise = ridgeY - wallTopY;       // 5
    const slopeRun = halfWidth;                // 8
    const slopeLength = Math.hypot(slopeRun, slopeRise); // ~9.434
    const slopeAngle = Math.atan2(slopeRise, slopeRun);  // ~32°

    // Wood Bases
    const leftBase = new THREE.Mesh(new THREE.BoxGeometry(0.2, baseHeight, totalLength), woodMat);
    leftBase.position.set(-8, baseHeight / 2, zCenter);
    ghGroup.add(leftBase);

    const rightBase = new THREE.Mesh(new THREE.BoxGeometry(0.2, baseHeight, totalLength), woodMat);
    rightBase.position.set(8, baseHeight / 2, zCenter);
    ghGroup.add(rightBase);

    const frontBase = new THREE.Mesh(new THREE.BoxGeometry(totalWidth, baseHeight, 0.2), woodMat);
    frontBase.position.set(0, baseHeight / 2, -45);
    ghGroup.add(frontBase);

    // Back base with a gap for the door
    const doorWidth = 2.0;
    const backBaseLeftWidth = (totalWidth - doorWidth) / 2;

    const backBaseLeft = new THREE.Mesh(new THREE.BoxGeometry(backBaseLeftWidth, baseHeight, 0.2), woodMat);
    backBaseLeft.position.set(-doorWidth / 2 - backBaseLeftWidth / 2, baseHeight / 2, 5);
    ghGroup.add(backBaseLeft);

    const backBaseRight = new THREE.Mesh(new THREE.BoxGeometry(backBaseLeftWidth, baseHeight, 0.2), woodMat);
    backBaseRight.position.set(doorWidth / 2 + backBaseLeftWidth / 2, baseHeight / 2, 5);
    ghGroup.add(backBaseRight);

    // Glass Walls (above the wood base)
    const leftWall = new THREE.Mesh(new THREE.BoxGeometry(0.2, wallHeight, totalLength), glassMat);
    leftWall.position.set(-8, baseHeight + wallHeight / 2, zCenter);
    ghGroup.add(leftWall);

    const rightWall = new THREE.Mesh(new THREE.BoxGeometry(0.2, wallHeight, totalLength), glassMat);
    rightWall.position.set(8, baseHeight + wallHeight / 2, zCenter);
    ghGroup.add(rightWall);

    const frontWall = new THREE.Mesh(new THREE.BoxGeometry(totalWidth, wallHeight, 0.2), glassMat);
    frontWall.position.set(0, baseHeight + wallHeight / 2, -45);
    ghGroup.add(frontWall);

    // Back Wall Glass (also with a gap for the door)
    const backWallLeft = new THREE.Mesh(new THREE.BoxGeometry(backBaseLeftWidth, wallHeight, 0.2), glassMat);
    backWallLeft.position.set(-doorWidth / 2 - backBaseLeftWidth / 2, baseHeight + wallHeight / 2, 5);
    ghGroup.add(backWallLeft);

    const backWallRight = new THREE.Mesh(new THREE.BoxGeometry(backBaseLeftWidth, wallHeight, 0.2), glassMat);
    backWallRight.position.set(doorWidth / 2 + backBaseLeftWidth / 2, baseHeight + wallHeight / 2, 5);
    ghGroup.add(backWallRight);

    // Top glass above the door
    const doorHeight = 4.0;
    if (baseHeight + wallHeight > doorHeight) {
        const topGlassHeight = (baseHeight + wallHeight) - doorHeight;
        const topGlass = new THREE.Mesh(new THREE.BoxGeometry(doorWidth, topGlassHeight, 0.2), glassMat);
        topGlass.position.set(0, doorHeight + topGlassHeight / 2, 5);
        ghGroup.add(topGlass);
    }

    // ---- Steeple (gable) roof: two flat panels meeting at the ridge ----
    const roofGeomBox = new THREE.BoxGeometry(slopeLength, 0.08, totalLength);

    const leftRoof = new THREE.Mesh(roofGeomBox, glassMat);
    leftRoof.position.set(-halfWidth / 2, (wallTopY + ridgeY) / 2, zCenter);
    leftRoof.rotation.z = slopeAngle;
    ghGroup.add(leftRoof);

    const rightRoof = new THREE.Mesh(roofGeomBox, glassMat);
    rightRoof.position.set(halfWidth / 2, (wallTopY + ridgeY) / 2, zCenter);
    rightRoof.rotation.z = -slopeAngle;
    ghGroup.add(rightRoof);

    // Triangular gable ends (front and back) — glass panes filling the gable
    const gableShape = new THREE.Shape();
    gableShape.moveTo(-halfWidth, 0);
    gableShape.lineTo(halfWidth, 0);
    gableShape.lineTo(0, slopeRise);
    gableShape.closePath();
    const gableGeom = new THREE.ExtrudeGeometry(gableShape, { depth: 0.1, bevelEnabled: false });

    const frontGable = new THREE.Mesh(gableGeom, glassMat);
    frontGable.position.set(0, wallTopY, -45);
    frontGable.rotation.y = Math.PI; // face inward
    ghGroup.add(frontGable);

    const backGable = new THREE.Mesh(gableGeom, glassMat);
    backGable.position.set(0, wallTopY, 5);
    ghGroup.add(backGable);

    // ---- Verdigris copper mullions (vertical bars dividing each glass wall into panes) ----
    const mullionThickness = 0.06;
    const mullionDepth = 0.08;

    // Long-wall mullions (left & right). 8 panes per wall → 9 mullions each.
    const longPanes = 8;
    const longMullionGeom = new THREE.BoxGeometry(mullionDepth, wallHeight, mullionThickness);
    for (let i = 0; i <= longPanes; i++) {
        const z = -45 + (totalLength / longPanes) * i;
        for (const x of [-halfWidth - 0.04, halfWidth + 0.04]) {
            const m = new THREE.Mesh(longMullionGeom, copperMat);
            m.position.set(x, wallTopY - wallHeight / 2, z);
            m.userData.detail = true;
            ghGroup.add(m);
        }
    }

    // Short-wall mullions (front & back). 4 panes per wall → 5 mullions, but skip door area on back.
    const shortPanes = 4;
    const shortMullionGeom = new THREE.BoxGeometry(mullionThickness, wallHeight, mullionDepth);
    for (let i = 0; i <= shortPanes; i++) {
        const x = -halfWidth + (totalWidth / shortPanes) * i;
        // Front wall (z = -45)
        const fm = new THREE.Mesh(shortMullionGeom, copperMat);
        fm.position.set(x, wallTopY - wallHeight / 2, -45 - 0.04);
        fm.userData.detail = true;
        ghGroup.add(fm);
        // Back wall (z = 5) — skip if mullion would land in the door gap
        if (Math.abs(x) > 1.05) {
            const bm = new THREE.Mesh(shortMullionGeom, copperMat);
            bm.position.set(x, wallTopY - wallHeight / 2, 5 + 0.04);
            bm.userData.detail = true;
            ghGroup.add(bm);
        }
    }

    // Horizontal mid-rail along all four walls (one long rail per wall)
    const midY = wallTopY - wallHeight / 2;
    const longRailGeom = new THREE.BoxGeometry(mullionDepth, mullionThickness, totalLength);
    for (const x of [-halfWidth - 0.04, halfWidth + 0.04]) {
        const rail = new THREE.Mesh(longRailGeom, copperMat);
        rail.position.set(x, midY, zCenter);
        rail.userData.detail = true;
        ghGroup.add(rail);
    }
    const frontRail = new THREE.Mesh(
        new THREE.BoxGeometry(totalWidth, mullionThickness, mullionDepth),
        copperMat
    );
    frontRail.position.set(0, midY, -45 - 0.04);
    frontRail.userData.detail = true;
    ghGroup.add(frontRail);

    // Cap rail at top of glass walls (along all four walls) — copper
    const longCapGeom = new THREE.BoxGeometry(0.12, 0.1, totalLength);
    for (const x of [-halfWidth, halfWidth]) {
        const cap = new THREE.Mesh(longCapGeom, copperMat);
        cap.position.set(x, wallTopY, zCenter);
        cap.userData.detail = true;
        ghGroup.add(cap);
    }
    const shortCapGeom = new THREE.BoxGeometry(totalWidth, 0.1, 0.12);
    const frontCap = new THREE.Mesh(shortCapGeom, copperMat);
    frontCap.position.set(0, wallTopY, -45);
    frontCap.userData.detail = true;
    ghGroup.add(frontCap);
    const backCap = new THREE.Mesh(shortCapGeom, copperMat);
    backCap.position.set(0, wallTopY, 5);
    backCap.userData.detail = true;
    ghGroup.add(backCap);

    // Glass Door at the back wall (entrance)
    const doorGroup = new THREE.Group();
    doorGroup.position.set(0, doorHeight / 2, 5.05); // Slightly offset from back wall

    // Door Frame
    const doorFrameMat = getMetalFrameMaterial();
    const doorFrameLeft = new THREE.Mesh(new THREE.BoxGeometry(0.1, doorHeight, 0.25), doorFrameMat);
    doorFrameLeft.position.set(-doorWidth / 2 + 0.05, 0, 0);
    doorGroup.add(doorFrameLeft);

    const doorFrameRight = new THREE.Mesh(new THREE.BoxGeometry(0.1, doorHeight, 0.25), doorFrameMat);
    doorFrameRight.position.set(doorWidth / 2 - 0.05, 0, 0);
    doorGroup.add(doorFrameRight);

    const doorFrameTop = new THREE.Mesh(new THREE.BoxGeometry(doorWidth, 0.1, 0.25), doorFrameMat);
    doorFrameTop.position.set(0, doorHeight / 2 - 0.05, 0);
    doorGroup.add(doorFrameTop);

    // Glass Pane
    const doorGlass = new THREE.Mesh(new THREE.BoxGeometry(doorWidth - 0.2, doorHeight - 0.1, 0.1), glassMat);
    doorGlass.position.set(0, -0.05, 0);
    doorGroup.add(doorGlass);

    // Door handle
    const handleMat = getDoorHandleMaterial();
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.4), handleMat);
    handle.position.set(doorWidth / 2 - 0.2, 0, 0.15);
    doorGroup.add(handle);

    ghGroup.add(doorGroup);

    // ---- Gable trusses (weathered wood) running across the width at intervals ----
    const trussCount = 9;
    const trussSpacing = totalLength / (trussCount - 1);
    const trussBeamThickness = 0.08;
    const trussBeamGeom = new THREE.BoxGeometry(slopeLength, trussBeamThickness, trussBeamThickness);
    const tieBeamGeom = new THREE.BoxGeometry(totalWidth, trussBeamThickness, trussBeamThickness);

    for (let i = 0; i < trussCount; i++) {
        const z = -45 + i * trussSpacing;
        // Left slope beam
        const leftBeam = new THREE.Mesh(trussBeamGeom, rafterMat);
        leftBeam.position.set(-halfWidth / 2, (wallTopY + ridgeY) / 2 - 0.06, z);
        leftBeam.rotation.z = slopeAngle;
        ghGroup.add(leftBeam);
        // Right slope beam
        const rightBeam = new THREE.Mesh(trussBeamGeom, rafterMat);
        rightBeam.position.set(halfWidth / 2, (wallTopY + ridgeY) / 2 - 0.06, z);
        rightBeam.rotation.z = -slopeAngle;
        ghGroup.add(rightBeam);
        // Horizontal tie beam at wall top
        const tieBeam = new THREE.Mesh(tieBeamGeom, rafterMat);
        tieBeam.position.set(0, wallTopY - 0.05, z);
        ghGroup.add(tieBeam);
    }

    // Ridge beam along the roof apex
    const ridge = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 0.12, totalLength),
        rafterMat
    );
    ridge.position.set(0, ridgeY - 0.06, zCenter);
    ghGroup.add(ridge);

    // ---- Trellis lattice over the central aisle with hanging Edison bulbs ----
    const trellisY = 3.0;
    const trellisBeamGeom = new THREE.BoxGeometry(0.07, 0.09, totalLength - 4);
    // Two long beams running the length
    for (const tx of [-1.8, 1.8]) {
        const beam = new THREE.Mesh(trellisBeamGeom, rafterMat);
        beam.position.set(tx, trellisY, zCenter);
        beam.userData.detail = true;
        ghGroup.add(beam);
    }
    // Cross slats every 1.5m
    const slatSpacing = 1.5;
    const slatCount = Math.floor((totalLength - 4) / slatSpacing);
    const slatStartZ = -45 + 2;
    const slatGeom = new THREE.BoxGeometry(3.8, 0.04, 0.04);
    for (let i = 0; i <= slatCount; i++) {
        const z = slatStartZ + i * slatSpacing;
        const slat = new THREE.Mesh(slatGeom, rafterMat);
        slat.position.set(0, trellisY + 0.05, z);
        slat.userData.detail = true;
        ghGroup.add(slat);
    }
    // A few diagonal lattice slats for visual texture
    const diagGeom = new THREE.BoxGeometry(0.04, 0.04, slatSpacing * 1.2);
    for (let i = 0; i < slatCount; i += 2) {
        const z = slatStartZ + i * slatSpacing + slatSpacing / 2;
        const diag = new THREE.Mesh(diagGeom, rafterMat);
        diag.position.set((i % 4 === 0 ? 1 : -1) * 0.9, trellisY + 0.1, z);
        diag.rotation.y = (i % 4 === 0 ? 1 : -1) * Math.PI / 5;
        diag.userData.detail = true;
        ghGroup.add(diag);
    }

    // Edison-style bulbs hanging from the trellis (one every 6m → ~9 bulbs)
    const bulbSpacing = 6;
    const bulbCount = Math.floor((totalLength - 4) / bulbSpacing) + 1;
    const cordMat = new THREE.MeshBasicMaterial({ color: 0x2b1f17 });
    const socketMat = new THREE.MeshStandardMaterial({ color: 0x363330, roughness: 0.55, metalness: 0.6 });
    const cordGeom = new THREE.CylinderGeometry(0.008, 0.008, 0.5, 6);
    const socketGeom = new THREE.CylinderGeometry(0.04, 0.045, 0.07, 10);
    const bulbGeom = new THREE.SphereGeometry(0.05, 12, 10);
    bulbGeom.scale(1, 1.25, 1);
    const filamentGeom = new THREE.TorusGeometry(0.012, 0.002, 4, 10);
    const filamentMat = new THREE.MeshBasicMaterial({ color: 0xffaa55 });

    for (let i = 0; i < bulbCount; i++) {
        const bz = slatStartZ + 1 + i * bulbSpacing;
        if (bz > 5 - 2) break;

        const cord = new THREE.Mesh(cordGeom, cordMat);
        cord.position.set(0, trellisY - 0.25, bz);
        cord.userData.detail = true;
        ghGroup.add(cord);

        const socket = new THREE.Mesh(socketGeom, socketMat);
        socket.position.set(0, trellisY - 0.5, bz);
        socket.userData.detail = true;
        ghGroup.add(socket);

        const bulbMat = makeBulbMaterial();
        const bulb = new THREE.Mesh(bulbGeom, bulbMat);
        bulb.position.set(0, trellisY - 0.6, bz);
        bulb.userData.detail = true;
        ghGroup.add(bulb);
        bulbMeshes.push(bulb);

        const filament = new THREE.Mesh(filamentGeom, filamentMat);
        filament.position.copy(bulb.position);
        filament.userData.detail = true;
        ghGroup.add(filament);

        const light = new THREE.PointLight(0xffaa55, 0, 6, 2);
        light.position.set(0, trellisY - 0.6, bz);
        light.castShadow = false;
        ghGroup.add(light);
        bulbLights.push(light);
    }

    // Selectively set shadow casting/receiving:
    // - Skip detail meshes (mullions, slats, bulbs) and transparent glass — they don't
    //   produce useful shadows but cost real GPU time.
    // - Opaque structural meshes still cast and receive shadows.
    ghGroup.traverse(obj => {
        if (!obj.isMesh) return;
        const isDetail = obj.userData.detail === true;
        const isGlass = obj.material === glassMat;
        obj.castShadow = !isDetail && !isGlass;
        obj.receiveShadow = !isDetail;
    });

    scene.add(ghGroup);
}

function createTable(x, z, material) {
    const tableGroup = new THREE.Group();

    // Top
    const topGeom = new THREE.BoxGeometry(2, 0.1, 3);
    const top = new THREE.Mesh(topGeom, material);
    top.position.y = 1.0; // Table height
    top.castShadow = true;
    top.receiveShadow = true;
    tableGroup.add(top);

    // Legs
    const legGeom = new THREE.BoxGeometry(0.1, 1.0, 0.1);
    const legPositions = [
        [-0.9, -1.4], [0.9, -1.4],
        [-0.9, 1.4], [0.9, 1.4]
    ];

    legPositions.forEach(pos => {
        const leg = new THREE.Mesh(legGeom, material);
        leg.position.set(pos[0], 0.5, pos[1]);
        leg.castShadow = true;
        tableGroup.add(leg);
    });

    tableGroup.position.set(x, 0, z);
    scene.add(tableGroup);
}

function buildPotMeshes(group) {
    const potMat = getPotMaterial();
    const soilMat = getSoilMaterial();

    const body = new THREE.Mesh(
        new THREE.CylinderGeometry(0.155, 0.105, 0.2, 28, 1),
        potMat
    );
    body.position.y = 0.1;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    const rim = new THREE.Mesh(
        new THREE.TorusGeometry(0.155, 0.012, 8, 28),
        potMat
    );
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.205;
    rim.castShadow = true;
    rim.receiveShadow = true;
    group.add(rim);

    // Soil mound (slightly domed)
    const soil = new THREE.Mesh(
        new THREE.SphereGeometry(0.142, 18, 12, 0, Math.PI * 2, 0, Math.PI / 2.5),
        soilMat
    );
    soil.position.y = 0.18;
    soil.scale.y = 0.45;
    soil.receiveShadow = true;
    soil.castShadow = true;
    group.add(soil);
}

function createEmptyPotsInstanced() {
    const count = tablePositions.length;
    const potMat = getPotMaterial();
    const soilMat = getSoilMaterial();

    // Lower-poly geometries for the instanced empty pots — they're seen at distance
    const bodyGeom = new THREE.CylinderGeometry(0.155, 0.105, 0.2, 18, 1);
    const rimGeom = new THREE.TorusGeometry(0.155, 0.012, 6, 18);
    const soilGeom = new THREE.SphereGeometry(0.142, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2.5);

    const bodies = new THREE.InstancedMesh(bodyGeom, potMat, count);
    const rims = new THREE.InstancedMesh(rimGeom, potMat, count);
    const soils = new THREE.InstancedMesh(soilGeom, soilMat, count);

    [bodies, rims, soils].forEach(m => {
        m.castShadow = true;
        m.receiveShadow = true;
        m.userData.isEmptyPotMesh = true;
    });

    const tmp = new THREE.Matrix4();
    const noRot = new THREE.Quaternion();
    const rimRot = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    const baseScale = new THREE.Vector3(1, 1, 1);
    const soilScale = new THREE.Vector3(1, 0.45, 1);

    for (let i = 0; i < count; i++) {
        const p = tablePositions[i];
        emptyPotOccupied.push(false);

        tmp.compose(new THREE.Vector3(p.x, p.y + 0.1, p.z), noRot, baseScale);
        bodies.setMatrixAt(i, tmp);

        tmp.compose(new THREE.Vector3(p.x, p.y + 0.205, p.z), rimRot, baseScale);
        rims.setMatrixAt(i, tmp);

        tmp.compose(new THREE.Vector3(p.x, p.y + 0.18, p.z), noRot, soilScale);
        soils.setMatrixAt(i, tmp);
    }
    bodies.instanceMatrix.needsUpdate = true;
    rims.instanceMatrix.needsUpdate = true;
    soils.instanceMatrix.needsUpdate = true;

    scene.add(bodies);
    scene.add(rims);
    scene.add(soils);

    emptyPotInstances = { bodies, rims, soils };
}

function setEmptyPotOccupied(index, occupied) {
    if (!emptyPotInstances) return;
    emptyPotOccupied[index] = occupied;
    const tmp = new THREE.Matrix4();
    const noRot = new THREE.Quaternion();
    const rimRot = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);

    if (occupied) {
        // Hide via zero-scale matrix
        const zero = new THREE.Matrix4().makeScale(0, 0, 0);
        emptyPotInstances.bodies.setMatrixAt(index, zero);
        emptyPotInstances.rims.setMatrixAt(index, zero);
        emptyPotInstances.soils.setMatrixAt(index, zero);
    } else {
        const p = tablePositions[index];
        const s = new THREE.Vector3(1, 1, 1);
        const soilS = new THREE.Vector3(1, 0.45, 1);
        tmp.compose(new THREE.Vector3(p.x, p.y + 0.1, p.z), noRot, s);
        emptyPotInstances.bodies.setMatrixAt(index, tmp);
        tmp.compose(new THREE.Vector3(p.x, p.y + 0.205, p.z), rimRot, s);
        emptyPotInstances.rims.setMatrixAt(index, tmp);
        tmp.compose(new THREE.Vector3(p.x, p.y + 0.18, p.z), noRot, soilS);
        emptyPotInstances.soils.setMatrixAt(index, tmp);
    }
    emptyPotInstances.bodies.instanceMatrix.needsUpdate = true;
    emptyPotInstances.rims.instanceMatrix.needsUpdate = true;
    emptyPotInstances.soils.instanceMatrix.needsUpdate = true;
}

function createPlant(todoData, isLoad = false) {
    let positionIndex = todoData.positionIndex;
    if (positionIndex === undefined) {
        console.error("No position index provided for plant.");
        return false;
    }

    const pos = tablePositions[positionIndex];

    // Hide the instanced empty pot at this slot (the plant brings its own pot meshes)
    setEmptyPotOccupied(positionIndex, true);

    // If a plant already lives here, dispose of it before creating the replacement
    const existingObjIndex = objects.findIndex(obj => obj.userData.positionIndex === positionIndex);
    if (existingObjIndex > -1) {
        const existingObj = objects[existingObjIndex];
        scene.remove(existingObj);
        disposeHierarchy(existingObj);
        objects.splice(existingObjIndex, 1);
    }

    // Plant Group
    const plantGroup = new THREE.Group();
    plantGroup.position.copy(pos);
    plantGroup.userData = {
        id: todoData.id,
        positionIndex: positionIndex,
        isEmpty: false
    };

    // 1. Pot + soil
    buildPotMeshes(plantGroup);

    if (todoData.completed) {
        // Render as blooming flower
        const stemGeom = new THREE.CylinderGeometry(0.014, 0.022, 0.4, 12);
        stemGeom.translate(0, 0.2, 0);
        const plantMat = new THREE.MeshStandardMaterial({
            color: 0x4caf50,
            roughness: 0.7,
            metalness: 0
        });
        const stem = new THREE.Mesh(stemGeom, plantMat);
        stem.position.y = 0.2;
        stem.castShadow = true;
        plantGroup.add(stem);

        const flower = buildFlower();
        flower.position.y = 0.42;
        stem.add(flower);

        // Slight bend
        stem.rotation.x = Math.PI / 12;
        stem.rotation.z = (Math.random() - 0.5) * 0.15;
    } else {
        // 3. Stem (will bend based on health)
        const stemGeom = new THREE.CylinderGeometry(0.014, 0.022, 0.4, 12);
        // Move pivot point to bottom of stem
        stemGeom.translate(0, 0.2, 0);
        const plantMat = new THREE.MeshStandardMaterial({
            color: 0x4caf50,
            roughness: 0.7,
            metalness: 0
        });
        const stem = new THREE.Mesh(stemGeom, plantMat);
        stem.position.y = 0.2; // Start at dirt level
        stem.castShadow = true;
        stem.name = "stem";
        plantGroup.add(stem);

        // 4. Leaves — multiple curved planes at varied angles for fullness
        if (!sharedLeafMat) sharedLeafMat = createLeafMaterial();
        const perPlantLeafMat = sharedLeafMat.clone(); // clone so we can color independently
        const leafGeom = createLeafGeometry();

        const leafConfigs = [
            { y: 0.18, ry: 0,           rz: -Math.PI / 3.2, scale: 1.0 },
            { y: 0.22, ry: Math.PI / 2, rz: -Math.PI / 3.2, scale: 0.95 },
            { y: 0.28, ry: Math.PI / 4, rz:  Math.PI / 3.2, scale: 1.05 },
            { y: 0.32, ry: Math.PI * 0.75, rz: -Math.PI / 4, scale: 0.9 },
            { y: 0.36, ry: Math.PI * 1.2, rz:  Math.PI / 3.5, scale: 0.85 }
        ];

        leafConfigs.forEach((cfg, i) => {
            const leaf = new THREE.Mesh(leafGeom, perPlantLeafMat);
            leaf.position.set(0, cfg.y, 0);
            leaf.rotation.set(0, cfg.ry, cfg.rz);
            leaf.scale.setScalar(cfg.scale * 0.9);
            // Shadow casting disabled — alpha-tested shadows are expensive and
            // leaves are too small to read clearly in shadow anyway.
            leaf.name = i === 0 ? "leaf1" : (i === 1 ? "leaf2" : `leaf${i + 1}`);
            stem.add(leaf);
        });
    }

    // Save reference for interaction and updates
    objects.push(plantGroup);
    scene.add(plantGroup);

    // Bind mesh to data
    todoData.mesh = plantGroup;

    if (!todoData.completed) {
        updatePlantVisual(todoData);
    }

    return true;
}

// --- Sun position (SunCalc algorithm) and day/night lighting ---

function computeSunPosition(date, lat, lng) {
    const rad = Math.PI / 180;
    const J1970 = 2440588;
    const J2000 = 2451545;
    const e = rad * 23.4397; // obliquity of the Earth

    const toJulian = (d) => d.getTime() / 86400000 - 0.5 + J1970;
    const toDays = (d) => toJulian(d) - J2000;

    const rightAsc = (l, b) => Math.atan2(Math.sin(l) * Math.cos(e) - Math.tan(b) * Math.sin(e), Math.cos(l));
    const decl = (l, b) => Math.asin(Math.sin(b) * Math.cos(e) + Math.cos(b) * Math.sin(e) * Math.sin(l));
    const azimuthFn = (H, phi, dec) => Math.atan2(Math.sin(H), Math.cos(H) * Math.sin(phi) - Math.tan(dec) * Math.cos(phi));
    const altitudeFn = (H, phi, dec) => Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(H));
    const sidereal = (d, lw) => rad * (280.16 + 360.9856235 * d) - lw;

    const sunMeanAnomaly = (d) => rad * (357.5291 + 0.98560028 * d);
    const eclipticLong = (M) => {
        const C = rad * (1.9148 * Math.sin(M) + 0.02 * Math.sin(2 * M) + 0.0003 * Math.sin(3 * M));
        const P = rad * 102.9372;
        return M + C + P + Math.PI;
    };

    const d = toDays(date);
    const M = sunMeanAnomaly(d);
    const L = eclipticLong(M);
    const ra = rightAsc(L, 0);
    const dec = decl(L, 0);

    const lw = rad * -lng;
    const phi = rad * lat;
    const H = sidereal(d, lw) - ra;

    return {
        altitude: altitudeFn(H, phi, dec),     // radians; >0 means above horizon
        azimuth: azimuthFn(H, phi, dec) + Math.PI // radians from north (0=N, π/2=E, π=S, 3π/2=W)
    };
}

function updateSunAndLighting() {
    if (!sky || !sunLight) return;

    const sun = computeSunPosition(new Date(), SUN_LOCATION.lat, SUN_LOCATION.lng);
    const elev = sun.altitude;
    const azim = sun.azimuth;
    const altDeg = elev * 180 / Math.PI;

    // World convention: north = -Z, east = +X.
    const dir = new THREE.Vector3(
        Math.sin(azim) * Math.cos(elev),
        Math.sin(elev),
        -Math.cos(azim) * Math.cos(elev)
    );

    sky.material.uniforms.sunPosition.value.copy(dir);
    sunLight.position.copy(dir).multiplyScalar(40);

    // dayness: 1 fully day, 0 fully night, smooth between altitude -6° → +5°
    const dayness = THREE.MathUtils.clamp((altDeg + 6) / 11, 0, 1);
    const nightness = 1 - dayness;
    currentDayness = dayness;

    sunLight.intensity = 3.0 * dayness;
    skyFill.intensity = 0.45 * dayness + 0.04 * nightness;
    warmFill.intensity = 0.6 * dayness + 0.05 * nightness;

    // Atmosphere: thinner Rayleigh and lower turbidity at night → darker sky
    sky.material.uniforms.rayleigh.value = 1.4 * dayness + 0.25 * nightness;
    sky.material.uniforms.turbidity.value = 6 * dayness + 1.2 * nightness;

    // Edison bulbs glow at night. Setting visible=false prunes them from the
    // PBR shader's light list entirely — big win during the day.
    const bulbsOn = nightness > 0.01;
    bulbLights.forEach(light => {
        light.visible = bulbsOn;
        light.intensity = nightness * 4.5;
    });
    bulbMeshes.forEach(mesh => {
        mesh.material.emissiveIntensity = nightness * 1.6;
    });
}

// --- Raycasting helpers — handle both regular Plant Groups and InstancedMesh empty pots ---
function gatherIntersectables() {
    const list = [];
    for (const group of objects) {
        for (const child of group.children) list.push(child);
    }
    if (emptyPotInstances) {
        list.push(emptyPotInstances.bodies);
        list.push(emptyPotInstances.rims);
        list.push(emptyPotInstances.soils);
    }
    return list;
}

// Classify the first raycast hit. Returns { kind: 'empty'|'plant'|null, ... }
function classifyHit(hit) {
    if (!hit || !hit.object) return { kind: null };
    const obj = hit.object;
    if (obj.userData && obj.userData.isEmptyPotMesh) {
        const idx = hit.instanceId;
        if (idx === undefined || emptyPotOccupied[idx]) return { kind: null };
        return { kind: 'empty', index: idx };
    }
    let target = obj;
    while (target && target.userData && target.userData.id === undefined) {
        target = target.parent;
        if (!target || target === scene) return { kind: null };
    }
    if (target.userData && target.userData.id) {
        const todo = todos.find(t => t.id === target.userData.id);
        if (todo) return { kind: 'plant', todo };
    }
    return { kind: null };
}

// Helper to clean up 3D objects
function disposeHierarchy(node) {
    if (!node) return;

    if (node.children) {
        for (let i = node.children.length - 1; i >= 0; i--) {
            disposeHierarchy(node.children[i]);
        }
    }

    if (node.geometry) {
        node.geometry.dispose();
    }

    if (node.material) {
        if (Array.isArray(node.material)) {
            node.material.forEach(mat => mat.dispose());
        } else {
            node.material.dispose();
        }
    }
}

// UI Event Listeners for adding todos
document.getElementById('add-todo-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const title = document.getElementById('todo-title').value;
    const desc = document.getElementById('todo-desc').value;
    const urgency = parseInt(document.getElementById('todo-urgency').value);

    if (activePotIndex === null) {
        console.error("No pot selected to plant seed.");
        return;
    }

    const newTodo = {
        id: Date.now(),
        title: title,
        desc: desc,
        urgency: urgency,
        createdAt: Date.now(),
        lastUpdated: Date.now(),
        health: 100, // 0 to 100
        positionIndex: activePotIndex,
        status: "Not Started",
        completed: false
    };

    if (createPlant(newTodo)) {
        todos.push(newTodo);
        saveTodosToLocal();
        this.reset();
        closeAddTodoModal();
    }
});

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (composer) composer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    const time = performance.now();

    if (controls.isLocked === true || mobileActive) {
        const delta = (time - prevTime) / 1000;

        velocity.x -= velocity.x * 10.0 * delta;
        velocity.z -= velocity.z * 10.0 * delta;

        direction.z = Number(moveForward) - Number(moveBackward);
        direction.x = Number(moveRight) - Number(moveLeft);
        direction.normalize(); // this ensures consistent movements in all directions

        if (moveForward || moveBackward) velocity.z -= direction.z * 40.0 * delta;
        if (moveLeft || moveRight) velocity.x -= direction.x * 40.0 * delta;

        controls.moveRight(-velocity.x * delta);
        controls.moveForward(-velocity.z * delta);

        // Boundary collision detection (keep player inside greenhouse)
        const pos = controls.getObject().position;
        if (pos.x < -7.5) pos.x = -7.5;
        if (pos.x > 7.5) pos.x = 7.5;
        if (pos.z < -44.5) pos.z = -44.5;
        if (pos.z > 4.5) pos.z = 4.5;

        // Hover raycasting
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(gatherIntersectables(), false);
        const tooltip = document.getElementById('hover-tooltip');
        const hit = classifyHit(intersects[0]);
        if (hit.kind === 'empty') {
            tooltip.textContent = "Click to plant a new to-do";
            tooltip.style.display = 'block';
        } else if (hit.kind === 'plant') {
            const todo = hit.todo;
            tooltip.textContent = todo.completed
                ? `Completed: ${todo.title}`
                : `${todo.title}\n[${todo.status || "Not Started"}]`;
            tooltip.style.display = 'block';
        } else {
            tooltip.style.display = 'none';
        }
    } else {
        document.getElementById('hover-tooltip').style.display = 'none';
    }

    // Update plant decay
    updateDecay();

    // Refresh sun position every 30s — slow real-time motion
    if (time - lastSunUpdate > 30000) {
        lastSunUpdate = time;
        updateSunAndLighting();
    }

    // Atmosphere — wind sway (GPU-side, just a uniform write) + glowing eyes state
    updateTreeWind(time);
    updateHauntedEyes(time);

    prevTime = time;

    if (composer) {
        composer.render();
    } else {
        renderer.render(scene, camera);
    }
}

// --- Decay Logic ---

function getCurrentSimulatedTime() {
    return Date.now() + simulatedTimeOffset;
}

function updateDecay() {
    const currentTime = getCurrentSimulatedTime();

    todos.forEach(todo => {
        if (todo.completed) return; // No decay for completed/blooming plants

        // Calculate time since last update in ms
        const timeElapsed = currentTime - todo.lastUpdated;

        // Convert to "days" for calculation (e.g., 1 day real-time = decay by X)
        // For testing, let's say 1 "real" day = 86400000 ms.
        // We will speed it up for visual purposes if not using fast forward,
        // but fast forward jumps time by 1 day.
        // Base decay rate: loose 20% health per day.
        const daysElapsed = timeElapsed / (1000 * 60 * 60 * 24);

        // Urgency multiplier:
        // Low (1) = 0.5x (decay slower)
        // Medium (2) = 1.0x (normal)
        // High (3) = 2.0x (decay faster)
        let multiplier = 1.0;
        if (todo.urgency === 1) multiplier = 0.5;
        if (todo.urgency === 3) multiplier = 2.0;

        // Calculate new health
        // E.g., drops 20 health per day * multiplier
        let healthLoss = daysElapsed * 20 * multiplier;
        todo.health = Math.max(0, 100 - healthLoss);

        // Update visual
        updatePlantVisual(todo);
    });
}

function updatePlantVisual(todo) {
    if (!todo.mesh) return;

    const stem = todo.mesh.getObjectByName("stem");
    if (!stem) return;

    // Health is 0 to 100
    const healthRatio = todo.health / 100;

    // Color transition: Green (0x2ecc71) to Brown/Yellow (0x8b8000 / 0x8b4513)
    const healthyColor = new THREE.Color(0x2ecc71);
    const deadColor = new THREE.Color(0x8b4513);
    const currentColor = healthyColor.clone().lerp(deadColor, 1 - healthRatio);

    // Update materials
    stem.material.color.copy(currentColor);

    const leaf1 = stem.getObjectByName("leaf1");
    if (leaf1) leaf1.material.color.copy(currentColor);

    const leaf2 = stem.getObjectByName("leaf2");
    if (leaf2) leaf2.material.color.copy(currentColor);

    // Drooping effect (rotate stem based on health)
    // 100 health = 0 rotation, 0 health = Math.PI / 2.5 (bent over)
    const targetRotationX = (1 - healthRatio) * (Math.PI / 2.5);
    stem.rotation.x = targetRotationX;

    // Scale effect (shrinks slightly as it dies)
    const targetScale = 0.5 + (healthRatio * 0.5);
    stem.scale.set(1, targetScale, 1);
}

document.getElementById('fast-forward-btn').addEventListener('click', function() {
    // Jump forward 1 day (86400000 ms)
    simulatedTimeOffset += 86400000;
    console.log("Fast forwarded 1 day. Current offset:", simulatedTimeOffset);
});

document.getElementById('clear-save-btn').addEventListener('click', function() {
    if (confirm("Are you sure you want to delete all your plants and clear save data?")) {
        localStorage.removeItem(STORAGE_KEY);
        location.reload();
    }
});

// --- Interaction Logic ---
let activeTodo = null;
let activePotIndex = null;

// Add click listener for raycasting
document.addEventListener('click', function() {
    if (!controls.isLocked) return;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(gatherIntersectables(), false);
    const hit = classifyHit(intersects[0]);
    if (hit.kind === 'empty') {
        activePotIndex = hit.index;
        openAddTodoModal();
    } else if (hit.kind === 'plant' && !hit.todo.completed) {
        openTodoModal(hit.todo);
    }
});

function openAddTodoModal() {
    pauseForModal();
    document.getElementById('add-todo-modal').style.display = 'flex';
}

function closeAddTodoModal() {
    document.getElementById('add-todo-modal').style.display = 'none';
    activePotIndex = null;
    startExploring();
}

document.getElementById('close-add-modal').addEventListener('click', closeAddTodoModal);

function openTodoModal(todo) {
    activeTodo = todo;
    pauseForModal();

    document.getElementById('modal-title').textContent = todo.title;
    document.getElementById('modal-desc').textContent = todo.desc;
    document.getElementById('modal-health').textContent = Math.round(todo.health) + '%';
    document.getElementById('modal-status').textContent = todo.status || "Not Started";

    let urgencyText = "Medium";
    if (todo.urgency === 1) urgencyText = "Low";
    if (todo.urgency === 3) urgencyText = "High";
    document.getElementById('modal-urgency').textContent = urgencyText;

    document.getElementById('todo-effort').value = "0";

    document.getElementById('todo-modal').style.display = 'flex';
}

function closeTodoModal() {
    document.getElementById('todo-modal').style.display = 'none';
    activeTodo = null;
    startExploring();
}

document.getElementById('close-modal').addEventListener('click', closeTodoModal);

// Status buttons
const statusButtons = [
    { id: 'btn-status-procrastinating', text: 'Procrastinating' },
    { id: 'btn-status-inprogress', text: 'In Progress' },
    { id: 'btn-status-almostdone', text: 'Almost Done' }
];

statusButtons.forEach(btnInfo => {
    document.getElementById(btnInfo.id).addEventListener('click', () => {
        if (activeTodo) {
            activeTodo.status = btnInfo.text;
            document.getElementById('modal-status').textContent = btnInfo.text;
            saveTodosToLocal();
        }
    });
});

document.getElementById('btn-checkin').addEventListener('click', function() {
    if (activeTodo) {
        const effortBoost = parseInt(document.getElementById('todo-effort').value);
        activeTodo.health = Math.min(100, activeTodo.health + effortBoost);
        activeTodo.lastUpdated = getCurrentSimulatedTime();

        updatePlantVisual(activeTodo);
        saveTodosToLocal();
        closeTodoModal();
    }
});

init();
animate();

document.getElementById('btn-complete').addEventListener('click', function() {
    if (activeTodo) {
        activeTodo.completed = true;
        activeTodo.health = 100;
        activeTodo.status = "Completed";

        saveTodosToLocal();

        // Recreate the plant visually to show the flower
        createPlant(activeTodo);

        closeTodoModal();
    }
});

// --- Mobile / touch helpers ---

function resetMovement() {
    moveForward = false;
    moveBackward = false;
    moveLeft = false;
    moveRight = false;
}

function startExploring() {
    if (isTouchDevice) {
        mobileActive = true;
        blocker.style.display = 'none';
        uiContainer.style.display = 'none';
        document.getElementById('mobile-controls').classList.add('active');
    } else {
        controls.lock();
    }
}

function pauseForModal() {
    if (isTouchDevice) {
        mobileActive = false;
        resetMovement();
        document.getElementById('mobile-controls').classList.remove('active');
    } else {
        controls.unlock();
    }
}

function showMobileMenu() {
    mobileActive = false;
    resetMovement();
    document.getElementById('mobile-controls').classList.remove('active');
    uiContainer.style.display = 'block';
}

function setupTouchControls() {
    const joystick = document.getElementById('joystick');
    const stick = document.getElementById('stick');
    const lookZone = document.getElementById('look-zone');
    const menuBtn = document.getElementById('mobile-menu-btn');
    const lookEuler = new THREE.Euler(0, 0, 0, 'YXZ');
    const JOY_RADIUS = 50;
    const TAP_THRESHOLD_PX = 10;
    const LOOK_SENSITIVITY = 0.005;

    // ----- Joystick -----
    let joyTouchId = null;
    let joyCenterX = 0;
    let joyCenterY = 0;

    joystick.addEventListener('touchstart', (e) => {
        if (joyTouchId !== null) return;
        e.preventDefault();
        const touch = e.changedTouches[0];
        joyTouchId = touch.identifier;
        const rect = joystick.getBoundingClientRect();
        joyCenterX = rect.left + rect.width / 2;
        joyCenterY = rect.top + rect.height / 2;
    }, { passive: false });

    joystick.addEventListener('touchmove', (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier !== joyTouchId) continue;
            e.preventDefault();
            const dx = touch.clientX - joyCenterX;
            const dy = touch.clientY - joyCenterY;
            const dist = Math.min(JOY_RADIUS, Math.hypot(dx, dy));
            const angle = Math.atan2(dy, dx);
            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;
            stick.style.transform = `translate(${sx}px, ${sy}px)`;
            const nx = sx / JOY_RADIUS;
            const ny = sy / JOY_RADIUS;
            moveLeft = nx < -0.3;
            moveRight = nx > 0.3;
            moveForward = ny < -0.3;
            moveBackward = ny > 0.3;
        }
    }, { passive: false });

    function endJoystick(e) {
        for (const touch of e.changedTouches) {
            if (touch.identifier !== joyTouchId) continue;
            e.preventDefault();
            joyTouchId = null;
            stick.style.transform = '';
            resetMovement();
        }
    }
    joystick.addEventListener('touchend', endJoystick, { passive: false });
    joystick.addEventListener('touchcancel', endJoystick, { passive: false });

    // ----- Look zone (drag to rotate, tap to interact) -----
    let lookTouchId = null;
    let lookLastX = 0;
    let lookLastY = 0;
    let lookMoved = 0;

    lookZone.addEventListener('touchstart', (e) => {
        if (lookTouchId !== null) return;
        e.preventDefault();
        const touch = e.changedTouches[0];
        lookTouchId = touch.identifier;
        lookLastX = touch.clientX;
        lookLastY = touch.clientY;
        lookMoved = 0;
    }, { passive: false });

    lookZone.addEventListener('touchmove', (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier !== lookTouchId) continue;
            e.preventDefault();
            const dx = touch.clientX - lookLastX;
            const dy = touch.clientY - lookLastY;
            lookLastX = touch.clientX;
            lookLastY = touch.clientY;
            lookMoved += Math.abs(dx) + Math.abs(dy);

            lookEuler.setFromQuaternion(camera.quaternion);
            lookEuler.y -= dx * LOOK_SENSITIVITY;
            lookEuler.x -= dy * LOOK_SENSITIVITY;
            lookEuler.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, lookEuler.x));
            camera.quaternion.setFromEuler(lookEuler);
        }
    }, { passive: false });

    function endLook(e) {
        for (const touch of e.changedTouches) {
            if (touch.identifier !== lookTouchId) continue;
            const tappedX = touch.clientX;
            const tappedY = touch.clientY;
            const wasTap = lookMoved < TAP_THRESHOLD_PX;
            lookTouchId = null;
            if (wasTap && mobileActive) {
                performTapInteraction(tappedX, tappedY);
            }
        }
    }
    lookZone.addEventListener('touchend', endLook, { passive: false });
    lookZone.addEventListener('touchcancel', endLook, { passive: false });

    // ----- Menu button -----
    menuBtn.addEventListener('click', showMobileMenu);
}

function performTapInteraction(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    const tapMouse = new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(tapMouse, camera);
    const intersects = raycaster.intersectObjects(gatherIntersectables(), false);
    const hit = classifyHit(intersects[0]);
    if (hit.kind === 'empty') {
        activePotIndex = hit.index;
        openAddTodoModal();
    } else if (hit.kind === 'plant' && !hit.todo.completed) {
        openTodoModal(hit.todo);
    }
}