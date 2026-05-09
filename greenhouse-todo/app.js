let camera, scene, renderer, controls;
let raycaster, mouse;

const objects = []; // Interactable objects (plants)
let todos = []; // Data for todos

// Time tracking
let simulatedTimeOffset = 0; // Fast forward offset in ms

// Local Storage keys
const STORAGE_KEY = 'greenhouse-todos-data';

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
    // 1. Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB); // Sky blue
    scene.fog = new THREE.FogExp2(0x87CEEB, 0.02);

    // 2. Camera setup
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.y = 1.6; // Average human height

    // 3. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.top = 20;
    directionalLight.shadow.camera.bottom = -20;
    directionalLight.shadow.camera.left = -20;
    directionalLight.shadow.camera.right = 20;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // 4. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.body.appendChild(renderer.domElement);

    // 5. Controls
    controls = new THREE.PointerLockControls(camera, document.body);

    instructions.addEventListener('click', function () {
        controls.lock();
    });

    resumeBtn.addEventListener('click', function() {
        controls.lock();
    });

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

    // 9. Load Saved Data
    loadTodosFromLocal();

    // Window resize handler
    window.addEventListener('resize', onWindowResize);
}

// --- Texture Generation ---
function createWoodTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = "#8b5a2b";
    ctx.fillRect(0, 0, 512, 512);
    for(let i = 0; i < 200; i++) {
        ctx.fillStyle = "rgba(0, 0, 0, " + (Math.random() * 0.1) + ")";
        ctx.fillRect(Math.random() * 512, 0, Math.random() * 10, 512);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(1, 1);
    return texture;
}

function createDirtTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = "#3d2817";
    ctx.fillRect(0, 0, 512, 512);
    for(let i = 0; i < 5000; i++) {
        const shade = Math.random() > 0.5 ? 0 : 255;
        ctx.fillStyle = "rgba(" + shade + "," + shade + "," + shade + "," + (Math.random() * 0.05) + ")";
        ctx.fillRect(Math.random() * 512, Math.random() * 512, 2, 2);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(20, 20); // Tile it a bit
    return texture;
}

function createGlassTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Tinted green background
    ctx.fillStyle = "rgba(180, 240, 200, 0.2)";
    ctx.fillRect(0, 0, 512, 512);

    // Add metal scaffolding lines
    ctx.strokeStyle = "#405040"; // Dark greenish-grey metal
    ctx.lineWidth = 16;

    // Border
    ctx.strokeRect(0, 0, 512, 512);

    // Crossbars for scaffolding
    ctx.beginPath();
    ctx.moveTo(256, 0);
    ctx.lineTo(256, 512);
    ctx.moveTo(0, 256);
    ctx.lineTo(512, 256);
    ctx.stroke();

    // Smudges
    for(let i = 0; i < 20; i++) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
        ctx.beginPath();
        ctx.arc(Math.random() * 512, Math.random() * 512, Math.random() * 50, 0, Math.PI * 2);
        ctx.fill();
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(4, 4); // Tile the glass panels
    return texture;
}

function createLeafTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = "#2ecc71"; // Base green
    ctx.fillRect(0, 0, 256, 256);

    // Veins
    ctx.strokeStyle = "#27ae60";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(128, 0);
    ctx.lineTo(128, 256); // center vein

    // Side veins
    for(let i = 0; i < 8; i++) {
        let y = i * 30 + 15;
        ctx.moveTo(128, y);
        ctx.lineTo(128 + 60, y - 40);
        ctx.moveTo(128, y);
        ctx.lineTo(128 - 60, y - 40);
    }
    ctx.stroke();
    return new THREE.CanvasTexture(canvas);
}

function createLeafAlphaTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Black background (transparent)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 256, 256);

    // White leaf shape (opaque)
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.ellipse(128, 128, 60, 120, 0, 0, Math.PI * 2);
    ctx.fill();

    return new THREE.CanvasTexture(canvas);
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

    // Instantiate empty pots at all positions
    for (let i = 0; i < tablePositions.length; i++) {
        createEmptyPot(i, tablePositions[i]);
    }

    // Floor
    const floorGeometry = new THREE.PlaneGeometry(100, 100);
    const floorMaterial = new THREE.MeshStandardMaterial({
        map: createDirtTexture(),
        roughness: 1.0
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Tables (Procedural layout along a central aisle)
    const tableMaterial = new THREE.MeshStandardMaterial({ map: createWoodTexture() });
    const numTables = 10;
    const tableSpacing = 4;

    for (let i = 0; i < numTables; i++) {
        const zPos = -i * tableSpacing;

        // Left table
        createTable(-3, zPos, tableMaterial);

        // Right table
        createTable(3, zPos, tableMaterial);
    }

    // Greenhouse Structure
    const glassMat = new THREE.MeshPhysicalMaterial({
        map: createGlassTexture(),
        color: 0xaaffaa, // Pale green tint
        transmission: 0.9,
        opacity: 1,
        transparent: true,
        roughness: 0.1,
        metalness: 0.1,
        side: THREE.DoubleSide
    });

    const ghGroup = new THREE.Group();

    const woodMat = new THREE.MeshStandardMaterial({ map: createWoodTexture() });

    // Waist-level Wood Bases
    const baseHeight = 1.2;
    const wallHeight = 4.8; // Total height 6m before roof
    const totalLength = 50;
    const totalWidth = 16;
    const zCenter = -20;

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

    // Curved Arch Roof
    const roofGeom = new THREE.CylinderGeometry(8, 8, totalLength, 12, 1, false, 0, Math.PI);
    const roof = new THREE.Mesh(roofGeom, glassMat);
    roof.rotation.x = -Math.PI / 2;
    roof.rotation.y = -Math.PI / 2;
    roof.position.set(0, baseHeight + wallHeight, zCenter);
    ghGroup.add(roof);

    // Front and Back Glass Half-Circles (filling the arches)
    const archGeom = new THREE.CylinderGeometry(8, 8, 0.2, 12, 1, false, 0, Math.PI);

    const frontArch = new THREE.Mesh(archGeom, glassMat);
    frontArch.rotation.x = -Math.PI / 2;
    frontArch.position.set(0, baseHeight + wallHeight, -45);
    ghGroup.add(frontArch);

    const backArch = new THREE.Mesh(archGeom, glassMat);
    backArch.rotation.x = -Math.PI / 2;
    backArch.position.set(0, baseHeight + wallHeight, 5);
    ghGroup.add(backArch);

    // Glass Door at the back wall (entrance)
    const doorGroup = new THREE.Group();
    doorGroup.position.set(0, doorHeight / 2, 5.05); // Slightly offset from back wall

    // Door Frame
    const doorFrameMat = new THREE.MeshStandardMaterial({ color: 0x405040 }); // Dark metal frame
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
    const handleMat = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.8, roughness: 0.2 });
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.4), handleMat);
    handle.position.set(doorWidth / 2 - 0.2, 0, 0.15);
    doorGroup.add(handle);

    ghGroup.add(doorGroup);

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

function createEmptyPot(index, pos) {
    const potGroup = new THREE.Group();
    potGroup.position.copy(pos);
    potGroup.userData = {
        isEmpty: true,
        positionIndex: index
    };

    const potGeom = new THREE.CylinderGeometry(0.15, 0.1, 0.2, 16);
    const potMat = new THREE.MeshStandardMaterial({ color: 0xcd5c5c, roughness: 0.9 });
    const pot = new THREE.Mesh(potGeom, potMat);
    pot.position.y = 0.1;
    pot.castShadow = true;
    potGroup.add(pot);

    const dirtGeom = new THREE.CylinderGeometry(0.14, 0.14, 0.05, 16);
    const dirtMat = new THREE.MeshStandardMaterial({ color: 0x2e1a0b });
    const dirt = new THREE.Mesh(dirtGeom, dirtMat);
    dirt.position.y = 0.18;
    potGroup.add(dirt);

    objects.push(potGroup);
    scene.add(potGroup);
}

function createPlant(todoData, isLoad = false) {
    let positionIndex = todoData.positionIndex;
    if (positionIndex === undefined) {
        console.error("No position index provided for plant.");
        return false;
    }

    const pos = tablePositions[positionIndex];

    // Find and remove the existing empty pot or plant at this position
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

    // 1. Pot
    const potGeom = new THREE.CylinderGeometry(0.15, 0.1, 0.2, 16);
    const potMat = new THREE.MeshStandardMaterial({ color: 0xcd5c5c, roughness: 0.9 });
    const pot = new THREE.Mesh(potGeom, potMat);
    pot.position.y = 0.1;
    pot.castShadow = true;
    plantGroup.add(pot);

    // 2. Dirt in pot
    const dirtGeom = new THREE.CylinderGeometry(0.14, 0.14, 0.05, 16);
    const dirtMat = new THREE.MeshStandardMaterial({ color: 0x2e1a0b });
    const dirt = new THREE.Mesh(dirtGeom, dirtMat);
    dirt.position.y = 0.18;
    plantGroup.add(dirt);

    if (todoData.completed) {
        // Render as blooming flower
        const stemGeom = new THREE.CylinderGeometry(0.015, 0.02, 0.4, 8);
        stemGeom.translate(0, 0.2, 0);
        const plantMat = new THREE.MeshStandardMaterial({ color: 0x2ecc71 });
        const stem = new THREE.Mesh(stemGeom, plantMat);
        stem.position.y = 0.2;
        stem.castShadow = true;
        plantGroup.add(stem);

        const flowerGeom = new THREE.DodecahedronGeometry(0.1);
        const flowerMat = new THREE.MeshStandardMaterial({ color: 0xff69b4, roughness: 0.4 }); // Hot pink bloom
        const flower = new THREE.Mesh(flowerGeom, flowerMat);
        flower.position.y = 0.4; // Top of stem
        stem.add(flower);

        // Droop slightly less
        stem.rotation.x = Math.PI / 8;
    } else {
        // 3. Stem (will bend based on health)
        const stemGeom = new THREE.CylinderGeometry(0.015, 0.02, 0.4, 8);
        // Move pivot point to bottom of stem
        stemGeom.translate(0, 0.2, 0);
        const plantMat = new THREE.MeshStandardMaterial({ color: 0x2ecc71 });
        const stem = new THREE.Mesh(stemGeom, plantMat);
        stem.position.y = 0.2; // Start at dirt level
        stem.castShadow = true;
        stem.name = "stem";
        plantGroup.add(stem);

        // 4. Leaves (attached to stem)
        const leafGeom = new THREE.PlaneGeometry(0.16, 0.16);
        if (!window.sharedLeafMat) {
            window.sharedLeafMat = new THREE.MeshStandardMaterial({
                map: createLeafTexture(),
                alphaMap: createLeafAlphaTexture(),
                transparent: true,
                side: THREE.DoubleSide,
                alphaTest: 0.5
            });
        }
        const leafMat = window.sharedLeafMat;

        const leaf1 = new THREE.Mesh(leafGeom, leafMat);
        leaf1.position.set(0.05, 0.2, 0);
        leaf1.rotation.z = -Math.PI / 4;
        leaf1.name = "leaf1";
        stem.add(leaf1);

        const leaf2 = new THREE.Mesh(leafGeom, leafMat);
        leaf2.position.set(-0.05, 0.3, 0);
        leaf2.rotation.z = Math.PI / 4;
        leaf2.name = "leaf2";
        stem.add(leaf2);
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
}

function animate() {
    requestAnimationFrame(animate);

    const time = performance.now();

    if (controls.isLocked === true) {
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
        const intersectableMeshes = [];
        objects.forEach(group => {
            group.children.forEach(child => intersectableMeshes.push(child));
        });

        const intersects = raycaster.intersectObjects(intersectableMeshes);
        const tooltip = document.getElementById('hover-tooltip');

        if (intersects.length > 0) {
            let target = intersects[0].object;
            while (target && target.userData && target.userData.id === undefined && target.userData.isEmpty === undefined) {
                target = target.parent;
                if (target === scene) break;
            }

            if (target && target.userData) {
                if (target.userData.isEmpty) {
                    tooltip.textContent = "Click to plant a new to-do";
                    tooltip.style.display = 'block';
                } else if (target.userData.id) {
                    const todo = todos.find(t => t.id === target.userData.id);
                    if (todo) {
                        if (todo.completed) {
                            tooltip.textContent = `Completed: ${todo.title}`;
                        } else {
                            const statusText = todo.status || "Not Started";
                            tooltip.textContent = `${todo.title}\n[${statusText}]`;
                        }
                        tooltip.style.display = 'block';
                    }
                } else {
                    tooltip.style.display = 'none';
                }
            } else {
                tooltip.style.display = 'none';
            }
        } else {
            tooltip.style.display = 'none';
        }
    } else {
        document.getElementById('hover-tooltip').style.display = 'none';
    }

    // Update plant decay
    updateDecay();

    prevTime = time;

    renderer.render(scene, camera);
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
    if (controls.isLocked) {
        // Only raycast if controls are locked (user is exploring)
        raycaster.setFromCamera(mouse, camera);

        // Intersect against all children of the plant groups
        const intersectableMeshes = [];
        objects.forEach(group => {
            group.children.forEach(child => intersectableMeshes.push(child));
        });

        const intersects = raycaster.intersectObjects(intersectableMeshes);

        if (intersects.length > 0) {
            // Find the parent group which holds the userData
            let target = intersects[0].object;
            while (target && target.userData && target.userData.id === undefined && target.userData.isEmpty === undefined) {
                target = target.parent;
                if(target === scene) break; // sanity check
            }

            if (target && target.userData) {
                if (target.userData.isEmpty) {
                    activePotIndex = target.userData.positionIndex;
                    openAddTodoModal();
                } else if (target.userData.id) {
                    const todoId = target.userData.id;
                    const todo = todos.find(t => t.id === todoId);

                    if (todo && !todo.completed) {
                        openTodoModal(todo);
                    }
                }
            }
        }
    }
});

function openAddTodoModal() {
    controls.unlock();
    document.getElementById('add-todo-modal').style.display = 'flex';
}

function closeAddTodoModal() {
    document.getElementById('add-todo-modal').style.display = 'none';
    activePotIndex = null;
    controls.lock();
}

document.getElementById('close-add-modal').addEventListener('click', closeAddTodoModal);

function openTodoModal(todo) {
    activeTodo = todo;
    controls.unlock(); // Unlock camera to use mouse

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
    controls.lock(); // Re-lock to return to walking
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