let camera, scene, renderer, controls;
let raycaster, mouse;

const objects = []; // Interactable objects (plants)
const todos = []; // Data for todos

// Time tracking
let simulatedTimeOffset = 0; // Fast forward offset in ms

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

init();
animate();

function init() {
    // 1. Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB); // Sky blue
    scene.fog = new THREE.Fog(0x87CEEB, 0, 50);

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

    // Window resize handler
    window.addEventListener('resize', onWindowResize);
}

// --- Plant Generation Logic ---
const tablePositions = []; // To track where to put next plant
const availablePositions = []; // Stack of available indices

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

    // Initialize available positions with all indices
    for (let i = 0; i < tablePositions.length; i++) {
        availablePositions.push(i);
    }
    // Reverse so we pop from the beginning (index 0 first)
    availablePositions.reverse();

    // Floor
    const floorGeometry = new THREE.PlaneGeometry(100, 100);
    const floorMaterial = new THREE.MeshStandardMaterial({
        color: 0x3d2817, // Dirt color
        roughness: 0.8
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Tables (Procedural layout along a central aisle)
    const tableMaterial = new THREE.MeshStandardMaterial({ color: 0x8b5a2b });
    const numTables = 10;
    const tableSpacing = 4;

    for (let i = 0; i < numTables; i++) {
        const zPos = -i * tableSpacing;

        // Left table
        createTable(-3, zPos, tableMaterial);

        // Right table
        createTable(3, zPos, tableMaterial);
    }
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

function createPlant(todoData) {
    if (availablePositions.length === 0) {
        console.warn("Greenhouse is full! Cannot add more plants.");
        alert("Greenhouse is full! Complete some tasks first.");
        return false;
    }

    const positionIndex = availablePositions.pop();
    const pos = tablePositions[positionIndex];

    // Plant Group
    const plantGroup = new THREE.Group();
    plantGroup.position.copy(pos);
    plantGroup.userData = {
        id: todoData.id,
        positionIndex: positionIndex
    }; // Link to data

    // 1. Pot
    const potGeom = new THREE.CylinderGeometry(0.15, 0.1, 0.2, 16);
    const potMat = new THREE.MeshStandardMaterial({ color: 0xcd5c5c, roughness: 0.9 }); // Terracotta
    const pot = new THREE.Mesh(potGeom, potMat);
    pot.position.y = 0.1; // Rest on table
    pot.castShadow = true;
    plantGroup.add(pot);

    // 2. Dirt in pot
    const dirtGeom = new THREE.CylinderGeometry(0.14, 0.14, 0.05, 16);
    const dirtMat = new THREE.MeshStandardMaterial({ color: 0x2e1a0b });
    const dirt = new THREE.Mesh(dirtGeom, dirtMat);
    dirt.position.y = 0.18;
    plantGroup.add(dirt);

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
    const leafGeom = new THREE.SphereGeometry(0.08, 8, 8);
    leafGeom.scale(1, 0.2, 1);

    const leaf1 = new THREE.Mesh(leafGeom, plantMat);
    leaf1.position.set(0.05, 0.2, 0);
    leaf1.rotation.z = -Math.PI / 4;
    leaf1.name = "leaf1";
    stem.add(leaf1);

    const leaf2 = new THREE.Mesh(leafGeom, plantMat);
    leaf2.position.set(-0.05, 0.3, 0);
    leaf2.rotation.z = Math.PI / 4;
    leaf2.name = "leaf2";
    stem.add(leaf2);

    // Save reference for interaction and updates
    objects.push(plantGroup);
    scene.add(plantGroup);

    // Bind mesh to data
    todoData.mesh = plantGroup;

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

    const newTodo = {
        id: Date.now(),
        title: title,
        desc: desc,
        urgency: urgency,
        createdAt: Date.now(),
        lastUpdated: Date.now(),
        health: 100 // 0 to 100
    };

    // Only add if there is space
    if (createPlant(newTodo)) {
        todos.push(newTodo);
        // Reset form
        this.reset();
        document.getElementById('todo-title').focus();
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

// --- Interaction Logic ---
let activeTodo = null;

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
            while(target && !target.userData.id) {
                target = target.parent;
                if(target === scene) break; // sanity check
            }

            if (target && target.userData.id) {
                const todoId = target.userData.id;
                const todo = todos.find(t => t.id === todoId);

                if (todo) {
                    openTodoModal(todo);
                }
            }
        }
    }
});

function openTodoModal(todo) {
    activeTodo = todo;
    controls.unlock(); // Unlock camera to use mouse

    document.getElementById('modal-title').textContent = todo.title;
    document.getElementById('modal-desc').textContent = todo.desc;
    document.getElementById('modal-health').textContent = Math.round(todo.health) + '%';

    let urgencyText = "Medium";
    if (todo.urgency === 1) urgencyText = "Low";
    if (todo.urgency === 3) urgencyText = "High";
    document.getElementById('modal-urgency').textContent = urgencyText;

    document.getElementById('todo-modal').style.display = 'flex';
}

function closeTodoModal() {
    document.getElementById('todo-modal').style.display = 'none';
    activeTodo = null;
    controls.lock(); // Re-lock to return to walking
}

document.getElementById('close-modal').addEventListener('click', closeTodoModal);

document.getElementById('btn-acknowledge').addEventListener('click', function() {
    if (activeTodo) {
        // Reset health to 100% and update time
        activeTodo.health = 100;
        activeTodo.lastUpdated = getCurrentSimulatedTime();

        // Update visually immediately
        updatePlantVisual(activeTodo);

        closeTodoModal();
    }
});

document.getElementById('btn-complete').addEventListener('click', function() {
    if (activeTodo) {
        // Remove from data
        const index = todos.indexOf(activeTodo);
        if (index > -1) {
            todos.splice(index, 1);
        }

        // Reclaim the position
        if (activeTodo.mesh && activeTodo.mesh.userData.positionIndex !== undefined) {
            availablePositions.push(activeTodo.mesh.userData.positionIndex);
            // Re-sort descending so we pop the lowest available index next
            availablePositions.sort((a, b) => b - a);
        }

        // Remove mesh from scene
        if (activeTodo.mesh) {
            scene.remove(activeTodo.mesh);
            disposeHierarchy(activeTodo.mesh);

            const objIndex = objects.indexOf(activeTodo.mesh);
            if(objIndex > -1) {
                objects.splice(objIndex, 1);
            }
        }

        closeTodoModal();
    }
});