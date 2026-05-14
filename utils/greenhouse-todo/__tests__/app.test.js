// Mock localStorage globally
const localStorageMock = (function() {
    let store = {};
    return {
        getItem: jest.fn(key => store[key] || null),
        setItem: jest.fn((key, value) => {
            store[key] = value.toString();
        }),
        clear: jest.fn(() => {
            store = {};
        })
    };
})();
Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// We just need a basic mock for three and its addons so require() doesn't fail
jest.mock('three', () => ({
    Cache: { enabled: false },
    TextureLoader: class { load() { return {}; } },
    Vector3: class { set() { return this; } },
    Vector2: class {},
    Scene: class {},
    PerspectiveCamera: class {},
    WebGLRenderer: class { setPixelRatio() {} setSize() {} },
    Raycaster: class {},
}));
jest.mock('three/addons/controls/PointerLockControls.js', () => ({ PointerLockControls: class {} }));
jest.mock('three/addons/environments/RoomEnvironment.js', () => ({ RoomEnvironment: class {} }));
jest.mock('three/addons/objects/Sky.js', () => ({ Sky: class {} }));
jest.mock('three/addons/postprocessing/EffectComposer.js', () => ({ EffectComposer: class {} }));
jest.mock('three/addons/postprocessing/RenderPass.js', () => ({ RenderPass: class {} }));
jest.mock('three/addons/postprocessing/UnrealBloomPass.js', () => ({ UnrealBloomPass: class {} }));
jest.mock('three/addons/postprocessing/OutputPass.js', () => ({ OutputPass: class {} }));
jest.mock('three/addons/utils/BufferGeometryUtils.js', () => ({ mergeGeometries: () => {} }));
jest.mock('three/addons/libs/stats.module.js', () => { return function() { return { showPanel: () => {}, dom: { style: {} } }; }; });

// Provide enough DOM elements for module load
const mockElement = { addEventListener: jest.fn(), style: {} };
document.getElementById = jest.fn(() => mockElement);
document.body.innerHTML = ``;

Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation(query => ({
        matches: false,
    })),
});

describe('app.js pure data logic', () => {
    let app;
    let consoleErrorMock;

    beforeEach(() => {
        jest.isolateModules(() => {
            localStorageMock.clear();
            localStorageMock.getItem.mockClear();
            localStorageMock.setItem.mockClear();
            consoleErrorMock = jest.spyOn(console, 'error').mockImplementation(() => {});

            app = require('../app.js');
        });
    });

    afterEach(() => {
        consoleErrorMock.mockRestore();
    });

    it('should save data correctly to localStorage, stripping mesh property', () => {
        const testTodos = [
            { id: 1, text: 'Test 1', mesh: { uuid: '123' }, otherProp: 'value1' },
            { id: 2, text: 'Test 2', otherProp: 'value2' }
        ];

        app.todos = testTodos;
        app.simulatedTimeOffset = 1234;

        app.saveTodosToLocal();

        expect(localStorageMock.setItem).toHaveBeenCalledWith(
            'greenhouse-todos-data',
            expect.any(String)
        );

        const savedDataStr = localStorageMock.setItem.mock.calls[0][1];
        const savedData = JSON.parse(savedDataStr);

        expect(savedData.simulatedTimeOffset).toBe(1234);
        expect(savedData.todos).toHaveLength(2);

        expect(savedData.todos[0]).not.toHaveProperty('mesh');
        expect(savedData.todos[0].id).toBe(1);
        expect(savedData.todos[0].text).toBe('Test 1');
        expect(savedData.todos[0].otherProp).toBe('value1');

        expect(savedData.todos[1]).not.toHaveProperty('mesh');
        expect(savedData.todos[1].id).toBe(2);
    });

    it('should handle empty todos array', () => {
        app.todos = [];
        app.simulatedTimeOffset = 1234;
        app.saveTodosToLocal();

        const savedDataStr = localStorageMock.setItem.mock.calls[0][1];
        const savedData = JSON.parse(savedDataStr);

        expect(savedData.todos).toEqual([]);
        expect(savedData.simulatedTimeOffset).toBe(1234);
    });

    it('should load data from localStorage', () => {
        const mockData = {
            todos: [
                { id: 1, text: 'Todo 1' },
                { id: 2, text: 'Todo 2' }
            ],
            simulatedTimeOffset: 5000
        };

        localStorageMock.getItem.mockReturnValue(JSON.stringify(mockData));

        app.loadTodosFromLocal();

        expect(localStorageMock.getItem).toHaveBeenCalledWith('greenhouse-todos-data');
        expect(app.todos).toEqual(mockData.todos);
        expect(app.simulatedTimeOffset).toBe(5000);
    });

    it('should handle missing simulatedTimeOffset in saved data', () => {
        const mockData = {
            todos: [
                { id: 1, text: 'Todo 1' }
            ]
        };

        localStorageMock.getItem.mockReturnValue(JSON.stringify(mockData));

        app.loadTodosFromLocal();

        expect(app.simulatedTimeOffset).toBe(0);
    });

    it('should handle missing todos in saved data', () => {
        const mockData = {
            simulatedTimeOffset: 1234
        };

        localStorageMock.getItem.mockReturnValue(JSON.stringify(mockData));

        app.loadTodosFromLocal();

        expect(app.todos).toEqual([]);
    });

    it('should handle invalid JSON in localStorage gracefully', () => {
        localStorageMock.getItem.mockReturnValue('invalid-json');

        app.loadTodosFromLocal();

        expect(consoleErrorMock).toHaveBeenCalledWith("Failed to parse saved data:", expect.any(Error));
    });
});
