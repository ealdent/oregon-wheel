module.exports = {
  testEnvironment: 'jsdom',
  moduleNameMapper: {
    '^three/addons/(.*)$': '<rootDir>/node_modules/three/examples/jsm/$1',
  },
  transformIgnorePatterns: [
    'node_modules/(?!(three)/)'
  ]
};
