// Mock the global variables before requiring gameplay.js
global.state = { phase: "planning" };
global.towerById = {
  "basic": { cost: 100 }
};

const { isFullRefundEligible } = require('../gameplay.js');

describe('isFullRefundEligible', () => {
  beforeEach(() => {
    global.state = { phase: "planning" };
    global.towerById = {
      "basic": { cost: 100 }
    };
  });

  it('should return false if no tower provided', () => {
    expect(isFullRefundEligible(null)).toBe(false);
    expect(isFullRefundEligible(undefined)).toBe(false);
  });

  it('should return true when all conditions are met', () => {
    const validTower = {
      type: 'basic',
      refundable: true,
      level: 1,
      spent: 100
    };
    expect(isFullRefundEligible(validTower)).toBe(true);
  });

  it('should return false if state phase is not planning', () => {
    global.state.phase = "active";
    const tower = { type: 'basic', refundable: true, level: 1, spent: 100 };
    expect(isFullRefundEligible(tower)).toBe(false);
  });

  it('should return false if tower is not refundable', () => {
    const tower = { type: 'basic', refundable: false, level: 1, spent: 100 };
    expect(isFullRefundEligible(tower)).toBe(false);
  });

  it('should return false if tower level is not 1', () => {
    const tower = { type: 'basic', refundable: true, level: 2, spent: 100 };
    expect(isFullRefundEligible(tower)).toBe(false);
  });

  it('should return false if spent does not match cost', () => {
    const tower = { type: 'basic', refundable: true, level: 1, spent: 50 };
    expect(isFullRefundEligible(tower)).toBe(false);
  });
});
