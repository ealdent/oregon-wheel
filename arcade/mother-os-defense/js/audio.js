"use strict";

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  toastTimer = 2.2;
}

function playSound(kind) {
  if (!state.sound) return;
  try {
    if (!audioContext && navigator.userActivation && !navigator.userActivation.hasBeenActive) {
      return;
    }
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }
    const now = audioContext.currentTime;
    const sound = {
      place: [220, 330, 0.055, "square", 0.045],
      upgrade: [330, 660, 0.085, "triangle", 0.06],
      wave: [130, 390, 0.18, "sawtooth", 0.035],
      hit: [180, 100, 0.055, "square", 0.025],
      blast: [90, 45, 0.18, "sawtooth", 0.075],
      leak: [90, 70, 0.22, "triangle", 0.08],
      click: [260, 260, 0.035, "square", 0.025]
    }[kind] || [220, 220, 0.04, "sine", 0.03];
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    osc.type = sound[3];
    osc.frequency.setValueAtTime(sound[0], now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(20, sound[1]), now + sound[2]);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(sound[4], now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + sound[2]);
    osc.connect(gain);
    gain.connect(audioContext.destination);
    osc.start(now);
    osc.stop(now + sound[2] + 0.03);
  } catch (error) {
    state.sound = false;
  }
}
