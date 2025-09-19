  #!/usr/bin/env python3
"""
Monte Carlo simulator for a slots strategy.

How to use:
1) Put the probabilities you previously determined into OUTCOMES below.
   - Each item: ("name", probability, payout_multiplier)
   - Example multipliers (adjust if your table differs):
       eggplant x3 -> 1.0 (push), hearts x3 -> 2.0, cherries x3 -> 3.0,
       cowoncy x3 -> 4.0, o-w-o -> 10.0
   - If probabilities sum to < 1, the leftover is treated as a 0x loss.
   - If probabilities sum to > 1 (beyond tiny rounding), an error is raised.

2) Choose STRATEGY in the CONFIG section. Included options:
   - "flat":             Constant bet each spin
   - "martingale":       Multiply bet after *loss*, reset after *win*
   - "paroli":           Multiply bet after *win*, reset after *loss* (anti-martingale)
   - "proportional":     Bet fixed fraction of bankroll
   - "custom":           Fill in `next_bet_custom(...)` with your exact rule

3) Run:
   python simulate_strategy.py
   (Optional) tweak NUM_SESSIONS / MAX_SPINS / bankroll / targets / seed, etc.

Outputs:
- Theoretical EV per spin (from your probabilities)
- Monte Carlo summary (mean/CI of final bankroll & ROI, risk of ruin, target-hit%)
- Longest losing/winning streak stats and typical bet sizes

Note: This is a stochastic simulator. Larger NUM_SESSIONS improves precision.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import statistics

# =========================
# 1) >>> OUTCOME PROBABILITIES <<<
# Replace these placeholder probabilities with the ones you previously estimated.
# Format: ("label", probability, payout_multiplier)
# If your set already includes the "loss" case, keep its multiplier at 0.0 and
# make sure total probability ~= 1.0; otherwise we'll infer loss as the remainder.
OUTCOMES: List[Tuple[str, float, float]] = [
    # ----- FILL THESE WITH YOUR ESTIMATES -----
    # ("eggplant x3", 0.1000, 1.0),  # example: push (break-even)
    # ("hearts x3",   0.0600, 2.0),
    # ("cherries x3", 0.0300, 3.0),
    # ("cowoncy x3",  0.0200, 4.0),
    # ("o-w-o",       0.0010, 10.0),
    # ------------------------------------------
    # Leave empty to force you to paste real numbers; or uncomment the above as a starting point.
]


# =========================
# 2) >>> STRATEGY CHOICES & CONFIG <<<
@dataclass
class StrategyParams:
    base_bet: int = 5000  # starting/flat bet (or minimum progression seed)
    loss_mult: float = 2.0  # martingale factor
    win_mult: float = 2.0  # paroli factor
    paroli_steps: int = 3  # e.g., press wins up to 3 times, then reset
    frac_of_bankroll: float = 0.02  # proportional fraction (e.g., 2% of bankroll)
    treat_push_as_win: bool = False  # consider 1.0x a "win" for progression logic?


# Choose one: "flat" | "martingale" | "paroli" | "proportional" | "custom"
STRATEGY: str = "martingale"


# =========================
# 3) >>> SIMULATION CONFIG <<<
@dataclass
class SimConfig:
    # Bankroll / betting constraints
    starting_bankroll: int = 1_000_000
    min_bet: int = 5_000
    max_bet: int = 250_000
    max_spins: int = 500  # per session cap
    stop_profit: Optional[int] = None  # e.g., +200_000 -> stop once profit >= 200k
    stop_drawdown: Optional[int] = None  # e.g., -200_000 -> stop once PnL <= -200k
    # Monte Carlo
    num_sessions: int = 20_000
    seed: Optional[int] = 42


CONFIG = SimConfig()
PARAMS = StrategyParams()


# =========================
# 4) >>> STRATEGY LOGIC IMPLEMENTATIONS <<<
def next_bet_flat(
    bankroll: int, last_bet: int, last_mult: float, w_streak: int, l_streak: int
) -> int:
    return PARAMS.base_bet


def next_bet_martingale(
    bankroll: int, last_bet: int, last_mult: float, w_streak: int, l_streak: int
) -> int:
    """Double after *loss*, reset after *win* (push counts per PARAMS.treat_push_as_win)."""
    if last_bet <= 0:
        return PARAMS.base_bet
    was_win = last_mult > 1.0 or (PARAMS.treat_push_as_win and last_mult == 1.0)
    return (
        PARAMS.base_bet
        if was_win
        else int(max(PARAMS.base_bet, min(CONFIG.max_bet, last_bet * PARAMS.loss_mult)))
    )


def next_bet_paroli(
    bankroll: int, last_bet: int, last_mult: float, w_streak: int, l_streak: int
) -> int:
    """Increase after *win*, reset after *loss*; cap presses at PARAMS.paroli_steps."""
    if last_bet <= 0:
        return PARAMS.base_bet
    was_win = last_mult > 1.0 or (PARAMS.treat_push_as_win and last_mult == 1.0)
    if was_win and w_streak < PARAMS.paroli_steps:
        return int(
            min(CONFIG.max_bet, max(PARAMS.base_bet, last_bet * PARAMS.win_mult))
        )
    else:
        return PARAMS.base_bet


def next_bet_proportional(
    bankroll: int, last_bet: int, last_mult: float, w_streak: int, l_streak: int
) -> int:
    """Bet a fixed fraction of current bankroll, rounded to nearest min_bet step."""
    raw = bankroll * PARAMS.frac_of_bankroll
    # round to nearest multiple of min_bet
    stepped = max(CONFIG.min_bet, int(round(raw / CONFIG.min_bet) * CONFIG.min_bet))
    return min(stepped, CONFIG.max_bet)


def next_bet_custom(
    bankroll: int, last_bet: int, last_mult: float, w_streak: int, l_streak: int
) -> int:
    """
    <<< YOUR EXACT RULE HERE >>>
    Example: "Press to X on consecutive Y, otherwise reset", or
    "Decrease after win, increase after 2 losses", etc.
    Use the inputs as needed:
      - bankroll: current bankroll (after settling last spin)
      - last_bet: bet just used (0 on first spin)
      - last_mult: payout multiplier from last spin (1.0 push, 0.0 loss, >1 win)
      - w_streak/l_streak: current streak counts AFTER last spin
    Return the next bet (int).
    """
    # Default placeholder: same as flat
    return PARAMS.base_bet


STRATEGIES = {
    "flat": next_bet_flat,
    "martingale": next_bet_martingale,
    "paroli": next_bet_paroli,
    "proportional": next_bet_proportional,
    "custom": next_bet_custom,
}


# =========================
# 5) >>> CORE SIM LOGIC <<<
@dataclass
class Outcome:
    name: str
    prob: float
    mult: float


def build_distribution(outcomes_cfg: List[Tuple[str, float, float]]) -> List[Outcome]:
    if not outcomes_cfg:
        raise ValueError(
            "Please paste your previously determined probabilities into OUTCOMES."
        )
    total_p = sum(p for _, p, _ in outcomes_cfg)
    if total_p > 1.0000000001:
        raise ValueError(
            f"Outcome probabilities sum to {total_p:.6f} (>1). Fix your inputs."
        )
    dist: List[Outcome] = [Outcome(n, p, m) for (n, p, m) in outcomes_cfg]
    if total_p < 0.9999999999:
        # Add the residual as a pure loss (0x)
        residual = 1.0 - total_p
        dist.append(Outcome("loss (residual)", residual, 0.0))
    # Build cumulative
    running = 0.0
    for d in dist:
        running += d.prob
    # tiny normalization safeguard
    s = sum(d.prob for d in dist)
    for d in dist:
        d.prob = d.prob / s
    return dist


@dataclass
class SessionStats:
    final_bankroll: int
    spins: int
    ruin: bool
    hit_target: bool
    total_bet: int
    total_return: int
    max_bet_seen: int
    longest_win_streak: int
    longest_loss_streak: int


def sample_outcome(
    rng: np.random.Generator, cum_p: np.ndarray, mults: np.ndarray, labels: List[str]
) -> Tuple[str, float]:
    u = rng.random()
    idx = int(np.searchsorted(cum_p, u, side="right"))
    return labels[idx], float(mults[idx])


def theoretical_ev(dist: List[Outcome]) -> float:
    """Expected return per 1 unit bet (multiplier expectation) minus 1."""
    return sum(o.prob * o.mult for o in dist) - 1.0


def simulate_one_session(
    rng: np.random.Generator, dist: List[Outcome], strategy: str, config: SimConfig
) -> SessionStats:
    labels = [o.name for o in dist]
    probs = np.array([o.prob for o in dist], dtype=float)
    mults = np.array([o.mult for o in dist], dtype=float)
    cum_p = np.cumsum(probs)

    bankroll = config.starting_bankroll
    total_bet = 0
    total_ret = 0
    max_bet_seen = 0

    # Streaks and last spin info
    last_bet = 0
    last_mult = 0.0
    w_streak = 0
    l_streak = 0
    longest_w = 0
    longest_l = 0

    next_bet_fn = STRATEGIES[strategy]

    for spin in range(1, config.max_spins + 1):
        # Determine bet for this spin
        bet = next_bet_fn(bankroll, last_bet, last_mult, w_streak, l_streak)
        bet = max(config.min_bet, min(config.max_bet, bet))
        bet = min(bet, bankroll)  # cannot bet more than you have

        if bet < config.min_bet:
            # Can't place table minimum => ruin (or end)
            return SessionStats(
                final_bankroll=bankroll,
                spins=spin - 1,
                ruin=True,
                hit_target=False,
                total_bet=total_bet,
                total_return=total_ret,
                max_bet_seen=max_bet_seen,
                longest_win_streak=longest_w,
                longest_loss_streak=longest_l,
            )

        # Place the bet
        bankroll -= bet
        total_bet += bet
        if bet > max_bet_seen:
            max_bet_seen = bet

        # Spin outcome
        _, mult = sample_outcome(rng, cum_p, mults, labels)
        payout = int(round(bet * mult))
        bankroll += payout
        total_ret += payout

        # Update streaks
        if mult > 1.0 or (PARAMS.treat_push_as_win and mult == 1.0):
            w_streak += 1
            l_streak = 0
            longest_w = max(longest_w, w_streak)
        elif mult == 1.0:
            # push: neither win nor loss (unless treat_push_as_win=True above)
            # keep streaks unchanged
            pass
        else:
            l_streak += 1
            w_streak = 0
            longest_l = max(longest_l, l_streak)

        # Stop conditions: profit target / drawdown
        pnl = bankroll - config.starting_bankroll
        if config.stop_profit is not None and pnl >= config.stop_profit:
            return SessionStats(
                final_bankroll=bankroll,
                spins=spin,
                ruin=False,
                hit_target=True,
                total_bet=total_bet,
                total_return=total_ret,
                max_bet_seen=max_bet_seen,
                longest_win_streak=longest_w,
                longest_loss_streak=longest_l,
            )
        if config.stop_drawdown is not None and pnl <= -abs(config.stop_drawdown):
            return SessionStats(
                final_bankroll=bankroll,
                spins=spin,
                ruin=True,
                hit_target=False,
                total_bet=total_bet,
                total_return=total_ret,
                max_bet_seen=max_bet_seen,
                longest_win_streak=longest_w,
                longest_loss_streak=longest_l,
            )

        # Prepare for next loop
        last_bet = bet
        last_mult = mult

    return SessionStats(
        final_bankroll=bankroll,
        spins=config.max_spins,
        ruin=False,
        hit_target=False,
        total_bet=total_bet,
        total_return=total_ret,
        max_bet_seen=max_bet_seen,
        longest_win_streak=longest_w,
        longest_loss_streak=longest_l,
    )


def ci_95(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan"), float("nan"))
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) < 30 else statistics.stdev(xs)
    # 1.96 ~ normal approx
    half = 1.96 * (s / math.sqrt(len(xs)))
    return (m - half, m + half)


def main():
    # Build RNG
    rng = np.random.default_rng(CONFIG.seed)

    # Build outcome distribution; add residual loss if needed
    dist = build_distribution(OUTCOMES)

    # Quick report of theoretical EV per spin (independent of strategy)
    ev = theoretical_ev(dist)
    print("=== Game (theoretical) ===")
    print(
        f"Sum of provided probabilities: {sum(o.prob for o in dist):.6f} (includes residual loss if any)"
    )
    print("Outcome table:")
    for o in dist:
        print(f"  - {o.name:<18}  p={o.prob:>8.5f}  multiplier={o.mult:>5.2f}")
    print(
        f"Expected return per spin (multiplier - 1): {ev:+.6f}  ==> house edge ≈ {-ev:+.6f}\n"
    )

    # Sanity on strategy name
    if STRATEGY not in STRATEGIES:
        raise ValueError(
            f"Unknown STRATEGY '{STRATEGY}'. Choose from {list(STRATEGIES)}."
        )

    # Run Monte Carlo
    finals: List[int] = []
    rois: List[float] = []  # (total_return - total_bet) / total_bet
    ruins = 0
    target_hits = 0
    max_bets = []
    longest_losses = []
    longest_wins = []
    spins_per = []

    for _ in range(CONFIG.num_sessions):
        s = simulate_one_session(rng, dist, STRATEGY, CONFIG)
        finals.append(s.final_bankroll)
        # ROI relative to wagered amount
        profit = s.total_return - s.total_bet
        roi = (profit / s.total_bet) if s.total_bet > 0 else 0.0
        rois.append(roi)
        ruins += int(s.ruin)
        target_hits += int(s.hit_target)
        max_bets.append(s.max_bet_seen)
        longest_losses.append(s.longest_loss_streak)
        longest_wins.append(s.longest_win_streak)
        spins_per.append(s.spins)

    start = CONFIG.starting_bankroll
    profits = [fb - start for fb in finals]
    mean_final = statistics.mean(finals)
    mean_profit = statistics.mean(profits)
    mean_roi = statistics.mean(rois)
    ci_final = ci_95(finals)
    ci_roi = ci_95(rois)

    print("=== Strategy & Sim ===")
    print(f"Strategy: {STRATEGY}")
    print(
        f"Sessions: {CONFIG.num_sessions:,}  |  Max spins per session: {CONFIG.max_spins:,}"
    )
    print(
        f"Bankroll: start={start:,}  min_bet={CONFIG.min_bet:,}  max_bet={CONFIG.max_bet:,}"
    )
    if CONFIG.stop_profit is not None:
        print(f"Stop on profit ≥ +{CONFIG.stop_profit:,}")
    if CONFIG.stop_drawdown is not None:
        print(f"Stop on drawdown ≤ -{abs(CONFIG.stop_drawdown):,}")
    print()

    print("=== Monte Carlo results ===")
    print(
        f"Mean final bankroll: {mean_final:,.0f}   (95% CI: {ci_final[0]:,.0f} … {ci_final[1]:,.0f})"
    )
    print(f"Mean profit per session: {mean_profit:,.0f}")
    print(
        f"Mean ROI on amount wagered: {mean_roi:+.4%}   (95% CI: {ci_roi[0]:+.4%} … {ci_roi[1]:+.4%})"
    )
    print(f"Risk of ruin: {ruins/CONFIG.num_sessions:.2%}")
    print(f"Profit target hit: {target_hits/CONFIG.num_sessions:.2%}")
    print(f"Avg spins completed per session: {statistics.mean(spins_per):.1f}")
    print(
        f"Median longest loss streak: {int(statistics.median(longest_losses))}   "
        f"|  Median longest win streak: {int(statistics.median(longest_wins))}"
    )
    print(f"Median max bet observed: {int(statistics.median(max_bets)):,}")


if __name__ == "__main__":
    main()
