import pytest
import importlib.util
import sys
import math
import os

# Load the target module dynamically due to hyphen in filename
spec = importlib.util.spec_from_file_location("slots_mc", "scripts/slots-mc.py")
slots_mc = importlib.util.module_from_spec(spec)
sys.modules["slots_mc"] = slots_mc
spec.loader.exec_module(slots_mc)

def test_build_distribution_empty():
    with pytest.raises(ValueError, match="Please paste your previously determined probabilities into OUTCOMES."):
        slots_mc.build_distribution([])

def test_build_distribution_sum_too_high():
    outcomes = [
        ("outcome1", 0.6, 1.0),
        ("outcome2", 0.5, 2.0),
    ]
    with pytest.raises(ValueError, match=r"Outcome probabilities sum to 1\.100000 \(>1\)\. Fix your inputs\."):
        slots_mc.build_distribution(outcomes)

def test_build_distribution_exact_sum():
    outcomes = [
        ("outcome1", 0.5, 1.0),
        ("outcome2", 0.5, 2.0),
    ]
    dist = slots_mc.build_distribution(outcomes)

    assert len(dist) == 2
    assert dist[0].name == "outcome1"
    assert math.isclose(dist[0].prob, 0.5)
    assert dist[0].mult == 1.0

    assert dist[1].name == "outcome2"
    assert math.isclose(dist[1].prob, 0.5)
    assert dist[1].mult == 2.0

    assert math.isclose(sum(d.prob for d in dist), 1.0)

def test_build_distribution_with_residual():
    outcomes = [
        ("outcome1", 0.4, 1.0),
        ("outcome2", 0.3, 2.0),
    ]
    dist = slots_mc.build_distribution(outcomes)

    assert len(dist) == 3
    assert dist[0].name == "outcome1"
    assert math.isclose(dist[0].prob, 0.4)

    assert dist[1].name == "outcome2"
    assert math.isclose(dist[1].prob, 0.3)

    # Residual loss
    assert dist[2].name == "loss (residual)"
    assert math.isclose(dist[2].prob, 0.3)  # 1.0 - 0.7
    assert dist[2].mult == 0.0

    assert math.isclose(sum(d.prob for d in dist), 1.0)

# Define test fixtures/helper to reset parameters if needed
@pytest.fixture(autouse=True)
def reset_globals():
    """Reset the global PARAMS and CONFIG to their default states before each test."""
    slots_mc.PARAMS.base_bet = 5000
    slots_mc.PARAMS.loss_mult = 2.0
    slots_mc.PARAMS.treat_push_as_win = False

    slots_mc.CONFIG.max_bet = 250000
    yield

def test_martingale_zero_bet():
    """Test that if the last bet was 0 or negative, we return to the base bet."""
    # Bankroll, last_bet, last_mult, w_streak, l_streak
    next_bet = slots_mc.next_bet_martingale(1000000, 0, 0.0, 0, 1)
    assert next_bet == slots_mc.PARAMS.base_bet

    next_bet = slots_mc.next_bet_martingale(1000000, -5000, 0.0, 0, 1)
    assert next_bet == slots_mc.PARAMS.base_bet

def test_martingale_win():
    """Test that a win resets the bet to the base bet."""
    slots_mc.PARAMS.base_bet = 5000
    # last_bet = 20000, last_mult = 2.0 (win)
    next_bet = slots_mc.next_bet_martingale(1000000, 20000, 2.0, 1, 0)
    assert next_bet == 5000

def test_martingale_loss():
    """Test that a loss doubles the last bet."""
    slots_mc.PARAMS.base_bet = 5000
    slots_mc.PARAMS.loss_mult = 2.0

    # last_bet = 5000, last_mult = 0.0 (loss)
    next_bet = slots_mc.next_bet_martingale(1000000, 5000, 0.0, 0, 1)
    assert next_bet == 10000

    # last_bet = 10000, last_mult = 0.0 (loss)
    next_bet = slots_mc.next_bet_martingale(1000000, 10000, 0.0, 0, 2)
    assert next_bet == 20000

def test_martingale_max_bet():
    """Test that the bet after a loss is capped by max_bet."""
    slots_mc.PARAMS.base_bet = 5000
    slots_mc.PARAMS.loss_mult = 2.0
    slots_mc.CONFIG.max_bet = 250000

    # last_bet = 200000, loss would double to 400000 but max is 250000
    next_bet = slots_mc.next_bet_martingale(1000000, 200000, 0.0, 0, 1)
    assert next_bet == 250000

def test_martingale_push_treated_as_win():
    """Test handling of a push (1.0x payout) when it is treated as a win."""
    slots_mc.PARAMS.base_bet = 5000
    slots_mc.PARAMS.loss_mult = 2.0
    slots_mc.PARAMS.treat_push_as_win = True

    # last_bet = 20000, last_mult = 1.0 (push treated as win)
    next_bet = slots_mc.next_bet_martingale(1000000, 20000, 1.0, 1, 0)
    assert next_bet == 5000

def test_martingale_push_treated_as_loss():
    """Test handling of a push (1.0x payout) when it is NOT treated as a win."""
    slots_mc.PARAMS.base_bet = 5000
    slots_mc.PARAMS.loss_mult = 2.0
    slots_mc.PARAMS.treat_push_as_win = False

    # last_bet = 20000, last_mult = 1.0 (push treated as loss)
    next_bet = slots_mc.next_bet_martingale(1000000, 20000, 1.0, 0, 1)
    assert next_bet == 40000
