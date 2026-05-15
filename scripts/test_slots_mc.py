import pytest
import importlib.util
import sys
import math

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
