import math
import sys
from pathlib import Path
import importlib.util

# Load slots-mc.py dynamically since it has a hyphen in the name
script_path = Path(__file__).parent.parent / "slots-mc.py"
spec = importlib.util.spec_from_file_location("slots_mc", script_path)
slots_mc = importlib.util.module_from_spec(spec)
sys.modules["slots_mc"] = slots_mc
spec.loader.exec_module(slots_mc)

def test_ci_95_empty():
    """Test ci_95 with an empty list."""
    result = slots_mc.ci_95([])
    assert math.isnan(result[0])
    assert math.isnan(result[1])

def test_ci_95_single_element():
    """Test ci_95 with a single element list."""
    val = 42.0
    result = slots_mc.ci_95([val])
    assert result[0] == val
    assert math.isnan(result[1])

def test_ci_95_small_list():
    """Test ci_95 with a small list (n < 30)."""
    xs = [10.0, 20.0]
    result = slots_mc.ci_95(xs)
    mean = 15.0
    half = 1.96 * (5.0 / math.sqrt(2))
    assert math.isclose(result[0], mean - half)
    assert math.isclose(result[1], mean + half)

def test_ci_95_large_list():
    """Test ci_95 with a large list (n >= 30)."""
    xs = [0.0] * 15 + [10.0] * 15
    result = slots_mc.ci_95(xs)
    mean = 5.0
    stdev = math.sqrt(750.0 / 29.0)
    half = 1.96 * (stdev / math.sqrt(30))
    assert math.isclose(result[0], mean - half)
    assert math.isclose(result[1], mean + half)

def test_ci_95_constant_list():
    """Test ci_95 with a constant list to verify no math domain errors (div by zero etc)."""
    xs = [10.0] * 5
    result = slots_mc.ci_95(xs)
    assert result[0] == 10.0
    assert result[1] == 10.0
