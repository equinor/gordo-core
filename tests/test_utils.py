import pandas as pd
import numpy as np
from gordo_core.utils import find_gaps, capture_args, capture_args_ext


def test_find_gaps():
    date_range1 = pd.date_range(
        start="2020-01-01 00:00", periods=6, freq="10T", tz="UTC"
    )
    date_range2 = pd.date_range(
        start="2020-01-01 01:30", periods=3, freq="10T", tz="UTC"
    )
    index = pd.concat([pd.Series(date_range1), pd.Series(date_range2)])
    data = np.random.random(9)
    series = pd.Series(data, index=index)
    gaps = find_gaps(
        series,
        resolution="10T",
        resampling_startpoint=pd.Timestamp("2020-01-01 00:00", tz="UTC"),
        resampling_endpoint=pd.Timestamp("2020-01-01 02:00", tz="UTC"),
    )
    expected = pd.DataFrame(
        {
            "start": [pd.Timestamp("2020-01-01 00:50:00", tz="UTC")],
            "end": [pd.Timestamp("2020-01-01 01:30:00", tz="UTC")],
        }
    )
    assert gaps.equals(expected)


def test_capture_args():
    class _CapturingTest:
        @capture_args
        def __init__(self, arg, first=42, second=2):
            self.arg = arg
            self.first = first
            self.second = second

    instance = _CapturingTest(0, first=1)
    assert hasattr(instance, "_params")
    assert getattr(instance, "_params") == {"first": 1, "second": 2, "arg": 0}


def test_capture_args_ext():
    class _CapturingTest:
        @capture_args_ext(ignore=["arg", "second"])
        def __init__(self, arg, first=42, second=2, third=3):
            self.arg = arg
            self.first = first
            self.second = second
            self.third = third

    instance = _CapturingTest(0)
    assert hasattr(instance, "_params")
    assert getattr(instance, "_params") == {"first": 42, "third": 3}
