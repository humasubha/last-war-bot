import numpy as np
import pytest

from game_vision_agent.capture.coordinates import desktop_to_client, frame_to_desktop
from game_vision_agent.control.executors import NoOpExecutor
from game_vision_agent.interfaces import ActionProposal, Box, WindowInfo
from game_vision_agent.perception.change import NumpyChangeDetector, Region
from game_vision_agent.replay.sources import SyntheticFrameSource


def test_static_and_large_change():
    detector = NumpyChangeDetector(sample_step=1)
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    assert not detector.compare(a, a.copy()).changed
    b = a.copy(); b[8:20, 8:20] = 255
    result = detector.compare(a, b)
    assert result.changed and result.changed_region is not None
    assert result.changed_region.x == 8


def test_small_change_and_roi_exclusion():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = a.copy(); b[2:6, 2:6] = 20
    assert NumpyChangeDetector(sample_step=1, threshold=0.1).compare(a, b).changed
    assert not NumpyChangeDetector(sample_step=1, threshold=0.1,
        ignore_regions=(Region(0, 0, 8, 8),)).compare(a, b).changed
    assert not NumpyChangeDetector(sample_step=1, threshold=0.1,
        roi=Region(16, 16, 8, 8)).compare(a, b).changed


def test_normalized_coordinates_and_invalid_dimensions():
    assert Box(25, 10, 50, 20).normalized(100, 40) == Box(.25, .25, .5, .5)
    with pytest.raises(ValueError):
        Box(0, 0, 1, 1).normalized(0, 10)


def test_negative_desktop_coordinate_conversion():
    win = WindowInfo(4, "fixture", -1920, 50, 640, 480, True, False,
                    -1900, 80, 600, 400)
    frame_box = Box(10, 20, 50, 60)
    desktop = frame_to_desktop(frame_box, win)
    assert desktop == Box(-1890, 100, 50, 60)
    assert desktop_to_client(desktop, win) == frame_box


def test_synthetic_deterministic():
    a, b = SyntheticFrameSource(), SyntheticFrameSource()
    assert np.array_equal(a.read().image, b.read().image)


def test_live_execution_never_runs():
    outcome = NoOpExecutor().execute(ActionProposal("x", "test"), simulator=False)
    assert not outcome.executed


def test_no_prior_frame():
    result = NumpyChangeDetector().compare(None, np.zeros((10, 10), dtype=np.uint8))
    assert not result.changed
