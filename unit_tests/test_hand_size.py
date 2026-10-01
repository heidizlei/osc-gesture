"""Small hands are dropped and at most the two largest hands are kept."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_keep_hands():
    path = Path(__file__).resolve().parents[1] / 'classes' / 'tracker.py'
    spec = importlib.util.spec_from_file_location('classes._test_tracker_size', path)
    module = importlib.util.module_from_spec(spec)
    dependencies = {name: Mock() for name in (
        'cv2', 'numpy', 'mediapipe', 'mediapipe.framework',
        'mediapipe.framework.formats', 'mediapipe.python',
        'mediapipe.python.solutions', 'classes.mjpeg_capture')}
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module.keep_hands


keep_hands = load_keep_hands()


@dataclass
class Result:
    handedness: list
    hand_landmarks: list
    hand_world_landmarks: list


def hand(palm):
    """21 landmarks with the wrist-to-middle-knuckle span = palm (normalized y)."""
    points = [SimpleNamespace(x=.5, y=.5) for _ in range(21)]
    points[9] = SimpleNamespace(x=.5, y=.5 - palm)
    return points


def result(*palms):
    return Result(handedness=[f'h{i}' for i in range(len(palms))],
                  hand_landmarks=[hand(p) for p in palms],
                  hand_world_landmarks=[f'w{i}' for i in range(len(palms))])


class KeepHandsTests(unittest.TestCase):
    def test_two_hands_without_minimum_pass_through(self):
        r = result(.1, .2)
        self.assertIs(keep_hands(r, 640, 480), r)

    def test_keeps_two_largest_in_original_order(self):
        kept = keep_hands(result(.2, .05, .1, .3), 640, 480)
        self.assertEqual(kept.handedness, ['h0', 'h3'])
        self.assertEqual(kept.hand_world_landmarks, ['w0', 'w3'])

    def test_drops_hands_under_minimum(self):
        kept = keep_hands(result(.05, .2), 640, 480, min_size=.1)
        self.assertEqual(kept.handedness, ['h1'])

    def test_empty_result(self):
        r = Result([], [], [])
        self.assertIs(keep_hands(r, 640, 480, .1), r)


if __name__ == '__main__':
    unittest.main()
