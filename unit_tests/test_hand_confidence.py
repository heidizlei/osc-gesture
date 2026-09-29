"""Confidence updates are validated and applied between camera frames."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


def load_tracker():
    path = Path(__file__).resolve().parents[1] / 'classes' / 'tracker.py'
    spec = importlib.util.spec_from_file_location('classes._test_tracker', path)
    module = importlib.util.module_from_spec(spec)
    dependencies = {name: Mock() for name in (
        'cv2', 'numpy', 'mediapipe', 'mediapipe.framework',
        'mediapipe.framework.formats', 'mediapipe.python',
        'mediapipe.python.solutions', 'classes.mjpeg_capture')}
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module.HandTracker


HandTracker = load_tracker()


class HandConfidenceTests(unittest.TestCase):
    def setUp(self):
        with patch.object(HandTracker, '_init_landmarker', return_value=Mock()):
            self.tracker = HandTracker(use_gpu=False)

    def test_updates_merge_and_wait_for_capture_thread(self):
        original = self.tracker.landmarker
        self.tracker.request_confidence({'detection': .9})
        self.tracker.request_confidence({'presence': .95})
        self.assertIs(self.tracker.landmarker, original)
        self.assertTrue(self.tracker.confidence_state()['pending'])
        replacement = Mock()
        with patch.object(self.tracker, '_init_landmarker', return_value=replacement) as build:
            self.tracker._apply_pending_confidence()
        build.assert_called_once_with(False, {'detection': .9, 'presence': .95, 'tracking': .8})
        self.assertIs(self.tracker.landmarker, replacement)
        original.close.assert_called_once()
        self.assertFalse(self.tracker.confidence_state()['pending'])

    def test_failure_keeps_previous_model_and_values(self):
        original = self.tracker.landmarker
        self.tracker.request_confidence({'tracking': .7})
        with patch.object(self.tracker, '_init_landmarker', side_effect=RuntimeError('unavailable')):
            self.tracker._apply_pending_confidence()
        state = self.tracker.confidence_state()
        self.assertIs(self.tracker.landmarker, original)
        original.close.assert_not_called()
        self.assertEqual(state['values']['tracking'], .8)
        self.assertIn('unavailable', state['error'])
        self.assertFalse(state['pending'])

    def test_invalid_updates_are_atomic(self):
        for values in (None, [], {}, {'unknown': .5}, {'tracking': True},
                       {'tracking': None}, {'tracking': float('nan')},
                       {'tracking': float('inf')}, {'tracking': -.1},
                       {'detection': .9, 'presence': 1.1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.tracker.request_confidence(values)
            self.assertFalse(self.tracker.confidence_state()['pending'])

    def test_unchanged_values_do_not_rebuild(self):
        self.tracker.request_confidence({'detection': .8})
        with patch.object(self.tracker, '_init_landmarker') as build:
            self.tracker._apply_pending_confidence()
        build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
