"""Saved sliders and toggles are written to disk and restored on the next start."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Load NumPy before temporarily patching sys.modules; its native modules must
# remain cached after the camera stubs are removed.
import numpy

sys.path.insert(0, str(Path(__file__).parents[1]))
with patch.dict(sys.modules, {'cv2': Mock(), 'classes.tracker': Mock()}):
    from classes.app import OSCGestureApp
    import classes.app as app_module


def make_app(path):
    tracker = Mock()
    tracker.confidence_state.return_value = {
        'values': {'detection': .8, 'presence': .8, 'tracking': .8}}
    with patch.object(app_module, 'HandTracker', return_value=tracker):
        app = OSCGestureApp(settings_path=path)
    app.orchestra_control.close()
    return app


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.dir.name) / 'nested' / 'settings.json')

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file_keeps_defaults(self):
        app = make_app(self.path)
        self.assertEqual(app.hand_grace_ms, 150)
        self.assertEqual(app.preset, 'range')

    def test_saved_settings_restore_on_next_start(self):
        app = make_app(self.path)
        app.apply_control({'preset': 'tempo', 'active_area': .6, 'piano_only': False,
                           'hand_grace_ms': 300, 'draw_landmarks': True,
                           'min_hand_size': .12,
                           'range_window': {'brass': 10},
                           'instr_range': {'strings': [40, 80]}})
        app.orchestra_split_ratio = .3
        app.apply_control({'save_settings': True})

        restored = make_app(self.path)
        self.assertEqual(restored.preset, 'tempo')
        self.assertAlmostEqual(restored.active_area_ratio, .6)
        self.assertFalse(restored.piano_only)
        self.assertEqual(restored.hand_grace_ms, 300)
        self.assertAlmostEqual(restored.min_hand_size, .12)
        self.assertTrue(restored.draw_landmarks)
        self.assertEqual(restored.range_windows['brass'], 10)
        self.assertEqual(restored.instr_ranges['strings'], [40, 80])
        self.assertAlmostEqual(restored.orchestra_split_ratio, .3)

    def test_bad_value_skips_only_that_setting(self):
        Path(self.path).parent.mkdir(parents=True)
        Path(self.path).write_text(json.dumps({'preset': 'nope', 'hand_grace_ms': 400}))
        app = make_app(self.path)
        self.assertEqual(app.preset, 'range')
        self.assertEqual(app.hand_grace_ms, 400)

    def test_corrupt_file_is_ignored(self):
        Path(self.path).parent.mkdir(parents=True)
        Path(self.path).write_text('{not json')
        self.assertEqual(make_app(self.path).hand_grace_ms, 150)


if __name__ == '__main__':
    unittest.main()
