"""Piano range transitions through the real camera app routing, without hardware."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Load NumPy before temporarily patching sys.modules; its native modules must
# remain cached after the camera stubs are removed.
import numpy

sys.path.insert(0, str(Path(__file__).parents[1]))
# These tests exercise range routing, not camera acquisition or rendering.
with patch.dict(sys.modules, {'cv2': Mock(), 'classes.tracker': Mock()}):
    from classes.app import OSCGestureApp
from classes.orchestra_control import HandGrace


class PianoHandLockTests(unittest.TestCase):
    def setUp(self):
        self.app = a = OSCGestureApp.__new__(OSCGestureApp)
        a.orchestra_mode = True
        a.orchestra_split_ratio = .5
        a.active_area_ratio = .75
        a.orchestra_control = SimpleNamespace(enabled=True, gesture_enabled=True,
                                             status={}, set_occupancy=Mock())
        a.hand_grace = HandGrace()
        a._piano_hands = {}
        a._last_piano_layout = None
        a._last_piano_centres = ()
        a._engaged = dict.fromkeys(('piano', 'brass', 'strings'), False)
        a._instr_state = {r: {'last_val': None, 'last_time': 0}
                          for r in ('brass', 'strings')}
        a._instr_ids = {0: 1, 1: 48, 2: 61}
        a.instr_ranges = {r: [24, 108] for r in a._engaged}
        a.interval = (-6, 6)
        a.last_left = a.last_right = None
        a.left_val = a.right_val = None
        a.last_osc_time = 0
        a.osc_interval = .2
        a.change_threshold = 4
        a.mode = 'range'
        a.osc_client = Mock()
        a._view_lock = threading.Lock()
        a._update_hand_presence = Mock()
        self.clock = patch.object(time, 'time', return_value=100)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def frame(self, *positions):
        a = self.app
        a.active_regions = sorted({a._hand_region(*p) for p in positions})
        if positions:
            a._update_output_range(positions)
        a._update_region_engagement()

    def piano_messages(self):
        return [call.args[1] for call in self.app.osc_client.send_message.call_args_list
                if call.args[0] == '/setOutputRange' and call.args[1][0] == 1]

    def test_two_hands_lock_and_return_independently_even_when_order_changes(self):
        self.frame((.25, .5, 'Left'), (.75, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 42, 54, 78, 90])
        # Immediate transition bypasses movement throttling; upper x is ignored.
        self.frame((.75, .5, 'Right'), (.9, .2, 'Left'))
        self.assertEqual(self.piano_messages()[-1], [1, 36, 60, 78, 90])
        self.app.last_osc_time = 0
        self.frame((.9, .2, 'Left'), (.5, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 36, 60, 60, 72])
        self.frame((.1, .2, 'Right'), (.9, .2, 'Left'))
        self.assertEqual(self.piano_messages()[-1], [1, 36, 60, 54, 78])
        count = len(self.piano_messages())
        self.frame((.1, .2, 'Left'), (.9, .2, 'Right'))
        self.assertEqual(len(self.piano_messages()), count)
        self.frame((.5, .5, 'Left'), (.9, .2, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 60, 72, 54, 78])

    def test_single_hand_lock_is_not_reset_when_piano_region_empties(self):
        self.frame((.25, .5, 'Left'))
        self.frame((.8, .2, 'Left'))
        self.assertEqual(self.piano_messages()[-1], [1, 36, 60, -1, -1])
        bars = {bar['instr']: bar for bar in self.app._range_bars()}
        self.assertEqual(bars['piano']['spans'], [[36, 60]])
        self.frame()
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])
        self.assertFalse(self.app._piano_hands)
        count = len(self.piano_messages())
        self.frame()
        self.assertEqual(len(self.piano_messages()), count)
        self.frame((.8, .2, 'Left'))
        self.assertFalse(self.app._piano_hands)

    def test_red_zone_clears_both_locks_through_mock_camera_path(self):
        a = self.app
        a.mock_hands = {'Left': (.25, .5), 'Right': (.75, .5)}
        a._step_mock()
        a.mock_hands = {'Left': (.25, .2), 'Right': (.75, .2)}
        a._step_mock()
        a.mock_hands = {'Left': (.25, .9), 'Right': (.75, .9)}
        a._step_mock()
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])
        self.assertFalse(a._piano_hands)
        self.assertEqual(a.orchestra_control.set_occupancy.call_args.args[-1], 0)

    def test_missing_locked_hand_stays_locked_until_all_hands_leave(self):
        self.frame((.25, .5, 'Left'), (.75, .5, 'Right'))
        self.frame((.25, .2, 'Left'), (.75, .5, 'Right'))
        self.app.last_osc_time = 0
        self.frame((.5, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 36, 60, 60, 72])
        self.frame()
        self.assertFalse(self.app._piano_hands)

    def test_absence_grace_retains_lock_until_expiry(self):
        a = self.app
        self.frame((.25, .5, 'Left'))
        self.frame((.25, .2, 'Left'))
        a.hand_grace.update([(.25, .2, 'Left')], ['Left'], a._hand_region, 10, .15)
        for now, locked in [(10.1, True), (10.2, False)]:
            a.active_regions = a.hand_grace.update([], [], a._hand_region, now, .15)
            a._update_region_engagement()
            self.assertEqual(bool(a._piano_hands), locked)

    def test_window_settings_do_not_resize_a_locked_hand(self):
        self.frame((.25, .5, 'Left'), (.75, .5, 'Right'))
        self.frame((.25, .2, 'Left'), (.75, .5, 'Right'))
        self.app._set_range_window(8)
        self.frame((.9, .2, 'Left'), (.75, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1][:3], [1, 36, 60])
        self.assertEqual(self.piano_messages()[-1][4] - self.piano_messages()[-1][3], 8)

    def test_failed_camera_frame_clears_locks_after_grace(self):
        a = self.app
        self.frame((.25, .5, 'Left'))
        self.frame((.25, .2, 'Left'))
        a._sync_gesture_control = Mock()
        a._apply_pending_view = Mock()
        a._pedal_mode_seen = True
        a.view = 'gesture'
        a.hand_grace_ms = 150
        a.hand_tracker = Mock()
        a.hand_tracker.get_frame_and_landmarks.return_value = (None, None)
        a._step()
        self.assertFalse(a._piano_hands)
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])

    def test_non_orchestra_piano_keeps_original_window(self):
        self.app.orchestra_mode = False
        self.frame((.25, .5, 'Left'))
        self.assertFalse(self.app._piano_hands)
        self.app.osc_client.send_message.assert_any_call('/setOutputRange', [1, 42, 54, -1, -1])


if __name__ == '__main__':
    unittest.main()
