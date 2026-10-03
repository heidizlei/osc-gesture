"""Stand-in hands: placement, the detected-hand gate, auto-drop, dead camera."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy

sys.path.insert(0, str(Path(__file__).parents[1]))
# Only the merge/placement logic is under test, not capture or rendering.
with patch.dict(sys.modules, {'cv2': Mock(), 'classes.tracker': Mock()}):
    from classes.app import OSCGestureApp
from classes.orchestra_control import HandGrace
from classes.boundary_gate import BoundaryGate


class ManualHandTests(unittest.TestCase):
    def setUp(self):
        self.app = a = OSCGestureApp.__new__(OSCGestureApp)
        a.manual_hands = {}
        a._tracked_labels = []
        a._view_lock = threading.Lock()

    # ---- placement ----

    def test_place_and_clear_one_hand(self):
        self.app.set_manual_hands({'Left': [0.2, 0.3]})
        self.assertEqual(self.app.manual_hands, {'Left': (0.2, 0.3)})
        self.app.set_manual_hands({'Left': None})
        self.assertEqual(self.app.manual_hands, {})

    def test_coordinates_are_clamped_to_the_frame(self):
        self.app.set_manual_hands({'Right': [1.8, -0.4]})
        self.assertEqual(self.app.manual_hands, {'Right': (1.0, 0.0)})

    def test_bad_input_is_rejected(self):
        for bad in ({'Middle': [0.1, 0.1]}, {'Left': [0.1]}, {'Left': 'x'}):
            with self.assertRaises(ValueError):
                self.app.set_manual_hands(bad)
        with self.assertRaises(ValueError):
            self.app.set_manual_hands(['Left'])

    def test_clear_removes_both(self):
        self.app.set_manual_hands({'Left': [0.2, 0.3], 'Right': [0.8, 0.3]})
        self.app.clear_manual_hands()
        self.assertEqual(self.app.manual_hands, {})

    # ---- merging into detection ----

    def test_fills_the_slot_the_camera_lost(self):
        self.app.set_manual_hands({'Right': [0.8, 0.4]})
        merged = self.app._merge_manual_hands([(0.2, 0.5, 'Left')])
        self.assertEqual(merged, [(0.2, 0.5, 'Left'), (0.8, 0.4, 'Right')])

    def test_drops_itself_when_the_real_hand_returns(self):
        self.app.set_manual_hands({'Right': [0.8, 0.4]})
        merged = self.app._merge_manual_hands([(0.7, 0.5, 'Right')])
        self.assertEqual(merged, [(0.7, 0.5, 'Right')])
        self.assertEqual(self.app.manual_hands, {})     # placement is gone for good

    def test_a_detected_hand_never_loses_to_a_placement(self):
        self.app.set_manual_hands({'Left': [0.1, 0.1], 'Right': [0.9, 0.1]})
        merged = self.app._merge_manual_hands([(0.3, 0.5, 'Left'), (0.7, 0.5, 'Right')])
        self.assertEqual(merged, [(0.3, 0.5, 'Left'), (0.7, 0.5, 'Right')])
        self.assertEqual(self.app.manual_hands, {})

    def test_never_produces_a_third_hand(self):
        # Two unlabelled detections already fill the pair, so a placement for a
        # label that is technically "unseen" must still not be appended.
        self.app.set_manual_hands({'Left': [0.1, 0.1]})
        merged = self.app._merge_manual_hands([(0.3, 0.5, None), (0.7, 0.5, None)])
        self.assertEqual(len(merged), 2)

    def test_both_hands_stand_in_when_nothing_is_detected(self):
        self.app.set_manual_hands({'Left': [0.2, 0.4], 'Right': [0.8, 0.4]})
        merged = self.app._merge_manual_hands([])
        self.assertEqual(merged, [(0.2, 0.4, 'Left'), (0.8, 0.4, 'Right')])

    def test_tracked_labels_are_published_for_the_ui_gate(self):
        self.app._merge_manual_hands([(0.3, 0.5, 'Left')])
        self.assertEqual(self.app._tracked_labels, ['Left'])
        self.app._merge_manual_hands([])
        self.assertEqual(self.app._tracked_labels, [])


class LandmarkIndexTests(unittest.TestCase):
    """A stand-in has no landmarks. Orchestra mode rebuilds the active-hand
    indices by position, and those index straight into the detection results,
    so a stand-in reaching that list indexes past the end and the loop dies."""

    def setUp(self):
        self.app = a = OSCGestureApp.__new__(OSCGestureApp)
        a.orchestra_mode = True
        a.piano_only = True
        a.orchestra_split_ratio = .5
        a.active_area_ratio = .75
        a.min_hand_size = 0.0
        a.orchestra_control = SimpleNamespace(enabled=True, gesture_enabled=True,
                                              status={}, set_occupancy=Mock())
        a.hand_grace = HandGrace()
        a.hand_grace_ms = 150
        a.boundary_gate = BoundaryGate()
        a._piano_hands = {}
        a._last_piano_layout = None
        a._last_piano_centres = ()
        a._engaged = dict.fromkeys(('piano', 'brass', 'strings'), False)
        a._instr_state = {r: {'last_val': None, 'last_time': 0}
                          for r in ('brass', 'strings')}
        a._instr_ids = {0: 1, 1: 48, 2: 61}
        a.instr_ranges = {r: [24, 108] for r in a._engaged}
        a.intervals = {'piano': (-6, 6)}
        a.last_left = a.last_right = None
        a.left_val = a.right_val = None
        a.last_osc_time = 0
        a.osc_interval = .2
        a.change_threshold = 4
        a.mode = 'range'
        a.osc_client = Mock()
        a._view_lock = threading.Lock()
        a.manual_hands = {}
        a._tracked_labels = []
        a._update_hand_presence = Mock()
        a._sync_gesture_control = Mock()
        a._apply_pending_view = Mock()
        a._tick_fps = Mock()
        a._pedal_mode_seen = True
        a.view = 'gesture'
        a.draw_landmarks = False
        a._frame_lock = threading.Lock()
        a._latest_frame = None
        a._latest_hands = []
        a._frame_seq = 0
        a.gesture_detector = Mock()
        a.gesture_detector.update.return_value = ('noop', 0)
        a.gesture_result = ('noop', 0)
        # One real hand on the left; its world landmarks are the only ones.
        hand = [SimpleNamespace(x=.25, y=.5, z=0.0) for _ in range(21)]
        results = SimpleNamespace(
            hand_landmarks=[hand],
            hand_world_landmarks=[hand],
            handedness=[[SimpleNamespace(category_name='Right')]])   # mirrored -> Left
        a.hand_tracker = Mock(camera_url=None)
        a.hand_tracker.get_frame_and_landmarks.return_value = (
            numpy.zeros((10, 10, 3)), results)
        self.clock = patch.object(time, 'time', return_value=100)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_a_stand_in_beside_a_real_hand_does_not_index_past_the_landmarks(self):
        a = self.app
        a.set_manual_hands({'Right': [0.75, 0.50]})     # the slot the camera lost
        for i in range(20):                             # let the gate confirm both
            with patch.object(time, 'monotonic', return_value=i * 0.05):
                a._step_input()                         # used to raise IndexError
        # Both hands drive, but only the detected one can offer gesture landmarks.
        self.assertEqual(len(a._merge_manual_hands(
            [(0.25, 0.5, 'Left')])), 2)


class DeadCameraTests(unittest.TestCase):
    """A feed that has stopped is the case stand-ins exist for, so the output
    has to keep following them with no frame arriving at all."""

    def setUp(self):
        self.app = a = OSCGestureApp.__new__(OSCGestureApp)
        a.orchestra_mode = False
        a.orchestra_split_ratio = .5
        a.active_area_ratio = .75
        a.min_hand_size = 0.0
        a.piano_only = False
        a.orchestra_control = SimpleNamespace(enabled=True, gesture_enabled=True,
                                              status={}, set_occupancy=Mock())
        a.hand_grace = HandGrace()
        a.hand_grace_ms = 150
        a.boundary_gate = BoundaryGate()
        a._piano_hands = {}
        a._last_piano_layout = None
        a._last_piano_centres = ()
        a._engaged = dict.fromkeys(('piano', 'brass', 'strings'), False)
        a._instr_state = {r: {'last_val': None, 'last_time': 0}
                          for r in ('brass', 'strings')}
        a._instr_ids = {0: 1, 1: 48, 2: 61}
        a.instr_ranges = {r: [24, 108] for r in a._engaged}
        a.intervals = {'piano': (-6, 6)}
        a.last_left = a.last_right = None
        a.left_val = a.right_val = None
        a.last_osc_time = 0
        a.osc_interval = .2
        a.change_threshold = 4
        a.mode = 'range'
        a.osc_client = Mock()
        a._view_lock = threading.Lock()
        a.manual_hands = {}
        a._tracked_labels = ['Left', 'Right']   # what the camera saw before dying
        a._update_hand_presence = Mock()
        a._sync_gesture_control = Mock()
        a._apply_pending_view = Mock()
        a._pedal_mode_seen = True
        a.view = 'gesture'
        a.gesture_detector = Mock()
        a.gesture_result = ('noop', 0)
        # The camera is dead: every grab comes back with nothing.
        a.hand_tracker = Mock(camera_url=None)
        a.hand_tracker.get_frame_and_landmarks.return_value = (None, None)
        self.now = 0.0
        self.clock = patch.object(time, 'time', return_value=100)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def ranges(self):
        return [c.args[1] for c in self.app.osc_client.send_message.call_args_list
                if c.args[0] == '/setOutputRange']

    def test_a_stand_in_still_drives_with_no_frame(self):
        self.app.set_manual_hands({'Left': [0.25, 0.5]})
        with patch.object(time, 'monotonic', return_value=0):
            self.app._step_input()
        self.assertTrue(self.app.hand_present)
        self.assertEqual(self.app.active_regions, ['piano'])
        self.assertTrue(self.ranges(), "a placed hand sent no range")

    def test_moving_it_moves_the_output_with_no_frame(self):
        a = self.app
        a.set_manual_hands({'Left': [0.10, 0.5]})
        with patch.object(time, 'monotonic', return_value=0):
            a._step_input()
        low = a.left_val
        a.set_manual_hands({'Left': [0.90, 0.5]})
        a.last_osc_time = 0
        with patch.object(time, 'monotonic', return_value=1):
            a._step_input()
        self.assertGreater(a.left_val, low)

    def test_nothing_placed_still_resets_as_before(self):
        with patch.object(time, 'monotonic', return_value=0):
            self.app._step_input()
        self.assertFalse(self.app.hand_present)
        self.assertEqual(self.ranges(), [])

    def test_a_dead_camera_frees_both_slots_for_placement(self):
        with patch.object(time, 'monotonic', return_value=0):
            self.app._step_input()
        # Otherwise the UI would refuse to stand in for the very hands it lost.
        self.assertEqual(self.app._tracked_labels, [])

    def test_one_below_the_boundary_is_out_of_play(self):
        self.app.set_manual_hands({'Left': [0.25, 0.92]})
        with patch.object(time, 'monotonic', return_value=0):
            self.app._step_input()
        self.assertFalse(self.app.hand_present)
        self.assertEqual(self.app.active_regions, [])

    def run_loop(self, seconds=1.0, step=0.05):
        """Spin the loop as it spins live, so the gate settles and the send
        throttle expires. Both clocks advance: setUp freezes time.time() for
        the placement tests, and leaving it frozen here would throttle every
        range after the first and hide a drag that does not follow."""
        for _ in range(int(seconds / step)):
            self.now += step
            with patch.object(time, 'monotonic', return_value=self.now), \
                 patch.object(time, 'time', return_value=100 + self.now):
                self.app._step_input()

    def test_orchestra_mode_sends_a_range_with_no_frame(self):
        # Orchestra routing goes through the boundary gate, which a one-shot
        # step never satisfies: this is the path the show actually runs.
        a = self.app
        a.orchestra_mode = True
        a.piano_only = True
        a.set_manual_hands({'Left': [0.25, 0.50]})
        self.run_loop()
        self.assertIn('Left', a.boundary_gate.ready)
        self.assertEqual(self.ranges(), [[1, 42, 54, -1, -1]])

    def test_orchestra_mode_follows_a_drag_with_no_frame(self):
        a = self.app
        a.orchestra_mode = True
        a.piano_only = True
        a.set_manual_hands({'Left': [0.10, 0.50]})
        self.run_loop()
        a.set_manual_hands({'Left': [0.90, 0.50]})
        self.run_loop()
        sent = self.ranges()
        self.assertGreater(len(sent), 1, "the drag sent no new range")
        self.assertGreater(sent[-1][1], sent[0][1])

    def test_a_dead_camera_with_a_url_still_drives(self):
        # The camera-URL branch returns early unless the stream reads stale,
        # so it needs its own cover: a stand-in must survive that path too.
        a = self.app
        a.hand_tracker = Mock(camera_url='http://198.51.100.9/dead')
        a.hand_tracker.cap = SimpleNamespace(stale=True)
        a.hand_tracker.get_frame_and_landmarks.return_value = (None, None)
        a._publish_frame = Mock()
        a.orchestra_mode = True
        a.piano_only = True
        a.set_manual_hands({'Left': [0.25, 0.50]})
        self.run_loop()
        self.assertTrue(a.hand_present)
        self.assertEqual(self.ranges(), [[1, 42, 54, -1, -1]])


if __name__ == '__main__':
    unittest.main()
