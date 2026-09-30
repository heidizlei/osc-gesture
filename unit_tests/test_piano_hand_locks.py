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
from classes.boundary_gate import BoundaryGate


class PianoHandLockTests(unittest.TestCase):
    def setUp(self):
        self.app = a = OSCGestureApp.__new__(OSCGestureApp)
        a.orchestra_mode = True
        a.orchestra_split_ratio = .5
        a.active_area_ratio = .75
        a.orchestra_control = SimpleNamespace(enabled=True, gesture_enabled=True,
                                             status={}, set_occupancy=Mock())
        a.hand_grace = HandGrace()
        a.boundary_gate = BoundaryGate()
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
        self.assertEqual(self.piano_messages()[-1], [1, 42, 54, 78, 90])
        self.app.last_osc_time = 0
        self.frame((.9, .2, 'Left'), (.5, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 42, 54, 60, 72])
        self.frame((.1, .2, 'Right'), (.9, .2, 'Left'))
        self.assertEqual(self.piano_messages()[-1], [1, 42, 54, 60, 72])
        count = len(self.piano_messages())
        self.frame((.1, .2, 'Left'), (.9, .2, 'Right'))
        self.assertEqual(len(self.piano_messages()), count)
        self.frame((.5, .5, 'Left'), (.9, .2, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 60, 72, 60, 72])

    def test_single_hand_lock_is_not_reset_when_piano_region_empties(self):
        self.frame((.25, .5, 'Left'))
        self.frame((.8, .2, 'Left'))
        self.assertEqual(self.piano_messages()[-1], [1, 42, 54, -1, -1])
        bars = {bar['instr']: bar for bar in self.app._range_bars()}
        self.assertEqual(bars['piano']['spans'], [[42, 54]])
        self.frame()
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])
        self.assertFalse(self.app._piano_hands)
        count = len(self.piano_messages())
        self.frame()
        self.assertEqual(len(self.piano_messages()), count)
        self.frame((.8, .2, 'Left'))
        self.assertFalse(self.app._piano_hands)

    def test_return_from_strings_immediately_restores_direct_control(self):
        self.frame((.75, .5, 'Right'))
        self.frame((.75, .2, 'Right'))
        self.assertEqual(self.piano_messages()[-1][1:3], [77, 89])
        self.frame((.75, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1][1:3], [78, 90])
        self.assertFalse(self.app._piano_hands['Right']['locked'])

    def test_other_hand_update_preserves_a_lock(self):
        self.frame((.25, .5, 'Left'), (.75, .5, 'Right'))
        self.frame((.25, .5, 'Left'), (.75, .2, 'Right'))
        held_span = self.piano_messages()[-1][3:]
        self.app.last_osc_time = 0
        self.frame((.5, .5, 'Left'), (.75, .2, 'Right'))
        self.assertEqual(self.piano_messages()[-1][3:], held_span)
        self.assertTrue(self.app._piano_hands['Right']['locked'])

    def test_red_zone_clears_both_locks_through_mock_camera_path(self):
        a = self.app
        a.mock_hands = {'Left': (.25, .5), 'Right': (.75, .5)}
        for now in (0, .1, .2, .26):
            with patch.object(time, 'monotonic', return_value=now):
                a._step_mock()
        self.assertEqual(len(a._piano_hands), 2)
        a.mock_hands = {'Left': (.25, .2), 'Right': (.75, .2)}
        with patch.object(time, 'monotonic', return_value=.3):
            a._step_mock()
        a.mock_hands = {'Left': (.25, .9), 'Right': (.75, .9)}
        with patch.object(time, 'monotonic', return_value=.35):
            a._step_mock()
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])
        self.assertFalse(a._piano_hands)
        self.assertEqual(a.orchestra_control.set_occupancy.call_args.args[-1], 0)

    def test_boundary_entry_sends_initial_range_and_suppresses_bounce(self):
        a = self.app

        def step(y, now):
            a.mock_hands = {'Left': (.25, y), 'Right': (.75, .9)}
            with patch.object(time, 'monotonic', return_value=now):
                a._step_mock()
            return a.orchestra_control.set_occupancy.call_args.args[-1]

        self.assertEqual(step(.5, 0), 4)
        self.assertEqual(self.piano_messages(), [[1, 42, 54, -1, -1]])
        step(.5, .1)
        step(.5, .2)
        step(.5, .26)
        self.assertEqual(len(self.piano_messages()), 1)
        self.assertEqual(step(.755, .3), 4)
        self.assertEqual(step(.745, .35), 0)
        count = a.osc_client.send_message.call_count
        for now, y in ((.4, .755), (.45, .745), (.5, .755)):
            self.assertEqual(step(y, now), 0)
        self.assertEqual(a.osc_client.send_message.call_count, count)
        for now in (.55, .65, .75):
            self.assertEqual(step(.5, now), 0)
        self.assertEqual(step(.5, .81), 4)
        self.assertEqual(len(self.piano_messages()), 3)  # initial, reset, stable re-entry

    def test_camera_path_uses_the_same_entry_gate(self):
        a = self.app
        a._sync_gesture_control = Mock()
        a._apply_pending_view = Mock()
        a._tick_fps = Mock()
        a._pedal_mode_seen = True
        a.view = 'gesture'
        a.hand_grace_ms = 150
        a.hand_tracker = Mock(camera_url=None)
        hand = [SimpleNamespace(x=.25, y=.5) for _ in range(21)]
        results = SimpleNamespace(hand_landmarks=[hand],
                                  handedness=[[SimpleNamespace(category_name='Right')]])
        a.hand_tracker.get_frame_and_landmarks.return_value = (numpy.zeros((10, 10, 3)), results)
        a.gesture_detector = Mock()
        a.gesture_detector.update.return_value = ('noop', 0)
        a.gesture_result = ('noop', 0)
        a._extract_world_landmarks = Mock(return_value=[])
        a._extract_wrist_img = Mock(return_value=[])
        a._extract_image_landmarks = Mock(return_value=[])
        with patch.object(time, 'monotonic', return_value=0):
            a._step()
        self.assertEqual(a.orchestra_control.set_occupancy.call_args.args[-1], 4)
        self.assertEqual(self.piano_messages(), [[1, 42, 54, -1, -1]])
        for now in (.1, .2, .26):
            with patch.object(time, 'monotonic', return_value=now):
                a._step()
        self.assertEqual(len(self.piano_messages()), 1)
        for now, y in ((.3, .755), (.35, .745), (.4, .755)):
            for point in hand:
                point.y = y
            with patch.object(time, 'monotonic', return_value=now):
                a._step()
        self.assertEqual(a.orchestra_control.set_occupancy.call_args.args[-1], 0)
        self.assertFalse(a.hand_grace.seen)

    def test_first_range_precedes_activation_and_movement_waits_for_settling(self):
        a = self.app
        messages = Mock()
        messages.attach_mock(a.orchestra_control.set_occupancy, 'occupancy')
        messages.attach_mock(a.osc_client.send_message, 'send')
        messages.attach_mock(a._update_hand_presence, 'presence')

        def step(x, now):
            a.mock_hands = {'Left': (x, .5), 'Right': (.75, .9)}
            with patch.object(time, 'monotonic', return_value=now), \
                    patch.object(time, 'time', return_value=100 + now):
                a._step_mock()

        step(.25, 0)
        calls = messages.mock_calls
        first_range = next(i for i, call in enumerate(calls) if call[0] == 'send')
        first_activation = next(i for i, call in enumerate(calls)
                                if call[0] == 'occupancy' and call.args[-1] == 4)
        first_presence = next(i for i, call in enumerate(calls) if call[0] == 'presence')
        self.assertLess(first_range, first_activation)
        self.assertLess(first_range, first_presence)
        self.assertEqual(self.piano_messages(), [[1, 42, 54, -1, -1]])
        step(.5, .1)
        step(.5, .2)
        self.assertEqual(len(self.piano_messages()), 1)
        step(.5, .26)
        self.assertEqual(self.piano_messages()[-1], [1, 60, 72, -1, -1])
        self.assertEqual(len(self.piano_messages()), 2)

    def test_missing_locked_hand_stays_locked_until_all_hands_leave(self):
        self.frame((.25, .5, 'Left'), (.75, .5, 'Right'))
        self.frame((.25, .2, 'Left'), (.75, .5, 'Right'))
        self.app.last_osc_time = 0
        self.frame((.5, .5, 'Right'))
        self.assertEqual(self.piano_messages()[-1], [1, 42, 54, 60, 72])
        self.frame()
        self.assertFalse(self.app._piano_hands)

    def test_upper_entry_range_precedes_activation_in_both_transports(self):
        a = self.app
        for pedal in (True, False):
            for region, label, x, mask in (('brass', 'Left', .25, 1),
                                           ('strings', 'Right', .75, 2)):
                with self.subTest(pedal=pedal, region=region):
                    a.orchestra_control.enabled = pedal
                    a.piano_only = True
                    a._last_active = [1]
                    a._last_forced = []
                    a._engaged = dict.fromkeys(a._engaged, False)
                    a.boundary_gate.clear()
                    a.osc_client.send_message.reset_mock()
                    a.orchestra_control.set_occupancy.reset_mock()
                    messages = Mock()
                    messages.attach_mock(a.orchestra_control.set_occupancy, 'occupancy')
                    messages.attach_mock(a.osc_client.send_message, 'send')
                    # Even a recent identical range must not suppress entry.
                    a._instr_state[region] = dict(last_time=100,
                        last_val=a.map_hand_x_to_val(x, region))
                    a.mock_hands = {'Left': (.25, .9), 'Right': (.75, .9)}
                    a.mock_hands[label] = (x, .2)
                    with patch.object(time, 'monotonic', return_value=0):
                        a._step_mock()
                    calls = messages.mock_calls
                    midi_id = a._instrument_id(region)
                    range_index = next(i for i, call in enumerate(calls)
                        if call[0] == 'send' and call.args[0] == '/setOutputRange'
                        and call.args[1][0] == midi_id)
                    if pedal:
                        activations = [i for i, call in enumerate(calls)
                            if call[0] == 'occupancy' and call.args[-1] == mask]
                    else:
                        activations = [i for i, call in enumerate(calls)
                            if call[0] == 'send' and call.args[0] in
                            ('/setActiveInstruments', '/setForcedInstruments')
                            and midi_id in call.args[1]]
                        self.assertEqual(len(activations), 2)
                    self.assertTrue(activations)
                    self.assertTrue(all(range_index < i for i in activations))
                    self.assertFalse(a._piano_hands)
                    count = a.osc_client.send_message.call_count
                    a.mock_hands[label] = (1 - x, .2)
                    for now in (.1, .2):
                        with patch.object(time, 'monotonic', return_value=now):
                            a._step_mock()
                    self.assertEqual(a.osc_client.send_message.call_count, count)

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
        self.assertEqual(self.piano_messages()[-1][:3], [1, 42, 54])
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
        a.hand_tracker = Mock(camera_url=None)
        a.hand_tracker.get_frame_and_landmarks.return_value = (None, None)
        a._step()
        self.assertFalse(a._piano_hands)
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])

    def test_remote_frame_gap_retains_locks_until_stream_is_stale(self):
        a = self.app
        self.frame((.25, .5, 'Left'))
        self.frame((.25, .2, 'Left'))
        a._sync_gesture_control = Mock()
        a._apply_pending_view = Mock()
        a._pedal_mode_seen = True
        a.view = 'gesture'
        a.hand_grace_ms = 150
        a.hand_tracker = Mock(camera_url='http://stage/stream')
        a.hand_tracker.cap.stale = False
        a.hand_tracker.get_frame_and_landmarks.return_value = (None, None)
        a.gesture_detector = Mock()
        a._publish_frame = Mock()
        a._step()
        self.assertTrue(a._piano_hands)
        a._update_hand_presence.assert_not_called()
        a.gesture_detector.reset.assert_not_called()

        a.hand_tracker.cap.stale = True
        a._step()
        self.assertFalse(a._piano_hands)
        self.assertFalse(a.hand_present)
        self.assertEqual(self.piano_messages()[-1], [1, 24, 108, -1, -1])
        a._update_hand_presence.assert_called_once()
        a.gesture_detector.reset.assert_called_once()
        a._publish_frame.assert_called_once_with(None, None, [])

    def test_non_orchestra_piano_keeps_original_window(self):
        self.app.orchestra_mode = False
        self.frame((.25, .5, 'Left'))
        self.assertFalse(self.app._piano_hands)
        self.app.osc_client.send_message.assert_any_call('/setOutputRange', [1, 42, 54, -1, -1])

    def use_defaults(self):
        self.app.instr_ranges = {'piano': (26, 89), 'strings': (33, 94), 'brass': (36, 92)}
        self.app.interval = (-8, 8)

    def advance(self, *positions):
        self.app.last_osc_time = 0
        for state in self.app._instr_state.values():
            state['last_time'] = 0
        self.frame(*positions)

    def test_strings_offset_ceiling_and_display(self):
        self.use_defaults()
        self.frame((.75, .2, 'Right'))
        self.app.osc_client.send_message.assert_any_call('/setOutputRange', [48, 69, 85, -1, -1])
        self.advance((1, .2, 'Right'))
        self.app.osc_client.send_message.assert_any_call('/setOutputRange', [48, 78, 94, -1, -1])
        bars = {bar['instr']: bar for bar in self.app._range_bars()}
        self.assertEqual(bars['strings']['spans'], [[78, 94]])
        self.app._set_instrument_range('strings', 33, 90)
        self.advance((1, .2, 'Right'))
        self.app.osc_client.send_message.assert_any_call('/setOutputRange', [48, 74, 90, -1, -1])

    def test_descent_and_ascent_restore_original_lock_without_drift(self):
        self.use_defaults()
        self.frame((.75, .5, 'Right'))
        for _ in range(2):
            for x, expected in ((.75, [61, 77]), (.6, [55, 71]),
                                (.5, [49, 65]), (.1, [25, 41]),
                                (.5, [49, 65]), (.6, [55, 71]),
                                (.75, [61, 77]), (1, [61, 77])):
                self.advance((x, .2, 'Right'))
                self.assertEqual(self.piano_messages()[-1][1:3], expected)
                self.assertEqual(self.app._piano_hands['Right']['home_window'], (61, 77))

    def test_avoidance_uses_sent_strings_not_throttled_hand_position(self):
        self.use_defaults()
        self.frame((.75, .5, 'Right'))
        self.frame((.75, .2, 'Right'))
        count = len(self.piano_messages())
        self.frame((.6, .2, 'Right'))
        self.assertEqual(len(self.piano_messages()), count)
        self.assertEqual(self.piano_messages()[-1][1:3], [61, 77])
        self.advance((.6, .2, 'Right'))
        self.assertEqual(self.piano_messages()[-1][1:3], [55, 71])

    def test_combined_piano_overlap_counts_unique_notes(self):
        self.use_defaults()
        # The two home windows together cover
        # all 17 strings pitches. Enforce the cap on their union.
        self.frame((.4, .5, 'Left'), (.75, .5, 'Right'))
        self.advance((.25, .2, 'Left'), (.5, .2, 'Right'))
        spans = self.piano_messages()[-1][1:]
        piano = set(range(spans[0], spans[1] + 1)) | set(range(spans[2], spans[3] + 1))
        self.assertLessEqual(len(piano & set(range(54, 71))), 12)
        for x in (.4, .3, .2, .4, .6, .8, 1):
            self.advance((.25, .2, 'Left'), (x, .2, 'Right'))
            spans = self.piano_messages()[-1][1:]
            piano = set(range(spans[0], spans[1] + 1)) | set(range(spans[2], spans[3] + 1))
            centre = self.app._instr_state['strings']['last_val']
            self.assertLessEqual(len(piano & set(range(centre - 8, centre + 9))), 12)
        self.assertEqual(spans, [45, 61, 61, 77])

    def test_direct_piano_control_has_priority_over_avoidance(self):
        self.use_defaults()
        self.frame((.5, .5, 'Left'), (.5, .2, 'Right'))
        self.assertEqual(self.piano_messages()[-1][1:3], [50, 66])
        self.assertFalse(self.app._piano_hands['Left']['locked'])

    def test_midi_floor_and_strings_exit(self):
        self.use_defaults()
        self.app.instr_ranges['strings'] = (0, 32)
        self.frame((.75, .5, 'Right'))
        # Simulate an already displaced lock descending toward MIDI zero.
        hand = self.app._piano_hands['Right']
        hand.update(locked=True, home_window=(61, 77), avoidance_shift=30)
        self.advance((0, .2, 'Right'))
        lo, hi = self.piano_messages()[-1][1:3]
        self.assertGreaterEqual(lo, 0)
        centre = self.app._instr_state['strings']['last_val']
        self.assertLessEqual(len(set(range(lo, hi + 1)) & set(range(centre - 8, centre + 9))), 12)
        self.advance((.25, .2, 'Left'))
        self.assertEqual(self.piano_messages()[-1][1:3], [61, 77])


if __name__ == '__main__':
    unittest.main()
