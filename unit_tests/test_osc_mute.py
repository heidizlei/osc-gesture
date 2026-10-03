"""The OSC master mute: stops the wire, not merely the display."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy

sys.path.insert(0, str(Path(__file__).parents[1]))
# Only the send gate is under test, not capture or rendering.
with patch.dict(sys.modules, {'cv2': Mock(), 'classes.tracker': Mock()}):
    from classes.app import OSCGestureApp, _OSCLog


class OscMuteTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.app = a = OSCGestureApp.__new__(OSCGestureApp)
        client = Mock()
        client.send_message = lambda addr, val: self.sent.append((addr, val))
        a.osc_log = _OSCLog(client)
        a.osc_client = a.osc_log
        a.osc_sending = True
        a.osc_log.should_send = lambda address: a.osc_sending
        a.last_left = a.last_right = 60
        a._last_piano_layout = object()
        a._instr_state = {r: {'last_val': 7, 'last_time': 5.0}
                          for r in ('brass', 'strings')}

    def test_it_starts_on(self):
        self.assertTrue(self.app.osc_sending)

    def test_muting_stops_the_wire(self):
        self.app.osc_client.send_message('/setOutputRange', [1, 2])
        self.app._set_osc_sending(False)
        self.app.osc_client.send_message('/setOutputRange', [3, 4])
        self.assertEqual(self.sent, [('/setOutputRange', [1, 2])])

    def test_a_muted_send_reports_that_it_did_not_go(self):
        # Callers print off this return, so a suppressed message must not
        # claim to have reached the wire.
        self.assertTrue(self.app.osc_client.send_message('/x', [1]))
        self.app._set_osc_sending(False)
        self.assertFalse(self.app.osc_client.send_message('/x', [1]))

    def test_a_muted_send_is_not_logged_either(self):
        self.app._set_osc_sending(False)
        self.app.osc_client.send_message('/x', [1])
        self.assertEqual(self.app.osc_log.since(None)[0], [])

    def test_resuming_forgets_the_throttle_so_the_receiver_resyncs(self):
        # Whatever moved while muted never arrived; without this the next
        # value could be throttled away and leave the receiver stale.
        a = self.app
        a._set_osc_sending(False)
        a._set_osc_sending(True)
        self.assertIsNone(a.last_left)
        self.assertIsNone(a.last_right)
        self.assertIsNone(a._last_piano_layout)
        self.assertEqual([s['last_val'] for s in a._instr_state.values()], [None, None])

    def test_repeating_a_state_is_a_no_op(self):
        a = self.app
        a._set_osc_sending(False)
        a.last_left = 60
        a._set_osc_sending(False)      # already muted: must not resync
        self.assertEqual(a.last_left, 60)

    def test_resuming_sends_again(self):
        a = self.app
        a._set_osc_sending(False)
        a.osc_client.send_message('/x', [1])
        a._set_osc_sending(True)
        a.osc_client.send_message('/x', [2])
        self.assertEqual(self.sent, [('/x', [2])])


if __name__ == '__main__':
    unittest.main()
