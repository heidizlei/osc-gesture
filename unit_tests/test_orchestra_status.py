"""Status subscription timing without sockets, camera, or wall-clock sleeps."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'orchestra_status_transport', Path(__file__).parents[1] / 'classes/orchestra_control.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


class StatusSubscriptionTests(unittest.TestCase):
    def controller(self):
        control = transport.OrchestraControl.__new__(transport.OrchestraControl)
        control.send = Mock()
        control.lock = threading.RLock()
        control.requested = None
        control.gesture_requested = None
        control.gesture_preference = None
        control.status = {}
        control.received = 0
        control.last_subscription = float('-inf')
        control.simulated = False
        control.down = True
        control.occupancy = [1, 48, 61, 0]
        control.occupancy_seen = 100
        control.browser_seen = 100
        control.reply_host = '127.0.0.1'
        control.server = SimpleNamespace(server_address=('127.0.0.1', 19001))
        return control

    def test_idle_renews_without_continuous_queries(self):
        control = self.controller()
        now = [100.0]
        ticks = iter([100.0, 100.3, 101.0, 109.9, 110.0, 110.3, 119.9, 120.0])

        def wait(_):
            tick = next(ticks, None)
            if tick is None:
                return True
            now[0] = tick
            return False

        control.stop_event = SimpleNamespace(wait=wait, is_set=lambda: False)
        with patch.object(transport.time, 'monotonic', side_effect=lambda: now[0]):
            control._run()
        self.assertEqual(control.send.call_count, 3)
        for call in control.send.call_args_list:
            self.assertEqual(call.args, ('/subscribeOrchestraState', ['127.0.0.1', 19001, 30]))

    def test_quiet_subscription_stays_confirmed_until_stale(self):
        control = self.controller()
        with patch.object(transport.time, 'monotonic', return_value=100):
            self.assertFalse(control.state()['confirmed'])
            control._receive('/orchestraState', '{"mode":false}')
        with patch.object(transport.time, 'monotonic', return_value=110):
            self.assertTrue(control.state()['confirmed'])
        with patch.object(transport.time, 'monotonic', return_value=116):
            self.assertFalse(control.state()['confirmed'])

    def test_mode_change_requests_immediate_confirmation(self):
        control = self.controller()
        with patch.object(transport.time, 'monotonic', return_value=100):
            control.set_mode(True)
        self.assertEqual(control.send.call_args_list[0].args,
                         ('/setModelConfig', ['orchestraPianoPedalMode', 1]))
        self.assertEqual(control.send.call_args_list[1].args,
                         ('/subscribeOrchestraState', ['127.0.0.1', 19001, 30]))

    def test_bypass_stops_sends_immediately_and_waits_for_enable_ack(self):
        control = self.controller()
        control.status = {'mode': True, 'gestureControlEnabled': True}
        control.set_gesture_control(False)
        self.assertFalse(control.gesture_enabled)
        self.assertFalse(control.enabled)
        control._receive('/orchestraState', '{"mode":false,"gestureControlEnabled":false}')
        self.assertIsNone(control.gesture_requested)
        control.set_gesture_control(True)
        self.assertFalse(control.gesture_enabled)
        control._receive('/orchestraState', '{"mode":true,"gestureControlEnabled":true}')
        self.assertTrue(control.gesture_enabled)
        self.assertTrue(control.enabled)

    def test_normal_pedal_simulation_available_while_gestures_off(self):
        control = self.controller()
        control.status = {'mode': False, 'gestureControlEnabled': False, 'pedalDown': False}
        control.set_source(True)
        self.assertFalse(control.down)
        control.pedal(True)
        self.assertTrue(control.down)
        control.send.assert_any_call('/setOrchestraPianoPedal', 1)
        with self.assertRaises(ValueError):
            control.set_mode(True)

    def test_reconnect_reasserts_user_bypass_preference(self):
        control = self.controller()
        control.set_gesture_control(False)
        control._receive('/orchestraState', '{"mode":false,"gestureControlEnabled":false}')
        control._receive('/orchestraState', '{"mode":true,"gestureControlEnabled":true}')
        self.assertFalse(control.gesture_enabled)
        self.assertIs(control.gesture_requested, False)

    def test_enable_request_does_not_trust_an_old_on_snapshot(self):
        control = self.controller()
        control.status = {'mode': True, 'gestureControlEnabled': True}
        control.set_gesture_control(False)
        control.set_gesture_control(True)
        self.assertFalse(control.gesture_enabled)
        control._receive('/orchestraState', '{"mode":true,"gestureControlEnabled":true}')
        self.assertTrue(control.gesture_enabled)


if __name__ == '__main__':
    unittest.main()
