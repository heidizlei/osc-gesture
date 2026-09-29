"""Deterministic coverage of piano/accompaniment OSC ordering."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'orchestra_timing', Path(__file__).parents[1] / 'classes/orchestra_timing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OrchestraTimingTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(module.time, 'monotonic', return_value=10).start()
        patch.object(module.threading, 'Timer').start()
        self.addCleanup(patch.stopall)
        self.send = Mock()
        self.control = module.OrchestraTiming(self.send, 1)

    def test_coalesces_latest_state_and_preserves_heartbeat(self):
        c = self.control
        c.send_message('/setOrchestraOccupancy', [1, 48, 61, 4])
        c.send_message('/setOutputRange', [1, 40, 60, -1, -1])
        self.clock.return_value = 10.1
        c.send_message('/setOrchestraOccupancy', [1, 48, 61, 2])
        c.send_message('/setOrchestraOccupancy', [1, 48, 61, 1])
        self.send.assert_called_with('/setOrchestraOccupancy', [1, 48, 61, 4])
        self.clock.return_value = 10.5
        c._flush()
        self.send.assert_called_with('/setOrchestraOccupancy', [1, 48, 61, 1])
        self.assertFalse(c.pending)

    def test_each_piano_update_extends_cooldown(self):
        c = self.control
        c.send_message('/setOutputRange', [1, 40, 60, -1, -1])
        c.send_message('/setActiveInstruments', [1, 48])
        self.clock.return_value = 10.4
        c.send_message('/setOutputRange', [1, 41, 61, -1, -1])
        self.clock.return_value = 10.5
        c._flush()
        self.assertTrue(c.pending)
        self.clock.return_value = 10.9
        c._flush()
        self.send.assert_called_with('/setActiveInstruments', [1, 48])

    def test_start_stop_and_shutdown_are_immediate(self):
        c = self.control
        c.send_message('/setOutputRange', [1, 40, 60, -1, -1])
        c.send_message('/setOrchestraOccupancy', [1, 48, 61, 4])
        self.send.assert_called_with('/setOrchestraOccupancy', [1, 48, 61, 4])
        c.send_message('/setOrchestraOccupancy', [1, 48, 61, 2])
        c.send_message('/setOrchestraOccupancy', [1, 48, 61, 0])
        self.assertFalse(c.pending)
        c.send_message('/setForcedInstruments', [48])
        c.close()
        c._flush()
        self.send.assert_called_with('/setOrchestraOccupancy', [1, 48, 61, 0])

    def test_mode_change_cancels_queued_updates(self):
        c = self.control
        c.send_message('/setOutputRange', [1, 40, 60, -1, -1])
        c.send_message('/setForcedInstruments', [48])
        c.send_message('/setGestureControl', 0)
        self.clock.return_value = 11
        c._flush()
        self.send.assert_called_with('/setGestureControl', 0)
