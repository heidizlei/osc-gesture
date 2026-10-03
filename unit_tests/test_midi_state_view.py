"""Stage view controller (CC 30) in MIDI State gesture snapshots, without sockets."""
import importlib.util
from pathlib import Path
import struct
import threading
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    'midi_state_sender', Path(__file__).parents[1] / 'classes/midi_state_sender.py')
sender_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sender_module)


def controllers(clip):
    """{controller: value} from an encoded snapshot's body."""
    words = struct.unpack('>' + 'I' * ((len(clip) - 32) // 4), clip[32:])
    return {(word >> 8) & 0x7f: value
            for word, value in zip(words[3::3], words[4::3])}


class ViewControllerTests(unittest.TestCase):
    def test_every_snapshot_carries_the_view(self):
        score = sender_module.encode_snapshot({}, .375, .75, 0, 1, group=2)
        gesture = sender_module.encode_snapshot({}, .375, .75, 0, 1, group=2, view=True)
        self.assertEqual(len(gesture), 172)
        self.assertEqual(controllers(score)[30], 0)
        self.assertEqual(controllers(gesture)[30], 0xffffffff)
        self.assertEqual(sorted(controllers(gesture)), list(range(20, 31)))
        # UMP Group nibble is Group 2 minus one.
        self.assertEqual(struct.unpack('>I', gesture[44:48])[0] >> 24 & 0xf, 1)

    def sender(self):
        sender = sender_module.MidiStateSender.__new__(sender_module.MidiStateSender)
        sender.lock = threading.Lock()
        sender.snapshot = ({}, .375, .75, float('-inf'))
        sender.view = False
        sender.sequence = 0
        sender.targets = [(2, '/midi-state/2/0')]
        sender.client = Mock()
        return sender

    def sent_view(self, sender, missing=False):
        sender._send(missing=missing)
        return controllers(sender.client.send_message.call_args.args[1])[30]

    def test_toggle_and_osc_control(self):
        sender = self.sender()
        self.assertEqual(self.sent_view(sender), 0)
        sender.set_view(True)
        self.assertEqual(self.sent_view(sender), 0xffffffff)
        sender._receive_view('/showGestureView', 0)
        self.assertFalse(sender.view)
        sender._receive_view('/showGestureView', 127)
        self.assertTrue(sender.view)
        for ignored in ((), ('on',), (1, 2)):
            sender._receive_view('/showGestureView', *ignored)
            self.assertTrue(sender.view)

    def test_every_group_gets_the_snapshot(self):
        sender = self.sender()
        sender.targets = [(2, '/midi-state/2/1'), (3, '/midi-state/3/1')]
        sender.set_view(True)
        sender._send()
        calls = sender.client.send_message.call_args_list
        self.assertEqual([c.args[0] for c in calls], ['/midi-state/2/1', '/midi-state/3/1'])
        for call, group in zip(calls, (2, 3)):
            clip = call.args[1]
            self.assertEqual(struct.unpack('>I', clip[44:48])[0] >> 24 & 0xf, group - 1)
            self.assertEqual(sorted(controllers(clip)), list(range(20, 31)))
            self.assertEqual(controllers(clip)[30], 0xffffffff)

    def test_shutdown_snapshot_returns_to_score(self):
        sender = self.sender()
        sender.set_view(True)
        self.assertEqual(self.sent_view(sender, missing=True), 0)
        self.assertTrue(sender.view)


if __name__ == '__main__':
    unittest.main()
