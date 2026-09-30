import unittest

from classes.boundary_gate import BoundaryGate


class BoundaryGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = BoundaryGate()

    def frame(self, y, t, x=0.5):
        return self.gate.update([(x, y, 'Left')], .75, t)

    def settle(self):
        for t in (0, .1, .2, .26):
            self.frame(.5, t)

    def test_first_entry_preserves_region_but_defers_later_updates(self):
        self.assertEqual(self.frame(.2, 0), [(0.5, .2, 'Left')])
        self.assertEqual(self.gate.entered, {'Left'})
        self.assertFalse(self.gate.ready)
        self.assertEqual(self.frame(.5, .1), [(0.5, .2, 'Left')])
        self.assertFalse(self.gate.entered)
        self.frame(.2, .2)
        self.assertEqual(self.frame(.2, .26), [(0.5, .2, 'Left')])
        self.assertEqual(self.gate.ready, {'Left'})

    def test_repeated_crossings_stay_red_then_clear_lift_recovers(self):
        self.settle()
        self.assertTrue(self.frame(.755, .3))  # hysteresis holds the first excursion
        self.assertFalse(self.gate.ready)
        for i, y in enumerate((.745, .755, .745, .755, .745)):
            self.assertFalse(self.frame(y, .35 + i * .05))
        for t in (.6, .7, .8):
            self.assertFalse(self.frame(.5, t))
        self.assertTrue(self.frame(.5, .86))
        self.assertEqual(self.gate.ready, {'Left'})

    def test_hovering_at_boundary_times_out_without_further_crossings(self):
        self.assertTrue(self.frame(.745, 0))
        self.frame(.745, .1)
        self.frame(.745, .2)
        self.assertFalse(self.frame(.745, .26))
        self.assertFalse(self.frame(.745, .35))

    def test_deep_red_exits_and_rearms_only_after_quiet_period(self):
        self.settle()
        self.assertFalse(self.frame(.9, .3))
        for t in (.4, .5, .6, .7, .8, .91):
            self.frame(.9, t)
        self.assertTrue(self.frame(.5, .95))
        self.assertFalse(self.gate.ready)

    def test_short_tracking_loss_does_not_confirm_entry(self):
        self.frame(.5, 0)
        self.gate.update([], .75, .1)
        self.frame(.5, .2)
        self.frame(.5, .3)
        self.assertFalse(self.gate.ready)

    def test_hands_are_independent(self):
        self.settle()
        for t, y in ((.3, .755), (.35, .745), (.4, .755)):
            positions = self.gate.update([(.2, y, 'Left'), (.8, .5, 'Right')],
                                         .75, t)
        self.assertEqual([p[2] for p in positions], ['Right'])


if __name__ == '__main__':
    unittest.main()
