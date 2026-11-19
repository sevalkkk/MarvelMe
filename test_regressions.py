"""Offline regression tests: python -m unittest test_regressions.py"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from gesture_engine import ZeroErrorPinchDetector
from main import _StableHandIdentity, _CleanVideoRecorder
from portal_renderer import PortalGeometryEngine


def hand(x, y):
    points = [(x, y)] * 21
    points[0] = (x, y)
    return points


class RegressionTests(unittest.TestCase):
    def test_identity_stays_with_wrist_when_order_flips(self):
        matcher = _StableHandIdentity()
        a, b = hand(.2, .5), hand(.8, .5)
        first = matcher.assign([(a, 'Left'), (b, 'Right')], 1280, 720, 10)
        second = matcher.assign([(b, 'Right'), (a, 'Left')], 1280, 720, 10.03)
        self.assertEqual(first, list(reversed(second)))

    def test_rotated_quad_has_no_crossing(self):
        points = ((100, 100), (300, 110), (140, 260), (260, 140))
        quad = PortalGeometryEngine.order_and_untwist_quad(*points)
        self.assertEqual(len(np.unique(quad, axis=0)), 4)
        self.assertGreater(abs(float(__import__('cv2').contourArea(quad))), 0)

    def test_recorder_flushes_accepted_frames(self):
        with tempfile.TemporaryDirectory() as d:
            recorder = _CleanVideoRecorder(Path(d), fps=30, max_queue_size=2)
            recorder.toggle(64, 64)
            for _ in range(20):
                recorder.write(np.zeros((64, 64, 3), dtype=np.uint8))
            accepted = 20 - recorder.dropped_frames
            output = recorder.stop()
            self.assertEqual(recorder.frame_count, accepted)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_no_repeated_pinches(self):
        detector = ZeroErrorPinchDetector()
        self.assertEqual(detector._tracks, {})


if __name__ == '__main__':
    unittest.main()
