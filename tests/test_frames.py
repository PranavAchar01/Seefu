"""Unit tests for core/frames.py — crop persistence + frame ingest geometry.

Pure unit tests: CROP_FILE is redirected into a TemporaryDirectory so the real
data/crop.json is never touched, and the module-global _warned_sizes set is
saved/cleared/restored around every test so the one-shot warning is
deterministic regardless of test order.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import frames


class FramesTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_crop_file = frames.CROP_FILE
        frames.CROP_FILE = Path(self._tmp.name) / "crop.json"
        # _warned_sizes is module-global mutable state; isolate it per test.
        self._orig_warned = set(frames._warned_sizes)
        frames._warned_sizes.clear()

    def tearDown(self):
        frames.CROP_FILE = self._orig_crop_file
        frames._warned_sizes.clear()
        frames._warned_sizes.update(self._orig_warned)
        self._tmp.cleanup()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _frame(w, h):
        """A (h, w, 3) uint8 frame whose pixels encode their own coordinates
        so we can verify WHICH region a crop extracted, not just its shape."""
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[..., 0] = (np.arange(h, dtype=np.uint32) % 256)[:, None]  # row id
        f[..., 1] = (np.arange(w, dtype=np.uint32) % 256)[None, :]  # col id
        return f

    def _write_crop(self, x, y, w, h, note=""):
        frames.CROP_FILE.write_text(
            json.dumps({"x": x, "y": y, "w": w, "h": h, "note": note}))

    # -- load_crop ----------------------------------------------------------

    def test_load_crop_missing_returns_none(self):
        self.assertFalse(frames.CROP_FILE.exists())
        self.assertIsNone(frames.load_crop())

    def test_load_crop_present_returns_int_tuple(self):
        self._write_crop(100, 50, 640, 480)
        self.assertEqual(frames.load_crop(), (100, 50, 640, 480))

    def test_load_crop_coerces_floats_to_ints(self):
        # crop.json written by other tooling may hold floats; contract is ints.
        self._write_crop(10.0, 20.9, 300.5, 400.0)
        crop = frames.load_crop()
        self.assertEqual(crop, (10, 20, 300, 400))
        self.assertTrue(all(isinstance(v, int) for v in crop))

    # -- save_crop ----------------------------------------------------------

    def test_save_crop_in_bounds_is_untouched(self):
        self.assertEqual(frames.save_crop(100, 50, 640, 480),
                         (100, 50, 640, 480))

    def test_save_crop_clamps_negative_origin_to_zero(self):
        x, y, w, h = frames.save_crop(-30, -5, 640, 480)
        self.assertEqual((x, y), (0, 0))
        self.assertEqual((w, h), (640, 480))

    def test_save_crop_clamps_size_to_sensor_bounds(self):
        # w/h may not extend past the 1920x1080 sensor from (x, y).
        x, y, w, h = frames.save_crop(1800, 1000, 500, 500)
        self.assertEqual((x, y), (1800, 1000))
        self.assertEqual(w, frames.SENSOR_W - 1800)  # 120
        self.assertEqual(h, frames.SENSOR_H - 1000)  # 80

    def test_save_crop_full_sensor_passes_unchanged(self):
        self.assertEqual(frames.save_crop(0, 0, 1920, 1080),
                         (0, 0, 1920, 1080))

    def test_save_crop_coerces_floats(self):
        self.assertEqual(frames.save_crop(10.7, 20.2, 100.9, 50.1),
                         (10, 20, 100, 50))

    def test_save_crop_writes_note(self):
        frames.save_crop(1, 2, 3, 4, note="bench B, camera 2")
        data = json.loads(frames.CROP_FILE.read_text())
        self.assertEqual(data["note"], "bench B, camera 2")

    def test_save_crop_origin_past_sensor_is_clamped_positive(self):
        # origin beyond the sensor is clamped inside it and sizes stay >= 1,
        # so crop.json can never hold a degenerate/negative rectangle
        x, y, w, h = frames.save_crop(2000, -50, 100, 0)
        self.assertEqual(x, frames.SENSOR_W - 1)
        self.assertEqual(y, 0)
        self.assertGreaterEqual(w, 1)
        self.assertGreaterEqual(h, 1)
        self.assertLessEqual(x + w, frames.SENSOR_W)
        self.assertLessEqual(y + h, frames.SENSOR_H)
        # the clamped rect is degenerate (<50px) so load_crop refuses it
        self.assertIsNone(frames.load_crop())

    # -- crop file round-trip ----------------------------------------------

    def test_save_then_load_round_trip(self):
        saved = frames.save_crop(750, 210, 480, 480, note="demo bench")
        self.assertEqual(frames.load_crop(), saved)
        # Clamped save round-trips as the clamped values, not the request.
        saved = frames.save_crop(-10, 900, 4000, 4000)
        self.assertEqual(saved, (0, 900, 1920, 180))
        self.assertEqual(frames.load_crop(), saved)

    # -- ingest -------------------------------------------------------------

    def test_ingest_no_crop_defined_passes_frame_through(self):
        f = self._frame(1920, 1080)
        self.assertIs(frames.ingest(f), f)

    def test_ingest_crops_sensor_frame_exactly_per_crop_json(self):
        self._write_crop(300, 200, 640, 360)
        f = self._frame(1920, 1080)
        out = frames.ingest(f)
        self.assertEqual(out.shape, (360, 640, 3))
        # Pixel coordinate channels prove the region is [y:y+h, x:x+w].
        np.testing.assert_array_equal(out, f[200:560, 300:940])
        self.assertEqual(int(out[0, 0, 0]), 200 % 256)  # first row id = y
        self.assertEqual(int(out[0, 0, 1]), 300 % 256)  # first col id = x

    def test_ingest_crop_sized_frame_passes_through_silently(self):
        # Idempotence: a frame already at crop size is returned untouched,
        # with no warning.
        self._write_crop(300, 200, 640, 360)
        f = self._frame(640, 360)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = frames.ingest(f)
        self.assertIs(out, f)
        self.assertEqual(buf.getvalue(), "")

    def test_ingest_is_idempotent(self):
        # ingest(ingest(sensor_frame)) == ingest(sensor_frame).
        self._write_crop(300, 200, 640, 360)
        once = frames.ingest(self._frame(1920, 1080))
        twice = frames.ingest(once)
        self.assertIs(twice, once)

    def test_ingest_foreign_size_warns_once_and_passes_through(self):
        self._write_crop(300, 200, 640, 360)
        f = self._frame(1280, 720)  # neither sensor nor crop size
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = frames.ingest(f)
        self.assertIs(out, f)  # passed through UNCROPPED
        warning = buf.getvalue()
        self.assertIn("WARNING", warning)
        self.assertIn("1280x720", warning)
        self.assertIn("1920x1080", warning)  # names the expected sensor
        self.assertIn("640x360", warning)    # names the expected crop

        # Same foreign size again: silent (once per size).
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            out2 = frames.ingest(self._frame(1280, 720))
        self.assertEqual(out2.shape, (720, 1280, 3))
        self.assertEqual(buf2.getvalue(), "")

    def test_ingest_warns_again_for_each_new_foreign_size(self):
        self._write_crop(300, 200, 640, 360)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            frames.ingest(self._frame(1280, 720))
            frames.ingest(self._frame(800, 600))
        text = buf.getvalue()
        self.assertIn("1280x720", text)
        self.assertIn("800x600", text)
        self.assertEqual(text.count("WARNING"), 2)

    def test_ingest_grayscale_sensor_frame_is_cropped(self):
        # ingest only relies on shape[:2]; 2-D (no channel axis) frames work.
        self._write_crop(10, 20, 100, 50)
        f = np.zeros((1080, 1920), dtype=np.uint8)
        out = frames.ingest(f)
        self.assertEqual(out.shape, (50, 100))

    def test_ingest_full_sensor_crop_returns_whole_frame(self):
        self._write_crop(0, 0, 1920, 1080)
        f = self._frame(1920, 1080)
        out = frames.ingest(f)
        self.assertEqual(out.shape, (1080, 1920, 3))
        np.testing.assert_array_equal(out, f)


if __name__ == "__main__":
    unittest.main()
