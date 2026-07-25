"""Unit tests for capture/capture.py.

Pure unit tests: no camera, no server, no port, no OpenAI network, no PatchCore
checkpoint. All path constants (core.frames.CROP_FILE) are redirected into a
TemporaryDirectory so real data/, runs/, and the sqlite db are never touched.

Run: .venv/bin/python -m unittest tests.test_capture_tools -v   (from repo root)
"""

import contextlib
import io
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import frames  # noqa: E402
from capture import capture  # noqa: E402


# ---------------------------------------------------------------------------
# capture.py
# ---------------------------------------------------------------------------

class CaptureBase(unittest.TestCase):
    """Redirect the crop file into a tmp dir; nothing touches data/ or runs/."""

    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.tmp_path = Path(self.tmp)
        self.crop_file = self.tmp_path / "crop.json"          # does not exist yet
        self.enterContext(mock.patch.object(frames, "CROP_FILE", self.crop_file))
        # _warned_sizes is module-global once-per-size state; isolate it.
        self._saved_warned = set(frames._warned_sizes)
        frames._warned_sizes.clear()
        self.addCleanup(self._restore_warned)

    def _restore_warned(self):
        frames._warned_sizes.clear()
        frames._warned_sizes.update(self._saved_warned)

    def write_crop(self, x=100, y=50, w=640, h=480):
        self.crop_file.write_text(json.dumps({"x": x, "y": y, "w": w, "h": h}))
        return x, y, w, h


class TestMeanBrightness(CaptureBase):

    def test_black_frame_is_zero(self):
        self.assertEqual(capture.mean_brightness(np.zeros((8, 8, 3), np.uint8)), 0.0)

    def test_white_frame_is_255(self):
        self.assertEqual(capture.mean_brightness(np.full((8, 8, 3), 255, np.uint8)), 255.0)

    def test_uniform_gray_equals_value(self):
        # equal BGR channels -> gray == value exactly (cv2 coeffs sum to 1)
        self.assertEqual(capture.mean_brightness(np.full((10, 10, 3), 100, np.uint8)), 100.0)

    def test_half_black_half_white(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        frame[5:, :] = 255
        self.assertAlmostEqual(capture.mean_brightness(frame), 127.5)

    def test_channel_weighting_blue_is_dim(self):
        # pure blue in BGR -> gray ~= 0.114 * 255 ~= 29, far below 255/3
        frame = np.zeros((10, 10, 3), np.uint8)
        frame[:, :, 0] = 255
        self.assertAlmostEqual(capture.mean_brightness(frame), 29.0, delta=1.0)

    def test_returns_python_float(self):
        self.assertIsInstance(capture.mean_brightness(np.zeros((4, 4, 3), np.uint8)), float)


class TestSaveFrame(CaptureBase):

    def _save(self, frame, sub="out"):
        out_dir = self.tmp_path / sub
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            path = capture.save_frame(frame, out_dir)
        return path, buf.getvalue()

    def test_sensor_frame_is_cropped_on_save(self):
        x, y, w, h = self.write_crop(100, 50, 640, 480)
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        path, out = self._save(frame)
        self.assertTrue(path.exists())
        saved = cv2.imread(str(path))
        self.assertEqual(saved.shape, (480, 640, 3))
        # PNG is lossless: content must be exactly the crop window
        np.testing.assert_array_equal(saved, frame[y:y + h, x:x + w])
        self.assertIn("saved", out)
        self.assertIn("mean_brightness=", out)

    def test_precropped_frame_passes_through_unchanged(self):
        self.write_crop(100, 50, 640, 480)
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        path, out = self._save(frame)
        saved = cv2.imread(str(path))
        np.testing.assert_array_equal(saved, frame)
        # matches the crop size exactly -> no resolution warning
        self.assertNotIn("WARNING", out)

    def test_unknown_size_passes_through_with_warning(self):
        self.write_crop(100, 50, 640, 480)
        frame = np.full((200, 300, 3), 40, np.uint8)
        path, out = self._save(frame)
        saved = cv2.imread(str(path))
        self.assertEqual(saved.shape, (200, 300, 3))
        self.assertIn("WARNING", out)  # neither sensor-sized nor crop-sized

    def test_no_crop_file_sensor_frame_uncropped(self):
        frame = np.full((1080, 1920, 3), 10, np.uint8)
        path, _ = self._save(frame)
        saved = cv2.imread(str(path))
        self.assertEqual(saved.shape, (1080, 1920, 3))

    def test_creates_out_dir_and_png_name(self):
        path, _ = self._save(np.zeros((16, 16, 3), np.uint8), sub="deep/nested")
        self.assertEqual(path.parent, self.tmp_path / "deep/nested")
        self.assertEqual(path.suffix, ".png")
        self.assertTrue(path.name.startswith("frame_"))


class TestCropFrameGuide(CaptureBase):

    def test_guide_math(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        out = capture.crop_frame(frame, (0.1, 0.2, 0.9, 1.0))
        self.assertEqual(out.shape, (80, 160, 3))

    def test_full_guide_is_identity_shape(self):
        frame = np.zeros((50, 60, 3), np.uint8)
        out = capture.crop_frame(frame, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(out.shape, frame.shape)


class TestDrawHud(CaptureBase):

    def test_without_crop_file_same_shape_and_in_place(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        out = capture.draw_hud(frame, saved_count=3, note="hello")
        self.assertIs(out, frame)                 # contract: mutates in place
        self.assertEqual(out.shape, (240, 320, 3))
        self.assertGreater(int(out.sum()), 0)     # HUD actually drawn

    def test_with_crop_file_on_sensor_frame_draws_crop_box(self):
        x, y, w, h = self.write_crop(100, 50, 640, 480)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        out = capture.draw_hud(frame, saved_count=0)
        self.assertEqual(out.shape, (1080, 1920, 3))
        # crop rectangle color (255, 200, 60) lands on the top-left corner
        self.assertTrue((out[y, x] == np.array([255, 200, 60])).all())

    def test_with_crop_file_on_small_frame_no_crop_box_no_crash(self):
        # crop box only drawn when the preview is exactly 1920x1080 (by design)
        self.write_crop(100, 50, 640, 480)
        frame = np.zeros((240, 320, 3), np.uint8)
        out = capture.draw_hud(frame, saved_count=1, cropping=True)
        self.assertEqual(out.shape, (240, 320, 3))

    def test_tiny_frame_text_offscreen_does_not_crash(self):
        # counter is drawn at y=110; a 60px-tall frame just clips it
        out = capture.draw_hud(np.zeros((60, 80, 3), np.uint8), saved_count=9)
        self.assertEqual(out.shape, (60, 80, 3))


class TestStats(CaptureBase):

    def _img(self, name, value, size=(10, 10)):
        p = self.tmp_path / name
        cv2.imwrite(str(p), np.full((*size, 3), value, np.uint8))
        return p

    def test_stats_math_two_frames(self):
        self._img("a.png", 100)
        self._img("b.png", 50)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            capture.stats(self.tmp_path)
        out = buf.getvalue()
        self.assertIn("a.png  mean_brightness=100.00", out)
        self.assertIn("b.png  mean_brightness=50.00", out)
        self.assertIn("2 frames  mean=75.00  std=25.000  range=50.00", out)

    def test_stats_ignores_non_image_files(self):
        self._img("a.png", 60)
        (self.tmp_path / "notes.txt").write_text("not an image")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            capture.stats(self.tmp_path)
        out = buf.getvalue()
        self.assertIn("1 frames", out)
        self.assertNotIn("notes.txt", out)

    def test_stats_empty_folder_exits(self):
        with self.assertRaises(SystemExit):
            capture.stats(self.tmp_path)

    def test_stats_unreadable_image_raises_cv2_error(self):
        # CURRENT BEHAVIOR (documented, not a spec): a file with an image
        # suffix that cv2.imread cannot decode returns None, and
        # mean_brightness(None) raises cv2.error instead of a clean message.
        # A corrupt frame in the folder therefore crashes --stats.
        (self.tmp_path / "corrupt.png").write_text("this is not a png")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(cv2.error):
            capture.stats(self.tmp_path)

if __name__ == "__main__":
    unittest.main()
