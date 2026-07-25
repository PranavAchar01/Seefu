"""Unit tests for the pure helpers in inspection/serve.py.

Covers ws_score, extract_blobs, next_dish_id, read_threshold /
model_fallback_threshold, dashed_circle / save_overlay, build_findings,
and image_b64. Never loads the model, never starts the server, never
touches the camera; CALIB / RUNS / DISH_SEQ are redirected into a
TemporaryDirectory for every test.
"""

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inspection"))

import cv2  # noqa: E402

import serve  # noqa: E402


class RedirectedPathsTestCase(unittest.TestCase):
    """Base: every test runs with serve's path constants inside a tmp dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self._orig = (serve.CALIB, serve.RUNS, serve.DISH_SEQ)
        serve.CALIB = tmp / "calibration.json"
        serve.RUNS = tmp / "runs"
        serve.DISH_SEQ = serve.RUNS / ".dish_seq"

    def tearDown(self):
        serve.CALIB, serve.RUNS, serve.DISH_SEQ = self._orig
        self._tmp.cleanup()


# ---------------- ws_score ----------------

class TestWsScore(RedirectedPathsTestCase):

    def test_zero_raw_maps_to_zero(self):
        self.assertEqual(serve.ws_score(0.0, 10.0), 0.0)

    def test_raw_at_threshold_maps_to_half(self):
        self.assertEqual(serve.ws_score(10.0, 10.0), 0.5)
        self.assertEqual(serve.ws_score(43.7, 43.7), 0.5)

    def test_raw_at_twice_threshold_maps_to_one(self):
        self.assertEqual(serve.ws_score(20.0, 10.0), 1.0)

    def test_raw_beyond_twice_threshold_clamps_to_one(self):
        self.assertEqual(serve.ws_score(55.0, 10.0), 1.0)
        self.assertEqual(serve.ws_score(1e9, 10.0), 1.0)

    def test_negative_raw_clamps_to_zero(self):
        self.assertEqual(serve.ws_score(-4.0, 10.0), 0.0)

    def test_nonpositive_threshold_returns_zero(self):
        self.assertEqual(serve.ws_score(5.0, 0.0), 0.0)
        self.assertEqual(serve.ws_score(5.0, -3.0), 0.0)

    def test_rounding_to_three_decimals(self):
        # 1 / (2*3) = 0.1666... -> 0.167
        self.assertEqual(serve.ws_score(1.0, 3.0), 0.167)


# ---------------- extract_blobs ----------------

class TestExtractBlobs(RedirectedPathsTestCase):

    def test_single_blob_at_known_location(self):
        full = np.zeros((100, 200), dtype=np.float32)
        full[20:30, 50:60] = 5.0                      # 10x10 = 100 px
        blobs = serve.extract_blobs(full, 1.0, "defect")
        self.assertEqual(len(blobs), 1)
        b = blobs[0]
        self.assertEqual(b["cx"], round(54.5 / 200, 4))   # centroid col / w
        self.assertEqual(b["cy"], round(24.5 / 100, 4))   # centroid row / h
        self.assertEqual(b["r"], round(float(np.sqrt(100 / np.pi)) / 200, 4))
        self.assertEqual(b["score"], 5.0)

    def test_speckle_under_40px_filtered_area_boundary(self):
        full = np.zeros((100, 100), dtype=np.float32)
        full[10, 10:49] = 5.0                         # 1x39 = 39 px: speckle
        full[60:65, 60:68] = 3.0                      # 5x8 = 40 px: kept
        blobs = serve.extract_blobs(full, 1.0, "pass")
        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0]["score"], 3.0)
        self.assertEqual(blobs[0]["r"], round(float(np.sqrt(40 / np.pi)) / 100, 4))

    def test_pure_speckle_with_pass_verdict_returns_empty(self):
        full = np.zeros((100, 100), dtype=np.float32)
        full[10:14, 10:15] = 5.0                      # 20 px < 40
        self.assertEqual(serve.extract_blobs(full, 1.0, "pass"), [])

    def test_top5_cap_and_descending_score_order(self):
        full = np.zeros((50, 700), dtype=np.float32)
        for i in range(7):                            # 7 blobs, scores 1..7
            full[20:30, i * 100 + 10:i * 100 + 20] = float(i + 1)
        blobs = serve.extract_blobs(full, 0.5, "defect")
        self.assertEqual(len(blobs), 5)
        self.assertEqual([b["score"] for b in blobs], [7.0, 6.0, 5.0, 4.0, 3.0])

    def test_defect_empty_mask_quantile_fallback_yields_blob(self):
        # no pixel clears the threshold, but the hot 0.2% tail is a >=40px
        # region: the quantile fallback should surface it as a real blob
        full = np.zeros((100, 100), dtype=np.float32)
        full[40:50, 60:70] = 0.5                      # 100 px, all below threshold 1.0
        blobs = serve.extract_blobs(full, 1.0, "defect")
        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0]["cx"], round(64.5 / 100, 4))
        self.assertEqual(blobs[0]["cy"], round(44.5 / 100, 4))
        self.assertEqual(blobs[0]["score"], 0.5)

    def test_defect_empty_mask_hottest_pixel_fallback(self):
        # strictly increasing map: the 0.2% quantile tail is ~20 px (< 40),
        # dies to the speckle filter -> guaranteed single hottest-pixel blob
        full = (np.arange(10000, dtype=np.float64) / 10000.0).reshape(100, 100)
        blobs = serve.extract_blobs(full, 2.0, "defect")
        self.assertEqual(len(blobs), 1)
        b = blobs[0]
        self.assertEqual(b["cx"], 0.99)               # argmax at (99, 99)
        self.assertEqual(b["cy"], 0.99)
        self.assertEqual(b["r"], 0.02)                # fixed fallback radius
        self.assertEqual(b["score"], 0.9999)

    def test_defect_all_speckle_hottest_pixel_fallback(self):
        # mask is non-empty but every component is speckle: defect verdict
        # still guarantees one blob at the hottest pixel
        full = np.zeros((100, 100), dtype=np.float32)
        full[10:14, 10:15] = 5.0                      # 20 px above threshold
        blobs = serve.extract_blobs(full, 1.0, "defect")
        self.assertEqual(len(blobs), 1)
        b = blobs[0]
        self.assertEqual((b["cx"], b["cy"]), (0.1, 0.1))  # first max: (10, 10)
        self.assertEqual(b["r"], 0.02)
        self.assertEqual(b["score"], 5.0)

    def test_defect_near_constant_map_quantile_ties_give_whole_frame_blob(self):
        # CURRENT-BEHAVIOR PIN, arguably a latent bug: when the map is almost
        # entirely one constant background value (here zeros + one hot pixel),
        # np.quantile(full_map, 0.998) EQUALS the background, so the fallback
        # mask covers the whole frame and produces one giant blob centered on
        # the image (r ~ 0.56 = whole-dish radius) instead of a tight blob at
        # the hot pixel. Real PatchCore maps are continuous, so this only
        # bites on degenerate/synthetic input; documented for the orchestrator.
        full = np.zeros((100, 100), dtype=np.float32)
        full[50, 50] = 0.5
        blobs = serve.extract_blobs(full, 1.0, "defect")
        self.assertEqual(len(blobs), 1)
        self.assertEqual((blobs[0]["cx"], blobs[0]["cy"]), (0.495, 0.495))
        self.assertEqual(blobs[0]["r"],
                         round(float(np.sqrt(10000 / np.pi)) / 100, 4))  # 0.5642
        self.assertEqual(blobs[0]["score"], 0.5)

    def test_pass_verdict_empty_mask_returns_empty(self):
        full = np.full((60, 60), 0.2, dtype=np.float32)
        self.assertEqual(serve.extract_blobs(full, 1.0, "pass"), [])


# ---------------- next_dish_id ----------------

class TestNextBoardId(RedirectedPathsTestCase):

    def test_first_call_seeds_at_004_and_persists(self):
        self.assertFalse(serve.DISH_SEQ.exists())
        self.assertEqual(serve.next_dish_id(), "DISH-004")
        self.assertTrue(serve.RUNS.is_dir())          # created inside tmp
        self.assertEqual(serve.DISH_SEQ.read_text(), "5")

    def test_sequential_calls_increment(self):
        ids = [serve.next_dish_id() for _ in range(3)]
        self.assertEqual(ids, ["DISH-004", "DISH-005", "DISH-006"])

    def test_zero_padding_from_seeded_sequence(self):
        serve.RUNS.mkdir(parents=True, exist_ok=True)
        serve.DISH_SEQ.write_text("7")
        self.assertEqual(serve.next_dish_id(), "DISH-007")
        serve.DISH_SEQ.write_text("41")
        self.assertEqual(serve.next_dish_id(), "DISH-041")

    def test_four_digit_sequence_not_truncated(self):
        serve.RUNS.mkdir(parents=True, exist_ok=True)
        serve.DISH_SEQ.write_text("999")
        self.assertEqual(serve.next_dish_id(), "DISH-999")
        self.assertEqual(serve.next_dish_id(), "DISH-1000")


# ---------------- read_threshold / model_fallback_threshold ----------------

class _Stub:
    pass


def _stub_model(pp):
    m = _Stub()
    m.post_processor = pp
    return m


class TestReadThreshold(RedirectedPathsTestCase):

    def test_reads_threshold_from_calibration_json(self):
        serve.CALIB.write_text(json.dumps({"threshold": 37.25, "normal_max": 30.1}))
        self.assertEqual(serve.read_threshold(), (37.25, "calibration.json"))

    def test_int_threshold_coerced_to_float(self):
        serve.CALIB.write_text(json.dumps({"threshold": 12}))
        value, source = serve.read_threshold()
        self.assertIsInstance(value, float)
        self.assertEqual(value, 12.0)
        self.assertEqual(source, "calibration.json")

    def test_missing_calib_falls_back_to_model_adaptive(self):
        pp = _Stub()
        pp.image_threshold = 3.5
        with mock.patch.object(serve, "get_model", return_value=_stub_model(pp)):
            value, source = serve.read_threshold()
        self.assertEqual(value, 3.5)
        self.assertEqual(source, "model adaptive (UNCALIBRATED - run --calibrate)")

    def test_calib_without_threshold_key_falls_back_to_model(self):
        serve.CALIB.write_text(json.dumps({"px_mm": {"mm_per_px": [0.1, 0.1]}}))
        pp = _Stub()
        pp.image_threshold = 2.25
        with mock.patch.object(serve, "get_model", return_value=_stub_model(pp)):
            value, source = serve.read_threshold()
        self.assertEqual(value, 2.25)
        self.assertIn("UNCALIBRATED", source)

    def test_no_calib_no_model_threshold_raises(self):
        with mock.patch.object(serve, "get_model", return_value=_stub_model(None)):
            with self.assertRaises(RuntimeError) as ctx:
                serve.read_threshold()
        self.assertIn("--calibrate", str(ctx.exception))


class TestModelFallbackThreshold(RedirectedPathsTestCase):

    def test_reads_image_threshold(self):
        pp = _Stub()
        pp.image_threshold = 2.0
        self.assertEqual(serve.model_fallback_threshold(_stub_model(pp)), 2.0)

    def test_numpy_scalar_converts(self):
        pp = _Stub()
        pp.image_threshold = np.float32(2.5)
        self.assertEqual(serve.model_fallback_threshold(_stub_model(pp)), 2.5)

    def test_unconvertible_value_falls_through_to_private_name(self):
        pp = _Stub()
        pp.image_threshold = "not-a-number"          # ValueError -> next name
        pp._image_threshold = 7.0
        self.assertEqual(serve.model_fallback_threshold(_stub_model(pp)), 7.0)

    def test_type_error_value_falls_through(self):
        pp = _Stub()
        pp.image_threshold = object()                # TypeError -> next name
        pp._image_threshold = 4.0
        self.assertEqual(serve.model_fallback_threshold(_stub_model(pp)), 4.0)

    def test_no_post_processor_returns_none(self):
        self.assertIsNone(serve.model_fallback_threshold(_stub_model(None)))
        self.assertIsNone(serve.model_fallback_threshold(_Stub()))  # attr absent

    def test_no_threshold_attrs_returns_none(self):
        self.assertIsNone(serve.model_fallback_threshold(_stub_model(_Stub())))


# ---------------- dashed_circle / save_overlay ----------------

class TestOverlayDrawing(RedirectedPathsTestCase):

    def test_dashed_circle_draws_in_place(self):
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        serve.dashed_circle(img, (40, 40), 20, (0, 255, 0))
        self.assertTrue(img.any())                    # something was painted
        self.assertEqual(img[0, 0].tolist(), [0, 0, 0])   # corner untouched
        self.assertEqual(img[40, 40].tolist(), [0, 0, 0])  # center untouched

    def test_dashed_circle_clips_offscreen_without_error(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        serve.dashed_circle(img, (2, 2), 100, (255, 0, 0))  # mostly off-frame
        self.assertEqual(img.shape, (40, 40, 3))

    def test_save_overlay_writes_png_into_runs(self):
        bgr = np.random.default_rng(0).integers(
            0, 255, size=(60, 80, 3), dtype=np.uint8)
        full = np.tile(np.linspace(0.0, 2.0, 80, dtype=np.float32), (60, 1))
        blobs = [{"cx": 0.5, "cy": 0.5, "r": 0.05, "score": 1.5}]
        path = serve.save_overlay(bgr, full, blobs, 1.0)
        self.assertEqual(path.parent, serve.RUNS)     # stayed inside tmp
        self.assertTrue(path.name.startswith("overlay_"))
        self.assertEqual(path.suffix, ".png")
        written = cv2.imread(str(path))
        self.assertIsNotNone(written)
        self.assertEqual(written.shape, (60, 80, 3))

    def test_save_overlay_threshold_zero_uses_max_normalization(self):
        bgr = np.full((32, 32, 3), 128, dtype=np.uint8)
        full = np.zeros((32, 32), dtype=np.float32)   # max()=0 -> 1e-6 guard
        path = serve.save_overlay(bgr, full, [], 0.0)
        self.assertTrue(path.exists())
        self.assertIsNotNone(cv2.imread(str(path)))


# ---------------- build_findings ----------------

class TestBuildFindings(RedirectedPathsTestCase):

    def test_blob_becomes_coord_finding_with_empty_fix(self):
        blob = {"cx": 0.25, "cy": 0.75, "r": 0.03, "score": 33.0}
        findings = serve.build_findings([blob], 640, 480)
        self.assertEqual(findings, [{
            "designator": None,
            "component": None,
            "description": "Anomalous region at (0.25, 0.75)",
            "fix": None,
            "score": 33.0,
            "source": "anomaly map",
        }])

    def test_empty_blobs_gives_empty_findings(self):
        self.assertEqual(serve.build_findings([], 640, 480), [])


# ---------------- image_b64 ----------------

class TestHuePresenceBlobs(RedirectedPathsTestCase):
    """PatchCore is position-blind; the ingredient check is the positional
    layer. Reference: a brown plate with a green garnish disc in the centre."""

    def ref_plate(self):
        ref = np.full((240, 240, 3), (40, 60, 120), np.uint8)
        cv2.circle(ref, (120, 120), 40, (60, 200, 60), -1)
        return ref

    def check(self, upload):
        with mock.patch.object(serve, "reference_frame",
                               return_value=(self.ref_plate(), (0, 0, 240, 240))), \
             mock.patch.object(serve, "register_to_reference", return_value=None):
            return serve.hue_presence_blobs(upload)

    def test_missing_green_garnish_is_flagged_at_its_zone(self):
        upload = np.full((240, 240, 3), (40, 60, 120), np.uint8)   # garnish gone
        blobs = self.check(upload)
        self.assertTrue(blobs)
        self.assertAlmostEqual(blobs[0]["cx"], 0.5, delta=0.2)
        self.assertAlmostEqual(blobs[0]["cy"], 0.5, delta=0.2)

    def test_identical_plate_is_not_flagged(self):
        self.assertEqual(self.check(self.ref_plate()), [])

    def test_normally_shifted_ingredient_is_not_flagged(self):
        upload = np.full((240, 240, 3), (40, 60, 120), np.uint8)
        cv2.circle(upload, (135, 128), 40, (60, 200, 60), -1)      # plating shift
        self.assertEqual(self.check(upload), [])

    def test_size_mismatch_returns_empty(self):
        upload = np.full((100, 100, 3), (40, 60, 120), np.uint8)
        self.assertEqual(self.check(upload), [])


class TestImageB64(RedirectedPathsTestCase):

    def test_small_frame_roundtrips_as_jpeg(self):
        bgr = np.full((10, 20, 3), 200, dtype=np.uint8)
        out = serve.image_b64(bgr)
        raw = base64.b64decode(out)
        self.assertEqual(raw[:2], b"\xff\xd8")        # JPEG magic
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (10, 20, 3))

    def test_wide_frame_downscaled_to_max_width(self):
        bgr = np.zeros((100, 2000, 3), dtype=np.uint8)
        raw = base64.b64decode(serve.image_b64(bgr))
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (48, 960, 3))  # int(100*960/2000)=48


# ---------------- region_label / label_locations ----------------

class TestRegionLabel(unittest.TestCase):
    """The anomaly map is the only thing that knows WHERE a defect is; the vision
    model sees a context-free crop and used to guess a side of the dish, which
    landed wrong whenever upload normalization rotated the photo."""

    def test_corners(self):
        self.assertEqual(serve.region_label(0.1, 0.1), "upper left")
        self.assertEqual(serve.region_label(0.9, 0.1), "upper right")
        self.assertEqual(serve.region_label(0.1, 0.9), "lower left")
        self.assertEqual(serve.region_label(0.9, 0.9), "lower right")

    def test_centre_and_edges(self):
        self.assertEqual(serve.region_label(0.5, 0.5), "centre of dish")
        self.assertEqual(serve.region_label(0.1, 0.5), "left edge")
        self.assertEqual(serve.region_label(0.9, 0.5), "right edge")
        self.assertEqual(serve.region_label(0.5, 0.1), "upper centre")
        self.assertEqual(serve.region_label(0.5, 0.9), "lower centre")

    def test_boundaries_do_not_crash_and_stay_in_vocabulary(self):
        vocab = {"upper", "lower", "middle", "left", "right", "centre", "edge", "of", "dish"}
        for cx in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0):
            for cy in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0):
                label = serve.region_label(cx, cy)
                self.assertTrue(label)
                self.assertTrue(set(label.split()) <= vocab, label)


class TestLabelLocations(unittest.TestCase):
    def test_labels_each_finding_from_its_blob(self):
        result = {"blobs": [{"cx": 0.1, "cy": 0.1}, {"cx": 0.9, "cy": 0.9}],
                  "findings": [{"designator": None}, {"designator": "J1"}]}
        serve.label_locations(result)
        self.assertEqual([f["location"] for f in result["findings"]],
                         ["upper left", "lower right"])

    def test_no_findings_is_a_noop(self):
        result = {"blobs": [], "findings": []}
        self.assertEqual(serve.label_locations(result)["findings"], [])

    def test_upload_rotation_moves_the_label_with_the_blob(self):
        """A 180-degree normalization flips blob coords back in
        undo_display_transform; the label must follow the CORRECTED coords, which
        is the whole point of labelling after the undo rather than before."""
        before = {"blobs": [{"cx": 0.85, "cy": 0.15}], "findings": [{}]}
        serve.label_locations(before)
        self.assertEqual(before["findings"][0]["location"], "upper right")

        after_undo = {"blobs": [{"cx": round(1 - 0.85, 4), "cy": round(1 - 0.15, 4)}],
                      "findings": [{}]}
        serve.label_locations(after_undo)
        self.assertEqual(after_undo["findings"][0]["location"], "lower left")


if __name__ == "__main__":
    unittest.main()
