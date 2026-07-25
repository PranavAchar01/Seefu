"""Unit tests for inspection/explain.py - no network, client stubbed.

CACHE_FILE redirected to a tempdir; explain._client injected directly so the
real OpenAI SDK is never constructed. API-failure paths raise the SDK's real
exception classes (constructed with dummy httpx objects) since describe_crop
catches exactly those types.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inspection"))

import numpy as np

import explain

CANNED = "FAIL: Scallion garnish is missing from the beef. | FIX: re-garnish with scallion and send"


class FakeClient:
    """Stands in for OpenAI(); records calls, returns canned text or raises."""

    def __init__(self, text=CANNED, exc=None):
        self.text, self.exc, self.calls = text, exc, 0

    def with_options(self, **_kw):
        return self

    @property
    def responses(self):
        return self

    def create(self, **_kw):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return type("R", (), {"output_text": self.text})()


def timeout_exc():
    import httpx
    from openai import APITimeoutError
    return APITimeoutError(request=httpx.Request("POST", "https://api.test"))


def make_crop(seed=0, size=200):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


class ExplainTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_cache_file = explain.CACHE_FILE
        explain.CACHE_FILE = Path(self.tmp.name) / "explain_cache.json"
        explain._cache = None
        self._orig_client = explain._client

    def tearDown(self):
        explain.CACHE_FILE = self._orig_cache_file
        explain._client = self._orig_client
        explain._cache = None
        self.tmp.cleanup()


class TestCropBlob(unittest.TestCase):
    def test_min_size_square_centered(self):
        frame = np.zeros((800, 1000, 3), np.uint8)
        crop = explain.crop_blob(frame, {"cx": 0.5, "cy": 0.5, "r": 0.001})
        self.assertEqual(crop.shape[:2], (160, 160))            # min_px floor

    def test_pad_scales_with_radius(self):
        frame = np.zeros((800, 1000, 3), np.uint8)
        crop = explain.crop_blob(frame, {"cx": 0.5, "cy": 0.5, "r": 0.1})
        half = int(0.1 * 1000 * 2.2)
        self.assertEqual(crop.shape[:2], (2 * half, 2 * half))

    def test_clamped_at_frame_corner(self):
        frame = np.zeros((800, 1000, 3), np.uint8)
        crop = explain.crop_blob(frame, {"cx": 0.01, "cy": 0.01, "r": 0.001})
        self.assertGreater(crop.size, 0)
        self.assertLessEqual(crop.shape[0], 160)                # clipped by edge


class TestDescribeCrop(ExplainTestBase):
    def test_no_api_key_falls_back(self):
        explain._client = None
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            text, fix, _cos, how = explain.describe_crop(
                make_crop(), make_crop(9), "Anomalous region at (0.5, 0.5)")
        self.assertEqual(text, "Anomalous region at (0.5, 0.5)")
        self.assertIsNone(fix)
        self.assertIn("no OPENAI_API_KEY", how)

    def test_success_returns_vision_fix_and_caches(self):
        fake = FakeClient()
        explain._client = fake
        text, fix, _cos, how = explain.describe_crop(make_crop(1), make_crop(9), "fb")
        self.assertEqual((text, how),
                         ("Scallion garnish is missing from the beef.", "vision"))
        self.assertEqual(fix, "re-garnish with scallion and send")
        self.assertTrue(explain.CACHE_FILE.exists())

        text2, fix2, _cos2, how2 = explain.describe_crop(make_crop(1), make_crop(9), "fb")
        self.assertEqual(how2, "vision (cached)")
        self.assertEqual(fix2, "re-garnish with scallion and send")
        self.assertEqual(fake.calls, 1)                         # cache hit, no 2nd call

    def test_different_crop_misses_cache(self):
        fake = FakeClient()
        explain._client = fake
        explain.describe_crop(make_crop(1), make_crop(9), "fb")
        explain.describe_crop(make_crop(2), make_crop(9), "fb")
        self.assertEqual(fake.calls, 2)

    def test_api_timeout_falls_back_with_exception_name(self):
        explain._client = FakeClient(exc=timeout_exc())
        text, fix, _cos, how = explain.describe_crop(
            make_crop(3), make_crop(9), "Anomalous region at (0.5, 0.5)")
        self.assertEqual(text, "Anomalous region at (0.5, 0.5)")
        self.assertIsNone(fix)
        self.assertEqual(how, "fallback (APITimeoutError)")

    def test_empty_response_falls_back_uncached(self):
        explain._client = FakeClient(text="")
        text, fix, _cos, how = explain.describe_crop(make_crop(4), make_crop(9), "fb")
        self.assertEqual(text, "fb")
        self.assertIn("empty response", how)
        self.assertEqual(explain._load_cache(), {})


class TestExplainFindings(ExplainTestBase):
    def frame_blob_finding(self):
        frame = make_crop(5, 400)
        blob = {"cx": 0.5, "cy": 0.5, "r": 0.02}
        finding = {"designator": None, "component": None, "description": "",
                   "fix": None, "score": 50.0, "source": "anomaly map"}
        return frame, [blob], [finding]

    def test_vision_fills_description_fix_and_source(self):
        explain._client = FakeClient()
        frame, blobs, findings = self.frame_blob_finding()
        out = explain.explain_findings(frame, frame.copy(), blobs, findings)
        self.assertEqual(out[0]["description"],
                         "Scallion garnish is missing from the beef.")
        self.assertEqual(out[0]["fix"], "re-garnish with scallion and send")
        self.assertEqual(out[0]["source"], "vision")

    def test_fallback_keeps_coord_text_and_source(self):
        explain._client = FakeClient(exc=timeout_exc())
        frame, blobs, findings = self.frame_blob_finding()
        out = explain.explain_findings(frame, frame.copy(), blobs, findings)
        self.assertEqual(out[0]["description"], "Anomalous region at (0.50, 0.50)")
        self.assertIsNone(out[0]["fix"])
        self.assertEqual(out[0]["source"], "anomaly map")       # unchanged on fallback


class TestScrub(unittest.TestCase):
    """The model describes a CROP of the normalized frame. Upload normalization
    rotates/warps the photo into the memory bank's geometry, so any plate
    position the model names is measured in the wrong frame. Positions come
    from serve.region_label instead; these get stripped."""

    def assertScrubbed(self, raw, expected):
        self.assertEqual(explain._parse_verdict(raw)[0], expected)

    def test_strips_wrong_corner_claim(self):
        self.assertScrubbed(
            "FAIL: The upper right corner of the plate is missing its garnish.",
            "The corner of the plate is missing its garnish.")

    def test_strips_trailing_position_phrase(self):
        self.assertScrubbed("FAIL: Missing broccoli at the top-left corner of the dish.",
                            "Missing broccoli.")
        self.assertScrubbed("FAIL: Sauce smear on the left side of the plate.",
                            "Sauce smear.")

    def test_image_phrasing_is_case_insensitive(self):
        self.assertScrubbed("FAIL: The broccoli is missing from the dish in Image 2.",
                            "The broccoli is missing from the dish.")
        self.assertScrubbed("COSMETIC: A steam reflection in image 2 not in image 1.",
                            "A steam reflection not on the reference.")

    def test_keeps_real_failure_wording(self):
        for text in ("Scallion garnish is missing from the beef.",
                     "The broccoli portion is short by half.",
                     "Sauce has run onto the rim of the bowl."):
            self.assertScrubbed(f"FAIL: {text}", text)

    def test_verdict_survives_scrubbing(self):
        _d, _f, cosmetic = explain._parse_verdict(
            "COSMETIC: Steam sheen in the upper right corner of the dish.")
        self.assertTrue(cosmetic)
        _d, _f, cosmetic = explain._parse_verdict(
            "FAIL: Missing garnish in the lower right region of the image.")
        self.assertFalse(cosmetic)

    def test_hedged_failure_still_demoted_to_cosmetic(self):
        _d, _f, cosmetic = explain._parse_verdict(
            "FAIL: Possibly a short rice portion on the left edge of the dish.")
        self.assertTrue(cosmetic)

    def test_hedged_contamination_is_never_demoted(self):
        """Food safety inverts the hedging rule: a doubtful foreign object is a
        held plate, not a cosmetic note."""
        desc, fix, cosmetic = explain._parse_verdict(
            "FAIL: Possible hair on the rice.")
        self.assertFalse(cosmetic)
        self.assertIn("hair", desc.lower())
        self.assertTrue(fix)                                    # safety default fix

    def test_fix_is_parsed_from_the_fail_line(self):
        desc, fix, cosmetic = explain._parse_verdict(
            "FAIL: Broccoli portion missing. | FIX: add the broccoli portion and send")
        self.assertEqual(desc, "Broccoli portion missing.")
        self.assertEqual(fix, "add the broccoli portion and send")
        self.assertFalse(cosmetic)

    def test_prompt_forbids_position_claims_and_version_bumped(self):
        """A prompt change with a stale PROMPT_VERSION would silently replay the
        old cached answers out of runs/explain_cache.json."""
        self.assertIn("NEVER name a plate position", explain.PROMPT)
        self.assertNotEqual(explain.PROMPT_VERSION, b"v3-crop-local")


if __name__ == "__main__":
    unittest.main()
