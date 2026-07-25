"""Unit tests for core/phone.py alert-call logic - no network, httpx stubbed."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import phone


RESULT = {"dish_id": "DISH-005", "case_id": "1042", "verdict": "defect",
          "score": 51.2, "threshold": 44.4,
          "findings": [{"designator": "J1", "component": "4-pin header",
                        "location": "upper left",
                        "description": "Pins 2 and 3 bent & contacting", "score": 51.2,
                        "source": "component map + vision"}]}


class TestAlertText(unittest.TestCase):
    def test_names_plate_and_designator_and_description(self):
        text = phone.defect_alert_text(RESULT)
        for needle in ("DISH-005", "J1", "Pins 2 and 3", "held at the pass"):
            self.assertIn(needle, text)

    def test_speaks_the_verdict(self):
        self.assertIn("failed the check", phone.defect_alert_text(RESULT))

    def test_speaks_score_against_threshold(self):
        text = phone.defect_alert_text(RESULT)
        self.assertIn("51.2", text)
        self.assertIn("44.4", text)
        self.assertIn("15 percent above", text)     # (51.2-44.4)/44.4

    def test_missing_score_just_omits_that_sentence(self):
        text = phone.defect_alert_text(dict(RESULT, score=None, threshold=None))
        self.assertNotIn("anomaly score is", text)
        self.assertIn("J1", text)

    def test_speaks_the_location_and_component(self):
        text = phone.defect_alert_text(RESULT)
        self.assertIn("in the upper left", text)
        self.assertIn("at J1, the 4-pin header", text)

    def test_unmapped_finding_speaks_the_location_not_unmapped(self):
        # an operator cannot walk to "unmapped"; they can walk to "lower right"
        r = dict(RESULT, findings=[{"designator": None, "location": "lower right",
                                    "description": "foreign object"}])
        text = phone.defect_alert_text(r)
        self.assertIn("in the lower right", text)
        self.assertIn("foreign object", text.lower())
        self.assertNotIn("unmapped", text)

    def test_edge_and_centre_locations_read_as_phrases(self):
        self.assertEqual(phone.spoken_place("left edge"), "on the left edge")
        self.assertEqual(phone.spoken_place("centre of dish"),
                         "in the centre of the dish")
        self.assertEqual(phone.spoken_place("upper right"), "in the upper right")
        self.assertEqual(phone.spoken_place(None), "")

    def test_no_coordinates_or_markdown_reach_the_speech(self):
        r = dict(RESULT, findings=[
            {"designator": None, "location": "upper centre",
             "description": "Anomalous region at (0.42, 0.31)"},
            {"designator": "R7", "location": "lower left",
             "description": "**solder bridge** between pads"}])
        text = phone.defect_alert_text(r)
        for banned in ("(0.42", "0.31)", "*", "{", "}", "["):
            self.assertNotIn(banned, text)
        self.assertIn("solder bridge between pads", text.lower())

    def test_speakable_strips_coordinates_and_trailing_preposition(self):
        self.assertEqual(phone.speakable("Anomalous region at (0.42, 0.31)"),
                         "an anomalous region")
        self.assertEqual(phone.speakable("bent pin & lifted pad"),
                         "bent pin and lifted pad")

    def test_no_findings_still_reads_sanely(self):
        r = dict(RESULT, findings=[])
        text = phone.defect_alert_text(r)
        self.assertIn("exceeded the threshold", text)

    def test_caps_at_three_findings(self):
        f = [{"designator": f"R{i}", "description": "d"} for i in range(6)]
        text = phone.defect_alert_text(dict(RESULT, findings=f))
        self.assertIn("R2", text)
        self.assertNotIn("R3,", text.replace("R3:", "R3,"))
        self.assertNotIn("R5", text)
        self.assertIn("Six problem regions were found", text)
        self.assertIn("first three", text)

    def test_reads_as_sentences_not_a_log_line(self):
        text = phone.defect_alert_text(RESULT)
        self.assertNotIn("\n", text)
        self.assertNotIn("designator", text.lower())
        self.assertGreater(text.count(". "), 4)       # breathing room


class PlaceAlertBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._tunnel = phone.TUNNEL_FILE
        self._root = phone.ROOT
        phone.ROOT = Path(self.tmp.name)
        phone.TUNNEL_FILE = Path(self.tmp.name) / ".tunnel_host"
        phone._last_alert_ts = 0.0
        phone._last_tts.update(voice="twilio-fallback", reason="not attempted",
                               model=None, voice_id=None, bytes=0, cached=False)
        self.emitted = []
        self._emit = phone.hub.emit
        phone.hub.emit = self.emitted.append
        self.requests = []
        self.eleven_status = 200
        self.eleven_body = b"ID3fake-elevenlabs-mp3"
        fake_httpx = mock.MagicMock()

        def post(url, data=None, auth=None, timeout=None, **kw):
            self.requests.append({"url": url, "data": data, "auth": auth, **kw})
            resp = mock.MagicMock()
            resp.raise_for_status = lambda: None
            if "elevenlabs" in url:
                resp.status_code = self.eleven_status
                resp.content = self.eleven_body
                resp.text = "" if self.eleven_status == 200 else "boom"
                resp.json = lambda: {"detail": {"message": "boom"}}
            else:
                resp.status_code = 201
                resp.json = lambda: {"sid": "CAtest"}
            return resp

        fake_httpx.post = post
        self.env = {"TWILIO_ACCOUNT_SID": "ACx", "TWILIO_AUTH_TOKEN": "tok",
                    "TWILIO_FROM": "+1000", "TWILIO_TO": "+1999",
                    "SEEFU_CALL_COOLDOWN": "60", "ELEVENLABS_API_KEY": ""}
        self.httpx_patch = mock.patch.dict(sys.modules, {"httpx": fake_httpx})
        self.httpx_patch.start()

    def tearDown(self):
        self.httpx_patch.stop()
        phone.TUNNEL_FILE = self._tunnel
        phone.ROOT = self._root
        phone.hub.emit = self._emit
        phone._last_alert_ts = 0.0
        self.tmp.cleanup()

    def place(self, extra_env=None):
        env = dict(self.env, **(extra_env or {}))
        with mock.patch.dict("os.environ", env, clear=False):
            return phone.place_defect_alert(RESULT)

    def eleven_request(self):
        return next(r for r in self.requests if "elevenlabs" in r["url"])

    def phone_events(self):
        return [m for m in self.emitted if m.get("type") == "phone"]


class TestPlaceDefectAlert(PlaceAlertBase):
    def test_no_tunnel_raises_clear_error(self):
        # trial accounts must fetch TwiML from a public URL - no tunnel, no call
        with self.assertRaises(RuntimeError):
            self.place()

    def test_tunnel_places_call_via_twiml_url_with_case_id(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        sid = self.place()
        self.assertEqual(sid, "CAtest")
        data = next(r for r in self.requests if "twilio" in r["url"])["data"]
        self.assertNotIn("Twiml", data)          # trial accounts reject inline TwiML
        self.assertEqual(data["Url"],
                         "https://demo.trycloudflare.com/phone/twiml?case_id=1042")

    def test_cooldown_skips_second_call(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        self.place()
        sid2 = self.place()
        self.assertIsNone(sid2)
        self.assertEqual(len([r for r in self.requests if "twilio" in r["url"]]), 1)
        self.assertTrue(any("cooldown" in m.get("message", "") for m in self.emitted))

    def test_disabled_via_env(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        sid = self.place({"SEEFU_CALL_ON_DEFECT": "0"})
        self.assertIsNone(sid)
        self.assertEqual(self.requests, [])

    def test_missing_creds_raises(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        with mock.patch.dict("os.environ",
                             {k: "" for k in self.env if k.startswith("TWILIO")},
                             clear=False):
            with self.assertRaises(RuntimeError):
                phone.place_defect_alert(RESULT)

    def test_placed_event_says_elevenlabs_when_synthesis_worked(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        self.place({"ELEVENLABS_API_KEY": "elk"})
        placed = self.phone_events()[-1]
        self.assertEqual(placed["status"], "placed")
        self.assertEqual(placed["voice"], "elevenlabs")
        self.assertEqual(placed["model"], phone.ELEVEN_MODEL)
        self.assertEqual(placed["bytes"], len(self.eleven_body))
        self.assertIn("ElevenLabs", placed["message"])   # the dashboard prints this
        self.assertNotIn("reason", placed)

    def test_placed_event_says_fallback_with_a_reason_when_key_missing(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        self.place()                                     # ELEVENLABS_API_KEY = ""
        placed = self.phone_events()[-1]
        self.assertEqual(placed["voice"], "twilio-fallback")
        self.assertIn("ELEVENLABS_API_KEY", placed["reason"])
        self.assertIn("Twilio fallback", placed["message"])

    def test_placed_event_carries_the_api_failure_reason(self):
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        self.eleven_status = 401
        self.place({"ELEVENLABS_API_KEY": "bad"})
        placed = self.phone_events()[-1]
        self.assertEqual(placed["voice"], "twilio-fallback")
        self.assertIn("401", placed["reason"])
        # ...and the call still went out
        self.assertTrue(any("twilio" in r["url"] for r in self.requests))

    def test_stashed_text_makes_the_twiml_leg_a_cache_hit(self):
        # /phone/twiml only knows the case_id, and a filed case carries no score;
        # the stash is what keeps both legs on the exact same words (and so on
        # the same mp3) instead of re-synthesizing while the operator waits
        phone.TUNNEL_FILE.write_text("demo.trycloudflare.com\n")
        self.place({"ELEVENLABS_API_KEY": "elk"})
        spoken = phone.defect_alert_text(RESULT)
        self.assertEqual(phone.case_alert_text("1042"), spoken)
        before = len(self.requests)
        with mock.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "elk"}, clear=False):
            again = phone.synthesize_alert_audio(spoken, "1042")
        self.assertIsNotNone(again)
        self.assertEqual(len(self.requests), before)     # served from disk
        self.assertTrue(phone.tts_status()["cached"])


class TestElevenLabsSynthesis(PlaceAlertBase):
    def synth(self, text="hello", case_id="1042", key="elk"):
        with mock.patch.dict("os.environ", {"ELEVENLABS_API_KEY": key}, clear=False):
            return phone.synthesize_alert_audio(text, case_id)

    def test_no_key_returns_none_and_records_a_loud_reason(self):
        self.assertIsNone(self.synth(key=""))
        status = phone.tts_status()
        self.assertEqual(status["voice"], "twilio-fallback")
        self.assertIn("ELEVENLABS_API_KEY", status["reason"])
        self.assertTrue(any(m.get("source") == "voice.elevenlabs"
                            for m in self.emitted))

    def test_stubbed_success_writes_and_caches(self):
        out = self.synth()
        self.assertIsNotNone(out)
        self.assertEqual(out.read_bytes(), self.eleven_body)
        again = self.synth()
        self.assertEqual(out, again)
        self.assertEqual(len([r for r in self.requests if "elevenlabs" in r["url"]]), 1)
        self.assertEqual(phone.tts_status()["voice"], "elevenlabs")

    def test_request_matches_the_documented_rest_shape(self):
        self.synth()
        req = self.eleven_request()
        self.assertTrue(req["url"].endswith(
            f"/v1/text-to-speech/{phone.ELEVEN_DEFAULT_VOICE}"))
        # output_format is a QUERY parameter on this endpoint, not a body field
        self.assertEqual(req["params"], {"output_format": phone.ELEVEN_OUTPUT_FORMAT})
        self.assertEqual(req["headers"]["xi-api-key"], "elk")
        body = req["json"]
        self.assertEqual(body["model_id"], phone.ELEVEN_MODEL)
        self.assertNotIn("output_format", body)
        for field in ("stability", "similarity_boost", "style",
                      "use_speaker_boost", "speed"):
            self.assertIn(field, body["voice_settings"])
        self.assertLessEqual(body["voice_settings"]["stability"], 1.0)

    def test_custom_voice_id_is_honoured(self):
        with mock.patch.dict("os.environ",
                             {"ELEVENLABS_API_KEY": "elk",
                              "ELEVENLABS_VOICE_ID": "customVoice1"}, clear=False):
            phone.synthesize_alert_audio("hello", "1042")
        self.assertTrue(self.eleven_request()["url"].endswith("customVoice1"))
        self.assertEqual(phone.tts_status()["voice_id"], "customVoice1")

    def test_http_error_records_the_status_code_and_detail(self):
        self.eleven_status = 429
        self.assertIsNone(self.synth())
        status = phone.tts_status()
        self.assertEqual(status["voice"], "twilio-fallback")
        self.assertIn("429", status["reason"])
        self.assertIn("boom", status["reason"])

    def test_empty_body_is_not_written_as_audio(self):
        self.eleven_body = b""
        self.assertIsNone(self.synth())
        self.assertIn("empty body", phone.tts_status()["reason"])

    def test_transport_exception_is_reported_by_type(self):
        import sys as _sys
        fake = mock.MagicMock()

        def boom(*a, **k):
            raise TimeoutError("read timeout")

        fake.post = boom
        with mock.patch.dict(_sys.modules, {"httpx": fake}):
            self.assertIsNone(self.synth())
        self.assertIn("TimeoutError", phone.tts_status()["reason"])

    def test_empty_text_is_refused(self):
        self.assertIsNone(self.synth(text="   "))
        self.assertIn("empty", phone.tts_status()["reason"])

    def test_case_id_cannot_escape_the_alerts_folder(self):
        out = self.synth(case_id="../../.env")
        self.assertIsNotNone(out)
        self.assertEqual(out.parent, Path(self.tmp.name) / "runs/alerts")

    def test_audio_file_lookup_sanitized(self):
        alerts = Path(self.tmp.name) / "runs/alerts"
        alerts.mkdir(parents=True, exist_ok=True)
        (alerts / "1042.mp3").write_bytes(b"x")
        self.assertIsNotNone(phone.alert_audio_file("1042.mp3"))
        self.assertIsNone(phone.alert_audio_file("../../.env"))
        self.assertIsNone(phone.alert_audio_file("1042.png"))
        self.assertIsNone(phone.alert_audio_file("nope.mp3"))


class TestCaseAlertText(PlaceAlertBase):
    def test_rebuilds_from_a_filed_case_when_nothing_is_stashed(self):
        cases = Path(self.tmp.name) / "runs/cases"
        cases.mkdir(parents=True, exist_ok=True)
        (cases / "77.json").write_text(json.dumps({
            "case_id": "77", "dish_id": "DISH-009", "verdict": "defect",
            "findings": [{"designator": None, "location": "lower right",
                          "description": "solder splash across two pads"}]}))
        text = phone.case_alert_text("77")
        self.assertIn("DISH-009", text)
        self.assertIn("in the lower right", text)
        self.assertIn("solder splash", text.lower())
        self.assertNotIn("unmapped", text)

    def test_unknown_case_falls_back_to_a_test_call_line(self):
        self.assertIn("Test call successful", phone.case_alert_text(""))

    def test_stale_stash_from_an_earlier_rehearsal_is_ignored(self):
        # reset_demo.py restarts case ids at 1041; a stash naming a different
        # dish must never be spoken over the new one
        cases = Path(self.tmp.name) / "runs/cases"
        cases.mkdir(parents=True, exist_ok=True)
        (cases / "1042.json").write_text(json.dumps({
            "case_id": "1042", "dish_id": "DISH-011", "verdict": "defect",
            "findings": [{"designator": "U3", "location": "upper right",
                          "description": "cracked package"}]}))
        phone.stash_alert_text("1042", "Dish DISH-005 has failed inspection.")
        text = phone.case_alert_text("1042")
        self.assertIn("DISH-011", text)
        self.assertNotIn("DISH-005", text)
        self.assertIn("cracked package", text.lower())

    def test_matching_stash_is_still_preferred(self):
        cases = Path(self.tmp.name) / "runs/cases"
        cases.mkdir(parents=True, exist_ok=True)
        (cases / "1042.json").write_text(json.dumps({
            "case_id": "1042", "dish_id": "DISH-005", "verdict": "defect",
            "findings": []}))
        phone.stash_alert_text("1042", "Dish DISH-005, score 51.2, verbatim.")
        self.assertEqual(phone.case_alert_text("1042"),
                         "Dish DISH-005, score 51.2, verbatim.")


if __name__ == "__main__":
    unittest.main()
