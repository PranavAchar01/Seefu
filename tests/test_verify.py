"""Unit tests for inspection/verify.py (M6 verified resolutions) - stdlib unittest.

Run: .venv/bin/python -m unittest tests.test_verify -v

Pure unit tests: the OpenAI SDK is replaced with a fake module injected into
sys.modules (verify.py imports it lazily inside verify_case), so no network,
no key, no real client is ever touched. CASES is redirected to a temp dir so
runs/cases on disk is never read or written.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection import verify


def make_fake_openai(output_text=None, exc=None, calls=None, ctor_exc=None):
    """Build a fake `openai` module mirroring the exact call chain verify.py uses:

        OpenAI().with_options(timeout=..., max_retries=...).responses.create(**kw)
            -> object with .output_text

    - output_text: what resp.output_text returns
    - exc: raised from responses.create (simulates API failure)
    - calls: list that collects the kwargs of every create() call
    - ctor_exc: raised from OpenAI() itself (used to prove the client is
      never even constructed on early-return paths)
    """
    module = types.ModuleType("openai")

    class _Responses:
        def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            if exc is not None:
                raise exc
            return types.SimpleNamespace(output_text=output_text)

    class _Client:
        def __init__(self, *args, **kwargs):
            if ctor_exc is not None:
                raise ctor_exc
            self.responses = _Responses()

        def with_options(self, **kwargs):
            return self

    module.OpenAI = _Client
    return module


class VerifyCaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_cases = verify.CASES
        verify.CASES = Path(self._tmp.name) / "cases"
        verify.CASES.mkdir(parents=True)
        self.addCleanup(self._restore_cases)

        # Default env for most tests: key present (fake client, no network).
        self._env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-not-real"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _restore_cases(self):
        verify.CASES = self._orig_cases

    def _write_case(self, case_id="case-001", extra=None):
        data = {
            "dish_id": "BRD-42",
            "verdict": "defect",
            "status": "quarantined",
            "findings": [{"component": "C9", "defect": "tombstone",
                          "x_mm": 33.0, "y_mm": 112.0}],
            "overlay_path": "runs/overlays/case-001.png",
        }
        if extra:
            data.update(extra)
        (verify.CASES / f"{case_id}.json").write_text(json.dumps(data))
        return case_id, data

    def _run(self, fake_module, case_id):
        with mock.patch.dict(sys.modules, {"openai": fake_module}):
            return verify.verify_case(case_id)

    # ---- early-return paths (no API involved) --------------------------------

    def test_missing_case_record_skips_without_touching_api(self):
        fake = make_fake_openai(ctor_exc=AssertionError("client must not be built"))
        verified, note = self._run(fake, "no-such-case")
        self.assertFalse(verified)
        self.assertEqual(note, "verification skipped: case record missing")

    def test_no_api_key_returns_unavailable_without_touching_api(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(ctor_exc=AssertionError("client must not be built"))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):  # falsy -> treated as absent
            verified, note = self._run(fake, case_id)
        self.assertFalse(verified)
        self.assertEqual(note, "verification unavailable: no API key")

    # ---- happy paths: model verdict is passed through ------------------------

    def test_consistent_case_returns_true_with_note(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(
            output_text='{"verified": true, "note": "findings match blobs; verdict consistent"}')
        verified, note = self._run(fake, case_id)
        self.assertIs(verified, True)
        self.assertEqual(note, "findings match blobs; verdict consistent")

    def test_inconsistent_case_returns_false_with_note(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(
            output_text='{"verified": false, "note": "pass verdict but findings present"}')
        verified, note = self._run(fake, case_id)
        self.assertIs(verified, False)
        self.assertEqual(note, "pass verdict but findings present")

    def test_markdown_json_fence_is_tolerated(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(
            output_text='```json\n{"verified": true, "note": "ok"}\n```')
        verified, note = self._run(fake, case_id)
        self.assertTrue(verified)
        self.assertEqual(note, "ok")

    def test_plain_fence_without_json_label_is_tolerated(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(
            output_text='```\n{"verified": true, "note": "ok"}\n```')
        verified, note = self._run(fake, case_id)
        self.assertTrue(verified)
        self.assertEqual(note, "ok")

    def test_surrounding_whitespace_is_tolerated(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(
            output_text='  \n{"verified": true, "note": "ok"}\n  ')
        verified, note = self._run(fake, case_id)
        self.assertTrue(verified)

    def test_missing_note_defaults_to_empty_string(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(output_text='{"verified": true}')
        verified, note = self._run(fake, case_id)
        self.assertTrue(verified)
        self.assertEqual(note, "")

    def test_note_is_coerced_to_str_and_capped_at_200_chars(self):
        case_id, _ = self._write_case()
        long_note = "x" * 500
        fake = make_fake_openai(
            output_text=json.dumps({"verified": True, "note": long_note}))
        _, note = self._run(fake, case_id)
        self.assertEqual(note, "x" * 200)

        fake = make_fake_openai(output_text='{"verified": true, "note": 123}')
        _, note = self._run(fake, case_id)
        self.assertEqual(note, "123")

    def test_verified_requires_strict_true(self):
        # only JSON true or the string "true" verify; anything else - including
        # the string "false", numbers, or junk - stays unverified
        case_id, _ = self._write_case()
        for reply, expected in [('{"verified": true, "note": "n"}', True),
                                ('{"verified": "true", "note": "n"}', True),
                                ('{"verified": "false", "note": "n"}', False),
                                ('{"verified": 1, "note": "n"}', False),
                                ('{"verified": 0, "note": "n"}', False)]:
            fake = make_fake_openai(output_text=reply)
            verified, _ = self._run(fake, case_id)
            self.assertIs(verified, expected, f"reply={reply}")

    # ---- degradation: API/parse failures never raise -------------------------

    def test_api_exception_degrades_to_unverified_with_exception_name(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(exc=TimeoutError("deadline"))
        verified, note = self._run(fake, case_id)
        self.assertFalse(verified)
        self.assertEqual(note, "verification unavailable: TimeoutError")

    def test_client_constructor_failure_degrades_gracefully(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(ctor_exc=RuntimeError("no client"))
        verified, note = self._run(fake, case_id)
        self.assertFalse(verified)
        self.assertEqual(note, "verification unavailable: RuntimeError")

    def test_non_json_model_output_degrades_gracefully(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(output_text="Sure! The case looks consistent to me.")
        verified, note = self._run(fake, case_id)
        self.assertFalse(verified)
        self.assertEqual(note, "verification unavailable: JSONDecodeError")

    def test_json_but_not_an_object_degrades_gracefully(self):
        case_id, _ = self._write_case()
        fake = make_fake_openai(output_text="true")  # valid JSON, has no .get
        verified, note = self._run(fake, case_id)
        self.assertFalse(verified)
        self.assertEqual(note, "verification unavailable: AttributeError")

    # ---- prompt payload contract ---------------------------------------------

    def test_prompt_contains_whitelisted_fields_and_leaks_nothing_else(self):
        calls = []
        case_id, _ = self._write_case(
            extra={"operator_email": "secret@example.com", "raw_frame_b64": "AAAA"})
        fake = make_fake_openai(
            output_text='{"verified": true, "note": "ok"}', calls=calls)
        self._run(fake, case_id)

        self.assertEqual(len(calls), 1)
        kwargs = calls[0]
        sent = kwargs["input"][0]["content"][0]["text"]
        # only the 5 whitelisted record keys go to the model
        for key in ("dish_id", "verdict", "status", "findings", "overlay_path"):
            self.assertIn(f'"{key}"', sent)
        self.assertIn("BRD-42", sent)
        self.assertNotIn("operator_email", sent)
        self.assertNotIn("secret@example.com", sent)
        self.assertNotIn("raw_frame_b64", sent)
        self.assertIn("STRICT JSON", sent)  # the instruction block is included

    def test_absent_record_keys_are_sent_as_null_not_omitted(self):
        calls = []
        case_id = "sparse-case"
        (verify.CASES / f"{case_id}.json").write_text(json.dumps({"dish_id": "B1"}))
        fake = make_fake_openai(
            output_text='{"verified": false, "note": "incomplete"}', calls=calls)
        verified, note = self._run(fake, case_id)
        self.assertFalse(verified)
        record = json.loads(calls[0]["input"][0]["content"][0]["text"]
                            .split("CASE RECORD:\n", 1)[1])
        self.assertEqual(record["dish_id"], "B1")
        for key in ("verdict", "status", "findings", "overlay_path"):
            self.assertIsNone(record[key])

    # ---- known bug: docstring says "Never raises", corrupt file raises -------

    def test_corrupt_case_file_degrades_gracefully(self):
        # "never raises" holds even for a corrupt/truncated case record
        case_id = "corrupt"
        (verify.CASES / f"{case_id}.json").write_text("{not json")
        fake = make_fake_openai(ctor_exc=AssertionError("client must not be built"))
        with mock.patch.dict(sys.modules, {"openai": fake}):
            verified, note = verify.verify_case(case_id)
        self.assertIs(verified, False)
        self.assertIn("unreadable case record", note)


if __name__ == "__main__":
    unittest.main()
