import http.client
import json
import math
import random
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path

import watermark_lab as lab
from server import Handler


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.profile = lab.demo_profile()
        self.tokenizer = self.profile["tokenizer_id"]

    def test_signed_overflow(self):
        self.assertEqual(lab.signed64(1 << 63), -(1 << 63))
        self.assertEqual(lab.signed64((1 << 64) - 1), -1)
        self.assertEqual(lab.signed64(-(1 << 63) - 1), (1 << 63) - 1)

    def test_upstream_arithmetic_parity(self):
        from validate import check_reference
        result = check_reference()
        self.assertEqual(result["binary_scores_compared"], 2072)
        self.assertTrue(result["all_exact"])

    def test_empty_and_short_are_not_negative_detections(self):
        for tokens in ([], [1, 2, 3], list(range(40))):
            result = lab.inspect_tokens(tokens, self.profile, self.tokenizer)
            self.assertEqual(result["status"], "insufficient_contexts")
            self.assertEqual(result["claude_status"], "unavailable")
            self.assertIsNone(result["empirical_tail_rank"])

    def test_repetitions_do_not_inflate_evidence(self):
        result = lab.inspect_tokens([7] * 128, self.profile, self.tokenizer)
        self.assertEqual(result["scored_contexts"], 1)
        self.assertEqual(result["skipped_repeated_contexts"], 123)

    def test_rolling_context_history_expires(self):
        p = dict(self.profile, ngram_len=2, context_history_size=1)
        result = lab.inspect_tokens([8, 9, 8, 9, 9, 9], p, self.tokenizer)
        self.assertEqual(result["scored_contexts"], 4)
        self.assertEqual(result["skipped_repeated_contexts"], 1)

    def test_eos_and_everything_after_are_excluded(self):
        result = lab.inspect_tokens(list(range(128)), self.profile, self.tokenizer, eos_token_id=20)
        self.assertEqual(result["tokens_before_eos"], 20)
        self.assertEqual(result["scored_contexts"], 16)
        self.assertEqual(result["token_trace"][-1]["position"], 19)

    def test_tokenizer_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Tokenizer mismatch"):
            lab.inspect_tokens([1] * 128, self.profile, "claude-guessed")

    def test_no_guessed_vendor_algorithm(self):
        with self.assertRaises(ValueError):
            lab.inspect_tokens([1] * 128, dict(self.profile, algorithm="claude"), self.tokenizer)

    def test_invalid_tokens_rejected(self):
        for tokens in ([True], [-1], [1.5], [1 << 63], "text", list(range(4097))):
            with self.subTest(tokens=str(tokens)[:30]), self.assertRaises(ValueError):
                lab.inspect_tokens(tokens, self.profile, self.tokenizer)

    def test_invalid_profiles_rejected(self):
        for p in (None, {}, dict(self.profile, keys=[1, 1]), dict(self.profile, keys=[False]),
                  dict(self.profile, ngram_len=1), dict(self.profile, context_history_size=0)):
            with self.subTest(p=p), self.assertRaises(ValueError):
                lab.validate_profile(p)

    def test_no_calibration_no_verdict(self):
        result = lab.inspect_tokens(list(range(128)), self.profile, self.tokenizer)
        self.assertEqual(result["status"], "uncalibrated")

    def test_calibration_requires_profile_and_length_match(self):
        cal = json.loads((lab.ROOT / "evidence/demo-calibration.json").read_text())
        cal["profile_fingerprint"] = "wrong"
        result = lab.inspect_tokens(list(range(128)), self.profile, self.tokenizer, calibration=cal)
        self.assertEqual(result["status"], "uncalibrated")
        self.assertIsNone(result["empirical_tail_rank"])

    def test_invalid_calibration_rejected(self):
        for scores in ([0.5] * 98, [float("nan")] * 99, [2.0] * 99, [True] * 99):
            with self.assertRaises(ValueError):
                lab.inspect_tokens(list(range(128)), self.profile, self.tokenizer,
                                   calibration={"null_scores": scores})

    def test_empirical_rank_uses_plus_one_and_conservative_ties(self):
        tokens = list(range(128))
        score = lab.inspect_tokens(tokens, self.profile, self.tokenizer)["mean_g_score"]
        cal = {"profile_fingerprint": lab.fingerprint(self.profile), "tokenizer_id": self.tokenizer,
               "input_tokens": 128, "scored_contexts": 124, "null_scores": [score] * 99}
        result = lab.inspect_tokens(tokens, self.profile, self.tokenizer, calibration=cal)
        self.assertEqual(result["empirical_tail_rank"], 1.0)
        cal["null_scores"] = [0] * 99
        result = lab.inspect_tokens(tokens, self.profile, self.tokenizer, calibration=cal)
        self.assertEqual(result["empirical_tail_rank"], 0.01)

    def test_saved_demos_are_rescored_and_still_not_claude_verdicts(self):
        expected = {"marked_correct_key": "reference_signal_detected",
                    "unmarked": "reference_signal_not_detected",
                    "marked_other_key": "reference_signal_not_detected"}
        for name, status in expected.items():
            data = json.loads((lab.ROOT / ("evidence/demo-" + name + ".json")).read_text())
            result = lab.inspect_payload(data["payload"])
            self.assertEqual(result["status"], status)
            self.assertEqual(result["claude_status"], "unavailable")
            self.assertEqual(result["mean_g_score"], data["result"]["mean_g_score"])

    def test_tournament_probabilities_normalized_nonnegative(self):
        rng = random.Random(71)
        for _ in range(20):
            weights = [rng.random() for _ in range(16)]
            p = [w / sum(weights) for w in weights]
            bits = [[rng.randrange(2) for _ in range(8)] for _ in p]
            out = lab.tournament_probabilities(p, bits)
            self.assertAlmostEqual(sum(out), 1, places=12)
            self.assertTrue(all(x >= 0 for x in out))
        self.assertEqual(lab.tournament_probabilities([1, 0], [[0, 1], [1, 0]]), [1, 0])

    def test_plain_unicode_and_stereotypical_prose_never_imply_vendor(self):
        for text in ("", "This is not X. It is Y. Delve into the landscape.", "Привет!", "a\u200bb", "👩‍💻"):
            result = lab.analyze_text(text)
            self.assertEqual(result["status"], "vendor_verification_unavailable")
            self.assertFalse(result["modified"])
            self.assertFalse(result["external_requests"])
            self.assertNotIn("probability", result)
        self.assertEqual(sum(lab.analyze_text("a\u200bb")["unicode_findings"].values()), 1)

    def test_report_trace_is_bounded(self):
        result = lab.inspect_tokens(list(range(512)), self.profile, self.tokenizer)
        self.assertEqual(len(result["token_trace"]), 256)
        self.assertTrue(result["trace_truncated"])


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        connection.request("GET" if payload is None else "POST", path,
                           body=None if payload is None else json.dumps(payload), headers=h)
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def test_text_api_unknown_not_fake_negative(self):
        status, body, headers = self.request("/api/analyze", {"text": "Human or AI?"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["claude_status"], "unavailable")
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_cross_origin_is_rejected(self):
        self.assertEqual(self.request("/api/analyze", {"text": "private"}, {"Origin": "https://example.com"})[0], 403)

    def test_nonlocal_host_is_rejected(self):
        self.assertEqual(self.request("/api/status", headers={"Host": "example.com"})[0], 403)

    def test_path_traversal_is_rejected(self):
        self.assertEqual(self.request("/../PROJECT_STATE.md")[0], 404)

    def test_schema_failure_returns_400(self):
        self.assertEqual(self.request("/api/analyze", {"text": 42})[0], 400)
        self.assertEqual(self.request("/api/tokens", {"token_ids": []})[0], 400)

    def test_unknown_demo_does_not_read_arbitrary_files(self):
        self.assertEqual(self.request("/api/demo", {"kind": "../../PROJECT_STATE.md"})[0], 400)
        self.assertEqual(self.request("/api/demo", {"kind": []})[0], 400)
        self.assertEqual(self.request("/api/demo", {"kind": {}})[0], 400)

    def test_home_has_csp_and_russian_ui(self):
        status, body, headers = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("Метка в словах".encode(), body)
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])


if __name__ == "__main__":
    unittest.main()
