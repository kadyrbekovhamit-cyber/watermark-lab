import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preference_probe as probe


class PreferenceProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marked = probe.synthetic_corpus("reference_watermark", contexts=16)
        cls.null = probe.synthetic_corpus("no_watermark", contexts=16)

    def test_learner_and_evaluator_do_not_call_generator_or_receive_keys(self):
        train, test = probe.validate_corpus(self.marked)
        with patch("watermark_lab.ngram_g_values", side_effect=AssertionError("Generator access")):
            learned = probe.learn_preferences(train, 4)
            result = probe.evaluate(learned, test, permutations=999)
        self.assertGreater(result["balanced_accuracy_on_covered"], .65)
        self.assertLessEqual(result["paired_label_swap_p"], .01)

    def test_ordinary_context_effect_is_also_detectable_not_specific_to_watermark(self):
        corpus = probe.synthetic_corpus("ordinary_context_effect", contexts=16)
        result = probe.analyze(corpus, permutations=999)
        self.assertGreater(result["evaluation"]["mean_paired_log_ratio_difference"], .1)
        self.assertFalse(result["key_recovered"])
        self.assertEqual(result["claude_status"], "unavailable")

    def test_null_balanced_accuracy_is_near_chance(self):
        r = probe.analyze(self.null, permutations=999)
        self.assertLess(abs(r["evaluation"]["balanced_accuracy_on_covered"] - .5), .04)

    def test_no_coverage_means_abstention(self):
        train, test = probe.validate_corpus(self.marked)
        learned = probe.learn_preferences(train, 4)
        changed = {(c + "-unseen", p): v for (c, p), v in test.items()}
        result = probe.evaluate(learned, changed, permutations=99)
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["paired_label_swap_p"])
        self.assertIsNone(result["balanced_accuracy_on_covered"])

    def test_identical_group_counts_are_null_and_ties_are_chance(self):
        corpus = copy.deepcopy(self.marked)
        for split in ("training", "validation"):
            for row in corpus[split]:
                row["counts"] = [32, 32, 32, 32]
        result = probe.analyze(corpus, permutations=99)["evaluation"]
        self.assertEqual(result["paired_label_swap_p"], 1)
        self.assertEqual(result["mean_paired_log_ratio_difference"], 0)
        self.assertEqual(result["balanced_accuracy_on_covered"], .5)

    def test_training_validation_prefix_leakage_rejected(self):
        bad = copy.deepcopy(self.marked)
        for row in bad["validation"]:
            row["prefix"] = row["prefix"].replace("2", "0").replace("3", "1")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            probe.analyze(bad)

    def test_bad_counts_or_missing_controls_rejected(self):
        for mutate in (lambda x: x["training"].pop(),
                       lambda x: x["training"][0]["counts"].__setitem__(0, -1),
                       lambda x: x["training"][0]["counts"].__setitem__(0, True),
                       lambda x: x["training"].append(x["training"][0])):
            bad = copy.deepcopy(self.marked)
            mutate(bad)
            with self.assertRaises(ValueError):
                probe.analyze(bad)

    def test_input_unchanged_and_no_raw_identifiers_in_report(self):
        original = copy.deepcopy(self.marked)
        result = probe.analyze(self.marked, permutations=99)
        self.assertEqual(original, self.marked)
        encoded = json.dumps(result)
        self.assertNotIn("context-000", encoded)
        self.assertNotIn('"keys"', encoded)
        self.assertFalse(result["external_requests"])

    def test_reproducible_generation_and_inference(self):
        again = probe.synthetic_corpus("reference_watermark", contexts=16)
        self.assertEqual(again, self.marked)
        self.assertEqual(probe.analyze(again, 99), probe.analyze(self.marked, 99))

    def test_cli_does_not_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "corpus.json"
            p.write_text(json.dumps(self.marked))
            before = p.read_bytes()
            result = subprocess.run([sys.executable, "-B", str(Path(probe.__file__)), "analyze", str(p), "--output", str(p)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(p.read_bytes(), before)

    def test_cli_rejects_colliding_benchmark_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "output.json"
            result = subprocess.run([sys.executable, "-B", str(Path(probe.__file__)),
                                     "benchmark", "--output", str(p), "--fixture", str(p)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()
