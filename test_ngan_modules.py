import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_loader import load_problem
from summary_statistics import summarize_results
from statistical_tests import holm_adjust, compare_algorithms, build_final_table


class TestNganModules(unittest.TestCase):
    def test_standard_datasets_dimensions(self):
        expected = {
            "small_10x3.json": (10, 3),
            "medium_30x5.json": (30, 5),
            "large_100x10.json": (100, 10),
        }
        for filename, (n, m) in expected.items():
            problem = load_problem(ROOT / "data" / filename)
            self.assertEqual(problem.num_jobs, n)
            self.assertEqual(problem.num_machines, m)
            self.assertEqual(len(problem.processing_times), m)
            self.assertEqual(len(problem.processing_times[0]), n)

    def test_summary_statistics(self):
        rows = [
            {"algorithm": "GA", "dataset": "small", "seed": 1, "makespan": 10,
             "runtime": 0.2, "evaluation_count": 10000},
            {"algorithm": "GA", "dataset": "small", "seed": 2, "makespan": 20,
             "runtime": 0.4, "evaluation_count": 10000},
        ]
        result = summarize_results(rows)[0]
        self.assertEqual(result["Best"], 10)
        self.assertEqual(result["Worst"], 20)
        self.assertEqual(result["Mean"], 15)
        self.assertAlmostEqual(result["Sample StdDev"], 2 ** 0.5 * 5)

    def test_holm(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)

    def test_statistical_comparison(self):
        rows = []
        for seed in range(1, 6):
            rows.append({"algorithm": "GA", "dataset": "small", "seed": seed,
                         "makespan": 100 + seed, "runtime": 0.1, "evaluation_count": 10000})
            rows.append({"algorithm": "WOA", "dataset": "small", "seed": seed,
                         "makespan": 110 + seed, "runtime": 0.2, "evaluation_count": 10000})
        result = compare_algorithms(rows)[0]
        self.assertEqual(result["paired_n"], 5)
        self.assertIn("wilcoxon_p_holm", result)

    @staticmethod
    def _row(alg, ds, seed, ms, nfe=10000):
        return {"algorithm": alg, "dataset": ds, "seed": seed, "makespan": ms,
                "runtime": 0.1, "evaluation_count": nfe}

    def test_identical_results_give_p_one(self):
        rows = [self._row(a, "small", s, 50.0) for a in ("GA", "WOA") for s in range(1, 11)]
        r = compare_algorithms(rows)[0]
        self.assertEqual(r["mann_whitney_u_p"], 1.0)
        self.assertEqual(r["wilcoxon_p"], 1.0)
        self.assertEqual(r["better"], "tie")

    def test_duplicate_seed_rejected(self):
        rows = [self._row("GA", "small", 1, 10), self._row("GA", "small", 1, 11)]
        with self.assertRaises(ValueError):
            summarize_results(rows)
        with self.assertRaises(ValueError):
            compare_algorithms(rows)

    def test_wrong_nfe_rejected(self):
        with self.assertRaises(ValueError):
            summarize_results([self._row("GA", "small", 1, 10, nfe=3)])

    def test_holm_per_test_family(self):
        rows = []
        for ds, shift in (("small", 0), ("medium", 8), ("large", 8)):
            for s in range(1, 31):
                rows.append(self._row("GA", ds, s, 100 + (s % 7)))
                rows.append(self._row("WOA", ds, s, 100 + shift + (s % 5) + 0.01 * s))
        tests = compare_algorithms(rows)
        for t in tests:   # m = 3 dataset -> p_holm <= 3 * p thô
            self.assertLessEqual(t["mann_whitney_u_p_holm"], min(1.0, 3 * t["mann_whitney_u_p"]) + 1e-12)
        table = build_final_table(summarize_results(rows), tests)
        self.assertEqual(len(table), 6)
        self.assertIn("wilcoxon_p_holm", table[0])


if __name__ == "__main__":
    unittest.main()
