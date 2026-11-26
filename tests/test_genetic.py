"""
tests/test_genetic.py
---------------------
Bộ kiểm thử tự động (Unit Test) cho Giải thuật Di truyền (GA):
- Kiểm tra tính hợp lệ của tham số khởi tạo.
- Kiểm tra các toán tử: Chọn lọc giải đấu, Lai ghép Uniform/Two-point, Đột biến Resetting/Swap.
- Kiểm tra tính toàn vẹn của chu trình tiến hóa: Đúng 10,000 NFE, đúng 200 điểm hội tụ,
  tính đơn điệu Best-so-far, lịch trình Gantt chuẩn, callback hoạt động.
- Kiểm tra tính tái lập nghiệm khi dùng cùng seed ngẫu nhiên.
"""

import unittest
from core.config import (
    CONVERGENCE_POINTS,
    MAX_EVALUATIONS,
    MAX_GENERATIONS,
    POPULATION_SIZE,
)
from core.problem import UPMSProblem, validate_result_contract
from core.evaluator import Evaluator
from algorithms.genetic import GeneticAlgorithm, run_ga


class TestGeneticAlgorithmComponents(unittest.TestCase):
    """Kiểm tra từng toán tử độc lập của GeneticAlgorithm."""

    def setUp(self):
        # Bài toán nhỏ: 3 máy, 6 jobs
        self.matrix = [
            [10.0, 20.0, 30.0, 40.0, 15.0, 25.0],
            [15.0, 25.0, 35.0, 45.0, 20.0, 10.0],
            [20.0, 10.0, 20.0, 30.0, 25.0, 35.0],
        ]
        self.problem = UPMSProblem(num_jobs=6, num_machines=3, processing_times=self.matrix)
        self.ga = GeneticAlgorithm(self.problem, seed=42)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, population_size=1)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, max_generations=0)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, crossover_prob=1.5)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, mutation_prob=-0.1)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, tournament_k=100)  # k > pop_size
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, elite_count=50)   # elite >= pop_size
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, crossover_type="invalid_cx")
        with self.assertRaises(ValueError):
            GeneticAlgorithm(self.problem, mutation_type="invalid_mut")
        with self.assertRaises(TypeError):
            GeneticAlgorithm(self.problem, seed=True)

    def test_create_random_chromosome(self):
        chrom = self.ga._create_random_chromosome()
        self.assertEqual(len(chrom), self.problem.num_jobs)
        for m in chrom:
            self.assertIsInstance(m, int)
            self.assertGreaterEqual(m, 0)
            self.assertLess(m, self.problem.num_machines)

    def test_tournament_selection(self):
        population = [
            ([0, 0, 0, 0, 0, 0], 100.0),
            ([1, 1, 1, 1, 1, 1], 50.0),
            ([2, 2, 2, 2, 2, 2], 120.0),
        ]
        # k = 3 -> chọn lọc chắc chắn phải ra cá thể có makespan nhỏ nhất (50.0)
        ga_k3 = GeneticAlgorithm(self.problem, population_size=3, tournament_k=3, seed=1)
        winner = ga_k3._tournament_selection(population)
        self.assertEqual(winner, [1, 1, 1, 1, 1, 1])

    def test_crossover_uniform(self):
        p1 = [0, 0, 0, 0, 0, 0]
        p2 = [1, 1, 1, 1, 1, 1]
        c1, c2 = self.ga._crossover_uniform(p1, p2)
        self.assertEqual(len(c1), 6)
        self.assertEqual(len(c2), 6)
        # Mỗi gen chỉ có thể là 0 hoặc 1
        for g1, g2 in zip(c1, c2):
            self.assertIn(g1, [0, 1])
            self.assertIn(g2, [0, 1])

    def test_crossover_two_point(self):
        p1 = [0, 0, 0, 0, 0, 0]
        p2 = [1, 1, 1, 1, 1, 1]
        c1, c2 = self.ga._crossover_two_point(p1, p2)
        self.assertEqual(len(c1), 6)
        self.assertEqual(len(c2), 6)
        for g1, g2 in zip(c1, c2):
            self.assertIn(g1, [0, 1])
            self.assertIn(g2, [0, 1])

    def test_mutate_resetting(self):
        # Buộc mutation_prob = 1.0 để luôn xảy ra đột biến
        ga_mut = GeneticAlgorithm(self.problem, mutation_prob=1.0, seed=123)
        original = [0, 0, 0, 0, 0, 0]
        mutated = ga_mut._mutate_resetting(original)
        self.assertEqual(len(mutated), 6)
        # Phải có ít nhất 1 gen bị đổi sang máy khác 0
        diff_count = sum(1 for o, m in zip(original, mutated) if o != m)
        self.assertGreaterEqual(diff_count, 1)


class TestGeneticAlgorithmExecution(unittest.TestCase):
    """Kiểm tra toàn bộ chu trình thực thi GA."""

    def setUp(self):
        # Bài toán 3 máy, 10 jobs (Small)
        self.problem = UPMSProblem.generate_random_instance(
            num_jobs=10, num_machines=3, seed=100, p_min=10, p_max=100
        )

    def test_ga_solve_budget_and_contract(self):
        """Kiểm tra GA tiêu thụ chính xác 10,000 NFE và thỏa mãn hợp đồng dữ liệu."""
        evaluator = Evaluator(self.problem, max_evaluations=MAX_EVALUATIONS)
        callback_history = []

        def cb(gen, ms, chrom):
            callback_history.append((gen, ms))

        ga = GeneticAlgorithm(
            problem=self.problem,
            population_size=POPULATION_SIZE,
            max_generations=MAX_GENERATIONS,
            seed=42,
        )
        result = ga.solve(evaluator=evaluator, callback=cb)

        # 1. Kiểm tra ngân sách NFE chính xác 10,000
        self.assertEqual(result["evaluation_count"], 10000)
        self.assertEqual(evaluator.evaluation_count, 10000)
        self.assertTrue(evaluator.is_budget_exhausted)

        # 2. Kiểm tra số điểm hội tụ (đúng 200 điểm)
        self.assertEqual(len(result["convergence"]), CONVERGENCE_POINTS)

        # 3. Kiểm tra callback được gọi đúng 200 lần (từ thế hệ 0 đến 199)
        self.assertEqual(len(callback_history), 200)
        self.assertEqual(callback_history[0][0], 0)
        self.assertEqual(callback_history[-1][0], 199)

        # 4. Kiểm tra hợp đồng dữ liệu
        validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_ga_seed_reproducibility(self):
        """Kiểm tra tính tái lập 100%: Hai lần chạy cùng seed phải cho kết quả giống hệt nhau."""
        res1 = run_ga(self.problem, seed=999)
        res2 = run_ga(self.problem, seed=999)

        self.assertEqual(res1["assignment"], res2["assignment"])
        self.assertEqual(res1["makespan"], res2["makespan"])
        self.assertEqual(res1["convergence"], res2["convergence"])
        self.assertEqual(res1["schedule"], res2["schedule"])

    def test_ga_small_scale_custom_budget(self):
        """Kiểm tra GA chạy với ngân sách nhỏ tùy chỉnh (ví dụ 10 pop x 5 gen = 60 NFE)."""
        custom_pop = 10
        custom_gen = 5
        custom_budget = custom_pop + custom_gen * custom_pop  # 60 NFE

        res = run_ga(
            self.problem,
            params={"population_size": custom_pop, "max_generations": custom_gen},
            seed=12,
        )

        self.assertEqual(res["evaluation_count"], custom_budget)
        self.assertEqual(len(res["convergence"]), 1 + custom_gen)

    def test_ga_rejects_stale_or_mismatched_evaluator(self):
        """Kiểm tra GA bắt buộc Evaluator phải mới nguyên (evaluation_count == 0) và đúng problem."""
        ga = GeneticAlgorithm(self.problem, population_size=10, max_generations=5, seed=1)

        # Evaluator đã tiêu thụ NFE
        stale_evaluator = Evaluator(self.problem, max_evaluations=100)
        stale_evaluator.evaluate([0] * self.problem.num_jobs)
        self.assertEqual(stale_evaluator.evaluation_count, 1)

        with self.assertRaises(ValueError) as ctx:
            ga.solve(evaluator=stale_evaluator)
        self.assertIn("Evaluator đã được sử dụng trước đó", str(ctx.exception))

        # Evaluator khác bài toán
        other_problem = UPMSProblem(2, 2, [[10.0, 20.0], [30.0, 40.0]])
        wrong_evaluator = Evaluator(other_problem, max_evaluations=100)
        with self.assertRaises(ValueError) as ctx2:
            ga.solve(evaluator=wrong_evaluator)
        self.assertIn("không gắn với cùng đối tượng UPMSProblem", str(ctx2.exception))

        # Evaluator không khớp ngân sách (GA cần 10 + 5*10 = 60 NFE, nhưng Evaluator cấp 50)
        mismatched_budget_evaluator = Evaluator(self.problem, max_evaluations=50)
        with self.assertRaises(ValueError) as ctx3:
            ga.solve(evaluator=mismatched_budget_evaluator)
        self.assertIn("Ngân sách Evaluator", str(ctx3.exception))

    def test_ga_single_machine_boundary(self):
        """Kiểm tra trường hợp biên: Chỉ có 1 máy (M=1). Tất cả jobs phải gán vào máy 0."""
        single_machine_matrix = [[15.0, 25.0, 35.0, 45.0]]
        p_single = UPMSProblem(num_jobs=4, num_machines=1, processing_times=single_machine_matrix)
        res = run_ga(p_single, params={"population_size": 10, "max_generations": 5}, seed=7)

        self.assertEqual(res["assignment"], [0, 0, 0, 0])
        self.assertAlmostEqual(res["makespan"], 15.0 + 25.0 + 35.0 + 45.0)

    def test_ga_zero_pc_zero_pm_boundary(self):
        """Kiểm tra trường hợp biên Pc = 0.0, Pm = 0.0 (chỉ chọn lọc và nhân bản)."""
        res = run_ga(
            self.problem,
            params={
                "population_size": 10,
                "max_generations": 5,
                "crossover_prob": 0.0,
                "mutation_prob": 0.0,
            },
            seed=42,
        )
        self.assertEqual(res["evaluation_count"], 10 + 5 * 10)
        validate_result_contract(res, self.problem, expected_convergence_len=6, max_evaluations=60)

    def test_runtime_excludes_callback_duration(self):
        """
        Kiểm tra thực tế: Phép đo runtime thuần túy loại trừ thời gian thực thi callback.
        Lưu ý kỹ thuật: wall_elapsed bao gồm cả thời gian chạy thuật toán, thời gian callback,
        và một phần nhỏ thời gian xây dựng schedule / validate_result_contract sau vòng lặp.
        Kiểm tra này đóng vai trò xác nhận thực nghiệm gần đúng rằng thời gian callback đã được
        khấu trừ (độ chênh lệch giữa wall-clock và runtime tối thiểu bằng 80% thời gian callback).
        """
        import time

        def slow_callback(gen, ms, chrom):
            # Giả lập thao tác vẽ giao diện hoặc I/O chậm (sleep 5ms mỗi lần)
            time.sleep(0.005)

        ga = GeneticAlgorithm(self.problem, population_size=10, max_generations=5, seed=123)
        wall_start = time.perf_counter()
        res_with_slow_cb = ga.solve(callback=slow_callback)
        wall_elapsed = time.perf_counter() - wall_start

        callback_duration = 6 * 0.005  # 6 lần gọi (gen 0 đến 5)
        self.assertGreaterEqual(
            wall_elapsed - res_with_slow_cb["runtime"],
            callback_duration * 0.8
        )


if __name__ == "__main__":
    unittest.main()
