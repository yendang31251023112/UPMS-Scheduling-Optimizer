"""
Bộ kiểm thử tự động toàn diện (Unit Test) cho gói core:
1. TestCoreConfig:
   - Kiểm tra các hằng số cấu hình.
   - Kiểm tra hàm validate_config() tự động chạy và phát hiện các tham số sai:
     GA_PC là boolean (True), WOA_VARIANTS trùng tên hoặc sai tên,
     Dataset có num_jobs/num_machines <= 0, STATISTICAL_ALPHA là NaN/bool.
2. TestUPMSProblem:
   - Tính toán Makespan và phân bổ tải máy (cả Python list và NumPy ndarray).
   - Lịch trình Gantt: Job ID tăng dần, thời gian bắt đầu 0.0, tính liên tục thời gian.
   - Ngoại lệ ma trận: Jagged array, NaN, Inf, số âm, boolean, sai kích thước (M, N).
   - Constructor num_jobs / num_machines: Chặn float, bool, <= 0.
   - generate_random_instance: Kiểm tra tham số trước khi sinh ma trận (chặn seed=bool, num_jobs<=0),
     tính tái lập 100% khi cùng seed.
   - Serialization to_dict và from_dict.
3. TestEvaluator:
   - Cơ chế đếm NFE chính xác và chặn vượt 10,000 NFE.
   - Đầu vào sai (float, bool, sai máy, sai độ dài) KHÔNG làm tiêu hao ngân sách NFE.
   - Chặn Makespan không hợp lệ (NaN, Inf, số âm, bool) từ problem và KHÔNG tăng NFE.
   - Kiểm tra reset() đưa evaluation_count về 0 và cho phép đánh giá lại bình thường.
   - Kiểm tra constructor max_evaluations (chặn float, bool, <= 0).
4. TestResultContract:
   - Hợp đồng kết quả hợp lệ với 200 điểm hội tụ giảm dần và khớp Makespan.
   - Kiểm tra tham số expected_convergence_len (chặn <= 0, float, bool).
   - Bắt lỗi khi schedule sai máy, sai thời lượng, thiếu job, sai thứ tự Job ID / máy chuẩn.
   - Bắt lỗi khi convergence chứa NaN/Inf/bool/âm, không đơn điệu (tăng giá trị), hoặc điểm cuối không khớp Makespan.
   - Bắt lỗi khi runtime hoặc makespan chứa NaN/Inf/bool/âm.
"""

import math
import unittest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from core.config import (
    POPULATION_SIZE,
    MAX_GENERATIONS,
    MAX_EVALUATIONS,
    CONVERGENCE_POINTS,
    GA_PC,
    GA_PM,
    GA_TOURNAMENT_K,
    GA_ELITE_COUNT,
    GA_CROSSOVER_TYPE,
    GA_MUTATION_TYPE,
    WOA_B,
    WOA_A_MAX,
    WOA_A_MIN,
    WOA_VARIANTS,
    BENCHMARK_SEEDS,
    BENCHMARK_RUNS,
    TOTAL_BENCHMARK_RUNS,
    STATISTICAL_ALPHA,
    validate_config,
)
import core.config as cfg
from core.problem import UPMSProblem, validate_result_contract
from core.evaluator import Evaluator


class TestCoreConfig(unittest.TestCase):
    """Kiểm tra tính nhất quán toán học của config.py."""

    def test_constants_and_validation(self):
        self.assertEqual(POPULATION_SIZE, 50)
        self.assertEqual(MAX_GENERATIONS, 199)
        self.assertEqual(MAX_EVALUATIONS, 10000)
        self.assertEqual(CONVERGENCE_POINTS, 200)
        self.assertEqual(GA_PC, 0.8)
        self.assertEqual(GA_PM, 0.05)
        self.assertEqual(GA_TOURNAMENT_K, 3)
        self.assertEqual(GA_ELITE_COUNT, 1)
        self.assertEqual(GA_CROSSOVER_TYPE, "uniform")
        self.assertEqual(GA_MUTATION_TYPE, "resetting")
        self.assertEqual(WOA_B, 1.0)
        self.assertEqual(WOA_A_MAX, 2.0)
        self.assertEqual(WOA_A_MIN, 0.0)
        self.assertEqual(WOA_VARIANTS, ("baseline",))
        self.assertEqual(len(BENCHMARK_SEEDS), 30)
        self.assertEqual(BENCHMARK_RUNS, 30)
        self.assertEqual(TOTAL_BENCHMARK_RUNS, 180)
        self.assertEqual(STATISTICAL_ALPHA, 0.05)

        # Chạy kiểm tra tính nhất quán toán học
        validate_config()

    def test_validate_config_rejects_bool_for_float_params(self):
        """Kiểm tra validate_config() chặn giá trị boolean True cho GA_PC (không nhận isinstance True==1)."""
        original_pc = cfg.GA_PC
        try:
            cfg.GA_PC = True
            with self.assertRaises(ValueError):
                cfg.validate_config()
        finally:
            cfg.GA_PC = original_pc

    def test_validate_config_rejects_invalid_woa_variants(self):
        """Kiểm tra validate_config() chặn WOA_VARIANTS có cấu hình không đúng quy ước."""
        original_variants = cfg.WOA_VARIANTS
        try:
            # Rỗng hoặc nhiều hơn 1 cấu hình
            cfg.WOA_VARIANTS = ("baseline", "extra")
            with self.assertRaises(ValueError):
                cfg.validate_config()

            # Sai tên
            cfg.WOA_VARIANTS = ("invalid_variant",)
            with self.assertRaises(ValueError):
                cfg.validate_config()
        finally:
            cfg.WOA_VARIANTS = original_variants

    def test_validate_config_rejects_invalid_datasets(self):
        """Kiểm tra validate_config() chặn dataset có num_jobs <= 0 khi vẫn có đủ 3 datasets."""
        original_ds = cfg.BENCHMARK_DATASETS
        try:
            invalid_ds = {k: dict(v) for k, v in original_ds.items()}
            invalid_ds["small"]["num_jobs"] = 0
            cfg.BENCHMARK_DATASETS = invalid_ds
            with self.assertRaises(ValueError) as ctx:
                cfg.validate_config()
            self.assertIn("num_jobs phải là số nguyên > 0", str(ctx.exception))
        finally:
            cfg.BENCHMARK_DATASETS = original_ds

    def test_validate_config_rejects_missing_dataset_keys(self):
        """Kiểm tra validate_config() chặn cấu hình thiếu một trong 3 datasets bắt buộc."""
        original_ds = cfg.BENCHMARK_DATASETS
        try:
            cfg.BENCHMARK_DATASETS = {
                "small": {"num_jobs": 10, "num_machines": 3, "filename": "small_10x3.json"}
            }
            with self.assertRaises(ValueError) as ctx:
                cfg.validate_config()
            self.assertIn("phải có chính xác 3 datasets", str(ctx.exception))
        finally:
            cfg.BENCHMARK_DATASETS = original_ds

    def test_validate_config_rejects_nan_alpha(self):
        """Kiểm tra validate_config() chặn STATISTICAL_ALPHA là NaN hoặc bool."""
        original_alpha = cfg.STATISTICAL_ALPHA
        try:
            cfg.STATISTICAL_ALPHA = float("nan")
            with self.assertRaises(ValueError):
                cfg.validate_config()

            cfg.STATISTICAL_ALPHA = True
            with self.assertRaises(ValueError):
                cfg.validate_config()
        finally:
            cfg.STATISTICAL_ALPHA = original_alpha

    def test_validate_config_rejects_invalid_convergence_points(self):
        """Kiểm tra validate_config() chặn CONVERGENCE_POINTS <= 0 hoặc sai kiểu."""
        original_cp = cfg.CONVERGENCE_POINTS
        try:
            cfg.CONVERGENCE_POINTS = 0
            with self.assertRaises(ValueError):
                cfg.validate_config()

            cfg.CONVERGENCE_POINTS = "200"
            with self.assertRaises(ValueError):
                cfg.validate_config()
        finally:
            cfg.CONVERGENCE_POINTS = original_cp

    def test_validate_config_rejects_invalid_param_study(self):
        """Kiểm tra validate_config() chặn các giá trị khảo sát tham số GA sai miền hoặc trùng lặp."""
        original_pc_study = cfg.GA_PARAM_STUDY_PC
        original_pm_study = cfg.GA_PARAM_STUDY_PM
        original_k_study = cfg.GA_PARAM_STUDY_K
        try:
            # Sai miền giá trị
            cfg.GA_PARAM_STUDY_PC = (0.5, 1.5)  # 1.5 > 1.0
            with self.assertRaises(ValueError):
                cfg.validate_config()

            cfg.GA_PARAM_STUDY_K = (0, 3)  # k < 1
            with self.assertRaises(ValueError):
                cfg.validate_config()

            # Trùng lặp giá trị
            cfg.GA_PARAM_STUDY_PC = (0.6, 0.7, 0.6)
            with self.assertRaises(ValueError):
                cfg.validate_config()

            cfg.GA_PARAM_STUDY_PM = (0.01, 0.05, 0.01)
            with self.assertRaises(ValueError):
                cfg.validate_config()

            cfg.GA_PARAM_STUDY_K = (2, 3, 2)
            with self.assertRaises(ValueError):
                cfg.validate_config()
        finally:
            cfg.GA_PARAM_STUDY_PC = original_pc_study
            cfg.GA_PARAM_STUDY_PM = original_pm_study
            cfg.GA_PARAM_STUDY_K = original_k_study

    def test_validate_config_rejects_path_traversal_filename(self):
        """Kiểm tra validate_config() chặn filename chứa đường dẫn thư mục."""
        original_ds = cfg.BENCHMARK_DATASETS
        try:
            cfg.BENCHMARK_DATASETS = {
                "small": {"num_jobs": 10, "num_machines": 3, "filename": "../secret.json"},
                "medium": {"num_jobs": 30, "num_machines": 5, "filename": "medium_30x5.json"},
                "large": {"num_jobs": 100, "num_machines": 10, "filename": "large_100x10.json"},
            }
            with self.assertRaises(ValueError):
                cfg.validate_config()
        finally:
            cfg.BENCHMARK_DATASETS = original_ds


class TestUPMSProblem(unittest.TestCase):
    """Kiểm tra mô hình bài toán UPMSProblem và tính toàn vẹn dữ liệu."""

    def setUp(self):
        self.matrix = [
            [10.0, 20.0, 30.0, 40.0],
            [15.0, 25.0, 35.0, 45.0],
            [20.0, 10.0, 20.0, 30.0],
        ]
        self.problem = UPMSProblem(num_jobs=4, num_machines=3, processing_times=self.matrix)

    def test_initialization_and_makespan(self):
        self.assertEqual(self.problem.num_jobs, 4)
        self.assertEqual(self.problem.num_machines, 3)

        assignment = [0, 2, 0, 1]
        self.assertEqual(self.problem.calculate_makespan(assignment), 45.0)
        self.assertEqual(self.problem.get_machine_loads(assignment), [40.0, 45.0, 10.0])

    def test_assignment_with_numpy_array(self):
        if not _HAS_NUMPY:
            self.skipTest("Môi trường chưa cài numpy, bỏ qua test numpy.")
        np_assignment = np.array([0, 2, 0, 1], dtype=int)
        self.assertEqual(self.problem.calculate_makespan(np_assignment), 45.0)

    def test_gantt_schedule_continuity_and_ordering(self):
        assignment = [0, 2, 0, 1]
        schedule = self.problem.build_schedule(assignment)

        # Lọc công việc trên máy 0
        m0_items = [item for item in schedule if item[0] == 0]
        self.assertEqual(len(m0_items), 2)

        # Quy chuẩn: Thứ tự Job ID tăng dần -> job 0 trước, job 2 sau
        self.assertEqual(m0_items[0], (0, 0, 0.0, 10.0))
        self.assertEqual(m0_items[1], (0, 2, 10.0, 40.0))

        # Kiểm tra tính liên tục thời gian trên từng máy
        for m in range(self.problem.num_machines):
            m_items = [item for item in schedule if item[0] == m]
            if not m_items:
                continue
            self.assertEqual(m_items[0][2], 0.0)
            for idx in range(len(m_items) - 1):
                self.assertEqual(m_items[idx][3], m_items[idx + 1][2])

    def test_constructor_parameter_validation(self):
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=2.5, num_machines=3, processing_times=self.matrix)
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=4, num_machines=3.0, processing_times=self.matrix)
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=True, num_machines=3, processing_times=self.matrix)
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=4, num_machines=False, processing_times=self.matrix)
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=0, num_machines=3, processing_times=self.matrix)
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=4, num_machines=-1, processing_times=self.matrix)

    def test_matrix_validation_jagged_array(self):
        jagged_matrix = [
            [10.0, 20.0, 30.0, 40.0],
            [15.0, 25.0],
            [20.0, 10.0, 20.0, 30.0],
        ]
        with self.assertRaises(ValueError):
            UPMSProblem(num_jobs=4, num_machines=3, processing_times=jagged_matrix)

    def test_matrix_validation_nan_and_inf(self):
        nan_matrix = [
            [10.0, float("nan"), 30.0, 40.0],
            [15.0, 25.0, 35.0, 45.0],
            [20.0, 10.0, 20.0, 30.0],
        ]
        with self.assertRaises(ValueError):
            UPMSProblem(num_jobs=4, num_machines=3, processing_times=nan_matrix)

        inf_matrix = [
            [10.0, 20.0, 30.0, 40.0],
            [15.0, float("inf"), 35.0, 45.0],
            [20.0, 10.0, 20.0, 30.0],
        ]
        with self.assertRaises(ValueError):
            UPMSProblem(num_jobs=4, num_machines=3, processing_times=inf_matrix)

    def test_matrix_validation_negative_and_types(self):
        neg_matrix = [
            [10.0, -5.0, 30.0, 40.0],
            [15.0, 25.0, 35.0, 45.0],
            [20.0, 10.0, 20.0, 30.0],
        ]
        with self.assertRaises(ValueError):
            UPMSProblem(num_jobs=4, num_machines=3, processing_times=neg_matrix)

        bool_matrix = [
            [10.0, True, 30.0, 40.0],
            [15.0, 25.0, 35.0, 45.0],
            [20.0, 10.0, 20.0, 30.0],
        ]
        with self.assertRaises(TypeError):
            UPMSProblem(num_jobs=4, num_machines=3, processing_times=bool_matrix)

    def test_matrix_validation_dimension_mismatch(self):
        matrix_2rows = [
            [10.0, 20.0, 30.0, 40.0],
            [15.0, 25.0, 35.0, 45.0],
        ]
        with self.assertRaises(ValueError):
            UPMSProblem(num_jobs=4, num_machines=3, processing_times=matrix_2rows)

    def test_seed_reproducibility(self):
        p1 = UPMSProblem.generate_random_instance(num_jobs=20, num_machines=4, seed=123)
        p2 = UPMSProblem.generate_random_instance(num_jobs=20, num_machines=4, seed=123)
        self.assertEqual(p1.processing_times, p2.processing_times)

        p3 = UPMSProblem.generate_random_instance(num_jobs=20, num_machines=4, seed=456)
        self.assertNotEqual(p1.processing_times, p3.processing_times)

    def test_generate_random_instance_parameter_checks(self):
        """Kiểm tra generate_random_instance chặn tham số sai ngay từ đầu."""
        with self.assertRaises(TypeError):
            UPMSProblem.generate_random_instance(num_jobs=0, num_machines=3)
        with self.assertRaises(TypeError):
            UPMSProblem.generate_random_instance(num_jobs=10, num_machines=0)
        with self.assertRaises(TypeError):
            UPMSProblem.generate_random_instance(num_jobs=10, num_machines=3, seed=True)
        with self.assertRaises(ValueError):
            UPMSProblem.generate_random_instance(10, 3, p_min=0)
        with self.assertRaises(ValueError):
            UPMSProblem.generate_random_instance(10, 3, p_min=50, p_max=20)

    def test_serialization_dict(self):
        data = self.problem.to_dict()
        restored = UPMSProblem.from_dict(data)
        self.assertEqual(self.problem.num_jobs, restored.num_jobs)
        self.assertEqual(self.problem.num_machines, restored.num_machines)
        self.assertEqual(self.problem.processing_times, restored.processing_times)

    def test_from_dict_schema_validation(self):
        """Kiểm tra from_dict() xác thực cấu trúc schema nghiêm ngặt."""
        # Không phải dict
        with self.assertRaises(TypeError):
            UPMSProblem.from_dict([1, 2, 3])
        # Thiếu trường
        with self.assertRaises(ValueError):
            UPMSProblem.from_dict({"num_jobs": 2, "num_machines": 2})
        # Thừa trường
        valid_dict = self.problem.to_dict()
        valid_dict["extra_key"] = "bad"
        with self.assertRaises(ValueError):
            UPMSProblem.from_dict(valid_dict)

    def test_validate_benchmark_bounds(self):
        """Kiểm tra validate_benchmark_bounds() phát hiện thời gian ngoài [10, 100]."""
        # Hợp lệ (10-100)
        p_valid = UPMSProblem(2, 2, [[10.0, 50.0], [80.0, 100.0]])
        p_valid.validate_benchmark_bounds(10.0, 100.0)

        # Quá nhỏ (< 10)
        p_low = UPMSProblem(2, 2, [[5.0, 50.0], [80.0, 100.0]])
        with self.assertRaises(ValueError):
            p_low.validate_benchmark_bounds(10.0, 100.0)

        # Quá lớn (> 100)
        p_high = UPMSProblem(2, 2, [[10.0, 50.0], [80.0, 150.0]])
        with self.assertRaises(ValueError):
            p_high.validate_benchmark_bounds(10.0, 100.0)

        # Tham số min_time / max_time không hợp lệ
        with self.assertRaises(TypeError):
            p_valid.validate_benchmark_bounds(min_time="10.0")
        with self.assertRaises(TypeError):
            p_valid.validate_benchmark_bounds(min_time=True)
        with self.assertRaises(ValueError):
            p_valid.validate_benchmark_bounds(min_time=-1.0)
        with self.assertRaises(ValueError):
            p_valid.validate_benchmark_bounds(min_time=100.0, max_time=50.0)


class TestEvaluator(unittest.TestCase):
    """Kiểm tra lớp Evaluator và cơ chế ngân sách NFE chặt chẽ."""

    def setUp(self):
        self.matrix = [
            [10.0, 20.0],
            [15.0, 25.0],
        ]
        self.problem = UPMSProblem(num_jobs=2, num_machines=2, processing_times=self.matrix)

    def test_constructor_max_evaluations_validation(self):
        with self.assertRaises(TypeError):
            Evaluator(self.problem, max_evaluations=100.5)
        with self.assertRaises(TypeError):
            Evaluator(self.problem, max_evaluations=True)
        with self.assertRaises(ValueError):
            Evaluator(self.problem, max_evaluations=0)
        with self.assertRaises(ValueError):
            Evaluator(self.problem, max_evaluations=-5)

    def test_budget_exhaustion(self):
        evaluator = Evaluator(self.problem, max_evaluations=2)
        self.assertEqual(evaluator.remaining_evaluations, 2)
        self.assertFalse(evaluator.is_budget_exhausted)

        evaluator.evaluate([0, 1])
        self.assertEqual(evaluator.evaluation_count, 1)

        evaluator.evaluate([1, 0])
        self.assertEqual(evaluator.evaluation_count, 2)
        self.assertTrue(evaluator.is_budget_exhausted)

        # Lần 3 vượt ngân sách -> ném RuntimeError
        with self.assertRaises(RuntimeError):
            evaluator.evaluate([0, 0])

    def test_reject_floats_and_booleans_no_silent_coercion(self):
        evaluator = Evaluator(self.problem, max_evaluations=10)

        with self.assertRaises(TypeError):
            evaluator.evaluate([0, 1.8])
        self.assertEqual(evaluator.evaluation_count, 0)

        with self.assertRaises(TypeError):
            evaluator.evaluate([True, False])
        self.assertEqual(evaluator.evaluation_count, 0)

    def test_invalid_assignment_does_not_consume_nfe(self):
        evaluator = Evaluator(self.problem, max_evaluations=10)

        with self.assertRaises(ValueError):
            evaluator.evaluate([0, 5])
        self.assertEqual(evaluator.evaluation_count, 0)

        with self.assertRaises(ValueError):
            evaluator.evaluate([0, 1, 0])
        self.assertEqual(evaluator.evaluation_count, 0)

        with self.assertRaises(ValueError):
            evaluator.evaluate(None)
        self.assertEqual(evaluator.evaluation_count, 0)

        evaluator.evaluate([0, 1])
        self.assertEqual(evaluator.evaluation_count, 1)

    def test_evaluator_reset(self):
        evaluator = Evaluator(self.problem, max_evaluations=10)
        evaluator.evaluate([0, 1])
        evaluator.evaluate([1, 0])
        self.assertEqual(evaluator.evaluation_count, 2)

        evaluator.reset()
        self.assertEqual(evaluator.evaluation_count, 0)
        self.assertEqual(evaluator.remaining_evaluations, 10)
        self.assertFalse(evaluator.is_budget_exhausted)

        # Đánh giá lại sau khi reset
        evaluator.evaluate([0, 0])
        self.assertEqual(evaluator.evaluation_count, 1)

    def test_evaluator_rejects_nan_and_bool_makespan(self):
        """Nếu calculate_makespan trả về NaN, Inf hoặc bool, Evaluator phải ném lỗi và không tăng NFE."""
        class MockProblemNaN:
            num_jobs = 2
            num_machines = 2
            def calculate_makespan(self, assignment):
                return float("nan")

        class MockProblemBool:
            num_jobs = 2
            num_machines = 2
            def calculate_makespan(self, assignment):
                return True

        mock_nan = MockProblemNaN()
        evaluator_nan = Evaluator(mock_nan, max_evaluations=10)
        with self.assertRaises(ValueError):
            evaluator_nan.evaluate([0, 1])
        self.assertEqual(evaluator_nan.evaluation_count, 0)

        mock_bool = MockProblemBool()
        evaluator_bool = Evaluator(mock_bool, max_evaluations=10)
        with self.assertRaises(TypeError):
            evaluator_bool.evaluate([0, 1])
        self.assertEqual(evaluator_bool.evaluation_count, 0)


class TestResultContract(unittest.TestCase):
    """Kiểm tra hợp đồng dữ liệu kết quả (result dictionary)."""

    def setUp(self):
        self.matrix = [
            [10.0, 20.0],
            [15.0, 25.0],
        ]
        self.problem = UPMSProblem(num_jobs=2, num_machines=2, processing_times=self.matrix)

    def _create_valid_result(self):
        convergence = [40.0 - 15.0 * (i / 199.0) for i in range(200)]
        return {
            "assignment": [0, 1],
            "makespan": 25.0,
            "schedule": [(0, 0, 0.0, 10.0), (1, 1, 0.0, 25.0)],
            "convergence": convergence,
            "runtime": 0.45,
            "evaluation_count": 10000,
        }

    def test_valid_result_contract(self):
        result = self._create_valid_result()
        validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_expected_convergence_len_validation(self):
        result = self._create_valid_result()
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=0)
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=-10)

    def test_invalid_convergence_length(self):
        result = self._create_valid_result()
        result["convergence"] = result["convergence"][:199]
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_convergence_non_monotonic(self):
        result = self._create_valid_result()
        result["convergence"][50] = result["convergence"][49] + 5.0
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_convergence_final_mismatch(self):
        result = self._create_valid_result()
        result["convergence"][-1] = 28.0
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_convergence_nan_or_bool(self):
        result = self._create_valid_result()
        result["convergence"][10] = float("nan")
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

        result["convergence"][10] = True
        with self.assertRaises(TypeError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_runtime_nan_or_bool(self):
        result = self._create_valid_result()
        result["runtime"] = float("nan")
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

        result["runtime"] = True
        with self.assertRaises(TypeError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_makespan_mismatch(self):
        result = self._create_valid_result()
        result["makespan"] = 99.0
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_schedule_wrong_order_or_machine(self):
        result = self._create_valid_result()
        # Đảo thứ tự schedule (máy 1 trước, máy 0 sau)
        result["schedule"] = [(1, 1, 0.0, 25.0), (0, 0, 0.0, 10.0)]
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_schedule_wrong_time(self):
        result = self._create_valid_result()
        # Sai start_time
        result["schedule"] = [(0, 0, 2.0, 12.0), (1, 1, 0.0, 25.0)]
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_schedule_types_bool_and_float(self):
        """Kiểm tra chặn boolean (True) và float (0.0) trong machine_id và job_id của schedule."""
        result = self._create_valid_result()
        # machine_id là True (dù True == 1 nhưng phải chặn vì không phải int thuần)
        result["schedule"] = [(True, 0, 0.0, 10.0), (1, 1, 0.0, 25.0)]
        with self.assertRaises(TypeError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

        # job_id là float 0.0 (dù 0.0 == 0 nhưng phải chặn)
        result["schedule"] = [(0, 0.0, 0.0, 10.0), (1, 1, 0.0, 25.0)]
        with self.assertRaises(TypeError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_schedule_nan_and_inf_times(self):
        """Kiểm tra thời gian start_time / end_time là NaN hoặc Inf phải ném ValueError."""
        result = self._create_valid_result()
        # start_time là NaN
        result["schedule"] = [(0, 0, float("nan"), 10.0), (1, 1, 0.0, 25.0)]
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

        # end_time là Inf
        result["schedule"] = [(0, 0, 0.0, float("inf")), (1, 1, 0.0, 25.0)]
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_schedule_duplicate_jobs(self):
        """Kiểm tra schedule có job bị trùng lặp phải ném ValueError."""
        result = self._create_valid_result()
        # Trùng job 0 (không có job 1)
        result["schedule"] = [(0, 0, 0.0, 10.0), (0, 0, 10.0, 20.0)]
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

    def test_invalid_evaluation_count_bool_and_custom_max(self):
        """Kiểm tra evaluation_count là bool hoặc vượt trần max_evaluations tùy chỉnh."""
        result = self._create_valid_result()
        # evaluation_count là bool
        result["evaluation_count"] = True
        with self.assertRaises(TypeError):
            validate_result_contract(result, self.problem, expected_convergence_len=200)

        # evaluation_count = 200 vượt trần max_evaluations = 100
        result["evaluation_count"] = 200
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200, max_evaluations=100)

        # max_evaluations không hợp lệ (<= 0)
        with self.assertRaises(ValueError):
            validate_result_contract(result, self.problem, expected_convergence_len=200, max_evaluations=0)


    def test_validate_result_contract_rejects_extra_fields(self):
        """Kiểm tra validate_result_contract() phát hiện và từ chối result có trường thừa."""
        result = self._create_valid_result()
        result["extra_field"] = "unexpected_data"
        with self.assertRaises(ValueError) as ctx:
            validate_result_contract(result, self.problem, expected_convergence_len=200)
        self.assertIn("chứa các trường thừa không được phép", str(ctx.exception))

    def test_validate_result_contract_rejects_missing_fields(self):
        """Kiểm tra validate_result_contract() phát hiện và từ chối result bị thiếu trường bắt buộc."""
        result = self._create_valid_result()
        del result["runtime"]
        with self.assertRaises(ValueError) as ctx:
            validate_result_contract(result, self.problem, expected_convergence_len=200)
        self.assertIn("thiếu các trường bắt buộc", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
