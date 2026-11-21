"""
Package core: Chứa mô hình bài toán UPMS, cấu hình tham số chung và bộ đánh giá NFE.
"""

from .config import (
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
    VALID_GA_CROSSOVER_TYPES,
    VALID_GA_MUTATION_TYPES,
    WOA_B,
    WOA_A_MAX,
    WOA_A_MIN,
    WOA_VARIANTS,
    VALID_WOA_VARIANTS,
    BENCHMARK_SEEDS,
    BENCHMARK_DATASETS,
    PROCESSING_TIME_MIN,
    PROCESSING_TIME_MAX,
    BENCHMARK_RUNS,
    TOTAL_BENCHMARK_RUNS,
    STATISTICAL_ALPHA,
    validate_config,
)
from .evaluator import Evaluator
from .problem import UPMSProblem, validate_result_contract

__all__ = [
    # Cấu hình NFE & Quần thể
    "POPULATION_SIZE",
    "MAX_GENERATIONS",
    "MAX_EVALUATIONS",
    "CONVERGENCE_POINTS",
    # Tham số GA
    "GA_PC",
    "GA_PM",
    "GA_TOURNAMENT_K",
    "GA_ELITE_COUNT",
    "GA_CROSSOVER_TYPE",
    "GA_MUTATION_TYPE",
    "VALID_GA_CROSSOVER_TYPES",
    "VALID_GA_MUTATION_TYPES",
    # Tham số WOA
    "WOA_B",
    "WOA_A_MAX",
    "WOA_A_MIN",
    "WOA_VARIANTS",
    "VALID_WOA_VARIANTS",
    # Tham số Benchmark & Dữ liệu
    "BENCHMARK_SEEDS",
    "BENCHMARK_DATASETS",
    "PROCESSING_TIME_MIN",
    "PROCESSING_TIME_MAX",
    "BENCHMARK_RUNS",
    "TOTAL_BENCHMARK_RUNS",
    "STATISTICAL_ALPHA",
    "validate_config",
    # Lớp xử lý & Hợp đồng dữ liệu
    "Evaluator",
    "UPMSProblem",
    "validate_result_contract",
]
