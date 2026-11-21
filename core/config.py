"""
Tệp cấu hình tập trung chứa toàn bộ tham số kỹ thuật chuẩn của đồ án UPMS.
Thống nhất dùng chung cho toàn bộ thành viên (GA, WOA, GUI, Benchmark).
Không tự ý sửa đổi các hằng số này trong quá trình thực nghiệm chính thức.
"""

import math
from typing import Any, Dict, List, Tuple

# 1. CẤU HÌNH NGÂN SÁCH ĐÁNH GIÁ (NFE - NUMBER OF FUNCTION EVALUATIONS)
# Ngân sách cố định cho 1 lượt chạy: 50 khởi tạo + 199 thế hệ * 50 = 10,000 NFE
# Quy tắc Elitism trong GA:
#   - Khởi tạo: Đánh giá 50 cá thể ban đầu -> tiêu thụ 50 NFE.
#   - Mỗi thế hệ t = 1..199: Sinh đúng 50 cá thể con mới (offspring), đánh giá cả 50 con mới -> 50 NFE.
#   - Gộp 50 con mới + 1 cá thể tinh hoa (Elite giữ từ thế hệ trước, KHÔNG đánh giá lại) = 51 ứng viên.
#   - Chọn lọc top 50 cá thể tốt nhất làm quần thể thế hệ mới.
#   => Tổng NFE tiêu thụ = 50 + 199 * 50 = 10,000 NFE chính xác tuyệt đối mà không cần đánh giá lại Elite!
POPULATION_SIZE: int = 50           # Kích thước quần thể (GA) / bầy đàn (WOA)
MAX_GENERATIONS: int = 199          # Số thế hệ / vòng lặp tìm kiếm (sau khởi tạo)
MAX_EVALUATIONS: int = 10000        # Trần ngân sách đánh giá hàm mục tiêu cố định

# Số điểm trên đường cong hội tụ: 1 điểm sau khởi tạo (thế hệ 0) + 199 thế hệ sau = 200 điểm
CONVERGENCE_POINTS: int = 1 + MAX_GENERATIONS  # 200

# 2. CẤU HÌNH THUẬT TOÁN DI TRUYỀN (GA - GENETIC ALGORITHM)
GA_PC: float = 0.8                  # Xác suất lai ghép cặp (Crossover probability)
GA_PM: float = 0.05                 # Xác suất đột biến cá thể (Mutation probability)
GA_TOURNAMENT_K: int = 3            # Kích thước giải đấu chọn lọc (Tournament size)
GA_ELITE_COUNT: int = 1             # Số lượng cá thể tinh hoa giữ lại (không đánh giá lại)
GA_CROSSOVER_TYPE: str = "uniform"  # Loại lai ghép mặc định
GA_MUTATION_TYPE: str = "resetting" # Loại đột biến mặc định

# Bộ giá trị hợp lệ cho GA (dùng để xác thực tham số)
VALID_GA_CROSSOVER_TYPES: Tuple[str, ...] = ("uniform", "two_point")
VALID_GA_MUTATION_TYPES: Tuple[str, ...] = ("resetting", "swap")

# 3. CẤU HÌNH THUẬT TOÁN BẦY CÁ VOI (WOA - WHALE OPTIMIZATION ALGORITHM)
WOA_B: float = 1.0                  # Hằng số định hình xoắn ốc bọt nước (Spiral constant)
WOA_A_MAX: float = 2.0              # Giá trị ban đầu của hệ số a (khai phá)
WOA_A_MIN: float = 0.0              # Giá trị kết thúc của hệ số a (khai thác)

# Tên chuẩn 4 biến thể WOA (sử dụng Tuple bất biến để tránh bị thay đổi trong runtime)
WOA_VARIANTS: Tuple[str, ...] = (
    "baseline",                     # WOA cơ sở (Linear decay a, Clip boundary)
    "reinit",                       # Biến thể 1: Re-initialization khi vượt biên
    "cosine",                       # Biến thể 2: Cosine decay schedule cho a
    "quadratic",                    # Biến thể 3: Quadratic decay schedule cho a
)
VALID_WOA_VARIANTS: Tuple[str, ...] = WOA_VARIANTS

# 4. CẤU HÌNH BỘ DỮ LIỆU THỬ NGHIỆM (DATASETS)
# Phạm vi sinh thời gian xử lý p_ij ngẫu nhiên
PROCESSING_TIME_MIN: int = 10
PROCESSING_TIME_MAX: int = 100

# 3 quy mô bài toán thử nghiệm (M machines x N jobs) theo chuẩn (M, N)
BENCHMARK_DATASETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "num_jobs": 10,
        "num_machines": 3,
        "filename": "small_10x3.json",
    },
    "medium": {
        "num_jobs": 30,
        "num_machines": 5,
        "filename": "medium_30x5.json",
    },
    "large": {
        "num_jobs": 100,
        "num_machines": 10,
        "filename": "large_100x10.json",
    },
}

# 5. CẤU HÌNH BENCHMARK THỰC NGHIỆM & KIỂM ĐỊNH THỐNG KÊ
BENCHMARK_RUNS: int = 30            # Số lần chạy độc lập cho mỗi cấu hình trên 1 dataset
BENCHMARK_SEEDS: List[int] = list(range(1, 31))  # 30 seed ngẫu nhiên cố định [1..30]

# Tổng số lượt chạy thực nghiệm chính thức:
# (1 GA cơ sở + 4 WOA) x 3 datasets x 30 seeds = 5 x 3 x 30 = 450 runs
TOTAL_BENCHMARK_RUNS: int = (1 + len(WOA_VARIANTS)) * len(BENCHMARK_DATASETS) * BENCHMARK_RUNS  # 450

# Mức ý nghĩa thống kê alpha (kiểm định giả thuyết)
STATISTICAL_ALPHA: float = 0.05


def validate_config() -> None:
    """
    Kiểm tra toàn diện tính nhất quán toán học và tính hợp lệ của các tham số cấu hình.
    Sử dụng raise ValueError / TypeError thay vì assert để đảm bảo luôn chạy kể cả với cờ tối ưu -O.
    """
    # 1. Quần thể và NFE
    if type(POPULATION_SIZE) is not int or POPULATION_SIZE <= 0:
        raise ValueError(f"POPULATION_SIZE phải là số nguyên > 0, nhận: {POPULATION_SIZE}")
    if type(MAX_GENERATIONS) is not int or MAX_GENERATIONS <= 0:
        raise ValueError(f"MAX_GENERATIONS phải là số nguyên > 0, nhận: {MAX_GENERATIONS}")
    expected_nfe = POPULATION_SIZE + MAX_GENERATIONS * POPULATION_SIZE
    if type(MAX_EVALUATIONS) is not int or MAX_EVALUATIONS != expected_nfe:
        raise ValueError(
            f"Mâu thuẫn NFE: {POPULATION_SIZE} + {MAX_GENERATIONS} * {POPULATION_SIZE} = {expected_nfe} != {MAX_EVALUATIONS}"
        )
    if CONVERGENCE_POINTS != 1 + MAX_GENERATIONS:
        raise ValueError(f"CONVERGENCE_POINTS ({CONVERGENCE_POINTS}) phải bằng 1 + MAX_GENERATIONS ({1 + MAX_GENERATIONS})")

    # 2. Tham số GA
    if type(GA_PC) not in (int, float) or not math.isfinite(GA_PC) or not (0.0 <= GA_PC <= 1.0):
        raise ValueError(f"GA_PC phải là số thực hữu hạn trong [0, 1], nhận: {GA_PC} (kiểu {type(GA_PC).__name__})")
    if type(GA_PM) not in (int, float) or not math.isfinite(GA_PM) or not (0.0 <= GA_PM <= 1.0):
        raise ValueError(f"GA_PM phải là số thực hữu hạn trong [0, 1], nhận: {GA_PM} (kiểu {type(GA_PM).__name__})")
    if type(GA_TOURNAMENT_K) is not int or not (1 <= GA_TOURNAMENT_K <= POPULATION_SIZE):
        raise ValueError(f"GA_TOURNAMENT_K phải là số nguyên trong [1, {POPULATION_SIZE}], nhận: {GA_TOURNAMENT_K}")
    if type(GA_ELITE_COUNT) is not int or not (0 <= GA_ELITE_COUNT < POPULATION_SIZE):
        raise ValueError(f"GA_ELITE_COUNT phải là số nguyên trong [0, {POPULATION_SIZE - 1}], nhận: {GA_ELITE_COUNT}")
    if GA_CROSSOVER_TYPE not in VALID_GA_CROSSOVER_TYPES:
        raise ValueError(f"GA_CROSSOVER_TYPE không hợp lệ: {GA_CROSSOVER_TYPE}, miền hợp lệ: {VALID_GA_CROSSOVER_TYPES}")
    if GA_MUTATION_TYPE not in VALID_GA_MUTATION_TYPES:
        raise ValueError(f"GA_MUTATION_TYPE không hợp lệ: {GA_MUTATION_TYPE}, miền hợp lệ: {VALID_GA_MUTATION_TYPES}")

    # 3. Tham số WOA
    if type(WOA_B) not in (int, float) or not math.isfinite(WOA_B) or WOA_B <= 0.0:
        raise ValueError(f"WOA_B phải là số thực hữu hạn > 0, nhận: {WOA_B}")
    if type(WOA_A_MAX) not in (int, float) or not math.isfinite(WOA_A_MAX):
        raise ValueError(f"WOA_A_MAX phải là số thực hữu hạn, nhận: {WOA_A_MAX}")
    if type(WOA_A_MIN) not in (int, float) or not math.isfinite(WOA_A_MIN):
        raise ValueError(f"WOA_A_MIN phải là số thực hữu hạn, nhận: {WOA_A_MIN}")
    if not (WOA_A_MAX > WOA_A_MIN >= 0.0):
        raise ValueError(f"WOA_A_MAX ({WOA_A_MAX}) phải > WOA_A_MIN ({WOA_A_MIN}) >= 0")
    if not isinstance(WOA_VARIANTS, tuple) or len(WOA_VARIANTS) != 4 or len(set(WOA_VARIANTS)) != 4:
        raise ValueError("WOA_VARIANTS phải là tuple gồm đúng 4 biến thể phân biệt")
    if set(WOA_VARIANTS) != {"baseline", "reinit", "cosine", "quadratic"}:
        raise ValueError(f"Danh sách biến thể WOA không đúng quy ước: nhận {WOA_VARIANTS}")

    # 4. Tham số Dataset
    if type(PROCESSING_TIME_MIN) is not int or PROCESSING_TIME_MIN <= 0:
        raise ValueError(f"PROCESSING_TIME_MIN phải là số nguyên > 0, nhận: {PROCESSING_TIME_MIN}")
    if type(PROCESSING_TIME_MAX) is not int or PROCESSING_TIME_MAX < PROCESSING_TIME_MIN:
        raise ValueError(f"PROCESSING_TIME_MAX ({PROCESSING_TIME_MAX}) phải >= PROCESSING_TIME_MIN ({PROCESSING_TIME_MIN})")
    for ds_name, ds_info in BENCHMARK_DATASETS.items():
        if type(ds_info["num_jobs"]) is not int or ds_info["num_jobs"] <= 0:
            raise ValueError(f"Dataset '{ds_name}': num_jobs phải là số nguyên > 0, nhận: {ds_info['num_jobs']}")
        if type(ds_info["num_machines"]) is not int or ds_info["num_machines"] <= 0:
            raise ValueError(f"Dataset '{ds_name}': num_machines phải là số nguyên > 0, nhận: {ds_info['num_machines']}")

    # 5. Tham số Benchmark & Seeds
    if type(BENCHMARK_RUNS) is not int or BENCHMARK_RUNS <= 0:
        raise ValueError(f"BENCHMARK_RUNS phải là số nguyên > 0, nhận: {BENCHMARK_RUNS}")
    if len(BENCHMARK_SEEDS) != BENCHMARK_RUNS:
        raise ValueError(f"Số lượng seeds ({len(BENCHMARK_SEEDS)}) phải khớp với BENCHMARK_RUNS ({BENCHMARK_RUNS})")
    if len(set(BENCHMARK_SEEDS)) != BENCHMARK_RUNS:
        raise ValueError("Tất cả các seed trong BENCHMARK_SEEDS phải phân biệt (không được trùng lặp)")
    if not all(type(s) is int and s > 0 for s in BENCHMARK_SEEDS):
        raise ValueError("Tất cả các seed phải là số nguyên dương > 0")
    expected_total_runs = (1 + len(WOA_VARIANTS)) * len(BENCHMARK_DATASETS) * BENCHMARK_RUNS
    if TOTAL_BENCHMARK_RUNS != expected_total_runs:
        raise ValueError(f"TOTAL_BENCHMARK_RUNS ({TOTAL_BENCHMARK_RUNS}) != tính toán {expected_total_runs}")
    if type(STATISTICAL_ALPHA) not in (int, float) or not math.isfinite(STATISTICAL_ALPHA) or not (0.0 < STATISTICAL_ALPHA < 1.0):
        raise ValueError(f"STATISTICAL_ALPHA phải là số thực hữu hạn trong (0, 1), nhận: {STATISTICAL_ALPHA}")


# Tự động xác thực cấu hình ngay khi module được nạp
validate_config()
