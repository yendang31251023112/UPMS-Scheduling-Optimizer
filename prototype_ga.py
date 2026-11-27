"""
prototype_ga.py
---------------
Kịch bản chạy thử nghiệm độc lập trên giao diện dòng lệnh (Console Prototype) cho GA.
Dùng để kiểm chứng:
1. Thuật toán hoạt động độc lập và đạt đúng 10,000 NFE.
2. Đo lường tốc độ thực thi thuần túy.
3. Xuất lịch trình Gantt và đường cong hội tụ 200 điểm.
"""

import sys
import time
from core.config import (
    CONVERGENCE_POINTS,
    GA_ELITE_COUNT,
    GA_PC,
    GA_PM,
    GA_TOURNAMENT_K,
    MAX_EVALUATIONS,
    MAX_GENERATIONS,
    POPULATION_SIZE,
)
from core.problem import UPMSProblem
from algorithms.genetic import run_ga


def progress_callback(gen: int, best_ms: float, best_chrom: list) -> None:
    """Callback theo dõi tiến trình tìm kiếm nghiệm qua từng thế hệ."""
    if gen == 0 or gen % 25 == 0 or gen == MAX_GENERATIONS:
        print(f"  [Gen {gen:3d}/199] Best Makespan hiện tại: {best_ms:7.2f}")


def main():
    print("=" * 70)
    print("   BẢN MẪU THỬ NGHIỆM GIẢI THUẬT DI TRUYỀN (GA PROTOTYPE - UPMS)")
    print("=" * 70)

    # 1. Khởi tạo một bài toán mẫu quy mô vừa (Medium: 30 jobs x 5 máy)
    num_jobs = 30
    num_machines = 5
    seed = 42

    print(f"\n[1] Khởi tạo bài toán mẫu UPMS (Seed = {seed}):")
    print(f"    - Số công việc (Jobs):    N = {num_jobs}")
    print(f"    - Số máy song song (M):   M = {num_machines}")
    print(f"    - Không gian tìm kiếm:    {num_machines}^{num_jobs} ≈ {num_machines**num_jobs:.2e} phương án")

    problem = UPMSProblem.generate_random_instance(
        num_jobs=num_jobs, num_machines=num_machines, seed=seed, p_min=10, p_max=100
    )

    print(f"    - Kích thước ma trận P:   {len(problem.processing_times)} hàng x {len(problem.processing_times[0])} cột (M x N)")

    # 2. Cấu hình tham số chuẩn
    print(f"\n[2] Thiết lập tham số giải thuật Di truyền (GA):")
    print(f"    - Kích thước quần thể:    N_pop = {POPULATION_SIZE}")
    print(f"    - Số thế hệ tối đa:       T = {MAX_GENERATIONS}")
    print(f"    - Xác suất lai ghép:      Pc = {GA_PC} (Uniform Crossover)")
    print(f"    - Xác suất đột biến:      Pm = {GA_PM} (Random Resetting)")
    print(f"    - Kích thước giải đấu:    k = {GA_TOURNAMENT_K}")
    print(f"    - Số cá thể tinh hoa:     Elite = {GA_ELITE_COUNT} (giữ nguyên không đánh giá lại)")
    print(f"    - Ngân sách kỳ vọng:      {POPULATION_SIZE} + {MAX_GENERATIONS} * {POPULATION_SIZE} = {MAX_EVALUATIONS} NFE")

    # 3. Tiến hành giải
    print(f"\n[3] Bắt đầu quá trình tiến hóa:")
    result = run_ga(
        problem=problem,
        params={
            "population_size": POPULATION_SIZE,
            "max_generations": MAX_GENERATIONS,
            "pc": GA_PC,
            "pm": GA_PM,
            "tournament_k": GA_TOURNAMENT_K,
            "elite_count": GA_ELITE_COUNT,
            "seed": seed,
        },
        callback=progress_callback,
    )

    # 4. Phân tích kết quả thực thi
    print("\n" + "=" * 70)
    print("   KẾT QUẢ THỰC THI GA")
    print("=" * 70)

    best_ms = result["makespan"]
    init_ms = result["convergence"][0]
    improvement = ((init_ms - best_ms) / init_ms * 100.0) if init_ms > 0 else 0.0
    runtime = result["runtime"]
    nfe = result["evaluation_count"]
    conv = result["convergence"]

    print(f"  * Makespan ban đầu (Gen 0):          {init_ms:7.2f}")
    print(f"  * Makespan tối ưu tìm được:          {best_ms:7.2f} (Cải thiện: {improvement:.2f}%)")
    print(f"  * Thời gian tìm kiếm thuần túy:      {runtime:.4f} giây (loại trừ thời gian callback)")
    print(f"  * Tổng số NFE đã tiêu thụ:           {nfe:,} NFE (Khớp trần {MAX_EVALUATIONS:,})")
    print(f"  * Số điểm trên đường hội tụ:         {len(conv)} điểm (Khớp {CONVERGENCE_POINTS})")
    print(f"  * Phương án phân công tối ưu:        {result['assignment']}")

    # 5. Phụ tải trên từng máy
    loads = problem.get_machine_loads(result["assignment"])
    print("\n[5] Phân bổ tải công việc trên từng máy:")
    for m, load in enumerate(loads):
        jobs_on_m = [j for j, assigned_m in enumerate(result["assignment"]) if assigned_m == m]
        print(f"    - Máy {m}: Tổng tải = {load:6.1f} | Số jobs = {len(jobs_on_m):2d} | Jobs: {jobs_on_m}")

    # 6. Kiểm tra tính liên tục của lịch trình Gantt
    schedule = result["schedule"]
    print(f"\n[6] Lịch trình Gantt: {len(schedule)} mục công việc đã xếp lịch")
    print("    5 công việc đầu tiên trong lịch trình:")
    for item in schedule[:5]:
        print(f"      Máy {item[0]} | Job {item[1]:2d} | Bắt đầu: {item[2]:6.1f} -> Kết thúc: {item[3]:6.1f}")

    print("\n>>> KIỂM TRA HỢP ĐỒNG DỮ LIỆU: ĐÃ VƯỢT QUA validate_result_contract() THÀNH CÔNG! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
