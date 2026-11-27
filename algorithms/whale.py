"""
algorithms/whale.py
-------------------
Cài đặt Thuật toán Tối ưu hóa Bầy Cá voi nguyên bản (Whale Optimization Algorithm - WOA)
theo Mirjalili & Lewis (2016), kết hợp cơ chế rời rạc hóa Clamping & Floor Mapping
cho bài toán Lập lịch máy song song không đồng nhất (UPMS - R || Cmax).
"""

import os
import sys
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

# Hỗ trợ chạy trực tiếp file algorithms/whale.py từ VS Code hoặc console
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    MAX_EVALUATIONS,
    MAX_GENERATIONS,
    POPULATION_SIZE,
    WOA_A_MAX,
    WOA_A_MIN,
    WOA_B,
)
from core.evaluator import Evaluator
from core.problem import UPMSProblem, validate_result_contract


def clamping_floor_mapping(position: np.ndarray, num_machines: int) -> np.ndarray:
    """Rời rạc hóa vector vị trí thực về chỉ số máy nguyên trong [0, M - 1]."""
    clamped = np.clip(position, 0.0, num_machines - 1e-9)
    return np.floor(clamped).astype(int)


def run_woa(
    problem: UPMSProblem,
    params: Optional[Dict[str, Any]] = None,
    evaluator: Optional[Evaluator] = None,
    callback: Optional[Callable[[int, float, list], None]] = None,
) -> Dict[str, Any]:
    """
    Chạy WOA nguyên bản (Mirjalili & Lewis, 2016) cho bài toán UPMS.
    Mặc định: population_size=50, max_generations=199 -> 10.000 NFE và 200 điểm convergence.
    """
    params = params or {}
    pop_size = int(params.get("population_size", params.get("pop_size", POPULATION_SIZE)))
    if "max_generations" in params:
        max_iter = int(params["max_generations"])
        default_nfe = pop_size * (1 + max_iter)
        max_nfe = int(params.get("max_evaluations", params.get("max_nfe", default_nfe)))
    else:
        max_nfe = int(params.get("max_evaluations", params.get("max_nfe", MAX_EVALUATIONS)))
        max_iter = max_nfe // pop_size - 1

    b = float(params.get("b", WOA_B))
    a_max = float(params.get("a_max", WOA_A_MAX))
    a_min = float(params.get("a_min", WOA_A_MIN))
    seed = params.get("seed", None)
    initial_population = params.get("initial_population", None)

    n_jobs = problem.num_jobs
    n_machines = problem.num_machines

    if evaluator is None:
        evaluator = Evaluator(problem, max_evaluations=max_nfe)
    else:
        if evaluator.evaluation_count != 0:
            raise ValueError(
                f"Evaluator truyền vào đã được sử dụng (evaluation_count={evaluator.evaluation_count}, yêu cầu == 0)."
            )
        if evaluator.problem is not problem:
            raise ValueError("Evaluator truyền vào không gắn với đúng đối tượng UPMSProblem hiện tại.")
        if evaluator.max_evaluations != max_nfe:
            raise ValueError(
                f"Ngân sách của Evaluator ({evaluator.max_evaluations}) không khớp với cấu hình WOA ({max_nfe})."
            )

    rng = np.random.default_rng(seed)

    def evaluate(assign: np.ndarray) -> float:
        return float(evaluator.evaluate(assign.tolist()))

    callback_time = 0.0
    start_time = time.perf_counter()

    # 1. Khởi tạo quần thể liên tục trong [0, M)
    if initial_population is not None:
        positions = np.array(initial_population, dtype=float, copy=True)
    else:
        positions = rng.uniform(0.0, float(n_machines), size=(pop_size, n_jobs))

    best_pos = None
    best_assignment = None
    best_makespan = float("inf")

    for i in range(pop_size):
        assign = clamping_floor_mapping(positions[i], n_machines)
        val = evaluate(assign)
        if val < best_makespan:
            best_makespan = val
            best_pos = positions[i].copy()
            best_assignment = assign.copy()

    convergence = [best_makespan]  # convergence[0]: Best sau khởi tạo

    if callback is not None:
        cb_start = time.perf_counter()
        callback(0, best_makespan, best_assignment.tolist())
        callback_time += time.perf_counter() - cb_start

    # 2. Vòng lặp chính WOA nguyên bản
    for t in range(1, max_iter + 1):
        a = a_max - (a_max - a_min) * (t / max_iter)  # Giảm tuyến tính từ 2 về 0

        for i in range(pop_size):
            r1 = rng.random(n_jobs)
            r2 = rng.random(n_jobs)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            l = rng.uniform(-1.0, 1.0, size=n_jobs)

            if p < 0.5:
                if np.abs(A).mean() < 1.0:
                    # Bao vây con mồi (Encircling prey)
                    D = np.abs(C * best_pos - positions[i])
                    new_pos = best_pos - A * D
                else:
                    # Tìm kiếm con mồi ngẫu nhiên (Search for prey)
                    rand_idx = rng.integers(0, pop_size)
                    X_rand = positions[rand_idx]
                    D = np.abs(C * X_rand - positions[i])
                    new_pos = X_rand - A * D
            else:
                # Tấn công theo đường xoắn ốc (Spiral updating)
                D_prime = np.abs(best_pos - positions[i])
                new_pos = D_prime * np.exp(b * l) * np.cos(2.0 * np.pi * l) + best_pos

            # Clamping về miền [0, M) và rời rạc hóa về [0, M - 1]
            positions[i] = np.clip(new_pos, 0.0, n_machines - 1e-9)
            assign = clamping_floor_mapping(positions[i], n_machines)
            val = evaluate(assign)

            if val < best_makespan:
                best_makespan = val
                best_pos = positions[i].copy()
                best_assignment = assign.copy()

        convergence.append(best_makespan)

        if callback is not None:
            cb_start = time.perf_counter()
            callback(t, best_makespan, best_assignment.tolist())
            callback_time += time.perf_counter() - cb_start

    runtime = max(0.0, (time.perf_counter() - start_time) - callback_time)
    best_assign_list = best_assignment.tolist()
    schedule = problem.build_schedule(best_assign_list)

    result = {
        "assignment": best_assign_list,
        "makespan": float(best_makespan),
        "schedule": schedule,
        "convergence": convergence,
        "runtime": float(runtime),
        "evaluation_count": evaluator.evaluation_count,
    }

    validate_result_contract(
        result,
        problem,
        expected_convergence_len=1 + max_iter,
        max_evaluations=max_nfe,
    )
    return result


if __name__ == "__main__":
    problem = UPMSProblem.generate_random_instance(
        num_jobs=10, num_machines=3, seed=42
    )

    def progress_callback(gen: int, best_ms: float, best_assign: list) -> None:
        if gen == 0 or gen % 50 == 0 or gen == MAX_GENERATIONS:
            print(f"  [Gen {gen:3d}/{MAX_GENERATIONS}] Best Makespan: {best_ms:7.2f}")

    print("=== KẾT QUẢ CHẠY THỬ WOA PROTOTYPE (SMALL 10x3 - TÍCH HỢP CORE) ===")
    res = run_woa(
        problem,
        params={
            "population_size": POPULATION_SIZE,
            "max_generations": MAX_GENERATIONS,
            "max_evaluations": MAX_EVALUATIONS,
            "b": WOA_B,
            "seed": 42,
        },
        callback=progress_callback,
    )

    print(f"Best Makespan   : {res['makespan']:.2f}")
    print(f"Assignment      : {res['assignment']}")
    print(f"Evaluation Count: {res['evaluation_count']}")
    print(f"Convergence Len : {len(res['convergence'])} (Start: {res['convergence'][0]:.2f} -> End: {res['convergence'][-1]:.2f})")
    print(f"Runtime         : {res['runtime']:.4f} s (đã loại trừ thời gian callback)")
    print("validate_result_contract(): PASSED")
