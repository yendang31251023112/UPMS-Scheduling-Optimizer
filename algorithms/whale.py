import time
import numpy as np


def clamping_floor_mapping(position: np.ndarray, num_machines: int) -> np.ndarray:
    """Rời rạc hóa vector vị trí thực về chỉ số máy nguyên [0, M - 1]."""
    clamped = np.clip(position, 0.0, num_machines - 1e-9)
    return np.floor(clamped).astype(int)


def run_woa(problem, params: dict | None = None, evaluator=None, callback=None) -> dict:
    """
    Chạy WOA nguyên bản (Mirjalili & Lewis, 2016) cho bài toán UPMS.
    Mặc định: pop_size=50, max_nfe=10000 -> 199 vòng lặp + 1 khởi tạo = 200 điểm convergence.
    """
    params = params or {}
    pop_size = int(params.get("pop_size", 50))
    max_nfe = int(params.get("max_nfe", 10000))
    b = float(params.get("b", 1.0))
    seed = params.get("seed", None)
    initial_population = params.get("initial_population", None)

    rng = np.random.default_rng(seed)
    n_jobs, n_machines = problem.processing_times.shape
    max_iter = max_nfe // pop_size - 1  # 50 * (1 + 199) = 10.000 NFE

    eval_count = 0

    def evaluate(assignment: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        if evaluator is not None:
            return float(evaluator.evaluate(assignment))
        return float(problem.calculate_makespan(assignment))

    start_time = time.perf_counter()

    # 1. Khởi tạo quần thể liên tục trong [0, M)
    if initial_population is not None:
        positions = np.array(initial_population, dtype=float, copy=True)
    else:
        positions = rng.uniform(0.0, float(n_machines), size=(pop_size, n_jobs))

    fitness = np.empty(pop_size, dtype=float)
    best_pos = None
    best_assignment = None
    best_makespan = float("inf")

    for i in range(pop_size):
        assign = clamping_floor_mapping(positions[i], n_machines)
        val = evaluate(assign)
        fitness[i] = val
        if val < best_makespan:
            best_makespan = val
            best_pos = positions[i].copy()
            best_assignment = assign.copy()

    convergence = [best_makespan]  # convergence[0]: Best sau khởi tạo

    # 2. Vòng lặp chính WOA nguyên bản
    for t in range(1, max_iter + 1):
        a = 2.0 - 2.0 * (t / max_iter)  # Giảm tuyến tính từ 2 về 0

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

            # Clamping về miền [0, M) và rời rạc hóa
            positions[i] = np.clip(new_pos, 0.0, n_machines - 1e-9)
            assign = clamping_floor_mapping(positions[i], n_machines)
            val = evaluate(assign)
            fitness[i] = val

            if val < best_makespan:
                best_makespan = val
                best_pos = positions[i].copy()
                best_assignment = assign.copy()

        convergence.append(best_makespan)
        if callback is not None:
            callback(t, best_makespan)

    runtime = time.perf_counter() - start_time
    schedule = problem.build_schedule(best_assignment)

    return {
        "assignment": best_assignment.tolist(),
        "makespan": float(best_makespan),
        "schedule": schedule,
        "convergence": convergence,
        "runtime": float(runtime),
        "evaluation_count": eval_count,
    }


# --- Phần chạy thử console trên bộ dữ liệu Small (10x3) ---
class _TempUPMSProblem:
    """Lớp tạm để test độc lập; khi ghép repo sẽ dùng core.problem.Problem của Yến."""

    def __init__(self, processing_times: np.ndarray):
        self.processing_times = np.asarray(processing_times, dtype=float)

    def calculate_makespan(self, assignment: np.ndarray) -> float:
        n_jobs, n_machines = self.processing_times.shape
        loads = np.zeros(n_machines, dtype=float)
        for job_id in range(n_jobs):
            m = int(assignment[job_id])
            loads[m] += self.processing_times[job_id, m]
        return float(np.max(loads))

    def build_schedule(self, assignment: np.ndarray) -> list[tuple]:
        n_jobs, n_machines = self.processing_times.shape
        machine_time = np.zeros(n_machines, dtype=float)
        schedule = []
        for job_id in range(n_jobs):  # Duyệt Job ID tăng dần
            m = int(assignment[job_id])
            start = machine_time[m]
            end = start + float(self.processing_times[job_id, m])
            schedule.append((m, job_id, start, end))
            machine_time[m] = end
        return schedule


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    P_small = rng.integers(1, 51, size=(10, 3)).astype(float)
    problem = _TempUPMSProblem(P_small)

    res = run_woa(problem, params={"pop_size": 50, "max_nfe": 10000, "b": 1.0, "seed": 42})

    print("=== KẾT QUẢ CHẠY THỬ WOA PROTOTYPE (SMALL 10x3) ===")
    print(f"Best Makespan   : {res['makespan']:.2f}")
    print(f"Assignment      : {res['assignment']}")
    print(f"Evaluation Count: {res['evaluation_count']}")
    print(f"Convergence Len : {len(res['convergence'])} (Start: {res['convergence'][0]:.2f} -> End: {res['convergence'][-1]:.2f})")
    print(f"Runtime         : {res['runtime']:.4f} s")

    # Unit test nhanh trước khi tạo PR
    assert res["evaluation_count"] == 10000, "Sai ngân sách NFE!"
    assert len(res["convergence"]) == 200, "Sai độ dài convergence!"
    assert all(0 <= m < 3 for m in res["assignment"]), "Chỉ số máy vượt biên!"
    print("Unit test cơ bản: PASSED")
