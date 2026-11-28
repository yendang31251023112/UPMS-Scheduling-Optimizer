"""
fake_data.py - DỮ LIỆU GIẢ để tui vẽ thử biểu đồ khi chưa có code thật của Yến/Tuấn/Ngân.
Khi ghép code thật thì KHÔNG cần file này nữa (chỉ dùng để demo).

Trả về dict `result` đúng theo "hợp đồng dữ liệu" của nhóm:
    assignment, makespan, schedule, convergence (200 phần tử), runtime, evaluation_count
"""
import random
import numpy as np


def make_fake_matrix(n_jobs=10, n_machines=3, seed=0):
    """Ma trận thời gian xử lý P (n_jobs dòng x n_machines cột).
    P[j][m] = thời gian máy m xử lý công việc j."""
    rng = np.random.default_rng(seed)
    return rng.integers(5, 50, size=(n_jobs, n_machines)).astype(float)


def build_schedule(P, assignment):
    """Dựng lịch: mỗi máy chạy các job được giao theo Job ID tăng dần.
    Trả về list (machine_id, job_id, start, end), sắp theo job_id tăng dần."""
    n_machines = P.shape[1]
    finish = [0.0] * n_machines          # thời điểm máy m rảnh
    schedule = []
    for job_id, m in enumerate(assignment):   # job_id chạy từ 0 -> N-1
        start = finish[m]
        end = start + P[job_id][m]
        schedule.append((m, job_id, start, end))
        finish[m] = end
    return schedule, max(finish)         # max(finish) chính là Makespan


def make_fake_result(P, seed=0):
    """Giả lập kết quả của một thuật toán: phân công ngẫu nhiên + đường hội tụ giảm dần."""
    random.seed(seed)
    n_jobs, n_machines = P.shape
    # Phân công "tốt tương đối": mỗi job chọn máy xử lý nhanh nhất, rồi xáo trộn nhẹ
    assignment = [int(np.argmin(P[j])) for j in range(n_jobs)]
    for _ in range(max(1, n_jobs // 5)):
        assignment[random.randrange(n_jobs)] = random.randrange(n_machines)

    schedule, makespan = build_schedule(P, assignment)

    # Đường hội tụ: đúng 200 phần tử, giảm dần (không bao giờ tăng) và kết thúc ở makespan
    start_value = makespan * 1.8
    convergence = []
    best = start_value
    for i in range(200):
        target = makespan + (start_value - makespan) * (1 - i / 199) ** 2
        best = min(best, target + random.uniform(0, makespan * 0.05))
        convergence.append(best)
    convergence[-1] = makespan

    return {
        "assignment": assignment,
        "makespan": float(makespan),
        "schedule": schedule,
        "convergence": convergence,
        "runtime": round(random.uniform(0.5, 1.5), 3),
        "evaluation_count": 10000,
    }
