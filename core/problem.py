"""
Mô hình bài toán Lập lịch máy song song không đồng nhất (UPMS - R || Cmax).
Định nghĩa cấu trúc bài toán, tính toán Makespan và xây dựng lịch trình chi tiết (Schedule) cho biểu đồ Gantt.
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from .config import (
    CONVERGENCE_POINTS,
    MAX_EVALUATIONS,
    PROCESSING_TIME_MIN,
    PROCESSING_TIME_MAX,
)


class UPMSProblem:
    """
    Biểu diễn bài toán Unrelated Parallel Machine Scheduling (UPMS - R || Cmax).

    Quy chuẩn kiến trúc:
        - Ma trận thời gian xử lý P có kích thước cố định (M, N).
        - Hàng i là máy i (0 <= i < M).
        - Cột j là công việc j (0 <= j < N).
        - P[i][j] = p_ij là thời gian máy i xử lý công việc j.

    Thuộc tính:
        num_jobs (int): Số lượng công việc N (J_0, ..., J_{N-1}).
        num_machines (int): Số lượng máy song song không đồng nhất M (M_0, ..., M_{M-1}).
        processing_times (List[List[float]]): Ma trận P kích thước (M, N).
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        processing_times: Any,
    ) -> None:
        """
        Khởi tạo bài toán UPMS.

        Args:
            num_jobs: Số lượng công việc (N), bắt buộc là số nguyên dương (int > 0).
            num_machines: Số lượng máy (M), bắt buộc là số nguyên dương (int > 0).
            processing_times: Ma trận P kích thước chuẩn (M, N).
        """
        # Kiểm tra chặt chẽ kiểu dữ liệu của num_jobs và num_machines (loại trừ bool và float)
        if type(num_jobs) is not int or num_jobs <= 0:
            raise TypeError(
                f"num_jobs phải là số nguyên dương (int > 0), nhận được: {num_jobs} (kiểu {type(num_jobs).__name__})"
            )
        if type(num_machines) is not int or num_machines <= 0:
            raise TypeError(
                f"num_machines phải là số nguyên dương (int > 0), nhận được: {num_machines} (kiểu {type(num_machines).__name__})"
            )

        self.num_jobs: int = num_jobs
        self.num_machines: int = num_machines

        # Kiểm tra và chuẩn hóa ma trận thời gian xử lý theo chuẩn duy nhất (M, N)
        self.processing_times: List[List[float]] = self._validate_and_normalize_matrix(
            processing_times, self.num_machines, self.num_jobs
        )

    @staticmethod
    def _validate_and_normalize_matrix(
        raw_matrix: Any,
        expected_m: int,
        expected_n: int,
    ) -> List[List[float]]:
        """
        Kiểm tra toàn diện tính hợp lệ của ma trận thời gian xử lý đầu vào:
        - Không được là None hoặc rỗng.
        - Số hàng phải đúng bằng M (số máy).
        - Mỗi hàng phải có đúng N phần tử (không cho phép jagged array / hàng ngắn hàng dài).
        - Mọi phần tử phải là số thực hoặc số nguyên (loại trừ boolean), không được là NaN hoặc Inf.
        - Mọi phần tử phải >= 0.

        Quy định kiến trúc: Chỉ chấp nhận định dạng chuẩn duy nhất (M, N).
        """
        if raw_matrix is None:
            raise ValueError("Ma trận processing_times không được để trống (None).")

        # Chuyển đổi từ numpy ndarray nếu có
        if _HAS_NUMPY and isinstance(raw_matrix, np.ndarray):
            if raw_matrix.ndim != 2:
                raise ValueError(
                    f"Ma trận numpy phải là mảng 2 chiều, nhận được mảng {raw_matrix.ndim} chiều."
                )
            matrix_list = raw_matrix.tolist()
        elif isinstance(raw_matrix, (list, tuple)):
            matrix_list = [list(row) if isinstance(row, (list, tuple)) else row for row in raw_matrix]
        else:
            raise TypeError(
                f"Ma trận phải là list, tuple hoặc numpy.ndarray, nhận được: {type(raw_matrix).__name__}"
            )

        rows = len(matrix_list)
        if rows == 0:
            raise ValueError("Ma trận processing_times không được rỗng (0 hàng).")

        if rows != expected_m:
            raise ValueError(
                f"Số hàng của ma trận ({rows}) không khớp với số máy M={expected_m}. "
                f"Đồ án quy định duy nhất chuẩn (num_machines, num_jobs) = ({expected_m}, {expected_n})."
            )

        normalized: List[List[float]] = []

        for i, row in enumerate(matrix_list):
            if not isinstance(row, (list, tuple)):
                raise TypeError(
                    f"Hàng {i} trong ma trận không phải là một danh sách hợp lệ, nhận kiểu: {type(row).__name__}"
                )
            cols = len(row)
            if cols != expected_n:
                raise ValueError(
                    f"Hàng {i} (máy {i}) có {cols} cột, không khớp với số công việc N={expected_n}."
                )

            normalized_row: List[float] = []
            for j, val in enumerate(row):
                # Loại trừ kiểu bool (vì trong Python, isinstance(True, int) là True)
                if type(val) is bool or not isinstance(val, (int, float)):
                    raise TypeError(
                        f"Phần tử tại [{i}][{j}] không phải là số hợp lệ: {val} (kiểu {type(val).__name__})"
                    )

                # Kiểm tra NaN và Infinity
                float_val = float(val)
                if math.isnan(float_val):
                    raise ValueError(f"Phần tử tại [{i}][{j}] mang giá trị NaN (không xác định).")
                if math.isinf(float_val):
                    raise ValueError(f"Phần tử tại [{i}][{j}] mang giá trị vô cùng (Infinity).")
                if float_val < 0.0:
                    raise ValueError(f"Thời gian xử lý không được âm: P[{i}][{j}] = {float_val}")

                normalized_row.append(float_val)

            normalized.append(normalized_row)

        return normalized

    def _validate_assignment(self, assignment: Any) -> List[int]:
        """
        Xác thực nghiêm ngặt phương án phân công (assignment):
        - Không được là None.
        - Độ dài phải đúng bằng num_jobs (N).
        - Mọi phần tử bắt buộc phải là số nguyên (int), tuyệt đối không chấp nhận bool hoặc float.
        - Mọi chỉ số máy phải nằm trong khoảng [0, num_machines - 1].
        """
        if assignment is None:
            raise ValueError("Phương án assignment không được là None.")

        # Xử lý numpy array
        if _HAS_NUMPY and isinstance(assignment, np.ndarray):
            if assignment.ndim != 1:
                raise ValueError(
                    f"Mảng assignment phải là 1 chiều, nhận được {assignment.ndim} chiều."
                )
            if assignment.dtype == bool or not np.issubdtype(assignment.dtype, np.integer):
                raise TypeError(
                    f"Mảng numpy assignment phải có dtype là số nguyên, nhận được: {assignment.dtype}."
                )
            raw_list = assignment.tolist()
        elif isinstance(assignment, (list, tuple)):
            raw_list = list(assignment)
        else:
            raise TypeError(
                f"Assignment phải là list, tuple hoặc numpy.ndarray 1D, nhận: {type(assignment).__name__}"
            )

        if len(raw_list) != self.num_jobs:
            raise ValueError(
                f"Độ dài assignment ({len(raw_list)}) không khớp số công việc N={self.num_jobs}."
            )

        clean_assignment: List[int] = []
        for j, m in enumerate(raw_list):
            # Kiểm tra kiểu nghiêm ngặt (loại trừ bool và float)
            if type(m) is bool or type(m) is not int:
                raise TypeError(
                    f"Chỉ số máy tại job {j} phải là số nguyên (int), "
                    f"nhận được: {m} (kiểu {type(m).__name__}). Không được truyền số thực hoặc boolean."
                )
            if m < 0 or m >= self.num_machines:
                raise ValueError(
                    f"Chỉ số máy không hợp lệ tại job {j}: gán máy {m}, "
                    f"miền hợp lệ là [0, {self.num_machines - 1}]."
                )
            clean_assignment.append(m)

        return clean_assignment

    def calculate_makespan(self, assignment: Any) -> float:
        """
        Tính Makespan (C_max) cho một phương án phân công:
            C_max = max_{0 <= i < M} ( sum_{j: assignment[j] == i} p_{i, j} )

        Args:
            assignment: Danh sách/mảng gồm N phần tử nguyên, assignment[j] in [0, M-1].

        Returns:
            float: Makespan của phương án phân công (hữu hạn và >= 0).
        """
        loads = self.get_machine_loads(assignment)
        makespan = max(loads)
        if not math.isfinite(makespan) or makespan < 0.0:
            raise ValueError(f"Makespan tính toán không hợp lệ: {makespan}")
        return float(makespan)

    def get_machine_loads(self, assignment: Any) -> List[float]:
        """
        Tính tổng thời gian tải làm việc (workload) trên từng máy:
            Load_i = sum_{j: assignment[j] == i} p_{i, j}

        Args:
            assignment: Danh sách gồm N phần tử nguyên trong [0, M-1].

        Returns:
            List[float]: Danh sách độ dài M chứa tổng thời gian làm việc của từng máy.
        """
        clean_assignment = self._validate_assignment(assignment)
        loads = [0.0] * self.num_machines
        for j, m in enumerate(clean_assignment):
            loads[m] += self.processing_times[m][j]
        return loads

    def build_schedule(self, assignment: Any) -> List[Tuple[int, int, float, float]]:
        """
        Xây dựng lịch trình thực hiện chi tiết cho từng máy để hiển thị biểu đồ Gantt.

        Quy chuẩn kỹ thuật:
        - Danh sách gồm các tuple theo thứ tự: Máy 0 (jobs tăng dần), Máy 1 (jobs tăng dần), ...
        - Trên từng máy, các công việc được sắp xếp thực hiện theo thứ tự Job ID tăng dần.
        - Mỗi job bắt đầu ngay khi job trước đó trên cùng máy hoàn thành (start_time = previous_end_time).
        - Thời điểm bắt đầu của job đầu tiên trên máy là 0.0.

        Args:
            assignment: Danh sách phân công máy cho N công việc.

        Returns:
            List[Tuple[int, int, float, float]]: Danh sách các khoảng thời gian thực thi:
                (machine_id, job_id, start_time, end_time)
        """
        clean_assignment = self._validate_assignment(assignment)

        # Gom nhóm các job được gán cho từng máy
        machine_to_jobs: List[List[int]] = [[] for _ in range(self.num_machines)]
        for j, m in enumerate(clean_assignment):
            machine_to_jobs[m].append(j)

        schedule: List[Tuple[int, int, float, float]] = []

        for m in range(self.num_machines):
            jobs_on_m = machine_to_jobs[m]
            # Quy chuẩn 10: Sắp xếp theo Job ID tăng dần
            jobs_on_m.sort()

            current_time = 0.0
            for j in jobs_on_m:
                duration = self.processing_times[m][j]
                start_time = current_time
                end_time = start_time + duration
                schedule.append((m, j, start_time, end_time))
                current_time = end_time

        return schedule

    @classmethod
    def generate_random_instance(
        cls,
        num_jobs: int,
        num_machines: int,
        seed: Optional[int] = None,
        p_min: int = PROCESSING_TIME_MIN,
        p_max: int = PROCESSING_TIME_MAX,
    ) -> "UPMSProblem":
        """
        Tạo nhanh một bài toán UPMS ngẫu nhiên (dùng cho kiểm thử hoặc demo).
        Hai lần gọi cùng seed đảm bảo tái lập 100% cùng một ma trận.

        Args:
            num_jobs: Số lượng công việc (bắt buộc int > 0).
            num_machines: Số lượng máy (bắt buộc int > 0).
            seed: Seed ngẫu nhiên (bắt buộc int hoặc None, loại trừ bool).
            p_min: Thời gian xử lý nhỏ nhất (int > 0).
            p_max: Thời gian xử lý lớn nhất (int >= p_min).

        Returns:
            UPMSProblem: Đối tượng bài toán với ma trận P kích thước (M, N).
        """
        # Kiểm tra tính hợp lệ của tham số TRƯỚC KHI sinh ma trận
        if type(num_jobs) is not int or num_jobs <= 0:
            raise TypeError(f"num_jobs phải là số nguyên dương (int > 0), nhận: {num_jobs}")
        if type(num_machines) is not int or num_machines <= 0:
            raise TypeError(f"num_machines phải là số nguyên dương (int > 0), nhận: {num_machines}")
        if seed is not None and (type(seed) is bool or type(seed) is not int):
            raise TypeError(f"seed phải là số nguyên (int) hoặc None, nhận: {seed} (kiểu {type(seed).__name__})")
        if type(p_min) is not int or type(p_max) is not int:
            raise TypeError("p_min và p_max phải là số nguyên (int).")
        if p_min <= 0:
            raise ValueError(f"p_min phải là số nguyên dương (> 0), nhận: {p_min}")
        if p_max < p_min:
            raise ValueError(f"p_max ({p_max}) không được nhỏ hơn p_min ({p_min})")

        rng = random.Random(seed)
        matrix: List[List[float]] = [
            [float(rng.randint(p_min, p_max)) for _ in range(num_jobs)]
            for _ in range(num_machines)
        ]
        return cls(num_jobs=num_jobs, num_machines=num_machines, processing_times=matrix)

    def to_dict(self) -> Dict[str, Any]:
        """Xuất thông tin bài toán sang dictionary để lưu file JSON."""
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "processing_times": self.processing_times,
        }

    def validate_benchmark_bounds(
        self,
        min_time: float = PROCESSING_TIME_MIN,
        max_time: float = PROCESSING_TIME_MAX,
    ) -> None:
        """
        Kiểm tra ma trận thời gian xử lý có tuân thủ đúng quy chuẩn miền giá trị
        thực nghiệm của đồ án [min_time, max_time] hay không.
        Được gọi bởi Dataset Loader hoặc trước khi thực hiện benchmark chính thức.
        """
        if type(min_time) is bool or not isinstance(min_time, (int, float)) or not math.isfinite(min_time):
            raise TypeError(f"min_time phải là số thực hữu hạn, nhận: {min_time} (kiểu {type(min_time).__name__})")
        if type(max_time) is bool or not isinstance(max_time, (int, float)) or not math.isfinite(max_time):
            raise TypeError(f"max_time phải là số thực hữu hạn, nhận: {max_time} (kiểu {type(max_time).__name__})")
        if min_time < 0.0:
            raise ValueError(f"min_time không được âm, nhận: {min_time}")
        if min_time > max_time:
            raise ValueError(f"min_time ({min_time}) không được lớn hơn max_time ({max_time})")

        for i, row in enumerate(self.processing_times):
            for j, val in enumerate(row):
                if not (min_time <= val <= max_time):
                    raise ValueError(
                        f"Thời gian xử lý P[{i}][{j}] = {val} nằm ngoài phạm vi benchmark "
                        f"quy định [{min_time}, {max_time}]."
                    )

    @classmethod
    def from_dict(cls, data: Any) -> "UPMSProblem":
        """
        Khởi tạo bài toán từ dictionary nạp từ file JSON.
        Xác thực cấu trúc schema nghiêm ngặt: đúng 3 trường 'num_jobs', 'num_machines', 'processing_times'.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Dữ liệu khởi tạo bài toán phải là một dictionary, nhận được: {type(data).__name__}")

        required_keys = {"num_jobs", "num_machines", "processing_times"}
        actual_keys = set(data.keys())
        if actual_keys != required_keys:
            missing = required_keys - actual_keys
            extra = actual_keys - required_keys
            msg_parts = []
            if missing:
                msg_parts.append(f"thiếu các trường {sorted(missing)}")
            if extra:
                msg_parts.append(f"chứa các trường thừa {sorted(extra)}")
            raise ValueError(f"Dữ liệu bài toán không đúng schema: {' và '.join(msg_parts)}")

        return cls(
            num_jobs=data["num_jobs"],
            num_machines=data["num_machines"],
            processing_times=data["processing_times"],
        )

    def __repr__(self) -> str:
        return f"<UPMSProblem: {self.num_jobs} jobs x {self.num_machines} machines>"


def validate_result_contract(
    result: Dict[str, Any],
    problem: UPMSProblem,
    expected_convergence_len: int = CONVERGENCE_POINTS,
    max_evaluations: int = MAX_EVALUATIONS,
) -> None:
    """
    Hàm kiểm tra toàn diện hợp đồng dữ liệu kết quả (result dictionary) trả về từ run_ga() hoặc run_woa().
    Đảm bảo tính nhất quán tuyệt đối giữa GA và WOA trước khi đưa vào GUI hoặc phân tích thống kê.

    Kiểm tra 6 khía cạnh cốt lõi:
    1. Tham số kiểm tra: expected_convergence_len (int > 0), max_evaluations (int > 0).
    2. Cấu trúc trường bắt buộc: "assignment", "makespan", "schedule", "convergence", "runtime", "evaluation_count".
    3. "assignment": Hợp lệ theo mô hình problem (độ dài N, máy trong [0, M-1]).
    4. "makespan": Số thực hữu hạn >= 0, không phải bool, khớp chính xác với tính lại từ assignment.
    5. "runtime": Số thực hữu hạn >= 0, không phải bool.
    6. "evaluation_count": Số nguyên int trong khoảng (0, max_evaluations].
    7. "convergence": Đúng expected_convergence_len điểm (200), không bool, không NaN/Inf, không âm,
       thỏa mãn tính chất Best-so-far lũy kế không tăng (f_{t+1} <= f_t + 1e-9), và điểm cuối cùng khớp với makespan.
    8. "schedule": Khớp tuyệt đối với kết quả chuẩn sinh ra từ problem.build_schedule(assignment)
       về số lượng, kiểu dữ liệu nguyên (chặn bool/float trong machine_id/job_id), thứ tự máy,
       thứ tự Job ID tăng dần, thời gian bắt đầu/kết thúc hữu hạn và tính liên tục thời gian.
    """
    # 0. Xác thực tham số cấu hình validation
    if type(expected_convergence_len) is not int or expected_convergence_len <= 0:
        raise ValueError(
            f"expected_convergence_len phải là số nguyên dương (int > 0), nhận: {expected_convergence_len}"
        )
    if type(max_evaluations) is not int or max_evaluations <= 0:
        raise ValueError(
            f"max_evaluations phải là số nguyên dương (int > 0), nhận: {max_evaluations}"
        )

    if not isinstance(result, dict):
        raise TypeError(f"Kết quả phải là một dictionary, nhận được: {type(result).__name__}")

    required_keys = {"assignment", "makespan", "schedule", "convergence", "runtime", "evaluation_count"}
    actual_keys = set(result.keys())
    if actual_keys != required_keys:
        missing = required_keys - actual_keys
        extra = actual_keys - required_keys
        msg_parts = []
        if missing:
            msg_parts.append(f"thiếu các trường bắt buộc {sorted(missing)}")
        if extra:
            msg_parts.append(f"chứa các trường thừa không được phép {sorted(extra)}")
        raise ValueError(
            f"Result contract vi phạm: {' và '.join(msg_parts)}. "
            f"Kết quả phải chứa chính xác 6 trường: {sorted(required_keys)}"
        )

    # 1. Kiểm tra assignment
    assignment = result["assignment"]
    recalculated_ms = problem.calculate_makespan(assignment)
    clean_assignment = problem._validate_assignment(assignment)

    # 2. Kiểm tra makespan
    makespan = result["makespan"]
    if type(makespan) is bool or not isinstance(makespan, (int, float)):
        raise TypeError(f"makespan phải là số thực hợp lệ, nhận được: {makespan} (kiểu {type(makespan).__name__})")
    float_makespan = float(makespan)
    if not math.isfinite(float_makespan) or float_makespan < 0.0:
        raise ValueError(f"makespan phải là số thực hữu hạn và >= 0, nhận: {float_makespan}")
    if abs(float_makespan - recalculated_ms) > 1e-6:
        raise ValueError(
            f"Makespan trong result ({float_makespan}) không khớp với Makespan tính toán lại từ assignment ({recalculated_ms})"
        )

    # 3. Kiểm tra runtime
    runtime = result["runtime"]
    if type(runtime) is bool or not isinstance(runtime, (int, float)):
        raise TypeError(f"runtime phải là số thực, nhận: {runtime} (kiểu {type(runtime).__name__})")
    float_runtime = float(runtime)
    if not math.isfinite(float_runtime) or float_runtime < 0.0:
        raise ValueError(f"runtime phải là số thực hữu hạn và >= 0, nhận: {float_runtime}")

    # 4. Kiểm tra evaluation_count
    eval_count = result["evaluation_count"]
    if type(eval_count) is not int or type(eval_count) is bool:
        raise TypeError(f"evaluation_count phải là số nguyên int, nhận: {eval_count} (kiểu {type(eval_count).__name__})")
    if eval_count <= 0 or eval_count > max_evaluations:
        raise ValueError(
            f"evaluation_count ({eval_count}) phải là số nguyên trong khoảng (0, {max_evaluations}]"
        )

    # 5. Kiểm tra convergence (Đường cong hội tụ Best-so-far lũy kế)
    convergence = result["convergence"]
    if not isinstance(convergence, (list, tuple)):
        raise TypeError(f"convergence phải là danh sách số thực, nhận: {type(convergence).__name__}")
    if len(convergence) != expected_convergence_len:
        raise ValueError(
            f"convergence phải có đúng {expected_convergence_len} điểm hội tụ, "
            f"nhưng nhận được {len(convergence)} điểm."
        )

    for idx, val in enumerate(convergence):
        if type(val) is bool or not isinstance(val, (int, float)):
            raise TypeError(
                f"Điểm hội tụ tại vị trí {idx} không phải là số: {val} (kiểu {type(val).__name__})"
            )
        f_val = float(val)
        if not math.isfinite(f_val) or f_val < 0.0:
            raise ValueError(f"Điểm hội tụ tại vị trí {idx} phải là số hữu hạn >= 0, nhận: {f_val}")

        # Kiểm tra tính đơn điệu không tăng của Best-so-far lũy kế: f_{t+1} <= f_t
        if idx > 0:
            prev_val = float(convergence[idx - 1])
            if f_val > prev_val + 1e-9:
                raise ValueError(
                    f"Đường cong hội tụ vi phạm tính chất Best-so-far không tăng: "
                    f"tại điểm {idx} ({f_val}) lớn hơn điểm {idx - 1} ({prev_val})"
                )

    # Điểm hội tụ cuối cùng phải khớp với makespan tốt nhất tìm được
    final_conv_val = float(convergence[-1])
    if abs(final_conv_val - float_makespan) > 1e-6:
        raise ValueError(
            f"Điểm hội tụ cuối cùng ({final_conv_val}) không khớp với makespan tốt nhất ({float_makespan})"
        )

    # 6. Kiểm tra schedule đối chiếu với chuẩn build_schedule()
    schedule = result["schedule"]
    if not isinstance(schedule, (list, tuple)):
        raise TypeError(f"schedule phải là danh sách các tuple, nhận: {type(schedule).__name__}")

    expected_schedule = problem.build_schedule(clean_assignment)
    if len(schedule) != len(expected_schedule):
        raise ValueError(
            f"Số lượng mục trong schedule ({len(schedule)}) không khớp với lịch chuẩn ({len(expected_schedule)})"
        )

    # Kiểm tra tính phân biệt của các công việc trong schedule (chặn trùng job)
    raw_job_ids = [item[1] for item in schedule if isinstance(item, (tuple, list)) and len(item) == 4 and type(item[1]) is int]
    if len(raw_job_ids) != len(set(raw_job_ids)):
        seen = set()
        dup = [j for j in raw_job_ids if j in seen or seen.add(j)]
        raise ValueError(f"Schedule chứa công việc bị trùng lặp: job {dup[0]}")

    for idx, (actual, expected) in enumerate(zip(schedule, expected_schedule)):
        if not isinstance(actual, (tuple, list)) or len(actual) != 4:
            raise ValueError(
                f"Mục {idx} trong schedule phải là tuple (machine_id, job_id, start_time, end_time), "
                f"nhưng nhận được: {actual}"
            )
        act_m, act_j, act_start, act_end = actual
        exp_m, exp_j, exp_start, exp_end = expected

        # Kiểm tra kiểu nghiêm ngặt của machine_id và job_id (chặn bool như True==1, chặn float như 0.0==0)
        if type(act_m) is not int:
            raise TypeError(
                f"Mục {idx}: machine_id phải là số nguyên int, nhận: {act_m} (kiểu {type(act_m).__name__})"
            )
        if type(act_j) is not int:
            raise TypeError(
                f"Mục {idx}: job_id phải là số nguyên int, nhận: {act_j} (kiểu {type(act_j).__name__})"
            )

        # Kiểm tra machine_id và job_id có khớp chuẩn không
        if act_m != exp_m or act_j != exp_j:
            raise ValueError(
                f"Schedule không đúng thứ tự chuẩn tại vị trí {idx}: "
                f"nhận (máy {act_m}, job {act_j}), kỳ vọng chuẩn (máy {exp_m}, job {exp_j})"
            )

        # Phân loại lỗi kiểu (TypeError) cho start_time và end_time
        if type(act_start) is bool or not isinstance(act_start, (int, float)):
            raise TypeError(
                f"Mục {idx}: start_time phải là số thực, nhận: {act_start} (kiểu {type(act_start).__name__})"
            )
        if type(act_end) is bool or not isinstance(act_end, (int, float)):
            raise TypeError(
                f"Mục {idx}: end_time phải là số thực, nhận: {act_end} (kiểu {type(act_end).__name__})"
            )

        # Phân loại lỗi giá trị (ValueError) cho NaN / Inf
        f_start = float(act_start)
        f_end = float(act_end)
        if not math.isfinite(f_start):
            raise ValueError(f"Mục {idx}: start_time phải là số hữu hạn, nhận: {act_start}")
        if not math.isfinite(f_end):
            raise ValueError(f"Mục {idx}: end_time phải là số hữu hạn, nhận: {act_end}")

        # Kiểm tra giá trị thời gian khớp chuẩn với build_schedule()
        if abs(f_start - exp_start) > 1e-6:
            raise ValueError(
                f"Schedule sai start_time tại vị trí {idx} (job {act_j} trên máy {act_m}): "
                f"nhận {act_start}, kỳ vọng chuẩn {exp_start}"
            )
        if abs(f_end - exp_end) > 1e-6:
            raise ValueError(
                f"Schedule sai end_time tại vị trí {idx} (job {act_j} trên máy {act_m}): "
                f"nhận {act_end}, kỳ vọng chuẩn {exp_end}"
            )
