"""
Lớp Evaluator quản lý ngân sách đánh giá hàm mục tiêu (NFE - Number of Function Evaluations).
Đảm bảo tính công bằng khoa học tuyệt đối giữa GA, WOA cơ sở:
- Mỗi lần gọi evaluate() sẽ kiểm tra trần ngân sách MAX_EVALUATIONS (mặc định 10,000 NFE).
- Nếu vượt quá ngân sách, tự động ném ngoại lệ RuntimeError để ngăn chặn chạy lố thế hệ.
- Kiểm tra tính hợp lệ của phương án phân công (assignment) trước khi tính Makespan:
  TUYỆT ĐỐI KHÔNG tự ý ép kiểu số thực thành số nguyên (tránh bỏ qua lỗi mapping của WOA).
- Nếu dữ liệu đầu vào không hợp lệ hoặc tính Makespan thất bại,
  ném ngoại lệ ngay lập tức và KHÔNG tiêu hao ngân sách NFE.
"""

import math
from typing import Any, List

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from .config import MAX_EVALUATIONS


class Evaluator:
    """
    Bộ đánh giá độc lập dùng chung cho mọi thuật toán tối ưu hóa bài toán UPMS.

    Attributes:
        problem: Đối tượng bài toán UPMS (phải có hàm calculate_makespan(assignment)).
        max_evaluations (int): Ngân sách số lần đánh giá tối đa cho phép (mặc định 10,000).
        evaluation_count (int): Số lần đánh giá hợp lệ đã thực hiện thực tế.
    """

    def __init__(self, problem: Any, max_evaluations: int = MAX_EVALUATIONS) -> None:
        """
        Khởi tạo một bộ đánh giá mới cho một lượt chạy thuật toán.

        Args:
            problem: Thể hiện của lớp UPMSProblem.
            max_evaluations: Ngân sách số lần đánh giá tối đa (bắt buộc là int > 0, mặc định 10,000).
        """
        if problem is None:
            raise ValueError("Đối tượng problem không được để trống (None).")

        # Kiểm tra chặt chẽ kiểu dữ liệu max_evaluations (loại trừ bool và float)
        if type(max_evaluations) is not int:
            raise TypeError(
                f"max_evaluations phải là số nguyên (int), nhận được: {max_evaluations} (kiểu {type(max_evaluations).__name__})"
            )
        if max_evaluations <= 0:
            raise ValueError(f"max_evaluations phải là số nguyên dương > 0, nhận: {max_evaluations}")

        self.problem = problem
        self.max_evaluations: int = max_evaluations
        self.evaluation_count: int = 0

    def evaluate(self, assignment: Any) -> float:
        """
        Đánh giá một phương án phân công công việc (assignment) và trả về Makespan.

        Quy trình xử lý:
        1. Xác thực tính hợp lệ của assignment (kiểu dữ liệu nguyên, không bool, không float, độ dài N, miền [0, M-1]).
           Nếu dữ liệu không hợp lệ -> ném ngoại lệ và KHÔNG tính NFE.
        2. Kiểm tra trần ngân sách NFE (nếu evaluation_count >= max_evaluations -> ném RuntimeError).
        3. Tính Makespan từ problem.calculate_makespan() và kiểm tra tính hữu hạn, không âm.
        4. Tăng biến đếm evaluation_count thêm 1 khi đánh giá thành công.

        Args:
            assignment: Danh sách hoặc mảng numpy 1D chứa N số nguyên,
                        phần tử thứ j là chỉ số máy gán cho công việc j.

        Returns:
            float: Thời gian hoàn thành trễ nhất (Makespan) của phương án phân công.

        Raises:
            TypeError: Khi assignment không phải kiểu danh sách/mảng số nguyên, hoặc chứa float/bool.
            ValueError: Khi assignment sai kích thước, chỉ số máy ngoài [0, M-1], hoặc Makespan không hợp lệ.
            RuntimeError: Khi đã đạt trần ngân sách đánh giá (evaluation_count >= max_evaluations).
        """
        # 1. Xác thực dữ liệu đầu vào TRƯỚC (đầu vào sai KHÔNG được làm tiêu hao ngân sách NFE)
        clean_assignment = self._validate_assignment_input(assignment)

        # 2. Kiểm tra trần ngân sách NFE
        if self.evaluation_count >= self.max_evaluations:
            raise RuntimeError(
                f"Evaluation budget exceeded: Đã đạt trần ngân sách {self.max_evaluations} NFE! "
                f"Thuật toán không được phép gọi thêm lượt đánh giá nào nữa."
            )

        # 3. Tính toán Makespan qua mô hình problem và xác thực kết quả
        raw_ms = self.problem.calculate_makespan(clean_assignment)
        if type(raw_ms) is bool or not isinstance(raw_ms, (int, float)):
            raise TypeError(
                f"calculate_makespan phải trả về số thực, nhận được: {raw_ms} (kiểu {type(raw_ms).__name__})"
            )
        makespan = float(raw_ms)
        if not math.isfinite(makespan) or makespan < 0.0:
            raise ValueError(
                f"Makespan phải là số thực hữu hạn và không âm (>= 0), nhận được: {makespan}"
            )

        # 4. Ghi nhận lượt đánh giá thành công vào ngân sách
        self.evaluation_count += 1

        return makespan

    def _validate_assignment_input(self, assignment: Any) -> List[int]:
        """
        Kiểm tra tính hợp lệ của assignment một cách nghiêm ngặt:
        - Không được None.
        - Phải là list, tuple hoặc numpy 1D array.
        - Tuyệt đối KHÔNG tự ý ép kiểu số thực sang số nguyên (bắt buộc caller phải làm mapping).
        - Tuyệt đối loại trừ kiểu boolean.
        - Độ dài phải đúng bằng problem.num_jobs.
        - Chỉ số máy phải nằm trong [0, problem.num_machines - 1].
        """
        if assignment is None:
            raise ValueError("Phương án assignment không được là None.")

        # Xử lý numpy array
        if _HAS_NUMPY and isinstance(assignment, np.ndarray):
            if assignment.ndim != 1:
                raise ValueError(
                    f"Mảng numpy assignment phải là 1 chiều, nhận được {assignment.ndim} chiều."
                )
            # Kiểm tra kiểu dtype của numpy: không cho phép float hoặc bool
            if assignment.dtype == bool or not np.issubdtype(assignment.dtype, np.integer):
                raise TypeError(
                    f"Mảng numpy assignment phải có dtype là số nguyên, nhận được: {assignment.dtype}. "
                    f"Evaluator không tự động ép kiểu số thực thành số nguyên."
                )
            raw_list = assignment.tolist()
        elif isinstance(assignment, (list, tuple)):
            raw_list = list(assignment)
        else:
            raise TypeError(
                f"Assignment phải là list, tuple hoặc numpy.ndarray 1D, nhận: {type(assignment).__name__}"
            )

        # Kiểm tra độ dài khớp với số jobs của bài toán
        num_jobs = getattr(self.problem, "num_jobs", None)
        if num_jobs is not None and len(raw_list) != num_jobs:
            raise ValueError(
                f"Độ dài assignment không khớp: nhận {len(raw_list)} phần tử, "
                f"bài toán yêu cầu đúng {num_jobs} jobs."
            )

        # Kiểm tra từng phần tử: bắt buộc là int chuẩn, cấm float và bool
        num_machines = getattr(self.problem, "num_machines", None)
        clean_assignment: List[int] = []

        for j, m in enumerate(raw_list):
            if type(m) is bool or type(m) is not int:
                raise TypeError(
                    f"Chỉ số máy tại job {j} phải là số nguyên (int), "
                    f"nhận được: {m} (kiểu {type(m).__name__}). "
                    f"Evaluator không tự động làm tròn hay ép kiểu số thực."
                )

            if num_machines is not None:
                if m < 0 or m >= num_machines:
                    raise ValueError(
                        f"Chỉ số máy không hợp lệ tại job {j}: gán máy {m}, "
                        f"trong khi miền hợp lệ là [0, {num_machines - 1}]."
                    )

            clean_assignment.append(m)

        return clean_assignment

    @property
    def remaining_evaluations(self) -> int:
        """Số lượt đánh giá còn lại trong ngân sách."""
        return max(0, self.max_evaluations - self.evaluation_count)

    @property
    def is_budget_exhausted(self) -> bool:
        """Kiểm tra xem ngân sách đánh giá đã cạn kiệt hay chưa."""
        return self.evaluation_count >= self.max_evaluations

    def reset(self) -> None:
        """
        Đặt lại số lần đánh giá về 0.

        CẢNH BÁO QUAN TRỌNG:
        Hàm này CHỈ được gọi khi bắt đầu một lượt chạy thử nghiệm mới hoàn toàn (new independent run)
        hoặc trong các kịch bản Unit Test.
        TUYỆT ĐỐI KHÔNG gọi reset() ở giữa quá trình tối ưu hóa của GA hoặc WOA vì sẽ phá vỡ
        ngân sách đánh giá cố định 10,000 NFE.
        """
        self.evaluation_count = 0

    def __repr__(self) -> str:
        return f"<Evaluator: {self.evaluation_count}/{self.max_evaluations} NFE consumed>"
