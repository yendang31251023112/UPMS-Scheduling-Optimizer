# Phần việc TV4 – Ngân (UPMS)

## Phạm vi bàn giao
- `dataset_loader.py`: đọc/lưu JSON theo đúng `UPMSProblem.from_dict()` của Yến; sinh ma trận chuẩn.
- `data_widget.py`: PyQt6 widget nạp Small/Medium/Large, mở JSON, xem ma trận; API `get_selected_matrix() -> np.ndarray` và `get_problem()`.
- `summary_statistics.py`: Best, Worst, Mean, Sample StdDev, Runtime và NFE; đọc/ghi CSV.
- `statistical_tests.py`: Mann–Whitney U, Wilcoxon signed-rank ghép cặp theo seed và hiệu chỉnh Holm.
- `data/`: ba bộ dữ liệu JSON Small, Medium, Large.

## Quy ước dữ liệu quan trọng
`UPMSProblem` của Yến quy định ma trận `processing_times` có shape **(M, N)**: hàng là máy, cột là job. Vì vậy tên bộ dữ liệu Small 10 jobs × 3 machines được lưu thành ma trận 3 × 10; Medium là 5 × 30; Large là 10 × 100. `get_selected_matrix()` giữ nguyên quy ước này.

JSON có đúng ba trường `num_jobs`, `num_machines`, `processing_times`. Mọi dữ liệu được kiểm tra bằng chính `UPMSProblem.from_dict()` và `validate_benchmark_bounds(10, 100)`.

## Cài đặt
```bash
pip install numpy scipy PyQt6
```
Đặt thư mục `core/` của Yến ở cùng cấp với `dataset_loader.py` và `data_widget.py` (đã được đóng gói trong bản bàn giao này để kiểm tra độc lập).

## Chạy tạo lại dữ liệu
```python
from dataset_loader import generate_standard_datasets
generate_standard_datasets("data", seed=42, overwrite=True)
```
Mặc định seed lần lượt là 42, 43, 44. Nếu nhóm đã thống nhất seed khác, chỉ cần truyền seed gốc tương ứng.

## Sử dụng
```python
from dataset_loader import load_problem
problem = load_problem("data/small_10x3.json")
matrix = problem.processing_times  # M x N
```

Trong `main_window.py`:
```python
problem = data_widget.get_problem()
matrix = data_widget.get_selected_matrix()
```

## Benchmark và thống kê
Ghi mỗi lần chạy một dòng theo header trong `benchmark_raw.csv`:
`algorithm,dataset,seed,makespan,runtime,evaluation_count`.

Sau đó gọi:
```python
from summary_statistics import load_raw_results, summarize_results, save_summary_csv
from statistical_tests import compare_algorithms, save_statistical_tests

rows = load_raw_results("benchmark_raw.csv")
save_summary_csv(summarize_results(rows), "summary_statistics.csv")
tests = compare_algorithms(rows, "GA", "WOA", alpha=0.05)
save_statistical_tests(tests, "statistical_tests_holm.csv")

# Bảng tổng hợp cuối: Best/Worst/Mean/StdDev/Runtime + p-value đã hiệu chỉnh Holm
from statistical_tests import build_final_table, save_final_table
save_final_table(build_final_table(summarize_results(rows), tests), "summary_table_holm.csv")
```
Hoặc chạy nhanh: `python statistical_tests.py benchmark_raw.csv GA WOA`.
`load_raw_results` từ chối dòng sai NFE (khác 10.000) và dòng trùng (algorithm, dataset, seed).
Không tạo trước CSV kết quả thực nghiệm hoặc p-value giả: các file thống kê chỉ có ý nghĩa sau khi chạy benchmark thực tế. Wilcoxon yêu cầu ghép cặp theo seed; Mann–Whitney U xem hai mẫu độc lập. Holm được áp dụng RIÊNG cho từng phép kiểm định, trên họ các dataset (m = 3). Wilcoxon chỉ ghép cặp thật khi GA và WOA dùng chung quần thể khởi tạo theo seed (`initial_population`).

## Lưu ý tích hợp
- `data_widget.py` yêu cầu PyQt6; các module loader/thống kê không phụ thuộc GUI.
- WOA của Tuấn import `core.*`, nên cấu trúc cuối của nhóm cần giữ `core/` ở root project và đặt `algorithms/whale.py` vào `algorithms/`.
- WOA hiện tại nhận `initial_population` qua `params["initial_population"]`; phần loader không thay đổi API thuật toán.
