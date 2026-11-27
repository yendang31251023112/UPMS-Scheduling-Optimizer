# ĐỒ ÁN TRÍ TUỆ NHÂN TẠO
## Đề tài 7: Giải quyết bài toán Lập lịch máy song song không đồng nhất (UPMS - $R||C_{\max}$) bằng Genetic Algorithm (GA) và Whale Optimization Algorithm (WOA)

---

## 1. Cấu trúc thư mục dự án

```text
Project/
├── core/                       # Thành phần cốt lõi dùng chung
│   ├── __init__.py
│   ├── config.py               # Toàn bộ thông số kỹ thuật chuẩn (NFE, seeds, GA/WOA)
│   ├── problem.py              # Mô hình UPMS, tính Makespan, sinh lịch trình Gantt, validate_result_contract
│   └── evaluator.py            # Quản lý trần 10,000 NFE công bằng cho mọi thuật toán
├── algorithms/                 # Triển khai thuật toán tối ưu
│   ├── __init__.py
│   ├── genetic.py              # GA: Uniform/Two-point Crossover, Resetting/Swap Mutation, Tournament, Elitism
│   └── whale.py                # WOA cơ sở nguyên bản (Mirjalili 2016) + Floor Mapping, Clamping & Plateau Effect
├── gui/                        # Giao diện người dùng PyQt6
│   ├── __init__.py
│   ├── main_window.py          # Cửa sổ chính & điều phối đa luồng QThread
│   ├── gantt_chart.py          # Biểu đồ Gantt trực quan hóa tiến độ các máy
│   ├── convergence_chart.py    # Biểu đồ đường cong hội tụ qua các thế hệ
│   ├── ga_widget.py            # Widget thiết lập & điều khiển GA (kèm tùy chỉnh toán tử)
│   ├── woa_widget.py           # Widget thiết lập & điều khiển WOA
│   └── data_widget.py          # Widget nạp & xem dữ liệu ma trận P
├── data/                       # Bộ dữ liệu thực nghiệm chuẩn
│   ├── small_10x3.json         # Quy mô Small: 10 công việc x 3 máy
│   ├── medium_30x5.json        # Quy mô Medium: 30 công việc x 5 máy
│   └── large_100x10.json       # Quy mô Large: 100 công việc x 10 máy
├── prototype_ga.py             # Script chạy thử nghiệm độc lập GA trên console (10,000 NFE)
├── tests/                      # Kiểm thử tự động & thực nghiệm
│   ├── __init__.py
│   ├── test_core.py            # 36 unit tests kiểm thử toàn diện gói core
│   ├── test_genetic.py         # 9 unit tests kiểm thử Giải thuật Di truyền (GA)
│   ├── benchmark_runner.py     # Điều phối chạy tuần tự 180 lượt benchmark chính thức
│   └── statistical_tests.py    # Kiểm định Mann-Whitney U / Wilcoxon cho GA vs WOA
├── .gitattributes              # Chuẩn hóa ngắt dòng LF giữa macOS và Windows
├── .gitignore                  # Bỏ qua file rác cache, môi trường ảo
├── requirements.txt            # Thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn
```

---

## 2. Hướng dẫn cài đặt môi trường

Khuyến nghị sử dụng Python 3.9 trở lên trong môi trường ảo:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên macOS / Linux:
source venv/bin/activate
# Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Trên Windows (Command Prompt):
.\venv\Scripts\activate.bat

# Cài đặt các thư viện cần thiết
python -m pip install -r requirements.txt
```

---

## 3. Chạy kiểm thử tự động

```bash
python -m unittest discover -s tests -t . -v
```
