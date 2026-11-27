"""
Package gui: Chứa các thành phần giao diện người dùng PyQt6 cho đồ án UPMS.
- ga_widget.py: Widget cấu hình tham số cho Giải thuật Di truyền (GA)
- woa_widget.py: Widget cấu hình tham số cho Thuật toán Cá voi (WOA)
- data_widget.py: Widget nạp và chọn bộ dữ liệu
- gantt_chart.py: Trực quan hóa tiến độ các máy (Gantt Chart)
- convergence_chart.py: Trực quan hóa đường cong hội tụ
- main_window.py: Cửa sổ chính tích hợp toàn bộ hệ thống
"""

try:
    from .ga_widget import GAWidget
    __all__ = ["GAWidget"]
except ImportError:
    __all__ = []
