"""
demo_charts.py - Chạy file này để xem 2 biểu đồ mẫu bằng dữ liệu giả (sản phẩm "2 biểu đồ mẫu chạy được").
    python demo_charts.py
"""
import sys
from PyQt6.QtWidgets import QApplication, QTabWidget

from gui.gantt_chart import GanttChart
from gui.convergence_chart import ConvergenceChart
from data import fake_data

if __name__ == "__main__":
    app = QApplication(sys.argv)

    P = fake_data.make_fake_matrix(10, 3)                       # Small: 10 việc, 3 máy
    r_ga = fake_data.make_fake_result(P, seed=1)
    r_woa = fake_data.make_fake_result(P, seed=2)

    gantt = GanttChart()
    gantt.plot(r_ga["schedule"], 3, r_ga["makespan"])

    conv = ConvergenceChart()
    conv.plot_many({"GA": r_ga["convergence"], "WOA": r_woa["convergence"]})

    tabs = QTabWidget()
    tabs.addTab(gantt, "Gantt")
    tabs.addTab(conv, "Hội tụ")
    tabs.resize(900, 520)
    tabs.setWindowTitle("Demo biểu đồ (dữ liệu giả)")
    tabs.show()
    sys.exit(app.exec())
