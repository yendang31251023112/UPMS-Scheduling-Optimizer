"""
main_window.py - BẢN PHÁC THẢO cửa sổ chính (TV3 - Lâm).

Bố cục:
    Bên trái : chọn bộ dữ liệu (Small/Medium/Large), chọn thuật toán, nút "Chạy"
    Bên phải : các tab  [Kết quả] [Gantt] [Hội tụ]
Thuật toán chạy trong QThread để cửa sổ không bị đơ.

Hiện tại dùng dữ liệu giả (fake_data.py). Khi các bạn bàn giao thì sửa các chỗ có ghi
"TODO-GHÉP" bên dưới để dùng code thật.
"""
import sys
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLabel, QComboBox, QPushButton, QTabWidget,
                             QTableWidget, QTableWidgetItem, QMessageBox)

from gantt_chart import GanttChart
from convergence_chart import ConvergenceChart
import fake_data                                   # TODO-GHÉP: bỏ khi có code thật

# TODO-GHÉP: bỏ comment khi các bạn đã push code
# from genetic import run_ga
# from whale import run_woa
# from ga_widget import GAWidget
# from woa_widget import WOAWidget
# from data_widget import DataWidget

SIZES = {"Small (10 x 3)": (10, 3), "Medium (30 x 5)": (30, 5), "Large (100 x 10)": (100, 10)}


def fake_run(name):
    """Hàm chạy giả (thay cho run_ga / run_woa): chờ 1 giây rồi trả về result giả."""
    def run(P, params):
        time.sleep(1)
        return fake_data.make_fake_result(P, seed=1 if name == "GA" else 2)
    return run


ALGORITHMS = {"GA": fake_run("GA"), "WOA": fake_run("WOA")}   # TODO-GHÉP: {"GA": run_ga, "WOA": run_woa}


class Worker(QThread):
    """Luồng nền: chạy thuật toán xong thì báo về cửa sổ chính bằng signal."""
    finished_ok = pyqtSignal(dict)      # gửi result
    failed = pyqtSignal(str)            # gửi thông báo lỗi

    def __init__(self, run_fn, problem, params):
        super().__init__()
        self.run_fn, self.problem, self.params = run_fn, problem, params

    def run(self):                      # hàm này chạy ở luồng nền, KHÔNG được đụng vào giao diện
        try:
            self.finished_ok.emit(self.run_fn(self.problem, self.params))
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UPMS Scheduling Optimizer - GA & WOA")
        self.resize(1100, 650)
        self.results = {}               # lưu result của từng thuật toán để vẽ so sánh

        # ---------- Cột trái: điều khiển ----------
        self.cb_size = QComboBox();  self.cb_size.addItems(SIZES.keys())
        self.cb_algo = QComboBox();  self.cb_algo.addItems(ALGORITHMS.keys())
        self.btn_run = QPushButton("Chạy thuật toán")
        self.btn_run.clicked.connect(self.on_run)
        self.lbl_status = QLabel("Sẵn sàng")

        box = QGroupBox("Cấu hình")
        form = QVBoxLayout(box)
        form.addWidget(QLabel("Bộ dữ liệu:")); form.addWidget(self.cb_size)
        form.addWidget(QLabel("Thuật toán:"));  form.addWidget(self.cb_algo)
        # TODO-GHÉP: form.addWidget(GAWidget()), form.addWidget(WOAWidget()), form.addWidget(DataWidget())
        form.addWidget(self.btn_run)
        form.addWidget(self.lbl_status)
        form.addStretch()

        # ---------- Cột phải: các tab kết quả ----------
        self.lbl_summary = QLabel("Chưa có kết quả")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Công việc", "Máy được giao"])
        tab_result = QWidget()
        lay = QVBoxLayout(tab_result)
        lay.addWidget(self.lbl_summary); lay.addWidget(self.table)

        self.gantt = GanttChart()
        self.conv = ConvergenceChart()
        tabs = QTabWidget()
        tabs.addTab(tab_result, "Kết quả")
        tabs.addTab(self.gantt, "Biểu đồ Gantt")
        tabs.addTab(self.conv, "Đồ thị hội tụ")

        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(box, 1)
        root.addWidget(tabs, 3)
        self.setCentralWidget(central)

    # ---------- Xử lý khi bấm nút Chạy ----------
    def on_run(self):
        n_jobs, n_machines = SIZES[self.cb_size.currentText()]
        # TODO-GHÉP: problem = data_widget.get_problem()  /  P = data_widget.get_selected_matrix()
        self.P = fake_data.make_fake_matrix(n_jobs, n_machines)
        self.algo_name = self.cb_algo.currentText()
        params = {}                                    # TODO-GHÉP: ga_widget.get_params() / woa_widget.get_params()

        self.btn_run.setEnabled(False)                 # tránh bấm 2 lần
        self.lbl_status.setText(f"Đang chạy {self.algo_name}...")
        self.worker = Worker(ALGORITHMS[self.algo_name], self.P, params)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_done(self, result):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Hoàn thành")
        self.results[self.algo_name] = result
        n_machines = self.P.shape[1]

        self.lbl_summary.setText(f"Makespan tốt nhất: {result['makespan']:.2f}   |   "
                                 f"Thời gian chạy: {result['runtime']} s   |   "
                                 f"Số lần đánh giá: {result['evaluation_count']}")
        self.table.setRowCount(len(result["assignment"]))
        for job, m in enumerate(result["assignment"]):
            self.table.setItem(job, 0, QTableWidgetItem(f"Job {job}"))
            self.table.setItem(job, 1, QTableWidgetItem(f"Máy {m}"))

        self.gantt.plot(result["schedule"], n_machines, result["makespan"])
        self.conv.plot_many({n: r["convergence"] for n, r in self.results.items()})

    def on_error(self, message):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Lỗi")
        QMessageBox.critical(self, "Lỗi", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
