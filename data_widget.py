"""Widget PyQt6 chọn, xem và nạp ma trận thời gian xử lý UPMS."""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QComboBox, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox
    )
except ImportError as exc:  # Cho phép dùng dataset_loader trên máy chưa cài GUI
    raise ImportError("Cần cài PyQt6: pip install PyQt6") from exc

try:
    from dataset_loader import load_problem, generate_problem
    from core.config import BENCHMARK_DATASETS
except ImportError:
    from .dataset_loader import load_problem, generate_problem
    from .core.config import BENCHMARK_DATASETS


class DataWidget(QWidget):
    """API tích hợp cho main_window:
    get_selected_matrix() -> np.ndarray
    get_problem() -> UPMSProblem
    """
    problem_changed = pyqtSignal(object)

    def __init__(self, data_dir=None, parent=None):
        super().__init__(parent)
        # Mặc định: thư mục data/ cạnh file này, không phụ thuộc thư mục chạy.
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
        self._problem = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Small (10 jobs × 3 machines)", "small")
        self.dataset_combo.addItem("Medium (30 jobs × 5 machines)", "medium")
        self.dataset_combo.addItem("Large (100 jobs × 10 machines)", "large")
        self.load_button = QPushButton("Nạp bộ dữ liệu")
        self.generate_button = QPushButton("Tạo dữ liệu mới")
        self.open_button = QPushButton("Mở JSON...")
        row.addWidget(QLabel("Quy mô:"))
        row.addWidget(self.dataset_combo, 1)
        row.addWidget(self.load_button)
        row.addWidget(self.generate_button)
        row.addWidget(self.open_button)
        layout.addLayout(row)

        self.info_label = QLabel("Chưa có dữ liệu.")
        layout.addWidget(self.info_label)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        self.load_button.clicked.connect(self.load_selected_dataset)
        self.generate_button.clicked.connect(self.generate_selected_dataset)
        self.open_button.clicked.connect(self.open_json)
        self.dataset_combo.currentIndexChanged.connect(self._load_if_available)

    def _set_problem(self, problem):
        self._problem = problem
        matrix = np.asarray(problem.processing_times, dtype=float)
        self.table.clear()
        self.table.setRowCount(problem.num_machines)
        self.table.setColumnCount(problem.num_jobs)
        self.table.setVerticalHeaderLabels([f"Machine {i}" for i in range(problem.num_machines)])
        self.table.setHorizontalHeaderLabels([f"Job {j}" for j in range(problem.num_jobs)])
        for i in range(problem.num_machines):
            for j in range(problem.num_jobs):
                self.table.setItem(i, j, QTableWidgetItem(f"{matrix[i, j]:g}"))
        self.info_label.setText(
            f"Đã nạp {problem.num_jobs} jobs × {problem.num_machines} machines "
            f"(ma trận P: {problem.num_machines} × {problem.num_jobs})."
        )
        self.problem_changed.emit(problem)

    def _load_if_available(self, _index):
        key = self.dataset_combo.currentData()
        spec = BENCHMARK_DATASETS[key]
        path = self.data_dir / spec["filename"]
        if path.exists():
            try:
                self._set_problem(load_problem(path))
            except Exception as exc:
                QMessageBox.critical(self, "Lỗi dữ liệu", str(exc))

    def load_selected_dataset(self):
        key = self.dataset_combo.currentData()
        path = self.data_dir / BENCHMARK_DATASETS[key]["filename"]
        try:
            self._set_problem(load_problem(path))
        except Exception as exc:
            QMessageBox.critical(self, "Không thể nạp dữ liệu", str(exc))

    def generate_selected_dataset(self):
        key = self.dataset_combo.currentData()
        spec = BENCHMARK_DATASETS[key]
        try:
            problem = generate_problem(spec["num_jobs"], spec["num_machines"])
            self._set_problem(problem)
        except Exception as exc:
            QMessageBox.critical(self, "Không thể tạo dữ liệu", str(exc))

    def open_json(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Chọn ma trận UPMS", str(self.data_dir), "JSON (*.json)"
        )
        if filename:
            try:
                self._set_problem(load_problem(filename))
            except Exception as exc:
                QMessageBox.critical(self, "File không hợp lệ", str(exc))

    def get_selected_matrix(self) -> np.ndarray:
        if self._problem is None:
            raise RuntimeError("Chưa nạp dữ liệu UPMS.")
        return np.asarray(self._problem.processing_times, dtype=float).copy()

    def get_problem(self):
        if self._problem is None:
            raise RuntimeError("Chưa nạp dữ liệu UPMS.")
        return self._problem

    def set_problem(self, problem):
        """Cho phép main_window hoặc test truyền trực tiếp Problem."""
        self._set_problem(problem)
