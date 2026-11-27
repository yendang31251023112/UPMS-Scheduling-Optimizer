"""
gui/ga_widget.py
----------------
Widget giao diện PyQt6 cấu hình tham số cho Giải thuật Di truyền (GA).

Ggiao tiếp với TV3:
- Cung cấp phương thức: ga_widget.get_params() -> dict
- Cung cấp phương thức: ga_widget.reset_to_defaults() -> None
- Cung cấp phương thức: ga_widget.set_params(dict) -> None
"""

from typing import Any, Dict, Optional

from core.config import (
    GA_CROSSOVER_TYPE,
    GA_ELITE_COUNT,
    GA_MUTATION_TYPE,
    GA_PC,
    GA_PM,
    GA_TOURNAMENT_K,
    MAX_GENERATIONS,
    POPULATION_SIZE,
)

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    QWidget = object  # type: ignore


class GAWidget(QWidget):
    """
    Widget cấu hình tham số Giải thuật Di truyền (GA).

    Cung cấp giao diện đồ họa cho phép người dùng điều chỉnh các siêu tham số:
    - Kích thước quần thể (N_pop)
    - Số thế hệ tối đa (T)
    - Xác suất lai ghép (Pc)
    - Xác suất đột biến (Pm)
    - Kích thước giải đấu (k)
    - Số lượng cá thể tinh hoa (Elite)
    - Phương pháp lai ghép & đột biến
    - Seed ngẫu nhiên (tùy chọn)
    """

    # Tín hiệu phát ra khi có bất kỳ tham số nào thay đổi
    params_changed = pyqtSignal() if _HAS_PYQT6 else None

    def __init__(self, parent: Optional[Any] = None) -> None:
        if not _HAS_PYQT6:
            raise ImportError(
                "Thư viện PyQt6 chưa được cài đặt trong môi trường hiện tại. "
                "Vui lòng chạy: pip install PyQt6"
            )
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        group_box = QGroupBox("Tham số Giải thuật Di truyền (GA)")
        form_layout = QFormLayout(group_box)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        # 1. Kích thước quần thể (Population Size)
        self.spin_pop_size = QSpinBox()
        self.spin_pop_size.setRange(10, 500)
        self.spin_pop_size.setValue(POPULATION_SIZE)
        self.spin_pop_size.setToolTip("Số lượng cá thể trong quần thể mỗi thế hệ (Mặc định: 50).")
        form_layout.addRow(QLabel("Kích thước quần thể (N):"), self.spin_pop_size)

        # 2. Số thế hệ tối đa (Max Generations)
        self.spin_max_gen = QSpinBox()
        self.spin_max_gen.setRange(1, 2000)
        self.spin_max_gen.setValue(MAX_GENERATIONS)
        self.spin_max_gen.setToolTip("Số thế hệ tiến hóa sau bước khởi tạo ban đầu (Mặc định: 199).")
        form_layout.addRow(QLabel("Số thế hệ (Generations):"), self.spin_max_gen)

        # 3. Xác suất lai ghép (Pc)
        self.spin_pc = QDoubleSpinBox()
        self.spin_pc.setRange(0.0, 1.0)
        self.spin_pc.setSingleStep(0.05)
        self.spin_pc.setValue(GA_PC)
        self.spin_pc.setToolTip("Xác suất thực hiện phép lai ghép cặp cá thể cha mẹ (Mặc định: 0.8).")
        form_layout.addRow(QLabel("Xác suất lai ghép (Pc):"), self.spin_pc)

        # 4. Xác suất đột biến (Pm)
        self.spin_pm = QDoubleSpinBox()
        self.spin_pm.setRange(0.0, 1.0)
        self.spin_pm.setSingleStep(0.01)
        self.spin_pm.setValue(GA_PM)
        self.spin_pm.setToolTip("Xác suất đột biến gen gán máy ngẫu nhiên (Mặc định: 0.05).")
        form_layout.addRow(QLabel("Xác suất đột biến (Pm):"), self.spin_pm)

        # 5. Kích thước giải đấu chọn lọc (Tournament k)
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 20)
        self.spin_k.setValue(GA_TOURNAMENT_K)
        self.spin_k.setToolTip("Số cá thể ngẫu nhiên tham gia giải đấu chọn lọc cha mẹ (Mặc định: 3).")
        form_layout.addRow(QLabel("Kích thước Tournament (k):"), self.spin_k)

        # 6. Số lượng cá thể tinh hoa (Elite Count)
        self.spin_elite = QSpinBox()
        self.spin_elite.setRange(0, 10)
        self.spin_elite.setValue(GA_ELITE_COUNT)
        self.spin_elite.setToolTip("Số cá thể tốt nhất giữ lại cho thế hệ sau mà không đánh giá lại (Mặc định: 1).")
        form_layout.addRow(QLabel("Cá thể tinh hoa (Elite):"), self.spin_elite)

        # 7. Loại lai ghép (Crossover Type)
        self.combo_cx = QComboBox()
        self.combo_cx.addItems(["uniform", "two_point"])
        self.combo_cx.setCurrentText(GA_CROSSOVER_TYPE)
        self.combo_cx.setToolTip("Toán tử lai ghép: Uniform (Đồng nhất) hoặc Two-point (Hai điểm cắt).")
        form_layout.addRow(QLabel("Phương pháp lai ghép:"), self.combo_cx)

        # 8. Loại đột biến (Mutation Type)
        self.combo_mut = QComboBox()
        self.combo_mut.addItems(["resetting", "swap"])
        self.combo_mut.setCurrentText(GA_MUTATION_TYPE)
        self.combo_mut.setToolTip("Toán tử đột biến: Random Resetting (Đổi máy) hoặc Swap (Hoán vị).")
        form_layout.addRow(QLabel("Phương pháp đột biến:"), self.combo_mut)

        # 9. Tùy chọn Seed cố định
        seed_layout = QHBoxLayout()
        self.check_seed = QCheckBox("Cố định Seed")
        self.check_seed.setChecked(False)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(1, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setEnabled(False)
        self.check_seed.toggled.connect(self.spin_seed.setEnabled)
        seed_layout.addWidget(self.check_seed)
        seed_layout.addWidget(self.spin_seed)
        form_layout.addRow(QLabel("Seed ngẫu nhiên:"), seed_layout)

        # 10. Chế độ Benchmark chuẩn (Khóa tham số) vs Khảo sát (Tự do chỉnh sửa)
        self.check_benchmark_mode = QCheckBox("Khóa chế độ Benchmark chuẩn (Config cố định)")
        self.check_benchmark_mode.setChecked(False)
        self.check_benchmark_mode.setToolTip("Khi bật, khóa các tham số theo đúng quy chuẩn 180 runs của đồ án.")
        self.check_benchmark_mode.toggled.connect(self.set_benchmark_mode)
        form_layout.addRow(QLabel("Chế độ chạy:"), self.check_benchmark_mode)

        # Nút khôi phục mặc định
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Khôi phục mặc định")
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_reset)
        form_layout.addRow(btn_layout)

        main_layout.addWidget(group_box)

        # Kết nối sự kiện thay đổi tham số
        self.spin_pop_size.valueChanged.connect(self._on_param_changed)
        self.spin_max_gen.valueChanged.connect(self._on_param_changed)
        self.spin_pc.valueChanged.connect(self._on_param_changed)
        self.spin_pm.valueChanged.connect(self._on_param_changed)
        self.spin_k.valueChanged.connect(self._on_param_changed)
        self.spin_elite.valueChanged.connect(self._on_param_changed)
        self.combo_cx.currentTextChanged.connect(self._on_param_changed)
        self.combo_mut.currentTextChanged.connect(self._on_param_changed)
        self.check_seed.toggled.connect(self._on_param_changed)
        self.spin_seed.valueChanged.connect(self._on_param_changed)

    def _on_param_changed(self) -> None:
        if self.params_changed is not None:
            self.params_changed.emit()

    def get_params(self) -> Dict[str, Any]:
        """
        Lấy toàn bộ tham số người dùng đã cấu hình trên giao diện.

        Returns:
            Dict[str, Any]: Dictionary tham số chuẩn truyền vào run_ga().
        """
        seed_val = self.spin_seed.value() if self.check_seed.isChecked() else None
        return {
            "population_size": self.spin_pop_size.value(),
            "max_generations": self.spin_max_gen.value(),
            "pc": self.spin_pc.value(),
            "pm": self.spin_pm.value(),
            "tournament_k": self.spin_k.value(),
            "elite_count": self.spin_elite.value(),
            "crossover_type": self.combo_cx.currentText(),
            "mutation_type": self.combo_mut.currentText(),
            "seed": seed_val,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Nạp các tham số từ bên ngoài vào giao diện (dùng khi nạp preset hoặc cấu hình)."""
        if "population_size" in params:
            self.spin_pop_size.setValue(int(params["population_size"]))
        if "max_generations" in params:
            self.spin_max_gen.setValue(int(params["max_generations"]))
        if "pc" in params or "crossover_prob" in params:
            self.spin_pc.setValue(float(params.get("pc", params.get("crossover_prob"))))
        if "pm" in params or "mutation_prob" in params:
            self.spin_pm.setValue(float(params.get("pm", params.get("mutation_prob"))))
        if "tournament_k" in params or "k" in params:
            self.spin_k.setValue(int(params.get("tournament_k", params.get("k"))))
        if "elite_count" in params:
            self.spin_elite.setValue(int(params["elite_count"]))
        if "crossover_type" in params:
            self.combo_cx.setCurrentText(str(params["crossover_type"]))
        if "mutation_type" in params:
            self.combo_mut.setCurrentText(str(params["mutation_type"]))
        if "seed" in params:
            seed_val = params["seed"]
            if seed_val is not None:
                self.check_seed.setChecked(True)
                self.spin_seed.setValue(int(seed_val))
            else:
                self.check_seed.setChecked(False)

    def set_benchmark_mode(self, enabled: bool) -> None:
        """
        Bật/tắt chế độ Benchmark chuẩn.
        - Khi bật (True): Tự động khôi phục cấu hình chuẩn và khóa các ô nhập liệu
          để đảm bảo không bị vô tình sửa đổi tham số khi thực nghiệm đối đầu.
        - Khi tắt (False): Mở khóa toàn bộ cho phép khảo sát tham số tự do.
        """
        if enabled:
            self.reset_to_defaults()
        editable = not enabled
        self.spin_pop_size.setEnabled(editable)
        self.spin_max_gen.setEnabled(editable)
        self.spin_pc.setEnabled(editable)
        self.spin_pm.setEnabled(editable)
        self.spin_k.setEnabled(editable)
        self.spin_elite.setEnabled(editable)
        self.combo_cx.setEnabled(editable)
        self.combo_mut.setEnabled(editable)
        self.check_seed.setEnabled(editable)
        self.spin_seed.setEnabled(editable and self.check_seed.isChecked())
        self.btn_reset.setEnabled(editable)

    def reset_to_defaults(self) -> None:
        """Khôi phục lại toàn bộ tham số chuẩn đồ án từ config.py."""
        self.spin_pop_size.setValue(POPULATION_SIZE)
        self.spin_max_gen.setValue(MAX_GENERATIONS)
        self.spin_pc.setValue(GA_PC)
        self.spin_pm.setValue(GA_PM)
        self.spin_k.setValue(GA_TOURNAMENT_K)
        self.spin_elite.setValue(GA_ELITE_COUNT)
        self.combo_cx.setCurrentText(GA_CROSSOVER_TYPE)
        self.combo_mut.setCurrentText(GA_MUTATION_TYPE)
        self.check_seed.setChecked(False)
