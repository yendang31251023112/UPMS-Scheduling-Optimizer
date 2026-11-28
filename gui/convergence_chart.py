"""
convergence_chart.py - Đồ thị hội tụ: trục x = số thế hệ (0..199), trục y = Makespan tốt nhất đến lúc đó.
Đường đi xuống càng nhanh và càng thấp thì thuật toán càng tốt.

Cách dùng:
    self.conv = ConvergenceChart()
    self.conv.plot(result["convergence"], "GA")            # vẽ 1 đường
    self.conv.plot_many({"GA": r1["convergence"], "WOA": r2["convergence"]})  # so sánh
"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ConvergenceChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4), layout="constrained")
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

    def plot(self, convergence, label="Thuật toán"):
        self.plot_many({label: convergence})

    def plot_many(self, curves):
        """curves: dict {tên thuật toán: list 200 giá trị}"""
        ax = self.ax
        ax.clear()
        for name, values in curves.items():
            ax.plot(range(len(values)), values, label=name, linewidth=1.8)
            # đánh dấu giá trị cuối (kết quả tốt nhất)
            ax.annotate(f"{values[-1]:.1f}", (len(values) - 1, values[-1]),
                        textcoords="offset points", xytext=(-5, 8), fontsize=8, ha="right")
        ax.set_xlabel("Thế hệ / vòng lặp (0 = sau khởi tạo)")
        ax.set_ylabel("Makespan tốt nhất (best-so-far)")
        ax.set_title("Đồ thị hội tụ")
        ax.grid(alpha=0.3)
        ax.legend()
        self.draw()
