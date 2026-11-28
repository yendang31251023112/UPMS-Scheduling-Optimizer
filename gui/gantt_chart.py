"""
gantt_chart.py - Biểu đồ Gantt: mỗi MÁY là một hàng, mỗi CÔNG VIỆC là một thanh ngang.
Chiều dài thanh = thời gian xử lý; vạch đỏ = Makespan (thời điểm xong muộn nhất).

Cách dùng trong main_window:
    self.gantt = GanttChart()
    self.gantt.plot(result["schedule"], n_machines, result["makespan"])
"""
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class GanttChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4), layout="constrained")
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

    def plot(self, schedule, n_machines, makespan=None):
        """schedule: list (machine_id, job_id, start_time, end_time), Job ID tăng dần."""
        ax = self.ax
        ax.clear()                                  # xóa hình cũ trước khi vẽ lại
        cmap = colormaps["tab20"]                   # bảng 20 màu, mỗi job một màu

        n_jobs = len(schedule)
        for machine_id, job_id, start, end in schedule:
            # barh(y, width, left): thanh ngang tại hàng y, bắt đầu 'left', dài 'width'
            ax.barh(machine_id, end - start, left=start,
                    color=cmap(job_id % 20), edgecolor="white", height=0.6)
            # Chỉ ghi số Job lên thanh khi ít job, nếu không sẽ rối (bài Large 100 job)
            if n_jobs <= 30:
                ax.text((start + end) / 2, machine_id, f"J{job_id}",
                        ha="center", va="center", fontsize=8)

        if makespan is not None:
            ax.axvline(makespan, color="red", linestyle="--", linewidth=1)
            ax.text(makespan, -0.9, f"Makespan = {makespan:.1f} ",
                    color="red", fontsize=9, ha="right", va="center")

        ax.set_yticks(range(n_machines))
        ax.set_yticklabels([f"Máy {m}" for m in range(n_machines)])
        ax.set_ylim(n_machines - 0.4, -1.3)         # Máy 0 trên cùng, chừa chỗ cho chữ Makespan
        ax.set_xlabel("Thời gian")
        ax.set_title("Biểu đồ Gantt - lịch trình công việc")
        ax.grid(axis="x", alpha=0.3)
        self.draw()                                 # vẽ lên màn hình
