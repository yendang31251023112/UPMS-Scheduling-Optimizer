"""Tổng hợp thống kê mô tả cho kết quả benchmark (TV4 - Ngân).

Cột CSV thô bắt buộc: algorithm,dataset,seed,makespan,runtime,evaluation_count
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

try:
    from core.config import MAX_EVALUATIONS
except ImportError:  # chạy độc lập
    MAX_EVALUATIONS = 10000

REQUIRED_COLUMNS = {"algorithm", "dataset", "seed", "makespan", "runtime", "evaluation_count"}


def _validate_rows(rows, expected_nfe=MAX_EVALUATIONS):
    """Kiểm tra cột, giá trị hữu hạn, NFE đúng ngân sách và không trùng (algorithm, dataset, seed)."""
    if not rows:
        raise ValueError("Không có dữ liệu benchmark để thống kê.")
    seen = set()
    for i, row in enumerate(rows):
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ValueError(f"Dòng {i}: thiếu cột bắt buộc {sorted(missing)}")
        for key in ("makespan", "runtime"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Dòng {i}: {key} phải hữu hạn và không âm.")
        if expected_nfe is not None and int(row["evaluation_count"]) != expected_nfe:
            raise ValueError(
                f"Dòng {i}: evaluation_count = {row['evaluation_count']}, "
                f"yêu cầu đúng {expected_nfe} NFE.")
        key = (str(row["algorithm"]), str(row["dataset"]), int(row["seed"]))
        if key in seen:
            raise ValueError(f"Dòng {i}: trùng (algorithm, dataset, seed) = {key}.")
        seen.add(key)


def load_raw_results(path, expected_nfe=MAX_EVALUATIONS):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    _validate_rows(rows, expected_nfe)
    for row in rows:
        row["seed"] = int(row["seed"])
        row["makespan"] = float(row["makespan"])
        row["runtime"] = float(row["runtime"])
        row["evaluation_count"] = int(row["evaluation_count"])
    return rows


def summarize_results(rows, expected_nfe=MAX_EVALUATIONS):
    """Trả về list dict theo (dataset, algorithm): Best, Worst, Mean, Median,
    Sample StdDev, Runtime, NFE."""
    _validate_rows(rows, expected_nfe)
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["algorithm"]))].append(row)
    output = []
    for (dataset, algorithm), group in sorted(groups.items()):
        values = [float(r["makespan"]) for r in group]
        runtimes = [float(r["runtime"]) for r in group]
        output.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "runs": len(group),
            "Best": min(values),
            "Worst": max(values),
            "Mean": mean(values),
            "Median": median(values),
            "Sample StdDev": stdev(values) if len(values) > 1 else 0.0,
            "Runtime Mean": mean(runtimes),
            "Runtime Sample StdDev": stdev(runtimes) if len(runtimes) > 1 else 0.0,
            "Mean NFE": mean(int(r["evaluation_count"]) for r in group),
        })
    return output


def save_summary_csv(summary, path):
    if not summary:
        raise ValueError("Summary rỗng.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return path
