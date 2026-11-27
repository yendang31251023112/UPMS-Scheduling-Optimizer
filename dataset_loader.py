"""Đọc, sinh và lưu bộ dữ liệu UPMS theo schema của core.problem.UPMSProblem."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from core.problem import UPMSProblem
    from core.config import BENCHMARK_DATASETS, PROCESSING_TIME_MIN, PROCESSING_TIME_MAX
except ImportError:  # Cho phép import khi package được đặt trong thư mục khác
    from .core.problem import UPMSProblem
    from .core.config import BENCHMARK_DATASETS, PROCESSING_TIME_MIN, PROCESSING_TIME_MAX

PathLike = Union[str, Path]


def load_problem(path: PathLike, validate_benchmark: bool = True) -> UPMSProblem:
    """Đọc JSON schema {num_jobs, num_machines, processing_times} thành UPMSProblem."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError("Định dạng đầu vào hiện hỗ trợ là .json")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    problem = UPMSProblem.from_dict(payload)
    if validate_benchmark:
        problem.validate_benchmark_bounds(PROCESSING_TIME_MIN, PROCESSING_TIME_MAX)
    return problem


def save_problem(problem: UPMSProblem, path: PathLike) -> Path:
    """Lưu UPMSProblem theo đúng schema JSON của Yến."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(problem.to_dict(), f, ensure_ascii=False, indent=4)
    return path


def generate_problem(num_jobs: int, num_machines: int, seed: Optional[int] = None,
                     p_min: int = PROCESSING_TIME_MIN,
                     p_max: int = PROCESSING_TIME_MAX) -> UPMSProblem:
    """Sinh instance ngẫu nhiên; ma trận có shape (M, N), hàng là máy."""
    return UPMSProblem.generate_random_instance(
        num_jobs=num_jobs, num_machines=num_machines, seed=seed,
        p_min=p_min, p_max=p_max
    )


def generate_standard_datasets(output_dir: PathLike = "data", seed: int = 42,
                              overwrite: bool = False) -> dict:
    """Sinh 3 bộ Small/Medium/Large theo core/config.py, trả về {tên: đường dẫn}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for offset, (name, spec) in enumerate(BENCHMARK_DATASETS.items()):
        path = output_dir / spec["filename"]
        if path.exists() and not overwrite:
            # Đọc và xác minh file có sẵn, không tự ý ghi đè dữ liệu nhóm đã chốt.
            load_problem(path)
        else:
            problem = generate_problem(
                spec["num_jobs"], spec["num_machines"], seed=seed + offset
            )
            save_problem(problem, path)
        generated[name] = path
    return generated


def load_matrix_csv(path: PathLike, num_jobs: Optional[int] = None,
                    num_machines: Optional[int] = None) -> UPMSProblem:
    """Đọc CSV ma trận P dạng M hàng x N cột (không có header)."""
    matrix = np.loadtxt(Path(path), delimiter=",", dtype=float, ndmin=2)
    m, n = matrix.shape
    if num_jobs is not None and n != num_jobs:
        raise ValueError(f"CSV có {n} jobs, mong đợi {num_jobs}.")
    if num_machines is not None and m != num_machines:
        raise ValueError(f"CSV có {m} machines, mong đợi {num_machines}.")
    return UPMSProblem(num_jobs=n, num_machines=m, processing_times=matrix)
