"""Kiểm định Mann–Whitney U, Wilcoxon signed-rank và hiệu chỉnh Holm (TV4 - Ngân).

Hiệu chỉnh Holm được áp dụng RIÊNG cho từng phép kiểm định, trên họ các dataset
(vd. 3 dataset -> m = 3). Wilcoxon chỉ có ý nghĩa ghép cặp khi hai thuật toán
dùng chung quần thể khởi tạo theo seed (initial_population).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv

try:
    from scipy.stats import mannwhitneyu, wilcoxon
except ImportError as exc:
    raise ImportError("Cần scipy để kiểm định thống kê: pip install scipy") from exc

TESTS = (("mann_whitney_u_p", "mann_whitney_u_p_holm", "mann_whitney_u_significant"),
         ("wilcoxon_p", "wilcoxon_p_holm", "wilcoxon_significant"))


def holm_adjust(p_values):
    """Holm-Bonferroni step-down; trả p-value đã hiệu chỉnh theo đúng thứ tự đầu vào."""
    p_values = [float(p) for p in p_values]
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("Mọi p-value phải nằm trong [0, 1].")
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def _mwu_p(a, b):
    if len(set(a) | set(b)) == 1:      # mọi giá trị giống nhau -> không có khác biệt
        return 1.0
    return float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)


def _wilcoxon_p(pa, pb):
    if all(x == y for x, y in zip(pa, pb)):   # mọi hiệu bằng 0
        return 1.0
    try:
        return float(wilcoxon(pa, pb, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def compare_algorithms(rows, algorithm_a="GA", algorithm_b="WOA", alpha=0.05):
    """So sánh từng dataset. MWU: hai mẫu độc lập; Wilcoxon: ghép cặp theo seed.
    Holm áp dụng riêng cho từng phép kiểm định trên các dataset."""
    grouped = defaultdict(dict)
    for row in rows:
        ds, alg, seed = str(row["dataset"]), str(row["algorithm"]), int(row["seed"])
        bucket = grouped[ds].setdefault(alg, {})
        if seed in bucket:
            raise ValueError(f"Trùng seed {seed} cho {alg} trên dataset {ds}.")
        bucket[seed] = float(row["makespan"])

    records = []
    for dataset, algs in sorted(grouped.items()):
        if algorithm_a not in algs or algorithm_b not in algs:
            continue
        a_by, b_by = algs[algorithm_a], algs[algorithm_b]
        a, b = list(a_by.values()), list(b_by.values())
        common = sorted(set(a_by) & set(b_by))
        mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
        records.append({
            "dataset": dataset, "algorithm_a": algorithm_a, "algorithm_b": algorithm_b,
            "n_a": len(a), "n_b": len(b), "paired_n": len(common),
            "mean_a": mean_a, "mean_b": mean_b,
            "better": (algorithm_a if mean_a < mean_b else
                       algorithm_b if mean_b < mean_a else "tie"),
            "mann_whitney_u_p": _mwu_p(a, b),
            "wilcoxon_p": (_wilcoxon_p([a_by[s] for s in common], [b_by[s] for s in common])
                           if len(common) >= 2 else float("nan")),
        })

    for p_key, holm_key, sig_key in TESTS:
        refs = [i for i, r in enumerate(records) if r[p_key] == r[p_key]]  # bỏ NaN
        adj = holm_adjust([records[i][p_key] for i in refs])
        for i, p_adj in zip(refs, adj):
            records[i][holm_key] = p_adj
        for r in records:
            p = r.get(holm_key)
            r[holm_key] = p if p is not None else float("nan")
            r[sig_key] = bool(p < alpha) if p is not None else None
    return records


def build_final_table(summary, tests):
    """Ghép thống kê mô tả (summary_statistics) với p-value đã hiệu chỉnh Holm
    thành một bảng duy nhất: mỗi dòng = (dataset, algorithm)."""
    by_ds = {t["dataset"]: t for t in tests}
    table = []
    for s in summary:
        t = by_ds.get(s["dataset"], {})
        row = dict(s)
        for key in ("mann_whitney_u_p", "mann_whitney_u_p_holm",
                    "wilcoxon_p", "wilcoxon_p_holm", "better"):
            row[key] = t.get(key, "")
        table.append(row)
    return table


def _save(records, path):
    if not records:
        raise ValueError("Không có dữ liệu để lưu.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return path


def save_statistical_tests(records, path):
    return _save(records, path)


def save_final_table(table, path):
    return _save(table, path)


if __name__ == "__main__":
    # python statistical_tests.py benchmark_raw.csv [GA] [WOA]
    import sys
    from summary_statistics import load_raw_results, summarize_results, save_summary_csv
    src = sys.argv[1] if len(sys.argv) > 1 else "benchmark_raw.csv"
    a = sys.argv[2] if len(sys.argv) > 2 else "GA"
    b = sys.argv[3] if len(sys.argv) > 3 else "WOA"
    rows = load_raw_results(src)
    summary = summarize_results(rows)
    tests = compare_algorithms(rows, a, b)
    save_summary_csv(summary, "summary_statistics.csv")
    save_statistical_tests(tests, "statistical_tests_holm.csv")
    save_final_table(build_final_table(summary, tests), "summary_table_holm.csv")
    print("Đã ghi summary_statistics.csv, statistical_tests_holm.csv, summary_table_holm.csv")
