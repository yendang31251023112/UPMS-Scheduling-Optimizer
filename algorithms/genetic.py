"""
algorithms/genetic.py
---------------------
Thuật toán Di truyền (Genetic Algorithm - GA) giải bài toán UPMS (R || Cmax).
Tuân thủ nghiêm ngặt 15 quy chuẩn kỹ thuật và hợp đồng dữ liệu của đồ án:
- Biểu diễn nghiệm: Vector số nguyên (Assignment vector) độ dài N, m_j in [0, M-1].
- Khởi tạo quần thể: 50 cá thể ngẫu nhiên.
- Chọn lọc (Selection): Tournament Selection (k=3).
- Lai ghép (Crossover): Uniform Crossover (Pc = 0.8) mặc định, hỗ trợ Two-point.
- Đột biến (Mutation): Random Resetting Mutation (Pm = 0.05).
- Chiến lược sống sót (Elitism): Generational Replacement with Elitism - 50 cá thể con mới kết hợp
  bảo tồn 1 cá thể tinh hoa (Elite) từ thế hệ trước (tổng 51 ứng viên) -> chọn top 50 cá thể tốt nhất
  lập thành quần thể thế hệ sau. Cá thể Elite tuyệt đối không bị đánh giá lại NFE.
- Ngân sách đánh giá: 50 khởi tạo + 199 thế hệ * 50 con mới = 10,000 NFE chính xác tuyệt đối.
- Đường cong hội tụ: Đúng 200 điểm Best-so-far lũy kế.
"""

import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import (
    CONVERGENCE_POINTS,
    GA_CROSSOVER_TYPE,
    GA_ELITE_COUNT,
    GA_MUTATION_TYPE,
    GA_PC,
    GA_PM,
    GA_TOURNAMENT_K,
    MAX_EVALUATIONS,
    MAX_GENERATIONS,
    POPULATION_SIZE,
    VALID_GA_CROSSOVER_TYPES,
    VALID_GA_MUTATION_TYPES,
)
from core.evaluator import Evaluator
from core.problem import UPMSProblem, validate_result_contract


class GeneticAlgorithm:
    """
    Bộ giải thuật Di truyền (GA) chuẩn hóa cho bài toán UPMS.

    Attributes:
        problem (UPMSProblem): Thể hiện bài toán UPMS cần giải.
        population_size (int): Kích thước quần thể (mặc định 50).
        max_generations (int): Số thế hệ tìm kiếm sau khởi tạo (mặc định 199).
        crossover_prob (float): Xác suất lai ghép Pc trên từng cặp cha mẹ (mặc định 0.8).
        mutation_prob (float): Xác suất đột biến Pm trên từng cá thể (mặc định 0.05, Individual-level mutation).
        tournament_k (int): Kích thước giải đấu chọn lọc k (mặc định 3).
        elite_count (int): Số lượng cá thể tinh hoa giữ lại (mặc định 1).
        crossover_type (str): "uniform" hoặc "two_point".
        mutation_type (str): "resetting" hoặc "swap".
        seed (Optional[int]): Seed ngẫu nhiên để tái lập kết quả thực nghiệm.
    """

    def __init__(
        self,
        problem: UPMSProblem,
        population_size: int = POPULATION_SIZE,
        max_generations: int = MAX_GENERATIONS,
        crossover_prob: float = GA_PC,
        mutation_prob: float = GA_PM,
        tournament_k: int = GA_TOURNAMENT_K,
        elite_count: int = GA_ELITE_COUNT,
        crossover_type: str = GA_CROSSOVER_TYPE,
        mutation_type: str = GA_MUTATION_TYPE,
        seed: Optional[int] = None,
    ) -> None:
        if problem is None:
            raise ValueError("Đối tượng problem không được để trống (None).")

        # Xác thực tham số đầu vào
        if type(population_size) is not int or population_size <= 1:
            raise ValueError(f"population_size phải là số nguyên > 1, nhận: {population_size}")
        if type(max_generations) is not int or max_generations <= 0:
            raise ValueError(f"max_generations phải là số nguyên > 0, nhận: {max_generations}")
        if type(crossover_prob) not in (int, float) or not (0.0 <= crossover_prob <= 1.0):
            raise ValueError(f"crossover_prob phải nằm trong [0.0, 1.0], nhận: {crossover_prob}")
        if type(mutation_prob) not in (int, float) or not (0.0 <= mutation_prob <= 1.0):
            raise ValueError(f"mutation_prob phải nằm trong [0.0, 1.0], nhận: {mutation_prob}")
        if type(tournament_k) is not int or not (1 <= tournament_k <= population_size):
            raise ValueError(f"tournament_k phải trong [1, {population_size}], nhận: {tournament_k}")
        if type(elite_count) is not int or not (0 <= elite_count < population_size):
            raise ValueError(f"elite_count phải trong [0, {population_size - 1}], nhận: {elite_count}")
        if crossover_type not in VALID_GA_CROSSOVER_TYPES:
            raise ValueError(f"crossover_type không hợp lệ: {crossover_type}")
        if mutation_type not in VALID_GA_MUTATION_TYPES:
            raise ValueError(f"mutation_type không hợp lệ: {mutation_type}")
        if seed is not None and (type(seed) is bool or type(seed) is not int):
            raise TypeError(f"seed phải là int hoặc None, nhận: {seed}")

        self.problem: UPMSProblem = problem
        self.population_size: int = population_size
        self.max_generations: int = max_generations
        self.crossover_prob: float = float(crossover_prob)
        self.mutation_prob: float = float(mutation_prob)
        self.tournament_k: int = tournament_k
        self.elite_count: int = elite_count
        self.crossover_type: str = crossover_type
        self.mutation_type: str = mutation_type
        self.seed: Optional[int] = seed

        self.rng: random.Random = random.Random(seed)

    def _create_random_chromosome(self) -> List[int]:
        """Tạo một nhiễm sắc thể ngẫu nhiên: gán mỗi job cho một máy ngẫu nhiên [0, M-1]."""
        num_machines = self.problem.num_machines
        return [self.rng.randint(0, num_machines - 1) for _ in range(self.problem.num_jobs)]

    def _tournament_selection(
        self, population: List[Tuple[List[int], float]]
    ) -> List[int]:
        """
        Chọn lọc giải đấu (Tournament Selection):
        Lấy ngẫu nhiên k ứng viên từ quần thể, chọn cá thể có Makespan nhỏ nhất (tốt nhất).
        """
        candidates = self.rng.sample(population, self.tournament_k)
        # Sắp xếp theo Makespan tăng dần (nghiệm Makespan nhỏ hơn là tốt hơn)
        candidates.sort(key=lambda ind: ind[1])
        return candidates[0][0][:]  # Trả về bản sao của nhiễm sắc thể thắng cuộc

    def _crossover_uniform(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Lai ghép đồng nhất (Uniform Crossover):
        Với mỗi gen j, xác suất 50% lấy từ parent 1 và 50% lấy từ parent 2.
        """
        child1: List[int] = []
        child2: List[int] = []
        for g1, g2 in zip(parent1, parent2):
            if self.rng.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        return child1, child2

    def _crossover_two_point(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Lai ghép hai điểm cắt (Two-point Crossover) phục vụ khảo sát so sánh toán tử.
        Chọn hai điểm cắt ngẫu nhiên trên vector phân công và hoán đổi đoạn giữa.
        """
        length = len(parent1)
        if length < 2:
            return parent1[:], parent2[:]
        cx1 = self.rng.randint(0, length - 2)
        cx2 = self.rng.randint(cx1 + 1, length - 1)
        child1 = parent1[:cx1] + parent2[cx1:cx2] + parent1[cx2:]
        child2 = parent2[:cx1] + parent1[cx1:cx2] + parent2[cx2:]
        return child1, child2

    def _crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Điều phối lai ghép dựa trên xác suất Pc và loại crossover_type (Mặc định: Uniform)."""
        if self.rng.random() < self.crossover_prob:
            if self.crossover_type == "uniform":
                return self._crossover_uniform(parent1, parent2)
            elif self.crossover_type == "two_point":
                return self._crossover_two_point(parent1, parent2)
        return parent1[:], parent2[:]

    def _mutate_resetting(self, chromosome: List[int]) -> List[int]:
        """
        Đột biến thiết lập lại ngẫu nhiên (Random Resetting Mutation - Toán tử chuẩn đồ án):
        Cơ chế: Mutation Probability per Individual (Xác suất đột biến áp dụng trên toàn bộ cá thể).
        Nếu biến cố đột biến xảy ra với xác suất Pm (self.mutation_prob), thuật toán chọn ngẫu nhiên
        một công việc (job) và phân công lại sang một máy khác máy hiện tại (m' != m).
        Lưu ý nghiên cứu: Đây là xác suất đột biến trên từng cá thể (Individual-level mutation),
        không phải xác suất độc lập trên từng gen (Gene-level mutation).
        """
        mutated = chromosome[:]
        num_machines = self.problem.num_machines
        if num_machines <= 1:
            return mutated

        if self.rng.random() < self.mutation_prob:
            # Chọn ngẫu nhiên một gen để thay đổi sang máy khác
            job_idx = self.rng.randint(0, len(mutated) - 1)
            current_machine = mutated[job_idx]
            other_machines = [m for m in range(num_machines) if m != current_machine]
            mutated[job_idx] = self.rng.choice(other_machines)

        return mutated

    def _mutate_swap(self, chromosome: List[int]) -> List[int]:
        """
        Đột biến hoán đổi máy được phân công giữa hai công việc (Swap Mutation - Phục vụ khảo sát):
        Cơ chế: Mutation Probability per Individual (Xác suất đột biến áp dụng trên toàn bộ cá thể).
        Nếu biến cố đột biến xảy ra với xác suất Pm (self.mutation_prob), thuật toán chọn ngẫu nhiên
        hai công việc khác nhau (idx1, idx2) và hoán đổi máy phân công cho nhau (a_{j1} <-> a_{j2}).
        Lưu ý: Không làm thay đổi thứ tự trên máy vì lịch trình luôn sắp Job ID tăng dần qua build_schedule.
        """
        mutated = chromosome[:]
        length = len(mutated)
        if length >= 2 and self.rng.random() < self.mutation_prob:
            idx1, idx2 = self.rng.sample(range(length), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated

    def _mutate(self, chromosome: List[int]) -> List[int]:
        """Điều phối đột biến dựa trên mutation_type."""
        if self.mutation_type == "resetting":
            return self._mutate_resetting(chromosome)
        elif self.mutation_type == "swap":
            return self._mutate_swap(chromosome)
        return chromosome[:]

    def solve(
        self,
        evaluator: Optional[Evaluator] = None,
        callback: Optional[Callable[[int, float, List[int]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi tìm kiếm nghiệm bằng Giải thuật Di truyền.

        Args:
            evaluator: Bộ đánh giá Evaluator dùng chung. Nếu không truyền, sẽ tự tạo mới.
            callback: Hàm callback tùy chọn nhận (generation_idx, best_so_far_makespan, best_assignment).
                      Dùng để cập nhật giao diện người dùng GUI hoặc thanh tiến trình.

        Returns:
            Dict[str, Any]: Dictionary kết quả chuẩn thỏa mãn validate_result_contract:
                - "assignment": List[int] phương án phân công tốt nhất.
                - "makespan": float Makespan nhỏ nhất tìm được.
                - "schedule": List[Tuple] lịch trình chi tiết cho biểu đồ Gantt.
                - "convergence": List[float] đúng 200 điểm hội tụ Best-so-far lũy kế.
                - "runtime": float thời gian chạy thuần túy (giây).
                - "evaluation_count": int số lần gọi evaluate() thực tế (10,000 NFE).
        """
        # Ngân sách mặc định: population_size + max_generations * population_size
        target_budget = self.population_size + self.max_generations * self.population_size
        if evaluator is not None:
            if evaluator.evaluation_count != 0:
                raise ValueError(
                    f"Evaluator đã được sử dụng trước đó (evaluation_count = {evaluator.evaluation_count} != 0). "
                    "Mỗi lượt chạy thuật toán yêu cầu một Evaluator mới với ngân sách độc lập."
                )
            if evaluator.problem is not self.problem:
                raise ValueError("Evaluator được truyền vào không gắn với cùng đối tượng UPMSProblem của GA.")
            if evaluator.max_evaluations != target_budget:
                raise ValueError(
                    f"Ngân sách Evaluator ({evaluator.max_evaluations} NFE) không khớp với ngân sách của GA ({target_budget} NFE)."
                )
        else:
            evaluator = Evaluator(self.problem, max_evaluations=target_budget)

        total_callback_time = 0.0
        start_time = time.perf_counter()

        # BƯỚC 1: KHỞI TẠO QUẦN THỂ BAN ĐẦU (Thế hệ 0)
        # Quần thể lưu dạng danh sách các tuple: (chromosome, makespan)
        population: List[Tuple[List[int], float]] = []
        for _ in range(self.population_size):
            chrom = self._create_random_chromosome()
            ms = evaluator.evaluate(chrom)  # Tiêu thụ 1 NFE
            population.append((chrom, ms))

        # Sắp xếp quần thể thế hệ 0 theo Makespan tăng dần
        population.sort(key=lambda ind: ind[1])
        best_chromosome, best_makespan = population[0][0][:], population[0][1]

        # Khởi tạo danh sách hội tụ Best-so-far (Điểm 0 sau khởi tạo)
        convergence: List[float] = [best_makespan]

        if callback is not None:
            cb_start = time.perf_counter()
            callback(0, best_makespan, best_chromosome)
            total_callback_time += time.perf_counter() - cb_start

        # BƯỚC 2: VÒNG LẶP TIẾN HÓA CÁC THẾ HỆ (Thế hệ 1 đến max_generations)
        for gen in range(1, self.max_generations + 1):
            offspring_chromosomes: List[List[int]] = []

            # Sinh đúng số cá thể con mới bằng kích thước quần thể (50 cá thể)
            while len(offspring_chromosomes) < self.population_size:
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)

                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                offspring_chromosomes.append(c1)
                if len(offspring_chromosomes) < self.population_size:
                    offspring_chromosomes.append(c2)

            # Đánh giá toàn bộ 50 cá thể con mới sinh (tiêu thụ đúng 50 NFE)
            offspring_population: List[Tuple[List[int], float]] = []
            for chrom in offspring_chromosomes:
                ms = evaluator.evaluate(chrom)
                offspring_population.append((chrom, ms))

            # BƯỚC 3: CHIẾN LƯỢC SỐNG SÓT ELITISM (Generational Replacement with Elitism)
            # Lấy cá thể tinh hoa (Elite) từ thế hệ trước mà KHÔNG đánh giá lại
            elite_pool: List[Tuple[List[int], float]] = population[:self.elite_count]

            # Bể ứng viên: 50 con mới + elite_count cá thể tinh hoa (tổng cộng 51 ứng viên nếu elite_count=1)
            candidate_pool = offspring_population + elite_pool

            # Sắp xếp bể ứng viên theo Makespan tăng dần
            candidate_pool.sort(key=lambda ind: ind[1])

            # Chọn top 50 cá thể tốt nhất để lập thành quần thể thế hệ mới
            population = candidate_pool[:self.population_size]

            # Cập nhật Best-so-far toàn cục
            gen_best_chrom, gen_best_ms = population[0][0], population[0][1]
            if gen_best_ms < best_makespan:
                best_makespan = gen_best_ms
                best_chromosome = gen_best_chrom[:]

            # Ghi nhận điểm hội tụ Best-so-far (đảm bảo tính đơn điệu không tăng)
            convergence.append(best_makespan)

            if callback is not None:
                cb_start = time.perf_counter()
                callback(gen, best_makespan, best_chromosome)
                total_callback_time += time.perf_counter() - cb_start

        total_elapsed = time.perf_counter() - start_time
        # Thời gian tìm kiếm thuần túy của thuật toán (loại trừ toàn bộ thời gian callback I/O và GUI)
        elapsed_time = max(0.0, total_elapsed - total_callback_time)

        # Xây dựng lịch trình Gantt chi tiết từ nghiệm tốt nhất
        schedule = self.problem.build_schedule(best_chromosome)

        result: Dict[str, Any] = {
            "assignment": best_chromosome,
            "makespan": best_makespan,
            "schedule": schedule,
            "convergence": convergence,
            "runtime": elapsed_time,
            "evaluation_count": evaluator.evaluation_count,
        }

        # Kiểm chứng nghiêm ngặt tính hợp lệ của kết quả theo hợp đồng dữ liệu đồ án
        validate_result_contract(
            result=result,
            problem=self.problem,
            expected_convergence_len=1 + self.max_generations,
            max_evaluations=target_budget,
        )

        return result


def run_ga(
    problem: UPMSProblem,
    params: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[int, float, List[int]], None]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hàm giao diện cấp cao (API bàn giao cho TV3 - GUI và Benchmark Runner).

    Args:
        problem: Thể hiện bài toán UPMSProblem.
        params: Dictionary tham số tùy chỉnh từ GUI hoặc Benchmark (nếu None sẽ dùng chuẩn config).
        callback: Hàm nhận cập nhật tiến độ (thế hệ, makespan tốt nhất, assignment tốt nhất).
        seed: Seed ngẫu nhiên để tái lập thực nghiệm.

    Returns:
        Dict[str, Any]: Kết quả chuẩn đã được thẩm định qua validate_result_contract().
    """
    p = params or {}

    pop_size = p.get("population_size", POPULATION_SIZE)
    max_gen = p.get("max_generations", MAX_GENERATIONS)
    pc = p.get("crossover_prob", p.get("pc", GA_PC))
    pm = p.get("mutation_prob", p.get("pm", GA_PM))
    k = p.get("tournament_k", p.get("k", GA_TOURNAMENT_K))
    elite = p.get("elite_count", GA_ELITE_COUNT)
    cx_type = p.get("crossover_type", GA_CROSSOVER_TYPE)
    mut_type = p.get("mutation_type", GA_MUTATION_TYPE)
    effective_seed = p.get("seed", seed)

    ga = GeneticAlgorithm(
        problem=problem,
        population_size=pop_size,
        max_generations=max_gen,
        crossover_prob=pc,
        mutation_prob=pm,
        tournament_k=k,
        elite_count=elite,
        crossover_type=cx_type,
        mutation_type=mut_type,
        seed=effective_seed,
    )

    return ga.solve(callback=callback)
