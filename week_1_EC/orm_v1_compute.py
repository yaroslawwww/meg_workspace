#!/usr/bin/env python3
"""
Каноническая непрерывная задача оптимального разбиения множеств (ОРМ).
Постановка: Задача А2 / В2 (монография Е.М. Киселевой и Л.С. Коряшкиной, разд. 1.1.1.1, 6.3.1).

Архитектура: 25 модульных функций, сгруппированных по 6 математическим блокам:
- Модуль 1: Аналитическая нормировка сложности (функции 1–4)
- Модуль 2: Комбинаторное масштабирование и сборка матрицы V (функции 5–10)
- Модуль 3: Геометрия Вороного в L_2([0, T]) (функции 11–16)
- Модуль 4: Аналитический барицентрический пересчет центров (функции 17–20)
- Модуль 5: Контроль сходимости и метрики распределения (функции 21–24)
- Модуль 6: Канонизация и постобработка (функция 25)
"""

import os
import re
import sys
import time
import math
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
import polars as pl
import torch

# Фиксация воспроизводимости
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ORM_TASK_A2_STRICT")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DB_DIR = WORKSPACE_DIR / "04_processed_db"
ORM_RESULTS_DIR = PROCESSED_DB_DIR / "orm_results"

DATASETS = {
    0: {
        "name": "Type 1 (HSE Bigrams) - Chistova Alena",
        "parquet_path": PROCESSED_DB_DIR / "type1_features/type1_sub10_chistova_alena_entropy_db.parquet",
        "prefix": "type1_sub10_chistova"
    },
    1: {
        "name": "Type 2 (Ushakov Drowsiness) - Bainbridge Emily",
        "parquet_path": PROCESSED_DB_DIR / "type2_features/type2_bainbridge_emily_entropy_db.parquet",
        "prefix": "type2_bainbridge"
    },
    2: {
        "name": "Type 3 (Spanish BCBL) - Subject 01 Block 1",
        "parquet_path": PROCESSED_DB_DIR / "type3_features/type3_bcbl_01_block1_entropy_db.parquet",
        "prefix": "type3_bcbl_01"
    }
}

K_GRID = [20, 35, 45, 50, 60 , 65,  70, 80,90, 100, 150, 200, 250, 300]
MAX_ITERATIONS = 400
TOL_REL_J = 1e-5
LABEL_STABILITY_TOL = 1e-3


# =============================================================================
# МОДУЛЬ 1. Аналитическая нормировка сложности (EC-математика)
# =============================================================================

def calculate_jensen_shannon_q0(num_patterns: int) -> float:
    """Функция 1: Вычисление константы Q_0 для дивергенции Йенсена-Шеннона."""
    n_float = float(num_patterns)
    term1 = ((n_float + 1.0) / n_float) * math.log2(n_float + 1.0)
    term2 = 2.0 * math.log2(2.0 * n_float)
    term3 = math.log2(n_float)
    return -2.0 / (term1 - term2 + term3)


def evaluate_jensen_shannon_divergence(
    prob_dist: np.ndarray, 
    uniform_dist: np.ndarray
) -> float:
    """Функция 2: Расчет дивергенции Йенсена-Шеннона D_JS между распределениями."""
    p_mix = 0.5 * (prob_dist + uniform_dist)
    shannon_p = -float(np.sum(prob_dist * np.log2(prob_dist)))
    shannon_pe = -float(np.sum(uniform_dist * np.log2(uniform_dist)))
    shannon_mix = -float(np.sum(p_mix * np.log2(p_mix)))
    return shannon_mix - 0.5 * shannon_p - 0.5 * shannon_pe


@lru_cache(maxsize=16)
def get_theoretical_c_max(permutation_order_d: int) -> float:
    """Функция 3: Аналитический супремум сложности C*_max(N) по Мартину-Пластино-Россо."""
    num_patterns = math.factorial(permutation_order_d)
    q0 = calculate_jensen_shannon_q0(num_patterns)
    uniform_dist = np.full(num_patterns, 1.0 / float(num_patterns), dtype=np.float64)

    p_grid = np.linspace(1.0 / float(num_patterns) + 1e-5, 1.0 - 1e-5, 2000)
    c_values = np.empty_like(p_grid)

    for idx, p_val in enumerate(p_grid):
        p_other = (1.0 - p_val) / (float(num_patterns) - 1.0)
        p_arr = np.full(num_patterns, p_other, dtype=np.float64)
        p_arr[0] = p_val
        
        d_js = evaluate_jensen_shannon_divergence(p_arr, uniform_dist)
        h_norm = (-float(np.sum(p_arr * np.log2(p_arr)))) / math.log2(float(num_patterns))
        c_values[idx] = q0 * d_js * h_norm

    return float(np.max(c_values))


def normalize_complexity_value(raw_complexity: np.ndarray, c_max: float) -> np.ndarray:
    """Функция 4: Приведение статистической сложности к интервалу [0, 1]."""
    return raw_complexity / c_max


# =============================================================================
# МОДУЛЬ 2. Комбинаторное масштабирование и сборка матрицы V
# =============================================================================

def parse_window_metadata_tag(window_tag_string: str) -> tuple[float, int]:
    """Функция 5: Извлечение длительности окна в секундах и порядка d из тега."""
    match = re.search(r"w_([0-9.]+)s_m(\d+)", window_tag_string)
    if not match:
        raise ValueError(f"Некорректный тег масштаба: {window_tag_string}")
    return float(match.group(1)), int(match.group(2))


def compute_window_sample_count(window_sec: float, sampling_freq: float) -> int:
    """Функция 6: Преобразование физической длительности окна в количество отсчетов W."""
    return int(round(window_sec * sampling_freq))


def compute_combinatorial_sampling_weight(
    window_samples: int, 
    permutation_order_d: int
) -> float:
    """Функция 7: Расчет комбинаторного фактора плотности фазового пространства sqrt(N_W / d!)."""
    n_effective = float(window_samples - permutation_order_d + 1)
    n_patterns = float(math.factorial(permutation_order_d))
    return math.sqrt(n_effective / n_patterns)


def compute_phase_space_displacement(
    pe_left: np.ndarray,
    pe_right: np.ndarray,
    sc_left_norm: np.ndarray,
    sc_right_norm: np.ndarray
) -> np.ndarray:
    """Функция 8: Евклидово расстояние между левым и правым состояниями на плоскости EC."""
    d_pe_sq = (pe_left - pe_right) ** 2
    d_sc_sq = (sc_left_norm - sc_right_norm) ** 2
    return np.sqrt(d_pe_sq + d_sc_sq)


def scale_displacement_trajectory(
    raw_displacement: np.ndarray, 
    sampling_weight: float
) -> np.ndarray:
    """Функция 9: Масштабирование фазового скачка на комбинаторный вес."""
    return raw_displacement * sampling_weight


def build_multiscale_feature_matrix(
    parquet_path: Path, 
    device: torch.device
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[dict], float]:
    """Функция 10: Сборка итогового тензора V [M, T] с метаданными."""
    logger.info(f"Чтение базы энтропий: {parquet_path.name}")
    df = pl.read_parquet(parquet_path).sort(["channel", "sample_idx"])

    channels = sorted(df["channel"].unique().to_list())
    num_channels = len(channels)

    pe_left_cols = [c for c in df.columns if c.startswith("pe_left_")]
    window_tags = [c.replace("pe_left_", "") for c in pe_left_cols]
    num_windows = len(window_tags)

    first_ch = df.filter(pl.col("channel") == channels[0])
    sample_indices = first_ch["sample_idx"].to_numpy()
    time_sec = first_ch["time_sec"].to_numpy().astype(np.float32)
    t_points = len(time_sec)

    diff_time = np.diff(time_sec[:500])
    diff_sample = np.diff(sample_indices[:500])
    sampling_freq = float(np.median(diff_sample[diff_time > 0] / diff_time[diff_time > 0]))

    total_objects = num_channels * num_windows
    logger.info(f"Пространство Omega: {num_channels} каналов x {num_windows} окон = {total_objects} объектов | T = {t_points} | fs = {sampling_freq:.1f} Гц")

    window_configs = []
    for w_tag in window_tags:
        w_sec, d_order = parse_window_metadata_tag(w_tag)
        w_samples = compute_window_sample_count(w_sec, sampling_freq)
        c_max = get_theoretical_c_max(d_order)
        weight = compute_combinatorial_sampling_weight(w_samples, d_order)
        window_configs.append((w_tag, c_max, weight))

    matrix_v = np.empty((total_objects, t_points), dtype=np.float32)
    object_catalog = []
    row_idx = 0

    for ch_idx, ch in enumerate(channels):
        ch_slice = slice(ch_idx * t_points, (ch_idx + 1) * t_points)
        for w_tag, c_max, weight in window_configs:
            pe_l = df[f"pe_left_{w_tag}"][ch_slice].to_numpy()
            pe_r = df[f"pe_right_{w_tag}"][ch_slice].to_numpy()
            c_l = df[f"c_left_{w_tag}"][ch_slice].to_numpy()
            c_r = df[f"c_right_{w_tag}"][ch_slice].to_numpy()

            sc_l_norm = normalize_complexity_value(c_l, c_max)
            sc_r_norm = normalize_complexity_value(c_r, c_max)

            disp = compute_phase_space_displacement(pe_l, pe_r, sc_l_norm, sc_r_norm)
            matrix_v[row_idx, :] = scale_displacement_trajectory(disp, weight)

            object_catalog.append({"channel": ch, "window_tag": w_tag})
            row_idx += 1

    del df
    tensor_v = torch.from_numpy(matrix_v).to(device)
    total_energy = float(torch.sum(tensor_v ** 2).item())
    logger.info(f"Тензор V загружен: {tensor_v.shape} ({tensor_v.element_size() * tensor_v.nelement() / (1024**2):.2f} MB)")

    return tensor_v, time_sec, sample_indices, object_catalog, total_energy


# =============================================================================
# МОДУЛЬ 3. Геометрия Вороного в L_2([0, T]) (Шаг разметки)
# =============================================================================

def compute_trajectory_energies(feature_tensor: torch.Tensor) -> torch.Tensor:
    """Функция 11: Вычисление интегральной кинетической энергии строк тензора ||V_x||^2."""
    return torch.sum(feature_tensor ** 2, dim=1)


def initialize_deterministic_centers(
    feature_tensor: torch.Tensor, 
    trajectory_energies: torch.Tensor, 
    K: int
) -> torch.Tensor:
    """Функция 12: Детерминированная квантильная инициализация центров tau_k."""
    m_objects, t_points = feature_tensor.shape
    centers = torch.zeros((K, t_points), dtype=feature_tensor.dtype, device=feature_tensor.device)
    sorted_indices = torch.argsort(trajectory_energies)
    quad_indices = torch.linspace(0, m_objects - 1, K, dtype=torch.long, device=feature_tensor.device)

    for k in range(K):
        centers[k] = feature_tensor[sorted_indices[quad_indices[k]]]

    return centers


def compute_squared_euclidean_distance_matrix(
    feature_tensor: torch.Tensor, 
    centers_tensor: torch.Tensor
) -> torch.Tensor:
    """Функция 13: Векторизованный расчет матрицы квадратов расстояний D = ||V_x - tau_k||^2."""
    v_sq_norms = torch.sum(feature_tensor ** 2, dim=1, keepdim=True)        # [M, 1]
    centers_sq = torch.sum(centers_tensor ** 2, dim=1, keepdim=True).T      # [1, K]
    cross_product = torch.matmul(feature_tensor, centers_tensor.T)          # [M, K]
    return torch.clamp_min(v_sq_norms - 2.0 * cross_product + centers_sq, 0.0)


def assign_voronoi_cells(distance_matrix: torch.Tensor) -> torch.Tensor:
    """Функция 14: Разметка ячеек Вороного: labels = argmin_k D[x, k]."""
    return torch.argmin(distance_matrix, dim=1)


def extract_pointwise_min_distances(
    distance_matrix: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """Функция 15: Извлечение квадрата расстояния до назначенного центра: D[x, labels[x]]."""
    return torch.gather(distance_matrix, 1, labels.unsqueeze(1)).squeeze(1)


def compute_primal_functional_value(min_distances: torch.Tensor) -> float:
    """Функция 16: Значение функционала Задачи А2: J = sum(min_distances)."""
    return float(torch.sum(min_distances).item())


# =============================================================================
# МОДУЛЬ 4. Аналитический барицентрический пересчет центров
# =============================================================================

def compute_cluster_sizes(labels: torch.Tensor, K: int) -> torch.Tensor:
    """Функция 17: Подсчет числа объектов в каждом кластере N_k = |Omega_k|."""
    return torch.bincount(labels, minlength=K)


def aggregate_cluster_trajectories(
    feature_tensor: torch.Tensor, 
    labels: torch.Tensor, 
    K: int
) -> torch.Tensor:
    """Функция 18: Агрегация сумм векторов по режимам: S_k(t) = sum_{x in Omega_k} V_x(t)."""
    _, t_points = feature_tensor.shape
    cluster_sums = torch.zeros((K, t_points), dtype=feature_tensor.dtype, device=feature_tensor.device)  # <-- ЭТА СТРОКА
    cluster_sums.index_add_(0, labels, feature_tensor)
    return cluster_sums


def compute_barycentric_centers(
    cluster_sums: torch.Tensor, 
    cluster_sizes: torch.Tensor
) -> torch.Tensor:
    """Функция 19: Аналитический пересчет барицентров tau_k = S_k / N_k (условие grad J = 0)."""
    valid_counts = cluster_sizes.unsqueeze(1).clamp_min(1).float()
    return cluster_sums / valid_counts


def remedy_empty_clusters(
    feature_tensor: torch.Tensor, 
    centers_tensor: torch.Tensor, 
    cluster_sizes: torch.Tensor, 
    min_distances: torch.Tensor
) -> torch.Tensor:
    """Функция 20: Восстановление пустых ячеек Вороного вектором с максимальной ошибкой."""
    empty_indices = torch.where(cluster_sizes == 0)[0]
    num_empty = len(empty_indices)
    if num_empty > 0:
        _, worst_point_indices = torch.topk(min_distances, k=num_empty, largest=True)
        for idx, empty_k in enumerate(empty_indices):
            centers_tensor[empty_k] = feature_tensor[worst_point_indices[idx]]
    return centers_tensor


# =============================================================================
# МОДУЛЬ 5. Контроль сходимости и метрики распределения
# =============================================================================

def compute_relative_loss_reduction(prev_J: float, curr_J: float) -> float:
    """Функция 21: Расчет относительного уменьшения функционала Delta J_rel."""
    return abs(prev_J - curr_J) / (prev_J + 1e-12)


def compute_label_churn_fraction(
    prev_labels: torch.Tensor, 
    curr_labels: torch.Tensor, 
    M: int
) -> float:
    """Функция 22: Доля объектов, сменивших режим на текущей итерации."""
    return float(torch.sum(prev_labels != curr_labels).item()) / float(M)


def evaluate_convergence_stopping_condition(
    iteration: int, 
    delta_J_rel: float, 
    churn_fraction: float, 
    tol_J: float = TOL_REL_J
) -> bool:
    """Функция 23: Оценка условий останова алгоритма."""
    return (iteration > 2) and (delta_J_rel < tol_J)


def calculate_cluster_occupancy_entropy(cluster_sizes: torch.Tensor, M: int) -> float:
    """Функция 24: Энтропия Шеннона распределения объемов кластеров."""
    prob_dist = cluster_sizes.cpu().numpy() / float(M)
    return -float(np.sum(prob_dist * np.log2(prob_dist + 1e-12)))


# =============================================================================
# МОДУЛЬ 6. Канонизация и постобработка
# =============================================================================

def sort_modes_canonically_by_energy(
    centers_tensor: torch.Tensor, 
    labels_vector: torch.Tensor, 
    K: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Функция 25: Упорядочивание режимов по энергии (режим покоя k = 0)."""
    energies = torch.sum(centers_tensor ** 2, dim=1).cpu().numpy()
    permutation = np.argsort(energies)

    reorder_map = torch.empty(K, dtype=torch.long, device=centers_tensor.device)
    reorder_map[permutation] = torch.arange(K, dtype=torch.long, device=centers_tensor.device)

    sorted_centers = centers_tensor[permutation]
    reindexed_labels = reorder_map[labels_vector]
    sorted_bincounts = torch.bincount(reindexed_labels, minlength=K).cpu().numpy()

    return sorted_centers, reindexed_labels, sorted_bincounts


# =============================================================================
# Исполнительный цикл и точка входа (Pipeline Runner)
# =============================================================================

def solve_orm_task_a2_single_k(
    V: torch.Tensor, 
    K: int, 
    total_energy: float, 
    max_iter: int = MAX_ITERATIONS
):
    """Исполнение Задачи А2 для одного значения K."""
    M, _ = V.shape
    start_time = time.time()

    energies = compute_trajectory_energies(V)
    centers = initialize_deterministic_centers(V, energies, K)

    prev_labels = torch.zeros(M, dtype=torch.long, device=V.device) - 1
    prev_J = float("inf")

    training_logs = []
    converged = False
    final_iteration = 0

    for iteration in range(1, max_iter + 1):
        iter_start = time.time()

        # Шаг 1. Разметка ячеек Вороного
        dist_matrix = compute_squared_euclidean_distance_matrix(V, centers)
        labels = assign_voronoi_cells(dist_matrix)
        min_dists = extract_pointwise_min_distances(dist_matrix, labels)
        curr_J = compute_primal_functional_value(min_dists)

        # Шаг 2. Аналитическое обновление барицентров
        cluster_sizes = compute_cluster_sizes(labels, K)
        cluster_sums = aggregate_cluster_trajectories(V, labels, K)
        centers = compute_barycentric_centers(cluster_sums, cluster_sizes)
        centers = remedy_empty_clusters(V, centers, cluster_sizes, min_dists)

        # Шаг 3. Диагностика сходимости
        delta_J_rel = compute_relative_loss_reduction(prev_J, curr_J) if iteration > 1 else 1.0
        churn = compute_label_churn_fraction(prev_labels, labels, M)

        prev_labels = labels.clone()
        prev_J = curr_J

        entropy_val = calculate_cluster_occupancy_entropy(cluster_sizes, M)
        cluster_sizes_np = cluster_sizes.cpu().numpy()

        iter_log = {
            "k_clusters": K,
            "iteration": iteration,
            "loss_primal_J": curr_J,
            "delta_j_rel": delta_J_rel,
            "switched_ratio": churn,
            "cluster_entropy": entropy_val,
            "min_cluster_size": int(np.min(cluster_sizes_np)),
            "max_cluster_size": int(np.max(cluster_sizes_np)),
            "variance_explained": float(1.0 - (curr_J / total_energy)),
            "step_sec": time.time() - iter_start
        }
        training_logs.append(iter_log)

        if evaluate_convergence_stopping_condition(iteration, delta_J_rel, churn):
            converged = True
            final_iteration = iteration
            break

    if not converged:
        final_iteration = max_iter

    total_duration = time.time() - start_time

    # Шаг 4. Канонизация
    sorted_centers, sorted_labels, sorted_counts = sort_modes_canonically_by_energy(centers, labels, K)

    summary = {
        "k_clusters": K,
        "final_loss_primal": training_logs[-1]["loss_primal_J"],
        "converged": converged,
        "iterations_count": final_iteration,
        "total_time_sec": total_duration,
        "baseline_ratio": float(sorted_counts[0]) / float(M),
        "final_cluster_entropy": training_logs[-1]["cluster_entropy"],
        "mode": "A2"
    }

    logger.info(
        f"[A2] K = {K:3d} | Время: {total_duration:5.2f} с | "
        f"Итераций: {final_iteration:2d} | "
        f"J = {summary['final_loss_primal']:.4e} | "
        f"R^2 = {training_logs[-1]['variance_explained'] * 100:.2f}% | "
        f"Покой tau_0 = {summary['baseline_ratio'] * 100:.1f}%"
    )

    return summary, training_logs, sorted_labels.cpu().numpy().astype(np.int32), min_dists.cpu().numpy().astype(np.float32), sorted_centers.cpu().numpy().astype(np.float32)


def main():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        dataset_code = int(os.environ["SLURM_ARRAY_TASK_ID"])
    elif len(sys.argv) > 1:
        dataset_code = int(sys.argv[1])
    else:
        logger.warning("dataset_code не указан. Запуск 0 (Тип 1).")
        dataset_code = 0

    if dataset_code not in DATASETS:
        raise ValueError(f"Недопустимый dataset_code: {dataset_code}.")

    cfg = DATASETS[dataset_code]
    prefix = cfg["prefix"]
    logger.info(f"=== Запуск ОРМ А2 (модульная архитектура: 25 функций) | Датасет: {cfg['name']} ===")

    if not cfg["parquet_path"].exists():
        raise FileNotFoundError(f"Файл не найден: {cfg['parquet_path']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Вычислительное устройство: {device}")

    target_output_dir = ORM_RESULTS_DIR / prefix
    target_output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = target_output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    tensor_v, time_sec, sample_indices, object_catalog, total_energy = build_multiscale_feature_matrix(
        cfg["parquet_path"], device
    )

    all_sweep_summaries = []
    all_training_logs = []
    all_memberships_frames = []

    m_total = len(object_catalog)
    channels_array = [obj["channel"] for obj in object_catalog]
    windows_array = [obj["window_tag"] for obj in object_catalog]

    for K_val in K_GRID:
        summary, logs, labels, dists, centers = solve_orm_task_a2_single_k(
            V=tensor_v,
            K=K_val,
            total_energy=total_energy,
            max_iter=MAX_ITERATIONS
        )

        all_sweep_summaries.append(summary)
        all_training_logs.extend(logs)

        df_mem_k = pl.DataFrame({
            "k_clusters": np.full(m_total, K_val, dtype=np.int32),
            "channel": channels_array,
            "window_tag": windows_array,
            "cluster_regime_id": labels,
            "dist_sq_to_center": dists
        })
        all_memberships_frames.append(df_mem_k)

        traj_dict = {"time_sec": time_sec, "sample_idx": sample_indices}
        for k_idx in range(K_val):
            traj_dict[f"tau_{k_idx}"] = centers[k_idx]

        traj_df = pl.DataFrame(traj_dict)
        traj_file_path = trajectories_dir / f"trajectories_k{K_val}.parquet"
        traj_df.write_parquet(traj_file_path, compression="snappy")

    # Сериализация баз данных
    df_sweep = pl.DataFrame(all_sweep_summaries)
    df_sweep.write_parquet(target_output_dir / "orm_v1_sweep_summary.parquet", compression="snappy")

    df_training = pl.DataFrame(all_training_logs)
    df_training.write_parquet(target_output_dir / "orm_v1_training_history.parquet", compression="snappy")

    df_memberships = pl.concat(all_memberships_frames)
    df_memberships.write_parquet(target_output_dir / "orm_v1_memberships.parquet", compression="snappy")

    logger.info(f"=== Расчет успешно завершен. Таблицы сохранены в: {target_output_dir} ===")


if __name__ == "__main__":
    main()
