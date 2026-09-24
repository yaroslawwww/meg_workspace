#!/usr/bin/env python3
"""
Модуль физико-математического анализа и анатомии функций режимов tau_k(t).

Анализирует свойства функций без навязывания гауссовости:
- Односторонняя область значений tau_k(t) >= 0.
- Фоновый пьедестал покоя (мода эмпирической плотности).
- Мера перемежаемости и разреженности энергии (кумулята энергии, индекс Джини).
- Морфология и асимметрия импульсов фазового перехода (FWHM, фронт vs спад).
- Соответствие ширины импульсов длительностям окон W из носителя Omega_k.

Архитектура: 28 атомарных функций, сгруппированных по 7 блокам.
"""

import os
import sys
import re
import math
import logging
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "legend.frameon": False,
    "figure.autolayout": True,
    "savefig.dpi": 300
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TAU_ANATOMY")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DB_DIR = WORKSPACE_DIR / "04_processed_db"
ORM_RESULTS_DIR = PROCESSED_DB_DIR / "orm_results"

DATASETS = {
    0: {
        "name": "Type 1 (HSE Bigrams) - Chistova Alena",
        "prefix": "type1_sub10_chistova"
    },
    1: {
        "name": "Type 2 (Ushakov Drowsiness) - Bainbridge Emily",
        "prefix": "type2_bainbridge"
    },
    2: {
        "name": "Type 3 (Spanish BCBL) - Subject 01 Block 1",
        "prefix": "type3_bcbl_01"
    }
}

# Репрезентативные K для детального препарирования функций
DEFAULT_ANALYTIC_K = [20, 50, 65, 100]


# =============================================================================
# МОДУЛЬ 1. Загрузка данных и извлечение траекторий (Функции 1–5)
# =============================================================================

def resolve_dataset_config(dataset_code: int, datasets_dict: dict, base_dir: Path) -> tuple[dict, Path, Path]:
    """Функция 1: Разрешение путей датасета и создание целевого каталога графиков."""
    if dataset_code not in datasets_dict:
        raise ValueError(f"Неизвестный dataset_code: {dataset_code}")
    cfg = datasets_dict[dataset_code]
    res_dir = base_dir / cfg["prefix"]
    if not res_dir.exists():
        raise FileNotFoundError(f"Каталог с результатами ОРМ не найден: {res_dir}")
    fig_dir = res_dir / "figures_tau_anatomy"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return cfg, res_dir, fig_dir


def parse_window_duration_seconds(window_tag: str) -> float:
    """Функция 2: Извлечение физической длительности окна в секундах из тега."""
    match = re.search(r"w_([0-9.]+)s", window_tag)
    return float(match.group(1)) if match else 0.0


def load_trajectories_table(trajectories_dir: Path, k_val: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Функция 3: Загрузка матрицы траекторий всех центров [K, T] и вектора времени."""
    traj_file = trajectories_dir / f"trajectories_k{k_val}.parquet"
    if not traj_file.exists():
        raise FileNotFoundError(f"Файл траекторий K={k_val} не найден: {traj_file}")
    df = pl.read_parquet(traj_file)
    time_sec = df["time_sec"].to_numpy().astype(np.float64)
    tau_cols = [c for c in df.columns if c.startswith("tau_")]
    tau_cols = sorted(tau_cols, key=lambda x: int(x.split("_")[1]))
    tau_mat = df.select(tau_cols).to_numpy().T.astype(np.float64)  # [K, T]
    return time_sec, tau_mat, tau_cols


def load_memberships_for_k(results_dir: Path, k_val: int) -> pl.DataFrame:
    """Функция 4: Загрузка принадлежности ячеек Вороного для конкретного K."""
    mem_file = results_dir / "orm_v1_memberships.parquet"
    if not mem_file.exists():
        raise FileNotFoundError(f"Файл memberships не найден: {mem_file}")
    df_all = pl.read_parquet(mem_file)
    return df_all.filter(pl.col("k_clusters") == k_val)


def extract_constituent_windows_of_cluster(df_mem_k: pl.DataFrame, k_id: int) -> list[float]:
    """Функция 5: Извлечение всех длительностей окон W, вошедших в подмножество Omega_k."""
    sub = df_mem_k.filter(pl.col("cluster_regime_id") == k_id)
    if len(sub) == 0:
        return []
    tags = sub["window_tag"].unique().to_list()
    return sorted([parse_window_duration_seconds(t) for t in tags])


# =============================================================================
# МОДУЛЬ 2. Анатомия распределений: пьедестал покоя и плотность (Функции 6–9)
# =============================================================================

def compute_empirical_density_and_pedestal(
    tau_series: np.ndarray, 
    num_bins: int = 120
) -> tuple[float, np.ndarray, np.ndarray]:
    """Функция 6: Расчет моды эмпирической плотности (физического фонового пьедестала)."""
    val_min = float(np.min(tau_series))
    val_max = float(np.max(tau_series))
    if math.isclose(val_min, val_max):
        return val_min, np.array([1.0]), np.array([val_min, val_max])

    counts, bin_edges = np.histogram(tau_series, bins=num_bins, density=True)
    max_bin_idx = int(np.argmax(counts))
    pedestal_mode = float(0.5 * (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return pedestal_mode, bin_centers, counts


def compute_intermittency_factor(
    tau_series: np.ndarray, 
    pedestal: float, 
    burst_threshold_ratio: float = 1.3
) -> dict:
    """Функция 7: Коэффициент перемежаемости (доля времени в покое vs фаза перестройки)."""
    burst_level = pedestal * burst_threshold_ratio
    total_points = len(tau_series)
    quiet_mask = tau_series <= burst_level
    quiet_fraction = float(np.sum(quiet_mask)) / float(total_points)
    burst_fraction = 1.0 - quiet_fraction
    max_burst_ratio = float(np.max(tau_series)) / (pedestal + 1e-12)
    
    return {
        "pedestal": pedestal,
        "burst_threshold": burst_level,
        "quiet_time_fraction": quiet_fraction,
        "burst_time_fraction": burst_fraction,
        "max_burst_to_pedestal": max_burst_ratio
    }


def compute_non_gaussian_tail_quantiles(tau_series: np.ndarray) -> dict:
    """Функция 8: Квантильный срез правого хвоста распределения (без предположений о нормальности)."""
    q_vals = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    res = np.quantile(tau_series, q_vals)
    return {f"q_{int(q * 1000)}": float(v) for q, v in zip(q_vals, res)}


def render_empirical_density_axis(
    ax: plt.Axes, 
    bin_centers: np.ndarray, 
    density: np.ndarray, 
    pedestal: float, 
    k_id: int, 
    color: str
):
    """Функция 9: Отрисовка эмпирической плотности p(tau_k) с отметкой пьедестала."""
    ax.fill_between(bin_centers, density, color=color, alpha=0.35)
    ax.plot(bin_centers, density, color=color, linewidth=1.2)
    ax.axvline(pedestal, color="black", linestyle="--", linewidth=1.0, label=f"Пьедестал: {pedestal:.3f}")
    ax.set_title(f"Плотность p(tau_{k_id})")
    ax.set_xlabel("Значение tau")
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=8)


# =============================================================================
# МОДУЛЬ 3. Энергетическая кумулята и мера разреженности (Функции 10–13)
# =============================================================================

def compute_energy_accumulation_curve(tau_series: np.ndarray) -> np.ndarray:
    """Функция 10: Интеграл накопления кинетической энергии E(t) = int_0^t tau^2 dt / int_0^T tau^2 dt."""
    instant_energy = tau_series ** 2
    cum_energy = np.cumsum(instant_energy)
    total_energy = cum_energy[-1]
    if total_energy <= 0.0:
        return np.linspace(0.0, 1.0, len(tau_series))
    return cum_energy / total_energy


def compute_energy_gini_sparsity(cum_energy: np.ndarray) -> float:
    """Функция 11: Коэффициент Джини распределения энергии (мера временной компактности перестроек)."""
    n = len(cum_energy)
    u_grid = np.linspace(0.0, 1.0, n)
    # Площадь под кумулятой
    area_under_curve = float(np.trapz(cum_energy, u_grid))
    # Если процесс абсолютно равномерен (диагональ), area = 0.5 -> Gini = 0.0
    # Если вся энергия в начале или конце -> Gini -> 1.0
    gini = 1.0 - 2.0 * min(area_under_curve, 1.0 - area_under_curve)
    return float(np.clip(gini, 0.0, 1.0))


def render_cumulative_energy_axis(
    ax: plt.Axes, 
    time_normalized: np.ndarray, 
    cum_energy: np.ndarray, 
    k_id: int, 
    color: str, 
    gini: float
):
    """Функция 12: Отрисовка кривой накопления энергии E_k(t) против идеальной диагонали."""
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linestyle=":", label="Равномерный дрейф")
    ax.plot(time_normalized, cum_energy, color=color, linewidth=1.2, label=f"tau_{k_id} (Gini={gini:.2f})")
    ax.set_title(f"Локализация энергии tau_{k_id}")
    ax.set_xlabel("Нормированное время t / T")
    ax.set_ylabel("Доля энергии E(t) / E_total")
    ax.legend(fontsize=8, loc="upper left")


def render_all_cumulatives_stack(
    ax: plt.Axes, 
    time_normalized: np.ndarray, 
    cum_energies: list[np.ndarray], 
    selected_indices: list[int]
):
    """Функция 13: Стекированное сравнение кумулят для группы ключевых режимов."""
    cmap = mpl.colormaps["turbo"].resampled(len(selected_indices))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0)
    for i, (k_idx, cum_e) in enumerate(zip(selected_indices, cum_energies)):
        ax.plot(time_normalized, cum_e, color=cmap(i), linewidth=0.9, label=f"k={k_idx}")
    ax.set_title("Сравнение локализации энергии по режимам")
    ax.set_xlabel("t / T")
    ax.set_ylabel("Кумулята E(t)")
    ax.legend(ncol=2, fontsize=8)


# =============================================================================
# МОДУЛЬ 4. Морфология отдельных импульсов: FWHM и асимметрия (Функции 14–18)
# =============================================================================

def locate_prominent_local_maxima(
    tau_series: np.ndarray, 
    pedestal: float, 
    min_prominence_ratio: float = 1.5,
    min_distance_points: int = 50
) -> list[int]:
    """Функция 14: Поиск вершин импульсов, существенно превышающих фоновый пьедестал."""
    threshold = pedestal * min_prominence_ratio
    peaks = []
    n_points = len(tau_series)

    for i in range(1, n_points - 1):
        val = tau_series[i]
        if val > threshold and val >= tau_series[i - 1] and val > tau_series[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance_points:
                peaks.append(i)
            else:
                if val > tau_series[peaks[-1]]:
                    peaks[-1] = i
    return peaks


def compute_pulse_fwhm_and_geometry(
    tau_series: np.ndarray, 
    time_sec: np.ndarray, 
    peak_idx: int, 
    pedestal: float
) -> dict | None:
    """Функция 15: Расчет полуширины на полувысоте (FWHM), крутизны фронта и асимметрии импульса."""
    peak_val = tau_series[peak_idx]
    peak_time = time_sec[peak_idx]
    half_height = pedestal + 0.5 * (peak_val - pedestal)

    # Поиск левой границы полувысоты
    left_idx = peak_idx
    while left_idx > 0 and tau_series[left_idx] > half_height:
        left_idx -= 1
    t_left = time_sec[left_idx]

    # Поиск правой границы полувысоты
    right_idx = peak_idx
    n_max = len(tau_series) - 1
    while right_idx < n_max and tau_series[right_idx] > half_height:
        right_idx += 1
    t_right = time_sec[right_idx]

    fwhm_duration = t_right - t_left
    rise_duration = peak_time - t_left
    decay_duration = t_right - peak_time

    if math.isclose(decay_duration, 0.0):
        asymmetry_ratio = 1.0
    else:
        asymmetry_ratio = rise_duration / decay_duration

    return {
        "peak_idx": peak_idx,
        "peak_time_sec": peak_time,
        "peak_val": peak_val,
        "half_height": half_height,
        "t_left": t_left,
        "t_right": t_right,
        "fwhm_sec": fwhm_duration,
        "rise_sec": rise_duration,
        "decay_sec": decay_duration,
        "front_to_tail_asymmetry": asymmetry_ratio
    }


def extract_zoom_window_around_peak(
    tau_series: np.ndarray, 
    time_sec: np.ndarray, 
    peak_info: dict, 
    window_padding_sec: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Функция 16: Срез траектории в окрестности импульса для визуализации профиля волны."""
    t_center = peak_info["peak_time_sec"]
    t_start = max(time_sec[0], t_center - window_padding_sec)
    t_end = min(time_sec[-1], t_center + window_padding_sec)

    mask = (time_sec >= t_start) & (time_sec <= t_end)
    return time_sec[mask], tau_series[mask]


def render_pulse_morphology_axis(
    ax: plt.Axes, 
    t_zoom: np.ndarray, 
    tau_zoom: np.ndarray, 
    p_info: dict, 
    constituent_windows: list[float]
):
    """Функция 17: Детальный рендеринг одного импульса с геометрией фронта, FWHM и окнами W."""
    ax.plot(t_zoom, tau_zoom, color="#d62728", linewidth=1.4, label="tau(t)")
    
    # Линия полувысоты (FWHM)
    ax.hlines(
        y=p_info["half_height"], 
        xmin=p_info["t_left"], 
        xmax=p_info["t_right"], 
        color="black", 
        linewidth=1.2, 
        linestyle="--",
        label=f"FWHM = {p_info['fwhm_sec']:.2f}с"
    )
    # Пик и координаты
    ax.plot(p_info["peak_time_sec"], p_info["peak_val"], "o", color="black", markersize=4)

    # Отображение диапазона характерных окон W из носителя
    if constituent_windows:
        w_min = min(constituent_windows)
        w_max = max(constituent_windows)
        ax.set_title(
            f"Импульс t={p_info['peak_time_sec']:.2f}с | FWHM={p_info['fwhm_sec']:.2f}с | "
            f"Асимметрия={p_info['front_to_tail_asymmetry']:.2f}\n"
            f"Окна в носителе: W in [{w_min:.2f}с .. {w_max:.2f}с]",
            fontsize=9
        )
    else:
        ax.set_title(
            f"Импульс t={p_info['peak_time_sec']:.2f}с | FWHM={p_info['fwhm_sec']:.2f}с | "
            f"Асимметрия={p_info['front_to_tail_asymmetry']:.2f}",
            fontsize=9
        )

    ax.set_xlabel("Время (сек)")
    ax.set_ylabel("tau(t)")
    ax.legend(fontsize=8, loc="upper right")


def compute_cluster_pulse_statistics(
    tau_series: np.ndarray, 
    time_sec: np.ndarray, 
    pedestal: float
) -> list[dict]:
    """Функция 18: Сбор морфологической статистики по всем выраженным импульсам траектории."""
    peak_indices = locate_prominent_local_maxima(tau_series, pedestal)
    pulse_stats = []
    for p_idx in peak_indices:
        geom = compute_pulse_fwhm_and_geometry(tau_series, time_sec, p_idx, pedestal)
        if geom is not None and geom["fwhm_sec"] > 0.0:
            pulse_stats.append(geom)
    return pulse_stats


# =============================================================================
# МОДУЛЬ 5. Связь масштаба окон W и геометрии импульсов (Функции 19–21)
# =============================================================================

def aggregate_cluster_scale_concordance(
    pulse_stats: list[dict], 
    constituent_windows: list[float]
) -> dict:
    """Функция 19: Сравнение средней длительности FWHM с теоретической длительностью окна W."""
    if not pulse_stats or not constituent_windows:
        return {"mean_fwhm": 0.0, "mean_w": 0.0, "concordance_ratio": 0.0}

    fwhm_array = np.array([p["fwhm_sec"] for p in pulse_stats])
    w_array = np.array(constituent_windows)

    mean_fwhm = float(np.median(fwhm_array))
    mean_w = float(np.median(w_array))
    concordance = mean_fwhm / (mean_w + 1e-12)

    return {
        "mean_fwhm": mean_fwhm,
        "mean_w": mean_w,
        "concordance_ratio": concordance,
        "pulses_count": len(pulse_stats)
    }


def render_scale_concordance_axis(
    ax: plt.Axes, 
    all_k_w_means: list[float], 
    all_k_fwhm_means: list[float], 
    all_k_ids: list[int]
):
    """Функция 20: График связи между масштабом окон кластера W и наблюдаемой шириной FWHM."""
    ax.plot([0.0, max(all_k_w_means, default=1.0)], [0.0, max(all_k_w_means, default=1.0)], 
            color="#999999", linestyle="--", label="Теоретическая линия FWHM ~ W")
    
    ax.scatter(all_k_w_means, all_k_fwhm_means, c="#1f77b4", edgecolor="#111111", s=35, alpha=0.8)
    for w_val, f_val, k_id in zip(all_k_w_means, all_k_fwhm_means, all_k_ids):
        if k_id % max(1, len(all_k_ids) // 10) == 0:
            ax.annotate(f"k={k_id}", (w_val, f_val), fontsize=7, alpha=0.8)

    ax.set_title("Согласованность масштабов: Окно W vs Наблюдаемая FWHM")
    ax.set_xlabel("Медианное окно W носителя (сек)")
    ax.set_ylabel("Медианная полуширина FWHM (сек)")
    ax.legend(fontsize=8)


def render_asymmetry_spectrum_axis(ax: plt.Axes, asymmetries: list[float], k_ids: list[int]):
    """Функция 21: Спектр асимметрии фронта (скорость нарастания vs скорость спада)."""
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Симметричный импульс (1.0)")
    ax.scatter(k_ids, asymmetries, c="#ff7f0e", edgecolor="#222222", s=30)
    ax.set_yscale("log")
    ax.set_title("Стрела времени: Асимметрия фронта импульса (Rise / Decay)")
    ax.set_xlabel("ID Режима k")
    ax.set_ylabel("Отношение фронтов (log)")
    ax.legend(fontsize=8)


# =============================================================================
# МОДУЛЬ 6. Генераторы исследовательских дашбордов (Функции 22–27)
# =============================================================================

def plot_pedestal_and_intermittency_overview(
    time_sec: np.ndarray, 
    tau_mat: np.ndarray, 
    k_val: int, 
    output_path: Path
):
    """Функция 22: Дашборд 1 — Фоновые пьедесталы, плотности p(tau) и перемежаемость."""
    k_total = tau_mat.shape[0]
    sample_indices = [0, 1, k_total // 4, k_total // 2, (3 * k_total) // 4, k_total - 1]
    sample_indices = sorted(list(set(sample_indices)))
    n_samples = len(sample_indices)

    fig, axes = plt.subplots(n_samples, 2, figsize=(13, 2.5 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    cmap = mpl.colormaps["turbo"].resampled(k_total)

    for row_idx, k_idx in enumerate(sample_indices):
        tau_series = tau_mat[k_idx, :]
        pedestal, bin_centers, density = compute_empirical_density_and_pedestal(tau_series)
        interm = compute_intermittency_factor(tau_series, pedestal)

        # Левая колонка: плотность распределения
        render_empirical_density_axis(axes[row_idx, 0], bin_centers, density, pedestal, k_idx, cmap(k_idx))

        # Правая колонка: траектория во времени с линией пьедестала
        ax_t = axes[row_idx, 1]
        ax_t.plot(time_sec, tau_series, color=cmap(k_idx), linewidth=0.7)
        ax_t.axhline(pedestal, color="black", linestyle="--", linewidth=0.9)
        ax_t.set_title(f"tau_{k_idx}(t) | Покой: {interm['quiet_time_fraction']*100:.1f}% времени | Макс/Пьедестал: {interm['max_burst_to_pedestal']:.1f}x")
        ax_t.set_xlabel("Время (сек)")
        ax_t.set_ylabel("tau(t)")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен дашборд пьедесталов: {output_path.name}")


def plot_energy_localization_and_cumulants(
    time_sec: np.ndarray, 
    tau_mat: np.ndarray, 
    k_val: int, 
    output_path: Path
):
    """Функция 23: Дашборд 2 — Кумуляты накопления энергии и индекс разреженности Джини."""
    k_total = tau_mat.shape[0]
    time_norm = np.linspace(0.0, 1.0, len(time_sec))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Панель 0,0: Все кумуляты вместе
    step = max(1, k_total // 15)
    subset_k = list(range(0, k_total, step))
    cum_list = [compute_energy_accumulation_curve(tau_mat[k, :]) for k in subset_k]
    render_all_cumulatives_stack(axes[0, 0], time_norm, cum_list, subset_k)

    # Панель 0,1: Спектр коэффициента Джини по всем режимам
    gini_values = [compute_energy_gini_coefficient(compute_energy_accumulation_curve(tau_mat[k, :])) for k in range(k_total)]
    axes[0, 1].bar(range(k_total), gini_values, color="#2ca02c", edgecolor="none")
    axes[0, 1].set_title("Спектр разреженности энергии (Индекс Джини)")
    axes[0, 1].set_xlabel("ID Режима k")
    axes[0, 1].set_ylabel("Gini (0 = постоянен, 1 = одиночный взрыв)")

    # Панели 1,0 и 1,1: Детальные кумуляты для контрастных режимов
    min_gini_k = int(np.argmin(gini_values))
    max_gini_k = int(np.argmax(gini_values))

    render_cumulative_energy_axis(axes[1, 0], time_norm, compute_energy_accumulation_curve(tau_mat[min_gini_k, :]), min_gini_k, "#1f77b4", gini_values[min_gini_k])
    render_cumulative_energy_axis(axes[1, 1], time_norm, compute_energy_accumulation_curve(tau_mat[max_gini_k, :]), max_gini_k, "#d62728", gini_values[max_gini_k])

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен дашборд локализации энергии: {output_path.name}")


def plot_pulse_geometry_gallery(
    time_sec: np.ndarray, 
    tau_mat: np.ndarray, 
    df_mem_k: pl.DataFrame, 
    k_val: int, 
    output_path: Path
):
    """Функция 24: Дашборд 3 — Галерея импульсов перестройки (зум фронтов, FWHM)."""
    k_total = tau_mat.shape[0]
    # Выбираем высокоэнергетические режимы, где заведомо есть четкие перестройки
    high_energy_k = list(range(max(1, k_total - 6), k_total))
    
    n_plots = len(high_energy_k)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.8 * n_plots))
    if n_plots == 1:
        axes = np.array([axes])

    for i, k_idx in enumerate(high_energy_k):
        tau_series = tau_mat[k_idx, :]
        pedestal, _, _ = compute_empirical_density_and_pedestal(tau_series)
        pulses = compute_cluster_pulse_statistics(tau_series, time_sec, pedestal)
        windows = extract_constituent_windows_of_cluster(df_mem_k, k_idx)

        if pulses:
            # Берем самый мощный импульс
            main_pulse = max(pulses, key=lambda p: p["peak_val"])
            t_zoom, tau_zoom = extract_zoom_window_around_peak(tau_series, time_sec, main_pulse, window_padding_sec=6.0)
            render_pulse_morphology_axis(axes[i], t_zoom, tau_zoom, main_pulse, windows)
        else:
            axes[i].text(0.5, 0.5, f"tau_{k_idx}: Нет выраженных изолированных пиков над пьедесталом", 
                         ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"tau_{k_idx}")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен дашборд геометрии импульсов: {output_path.name}")


def plot_window_scale_vs_pulse_geometry(
    time_sec: np.ndarray, 
    tau_mat: np.ndarray, 
    df_mem_k: pl.DataFrame, 
    k_val: int, 
    output_path: Path
):
    """Функция 25: Дашборд 4 — Сопоставление шкалы носителя W и физической ширины перестройки FWHM."""
    k_total = tau_mat.shape[0]
    all_w_medians = []
    all_fwhm_medians = []
    all_asymmetries = []
    valid_k = []

    for k in range(1, k_total):  # пропускаем tau_0 (покой)
        tau_series = tau_mat[k, :]
        pedestal, _, _ = compute_empirical_density_and_pedestal(tau_series)
        pulses = compute_cluster_pulse_statistics(tau_series, time_sec, pedestal)
        windows = extract_constituent_windows_of_cluster(df_mem_k, k)

        if pulses and windows:
            stat = aggregate_cluster_scale_concordance(pulses, windows)
            asym_median = float(np.median([p["front_to_tail_asymmetry"] for p in pulses]))
            all_w_medians.append(stat["mean_w"])
            all_fwhm_medians.append(stat["mean_fwhm"])
            all_asymmetries.append(asym_median)
            valid_k.append(k)

    if not valid_k:
        logger.warning(f"Недостаточно импульсов для сопоставления масштабов K={k_val}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    render_scale_concordance_axis(axes[0], all_w_medians, all_fwhm_medians, valid_k)
    render_asymmetry_spectrum_axis(axes[1], all_asymmetries, valid_k)

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен дашборд связи масштабов: {output_path.name}")


def generate_all_tau_anatomy_figures_for_k(
    time_sec: np.ndarray, 
    tau_mat: np.ndarray, 
    df_mem_k: pl.DataFrame, 
    k_val: int, 
    figures_dir: Path
):
    """Функция 26: Построение полного комплекта анатомических дашбордов для одного K."""
    logger.info(f"Препарирование траекторий для K={k_val}...")
    
    # 1. Пьедесталы покоя и плотности распределения
    plot_pedestal_and_intermittency_overview(
        time_sec, tau_mat, k_val, figures_dir / f"tau_anatomy_pedestal_density_k{k_val}.png"
    )

    # 2. Локализация энергии и кривые кумулят (Джини)
    plot_energy_localization_and_cumulants(
        time_sec, tau_mat, k_val, figures_dir / f"tau_anatomy_energy_cumulants_k{k_val}.png"
    )

    # 3. Морфология импульсов в высоком разрешении (FWHM, фронты)
    plot_pulse_geometry_gallery(
        time_sec, tau_mat, df_mem_k, k_val, figures_dir / f"tau_anatomy_pulse_geometry_k{k_val}.png"
    )

    # 4. Согласованность масштабов W носителя и длительности перестройки FWHM
    plot_window_scale_vs_pulse_geometry(
        time_sec, tau_mat, df_mem_k, k_val, figures_dir / f"tau_anatomy_scale_concordance_k{k_val}.png"
    )


# =============================================================================
# МОДУЛЬ 7. Главный исполнительный цикл (Функции 27–28)
# =============================================================================

def run_tau_anatomy_pipeline(dataset_code: int, custom_k_list: list[int] | None = None):
    """Функция 27: Исполнение пайплайна исследования анатомии функций tau_k(t)."""
    cfg, results_dir, figures_dir = resolve_dataset_config(dataset_code, DATASETS, ORM_RESULTS_DIR)
    trajectories_dir = results_dir / "trajectories"

    logger.info(f"=== Анализ анатомии функций tau_k(t) | Датасет: {cfg['name']} ===")

    # Поиск доступных рассчитанных K
    avail_files = list(trajectories_dir.glob("trajectories_k*.parquet"))
    if not avail_files:
        raise FileNotFoundError(f"В {trajectories_dir} отсутствуют файлы траекторий!")

    all_k_in_dir = []
    for f in avail_files:
        m = re.search(r"trajectories_k(\d+)\.parquet", f.name)
        if m:
            all_k_in_dir.append(int(m.group(1)))
    all_k_in_dir = sorted(all_k_in_dir)

    target_k_list = custom_k_list if custom_k_list else [k for k in DEFAULT_ANALYTIC_K if k in all_k_in_dir]
    if not target_k_list:
        target_k_list = [all_k_in_dir[0], all_k_in_dir[len(all_k_in_dir) // 2], all_k_in_dir[-1]]

    logger.info(f"Выбранные значения K для углубленного анализа: {target_k_list}")

    for k_val in target_k_list:
        time_sec, tau_mat, _ = load_trajectories_table(trajectories_dir, k_val)
        df_mem_k = load_memberships_for_k(results_dir, k_val)
        generate_all_tau_anatomy_figures_for_k(time_sec, tau_mat, df_mem_k, k_val, figures_dir)

    logger.info(f"=== Препарирование завершено. Все графики сохранены в: {figures_dir} ===")


def main():
    """Функция 28: Точка входа скрипта (CLI / SLURM)."""
    dataset_code = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else 0))
    run_tau_anatomy_pipeline(dataset_code)


if __name__ == "__main__":
    main()
