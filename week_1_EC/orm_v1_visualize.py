#!/usr/bin/env python3
"""
Модуль полной объективной визуализации результатов ОРМ (Задача А2).
Принцип: визуализация всех вычисленных данных (все K, все режимы, полное время T)
без фильтрации, обрезки интервалов и выбора гиперпараметров.
Архитектура: 30 атомарных функций.
"""

import logging
import math  # ИСПРАВЛЕНО: добавлен отсутствовавший импорт
import os
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Единый строгий стиль оформления
mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "legend.frameon": False,
    "figure.autolayout": True,
    "savefig.dpi": 300,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ORM_VISUALIZE_ALL")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DB_DIR = WORKSPACE_DIR / "04_processed_db"
ORM_RESULTS_DIR = PROCESSED_DB_DIR / "orm_results"
METADATA_DIR = WORKSPACE_DIR / "02_metadata"

DATASETS = {
    0: {
        "name": "Type 1 (HSE Bigrams) - Chistova Alena",
        "prefix": "type1_sub10_chistova",
        "events_csv": METADATA_DIR
        / "annotations/type1_sub10_chistova_alena_events.csv",
        "type": "type1",
    },
    1: {
        "name": "Type 2 (Ushakov Drowsiness) - Bainbridge Emily",
        "prefix": "type2_bainbridge",
        "events_csv": METADATA_DIR
        / "annotations/type2_bainbridge_emily_220519_buttons.csv",
        "type": "type2",
    },
    2: {
        "name": "Type 3 (Spanish BCBL) - Subject 01 Block 1",
        "prefix": "type3_bcbl_01",
        "events_csv": None,
        "type": "type3",
    },
}


# =============================================================================
# МОДУЛЬ 1. Метаданные и загрузка данных (Функции 1–5)
# =============================================================================


def parse_window_scale_seconds(window_tag: str) -> float:
    """Функция 1: Извлечение длительности окна в секундах из тега."""
    match = re.search(r"w_([0-9.]+)s", window_tag)
    return float(match.group(1)) if match else 0.0


def sort_window_tags(window_tags: list[str]) -> list[str]:
    """Функция 2: Монотонная сортировка списка оконных шкал по возрастанию длительности."""
    return sorted(window_tags, key=parse_window_scale_seconds)


def resolve_dataset_directories(
    dataset_code: int, datasets_dict: dict, base_results_dir: Path
) -> tuple[dict, Path, Path]:
    """Функция 3: Разрешение директорий результатов и графиков."""
    if dataset_code not in datasets_dict:
        raise ValueError(f"Неизвестный dataset_code: {dataset_code}")
    cfg = datasets_dict[dataset_code]
    results_dir = base_results_dir / cfg["prefix"]
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Каталог с результатами не найден: {results_dir}"
        )
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return cfg, results_dir, figures_dir


def load_all_orm_tables(
    results_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Path]:
    """Функция 4: Загрузка всех рассчитанных таблиц ОРМ."""
    df_sweep = pl.read_parquet(results_dir / "orm_v1_sweep_summary.parquet")
    df_history = pl.read_parquet(
        results_dir / "orm_v1_training_history.parquet"
    )
    df_memberships = pl.read_parquet(
        results_dir / "orm_v1_memberships.parquet"
    )
    trajectories_dir = results_dir / "trajectories"
    return df_sweep, df_history, df_memberships, trajectories_dir


def parse_all_experimental_events(
    events_csv: Path | None, dataset_type: str
) -> list[tuple[float, str]]:
    """Функция 5: Чтение полного списка экспериментальных меток времени без фильтрации."""
    events = []
    if not events_csv or not events_csv.exists():
        return events
    try:
        df_ev = pl.read_csv(events_csv, ignore_errors=True)
        if dataset_type == "type1" and "key_resp.rt" in df_ev.columns:
            valid = df_ev.filter(pl.col("key_resp.rt").is_not_null())
            rts = valid["key_resp.rt"].to_numpy()
            curr_t = 5.0
            for rt in rts:
                events.append((curr_t + 0.5, "Stim"))
                events.append((curr_t + 0.5 + rt, "Resp"))
                curr_t += 0.5 + rt + 0.02
        elif dataset_type == "type2" and "time" in df_ev.columns:
            for t in df_ev["time"].to_numpy():
                events.append((float(t), "Btn"))
    except Exception as err:
        logger.warning(f"Ошибка загрузки меток событий: {err}")
    return events


# =============================================================================
# МОДУЛЬ 2. Преобразования пространственно-временных матриц (Функции 6–9)
# =============================================================================


def extract_channel_window_topology(
    df_memberships: pl.DataFrame, k_val: int
) -> tuple[np.ndarray, list[str], list[str]]:
    """Функция 6: Преобразование принадлежностей в полную матрицу (каналы x окна)."""
    sub = df_memberships.filter(pl.col("k_clusters") == k_val)
    channels = sorted(sub["channel"].unique().to_list())
    windows = sort_window_tags(sub["window_tag"].unique().to_list())

    ch_map = {ch: i for i, ch in enumerate(channels)}
    win_map = {w: j for j, w in enumerate(windows)}

    mat = np.zeros((len(channels), len(windows)), dtype=np.int32)
    for row in sub.iter_rows(named=True):
        mat[ch_map[row["channel"]], win_map[row["window_tag"]]] = row[
            "cluster_regime_id"
        ]
    return mat, channels, windows


def compute_full_cluster_sizes(
    df_memberships: pl.DataFrame, k_val: int
) -> tuple[np.ndarray, np.ndarray]:
    """Функция 7: Полный спектр размеров всех K подмножеств без исключений."""
    sub = df_memberships.filter(pl.col("k_clusters") == k_val)
    counts = sub["cluster_regime_id"].value_counts().sort("cluster_regime_id")
    ids = counts["cluster_regime_id"].to_numpy()
    sizes = counts["count"].to_numpy()
    return ids, sizes


def compute_all_scale_distributions(
    df_memberships: pl.DataFrame, k_val: int, sorted_windows: list[str]
) -> np.ndarray:
    """Функция 8: Распределение всех K режимов по оконным масштабам."""
    sub = df_memberships.filter(pl.col("k_clusters") == k_val)
    dist_matrix = np.zeros((k_val, len(sorted_windows)), dtype=np.float32)
    win_map = {w: j for j, w in enumerate(sorted_windows)}

    for row in sub.iter_rows(named=True):
        dist_matrix[
            row["cluster_regime_id"], win_map[row["window_tag"]]
        ] += 1.0

    row_sums = dist_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dist_matrix / row_sums


def compute_regime_cross_correlation(
    df_traj: pl.DataFrame, k_val: int
) -> np.ndarray:
    """Функция 9: Полная матрица парной корреляции между всеми K центрами режимов."""
    tau_cols = [
        f"tau_{k}" for k in range(k_val) if f"tau_{k}" in df_traj.columns
    ]
    data = df_traj.select(tau_cols).to_numpy()
    return np.corrcoef(data, rowvar=False)


# =============================================================================
# МОДУЛЬ 3. Панели динамики обучения (Функции 10–13)
# =============================================================================


def render_history_loss_panel(
    ax: plt.Axes, df_history: pl.DataFrame, all_k: list[int]
):
    """Функция 10: Кривые функционала J(m) для всех K на всех итерациях."""
    cmap = mpl.colormaps["turbo"].resampled(len(all_k))
    for i, k in enumerate(all_k):
        sub = df_history.filter(pl.col("k_clusters") == k).sort("iteration")
        ax.plot(
            sub["iteration"].to_numpy(),
            sub["loss_primal_J"].to_numpy(),
            color=cmap(i),
            label=f"K={k}",
        )
    ax.set_title("Функционал J(m)")
    ax.set_xlabel("Итерация")
    ax.legend(ncol=2, fontsize=8)


def render_history_stability_panel(
    ax: plt.Axes, df_history: pl.DataFrame, all_k: list[int]
):
    """Функция 11: Доля переключений ячеек Вороного для всех K."""
    cmap = mpl.colormaps["turbo"].resampled(len(all_k))
    for i, k in enumerate(all_k):
        sub = df_history.filter(pl.col("k_clusters") == k).sort("iteration")
        ax.semilogy(
            sub["iteration"].to_numpy(),
            np.maximum(sub["switched_ratio"].to_numpy(), 1e-6),
            color=cmap(i),
        )
    ax.axhline(1e-3, color="black", linestyle="--")
    ax.set_title("Доля переключений меток")
    ax.set_xlabel("Итерация")


def render_history_r2_panel(
    ax: plt.Axes, df_history: pl.DataFrame, all_k: list[int]
):
    """Функция 12: Объясненная дисперсия R^2 (%) для всех K."""
    cmap = mpl.colormaps["turbo"].resampled(len(all_k))
    for i, k in enumerate(all_k):
        sub = df_history.filter(pl.col("k_clusters") == k).sort("iteration")
        ax.plot(
            sub["iteration"].to_numpy(),
            sub["variance_explained"].to_numpy() * 100.0,
            color=cmap(i),
        )
    ax.set_title("Объясненная дисперсия R^2 (%)")
    ax.set_xlabel("Итерация")


def render_history_entropy_panel(
    ax: plt.Axes, df_history: pl.DataFrame, all_k: list[int]
):
    """Функция 13: Энтропия распределения объемов подмножеств H(Omega)."""
    cmap = mpl.colormaps["turbo"].resampled(len(all_k))
    for i, k in enumerate(all_k):
        sub = df_history.filter(pl.col("k_clusters") == k).sort("iteration")
        ax.plot(
            sub["iteration"].to_numpy(),
            sub["cluster_entropy"].to_numpy(),
            color=cmap(i),
        )
    ax.set_title("Энтропия объемов H(Omega)")
    ax.set_xlabel("Итерация")


# =============================================================================
# МОДУЛЬ 4. Панели сравнения по сетке K (Функции 14–17)
# =============================================================================


def render_sweep_loss_panel(
    ax: plt.Axes, k_vals: np.ndarray, j_vals: np.ndarray
):
    """Функция 14: Невязка J по всей сетке K."""
    ax.plot(k_vals, j_vals, "o-", color="#1f77b4")
    ax.set_title("Итоговый функционал J(K)")
    ax.set_xlabel("K")


def render_sweep_r2_panel(
    ax: plt.Axes, k_vals: np.ndarray, r2_vals: np.ndarray
):
    """Функция 15: Итоговый R^2 (%) по всей сетке K."""
    ax.plot(k_vals, r2_vals, "s-", color="#2ca02c")
    ax.set_title("Качество покрытия R^2 (%)")
    ax.set_xlabel("K")


def render_sweep_baseline_ratio_panel(
    ax: plt.Axes, k_vals: np.ndarray, baseline_ratios: np.ndarray
):
    """Функция 16: Доля элементов в базовом режиме покоя tau_0."""
    ax.plot(k_vals, baseline_ratios * 100.0, "^-", color="#d62728")
    ax.set_title("Доля фонового режима tau_0 (%)")
    ax.set_xlabel("K")


def render_sweep_time_panel(
    ax: plt.Axes, k_vals: np.ndarray, times: np.ndarray
):
    """Функция 17: Полное время счета по всей сетке K."""
    ax.bar(k_vals, times, width=max(1.0, (k_vals[-1] - k_vals[0]) / 40.0), color="#7f7f7f")
    ax.set_title("Время вычислений (сек)")
    ax.set_xlabel("K")


# =============================================================================
# МОДУЛЬ 5. Отрисовка конкретных структур данных (Функции 18–22)
# =============================================================================


def render_topology_matrix_axis(
    ax: plt.Axes, matrix: np.ndarray, sorted_windows: list[str], k_val: int
):
    """Функция 18: Отрисовка топологической карты каналы x окна для одного K."""
    cmap = mpl.colormaps["turbo"].resampled(k_val).copy()
    cmap.set_under("black")
    ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0.5,
        vmax=k_val - 0.5,
        interpolation="nearest",
    )
    ax.set_title(f"Топология Omega (K={k_val})")
    ax.set_xlabel("Оконный масштаб")
    ax.set_ylabel("Канал")

    num_ticks = min(5, len(sorted_windows))
    ticks = np.unique(
        np.linspace(0, len(sorted_windows) - 1, num_ticks, dtype=int)
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{parse_window_scale_seconds(sorted_windows[i]):.2f}с" for i in ticks]
    )


def render_full_trajectories_axis(
    ax: plt.Axes,
    time_sec: np.ndarray,
    df_traj: pl.DataFrame,
    k_val: int,
    events: list[tuple[float, str]],
):
    """Функция 19: Отрисовка ВСЕХ K траекторий на ВСЕМ временном интервале [0, T]."""
    if "tau_0" in df_traj.columns:
        ax.plot(
            time_sec,
            df_traj["tau_0"].to_numpy(),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="tau_0 (Покой)",
        )
    cmap = mpl.colormaps["turbo"].resampled(max(k_val - 1, 1))

    for k in range(1, k_val):
        col_name = f"tau_{k}"
        if col_name in df_traj.columns:
            ax.plot(
                time_sec,
                df_traj[col_name].to_numpy(),
                color=cmap(k - 1),
                linewidth=0.8,
                alpha=0.85,
            )

    for t_ev, _ in events:
        ax.axvline(t_ev, color="red", alpha=0.4, linestyle=":", linewidth=0.8)

    ax.set_title(
        f"Все {k_val} траекторий tau_k(t) на интервале [0, {time_sec[-1]:.1f}] сек"
    )
    ax.set_xlabel("Время (сек)")


def render_occupancy_bar_axis(
    ax: plt.Axes, ids: np.ndarray, sizes: np.ndarray, k_val: int
):
    """Функция 20: Отрисовка объемов абсолютно всех кластеров для одного K."""
    cmap = mpl.colormaps["turbo"].resampled(max(len(ids) - 1, 1))
    colors = ["black"] + [cmap(i) for i in range(len(ids) - 1)]
    ax.bar(ids, sizes, color=colors, edgecolor="none")
    ax.set_title(f"Размеры всех кластеров (K={k_val})")
    ax.set_xlabel("ID Режима")
    ax.set_ylabel("Число объектов")


def render_scale_profile_axis(
    ax: plt.Axes, dist_mat: np.ndarray, sorted_windows: list[str], k_val: int
):
    """Функция 21: Отрисовка профилей шкал для ВСЕХ K режимов."""
    scale_seconds = [parse_window_scale_seconds(w) for w in sorted_windows]
    cmap = mpl.colormaps["turbo"].resampled(max(k_val - 1, 1))
    for k in range(1, k_val):
        ax.plot(
            scale_seconds,
            dist_mat[k, :],
            color=cmap(k - 1),
            linewidth=0.8,
            alpha=0.7,
        )
    ax.set_xscale("log")
    ax.set_title(f"Оконный спектр всех режимов (K={k_val})")
    ax.set_xlabel("W (сек, log)")


def render_correlation_matrix_axis(
    ax: plt.Axes, corr_mat: np.ndarray, k_val: int
):
    """Функция 22: Отрисовка полной матрицы корреляции K x K."""
    ax.imshow(
        corr_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest"
    )
    ax.set_title(f"Корреляция центров {k_val}x{k_val}")
    ax.set_xlabel("ID Режима")
    ax.set_ylabel("ID Режима")


# =============================================================================
# МОДУЛЬ 6. Генераторы итоговых графиков (Функции 23–29)
# =============================================================================


def plot_all_training_diagnostics(
    df_history: pl.DataFrame, all_k: list[int], output_path: Path
):
    """Функция 23: Дашборд сходимости для всех K."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    render_history_loss_panel(axes[0, 0], df_history, all_k)
    render_history_stability_panel(axes[0, 1], df_history, all_k)
    render_history_r2_panel(axes[1, 0], df_history, all_k)
    render_history_entropy_panel(axes[1, 1], df_history, all_k)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен график: {output_path.name}")


def plot_all_sweep_diagnostics(
    df_sweep: pl.DataFrame, df_history: pl.DataFrame, output_path: Path
):
    """Функция 24: Дашборд сравнения по всей сетке K с автоматическим извлечением R^2."""
    df_sorted = df_sweep.sort("k_clusters")

    # ИСПРАВЛЕНО: если final_variance_explained нет в df_sweep, подтягиваем из df_history
    if "final_variance_explained" not in df_sorted.columns:
        last_r2 = (
            df_history.sort(["k_clusters", "iteration"])
            .group_by("k_clusters", maintain_order=True)
            .agg(pl.col("variance_explained").last().alias("final_variance_explained"))
        )
        df_sorted = df_sorted.join(last_r2, on="k_clusters", how="left")

    k_vals = df_sorted["k_clusters"].to_numpy()
    j_vals = df_sorted["final_loss_primal"].to_numpy()
    r2_vals = df_sorted["final_variance_explained"].to_numpy() * 100.0
    baseline_ratios = df_sorted["baseline_ratio"].to_numpy()
    times = df_sorted["total_time_sec"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    render_sweep_loss_panel(axes[0, 0], k_vals, j_vals)
    render_sweep_r2_panel(axes[0, 1], k_vals, r2_vals)
    render_sweep_baseline_ratio_panel(axes[1, 0], k_vals, baseline_ratios)
    render_sweep_time_panel(axes[1, 1], k_vals, times)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен график: {output_path.name}")


def plot_all_topologies_grid(
    df_memberships: pl.DataFrame, all_k: list[int], output_path: Path
):
    """Функция 25: Сетка топологических карт каналы x окна для ВСЕХ вычисленных K."""
    n_cols = 3
    n_rows = math.ceil(len(all_k) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, k_val in enumerate(all_k):
        mat, _, sorted_windows = extract_channel_window_topology(
            df_memberships, k_val
        )
        render_topology_matrix_axis(axes_flat[idx], mat, sorted_windows, k_val)

    for empty_idx in range(len(all_k), len(axes_flat)):
        axes_flat[empty_idx].axis("off")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен график: {output_path.name}")


def plot_all_trajectories_full_session(
    trajectories_dir: Path,
    all_k: list[int],
    events: list[tuple[float, str]],
    output_dir: Path,
):
    """Функция 26: Отдельный график полного времени для КАЖДОГО K со ВСЕМИ траекториями."""
    for k_val in all_k:
        traj_file = trajectories_dir / f"trajectories_k{k_val}.parquet"
        if not traj_file.exists():
            continue
        df_traj = pl.read_parquet(traj_file)
        time_sec = df_traj["time_sec"].to_numpy()

        fig, ax = plt.subplots(figsize=(16, 6))
        render_full_trajectories_axis(ax, time_sec, df_traj, k_val, events)
        out_file = output_dir / f"trajectories_full_time_k{k_val}.png"
        fig.savefig(out_file)
        plt.close(fig)
        logger.info(f"Сохранен график: {out_file.name}")


def plot_all_occupancies_grid(
    df_memberships: pl.DataFrame, all_k: list[int], output_path: Path
):
    """Функция 27: Сетка распределений размеров подмножеств для ВСЕХ K."""
    n_cols = 3
    n_rows = math.ceil(len(all_k) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, k_val in enumerate(all_k):
        ids, sizes = compute_full_cluster_sizes(df_memberships, k_val)
        render_occupancy_bar_axis(axes_flat[idx], ids, sizes, k_val)

    for empty_idx in range(len(all_k), len(axes_flat)):
        axes_flat[empty_idx].axis("off")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен график: {output_path.name}")


def plot_all_scale_distributions_grid(
    df_memberships: pl.DataFrame, all_k: list[int], output_path: Path
):
    """Функция 28: Сетка маргинальных спектров оконных масштабов для ВСЕХ K."""
    n_cols = 3
    n_rows = math.ceil(len(all_k) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, k_val in enumerate(all_k):
        sub = df_memberships.filter(pl.col("k_clusters") == k_val)
        sorted_windows = sort_window_tags(
            sub["window_tag"].unique().to_list()
        )
        dist_mat = compute_all_scale_distributions(
            df_memberships, k_val, sorted_windows
        )
        render_scale_profile_axis(
            axes_flat[idx], dist_mat, sorted_windows, k_val
        )

    for empty_idx in range(len(all_k), len(axes_flat)):
        axes_flat[empty_idx].axis("off")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен график: {output_path.name}")


def plot_all_correlation_matrices_grid(
    trajectories_dir: Path, all_k: list[int], output_path: Path
):
    """Функция 29: Сетка матриц корреляции центров для ВСЕХ K."""
    n_cols = 3
    n_rows = math.ceil(len(all_k) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows)
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, k_val in enumerate(all_k):
        traj_file = trajectories_dir / f"trajectories_k{k_val}.parquet"
        if not traj_file.exists():
            continue
        df_traj = pl.read_parquet(traj_file)
        corr_mat = compute_regime_cross_correlation(df_traj, k_val)
        render_correlation_matrix_axis(axes_flat[idx], corr_mat, k_val)

    for empty_idx in range(len(all_k), len(axes_flat)):
        axes_flat[empty_idx].axis("off")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Сохранен график: {output_path.name}")


# =============================================================================
# МОДУЛЬ 7. Главный цикл пайплайна (Функция 30)
# =============================================================================


def main():
    """Функция 30: Точка входа пайплайна тотальной визуализации."""
    dataset_code = int(
        os.environ.get(
            "SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else 0
        )
    )
    cfg, results_dir, figures_dir = resolve_dataset_directories(
        dataset_code, DATASETS, ORM_RESULTS_DIR
    )
    logger.info(f"=== Полная объективная визуализация: {cfg['name']} ===")

    df_sweep, df_history, df_memberships, trajectories_dir = (
        load_all_orm_tables(results_dir)
    )
    events = parse_all_experimental_events(cfg["events_csv"], cfg["type"])
    all_k = sorted(df_sweep["k_clusters"].unique().to_list())

    # 1. Диагностика процесса обучения (все итерации, все K)
    plot_all_training_diagnostics(
        df_history, all_k, figures_dir / "01_all_training_diagnostics.png"
    )

    # 2. Итоговые показатели функционала по всей сетке K (ИСПРАВЛЕНО: передаем df_history)
    plot_all_sweep_diagnostics(
        df_sweep, df_history, figures_dir / "02_all_k_sweep_summary.png"
    )

    # 3. Полные топологические карты (каналы x окна) для каждого K
    plot_all_topologies_grid(
        df_memberships,
        all_k,
        figures_dir / "03_all_spatial_scale_topologies.png",
    )

    # 4. Временные траектории ВСЕХ режимов на ВСЕМ физическом интервале [0, T]
    plot_all_trajectories_full_session(
        trajectories_dir, all_k, events, figures_dir
    )

    # 5. Размеры ВСЕХ кластеров для каждого K
    plot_all_occupancies_grid(
        df_memberships, all_k, figures_dir / "04_all_cluster_sizes.png"
    )

    # 6. Спектральный охват оконных шкал для каждого K
    plot_all_scale_distributions_grid(
        df_memberships, all_k, figures_dir / "05_all_scale_distributions.png"
    )

    # 7. Матрицы ортогональности (корреляции) центров для каждого K
    plot_all_correlation_matrices_grid(
        trajectories_dir,
        all_k,
        figures_dir / "06_all_correlation_matrices.png",
    )

    logger.info(f"=== Все графики сохранены без купюр в: {figures_dir} ===")


if __name__ == "__main__":
    main()
