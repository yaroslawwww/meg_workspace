#!/usr/bin/env python3
r"""
week_1_EC/inspect_all_modes_variability.py
Глобальный анализ вариативности каналов и окон по ВСЕМ режимам ОРМ:
- Построение матрицы [K x Временные масштабы w];
- Построение матрицы [K x Анатомические зоны MNE VectorView ROI];
- Расчет энтропии пространственной и масштабной специализации режимов;
- Выявление узкоспециализированных модулей мозга vs глобальных системных мод;
- Исправление топокарты через mne.pick_info.
"""

import re
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ORM_GLOBAL_VARIABILITY")

# =============================================================================
# МОДУЛЬ 1: Обнаружение путей и заголовочной информации
# =============================================================================

def resolve_workspace_root(custom_path: Optional[str] = None) -> Path:
    """Функция 1: Определение корня проекта."""
    if custom_path:
        return Path(custom_path).resolve()
    cwd = Path.cwd()
    if (cwd / "04_processed_db").exists():
        return cwd
    if (cwd.parent / "04_processed_db").exists():
        return cwd.parent
    return Path("/workspace")


def get_dataset_descriptors(workspace_dir: Path, dataset_code: int) -> dict:
    """Функция 2: Пути к данным."""
    raw_dir = workspace_dir / "01_raw_data"
    registry = {
        0: {
            "name": "Type 1 (HSE Bigrams) - Chistova Alena",
            "prefix": "type1_sub10_chistova",
            "pysz_file": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw.fif.pysz",
            "raw_fif": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw.fif",
            "prep_fif": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif"
        }
    }
    if dataset_code not in registry:
        raise ValueError(f"Неизвестный dataset_code: {dataset_code}")
    return registry[dataset_code]


def extract_info_from_pysz_or_fif(cfg: dict) -> mne.Info:
    """Функция 3: Чтение структуры mne.Info напрямую из архива .fif.pysz."""
    for cand in [cfg.get("prep_fif"), cfg.get("raw_fif")]:
        if cand and cand.exists():
            return mne.io.read_info(cand, verbose=False)

    with open(cfg["pysz_file"], "rb") as f:
        payload = pickle.load(f)
    return payload["info"]


# =============================================================================
# МОДУЛЬ 2: Штатная разметка сенсоров MNE VectorView
# =============================================================================

def build_neuromag_roi_mapping(info: mne.Info) -> dict[str, str]:
    """Функция 4: Извлечение анатомических зон сенсоров через mne.read_vectorview_selection."""
    standard_regions = [
        "Left-frontal", "Right-frontal",
        "Left-temporal", "Right-temporal",
        "Left-parietal", "Right-parietal",
        "Left-occipital", "Right-occipital",
        "Vertex"
    ]
    mapping = {}
    for region_name in standard_regions:
        try:
            ch_list = mne.read_vectorview_selection(region_name, info=info, verbose=False)
            for ch in ch_list:
                mapping[ch] = region_name
                mapping[ch.replace(" ", "")] = region_name
        except Exception:
            pass
    return mapping


def build_sensor_types_mapping(info: mne.Info) -> dict[str, str]:
    """Функция 5: Привязка типов сенсоров (grad/mag) через mne.channel_type."""
    type_map = {}
    for idx, ch in enumerate(info["ch_names"]):
        c_type = mne.channel_type(info, idx)
        type_map[ch] = c_type
        type_map[ch.replace(" ", "")] = c_type
    return type_map


# =============================================================================
# МОДУЛЬ 3: Загрузка всей базы ОРМ и обогащение параметров
# =============================================================================

def parse_window_tag(window_tag: str) -> tuple[float, int]:
    """Функция 6: Разбор окна w (с) и порядка m из тега."""
    m = re.search(r"w_([0-9.]+)s_m(\d+)", window_tag)
    if not m:
        return 0.0, 0
    return float(m.group(1)), int(m.group(2))


def load_all_memberships_for_k(
    workspace_dir: Path, 
    prefix: str, 
    K: int, 
    roi_map: dict[str, str], 
    type_map: dict[str, str]
) -> pl.DataFrame:
    """Функция 7: Загрузка всех 15300 объектов для заданного K с метаданными."""
    mem_path = workspace_dir / "04_processed_db" / "orm_results" / prefix / "orm_v1_memberships.parquet"
    df_all = pl.read_parquet(mem_path).filter(pl.col("k_clusters") == K)

    channels = df_all["channel"].to_list()
    tags = df_all["window_tag"].to_list()

    durations, orders, rois, types = [], [], [], []
    for ch, tag in zip(channels, tags):
        w_sec, m_order = parse_window_tag(tag)
        durations.append(w_sec)
        orders.append(m_order)
        rois.append(roi_map.get(ch, roi_map.get(ch.replace(" ", ""), "Unknown")))
        types.append(type_map.get(ch, type_map.get(ch.replace(" ", ""), "meg")))

    df_enriched = df_all.with_columns([
        pl.Series("window_sec", durations, dtype=pl.Float32),
        pl.Series("order_m", orders, dtype=pl.Int32),
        pl.Series("mne_roi", rois, dtype=pl.Utf8),
        pl.Series("sensor_type", types, dtype=pl.Utf8)
    ])
    logger.info(f"Загружено {len(df_enriched)} объектов ячеек Вороного для K={K}")
    return df_enriched


# =============================================================================
# МОДУЛЬ 4: Расчет матриц вариативности и энтропии специализации
# =============================================================================

def compute_scale_occupancy_matrix(df_k: pl.DataFrame, K: int) -> tuple[np.ndarray, list[float]]:
    """Функция 8: Матрица [K x Число окон]: плотность каналов для каждой моды на каждом окне."""
    unique_windows = sorted(df_k["window_sec"].unique().to_list())
    n_windows = len(unique_windows)
    win_to_col = {w: i for i, w in enumerate(unique_windows)}

    scale_mat = np.zeros((K, n_windows), dtype=np.int32)
    for row in df_k.iter_rows(named=True):
        k = row["cluster_regime_id"]
        w = row["window_sec"]
        scale_mat[k, win_to_col[w]] += 1

    return scale_mat, unique_windows


def compute_roi_occupancy_matrix(df_k: pl.DataFrame, K: int) -> tuple[np.ndarray, list[str]]:
    """Функция 9: Матрица [K x Число анатомических ROI]: сколько пар режима лежит в каждой зоне."""
    unique_rois = sorted([r for r in df_k["mne_roi"].unique().to_list() if r != "Unknown"])
    roi_to_col = {r: i for i, r in enumerate(unique_rois)}

    roi_mat = np.zeros((K, len(unique_rois)), dtype=np.int32)
    for row in df_k.iter_rows(named=True):
        r = row["mne_roi"]
        if r in roi_to_col:
            k = row["cluster_regime_id"]
            roi_mat[k, roi_to_col[r]] += 1

    return roi_mat, unique_rois


def compute_diversity_entropies(matrix: np.ndarray) -> np.ndarray:
    """Функция 10: Расчет нормированной информационной энтропии Шеннона распределения."""
    K, n_bins = matrix.shape
    entropies = np.zeros(K, dtype=np.float32)
    max_ent = np.log2(n_bins) if n_bins > 1 else 1.0

    for k in range(K):
        row = matrix[k].astype(np.float64)
        total = np.sum(row)
        if total > 0:
            p = row / total
            p = p[p > 0]
            shannon = -np.sum(p * np.log2(p))
            entropies[k] = float(shannon / max_ent)
        else:
            entropies[k] = 0.0

    return entropies


def build_comprehensive_modes_catalog(
    df_k: pl.DataFrame, 
    K: int, 
    scale_mat: np.ndarray, 
    windows: list[float], 
    roi_mat: np.ndarray, 
    rois: list[str],
    scale_entropies: np.ndarray, 
    roi_entropies: np.ndarray
) -> pl.DataFrame:
    """Функция 11: Сводный каталог всех K режимов: доминанты, энтропии и архетипы."""
    catalog = []
    for k in range(K):
        sub = df_k.filter(pl.col("cluster_regime_id") == k)
        total_pairs = len(sub)
        
        # Доминирующее окно
        dom_win_idx = int(np.argmax(scale_mat[k]))
        dom_window = windows[dom_win_idx]
        
        # Доминирующий ROI
        dom_roi_idx = int(np.argmax(roi_mat[k]))
        dom_roi = rois[dom_roi_idx]
        
        # Самый центральный канал
        archetype_row = sub.sort("dist_sq_to_center").head(1).to_dicts()
        arch_ch = archetype_row[0]["channel"] if archetype_row else "None"
        arch_dist = archetype_row[0]["dist_sq_to_center"] if archetype_row else 0.0

        catalog.append({
            "mode_k": k,
            "total_pairs": total_pairs,
            "dom_window_sec": dom_window,
            "dom_roi": dom_roi,
            "archetype_channel": arch_ch,
            "min_dist_sq": arch_dist,
            "scale_entropy_norm": float(scale_entropies[k]),
            "spatial_roi_entropy_norm": float(roi_entropies[k])
        })

    return pl.DataFrame(catalog)


# =============================================================================
# МОДУЛЬ 5: Визуализация вариативности всех режимов
# =============================================================================

def plot_scale_occupancy_heatmap(scale_mat: np.ndarray, windows: list[float], K: int, output_path: Path):
    """Функция 12: Полная матрица [Режим k x Временное окно w]."""
    plt.figure(figsize=(18, 10), dpi=150)
    extent = [windows[0], windows[-1], 0, K]

    im = plt.imshow(
        scale_mat, 
        aspect="auto", 
        origin="lower", 
        extent=extent, 
        cmap="magma", 
        interpolation="none"
    )
    plt.colorbar(im, label="Число каналов в режиме на данном масштабе", pad=0.02)
    plt.xlabel("Временной масштаб окна w (секунды)", fontsize=12)
    plt.ylabel("Индекс режима ОРМ k (отсортированы по энергии)", fontsize=12)
    plt.title(f"Спектр временных масштабов по ВСЕМ режимам ОРМ (K = {K})", fontsize=14, fontweight="bold")
    plt.axhline(77, color="cyan", linestyle="--", linewidth=1.2, label=r"Режим $\tau_{77}$ (Топ-активатор)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранена матрица масштабов: {output_path.name}")


def plot_roi_occupancy_heatmap(roi_mat: np.ndarray, rois: list[str], K: int, output_path: Path):
    """Функция 13: Матрица анатомического распределения [Режим k x MNE ROI]."""
    plt.figure(figsize=(14, 10), dpi=150)

    # Нормируем каждую строку к 100% для прозрачного сравнения долей
    row_sums = roi_mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi_perc = np.true_divide(roi_mat, row_sums) * 100.0
        roi_perc[~np.isfinite(roi_perc)] = 0.0

    im = plt.imshow(roi_perc, aspect="auto", origin="lower", cmap="viridis", interpolation="none")
    plt.colorbar(im, label="Доля участия анатомической зоны в моде (%)", pad=0.02)

    plt.xticks(range(len(rois)), rois, rotation=40, ha="right", fontsize=10)
    plt.ylabel("Индекс режима ОРМ k (0 .. K-1)", fontsize=12)
    plt.title(f"Анатомическая специализация режимов ОРМ по зонам MNE VectorView (K = {K})", fontsize=14, fontweight="bold")
    plt.axhline(77, color="red", linestyle="--", linewidth=1.5, label=r"Режим $\tau_{77}$")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранена анатомическая матрица: {output_path.name}")


def plot_specialization_phase_space(catalog_df: pl.DataFrame, K: int, output_path: Path):
    """Функция 14: Фазовая диаграмма: Пространственная локализация vs Масштабная локализация."""
    plt.figure(figsize=(12, 9), dpi=150)
    
    x = catalog_df["scale_entropy_norm"].to_numpy()
    y = catalog_df["spatial_roi_entropy_norm"].to_numpy()
    energies = catalog_df["mode_k"].to_numpy()

    sc = plt.scatter(x, y, c=energies, cmap="turbo", s=60, edgecolors="black", alpha=0.85)
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label("Индекс режима k (кинетическая энергия)", fontsize=11)

    # Подсветка tau_77
    row_77 = catalog_df.filter(pl.col("mode_k") == 77).to_dicts()[0]
    plt.scatter([row_77["scale_entropy_norm"]], [row_77["spatial_roi_entropy_norm"]], 
                color="red", s=180, edgecolors="white", linewidth=2.0, zorder=6)
    plt.annotate(
        r"$\tau_{77}$ (Правый затылок, $w \approx 3$с)",
        xy=(row_77["scale_entropy_norm"], row_77["spatial_roi_entropy_norm"]),
        xytext=(row_77["scale_entropy_norm"] + 0.03, row_77["spatial_roi_entropy_norm"] - 0.05),
        arrowprops=dict(facecolor="red", arrowstyle="->", shrinkA=0),
        fontweight="bold"
    )

    plt.axvline(0.5, color="gray", linestyle=":", alpha=0.6)
    plt.axhline(0.5, color="gray", linestyle=":", alpha=0.6)

    plt.text(0.05, 0.95, "Мульти-локальные\nрежимы одного окна", transform=plt.gca().transAxes, fontsize=10, verticalalignment="top")
    plt.text(0.05, 0.05, "Фокальные микро-модули\n(узкая зона + узкое окно)", transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom")
    plt.text(0.65, 0.95, "Глобальный фоновый шум\n(весь мозг + все окна)", transform=plt.gca().transAxes, fontsize=10, verticalalignment="top")

    plt.xlabel("Нормированная энтропия временных окон (0 = строго одно окно, 1 = все окна)", fontsize=11)
    plt.ylabel("Нормированная энтропия зон мозга (0 = строго один ROI, 1 = равномерно)", fontsize=11)
    plt.title(f"Фазовое пространство специализации режимов ОРМ (K = {K})", fontsize=13, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранена фазовая диаграмма энтропии: {output_path.name}")


def plot_corrected_topomap_density(df_k: pl.DataFrame, info: mne.Info, target_mode: int, output_path: Path):
    """Функция 15: Корректная отрисовка топокарты MNE через предварительный mne.pick_info."""
    sub = df_k.filter(pl.col("cluster_regime_id") == target_mode)
    ch_counts = sub.group_by("channel").len()
    density_map = {row["channel"]: row["len"] for row in ch_counts.iter_rows(named=True)}

    meg_picks = mne.pick_types(info, meg=True, eeg=False)
    info_meg = mne.pick_info(info, meg_picks)
    meg_names = info_meg["ch_names"]

    weights = np.array([density_map.get(ch, density_map.get(ch.replace(" ", ""), 0)) for ch in meg_names], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    # Корректный вызов без picks (сенсоры уже отобраны в info_meg)
    im, _ = mne.viz.plot_topomap(weights, info_meg, axes=ax, show=False, cmap="turbo", sphere="eeglab")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Число масштабов")
    ax.set_title(f"MNE-топокарта плотности режима $\\tau_{{{target_mode}}}$ на шлеме", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранена исправленная топокарта: {output_path.name}")


# =============================================================================
# МОДУЛЬ 6: Точка входа
# =============================================================================

def main():
    """Функция 16: Главный диспетчер."""
    parser = argparse.ArgumentParser(description="Глобальная вариативность каналов и окон ОРМ.")
    parser.add_argument("--workspace", type=str, default=None, help="Путь к корню проекта")
    parser.add_argument("--dataset", type=int, default=0, choices=[0, 1, 2], help="Код датасета (0: Bigrams)")
    parser.add_argument("--k", type=int, default=100, help="Число кластеров K")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_root(args.workspace)
    cfg = get_dataset_descriptors(workspace_dir, args.dataset)
    prefix = cfg["prefix"]

    out_dir = workspace_dir / "04_processed_db" / "orm_results" / prefix / f"global_variability_k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Чтение info
    info = extract_info_from_pysz_or_fif(cfg)
    roi_map = build_neuromag_roi_mapping(info)
    type_map = build_sensor_types_mapping(info)

    # 2. Загрузка данных разбиения
    df_k = load_all_memberships_for_k(workspace_dir, prefix, args.k, roi_map, type_map)

    # 3. Расчет матриц
    scale_mat, windows = compute_scale_occupancy_matrix(df_k, args.k)
    roi_mat, rois = compute_roi_occupancy_matrix(df_k, args.k)

    # 4. Энтропии
    scale_ent = compute_diversity_entropies(scale_mat)
    roi_ent = compute_diversity_entropies(roi_mat)

    # 5. Сводный каталог всех режимов
    catalog_df = build_comprehensive_modes_catalog(df_k, args.k, scale_mat, windows, roi_mat, rois, scale_ent, roi_ent)
    catalog_df.write_csv(out_dir / f"all_modes_k{args.k}_catalog.csv")
    catalog_df.write_parquet(out_dir / f"all_modes_k{args.k}_catalog.parquet")

    # 6. Графика
    plot_scale_occupancy_heatmap(scale_mat, windows, args.k, out_dir / "01_scale_occupancy_all_modes.png")
    plot_roi_occupancy_heatmap(roi_mat, rois, args.k, out_dir / "02_roi_occupancy_all_modes.png")
    plot_specialization_phase_space(catalog_df, args.k, out_dir / "03_specialization_entropy_phase_space.png")
    
    # Исправленная топокарта для моды 77
    plot_corrected_topomap_density(df_k, info, 77, out_dir / "04_mode_77_corrected_topomap.png")

    logger.info(f"\nАнализ вариативности завершен. Все таблицы и фазовые карты сохранены в:\n{out_dir}")


if __name__ == "__main__":
    main()
