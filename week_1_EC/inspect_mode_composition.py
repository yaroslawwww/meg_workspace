#!/usr/bin/env python3
r"""
week_1_EC/inspect_mode_composition.py
Декомпозиция состава режима ОРМ на основе встроенных функций MNE:
- Извлечение структуры mne.Info напрямую из pickle-архива .fif.pysz без распаковки;
- Привязка сенсоров к анатомическим долям через mne.read_vectorview_selection;
- Определение типа сенсора (grad/mag) через mne.channel_type;
- Извлечение всех пар (канал, масштаб) ячейки Вороного Omega_k;
- Ранжирование по удаленности от центроида tau_k;
- Построение гистограммы временных масштабов и топокарты на шлеме МЭГ.
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
logger = logging.getLogger("ORM_MODE_DECOMPOSITION")

# =============================================================================
# МОДУЛЬ 1: Пути и извлечение mne.Info напрямую из .fif.pysz
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
    """Функция 2: Пути к базам ОРМ и архивам .fif.pysz."""
    raw_dir = workspace_dir / "01_raw_data"
    registry = {
        0: {
            "name": "Type 1 (HSE Bigrams) - Chistova Alena",
            "prefix": "type1_sub10_chistova",
            "pysz_file": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw.fif.pysz",
            "raw_fif": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw.fif",
            "prep_fif": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif"
        },
        1: {
            "name": "Type 2 (Ushakov Drowsiness) - Bainbridge Emily",
            "prefix": "type2_bainbridge",
            "pysz_file": raw_dir / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif.pysz",
            "raw_fif": raw_dir / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif",
            "prep_fif": raw_dir / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif"
        },
        2: {
            "name": "Type 3 (Spanish BCBL) - Subject 01 Block 1",
            "prefix": "type3_bcbl_01",
            "pysz_file": raw_dir / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw.fif.pysz",
            "raw_fif": raw_dir / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw.fif",
            "prep_fif": raw_dir / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw_preprocessed.fif"
        }
    }
    if dataset_code not in registry:
        raise ValueError(f"Неизвестный dataset_code: {dataset_code}")
    return registry[dataset_code]


def extract_info_from_pysz_or_fif(cfg: dict) -> mne.Info:
    """Функция 3: Извлечение структуры mne.Info напрямую из payload архива .fif.pysz без распаковки каналов."""
    # 1. Если существует распакованный .fif, читаем его заголовок
    for cand in [cfg.get("prep_fif"), cfg.get("raw_fif")]:
        if cand and cand.exists():
            logger.info(f"Чтение info из распакованного FIF: {cand.name}")
            return mne.io.read_info(cand, verbose=False)

    # 2. Иначе считываем mne.Info напрямую из pickle архива .fif.pysz
    pysz_path = cfg["pysz_file"]
    if not pysz_path.exists():
        raise FileNotFoundError(f"Файл архива .fif.pysz не найден: {pysz_path}")

    logger.info(f"Чтение mne.Info напрямую из структуры архива: {pysz_path.name}")
    with open(pysz_path, "rb") as f:
        payload = pickle.load(f)

    info = payload["info"]
    logger.info(f"Успешно извлечена структура mne.Info ({len(info['ch_names'])} каналов, sfreq={info['sfreq']} Гц)")
    return info


# =============================================================================
# МОДУЛЬ 2: Встроенная анатомическая и сенсорная разметка MNE
# =============================================================================

def build_builtin_neuromag_roi_mapping(info: mne.Info) -> dict[str, str]:
    """Функция 4: Извлечение анатомических зон сенсоров через штатную mne.read_vectorview_selection."""
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
            # Штатная функция MNE для Elekta/Neuromag VectorView
            ch_list = mne.read_vectorview_selection(region_name, info=info, verbose=False)
            for ch in ch_list:
                mapping[ch] = region_name
                mapping[ch.replace(" ", "")] = region_name
        except Exception as e:
            logger.warning(f"Не удалось получить выборку {region_name}: {e}")

    logger.info(f"Штатная разметка VectorView MNE сформирована для {len(mapping) // 2} сенсоров")
    return mapping


def build_channel_types_mapping(info: mne.Info) -> dict[str, str]:
    """Функция 5: Определение типа сенсора (grad/mag) через штатную mne.channel_type."""
    type_map = {}
    for idx, ch_name in enumerate(info["ch_names"]):
        c_type = mne.channel_type(info, idx)
        type_map[ch_name] = c_type
        type_map[ch_name.replace(" ", "")] = c_type
    return type_map


# =============================================================================
# МОДУЛЬ 3: Загрузка ячеек Вороного и обогащение метаданными
# =============================================================================

def parse_window_tag_attributes(window_tag: str) -> tuple[float, int]:
    """Функция 6: Разбор длительности окна (с) и порядка перестановки m из тега."""
    m = re.search(r"w_([0-9.]+)s_m(\d+)", window_tag)
    if not m:
        return 0.0, 0
    return float(m.group(1)), int(m.group(2))


def load_mode_memberships_enriched(
    workspace_dir: Path, 
    prefix: str, 
    K: int, 
    target_mode: int,
    roi_mapping: dict[str, str],
    type_mapping: dict[str, str]
) -> pl.DataFrame:
    """Функция 7: Загрузка состава ячейки Omega_k и привязка MNE-метаданных."""
    mem_path = workspace_dir / "04_processed_db" / "orm_results" / prefix / "orm_v1_memberships.parquet"
    if not mem_path.exists():
        raise FileNotFoundError(f"Файл memberships не найден: {mem_path}")

    df_all = pl.read_parquet(mem_path)
    df_mode = df_all.filter(
        (pl.col("k_clusters") == K) & 
        (pl.col("cluster_regime_id") == target_mode)
    ).sort("dist_sq_to_center")

    channels = df_mode["channel"].to_list()
    window_tags = df_mode["window_tag"].to_list()

    durations, orders, rois, types = [], [], [], []
    for ch, tag in zip(channels, window_tags):
        w_sec, m_order = parse_window_tag_attributes(tag)
        durations.append(w_sec)
        orders.append(m_order)
        rois.append(roi_mapping.get(ch, roi_mapping.get(ch.replace(" ", ""), "Unknown")))
        types.append(type_mapping.get(ch, type_mapping.get(ch.replace(" ", ""), "meg")))

    df_enriched = df_mode.with_columns([
        pl.Series("window_sec", durations, dtype=pl.Float32),
        pl.Series("order_m", orders, dtype=pl.Int32),
        pl.Series("mne_roi", rois, dtype=pl.Utf8),
        pl.Series("sensor_type", types, dtype=pl.Utf8)
    ])

    logger.info(f"В режим tau_{target_mode} (K={K}) вошло {len(df_enriched)} объектов (пар канал-масштаб)")
    return df_enriched


# =============================================================================
# МОДУЛЬ 4: Статистическая декомпозиция состава
# =============================================================================

def summarize_roi_and_channel_distribution(df_mode: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Функция 8: Агрегация по анатомическим ROI MNE и по отдельным каналам."""
    roi_summary = (
        df_mode.group_by("mne_roi")
        .agg([
            pl.len().alias("pair_count"),
            pl.col("channel").n_unique().alias("unique_channels"),
            pl.col("dist_sq_to_center").mean().alias("mean_dist_sq")
        ])
        .sort("pair_count", descending=True)
    )

    ch_summary = (
        df_mode.group_by(["channel", "mne_roi", "sensor_type"])
        .agg([
            pl.len().alias("scale_count"),
            pl.col("dist_sq_to_center").min().alias("min_dist_sq"),
            pl.col("dist_sq_to_center").mean().alias("mean_dist_sq")
        ])
        .sort(["scale_count", "min_dist_sq"], descending=[True, False])
    )
    return roi_summary, ch_summary


def summarize_scale_distribution(df_mode: pl.DataFrame) -> pl.DataFrame:
    """Функция 9: Агрегация по временным масштабам (окнам)."""
    scale_summary = (
        df_mode.group_by(["window_tag", "window_sec", "order_m"])
        .agg([
            pl.len().alias("channel_count"),
            pl.col("dist_sq_to_center").mean().alias("mean_dist_sq")
        ])
        .sort("window_sec")
    )
    return scale_summary


# =============================================================================
# МОДУЛЬ 5: Визуализация топографии и масштабов
# =============================================================================

def plot_scale_spectrum_distribution(scale_summary: pl.DataFrame, K: int, target_mode: int, output_path: Path):
    """Функция 10: Спектр временных масштабов в составе моды."""
    w_sec = scale_summary["window_sec"].to_numpy()
    ch_counts = scale_summary["channel_count"].to_numpy()
    orders = scale_summary["order_m"].to_numpy()

    plt.figure(figsize=(14, 6), dpi=150)
    color_map = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c", 6: "#d62728"}
    colors = [color_map.get(m, "gray") for m in orders]

    plt.bar(w_sec, ch_counts, width=0.07, color=colors, edgecolor="black", alpha=0.85)

    plt.title(f"Спектр временных масштабов ячейки $\\tau_{{{target_mode}}}$ (K = {K})", fontsize=13, fontweight="bold")
    plt.xlabel("Длительность окна энтропии w (секунды)", fontsize=11)
    plt.ylabel("Количество каналов на данном масштабе", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=f"m = {m}") for m, c in color_map.items()]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранено распределение масштабов: {output_path.name}")


def plot_mne_topomap_density(ch_summary: pl.DataFrame, info: mne.Info, K: int, target_mode: int, output_path: Path):
    """Функция 11: Официальная 2D-топокарта MNE плотности каналов на сенсорном шлеме."""
    meg_picks = mne.pick_types(info, meg=True, eeg=False)
    meg_ch_names = [info["ch_names"][i] for i in meg_picks]

    # Сопоставляем каждому сенсору число окон в моде
    density_dict = {row["channel"]: row["scale_count"] for row in ch_summary.iter_rows(named=True)}
    weights = np.array([density_dict.get(ch, density_dict.get(ch.replace(" ", ""), 0)) for ch in meg_ch_names], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    im, _ = mne.viz.plot_topomap(
        weights, 
        info, 
        picks=meg_picks, 
        axes=ax, 
        show=False, 
        cmap="turbo", 
        sphere="eeglab"
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Число масштабов в составе режима")
    ax.set_title(f"MNE-топокарта режима $\\tau_{{{target_mode}}}$ на сенсорах МЭГ (K = {K})", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранена официальная топокарта MNE: {output_path.name}")


def plot_top_channels_bar_chart(ch_summary: pl.DataFrame, K: int, target_mode: int, output_path: Path, top_n: int = 25):
    """Функция 12: Барчарт топ-каналов со штатной зоной MNE и типом датчика."""
    top_df = ch_summary.head(top_n)
    channels = top_df["channel"].to_list()[::-1]
    counts = top_df["scale_count"].to_list()[::-1]
    rois = top_df["mne_roi"].to_list()[::-1]
    types = top_df["sensor_type"].to_list()[::-1]

    plt.figure(figsize=(13, 10), dpi=150)
    bars = plt.barh(channels, counts, color="#2b5c8f", edgecolor="black", alpha=0.85)

    for bar, roi, stype in zip(bars, rois, types):
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2.0, 
                 f" [{stype}] {roi}", verticalalignment="center", fontsize=9, color="#222222")

    plt.xlim(0, max(counts) + 15)
    plt.xlabel("Число окон/масштабов, вошедших в моду", fontsize=11)
    plt.ylabel("МЭГ-канал", fontsize=11)
    plt.title(f"Топ-{top_n} каналов режима $\\tau_{{{target_mode}}}$ (K = {K}) | MNE VectorView ROI", fontsize=13, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранен барчарт каналов: {output_path.name}")


# =============================================================================
# МОДУЛЬ 6: Консольный вывод и координатор
# =============================================================================

def print_detailed_decomposition(
    df_mode: pl.DataFrame, 
    roi_summary: pl.DataFrame, 
    target_mode: int, 
    top_n: int = 20
):
    """Функция 13: Вывод в консоль анатомического распределения и ядра кластера."""
    logger.info(f"\n==================== РАСПРЕДЕЛЕНИЕ ПО MNE VECTORVIEW ROI ====================")
    logger.info(f"{'Анатомическая зона (MNE Selection)':<25} | {'Всего пар (c, w)':<16} | {'Уникальных каналов'}")
    logger.info("-" * 65)
    for row in roi_summary.iter_rows(named=True):
        logger.info(f"{row['mne_roi']:<25} | {row['pair_count']:<16} | {row['unique_channels']}")

    logger.info(f"\n==================== ЯДРО РЕЖИМА tau_{target_mode} (ТОП-{top_n} АРХЕТИПОВ) ====================")
    logger.info(f"{'Ранг':<5} | {'Канал':<10} | {'Тип':<6} | {'Масштаб (окно)':<20} | {'w (сек)':<8} | {'m':<3} | {'MNE ROI':<20} | {'D^2 до центра'}")
    logger.info("-" * 95)

    for rank, row in enumerate(df_mode.head(top_n).iter_rows(named=True), start=1):
        logger.info(
            f"{rank:<5} | {row['channel']:<10} | {row['sensor_type']:<6} | {row['window_tag']:<20} | "
            f"{row['window_sec']:<8.3f} | {row['order_m']:<3} | {row['mne_roi']:<20} | {row['dist_sq_to_center']:.4e}"
        )


def main():
    """Функция 14: Точка входа."""
    parser = argparse.ArgumentParser(description="Декомпозиция режима Вороного tau_k через встроенные функции MNE.")
    parser.add_argument("--workspace", type=str, default=None, help="Путь к корню проекта")
    parser.add_argument("--dataset", type=int, default=0, choices=[0, 1, 2], help="Код датасета (0: Bigrams)")
    parser.add_argument("--k", type=int, default=100, help="Размерность разбиения K")
    parser.add_argument("--mode", type=int, default=77, help="Номер исследуемого режима (tau_k)")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_root(args.workspace)
    cfg = get_dataset_descriptors(workspace_dir, args.dataset)
    prefix = cfg["prefix"]

    out_dir = workspace_dir / "04_processed_db" / "orm_results" / prefix / f"mode_{args.mode}_decomposition_k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Чтение mne.Info напрямую из .fif.pysz архива (без распаковки .fif)
    info = extract_info_from_pysz_or_fif(cfg)

    # 2. Построение штатной разметки MNE VectorView
    roi_mapping = build_builtin_neuromag_roi_mapping(info)
    type_mapping = build_channel_types_mapping(info)

    # 3. Извлечение и обогащение состава ячейки Вороного
    df_mode = load_mode_memberships_enriched(workspace_dir, prefix, args.k, args.mode, roi_mapping, type_mapping)

    # 4. Агрегация
    roi_summary, ch_summary = summarize_roi_and_channel_distribution(df_mode)
    scale_summary = summarize_scale_distribution(df_mode)

    # 5. Экспорт таблиц
    df_mode.write_parquet(out_dir / f"mode_{args.mode}_full_pairs.parquet")
    df_mode.write_csv(out_dir / f"mode_{args.mode}_full_pairs.csv")
    roi_summary.write_csv(out_dir / f"mode_{args.mode}_roi_distribution.csv")
    ch_summary.write_csv(out_dir / f"mode_{args.mode}_channels_ranking.csv")
    scale_summary.write_csv(out_dir / f"mode_{args.mode}_scales_ranking.csv")

    # 6. Вывод в консоль
    print_detailed_decomposition(df_mode, roi_summary, target_mode=args.mode, top_n=20)

    # 7. Визуализация
    plot_scale_spectrum_distribution(scale_summary, args.k, args.mode, out_dir / "01_scale_spectrum.png")
    plot_top_channels_bar_chart(ch_summary, args.k, args.mode, out_dir / "02_top_channels.png", top_n=30)
    plot_mne_topomap_density(ch_summary, info, args.k, args.mode, out_dir / "03_meg_helmet_topomap.png")

    logger.info(f"Декомпозиция успешно завершена. Все файлы в: {out_dir}")


if __name__ == "__main__":
    main()
