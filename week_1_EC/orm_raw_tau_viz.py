#!/usr/bin/env python3
r"""
week_1_EC/orm_raw_tau_viz.py
Индивидуальный корреляционный анализ режимов ОРМ после фильтра Савицкого — Голея (p = 2):
- Точечная декомпрессия STI101 из .fif.pysz (без потерь);
- Срез последних 500 секунд записи;
- Фильтрация Савицкого — Голея (p = 2) для каждой отдельной моды tau_k(t);
- Гауссова свертка стимулов с широким физиологическим окном (sigma_sec = 1.0 с);
- Помодовый расчет корреляций с крестом и биграммой (zero-lag и с поиском лага);
- 2D-карта отклика [Мода k x Временной лаг \Delta t];
- Наложение сильнейших мод-активаторов и мод-супрессоров на временную шкалу стимулов.
"""

import os
import sys
import re
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pysz import sz
except ImportError:
    sz = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ORM_SAVGOL_MODE_CORR")

# =============================================================================
# МОДУЛЬ 1: Обнаружение путей и декомпрессия STI101
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
    """Функция 2: Реестр путей к архивам .pysz и результатам ОРМ."""
    raw_dir = workspace_dir / "01_raw_data"
    registry = {
        0: {
            "name": "Type 1 (HSE Bigrams) - Chistova Alena",
            "prefix": "type1_sub10_chistova",
            "pysz_file": raw_dir / "type1_bigrams/type1_sub10_chistova_alena_raw.fif.pysz",
            "stim_ch": "STI101"
        },
        1: {
            "name": "Type 2 (Ushakov Drowsiness) - Bainbridge Emily",
            "prefix": "type2_bainbridge",
            "pysz_file": raw_dir / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif.pysz",
            "stim_ch": "STI101"
        },
        2: {
            "name": "Type 3 (Spanish BCBL) - Subject 01 Block 1",
            "prefix": "type3_bcbl_01",
            "pysz_file": raw_dir / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw.fif.pysz",
            "stim_ch": "STI101"
        }
    }
    if dataset_code not in registry:
        raise ValueError(f"Неизвестный dataset_code: {dataset_code}")
    return registry[dataset_code]


def decompress_stim_channel_from_pysz(pysz_path: Path, stim_channel_name: str = "STI101") -> tuple[np.ndarray, float]:
    """Функция 3: Точечная распаковка исключительно канала стимулов из архива pysz."""
    if sz is None:
        raise ImportError("Пакет 'pysz' не установлен в окружении.")

    if not pysz_path.exists():
        raise FileNotFoundError(f"Архив pysz не найден: {pysz_path}")

    logger.info(f"Распаковка {stim_channel_name} из архива: {pysz_path.name}")
    with open(pysz_path, "rb") as f:
        payload = pickle.load(f)

    info = payload["info"]
    n_times = payload["n_times"]
    chunks = payload["chunks"]
    sfreq = float(info["sfreq"])

    if stim_channel_name not in info["ch_names"]:
        raise ValueError(f"Канал {stim_channel_name} не найден в {pysz_path.name}")

    stim_idx = info["ch_names"].index(stim_channel_name)
    chunk = chunks[stim_idx]

    decompressed, _ = sz.decompress(chunk, np.float32, (n_times,))
    stim_data = np.round(decompressed).astype(np.int64)

    logger.info(f"Распакован канал {stim_channel_name}: {len(stim_data)} отсчетов (fs = {sfreq} Гц)")
    return stim_data, sfreq


def extract_event_onsets(stim_data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Функция 4: Извлечение фронтов фиксационного креста (код 8) и биграммы (+0.5с)."""
    is_cross = (stim_data == 8) | ((stim_data & 8) == 8)
    onsets = np.where((~is_cross[:-1]) & is_cross[1:])[0] + 1

    min_dist_samples = int(0.200 * sfreq)
    filtered = []
    last_s = -min_dist_samples

    for s in onsets:
        if s - last_s >= min_dist_samples:
            filtered.append(s)
            last_s = s

    cross_times_sec = np.array(filtered, dtype=np.int64) / sfreq
    bigram_times_sec = cross_times_sec + 0.500

    logger.info(f"Найдено событий: {len(cross_times_sec)} крестов, {len(bigram_times_sec)} биграмм")
    return cross_times_sec, bigram_times_sec


# =============================================================================
# МОДУЛЬ 2: Загрузка, обрезка и фильтр Савицкого — Голея (p = 2)
# =============================================================================

def load_trajectories_and_cut_tail(
    trajectories_dir: Path,
    K: int,
    trim_end_sec: float = 500.0
) -> tuple[np.ndarray, np.ndarray]:
    """Функция 5: Чтение tau_k(t) и жесткое отсечение последних 500 секунд."""
    fpath = trajectories_dir / f"trajectories_k{K}.parquet"
    df = pl.read_parquet(fpath)

    time_sec = df["time_sec"].to_numpy().astype(np.float64)
    tau_cols = sorted([c for c in df.columns if c.startswith("tau_")], key=lambda x: int(x.split("_")[1]))
    tau_matrix = np.vstack([df[c].to_numpy().astype(np.float32) for c in tau_cols])

    cutoff_time = time_sec[-1] - trim_end_sec
    mask = time_sec <= cutoff_time

    t_clean = time_sec[mask]
    tau_clean = tau_matrix[:, mask]

    logger.info(f"[K={K}] Отрезано {trim_end_sec:.1f}с. Диапазон: [{t_clean[0]:.1f}с .. {t_clean[-1]:.1f}с]")
    return t_clean, tau_clean


def apply_savitzky_golay_quadratic(
    tau_matrix: np.ndarray,
    dt: float,
    window_sec: float = 0.7
) -> np.ndarray:
    """Функция 6: Поканальная фильтрация Савицкого — Голея с квадратичным полиномом (p = 2)."""
    window_samples = int(round(window_sec / dt))
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = max(5, window_samples)

    polyorder = 2
    logger.info(f"Фильтр Савицкого-Голея: polyorder={polyorder}, window={window_samples} отсчетов ({window_samples * dt:.3f} с)")

    tau_smoothed = savgol_filter(tau_matrix, window_length=window_samples, polyorder=polyorder, axis=1, mode="interp")
    tau_smoothed = np.clip(tau_smoothed, a_min=0.0, a_max=None)
    return tau_smoothed


def build_smooth_stimulus_signal(time_sec: np.ndarray, event_times: np.ndarray, sigma_sec: float = 1.0) -> np.ndarray:
    """Функция 7: Непрерывная функция стимуляции через гауссову свертку (sigma_sec = 1.0 с)."""
    if len(event_times) == 0:
        return np.zeros_like(time_sec)

    dt = float(np.median(np.diff(time_sec)))
    sigma_samp = max(1.0, sigma_sec / dt)

    train = np.zeros(len(time_sec), dtype=np.float32)
    idxs = np.searchsorted(time_sec, event_times)
    valid = idxs[idxs < len(time_sec)]
    train[valid] = 1.0

    smooth = gaussian_filter1d(train, sigma=sigma_samp, mode="nearest")
    m_val = np.max(smooth)
    if m_val > 0:
        smooth /= m_val
    return smooth


# =============================================================================
# МОДУЛЬ 3: Помодовый корреляционный анализ (Zero-lag и с лагом)
# =============================================================================

def compute_individual_zero_lag_correlations(
    tau_matrix_sg: np.ndarray,
    regressor: np.ndarray
) -> np.ndarray:
    """Функция 8: Расчет корреляции Пирсона каждого очищенного режима с регрессором при лаге 0."""
    K = tau_matrix_sg.shape[0]
    corrs = np.zeros(K, dtype=np.float32)
    reg_std = np.std(regressor)

    if reg_std == 0:
        return corrs

    for k in range(K):
        sig = tau_matrix_sg[k]
        s_std = np.std(sig)
        if s_std > 0:
            corrs[k] = float(np.corrcoef(sig, regressor)[0, 1])
        else:
            corrs[k] = 0.0

    return corrs


def compute_mode_lagged_cross_correlations(
    tau_matrix_sg: np.ndarray,
    stimulus_signal: np.ndarray,
    dt: float,
    min_lag_sec: float = -0.5,
    max_lag_sec: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Функция 9: 2D-матрица кросс-корреляций [K x Lags] и извлечение оптимального лага для каждого k."""
    K = tau_matrix_sg.shape[0]
    min_lag_samp = int(round(min_lag_sec / dt))
    max_lag_samp = int(round(max_lag_sec / dt))
    lags_samples = np.arange(min_lag_samp, max_lag_samp + 1)
    lags_sec = lags_samples * dt

    n_lags = len(lags_samples)
    cross_corr_2d = np.zeros((K, n_lags), dtype=np.float32)

    stim_norm = (stimulus_signal - np.mean(stimulus_signal)) / (np.std(stimulus_signal) + 1e-12)
    n = len(stim_norm)

    peak_corrs = np.zeros(K, dtype=np.float32)
    optimal_lags = np.zeros(K, dtype=np.float32)

    for k in range(K):
        sig = tau_matrix_sg[k]
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-12)

        for l_idx, l in enumerate(lags_samples):
            if l < 0:
                c = np.mean(sig_norm[:l] * stim_norm[-l:])
            elif l > 0:
                c = np.mean(sig_norm[l:] * stim_norm[:n-l])
            else:
                c = np.mean(sig_norm * stim_norm)
            cross_corr_2d[k, l_idx] = float(c)

        # Поиск максимального положительного отклика в физиологическом окне [0 .. max_lag]
        valid_pos_lags = np.where(lags_sec >= 0.0)[0]
        if len(valid_pos_lags) > 0:
            best_idx_in_valid = np.argmax(cross_corr_2d[k, valid_pos_lags])
            best_global_idx = valid_pos_lags[best_idx_in_valid]
            peak_corrs[k] = cross_corr_2d[k, best_global_idx]
            optimal_lags[k] = lags_sec[best_global_idx]

    return lags_sec, cross_corr_2d, peak_corrs, optimal_lags


# =============================================================================
# МОДУЛЬ 4: Визуализация индивидуальных откликов мод
# =============================================================================

def plot_individual_correlations_spectrum(
    corr_cross: np.ndarray,
    corr_bigram: np.ndarray,
    peak_corr_bigram: np.ndarray,
    K: int,
    output_path: Path
):
    """Функция 10: Спектр индивидуальных корреляций всех K мод (zero-lag vs оптимальный лаг)."""
    plt.figure(figsize=(16, 7), dpi=150)
    modes = np.arange(K)

    plt.plot(modes, corr_cross, marker="o", markersize=3, label="Крест (zero-lag)", color="#1f77b4", alpha=0.7, linewidth=1.2)
    plt.plot(modes, corr_bigram, marker="s", markersize=3, label="Биграмма (zero-lag)", color="#ff7f0e", alpha=0.7, linewidth=1.2)
    plt.plot(modes, peak_corr_bigram, marker="^", markersize=4, label="Биграмма (Пик с учетом фазового лага)", color="#d62728", linewidth=1.8)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    # Подсветка сильнейшей моды-активатора
    top_k = int(np.argmax(peak_corr_bigram))
    plt.scatter([top_k], [peak_corr_bigram[top_k]], color="red", s=90, zorder=5)
    plt.annotate(
        f"Топ-активатор: k={top_k}\n$r^* = {peak_corr_bigram[top_k]:.3f}$",
        xy=(top_k, peak_corr_bigram[top_k]),
        xytext=(top_k, peak_corr_bigram[top_k] + 0.05),
        arrowprops=dict(facecolor="red", arrowstyle="->", shrinkA=0),
        fontweight="bold"
    )

    # Подсветка сильнейшей моды-супрессора
    min_k = int(np.argmin(corr_bigram))
    plt.scatter([min_k], [corr_bigram[min_k]], color="blue", s=90, zorder=5)
    plt.annotate(
        f"Топ-супрессор: k={min_k}\n$r = {corr_bigram[min_k]:.3f}$",
        xy=(min_k, corr_bigram[min_k]),
        xytext=(min_k, corr_bigram[min_k] - 0.06),
        arrowprops=dict(facecolor="blue", arrowstyle="->", shrinkA=0),
        fontweight="bold"
    )

    plt.title(f"Спектр индивидуальных корреляций мод ОРМ (K = {K}, $\\sigma = 1.0$ с)", fontsize=13, fontweight="bold")
    plt.xlabel("Индекс режима k (упорядочены по энергии: от покоя к турбулентности)", fontsize=11)
    plt.ylabel("Коэффициент корреляции Пирсона r", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранен спектр корреляций: {output_path.name}")


def plot_2d_mode_lag_cross_correlation_map(
    cross_corr_2d: np.ndarray,
    lags_sec: np.ndarray,
    K: int,
    output_path: Path
):
    """Функция 11: 2D-карта отклика: Режим k (ось Y) x Временной лаг (ось X)."""
    plt.figure(figsize=(14, 10), dpi=150)
    extent = [lags_sec[0] * 1000.0, lags_sec[-1] * 1000.0, 0, K]

    im = plt.imshow(
        cross_corr_2d,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        interpolation="none"
    )
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(r"Корреляция $r(k, \Delta t)$", fontsize=11)

    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Момент биграммы (t=0)")

    plt.xlabel(r"Фазовый лаг $\Delta t$ (миллисекунды): $>0$ — реакция ПОСЛЕ стимула", fontsize=11)
    plt.ylabel("Индекс режима ОРМ k (0 .. K-1)", fontsize=11)
    plt.title(f"2D-ландшафт фазовых откликов мод ОРМ на биграмму (K = {K}, $\\sigma = 1.0$ с)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранена 2D-карта откликов: {output_path.name}")


def plot_top_modes_overlay_with_stimuli(
    time_sec: np.ndarray,
    tau_matrix_sg: np.ndarray,
    s_cross: np.ndarray,
    s_bigram: np.ndarray,
    top_k_activators: list[int],
    top_k_suppressors: list[int],
    K: int,
    output_path: Path,
    zoom_span: tuple[float, float] = (150.0, 200.0)
):
    """Функция 12: Наглядное сопоставление топ-активаторов и топ-супрессоров со стимулами на 50с отрезке."""
    t0, t1 = zoom_span
    mask = (time_sec >= t0) & (time_sec <= t1)
    t_sub = time_sec[mask]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10), sharex=True, dpi=150)

    # 1. Стимулы
    ax1.plot(t_sub, s_cross[mask], color="blue", linewidth=1.5, alpha=0.7, label="Крест (t0)")
    ax1.plot(t_sub, s_bigram[mask], color="red", linewidth=1.5, alpha=0.7, label="Биграмма (t0+0.5с)")
    ax1.set_ylabel("Стимулы", fontsize=10)
    ax1.set_title(f"Динамика топ-мод ОРМ и стимуляции в окне [{t0:.0f}с .. {t1:.0f}с] (K = {K}, $\\sigma = 1.0$ с)", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(loc="upper right")

    # 2. Моды-активаторы (вспыхивают при чтении)
    for k in top_k_activators:
        sig = tau_matrix_sg[k, mask]
        sig_norm = sig / (np.max(sig) + 1e-12)
        ax2.plot(t_sub, sig_norm, linewidth=1.4, label=f"$\\tau_{{{k}}}^{{\\mathrm{{SG}}}}(t)$ (Активатор)")
    ax2.set_ylabel("Активаторы", fontsize=10)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right")

    # 3. Моды-супрессоры (подавляются при чтении)
    for k in top_k_suppressors:
        sig = tau_matrix_sg[k, mask]
        sig_norm = sig / (np.max(sig) + 1e-12)
        ax3.plot(t_sub, sig_norm, linewidth=1.4, linestyle="--", label=f"$\\tau_{{{k}}}^{{\\mathrm{{SG}}}}(t)$ (Супрессор)")
    ax3.set_ylabel("Супрессоры", fontsize=10)
    ax3.set_xlabel("Физическое время t (секунды)", fontsize=11)
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Сохранено наложение топ-мод: {output_path.name}")


# =============================================================================
# МОДУЛЬ 5: Исполнительный координатор
# =============================================================================

def process_k_individual_correlations(
    workspace_dir: Path,
    trajectories_dir: Path,
    out_dir_k: Path,
    cross_times: np.ndarray,
    bigram_times: np.ndarray,
    K: int,
    trim_end_sec: float,
    savgol_window_sec: float,
    sigma_sec: float
):
    """Функция 13: Расчет и визуализация индивидуальных корреляций для одного K."""
    logger.info(f"==================== ИНДИВИДУАЛЬНЫЕ КОРРЕЛЯЦИИ ДЛЯ K = {K} ====================")
    t_clean, tau_clean = load_trajectories_and_cut_tail(trajectories_dir, K, trim_end_sec=trim_end_sec)
    dt = float(np.median(np.diff(t_clean)))

    # 1. Фильтрация Савицкого — Голея
    tau_sg = apply_savitzky_golay_quadratic(tau_clean, dt=dt, window_sec=savgol_window_sec)

    # 2. Непрерывные функции стимуляции с заданным sigma_sec
    s_cross = build_smooth_stimulus_signal(t_clean, cross_times, sigma_sec=sigma_sec)
    s_bigram = build_smooth_stimulus_signal(t_clean, bigram_times, sigma_sec=sigma_sec)

    # 3. Мгновенные корреляции (Zero-lag)
    corr_cross = compute_individual_zero_lag_correlations(tau_sg, s_cross)
    corr_bigram = compute_individual_zero_lag_correlations(tau_sg, s_bigram)

    # 4. Лаговая кросс-корреляция по каждой моде
    lags_sec, cross_corr_2d, peak_corrs, opt_lags = compute_mode_lagged_cross_correlations(
        tau_sg, s_bigram, dt=dt, min_lag_sec=-0.5, max_lag_sec=2.0
    )

    # 5. Сохранение полной таблицы в Parquet
    df_corrs = pl.DataFrame({
        "k_mode": np.arange(K),
        "corr_cross_zero_lag": corr_cross,
        "corr_bigram_zero_lag": corr_bigram,
        "peak_corr_bigram": peak_corrs,
        "optimal_lag_sec": opt_lags,
        "optimal_lag_ms": opt_lags * 1000.0
    })
    df_corrs.write_parquet(out_dir_k / f"mode_correlations_k{K}.parquet")

    # 6. Отбор топ-активаторов и топ-супрессоров
    top_activators = list(np.argsort(peak_corrs)[-3:][::-1])
    top_suppressors = list(np.argsort(corr_bigram)[:3])

    logger.info(f"Топ-3 активатора: {top_activators} (r*={peak_corrs[top_activators]})")
    logger.info(f"Топ-3 супрессора: {top_suppressors} (r={corr_bigram[top_suppressors]})")

    # 7. Графика
    plot_individual_correlations_spectrum(corr_cross, corr_bigram, peak_corrs, K, out_dir_k / f"01_mode_correlations_spectrum_k{K}.png")
    plot_2d_mode_lag_cross_correlation_map(cross_corr_2d, lags_sec, K, out_dir_k / f"02_mode_lag_heatmap_k{K}.png")
    plot_top_modes_overlay_with_stimuli(
        t_clean, tau_sg, s_cross, s_bigram,
        top_activators, top_suppressors, K,
        out_dir_k / f"03_top_modes_vs_stimuli_k{K}.png"
    )


def main():
    """Функция 14: Точка входа."""
    parser = argparse.ArgumentParser(description="Помодовый корреляционный анализ ОРМ после Савицкого — Голея.")
    parser.add_argument("--workspace", type=str, default=None, help="Путь к корню проекта")
    parser.add_argument("--dataset", type=int, default=0, choices=[0, 1, 2], help="Код датасета (0, 1, 2)")
    parser.add_argument("--trim-end-sec", type=float, default=500.0, help="Длительность обрезки финала записи (с)")
    parser.add_argument("--savgol-win-sec", type=float, default=0.7, help="Длительность окна Савицкого-Голея в секундах (по умолчанию 0.7с)")
    parser.add_argument("--sigma-sec", type=float, default=1.0, help="Ширина гауссова сглаживания стимулов (по умолчанию 1.0с)")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_root(args.workspace)
    cfg = get_dataset_descriptors(workspace_dir, args.dataset)
    prefix = cfg["prefix"]

    base_results_dir = workspace_dir / "04_processed_db" / "orm_results" / prefix
    trajectories_dir = base_results_dir / "trajectories"

    # Автопоиск сетки K
    files = sorted(trajectories_dir.glob("trajectories_k*.parquet"))
    all_k = sorted(list(set([int(re.search(r"trajectories_k(\d+)\.parquet", f.name).group(1)) for f in files])))
    logger.info(f"Найдена сетка K: {all_k}")

    # Точечная декомпрессия STI101 из .pysz
    stim_data, sfreq = decompress_stim_channel_from_pysz(cfg["pysz_file"], cfg["stim_ch"])
    cross_times, bigram_times = extract_event_onsets(stim_data, sfreq)

    out_root = base_results_dir / "figures_mode_correlations_savgol"
    out_root.mkdir(parents=True, exist_ok=True)

    for k_val in all_k:
        out_dir_k = out_root / f"k_{k_val}"
        out_dir_k.mkdir(parents=True, exist_ok=True)
        process_k_individual_correlations(
            workspace_dir=workspace_dir,
            trajectories_dir=trajectories_dir,
            out_dir_k=out_dir_k,
            cross_times=cross_times,
            bigram_times=bigram_times,
            K=k_val,
            trim_end_sec=args.trim_end_sec,
            savgol_window_sec=args.savgol_win_sec,
            sigma_sec=args.sigma_sec
        )

    logger.info(f"Помодовый корреляционный анализ успешно завершен: {out_root}")


if __name__ == "__main__":
    main()
