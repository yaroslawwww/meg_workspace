#!/usr/bin/env python3
"""
visualize_qcml_multi_analysis.py
Комплексный анализ и визуализация внутренней размерности QCML:
1. Иерархия: figures/{subject}/{condition}/N{N}/
2. Честная детекция d=0 (включая бин [-0.5, 0.5) для Empty Room).
3. Биграммы (Type 1):
   - 3 отдельных графика по методам (Ratio Gap, RMT, Threshold 0.5).
   - Внутри каждого графика объединены 3 события (Крест, Биграмма, Ответ).
   - Единая шкала Y по всем валидным панелям (mean ± 1 SEM, ddof=1).
4. Сон (Type 2):
   - Поточечные (point-wise) Violin Plot и целочисленные гистограммы режимов.
   - Корректная синхронизация секунд (интервалы [t, t+1)).
5. Гистограммы сравнения:
   - Внутри каждого субъекта по доступным состояниям (с учетом Empty Room).
   - Межсубъектные сравнения для каждого физиологического состояния.
Чисто функциональный подход (без классов), строгий контроль памяти (RAM < 80 МБ).
"""

import gc
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = WORKSPACE_DIR / "04_processed_db"
RAW_DIR = WORKSPACE_DIR / "01_raw_data"
METADATA_DIR = WORKSPACE_DIR / "02_metadata"
ANNOTATIONS_DIR = METADATA_DIR / "annotations"
FIG_DIR = WORKSPACE_DIR / "week_2_QCML_graphs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Было:
# CHUNK_SIZE = 50000

# Стало:
CHUNK_SIZE = 50000
SFREQ_TYPE1 = 1000.0  # Биграммы (ВШЭ)
SFREQ_TYPE2 = 200.0   # Сонливость (Ушаков, после tSSS)

RAW_SOURCE_ROOTS = [
    RAW_DIR / "type1_bigrams",
    Path("/raw_data/MEG"),
    Path("~/git-reps/EEGs/EEG_final/eeg_pipeline/raw_data/MEG").expanduser(),
    WORKSPACE_DIR.parent / "raw_data" / "MEG",
]

TYPE1_SOURCE_MAP = {
    "sub00": ("20.08.24/2/240820_gromov/main.fif", "20.08.24/1/717446_untitled_2024-08-20_10h20.54.855.csv"),
    "sub01": ("20.08.24/2/240820_isaeva/main.fif", "20.08.24/1/F_6_untitled_2024-08-20_14h41.42.539.csv"),
    "sub02": ("20.08.24/2/240820_tomaschuk/part_1.fif", "20.08.24/1/460977_untitled_2024-08-20_12h31.49.749.csv"),
    "sub03": ("20.08.24/2/240820_tomaschuk/part_2.fif", "20.08.24/1/F_5_1_untitled_2024-08-20_13h27.53.116.csv"),
    "sub04": ("03.09.24/2/kirillova_agatha/main.fif", "03.09.24/1/F_7_fixed_final_untitled_2024-09-03_16h12.39.775.csv"),
    "sub05": ("03.09.24/2/shalamkova_alisa/main.fif", "03.09.24/1/F_8_untitled_2024-09-03_17h31.17.608.csv"),
    "sub06": ("03.09.24/2/borevskiy_andrey/main.fif", "03.09.24/1/F_9_untitled_2024-09-03_18h13.25.354.csv"),
    "sub07": ("02.10.24/2/nikolaev_evgeniy/main.fif", "02.10.24/1/Zhenya_final_untitled_2024-10-02_09h51.52.759.csv"),
    "sub08": ("02.10.24/2/spiridonov_dmitry/main.fif", "02.10.24/1/Dima_Dima_untitled_2024-10-02_10h39.35.813.csv"),
    "sub09": ("02.10.24/2/ostapets_valeria/main.fif", "02.10.24/1/Valeria_untitled_2024-10-02_11h25.29.235.csv"),
    "sub10": ("15.10.24/2/chistova_alena/main.fif", "15.10.24/1/alena_final_untitled_2024-10-15_13h00.00.308.csv"),
    "sub11": ("last_exps(don't use,please)/12.11.24/nedbay_pavel/241112/words.fif", None),
    "sub12": ("last_exps(don't use,please)/29.10.24/holinov_roman/words.fif", None),
    "sub13": ("last_exps(don't use,please)/29.10.24/saldan_anuar/words.fif", None),
    "sub14": ("last_exps(don't use,please)/29.10.24/sinitsin_ivan/words.fif", None),
}

EMPTY_ROOM_DATE_MAP = {
    "er_200824": ["sub00", "sub01", "sub02", "sub03"],
    "er_030924": ["sub04", "sub05", "sub06"],
    "er_021024": ["sub07", "sub08", "sub09"],
    "er_151024": ["sub10"],
    "er_291024": ["sub12", "sub13", "sub14"],
    "er_121124": ["sub11"]
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

PALETTE_TYPE2 = {
    "Верно": "#2ca02c",
    "Ошибка": "#d62728",
    "Сон": "#1f77b4"
}

PALETTE_METHODS = {
    "Algorithm 1 (Ratio Gap)": "#4a148c",
    "RMT (Marchenko-Pastur)": "#b71c1c",
    "Threshold 0.5": "#1b5e20"
}

# =============================================================================
# 1. МАТЕМАТИЧЕСКИЕ МЕТОДЫ CANDELORI ET AL. (2025)
# =============================================================================

# Кэш численной MP-медианы: ключ — аспект β (γ), значение — медиана s/σ.
_MP_MEDIAN_S_CACHE: Dict[float, float] = {}


def _mp_median_singular_numeric(beta: float, n_grid: int = 8192) -> float:
    """
    Численная медиана сингулярных чисел MP-распределения (в единицах σ).
    Плотность в шкале s = √λ (λ — собственные значения MP с аспектом β):

        ρ(s) = √((s_+² − s²)(s² − s_-²)) / (π β s),
        s_- = 1 − √β,   s_+ = 1 + √β,   β ∈ (0, 1].

    Значение кэшируется по β с точностью 5 знаков.
    """
    beta = float(np.clip(beta, 1e-3, 1.0))
    key = round(beta, 5)
    if key in _MP_MEDIAN_S_CACHE:
        return _MP_MEDIAN_S_CACHE[key]

    s_min = 1.0 - np.sqrt(beta)
    s_max = 1.0 + np.sqrt(beta)
    s = np.linspace(s_min + 1e-12, s_max - 1e-12, n_grid)

    num = np.maximum((s_max**2 - s**2) * (s**2 - s_min**2), 0.0)
    pdf = np.sqrt(num) / (np.pi * beta * s)

    cdf = np.cumsum(pdf)
    if cdf[-1] <= 0.0:
        # Вырожденный случай: β → 0, распределение стягивается к s = 1
        _MP_MEDIAN_S_CACHE[key] = 1.0
        return 1.0
    cdf /= cdf[-1]

    idx = int(np.searchsorted(cdf, 0.5))
    med = float(s[min(idx, len(s) - 1)])
    _MP_MEDIAN_S_CACHE[key] = med
    return med


def _gd_optimal_threshold(beta: float) -> float:
    """
    Точная оптимальная константа Gavish & Donoho (2014, Eq. 11)
    для hard threshold сингулярных чисел при аспекте β ∈ (0, 1]:

        τ(β) / σ = √[ 2(β+1) + 8β / ((β+1) + √(β² + 14β + 1)) ] / √β.

    При β → 1: τ(1)/σ = 4/√3 ≈ 2.309 — именно это значение
    фигурирует в статье Candelori et al. (2025).
    """
    b = float(np.clip(beta, 1e-3, 1.0))
    inner = 2.0 * (b + 1.0) + 8.0 * b / ((b + 1.0) + np.sqrt(b * b + 14.0 * b + 1.0))
    return float(np.sqrt(inner) / np.sqrt(b))


def compute_paper_methods(
    evals: np.ndarray,
    D_features: int = 306,
) -> Dict[str, np.ndarray]:
    """
    evals : (T, K) активный спектр квантовой метрики g(x),
            отсортированный по убыванию: λ_0 ≥ λ_1 ≥ … ≥ λ_{K-1} ≥ 0.
    D_features : размерность полного пространства g(x) — число сенсоров МЭГ
                 (используется только для оценки аспекта β в RMT).

    Возвращает три оценки внутренней размерности d:

      * "Algorithm 1 (Ratio Gap)" — строго по Candelori et al. (2025):
            γ = argmax_k λ_{k-1}/λ_k,  d = γ  (в порядке убывания).
        При d = 0 (все моды на уровне шума) возвращается 0.

      * "RMT (Marchenko-Pastur)" — Gavish & Donoho (2014):
            s = √λ;  σ оценивается по медиане шумового плато через
            численную MP-медиану;  порог τ = c(β)·σ.
        При β → 1 воспроизводит τ = (4/√3)·σ из статьи.

      * "Threshold 0.5" — baseline-эвристика (в статье отсутствует),
            выделяет моды, близкие к идеальному значению 1.
    """
    evals = np.maximum(evals, 0.0).astype(np.float32)
    T, K = evals.shape

    # -------------------------------------------------------------------------
    # 1. Algorithm 1 (Ratio Gap), Candelori et al. (2025)
    # -------------------------------------------------------------------------
    eps_float = np.finfo(np.float32).eps  # ≈ 1.19e-7
    ratios = evals[:, :-1] / (evals[:, 1:] + eps_float)
    d_ratio = (np.argmax(ratios, axis=1) + 1).astype(np.float32)

    # Защита от ложного d=0: если максимальная мода не превышает машинного
    # уровня, многообразие отсутствует (Empty Room).
    zero_tol = 10.0 * eps_float
    d_ratio = np.where(evals[:, 0] > zero_tol, d_ratio, 0.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. RMT: Marchenko-Pastur bulk + Gavish-Donoho hard threshold
    # -------------------------------------------------------------------------
    s = np.sqrt(evals)                                  # сингулярные числа g(x)

    bulk_start = max(1, K // 2)
    if K - bulk_start < 2:
        # Слишком короткий активный спектр — RMT не определён,
        # возвращаем консервативную оценку.
        d_rmt = np.zeros(T, dtype=np.float32)
    else:
        s_bulk = s[:, bulk_start:]                      # шумовое плато

        beta = float(min(1.0, K / float(D_features)))   # эффективный аспект MP
        mu_s = _mp_median_singular_numeric(beta)        # медиана s/σ при данном β
        c_gd = _gd_optimal_threshold(beta)              # константа GD для hard threshold

        sigma_est = np.median(s_bulk, axis=1) / mu_s    # оценка σ шума
        tau_sing = c_gd * sigma_est                     # порог по s

        d_rmt = np.sum(s > tau_sing[:, None], axis=1).astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Threshold 0.5 (baseline; в статье отсутствует как метод)
    # -------------------------------------------------------------------------
    # Идеализированный спектр: касательные моды ≈ 1, нормальные ≈ 0.
    d_05 = np.sum(evals > 0.5, axis=1).astype(np.float32)

    return {
        "Algorithm 1 (Ratio Gap)": d_ratio,
        "RMT (Marchenko-Pastur)": d_rmt,
        "Threshold 0.5": d_05,
    }

# =============================================================================
# 2. ПАРСИНГ МЕТАДАННЫХ
# =============================================================================
def parse_metadata(file_path: Path) -> Dict[str, Any]:
    name = file_path.name
    parent = file_path.parent.name

    n_match = re.search(r"_N(\d+)", name)
    n_val = int(n_match.group(1)) if n_match else 24

    if "type2" in parent or "type2" in name:
        condition = "drowsiness"
        sub_match = re.search(r"type2_([a-zA-Z]+_[a-zA-Z]+_\d+)", name)
        sub_id = sub_match.group(1) if sub_match else name.split("_")[1]
    elif "empty_room" in parent or "er_" in name:
        condition = "empty_room"
        sub_match = re.search(r"(er_\d+)", name)
        sub_id = sub_match.group(1) if sub_match else "empty_room"
    elif "resting" in parent or "resting" in name:
        condition = "resting_state"
        sub_match = re.search(r"(sub\d+|pilot\d+)", name)
        sub_id = sub_match.group(1) if sub_match else name.split("_")[1]
    else:
        condition = "task_bigrams"
        sub_match = re.search(r"(sub\d+|pilot\d+)", name)
        sub_id = sub_match.group(1) if sub_match else name.split("_")[1]

    return {
        "file_path": file_path,
        "condition": condition,
        "subject": sub_id,
        "N": n_val
    }


def get_subject_target_dir(subject: str, condition: str, n_val: int) -> Path:
    target_dir = FIG_DIR / subject / condition / f"N{n_val}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


# =============================================================================
# 3. ПОИСК ТРИГГЕРОВ И ВЫРЕЗКА ЭПОХ (DDOF = 1)
# =============================================================================
_TRIGGER_CACHE_T1: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

def extract_stim_channel_from_pysz(pysz_path: Path) -> Tuple[np.ndarray, int, float]:
    with open(pysz_path, "rb") as f:
        payload = pickle.load(f)

    info = payload["info"]
    ch_names = info["ch_names"]

    stim_idx = None
    for name in ["STI101", "STI 014", "STI001"]:
        if name in ch_names:
            stim_idx = ch_names.index(name)
            break

    if stim_idx is None:
        for idx, ch in enumerate(info["chs"]):
            if ch["kind"] == mne.io.constants.FIFF.FIFFV_STIM_CH:
                stim_idx = idx
                break

    if stim_idx is None or stim_idx not in payload["chunks"]:
        raise ValueError(f"STIM канал не найден в архиве {pysz_path.name}")

    from pysz import sz
    c_bytes = payload["chunks"][stim_idx]
    decomp, _ = sz.decompress(c_bytes, np.float32, (payload["n_times"],))
    return decomp, payload.get("first_samp", 0), float(info["sfreq"])


def get_type1_triggers(sub_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if sub_id in _TRIGGER_CACHE_T1:
        return _TRIGGER_CACHE_T1[sub_id]

    cross_idx = None

    # 1. Поиск локального .fif
    fif_candidates = list((RAW_DIR / "type1_bigrams").glob(f"*{sub_id}*.fif"))
    fif_candidates = [f for f in fif_candidates if not f.name.endswith("-1.fif") and not f.name.endswith("-2.fif")]
    if fif_candidates:
        try:
            raw = mne.io.read_raw_fif(fif_candidates[0], preload=False, verbose=False)
            events = mne.find_events(raw, stim_channel="STI101", min_duration=0.002, verbose=False)
            cross_events = events[events[:, 2] == 8]
            cross_idx = (cross_events[:, 0] - raw.first_samp).astype(np.float64)
        except Exception:
            cross_idx = None

    # 2. Поиск в каталогах исходных данных
    if cross_idx is None and sub_id in TYPE1_SOURCE_MAP:
        rel_fif, _ = TYPE1_SOURCE_MAP[sub_id]
        for root in RAW_SOURCE_ROOTS:
            cand = root / rel_fif
            if cand.exists():
                try:
                    raw = mne.io.read_raw_fif(cand, preload=False, verbose=False)
                    events = mne.find_events(raw, stim_channel="STI101", min_duration=0.002, verbose=False)
                    cross_events = events[events[:, 2] == 8]
                    cross_idx = (cross_events[:, 0] - raw.first_samp).astype(np.float64)
                    break
                except Exception:
                    continue

    # 3. Декомпрессия STI101 из .fif.pysz
    if cross_idx is None:
        pysz_candidates = list((RAW_DIR / "type1_bigrams").glob(f"*{sub_id}*.fif.pysz"))
        if pysz_candidates:
            try:
                stim_data, first_samp, sfreq = extract_stim_channel_from_pysz(pysz_candidates[0])
                raw_stim = mne.io.RawArray(
                    stim_data[None, :],
                    mne.create_info(["STI101"], sfreq=sfreq, ch_types=["stim"]),
                    first_samp=first_samp,
                    verbose=False
                )
                events = mne.find_events(raw_stim, stim_channel="STI101", min_duration=0.002, verbose=False)
                cross_events = events[events[:, 2] == 8]
                cross_idx = (cross_events[:, 0] - first_samp).astype(np.float64)
            except Exception:
                cross_idx = None

    if cross_idx is None or len(cross_idx) == 0:
        return None

    bigram_idx = cross_idx + 500.0

    csv_candidates = list(ANNOTATIONS_DIR.glob(f"*{sub_id}*.csv"))
    csv_p = csv_candidates[0] if csv_candidates else None
    if csv_p is None and sub_id in TYPE1_SOURCE_MAP:
        _, rel_csv = TYPE1_SOURCE_MAP[sub_id]
        if rel_csv:
            for root in RAW_SOURCE_ROOTS:
                if (root / rel_csv).exists():
                    csv_p = root / rel_csv
                    break

    resp_idx = np.array([], dtype=np.float64)
    if csv_p and csv_p.exists():
        try:
            df_csv = pd.read_csv(csv_p)
            mask = df_csv["text.started"].notna() if "text.started" in df_csv.columns else df_csv["bigrams"].notna()
            trials = df_csv[mask].copy().reset_index(drop=True)

            if "sub10" in sub_id and len(trials) == 808 and len(cross_idx) == 805:
                trials = trials.drop([780, 781, 782]).reset_index(drop=True)

            rt_col = next((c for c in ["key_resp.rt", "RT", "rt"] if c in trials.columns), None)
            if rt_col:
                rts = trials[rt_col].values.astype(np.float64)
                valid_rt_mask = ~np.isnan(rts)
                min_l = min(len(cross_idx), len(rts))
                valid_mask = valid_rt_mask[:min_l]
                resp_idx = (bigram_idx[:min_l] + rts[:min_l] * 1000.0)[valid_mask]
        except Exception:
            pass

    _TRIGGER_CACHE_T1[sub_id] = (cross_idx, bigram_idx, resp_idx)
    return _TRIGGER_CACHE_T1[sub_id]


def align_epoch_series(series: np.ndarray, triggers: np.ndarray, pre: int = 2500, post: int = 2500):
    """
    Вырезка триальных окон и несмещенная оценка SEM (ddof=1).
    """
    epochs = []
    T = len(series)
    for tr in triggers:
        if np.isnan(tr):
            continue
        idx = int(round(tr))
        if idx - pre >= 0 and idx + post + 1 <= T:
            epochs.append(series[idx - pre : idx + post + 1])
    if not epochs:
        return np.zeros(pre + post + 1, dtype=np.float32), np.zeros(pre + post + 1, dtype=np.float32), 0

    arr = np.array(epochs, dtype=np.float32)
    mean_prof = np.mean(arr, axis=0)
    sem_prof = np.std(arr, axis=0, ddof=1) / np.sqrt(len(epochs))
    return mean_prof, sem_prof, len(epochs)


_TYPE2_ANN_DF: Optional[pd.DataFrame] = None

def get_type2_subject_annotation(file_stem: str) -> Optional[pd.DataFrame]:
    global _TYPE2_ANN_DF
    ann_file = ANNOTATIONS_DIR / "Разметка интерполированная (все).csv"
    if not ann_file.exists():
        return None

    if _TYPE2_ANN_DF is None:
        try:
            _TYPE2_ANN_DF = pd.read_csv(ann_file)
        except Exception:
            return None

    unique_names = _TYPE2_ANN_DF["name"].dropna().unique()
    matched_name = next((n for n in unique_names if str(n).strip() in file_stem), None)
    if not matched_name:
        return None

    match_rows = _TYPE2_ANN_DF[_TYPE2_ANN_DF["name"] == matched_name].copy()
    return match_rows.sort_values("time").reset_index(drop=True)


# =============================================================================
# 4. ФУНКЦИИ ГРАФИЧЕСКОГО ОТОБРАЖЕНИЯ
# =============================================================================
def plot_bigrams_single_method(
    target_dir: Path,
    subject: str,
    n_val: int,
    method_name: str,
    cross_data: Tuple[np.ndarray, np.ndarray, int],
    bigram_data: Tuple[np.ndarray, np.ndarray, int],
    resp_data: Tuple[np.ndarray, np.ndarray, int]
) -> None:
    """
    3 отдельных файла по методам с единой шкалой Y по валидным событиям.
    """
    t_axis = np.linspace(-2.5, 2.5, 5001)
    color = PALETTE_METHODS.get(method_name, "navy")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)

    ev_configs = [
        ("Крест фиксации (t = 0)", cross_data, [(0.5, "Биграмма (+0.5с)", "crimson", ":")]),
        ("Предъявление биграммы (t = 0)", bigram_data, [(-0.5, "Крест (-0.5с)", "navy", ":")]),
        ("Моторный ответ (t = 0)", resp_data, [])
    ]

    valid_y_mins = []
    valid_y_maxs = []
    for _, (mean_d, sem_d, n_ev), _ in ev_configs:
        if n_ev > 0:
            valid_y_mins.append(np.min(mean_d - sem_d))
            valid_y_maxs.append(np.max(mean_d + sem_d))

    if valid_y_mins and valid_y_maxs:
        y_min_g = min(valid_y_mins)
        y_max_g = max(valid_y_maxs)
        pad = max(0.15, (y_max_g - y_min_g) * 0.15)
        unified_ylim = (y_min_g - pad, y_max_g + pad)
    else:
        unified_ylim = (-0.5, 10.0)

    for idx, (title, (mean_d, sem_d, n_ev), vlines) in enumerate(ev_configs):
        ax = axes[idx]
        if n_ev > 0:
            ax.plot(t_axis, mean_d, color=color, linewidth=2.0, label=f"{method_name} (N={n_ev} триалов)")
            ax.fill_between(t_axis, mean_d - sem_d, mean_d + sem_d, color=color, alpha=0.25, label="±1 SEM")
        else:
            ax.text(0.5, 0.5, "Нет валидных триалов", ha="center", va="center", transform=ax.transAxes)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.4)
        for x_pos, v_lbl, lc, st in vlines:
            ax.axvline(x_pos, color=lc, linestyle=st, linewidth=1.2, label=v_lbl)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Размерность (d)")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(unified_ylim)

        ax.grid(True, linestyle="--", alpha=0.5)
        if n_ev > 0:
            ax.legend(loc="upper right", framealpha=0.9)

    axes[2].set_xlabel("Время относительно события: dt (секунды)")
    fig.suptitle(f"Когнитивная динамика: {method_name} ({subject}, N={n_val})", fontweight="bold")

    slug = re.sub(r'[^a-zA-Z0-9]', '_', method_name).strip('_').lower()
    out_file = target_dir / f"{slug}.png"
    plt.savefig(out_file, dpi=160)
    plt.close()
    print(f"      [+] График сохранен: {out_file.relative_to(WORKSPACE_DIR)}")


def plot_type2_pointwise_plots(target_dir: Path, subject: str, n_val: int, df_points: pd.DataFrame) -> None:
    """Поточечный Violin Plot и дискретные гистограммы режимов (включая d=0)."""
    if df_points.empty:
        return

    methods = ["Algorithm 1 (Ratio Gap)", "RMT (Marchenko-Pastur)", "Threshold 0.5"]
    max_k = 2 * (n_val - 1)
    discrete_bins = np.arange(-0.5, max_k + 1.5, 1.0)

    # 1. Violin Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for i, m in enumerate(methods):
        ax = axes[i]
        sub = df_points[df_points["method"] == m]
        if sub.empty:
            continue
        sns.violinplot(
            data=sub,
            x="state",
            y="dimension",
            hue="state",
            legend=False,
            order=["Верно", "Ошибка", "Сон"],
            palette=PALETTE_TYPE2,
            inner="quartile",
            cut=0,
            ax=ax
        )
        ax.set_title(f"Метод: {m}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Физиологический режим (Ушаков)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.set_ylabel("Внутренняя размерность d (поточечно)")

    fig.suptitle(f"Поточечное распределение размерности (Сон): {subject} (N={n_val})", fontweight="bold")
    v_file = target_dir / "violin_states_pointwise.png"
    plt.savefig(v_file, dpi=160)
    plt.close()

    # 2. Дискретная гистограмма
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for i, m in enumerate(methods):
        ax = axes[i]
        sub = df_points[df_points["method"] == m]
        for st in ["Верно", "Ошибка", "Сон"]:
            vals = sub[sub["state"] == st]["dimension"]
            if len(vals) > 0:
                ax.hist(vals, bins=discrete_bins, density=True, alpha=0.35, label=st,
                        color=PALETTE_TYPE2[st], histtype="stepfilled", edgecolor="black")
        ax.set_title(f"Метод: {m}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Размерность (d)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.set_ylabel("Плотность вероятности P(d)")
        ax.legend()

    fig.suptitle(f"Целочисленные гистограммы режимов (Сон): {subject} (N={n_val})", fontweight="bold")
    h_file = target_dir / "hist_states_pointwise.png"
    plt.savefig(h_file, dpi=160)
    plt.close()
    print(f"      [+] Скрипки и гистограммы сохранены: {target_dir.relative_to(WORKSPACE_DIR)}")


def plot_stationary_plots(target_dir: Path, dataset_name: str, subject: str, n_val: int, d_1d: Dict[str, np.ndarray]) -> None:
    """Гистограммы и violin plot для Resting State и Empty Room (RAM < 5 МБ)."""
    max_k = 2 * (n_val - 1)
    discrete_bins = np.arange(-0.5, max_k + 1.5, 1.0)

    methods = ["Algorithm 1 (Ratio Gap)", "RMT (Marchenko-Pastur)", "Threshold 0.5"]

    # 1. Быстрое компактное прореживание для KDE в violinplot (<= 15k точек на метод)
    total_pts = len(d_1d["Threshold 0.5"])
    stride_viz = max(1, total_pts // 15000)
    sub_sample = {m: d_1d[m][::stride_viz] for m in methods}
    n_sub = len(sub_sample[methods[0]])

    df_kde = pd.DataFrame({
        "method": np.repeat(methods, n_sub),
        "dimension": np.concatenate([sub_sample[m] for m in methods])
    })

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    sns.violinplot(
        data=df_kde,
        x="method",
        y="dimension",
        hue="method",
        legend=False,
        palette=PALETTE_METHODS,
        inner="quartile",
        cut=0,
        ax=ax
    )
    ax.set_title(f"Распределение ({dataset_name}): {subject} (N={n_val})", fontweight="bold")
    ax.set_xlabel("Метод QCML")
    ax.set_ylabel("Внутренняя размерность (d)")
    ax.grid(True, linestyle="--", alpha=0.5)
    out_f = target_dir / "violin_methods.png"
    plt.savefig(out_f, dpi=160)
    plt.close()
    del df_kde, sub_sample

    # 2. Дискретная гистограмма напрямую из NumPy массивов (100% данных)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for m in methods:
        vals = d_1d[m]
        if len(vals) > 0:
            ax.hist(vals, bins=discrete_bins, density=True, alpha=0.3, label=m,
                    color=PALETTE_METHODS[m], histtype="stepfilled", edgecolor="black")
    ax.set_title(f"Гистограммы методов ({dataset_name}): {subject} (N={n_val})", fontweight="bold")
    ax.set_xlabel("Размерность (d)")
    ax.set_ylabel("Плотность вероятности P(d)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    out_h = target_dir / "histogram_methods.png"
    plt.savefig(out_h, dpi=160)
    plt.close()
    print(f"      [+] Скрипки и гистограммы сохранены: {target_dir.relative_to(WORKSPACE_DIR)}")


def plot_intra_subject_condition_comparison(subject: str, n_val: int, sub_samples_dict: Dict[str, List[pd.DataFrame]]) -> None:
    """Гистограммы сравнения состояний внутри одного субъекта."""
    conditions = list(sub_samples_dict.keys())
    if len(conditions) < 2:
        return

    out_file = FIG_DIR / subject / f"intra_subject_states_hist_N{n_val}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    methods = ["Algorithm 1 (Ratio Gap)", "RMT (Marchenko-Pastur)", "Threshold 0.5"]
    max_k = 2 * (n_val - 1)
    discrete_bins = np.arange(-0.5, max_k + 1.5, 1.0)

    # Однократная конкатенация до цикла
    cond_dfs = {c: pd.concat(sub_samples_dict[c], ignore_index=True) for c in conditions}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)

    cond_colors = {
        "task_bigrams": "#2ca02c",
        "resting_state": "#ff7f0e",
        "empty_room": "#1f77b4",
        "drowsiness": "#9c27b0"
    }

    for i, m in enumerate(methods):
        ax = axes[i]
        for c in conditions:
            vals = cond_dfs[c][cond_dfs[c]["method"] == m]["dimension"]
            if len(vals) > 0:
                ax.hist(vals, bins=discrete_bins, density=True, alpha=0.35, label=c,
                        color=cond_colors.get(c, "gray"), histtype="stepfilled", edgecolor="black")

        ax.set_title(f"Метод: {m}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Размерность (d)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.set_ylabel("Плотность вероятности P(d)")
        ax.legend()

    fig.suptitle(f"Сравнение состояний внутри субъекта: {subject} (N={n_val})", fontweight="bold")
    plt.savefig(out_file, dpi=160)
    plt.close()
    del cond_dfs
    print(f"  [+] Внутрисубъектное сравнение состояний ({subject}) сохранено: {out_file.relative_to(WORKSPACE_DIR)}")


def plot_inter_subject_comparisons(global_condition_records: Dict[str, Dict[int, List[pd.DataFrame]]]) -> None:
    """Межсубъектные сравнения для каждого состояния."""
    out_inter_dir = FIG_DIR / "inter_subject_comparisons"
    out_inter_dir.mkdir(parents=True, exist_ok=True)

    methods = ["Algorithm 1 (Ratio Gap)", "RMT (Marchenko-Pastur)", "Threshold 0.5"]

    for cond, n_dict in global_condition_records.items():
        for n_val, df_list in n_dict.items():
            if not df_list:
                continue
            df_cond = pd.concat(df_list, ignore_index=True)
            if df_cond["subject"].nunique() < 2:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True, constrained_layout=True)

            for i, m in enumerate(methods):
                ax = axes[i]
                sub = df_cond[df_cond["method"] == m]
                if sub.empty:
                    continue

                sns.boxplot(
                    data=sub,
                    x="subject",
                    y="dimension",
                    hue="subject",
                    legend=False,
                    ax=ax,
                    width=0.5,
                    showmeans=True,
                    palette="tab10"
                )
                ax.set_title(f"Метод: {m}", fontweight="bold")
                ax.set_xlabel("Субъект")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.tick_params(axis='x', rotation=30)
                if i == 0:
                    ax.set_ylabel("Размерность (d)")

            fig.suptitle(f"Межсубъектное сравнение ({cond}, N={n_val})", fontweight="bold")
            out_file = out_inter_dir / f"{cond}_comparison_N{n_val}.png"
            plt.savefig(out_file, dpi=160)
            plt.close()
            print(f"[+] Межсубъектное сравнение ({cond}) сохранено: {out_file.relative_to(WORKSPACE_DIR)}")


# =============================================================================
# 5. ГЛАВНЫЙ ИСПОЛНИТЕЛЬНЫЙ ЦИКЛ ОБРАБОТКИ (100% ДАННЫХ)
# =============================================================================
def process_all_files():
    parquet_files = sorted(DB_DIR.glob("**/*.parquet"))
    parquet_files = [f for f in parquet_files if "type3" not in str(f)]
    print(f"[*] Найдено {len(parquet_files)} файлов для комплексного анализа.")

    methods_keys = ["Algorithm 1 (Ratio Gap)", "RMT (Marchenko-Pastur)", "Threshold 0.5"]

    intra_subject_data: Dict[str, Dict[int, Dict[str, List[pd.DataFrame]]]] = {}
    inter_subject_data: Dict[str, Dict[int, List[pd.DataFrame]]] = {}

    for p_file in parquet_files:
        meta = parse_metadata(p_file)
        target_dir = get_subject_target_dir(meta["subject"], meta["condition"], meta["N"])

        try:
            pq_file = pq.ParquetFile(p_file)
            total_rows = pq_file.metadata.num_rows
            arrow_cols = set(pq_file.schema_arrow.names)
        except Exception as e:
            print(f"  [-] Ошибка открытия {p_file.name}: {e}")
            continue

        if "eigenvalues" not in arrow_cols:
            print(f"  [-] Пропуск {p_file.name}: отсутствует колонка 'eigenvalues'")
            continue

        print(f"\n▶ [{meta['condition'].upper()}] {meta['subject']} (N={meta['N']}) | Точек: {total_rows}")

        # Непрерывный массив d(t) на 100% отсчетов
        d_1d = {m: np.empty(total_rows, dtype=np.float32) for m in methods_keys}
        offset = 0

        for batch in pq_file.iter_batches(batch_size=CHUNK_SIZE, columns=["eigenvalues"]):
            col_arrow = batch.column(0)
            n_b = len(batch)
            evals_chunk = np.vstack(col_arrow.to_numpy(zero_copy_only=False))
            del col_arrow

            dims = compute_paper_methods(evals_chunk)
            for m in methods_keys:
                d_1d[m][offset : offset + n_b] = dims[m]

            offset += n_b
            del dims, evals_chunk

        # Сохранение компактной выборки (до 50k точек) для внутри- и межсубъектных сравнений
        stride_dist = max(1, total_rows // 50000)
        df_sampled = pd.DataFrame({
            "subject": meta["subject"],
            "condition": meta["condition"],
            "N": meta["N"],
            "method": np.repeat(methods_keys, len(d_1d["Threshold 0.5"][::stride_dist])),
            "dimension": np.concatenate([d_1d[m][::stride_dist] for m in methods_keys])
        })

        sub_n = meta["subject"]
        c_n = meta["condition"]
        n_v = meta["N"]

        # Связывание пустой комнаты с соответствующим субъектом Type 1
        target_subs_intra = [sub_n]
        if c_n == "empty_room" and sub_n in EMPTY_ROOM_DATE_MAP:
            target_subs_intra = EMPTY_ROOM_DATE_MAP[sub_n]

        for s_target in target_subs_intra:
            if s_target not in intra_subject_data:
                intra_subject_data[s_target] = {}
            if n_v not in intra_subject_data[s_target]:
                intra_subject_data[s_target][n_v] = {}
            if c_n not in intra_subject_data[s_target][n_v]:
                intra_subject_data[s_target][n_v][c_n] = []
            intra_subject_data[s_target][n_v][c_n].append(df_sampled)

        if c_n not in inter_subject_data:
            inter_subject_data[c_n] = {}
        if n_v not in inter_subject_data[c_n]:
            inter_subject_data[c_n][n_v] = []
        inter_subject_data[c_n][n_v].append(df_sampled)

        # -------------------------------------------------------------
        # 1. ОБРАБОТКА ТИПА 1 (TASK BIGRAMS)
        # -------------------------------------------------------------
        if meta["condition"] == "task_bigrams":
            triggers = get_type1_triggers(meta["subject"])
            if triggers is not None:
                cross_idx, bigram_idx, resp_idx = triggers
                for m in methods_keys:
                    c_mean, c_sem, n_c = align_epoch_series(d_1d[m], cross_idx, 2500, 2500)
                    b_mean, b_sem, n_b = align_epoch_series(d_1d[m], bigram_idx, 2500, 2500)
                    r_mean, r_sem, n_r = align_epoch_series(d_1d[m], resp_idx, 2500, 2500)

                    if n_c > 0:
                        plot_bigrams_single_method(
                            target_dir=target_dir,
                            subject=meta["subject"],
                            n_val=meta["N"],
                            method_name=m,
                            cross_data=(c_mean, c_sem, n_c),
                            bigram_data=(b_mean, b_sem, n_b),
                            resp_data=(r_mean, r_sem, n_r)
                        )
            else:
                print(f"    [-] Триггеры отсутствуют для {meta['subject']}")

        # -------------------------------------------------------------
        # 2. ОБРАБОТКА ТИПА 2 (DROWSINESS)
        # -------------------------------------------------------------
        elif meta["condition"] == "drowsiness":
            ann_df = get_type2_subject_annotation(p_file.name)
            if ann_df is not None:
                ann_lookup = dict(zip(ann_df["time"].astype(int), ann_df["beh_type"]))

                # Корректное соотнесение секунд (интервалы [t, t+1))
                indices_t2 = np.arange(total_rows)
                times_sec = (indices_t2 // int(SFREQ_TYPE2)).astype(int)

                states = np.array([ann_lookup.get(t, None) for t in times_sec])
                valid_mask = np.isin(states, ["Верно", "Ошибка", "Сон"])

                if np.any(valid_mask):
                    v_states = states[valid_mask]
                    v_idx = indices_t2[valid_mask]
                    df_t2_points = pd.concat([
                        pd.DataFrame({"method": m, "state": v_states, "dimension": d_1d[m][v_idx]})
                        for m in methods_keys
                    ], ignore_index=True)

                    plot_type2_pointwise_plots(target_dir, meta["subject"], meta["N"], df_t2_points)
                    del df_t2_points

        # -------------------------------------------------------------
        # 3. ОБРАБОТКА RESTING STATE И EMPTY ROOM (RAM-SAFE)
        # -------------------------------------------------------------
        elif meta["condition"] in ["resting_state", "empty_room"]:
            plot_stationary_plots(target_dir, meta["condition"], meta["subject"], meta["N"], d_1d)

        del d_1d
        gc.collect()

    # -----------------------------------------------------------------
    # 6. ГИСТОГРАММЫ СРАВНЕНИЯ
    # -----------------------------------------------------------------
    print("\n[*] Построение внутрисубъектных гистограмм сравнения состояний...")
    for sub_n, n_map in intra_subject_data.items():
        for n_v, cond_map in n_map.items():
            plot_intra_subject_condition_comparison(sub_n, n_v, cond_map)

    print("\n[*] Построение межсубъектных гистограмм по каждому состоянию...")
    plot_inter_subject_comparisons(inter_subject_data)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
def main():
    print("=== Комплексная визуализация QCML (100% данных, d=0 разрешено, RAM < 80 МБ) ===")
    print(f"Базовая директория сохранения: {FIG_DIR.resolve()}\n")
    process_all_files()
    print("\n[+] Все задачи успешно завершены!")


if __name__ == "__main__":
    main()
