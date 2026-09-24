#!/usr/bin/env python3
"""
stat_test_type2_full.py
Полный статистический контраст 4x3:
Ряд 1: Контурные плотности P(d) (Верно, Ошибка, Сон)
Ряд 2: Кумулятивные распределения (ECDF)
Ряд 3: Контраст ΔP = P(d | Ошибка) - P(d | Верно)
Ряд 4: Контраст ΔP = P(d | Сон) - P(d | Верно)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import matplotlib.pyplot as plt

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = WORKSPACE_DIR / "04_processed_db" / "type2_features"
ANNOTATIONS_FILE = WORKSPACE_DIR / "02_metadata" / "annotations" / "Разметка интерполированная (все).csv"
OUT_DIR = WORKSPACE_DIR / "week_2_QCML_graphs" / "figures" / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_STEM = "type2_chastkov_konstantin_220310_raw_metric_spectrum_N24"
SFREQ = 200.0

PALETTE = {
    "Верно": "#2ca02c",   # Зеленый
    "Ошибка": "#d62728",  # Красный
    "Сон": "#1f77b4"      # Синий
}

def main():
    pq_path = DB_DIR / f"{TARGET_STEM}.parquet"
    if not pq_path.exists():
        print(f"[!] Файл не найден: {pq_path}")
        return

    table = pq.read_table(pq_path)
    df_raw = table.to_pandas()
    evals = np.vstack(df_raw["eigenvalues"].values)
    T, K = evals.shape

    eps = np.finfo(np.float32).eps
    # 1. Ratio Gap
    ratios = evals[:, :-1] / (evals[:, 1:] + eps)
    d_ratio = (np.argmax(ratios, axis=1) + 1).astype(np.float32)
    d_ratio = np.where(evals[:, 0] > 10 * eps, d_ratio, 0.0)

    # 2. RMT (точное согласование с аспектным отношением бета)
    beta = float(min(1.0, K / 306.0))
    inner = 2.0 * (beta + 1.0) + 8.0 * beta / ((beta + 1.0) + np.sqrt(beta**2 + 14.0 * beta + 1.0))
    c_gd = float(np.sqrt(inner) / np.sqrt(beta))
    s = np.sqrt(np.maximum(evals, 0.0))
    s_bulk = s[:, K // 2:]
    sigma_est = np.median(s_bulk, axis=1) / 0.985
    tau = c_gd * sigma_est
    d_rmt = np.sum(s > tau[:, None], axis=1).astype(np.float32)

    # 3. Threshold 0.5
    d_05 = np.sum(evals > 0.5, axis=1).astype(np.float32)

    methods = {
        "Algorithm 1 (Ratio Gap)": d_ratio,
        "RMT (Marchenko-Pastur)": d_rmt,
        "Threshold 0.5": d_05
    }

    ann_df = pd.read_csv(ANNOTATIONS_FILE)
    subj_name = next(n for n in ann_df["name"].dropna().unique() if str(n).strip() in TARGET_STEM)
    sub_ann = ann_df[ann_df["name"] == subj_name]
    ann_map = dict(zip(sub_ann["time"].astype(int), sub_ann["beh_type"]))

    indices = np.arange(T)
    times_sec = (indices // int(SFREQ)).astype(int)
    states = np.array([ann_map.get(t, None) for t in times_sec])

    valid_mask = np.isin(states, ["Верно", "Ошибка", "Сон"])
    v_states = states[valid_mask]

    # --- Сетка 4 строки x 3 столбца ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 15), constrained_layout=True)

    for col_idx, (m_name, d_arr) in enumerate(methods.items()):
        v_d = d_arr[valid_mask]
        max_d = int(np.max(v_d)) + 2
        bins = np.arange(-0.5, max_d + 0.5, 1.0)
        centers = 0.5 * (bins[:-1] + bins[1:])

        d_v = v_d[v_states == "Верно"]
        d_e = v_d[v_states == "Ошибка"]
        d_s = v_d[v_states == "Сон"]

        counts_v, _ = np.histogram(d_v, bins=bins, density=True)
        counts_e, _ = np.histogram(d_e, bins=bins, density=True)
        counts_s, _ = np.histogram(d_s, bins=bins, density=True)

        # РЯД 1: Контурные плотности
        ax1 = axes[0, col_idx]
        for name, data, col in [("Верно", d_v, PALETTE["Верно"]),
                                ("Ошибка", d_e, PALETTE["Ошибка"]),
                                ("Сон", d_s, PALETTE["Сон"])]:
            cnts, _ = np.histogram(data, bins=bins, density=True)
            ax1.step(centers, cnts, where="mid", color=col, lw=2.2, label=f"{name} (mean={data.mean():.2f})")
            ax1.axvline(data.mean(), color=col, linestyle="--", alpha=0.7, lw=1.2)
        ax1.set_title(f"{m_name}\nКонтурные плотности P(d)", fontweight="bold")
        ax1.set_ylabel("Плотность P(d)")
        ax1.set_xlim(np.percentile(v_d, 0.1) - 1, np.percentile(v_d, 99.9) + 1)
        ax1.grid(True, ls="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=9)

        # РЯД 2: ECDF
        ax2 = axes[1, col_idx]
        for name, data, col in [("Верно", d_v, PALETTE["Верно"]),
                                ("Ошибка", d_e, PALETTE["Ошибка"]),
                                ("Сон", d_s, PALETTE["Сон"])]:
            x_sorted = np.sort(data)
            y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax2.plot(x_sorted, y_ecdf, color=col, lw=2.0, label=name)
        ax2.set_title("Кумулятивное распределение (ECDF)")
        ax2.set_ylabel("F(d) = P(X ≤ d)")
        ax2.set_xlim(ax1.get_xlim())
        ax2.grid(True, ls="--", alpha=0.5)
        ax2.legend(loc="lower right", fontsize=9)

        # РЯД 3: Контраст ΔP = P(Ошибка) - P(Верно)
        ax3 = axes[2, col_idx]
        delta_p_err = counts_e - counts_v
        col_err = np.where(delta_p_err >= 0, PALETTE["Ошибка"], PALETTE["Верно"])
        ax3.bar(centers, delta_p_err, width=0.8, color=col_err, edgecolor="black", alpha=0.8)
        ax3.axhline(0, color="black", lw=1.2)
        ax3.set_title("Контраст: P(d | Ошибка) − P(d | Верно)", fontweight="bold")
        ax3.set_ylabel("Δ Плотность (Ош − Вер)")
        ax3.set_xlim(ax1.get_xlim())
        ax3.grid(True, ls="--", alpha=0.5)

        # РЯД 4: Контраст ΔP = P(Сон) - P(Верно)
        ax4 = axes[3, col_idx]
        delta_p_slp = counts_s - counts_v
        col_slp = np.where(delta_p_slp >= 0, PALETTE["Сон"], PALETTE["Верно"])
        ax4.bar(centers, delta_p_slp, width=0.8, color=col_slp, edgecolor="black", alpha=0.8)
        ax4.axhline(0, color="black", lw=1.2)
        ax4.set_title("Контраст: P(d | Сон) − P(d | Верно)", fontweight="bold")
        ax4.set_xlabel("Внутренняя размерность (d)")
        ax4.set_ylabel("Δ Плотность (Сон − Вер)")
        ax4.set_xlim(ax1.get_xlim())
        ax4.grid(True, ls="--", alpha=0.5)

    fig.suptitle(f"Комплексный анализ когнитивных контрастов: {subj_name} (N=24)",
                 fontsize=15, fontweight="bold")
    out_file = OUT_DIR / f"full_contrast_4x3_{TARGET_STEM}.png"
    plt.savefig(out_file, dpi=160)
    plt.close()
    print(f"[+] Полная матрица контрастов 4x3 сохранена: {out_file.relative_to(WORKSPACE_DIR)}")

if __name__ == "__main__":
    main()
