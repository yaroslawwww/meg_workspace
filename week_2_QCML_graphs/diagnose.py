#!/usr/bin/env python3
"""
diagnose_type2_sync.py

Диагностика синхронизации Type2 (сонливость) с аннотациями.
Проверяет 6 гипотез:
  1. Реальная sfreq в .fif vs SFREQ_TYPE2
  2. Длительность parquet vs диапазон времени в аннотации
  3. Структура annotation CSV (колонки, значения, имена)
  4. Матчинг parquet-файла и имени субъекта в аннотации
  5. Распределение состояний по времени (пересечение)
  6. Распределение d, residual, trace, PR по состояниям
Плюс: сохранение sanity-plot d(t) с таймлайном переходов.

Запуск:
    python diagnose_type2_sync.py <subject_parquet_stem> [--fif <path>]

Пример:
    python diagnose_type2_sync.py type2_bainbridge_emily_220519_raw_metric_spectrum_N24
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("ERROR")

# --- Пути (настраиваются под вашу структуру) ---
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = WORKSPACE_DIR / "04_processed_db"
RAW_DIR = WORKSPACE_DIR / "01_raw_data" / "type2_drowsiness"
ANNOTATIONS_DIR = WORKSPACE_DIR / "02_metadata" / "annotations"
OUT_DIR = WORKSPACE_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_FILE = ANNOTATIONS_DIR / "Разметка интерполированная (все).csv"

# Значения состояний, которые мы ожидаем
EXPECTED_STATES = ["Верно", "Ошибка", "Сон"]
# Возможные синонимы (если в файле латиница или другой регистр)
STATE_ALIASES = {
    "Верно":   ["верно", "correct", "ok", "1"],
    "Ошибка":  ["ошибка", "error", "incorrect", "0"],
    "Сон":     ["сон", "sleep", "drowsy", "2"],
}

# SFREQ который использует visualize_qcml_multi_analysis.py
SFREQ_TYPE2_ASSUMED = 200.0


# =============================================================================
# 1. РЕАЛЬНАЯ SFREQ В .FIF
# =============================================================================
def check_sfreq(fif_path: Path) -> dict:
    print("\n" + "=" * 70)
    print("1. РЕАЛЬНАЯ SFREQ В .FIF")
    print("=" * 70)

    if not fif_path.exists():
        print(f"  [!] Файл не найден: {fif_path}")
        return {"ok": False}

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    n_times = raw.n_times
    duration = n_times / sfreq

    print(f"  Файл:        {fif_path.name}")
    print(f"  sfreq:       {sfreq} Hz")
    print(f"  n_times:     {n_times} сэмплов")
    print(f"  duration:    {duration:.2f} сек  ({duration/60:.1f} мин)")

    match = abs(sfreq - SFREQ_TYPE2_ASSUMED) < 1e-3
    print(f"\n  Ожидание в скрипте: SFREQ_TYPE2 = {SFREQ_TYPE2_ASSUMED} Hz")
    if match:
        print(f"  [OK] Совпадает.")
    else:
        ratio = sfreq / SFREQ_TYPE2_ASSUMED
        print(f"  [!!] НЕ СОВПАДАЕТ. Отношение = {ratio:.3f}")
        print(f"       Все times_sec будут смещены в {ratio:.1f} раз.")

    return {
        "ok": match,
        "sfreq": sfreq,
        "n_times": n_times,
        "duration": duration,
    }


# =============================================================================
# 2. ДЛИТЕЛЬНОСТЬ PARQUET VS АННОТАЦИЯ
# =============================================================================
def check_parquet_duration(parquet_path: Path, sfreq_real: float) -> dict:
    print("\n" + "=" * 70)
    print("2. ДЛИТЕЛЬНОСТЬ PARQUET")
    print("=" * 70)

    if not parquet_path.exists():
        print(f"  [!] Не найден: {parquet_path}")
        return {"ok": False}

    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    cols = list(pf.schema_arrow.names)

    print(f"  Файл:        {parquet_path.name}")
    print(f"  Колонки:     {cols}")
    print(f"  n_rows:      {n_rows}")

    dur_at_assumed = n_rows / SFREQ_TYPE2_ASSUMED
    dur_at_real = n_rows / sfreq_real if sfreq_real > 0 else float("nan")

    print(f"\n  Длительность @ {SFREQ_TYPE2_ASSUMED} Hz (как в скрипте): {dur_at_assumed:.2f} сек")
    print(f"  Длительность @ {sfreq_real:.1f} Hz (как в .fif):       {dur_at_real:.2f} сек")

    return {
        "ok": True,
        "n_rows": n_rows,
        "cols": cols,
        "dur_at_assumed": dur_at_assumed,
        "dur_at_real": dur_at_real,
        "parquet_path": parquet_path,
    }


# =============================================================================
# 3. СТРУКТУРА ANNOTATION CSV
# =============================================================================
def check_annotation_structure() -> dict:
    print("\n" + "=" * 70)
    print("3. СТРУКТУРА ANNOTATION CSV")
    print("=" * 70)

    if not ANNOTATION_FILE.exists():
        print(f"  [!] Файл не найден: {ANNOTATION_FILE}")
        return {"ok": False}

    df = pd.read_csv(ANNOTATION_FILE)
    print(f"  Файл:        {ANNOTATION_FILE.name}")
    print(f"  shape:       {df.shape}")
    print(f"  columns:     {list(df.columns)}")

    # Ищем колонку времени и колонку состояния
    time_col = next((c for c in ["time", "Время", "Time", "t"] if c in df.columns), None)
    beh_col  = next((c for c in ["beh_type", "Разметка", "label", "state", "type"] if c in df.columns), None)
    name_col = next((c for c in ["name", "Имя", "subject"] if c in df.columns), None)

    print(f"\n  time_col:    {time_col}")
    print(f"  beh_col:     {beh_col}")
    print(f"  name_col:    {name_col}")

    if not time_col or not beh_col or not name_col:
        print("  [!!] Не удалось определить обязательные колонки.")
        return {"ok": False, "df": df, "time_col": time_col, "beh_col": beh_col, "name_col": name_col}

    print(f"\n  Уникальные значения {beh_col}:")
    for v, cnt in df[beh_col].value_counts().items():
        print(f"    '{v}': {cnt}")

    print(f"\n  Уникальные значения {name_col} (первые 15):")
    for v in df[name_col].dropna().unique()[:15]:
        print(f"    '{v}'")

    print(f"\n  time range:  [{df[time_col].min():.2f}, {df[time_col].max():.2f}]")
    print(f"  time dtype:  {df[time_col].dtype}")

    return {
        "ok": True,
        "df": df,
        "time_col": time_col,
        "beh_col": beh_col,
        "name_col": name_col,
    }


# =============================================================================
# 4. МАТЧИНГ PARQUET-ИМЕНИ И ИМЕНИ СУБЪЕКТА
# =============================================================================
def check_subject_match(parquet_path: Path, ann_info: dict) -> dict:
    print("\n" + "=" * 70)
    print("4. МАТЧИНГ ИМЕНИ СУБЪЕКТА")
    print("=" * 70)

    if not ann_info.get("ok"):
        print("  [skip] аннотация недоступна")
        return {"ok": False}

    name_col = ann_info["name_col"]
    unique_names = ann_info["df"][name_col].dropna().unique()
    file_stem = parquet_path.name

    print(f"  Parquet stem: {file_stem}")
    print(f"\n  Поиск подстроки в {name_col}:")
    matches = [n for n in unique_names if str(n).strip() in file_stem]
    for n in matches:
        print(f"    [match] '{n}'")
    if not matches:
        print("    [!!] НИЧЕГО НЕ НАЙДЕНО.")

    matched = matches[0] if matches else None
    if matched:
        # сколько строк сматчилось
        sub_df = ann_info["df"][ann_info["df"][name_col] == matched]
        print(f"\n  Выбранный субъект: '{matched}'")
        print(f"  Строк в аннотации для него: {len(sub_df)}")
        print(f"  time range: [{sub_df[ann_info['time_col']].min():.2f}, "
              f"{sub_df[ann_info['time_col']].max():.2f}]")

    return {"ok": bool(matched), "matched_name": matched, "sub_df": sub_df if matched else None}


# =============================================================================
# 5. ПЕРЕСЕЧЕНИЕ ВРЕМЕНИ PARQUET И АННОТАЦИИ
# =============================================================================
def check_time_overlap(pq_info: dict, ann_info: dict, sub_info: dict, sfreq_real: float) -> dict:
    print("\n" + "=" * 70)
    print("5. ПЕРЕСЕЧЕНИЕ ВРЕМЕНИ PARQUET И АННОТАЦИИ")
    print("=" * 70)

    if not (pq_info.get("ok") and ann_info.get("ok") and sub_info.get("ok")):
        print("  [skip] предыдущие проверки не прошли")
        return {"ok": False}

    sub_df = sub_info["sub_df"]
    time_col = ann_info["time_col"]
    beh_col = ann_info["beh_col"]

    n_rows = pq_info["n_rows"]

    # Два сценария длительности parquet
    dur_assumed = pq_info["dur_at_assumed"]
    dur_real = pq_info["dur_at_real"]
    ann_max = sub_df[time_col].max()

    print(f"  Parquet n_rows:              {n_rows}")
    print(f"  Длительность @ 200 Hz:       {dur_assumed:.1f} сек")
    print(f"  Длительность @ {sfreq_real:.0f} Hz:  {dur_real:.1f} сек")
    print(f"  Annotation time max:         {ann_max:.1f} сек")

    print("\n  Совпадение длительностей:")
    for label, dur in [("200 Hz", dur_assumed), (f"{sfreq_real:.0f} Hz", dur_real)]:
        diff = abs(dur - ann_max)
        rel = diff / max(ann_max, 1)
        flag = "OK" if rel < 0.05 else "MISMATCH"
        print(f"    [{flag}] @ {label}: |{dur:.1f} - {ann_max:.1f}| = {diff:.1f} сек ({100*rel:.1f}%)")

    return {
        "ok": True,
        "n_rows": n_rows,
        "dur_assumed": dur_assumed,
        "dur_real": dur_real,
        "ann_max": ann_max,
    }


# =============================================================================
# 6. РАСПРЕДЕЛЕНИЕ МЕТРИК ПО СОСТОЯНИЯМ
# =============================================================================
def _normalize_states(raw_states: np.ndarray) -> np.ndarray:
    """Приводим любые синонимы к каноническим Верно/Ошибка/Сон."""
    canonical = np.full(len(raw_states), None, dtype=object)
    for canon, aliases in STATE_ALIASES.items():
        for alias in aliases:
            mask = np.array([str(s).strip().lower() == alias for s in raw_states])
            canonical[mask] = canon
    # также исходные канонические (на случай если регистр совпадает)
    for canon in EXPECTED_STATES:
        mask = raw_states == canon
        canonical[mask] = canon
    return canonical


def check_state_distribution(pq_info: dict, ann_info: dict, sub_info: dict,
                              sfreq_used: float) -> dict:
    print("\n" + "=" * 70)
    print(f"6. РАСПРЕДЕЛЕНИЕ МЕТРИК ПО СОСТОЯНИЯМ (sfreq = {sfreq_used} Hz)")
    print("=" * 70)

    if not (pq_info.get("ok") and ann_info.get("ok") and sub_info.get("ok")):
        print("  [skip] предыдущие проверки не прошли")
        return {"ok": False}

    # Собираем states(t) из аннотации
    sub_df = sub_info["sub_df"]
    time_col = ann_info["time_col"]
    beh_col = ann_info["beh_col"]

    # Раскладываем аннотацию в lookup {sec_int: raw_beh_type}
    ann_lookup = {}
    for _, row in sub_df.iterrows():
        t = int(row[time_col])
        ann_lookup[t] = row[beh_col]

    # Читаем parquet
    parquet_path = pq_info["parquet_path"]
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    n_rows = len(df)
    indices = np.arange(n_rows)
    times_sec = (indices // int(sfreq_used)).astype(int)

    raw_states = np.array([ann_lookup.get(t, None) for t in times_sec], dtype=object)
    canonical = _normalize_states(raw_states)

    print(f"  times_sec range: [{times_sec.min()}, {times_sec.max()}]")

    # Сколько попало в известные состояния
    print(f"\n  Распределение по состояниям:")
    print(f"    {'state':10s}  {'n':>8s}  {'%':>6s}  {'t_range (sec)':>20s}")
    for st in EXPECTED_STATES:
        mask = canonical == st
        n = int(mask.sum())
        pct = 100 * n / n_rows if n_rows else 0.0
        if n > 0:
            t_range = f"[{times_sec[mask].min()}, {times_sec[mask].max()}]"
        else:
            t_range = "-"
        print(f"    {st:10s}  {n:>8d}  {pct:>5.1f}%  {t_range:>20s}")

    n_unknown = int((canonical == None).sum())
    print(f"    {'(unknown)':10s}  {n_unknown:>8d}  {100*n_unknown/n_rows:>5.1f}%")

    # Если всё unknown — плохо
    if n_unknown == n_rows:
        print("\n  [!!] НИ ОДНА точка не сматчилась ни с одним состоянием.")
        print("       Проверьте: значения beh_type в аннотации, диапазон time,")
        print("       и совпадение sfreq (times_sec).")
        return {"ok": False, "canonical": canonical, "times_sec": times_sec,
                "df": df, "n_unknown": n_unknown}

    # Метрики по состояниям
    metrics = ["reconstruction_residual", "metric_trace", "participation_ratio"]
    metrics = [m for m in metrics if m in df.columns]

    print(f"\n  Метрики по состояниям (mean ± std):")
    header = f"    {'state':10s}" + "".join([f"  {m:>26s}" for m in metrics])
    print(header)
    print("    " + "-" * (len(header) - 4))

    for st in EXPECTED_STATES:
        mask = canonical == st
        if mask.sum() == 0:
            continue
        row = f"    {st:10s}"
        for m in metrics:
            vals = df[m].values[mask]
            row += f"  {vals.mean():>12.4f} ± {vals.std():>8.4f}"
        print(row)

    return {
        "ok": True,
        "canonical": canonical,
        "times_sec": times_sec,
        "df": df,
        "n_unknown": n_unknown,
        "metrics": metrics,
    }


# =============================================================================
# 7. SANITY-PLOT d(t) С ТАЙМЛАЙНОМ
# =============================================================================
def make_sanity_plot(state_info: dict, sub_info: dict, ann_info: dict,
                     out_dir: Path, subject_tag: str) -> None:
    if not state_info.get("ok"):
        return

    times_sec = state_info["times_sec"]
    canonical = state_info["canonical"]
    df = state_info["df"]

    if "eigenvalues" not in df.columns:
        print("[skip plot] нет колонки eigenvalues")
        return

    # Считаем d_Ratio Gap по всему массиву (векторно, не через compute_paper_methods)
    evals = np.vstack(df["eigenvalues"].values)  # (T, K)
    eps = np.finfo(np.float32).eps
    ratios = evals[:, :-1] / (evals[:, 1:] + eps)
    d_ratio = (np.argmax(ratios, axis=1) + 1).astype(np.float32)
    zero_tol = 10 * eps
    d_ratio = np.where(evals[:, 0] > zero_tol, d_ratio, 0.0)

    # Downsample для графика
    stride = max(1, len(times_sec) // 5000)
    t_plot = times_sec[::stride]
    d_plot = d_ratio[::stride]
    pr_plot = df["participation_ratio"].values[::stride] if "participation_ratio" in df.columns else None
    res_plot = df["reconstruction_residual"].values[::stride] if "reconstruction_residual" in df.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, constrained_layout=True)

    # --- d(t) ---
    ax = axes[0]
    ax.plot(t_plot, d_plot, lw=0.5, color="purple", alpha=0.7)
    ax.set_ylabel("d (Ratio Gap)")
    ax.set_title(f"Sanity check: {subject_tag}", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)

    # --- participation ratio ---
    if pr_plot is not None:
        ax = axes[1]
        ax.plot(t_plot, pr_plot, lw=0.5, color="teal", alpha=0.7)
        ax.set_ylabel("Participation Ratio")
        ax.grid(True, ls="--", alpha=0.4)

    # --- residual ---
    if res_plot is not None:
        ax = axes[2]
        ax.plot(t_plot, res_plot, lw=0.5, color="crimson", alpha=0.7)
        ax.set_ylabel("Residual")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel("Время, сек")

    # --- Таймлайн состояний на всех осях ---
    colors = {"Верно": "#2ca02c", "Ошибка": "#d62728", "Сон": "#1f77b4"}
    for ax in axes:
        prev_st = None
        for i in range(len(canonical)):
            st = canonical[i]
            if st != prev_st and st is not None:
                ax.axvline(times_sec[i], color=colors.get(st, "gray"),
                           alpha=0.35, lw=0.8)
                prev_st = st

    # Легенда состояний
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=s) for s, c in colors.items()]
    axes[0].legend(handles=handles, loc="upper right", ncol=3, fontsize=9)

    out_file = out_dir / f"sanity_{subject_tag}.png"
    plt.savefig(out_file, dpi=140)
    plt.close()
    print(f"\n[+] Sanity plot сохранён: {out_file}")


# =============================================================================
# MAIN
# =============================================================================
def find_fif_for_stem(stem: str, raw_dir: Path) -> Path:
    """По stem parquet найти исходный .fif (эвристика по ключевым словам)."""
    # Убираем _metric_spectrum_N\d+ и похожие суффиксы
    base = re.sub(r"_metric_spectrum_N\d+$", "", stem)
    base = re.sub(r"^(type2_)?", "", base)

    candidates = list(raw_dir.glob("*.fif"))
    # Пробуем разные варианты
    for cand in candidates:
        cname = cand.stem.replace("_raw_preprocessed", "").replace("_raw", "")
        if all(word in cname for word in base.split("_")[:2]):
            return cand
    return candidates[0] if candidates else raw_dir / "NOT_FOUND.fif"


def main():
    parser = argparse.ArgumentParser(description="Diagnose Type2 sync issues")
    parser.add_argument("stem", type=str,
                        help="Stem parquet-файла без расширения, "
                             "например type2_bainbridge_emily_220519_raw_metric_spectrum_N24")
    parser.add_argument("--fif", type=str, default=None,
                        help="Явный путь к .fif (иначе ищется автоматически)")
    parser.add_argument("--sfreq-override", type=float, default=None,
                        help="Использовать эту sfreq вместо SFREQ_TYPE2 (для теста)")
    args = parser.parse_args()

    # Ищем parquet
    parquets = list(DB_DIR.glob(f"**/{args.stem}.parquet"))
    if not parquets:
        print(f"[!] Parquet не найден: {args.stem}")
        print(f"    Поиск в {DB_DIR}")
        sys.exit(1)
    parquet_path = parquets[0]

    # Ищем fif
    if args.fif:
        fif_path = Path(args.fif)
    else:
        fif_path = find_fif_for_stem(args.stem, RAW_DIR)

    print(f"[*] Parquet: {parquet_path}")
    print(f"[*] FIF:     {fif_path}")

    # --- 1 ---
    sfreq_info = check_sfreq(fif_path)
    sfreq_real = sfreq_info.get("sfreq", SFREQ_TYPE2_ASSUMED)

    # --- 2 ---
    pq_info = check_parquet_duration(parquet_path, sfreq_real)

    # --- 3 ---
    ann_info = check_annotation_structure()

    # --- 4 ---
    sub_info = check_subject_match(parquet_path, ann_info)

    # --- 5 ---
    overlap_info = check_time_overlap(pq_info, ann_info, sub_info, sfreq_real)

    # --- 6 ---
    # Выбор sfreq для раскладки states:
    #   по умолчанию — та, что в скрипте (200)
    #   если явно передан --sfreq-override, использовать его
    #   если реальная sfreq в .fif отличается, показать оба варианта
    sfreq_used = args.sfreq_override or SFREQ_TYPE2_ASSUMED
    state_info = check_state_distribution(pq_info, ann_info, sub_info, sfreq_used)

    # Если результат плохой, попробуем с реальной sfreq
    if state_info.get("ok") and state_info.get("n_unknown", 0) == pq_info.get("n_rows", 1):
        if sfreq_real != sfreq_used:
            print(f"\n[*] Повтор проверки с реальной sfreq = {sfreq_real} Hz...")
            state_info = check_state_distribution(pq_info, ann_info, sub_info, sfreq_real)

    # --- 7 ---
    subject_tag = re.sub(r"_metric_spectrum_N\d+$", "", args.stem)
    make_sanity_plot(state_info, sub_info, ann_info, OUT_DIR, subject_tag)

    # --- ИТОГ ---
    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)
    print(f"  sfreq в .fif:              {sfreq_real} Hz")
    print(f"  SFREQ_TYPE2 в скрипте:     {SFREQ_TYPE2_ASSUMED} Hz")
    if abs(sfreq_real - SFREQ_TYPE2_ASSUMED) > 1e-3:
        print(f"  [!!] НЕСООТВЕТСТВИЕ. Везде, где в analyze используются индексы,")
        print(f"       соответствующие 200 Hz, реально используется {sfreq_real} Hz.")
        print(f"       Это сдвигает times_sec в {sfreq_real/SFREQ_TYPE2_ASSUMED:.2f} раз.")
    else:
        print(f"  [OK] частоты совпадают")

    if not sub_info.get("ok"):
        print(f"  [!!] Имя субъекта не найдено в аннотации.")

    if state_info.get("ok"):
        n_unk = state_info.get("n_unknown", 0)
        n_all = pq_info.get("n_rows", 1)
        print(f"  Unmapped samples: {n_unk} / {n_all} ({100*n_unk/n_all:.1f}%)")
        if n_unk == n_all:
            print(f"  [!!] Ни одна точка не сматчилась — states = None везде.")
        elif n_unk > 0.7 * n_all:
            print(f"  [!!] Больше 70% точек не сматчились — сильный сдвиг.")
        else:
            print(f"  [OK] Основная часть точек сматчена.")


if __name__ == "__main__":
    main()
