import mne
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = WORKSPACE_DIR / "04_processed_db" / "type1_features" / "type1_sub10_intrinsic_dimension.parquet"
FIF_PATH = WORKSPACE_DIR / "01_raw_data" / "type1_bigrams" / "type1_sub10_chistova_alena_raw_preprocessed.fif"
EVENTS_CSV = WORKSPACE_DIR / "02_metadata" / "annotations" / "type1_sub10_chistova_alena_events.csv"
OUT_DIR = WORKSPACE_DIR / "04_processed_db" / "type1_features"

def load_and_align_events(csv_path: Path, n_meg_triggers: int = 805) -> pd.DataFrame:
    """Загрузка CSV, удаление инструкций и коррекция пропущенных триггеров 780-782."""
    df = pd.read_csv(csv_path)
    mask = df["text.started"].notna() if "text.started" in df.columns else df["bigrams"].notna()
    df_trials = df[mask].copy().reset_index(drop=True)
    
    if len(df_trials) == 808 and n_meg_triggers == 805:
        print("[!] Применение коррекции sub_10: удаление строк 780, 781, 782...")
        df_trials = df_trials.drop([780, 781, 782]).reset_index(drop=True)
        
    return df_trials

def compute_event_triggered_profile(signal_array, trigger_indices, pre_samp=5000, post_samp=5000):
    """Вырезка окон [-5с, +5с] и усреднение по триалам для каждого момента времени dt (шаг 1 мс)."""
    T_total = len(signal_array)
    epochs = []
    
    for idx in trigger_indices:
        if np.isnan(idx):
            continue
        idx_int = int(round(idx))
        start_idx = idx_int - pre_samp
        end_idx = idx_int + post_samp + 1  # 10001 сэмпл
        
        if start_idx >= 0 and end_idx <= T_total:
            epochs.append(signal_array[start_idx:end_idx])
            
    epochs = np.array(epochs, dtype=np.float32)
    mean_d = np.mean(epochs, axis=0)
    sem_d = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    return mean_d, sem_d, len(epochs)

def main():
    print(f"[*] Чтение размерностей: {PARQUET_PATH.name}...")
    df = pl.read_parquet(PARQUET_PATH)
    d_vals = df["intrinsic_dimension"].to_numpy().astype(np.float32)
    
    print(f"[*] Чтение триггеров МЭГ: {FIF_PATH.name}...")
    raw = mne.io.read_raw_fif(FIF_PATH, preload=False, verbose=False)
    first_samp = raw.first_samp
    
    # 1. Точки Креста (t=0)
    events = mne.find_events(raw, stim_channel="STI101", min_duration=0.002, verbose=False)
    cross_events = events[events[:, 2] == 8]
    cross_idx = (cross_events[:, 0] - first_samp).astype(np.float64)
    n_cross = len(cross_idx)
    print(f"[+] Аппаратных триггеров креста (STI101=8): {n_cross}")
    
    # 2. Точки Биграммы (t=0): строго +500 мс от креста
    bigram_idx = cross_idx + 500.0
    
    # 3. Точки Моторного ответа (t=0, СДВИГ 0): Крест + 500 мс + RT
    df_trials = load_and_align_events(EVENTS_CSV, n_meg_triggers=n_cross)
    rt_col = next(c for c in ["key_resp.rt", "RT", "rt"] if c in df_trials.columns)
    rts = df_trials[rt_col].values.astype(np.float64)
    
    min_len = min(n_cross, len(rts))
    resp_idx = bigram_idx[:min_len] + rts[:min_len] * 1000.0
    
    # Временная шкала от -5.0 до +5.0 секунд
    t_axis = np.linspace(-2.5, 2.5, 10001)
    
    events_config = [
        {
            "id": "cross",
            "title": "Средняя внутренняя размерность в зависимости от времени, прошедшего с появления креста на экране",
            "indices": cross_idx,
            "color": "navy",
            "file": OUT_DIR / "type1_sub10_er_cross_5s.png",
            "xlabel": "Время относительно старта креста: dt = t - t_cross (секунды)",
            "vline_label": "t = 0: Крест фиксации",
            "add_lines": [(0.5, "Биграмма (+0.5с)", "crimson", ":")]
        },
        {
            "id": "bigram",
            "title": "Средняя внутренняя размерность в зависимости от времени, прошедшего с появления биграммы(через 0.5 сек после креста) на экране",
            "indices": bigram_idx,
            "color": "crimson",
            "file": OUT_DIR / "type1_sub10_er_bigram_5s.png",
            "xlabel": "Время относительно предъявления биграммы: dt = t - t_stim (секунды)",
            "vline_label": "t = 0: Предъявление биграммы",
            "add_lines": [(-0.5, "Крест фиксации (-0.5с)", "navy", ":")]
        },
        {
            "id": "response",
            "title": "Средняя внутренняя размерность в зависимости от времени, прошедшего с нажатия кнопки",
            "indices": resp_idx,
            "color": "darkgreen",
            "file": OUT_DIR / "type1_sub10_er_response_5s.png",
            "xlabel": "Время относительно нажатия кнопки: dt = t - t_RT (секунды)",
            "vline_label": "t = 0: Моторный ответ (RT)",
            "add_lines": []
        }
    ]
    
    # -------------------------------------------------------------
    # 1. СОХРАНЕНИЕ 3 РАЗДЕЛЬНЫХ КАРТИНОК
    # -------------------------------------------------------------
    results = []
    for cfg in events_config:
        mean_d, sem_d, n_valid = compute_event_triggered_profile(d_vals, cfg["indices"])
        results.append((cfg, mean_d, sem_d, n_valid))
        
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.plot(t_axis, mean_d, color=cfg["color"], linewidth=2.0, label=f"Средняя размерность <d> (N={n_valid} триалов)")
        ax.fill_between(t_axis, mean_d - sem_d, mean_d + sem_d, color=cfg["color"], alpha=0.25, label="±1 SEM")
        
        # Основная отметка события (t=0)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label=cfg["vline_label"])
        
        # Дополнительные временные засечки
        for pos, label, c, style in cfg["add_lines"]:
            ax.axvline(pos, color=c, linestyle=style, linewidth=1.3, label=label)
            
        ax.set_title(f"Локальная размерность d(t) вокруг события: {cfg['title']} [-2.5с, +2.5с]", fontsize=12, fontweight="bold")
        ax.set_xlabel(cfg["xlabel"], fontsize=11)
        ax.set_ylabel("Внутренняя размерность (d)", fontsize=11)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(9.0, 10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        
        plt.savefig(cfg["file"], dpi=160)
        plt.close()
        print(f"[+] Сохранен отдельный график: {cfg['file'].name}")

    # -------------------------------------------------------------
    # 2. СВОДНЫЙ 3-ПАНЕЛЬНЫЙ ГРАФИК (ВСЕ 3 СОБЫТИЯ ВМЕСТЕ)
    # -------------------------------------------------------------
    comp_file = OUT_DIR / "type1_sub10_er_all_three_events_5s.png"
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    
    for row, (cfg, mean_d, sem_d, n_valid) in enumerate(results):
        ax = axes[row]
        ax.plot(t_axis, mean_d, color=cfg["color"], linewidth=2.0, label=f"<d> ({cfg['title']})")
        ax.fill_between(t_axis, mean_d - sem_d, mean_d + sem_d, color=cfg["color"], alpha=0.25)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
        
        for pos, label, c, style in cfg["add_lines"]:
            ax.axvline(pos, color=c, linestyle=style, linewidth=1.2, alpha=0.8)
            
        ax.set_ylabel("Размерность (d)", fontsize=10)
        ax.set_title(f"{cfg['title']} (N = {n_valid})", fontsize=11, fontweight="bold")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(9.0, 10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)
        
    axes[2].set_xlabel("Разница во времени от триггера: dt = t - t_event (секунды)", fontsize=11)
    plt.savefig(comp_file, dpi=160)
    plt.close()
    print(f"[+] Сохранен сводный 3-панельный график: {comp_file.name}")

if __name__ == "__main__":
    main()
