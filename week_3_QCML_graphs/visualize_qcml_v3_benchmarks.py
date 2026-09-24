#!/usr/bin/env python3

import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import mne

mne.set_log_level("ERROR")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})

DIMENSION_THRESHOLD = 0.5
TYPE1_SAMPLING_FREQUENCY_HZ = 1000.0
TYPE2_SAMPLING_FREQUENCY_HZ = 200.0
FIXATION_CROSS_DURATION_SAMPLES = 500
TRIAL_HALF_WINDOW_SAMPLES = 2500
PARQUET_CHUNK_SIZE = 50_000
FIGURE_DPI = 160
TOP_ANNOTATED_CHANNELS = 10

STATE_CORRECT = "Верно"
STATE_ERROR = "Ошибка"
STATE_SLEEP = "Сон"

CONDITION_TASK = "task_bigrams"
CONDITION_RESTING = "resting_state"
CONDITION_EMPTY_ROOM = "empty_room"
CONDITION_DROWSINESS = "drowsiness"

PALETTE_STATES = {
    STATE_CORRECT: "#2ca02c",
    STATE_ERROR: "#d62728",
    STATE_SLEEP: "#1f77b4",
}

SPECTRUM_FILENAME_PATTERN = re.compile(
    r"(?P<stem>.+?)_spectrum_D(?P<D>\d+)_N(?P<N>\d+)_m(?P<m>\d+)"
    r"_tau(?P<tau>\d+)ms_(?P<norm>\w+)_w(?P<w>[\d.]+)_(?P<grad>\w+)\.parquet"
)

LEVERAGE_FILENAME_PATTERN = re.compile(
    r"(?P<stem>.+?)_leverage_D(?P<D>\d+)_N(?P<N>\d+)_m(?P<m>\d+)"
    r"_tau(?P<tau>\d+)ms_(?P<norm>\w+)_w(?P<w>[\d.]+)_(?P<grad>\w+)\.pt"
)


def gramian_dimension(hilbert_dimension: int) -> int:
    return 2 * (hilbert_dimension - 1)

def is_readable_parquet(parquet_path: Path) -> bool:
    try:
        pq.read_metadata(parquet_path)
        return True
    except Exception:
        return False
def integer_dimension_bins(max_dimension: int) -> np.ndarray:
    return np.arange(-0.5, max_dimension + 1.5, 1.0)


def detect_condition(stem: str) -> str:
    if stem.startswith("type2_") or "drowsiness" in stem:
        return CONDITION_DROWSINESS
    if stem.startswith("er_") or "empty_room" in stem:
        return CONDITION_EMPTY_ROOM
    if "resting" in stem:
        return CONDITION_RESTING
    return CONDITION_TASK


def parse_spectrum_filename(filename: str) -> Optional[Dict]:
    match = SPECTRUM_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "stem": groups["stem"],
        "subject": groups["stem"],
        "D": int(groups["D"]),
        "N": int(groups["N"]),
        "m": int(groups["m"]),
        "tau_ms": int(groups["tau"]),
        "norm": groups["norm"],
        "w": float(groups["w"]),
        "grad": groups["grad"],
        "condition": detect_condition(groups["stem"]),
        "parquet_name": filename,
    }


def parse_leverage_filename(filename: str) -> Optional[Dict]:
    match = LEVERAGE_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "stem": groups["stem"],
        "subject": groups["stem"],
        "D": int(groups["D"]),
        "N": int(groups["N"]),
        "m": int(groups["m"]),
        "tau_ms": int(groups["tau"]),
        "norm": groups["norm"],
        "w": float(groups["w"]),
        "grad": groups["grad"],
        "condition": detect_condition(groups["stem"]),
        "leverage_name": filename,
    }


def discover_spectrum_files(db_dir: Path) -> List[Dict]:
    discovered = []
    for parquet_path in sorted(db_dir.rglob("*_spectrum_*.parquet")):
        parsed = parse_spectrum_filename(parquet_path.name)
        if parsed is None:
            continue
        parsed["parquet_path"] = parquet_path
        discovered.append(parsed)
    return discovered


def discover_leverage_files(db_dir: Path) -> List[Dict]:
    discovered = []
    for leverage_path in sorted(db_dir.rglob("*_leverage_*.pt")):
        parsed = parse_leverage_filename(leverage_path.name)
        if parsed is None:
            continue
        parsed["leverage_path"] = leverage_path
        discovered.append(parsed)
    return discovered


def threshold_dimension_series(eigenvalues: np.ndarray) -> np.ndarray:
    return (eigenvalues > DIMENSION_THRESHOLD).sum(axis=1).astype(np.float32)


def read_threshold_dimension_series(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["eigenvalues"])
    eigenvalues = np.vstack(table.column("eigenvalues").to_numpy(zero_copy_only=False))
    return threshold_dimension_series(eigenvalues)


def load_leverage_payload(leverage_path: Path) -> Dict:
    return torch.load(leverage_path, map_location="cpu", weights_only=False)


def normalize_positive(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0:
        return values.copy()
    return values / total


def find_leverage_entry(spectrum_entry: Dict, leverage_entries: List[Dict]) -> Optional[Dict]:
    for entry in leverage_entries:
        if entry["stem"] == spectrum_entry["stem"] and entry["N"] == spectrum_entry["N"]:
            return entry
    return None


def sensor_2d_coordinates(
    channel_indices: Optional[List[int]],
    raw_dir: Path,
) -> Tuple[np.ndarray, List[str]]:
    sample_fif = None
    if raw_dir.exists():
        for fif_candidate in raw_dir.rglob("*.fif"):
            sample_fif = fif_candidate
            break

    if sample_fif is not None:
        raw = mne.io.read_raw_fif(sample_fif, preload=False, verbose=False)
        magnetometer_picks = mne.pick_types(raw.info, meg="mag", eeg=False, stim=False)
        gradiometer_picks = mne.pick_types(raw.info, meg="grad", eeg=False, stim=False)
        selected_picks = np.concatenate([magnetometer_picks, gradiometer_picks])

        layout = mne.channels.find_layout(raw.info, ch_type="meg")
        name_to_position = {
            layout.names[i]: layout.pos[i, :2] for i in range(len(layout.names))
        }
        channel_names = [raw.info["ch_names"][pick] for pick in selected_picks]
        coordinates = np.array([name_to_position[name] for name in channel_names])
        return coordinates, channel_names

    layout = mne.channels.read_layout("Vectorview-all")
    name_to_position = {
        layout.names[i]: layout.pos[i, :2] for i in range(len(layout.names))
    }
    magnetometer_names = [name for name in layout.names if name.endswith("1")]
    gradiometer_names = [
        name for name in layout.names
        if name.endswith("2") or name.endswith("3")
    ]
    channel_names = magnetometer_names + gradiometer_names
    coordinates = np.array([name_to_position[name] for name in channel_names])
    return coordinates, channel_names


def plot_channel_leverage_2d(
    output_path: Path,
    subject: str,
    condition: str,
    hilbert_dimension: int,
    coordinates: np.ndarray,
    channel_names: List[str],
    leverage_values: np.ndarray,
) -> None:
    normalized_leverage = normalize_positive(leverage_values)
    uniform_level = 1.0 / len(normalized_leverage)
    color_ceiling = float(np.percentile(normalized_leverage, 99))
    color_norm = Normalize(vmin=0.0, vmax=max(color_ceiling, uniform_level * 1.5))

    marker_sizes = 25 + np.clip(normalized_leverage / uniform_level, 0.2, 10.0) * 35

    fig, ax = plt.subplots(figsize=(10, 9), dpi=FIGURE_DPI)
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=normalized_leverage,
        s=marker_sizes,
        cmap="inferno",
        norm=color_norm,
        edgecolors="#222222",
        linewidth=0.6,
        alpha=0.92,
        zorder=3,
    )

    top_indices = np.argsort(normalized_leverage)[::-1][:TOP_ANNOTATED_CHANNELS]
    for index in top_indices:
        ax.annotate(
            f"{channel_names[index]}\n{normalized_leverage[index] * 100:.2f}%",
            xy=(coordinates[index, 0], coordinates[index, 1]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#111111",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#777777", alpha=0.8),
        )

    ax.set_title(f"Channel leverage: {subject} ({condition}), N={hilbert_dimension}")
    ax.set_xlabel("Лево  $\\longleftrightarrow$  Право")
    ax.set_ylabel("Затылок  $\\longleftrightarrow$  Лоб")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Вклад канала $\ell_k = g_{kk} / \mathrm{Tr}(g)$")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def detect_cross_triggers(fif_path: Path) -> Optional[np.ndarray]:
    if not fif_path.exists():
        return None
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    events = mne.find_events(raw, stim_channel="STI101", min_duration=0.002, verbose=False)
    cross_events = events[events[:, 2] == 8]
    if len(cross_events) == 0:
        return None
    return (cross_events[:, 0] - raw.first_samp).astype(np.float64)


def find_type1_fif(subject: str, raw_dir: Path) -> Optional[Path]:
    candidates = list((raw_dir / "type1_bigrams").glob(f"*{subject}*.fif"))
    candidates = [
        candidate for candidate in candidates
        if not candidate.name.endswith("-1.fif") and not candidate.name.endswith("-2.fif")
    ]
    return candidates[0] if candidates else None


def find_type1_psychopy_csv(subject: str, annotations_dir: Path) -> Optional[Path]:
    candidates = list(annotations_dir.glob(f"*{subject}*.csv"))
    return candidates[0] if candidates else None


def extract_response_latencies(csv_path: Path) -> Optional[np.ndarray]:
    if csv_path is None or not csv_path.exists():
        return None
    table = pd.read_csv(csv_path)
    latency_column = next(
        (name for name in ("key_resp.rt", "RT", "rt") if name in table.columns),
        None,
    )
    if latency_column is None:
        return None
    return table[latency_column].to_numpy(dtype=np.float64)


def trial_average_profile(
    series: np.ndarray, triggers: np.ndarray, half_window_samples: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    selected = []
    for trigger in triggers:
        if np.isnan(trigger):
            continue
        index = int(round(trigger))
        if index - half_window_samples >= 0 and index + half_window_samples < len(series):
            selected.append(series[index - half_window_samples : index + half_window_samples])
    if not selected:
        length = 2 * half_window_samples
        return np.zeros(length, dtype=np.float32), np.zeros(length, dtype=np.float32), 0
    epochs = np.stack(selected, axis=0)
    mean_profile = epochs.mean(axis=0)
    sem_profile = epochs.std(axis=0, ddof=1) / np.sqrt(epochs.shape[0])
    return mean_profile, sem_profile, epochs.shape[0]


def load_type2_annotations(annotations_dir: Path) -> Optional[pd.DataFrame]:
    annotation_path = annotations_dir / "Разметка интерполированная (все).csv"
    if not annotation_path.exists():
        return None
    return pd.read_csv(annotation_path)


def select_type2_annotation_rows(table: pd.DataFrame, stem: str) -> Optional[pd.DataFrame]:
    unique_names = table["name"].dropna().unique()
    matched = next((name for name in unique_names if str(name).strip() in stem), None)
    if matched is None:
        return None
    return table[table["name"] == matched].sort_values("time").reset_index(drop=True)


def plot_erid(
    output_path: Path,
    subject: str,
    hilbert_dimension: int,
    event_label: str,
    mean_profile: np.ndarray,
    sem_profile: np.ndarray,
    trial_count: int,
    sampling_frequency_hz: float,
    half_window_samples: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    if trial_count > 0:
        time_axis = (np.arange(len(mean_profile)) - half_window_samples) / sampling_frequency_hz
        color = "#b71c1c"
        ax.plot(time_axis, mean_profile, color=color, linewidth=2.0,
                label=f"d_th (N={trial_count})")
        ax.fill_between(time_axis, mean_profile - sem_profile, mean_profile + sem_profile,
                        color=color, alpha=0.22)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Время от события, с")
    ax.set_ylabel(r"$\langle d_{threshold}(t) \rangle_{триалы}$")
    ax.set_title(f"ERID ({event_label}): {subject}, N={hilbert_dimension}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_state_distributions(
    output_path: Path,
    subject: str,
    hilbert_dimension: int,
    state_table: pd.DataFrame,
    max_dimension: int,
) -> None:
    states = [STATE_CORRECT, STATE_ERROR, STATE_SLEEP]
    bins = integer_dimension_bins(max_dimension)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for state in states:
        state_values = state_table[state_table["state"] == state]["dimension"]
        if len(state_values) > 0:
            ax.hist(
                state_values,
                bins=bins,
                density=True,
                alpha=0.35,
                label=state,
                color=PALETTE_STATES[state],
                histtype="stepfilled",
                edgecolor="black",
            )
    ax.set_xlabel("Размерность d")
    ax.set_ylabel("Плотность P(d)")
    ax.set_title(f"Распределения d(x) по состояниям: {subject}, N={hilbert_dimension}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def build_state_table(
    dimension_series: np.ndarray,
    annotation_table: pd.DataFrame,
    sampling_frequency_hz: float,
) -> Optional[pd.DataFrame]:
    annotation_lookup = dict(zip(annotation_table["time"].astype(int), annotation_table["beh_type"]))
    indices = np.arange(len(dimension_series))
    times_seconds = (indices / sampling_frequency_hz).astype(int)
    states = np.array([annotation_lookup.get(time, None) for time in times_seconds])
    valid_mask = np.isin(states, [STATE_CORRECT, STATE_ERROR, STATE_SLEEP])
    if not np.any(valid_mask):
        return None
    return pd.DataFrame({
        "state": states[valid_mask],
        "dimension": dimension_series[valid_mask],
    })


def process_task_bigrams(
    entry: Dict,
    output_root: Path,
    raw_dir: Path,
    annotations_dir: Path,
) -> None:
    parquet_path: Path = entry["parquet_path"]
    fif_path = find_type1_fif(entry["subject"], raw_dir)
    if fif_path is None:
        print(f"    [-] FIF не найден для {entry['subject']}, ERID пропущен")
        return

    cross_events = detect_cross_triggers(fif_path)
    if cross_events is None or len(cross_events) == 0:
        print(f"    [-] Триггеры не найдены в {fif_path.name}, ERID пропущен")
        return

    bigram_events = cross_events + FIXATION_CROSS_DURATION_SAMPLES

    csv_path = find_type1_psychopy_csv(entry["subject"], annotations_dir)
    latencies = extract_response_latencies(csv_path) if csv_path else None
    response_events = np.array([], dtype=np.float64)
    if latencies is not None:
        paired_count = min(len(bigram_events), len(latencies))
        valid_mask = ~np.isnan(latencies[:paired_count])
        response_events = (
            bigram_events[:paired_count]
            + latencies[:paired_count] * TYPE1_SAMPLING_FREQUENCY_HZ
        )[valid_mask]

    dimension_series = read_threshold_dimension_series(parquet_path)

    subject_dir = output_root / "erid" / entry["subject"]
    subject_dir.mkdir(parents=True, exist_ok=True)

    event_map = {
        "cross": cross_events,
        "bigram": bigram_events,
        "response": response_events,
    }
    for event_label, events in event_map.items():
        if len(events) == 0:
            continue
        mean_profile, sem_profile, trial_count = trial_average_profile(
            dimension_series, events, TRIAL_HALF_WINDOW_SAMPLES
        )
        output_path = subject_dir / f"erid_{event_label}_N{entry['N']}_m{entry['m']}.png"
        plot_erid(
            output_path,
            entry["subject"],
            entry["N"],
            event_label,
            mean_profile,
            sem_profile,
            trial_count,
            TYPE1_SAMPLING_FREQUENCY_HZ,
            TRIAL_HALF_WINDOW_SAMPLES,
        )
        print(f"      [+] ERID сохранён: {output_path.name}")


def process_drowsiness(
    entry: Dict,
    output_root: Path,
    annotations_dir: Path,
) -> None:
    annotation_table = load_type2_annotations(annotations_dir)
    if annotation_table is None:
        print("    [-] Аннотации Type 2 не найдены, состояния пропущены")
        return

    subject_annotations = select_type2_annotation_rows(annotation_table, entry["stem"])
    if subject_annotations is None:
        print(f"    [-] Аннотации для {entry['subject']} не найдены")
        return

    dimension_series = read_threshold_dimension_series(entry["parquet_path"])
    state_table = build_state_table(dimension_series, subject_annotations, TYPE2_SAMPLING_FREQUENCY_HZ)
    if state_table is None:
        print(f"    [-] Нет валидных состояний для {entry['subject']}")
        return

    subject_dir = output_root / "states" / entry["subject"]
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / f"states_N{entry['N']}_m{entry['m']}.png"
    plot_state_distributions(
        output_path,
        entry["subject"],
        entry["N"],
        state_table,
        gramian_dimension(entry["N"]),
    )
    print(f"      [+] Состояния сохранены: {output_path.name}")


def process_channel_leverage(
    entry: Dict,
    output_root: Path,
    leverage_entries: List[Dict],
    raw_dir: Path,
) -> None:
    leverage_entry = find_leverage_entry(entry, leverage_entries)
    if leverage_entry is None:
        return

    payload = load_leverage_payload(leverage_entry["leverage_path"])
    channel_leverage = np.asarray(payload["channel_leverage"])
    if channel_leverage.ndim != 1 or channel_leverage.size == 0:
        return

    magnetometer_picks = payload.get("mag_picks", [])
    gradiometer_picks = payload.get("grad_picks", [])
    channel_indices = (
        list(magnetometer_picks) + list(gradiometer_picks)
        if magnetometer_picks or gradiometer_picks
        else None
    )

    coordinates, channel_names = sensor_2d_coordinates(channel_indices, raw_dir)
    if coordinates.shape[0] != channel_leverage.shape[0]:
        print(
            f"    [-] Размер координат ({coordinates.shape[0]}) "
            f"не совпадает с leverage ({channel_leverage.shape[0]})"
        )
        return

    subject_dir = output_root / "channel_leverage" / entry["subject"]
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / f"leverage_2d_{entry['condition']}_N{entry['N']}_m{entry['m']}.png"

    plot_channel_leverage_2d(
        output_path,
        entry["subject"],
        entry["condition"],
        entry["N"],
        coordinates,
        channel_names,
        channel_leverage,
    )
    print(f"      [+] Channel leverage сохранён: {output_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCML v3 visualization")
    parser.add_argument("--db_dir", type=str, default="04_processed_db")
    parser.add_argument("--raw_dir", type=str, default="01_raw_data")
    parser.add_argument("--annotations_dir", type=str, default="02_metadata/annotations")
    parser.add_argument("--output_dir", type=str, default="week_3_QCML_graphs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_dir = Path(args.db_dir)
    raw_dir = Path(args.raw_dir)
    annotations_dir = Path(args.annotations_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    spectrum_entries = discover_spectrum_files(db_dir)
    leverage_entries = discover_leverage_files(db_dir)
    print(f"[*] Найдено {len(spectrum_entries)} спектров и {len(leverage_entries)} leverage-файлов")

    for entry in spectrum_entries:
        parquet_path = entry["parquet_path"]
        if not is_readable_parquet(parquet_path):
            print(f"\n▶ {entry['subject']} ({entry['condition']}) N={entry['N']} m={entry['m']}")
            print(f"    [-] Parquet не читается (недописан?): {parquet_path.name}, пропуск")
            continue
        print(f"\n▶ {entry['subject']} ({entry['condition']}) N={entry['N']} m={entry['m']}")

        if entry["condition"] in (CONDITION_TASK, CONDITION_RESTING, CONDITION_EMPTY_ROOM):
            process_channel_leverage(entry, output_root, leverage_entries, raw_dir)
        if entry["condition"] == CONDITION_TASK:
            process_task_bigrams(entry, output_root, raw_dir, annotations_dir)
        if entry["condition"] == CONDITION_DROWSINESS:
            process_drowsiness(entry, output_root, annotations_dir)
        gc.collect()

    print(f"\n[✓] Все визуализации сохранены в {output_root.resolve()}")


if __name__ == "__main__":
    main()
