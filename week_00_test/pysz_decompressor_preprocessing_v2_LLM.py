import os
from pathlib import Path
import mne
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import numpy as np
from pysz import sz
import glob
import pickle
import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = WORKSPACE_DIR / "01_raw_data"
RESTING_DIR = RAW_DATA_DIR / "resting_state"
EMPTY_ROOMS_DIR = RAW_DATA_DIR / "shared_empty_rooms"


def pysz_decompression(file_path: Path) -> Path:
    """Распаковка архива SZ3 в .fif файл."""
    # Удаляем суффикс .pysz: 'name.fif.pysz' -> 'name.fif'
    fif_file_path = file_path.with_name(file_path.name.replace(".pysz", ""))

    if fif_file_path.exists():
        print(f"[Пропуск декомпрессии] Уже существует: {fif_file_path.name}")
        return fif_file_path

    print(f"[*] Декомпрессия: {file_path.name} -> {fif_file_path.name}")
    with open(file_path, "rb") as file_handle:
        payload = pickle.load(file_handle)

    info = payload["info"]
    first_samp = payload["first_samp"]
    n_times = payload["n_times"]
    n_channels = payload["n_channels"]
    chunks = payload["chunks"]

    data_picks = set(mne.pick_types(info, meg=True, eeg=True, eog=True, ecg=True, exclude=[]))
    data = np.zeros((n_channels, n_times), dtype=np.float32)

    for ch_idx, chunk in chunks.items():
        dtype = np.float64 if ch_idx in data_picks else np.float32
        decompressed, _ = sz.decompress(chunk, dtype, (n_times,))
        data[ch_idx] = decompressed

    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)
    raw.save(fif_file_path, overwrite=True)
    raw.close()
    del data, payload, raw
    return fif_file_path


def fif_preprocessing(file_path: Path) -> Path:
    """tSSS и проекторы только для биологических сигналов (Resting State)."""
    # Пустые комнаты не требуют tSSS, фильтрации и проекторов
    if "shared_empty_rooms" in str(file_path) or "type2" in str(file_path):
        print(f"[Инфо] Для {file_path.name} tSSS и проекторы не требуются.")
        return file_path

    out_preprocessed = file_path.with_name(file_path.stem + "_preprocessed.fif")
    if out_preprocessed.exists():
        print(f"[Пропуск предобработки] Уже существует: {out_preprocessed.name}")
        return out_preprocessed

    print(f"[*] tSSS и предобработка: {file_path.name} -> {out_preprocessed.name}")
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    # 1. Поиск плохих каналов для Maxwell
    bads, flats = mne.preprocessing.find_bad_channels_maxwell(raw, verbose=False)
    raw.info["bads"] = list(set(raw.info["bads"] + bads + flats))

    # 2. tSSS
    raw = maxwell_filter(raw, st_duration=10.0, st_correlation=0.900, verbose=False)

    # 3. Фильтр 1-45 Гц
    raw.filter(l_freq=1.0, h_freq=45.0, method='fir', phase='zero', fir_window='hamming', n_jobs=1, verbose=False)

    # 4. Проекторы ECG / EOG (с защитой от падений при отсутствии пиков)
    try:
        proj_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, reject=None, no_proj=True, verbose=False)
        raw.add_proj(proj_ecg)
    except Exception:
        pass

    try:
        proj_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0, reject=None, no_proj=True, verbose=False)
        raw.add_proj(proj_eog)
    except Exception:
        pass

    raw.apply_proj()

    raw.save(out_preprocessed, overwrite=True)
    raw.close()
    del raw
    return out_preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PreprocessPipeline')
    parser.add_argument('filecode', type=int, help='Индекс файла в общем пуле')
    args = parser.parse_args()

    # Собираем файлы из двух папок: 18 resting + 19 empty room = 37 файлов
    resting_files = sorted(glob.glob(str(RESTING_DIR / "*.fif.pysz")))
    empty_room_files = sorted(glob.glob(str(EMPTY_ROOMS_DIR / "*.fif.pysz")))
    all_files = resting_files + empty_room_files

    if args.filecode >= len(all_files):
        print(f"[Ошибка] filecode {args.filecode} >= {len(all_files)}")
        exit(1)

    target_file = Path(all_files[args.filecode])
    print(f"=== Задача #{args.filecode}/{len(all_files)-1} | Файл: {target_file.name} ===")

    fif_path = pysz_decompression(target_file)
    fif_preprocessing(fif_path)
    print(f"[+] Успешно завершено: {fif_path.name}")
