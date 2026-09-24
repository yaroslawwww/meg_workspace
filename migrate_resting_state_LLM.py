#!/usr/bin/env python3
"""
Скрипт миграции записей Resting State (МЭГ) в гибридный формат SZ3 (.pysz)
с автоматическим сопоставлением пустых комнат (Empty Rooms)
и генерацией манифеста 02_metadata/manifest_resting.csv.
"""

import csv
import gc
import logging
import os
from pathlib import Path
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

# Ограничение системных потоков для независимых процессов
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import mne
import numpy as np
from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RESTING_MIGRATION")

# Пути к директориям
WORKSPACE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = WORKSPACE_DIR / "01_raw_data"
METADATA_DIR = WORKSPACE_DIR / "02_metadata"
RESTING_DIR = RAW_DATA_DIR / "resting_state"
EMPTY_ROOMS_DIR = RAW_DATA_DIR / "shared_empty_rooms"

for d in [RESTING_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Определение корня сырых данных (контейнер / хост)
CONTAINER_RAW_PATH = Path("/raw_data")
HOST_RAW_PATH = Path("~/git-reps/EEGs/EEG_final/eeg_pipeline/raw_data").expanduser()

SOURCE_RAW_ROOT = CONTAINER_RAW_PATH if CONTAINER_RAW_PATH.exists() else HOST_RAW_PATH
logger.info(f"Корень исходных данных: {SOURCE_RAW_ROOT}")

# Физические границы погрешности SZ3 для сенсоров
ABS_BOUNDS = {
    "mag": 1e-15,   # 1 fT
    "grad": 1e-15,  # 0.2 fT/cm
    "eeg": 1e-9,
    "eog": 1e-9,
    "ecg": 1e-9,
}
DEFAULT_ABS_BOUND = 1e-7


def _compress_single_channel_task(args):
    ch_idx, single_ch, ch_type = args
    abs_bound = ABS_BOUNDS.get(ch_type, DEFAULT_ABS_BOUND)
    cfg = szConfig()
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = abs_bound
    c_bytes, _ = sz.compress(single_ch, cfg)
    return ch_idx, c_bytes


def compress_to_pysz(src_path: Path, dst_pysz_path: Path):
    if dst_pysz_path.exists():
        mb_size = dst_pysz_path.stat().st_size / (1024 * 1024)
        logger.info(f"[Пропуск] Архив уже существует: {dst_pysz_path.name} ({mb_size:.1f} MB)")
        return

    if not src_path.exists():
        logger.error(f"[Ошибка] Исходный файл не найден: {src_path}")
        return

    t0 = time.perf_counter()
    src_size_mb = src_path.stat().st_size / (1024 * 1024)
    logger.info(f"[Старт компрессии Resting] {src_path.name} ({src_size_mb:.1f} MB) -> {dst_pysz_path.name}")

    raw = mne.io.read_raw_fif(src_path, preload=False, verbose=False)
    n_channels = len(raw.ch_names)
    n_times = int(raw.n_times)
    ch_types = raw.get_channel_types()

    data_picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, ecg=True, exclude=[])
    pass_picks = np.array([i for i in range(n_channels) if i not in data_picks])

    compressed_chunks = {}

    # 1. Служебные каналы (STIM, триггеры) без потерь
    if len(pass_picks) > 0:
        ch_pass, _ = raw[pass_picks, :]
        cfg_lossless = szConfig()
        cfg_lossless.cmprAlgo = szAlgorithm.LOSSLESS
        for idx, orig_ch_idx in enumerate(pass_picks):
            track_data = ch_pass[idx].astype(np.float32)
            c_bytes, _ = sz.compress(track_data, cfg_lossless)
            compressed_chunks[orig_ch_idx] = c_bytes
        del ch_pass

    # 2. Параллельное сжатие сигнальных каналов (4 воркера)
    N_JOBS = 4
    BATCH_SIZE = 64
    n_batches = (len(data_picks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(data_picks), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(data_picks))
        batch_chs = data_picks[start:end]
        ch_data, _ = raw[batch_chs, :]

        tasks = [
            (ch_idx, np.ascontiguousarray(ch_data[i]), ch_types[ch_idx])
            for i, ch_idx in enumerate(batch_chs)
        ]

        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            for ch_idx, c_bytes in executor.map(_compress_single_channel_task, tasks):
                compressed_chunks[ch_idx] = c_bytes

        del ch_data, tasks
        gc.collect()

    payload = {
        "info": raw.info,
        "first_samp": raw.first_samp,
        "n_times": n_times,
        "n_channels": n_channels,
        "chunks": compressed_chunks,
    }

    with open(dst_pysz_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    raw.close()
    del compressed_chunks, payload
    gc.collect()

    elapsed = time.perf_counter() - t0
    dst_size_mb = dst_pysz_path.stat().st_size / (1024 * 1024)
    ratio = src_size_mb / dst_size_mb if dst_size_mb > 0 else 1.0
    logger.info(f"[Успех] {dst_pysz_path.name} | {src_size_mb:.1f} MB -> {dst_size_mb:.1f} MB ({ratio:.2f}x) за {elapsed:.1f}c")


def process_resting_migration():
    meg_raw_dir = SOURCE_RAW_ROOT / "MEG"
    manifest_rows = []

    # Точный реестр файлов resting state по структуре папок МЭГ ВШЭ
    # Формат: (session_id, subject_name, относительный путь к fif, привязанная пустая комната)
    resting_registry = [
        # Сессия 20.08.2024
        ("type1_sub00_resting", "gromov_vasily", "20.08.24/2/240820_gromov/resting_state.fif", "er_200824_raw.fif.pysz"),
        ("type1_sub01_resting", "isaeva_valeria", "20.08.24/2/240820_isaeva/resting_state.fif", "er_200824_raw.fif.pysz"),
        ("type1_sub02_resting", "tomaschuk_korney", "20.08.24/2/240820_tomaschuk/resting_state.fif", "er_200824_raw.fif.pysz"),

        # Сессия 03.09.2024
        ("type1_sub04_resting", "kirillova_agatha", "03.09.24/2/kirillova_agatha/resting_state.fif", "er_030924_raw.fif.pysz"),
        ("type1_sub05_resting", "shalamkova_alisa", "03.09.24/2/shalamkova_alisa/resting_state.fif", "er_030924_raw.fif.pysz"),
        ("type1_sub06_resting", "borevskiy_andrey", "03.09.24/2/borevskiy_andrey/resting_state.fif", "er_030924_raw.fif.pysz"),

        # Сессия 02.10.2024
        ("type1_sub07_resting", "nikolaev_evgeniy", "02.10.24/2/nikolaev_evgeniy/resting_state.fif", "er_021024_raw.fif.pysz"),
        ("type1_sub08_resting", "spiridonov_dmitry", "02.10.24/2/spiridonov_dmitry/resting_state.fif", "er_021024_raw.fif.pysz"),
        ("type1_sub09_resting", "ostapets_valeria", "02.10.24/2/ostapets_valeria/resting_state.fif", "er_021024_raw.fif.pysz"),

        # Сессия 15.10.2024 (У Алёны файл назван resting.fif)
        ("type1_sub10_resting", "chistova_alena", "15.10.24/2/chistova_alena/resting.fif", "er_151024_raw.fif.pysz"),

        # Сессия 12.11.2024
        ("type1_sub11_resting", "nedbay_pavel", "last_exps(don't use,please)/12.11.24/nedbay_pavel/241112/resting_state.fif", "er_121124_raw.fif.pysz"),
        ("type1_sub14_resting_1211", "sinitsin_ivan", "last_exps(don't use,please)/12.11.24/sinitsin_ivan/resting_state.fif", "er_121124_raw.fif.pysz"),

        # Сессия 29.10.2024
        ("type1_sub12_resting", "holinov_roman", "last_exps(don't use,please)/29.10.24/holinov_roman/resting_state.fif", "er_291024_raw.fif.pysz"),
        ("type1_sub13_resting", "saldan_anuar", "last_exps(don't use,please)/29.10.24/saldan_anuar/resting_state.fif", "er_291024_raw.fif.pysz"),
        ("type1_sub14_resting_2910", "sinitsin_ivan", "last_exps(don't use,please)/29.10.24/sinitsin_ivan/resting_state.fif", "er_291024_raw.fif.pysz"),

        # Пилотные записи 13.08.2024
        ("type1_pilot00_resting", "gromov_vasily", "13.08.24/2/gromov_vasily/resting_state.fif", "er_200824_raw.fif.pysz"),
        ("type1_pilot01_resting", "kirillova_agata", "13.08.24/2/kirillova_agata/resting_state.fif", "er_200824_raw.fif.pysz"),
        ("type1_pilot02_resting", "tomaschuk_korney", "13.08.24/2/tomaschuk_korney/resting_state.fif", "er_200824_raw.fif.pysz"),
    ]

    logger.info(f"=== Начало миграции Resting State ({len(resting_registry)} файлов) ===")

    for sess_id, subj, rel_fif, er_file in resting_registry:
        src_fif = meg_raw_dir / rel_fif

        if not src_fif.exists():
            logger.warning(f"[Файл не найден] {src_fif}")
            continue

        dst_pysz_name = f"{sess_id}_{subj}_raw.fif.pysz"
        dst_pysz_path = RESTING_DIR / dst_pysz_name

        compress_to_pysz(src_fif, dst_pysz_path)

        er_rel_path = ""
        er_full = EMPTY_ROOMS_DIR / er_file
        if er_full.exists():
            er_rel_path = str(er_full.relative_to(WORKSPACE_DIR))

        manifest_rows.append({
            "session_id": sess_id,
            "data_type": "type1_resting",
            "subject_name": subj,
            "raw_pysz_rel_path": str(dst_pysz_path.relative_to(WORKSPACE_DIR)),
            "events_rel_path": "",
            "empty_room_rel_path": er_rel_path,
            "sfreq": 1000.0,
            "event_source": "none",
            "notes_and_fixes": f"Spontaneous Resting State recording from {rel_fif}",
            "status": "ready"
        })

    manifest_path = METADATA_DIR / "manifest_resting.csv"
    fieldnames = [
        "session_id", "data_type", "subject_name", "raw_pysz_rel_path",
        "events_rel_path", "empty_room_rel_path", "sfreq",
        "event_source", "notes_and_fixes", "status"
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"Миграция Resting State завершена. Записано сессий в манифест: {len(manifest_rows)}")
    logger.info(f"Файл манифеста: {manifest_path}")


if __name__ == "__main__":
    process_resting_migration()
