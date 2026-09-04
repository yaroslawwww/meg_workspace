#!/usr/bin/env python3
import gc
import logging
from pathlib import Path
import pickle
import time
import mne
import numpy as np
from pysz import sz

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("VERIFY_PERMUTATIONS")


def calc_permutation_match(
    sig_orig: np.ndarray,
    sig_decomp: np.ndarray,
    m: int = 3,
    tau: int = 5,
) -> float:
    """Вычисляет процент точно совпадающих ординарных перестановок (рангов)."""
    n_times = len(sig_orig)
    n_vectors = n_times - (m - 1) * tau
    if n_vectors <= 0:
        return 0.0

    idx = np.arange(n_vectors)[:, None] + np.arange(m)[None, :] * tau
    vec_orig = sig_orig[idx]
    vec_decomp = sig_decomp[idx]

    ranks_orig = np.argsort(vec_orig, axis=1)
    ranks_decomp = np.argsort(vec_decomp, axis=1)

    matches = np.all(ranks_orig == ranks_decomp, axis=1)
    return float(np.mean(matches) * 100.0)


def verify_20_random_channels(src_fif_path: Path, pysz_path: Path, n_select: int = 20) -> dict:
    """Проверяет 20 случайных каналов для одной пары (FIF, PYSZ)."""
    if not src_fif_path.exists() or not pysz_path.exists():
        logger.warning(f"[Пропуск] Файл не найден: {src_fif_path.name} или {pysz_path.name}")
        return {}

    print("=" * 75)
    print(f" ПРОВЕРКА ПЕРЕСТАНОВОК ПО {n_select} СЛУЧАЙНЫМ КАНАЛАМ")
    print(f" Исходник FIF:  {src_fif_path.name}")
    print(f" Архив PYSZ:    {pysz_path.name}")
    print("=" * 75)

    raw_orig = mne.io.read_raw_fif(src_fif_path, preload=False, verbose=False)
    sfreq = raw_orig.info["sfreq"]
    ch_names = raw_orig.ch_names
    ch_types = raw_orig.get_channel_types()

    data_picks = mne.pick_types(raw_orig.info, meg=True, eeg=True, eog=True, ecg=True, exclude=[])

    np.random.seed(42)
    selected_picks = np.sort(
        np.random.choice(data_picks, size=min(n_select, len(data_picks)), replace=False)
    )

    # 1. Загрузка и фильтрация 20 каналов из FIF
    orig_20_raw = raw_orig.get_data(picks=selected_picks)
    orig_20_filtered = mne.filter.filter_data(
        orig_20_raw,
        sfreq=sfreq,
        l_freq=1.0,
        h_freq=45.0,
        method="fir",
        phase="zero",
        fir_window="hamming",
        n_jobs=4,
        verbose=False,
    ).astype(np.float32)

    del orig_20_raw
    gc.collect()

    # 2. Декомпрессия 20 каналов из PYSZ
    with open(pysz_path, "rb") as f:
        payload = pickle.load(f)

    n_times = int(payload["n_times"])
    chunks = payload["chunks"]

    decomp_20_data = np.empty((len(selected_picks), n_times), dtype=np.float32)
    for i, ch_idx in enumerate(selected_picks):
        c_bytes = chunks[ch_idx]
        decomp, _ = sz.decompress(c_bytes, np.float32, (n_times,))
        decomp_20_data[i, :] = decomp

    del payload, chunks
    gc.collect()

    # 3. Расчет точности перестановок
    test_params = [(3, 1), (3, 5), (5, 5)]
    file_results = {p: [] for p in test_params}

    print("-" * 75)
    print(f"{'Канал':<12} | {'Тип':<6} | {'Max ABS Ошибка':<15} | {'m=3, tau=1':<10} | {'m=3, tau=5':<10} | {'m=5, tau=5':<10}")
    print("-" * 75)

    for i, ch_idx in enumerate(selected_picks):
        c_name = ch_names[ch_idx]
        c_type = ch_types[ch_idx]

        orig_sig = orig_20_filtered[i]
        decomp_sig = decomp_20_data[i]

        max_err = float(np.max(np.abs(orig_sig - decomp_sig)))

        res_row = []
        for m, tau in test_params:
            pct = calc_permutation_match(orig_sig, decomp_sig, m=m, tau=tau)
            file_results[(m, tau)].append(pct)
            res_row.append(pct)

        print(
            f"{c_name:<12} | {c_type:<6} | {max_err:<15.3e} | "
            f"{res_row[0]:6.2f}%    | {res_row[1]:6.2f}%    | {res_row[2]:6.2f}%"
        )

    print("-" * 75)
    print(" СРЕДНИЕ ПОКАЗАТЕЛИ ДЛЯ ДАННОГО ФАЙЛА:")
    for m, tau in test_params:
        avg_p = np.mean(file_results[(m, tau)])
        print(f"  * m={m}, tau={tau:2d} -> Среднее: {avg_p:6.2f}%")
    print("=" * 75 + "\n")

    return file_results


if __name__ == "__main__":
    WORKSPACE_DIR = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parent
    RAW_DATA_DIR = WORKSPACE_DIR / "01_raw_data"

    CONTAINER_RAW_PATH = Path("/raw_data")
    HOST_RAW_PATH = Path("~/git-reps/EEGs/EEG_final/eeg_pipeline/raw_data").expanduser()
    SOURCE_RAW_ROOT = CONTAINER_RAW_PATH if CONTAINER_RAW_PATH.exists() else HOST_RAW_PATH

    # --- СПИСОК ТЕСТОВЫХ ПАР ДЛЯ ВСЕХ ТРЕХ ТИПОВ ДАТАСЕТОВ ---
    test_cases = {
        "ТИП 1 (Биграммы ВШЭ)": [
            (
                SOURCE_RAW_ROOT / "MEG/20.08.24/2/240820_gromov/main.fif",
                RAW_DATA_DIR / "type1_bigrams/type1_sub00_gromov_vasily_raw.fif.pysz",
            ),
            (
                SOURCE_RAW_ROOT / "MEG/03.09.24/2/kirillova_agatha/main.fif",
                RAW_DATA_DIR / "type1_bigrams/type1_sub04_kirillova_agatha_raw.fif.pysz",
            ),
            (
                SOURCE_RAW_ROOT / "MEG/02.10.24/2/nikolaev_evgeniy/main.fif",
                RAW_DATA_DIR / "type1_bigrams/type1_sub07_nikolaev_evgeniy_raw.fif.pysz",
            ),
        ],
        "ТИП 2 (Микросон / Ушаков)": [
            (
                SOURCE_RAW_ROOT / "МЭГ обработанные/Данные МЭГ/bainbridge_emily_220519_raw_tsss.fif",
                RAW_DATA_DIR / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif.pysz",
            ),
            (
                SOURCE_RAW_ROOT / "МЭГ обработанные/Данные МЭГ/fadeev_kirill_220407_raw_tsss.fif",
                RAW_DATA_DIR / "type2_drowsiness/type2_fadeev_kirill_220407_raw.fif.pysz",
            ),
            (
                SOURCE_RAW_ROOT / "МЭГ обработанные/Данные МЭГ/roslyakova_sophia_220602_raw_tsss.fif",
                RAW_DATA_DIR / "type2_drowsiness/type2_roslyakova_sophia_220602_raw.fif.pysz",
            ),
        ],
        "ТИП 3 (SpanishBCBL)": [
            (
                SOURCE_RAW_ROOT / "SpanishBCBL/MEG/FIF/01_9228/220404/Block1.fif",
                RAW_DATA_DIR / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw.fif.pysz",
            ),
            (
                SOURCE_RAW_ROOT / "SpanishBCBL/MEG/FIF/02_2810/220412/block1.fif",
                RAW_DATA_DIR / "type3_innerspeech/type3_bcbl_02_2810_220412_block1_raw.fif.pysz",
            ),
            (
                SOURCE_RAW_ROOT / "SpanishBCBL/MEG/FIF/03_11123/230313/block1.fif",
                RAW_DATA_DIR / "type3_innerspeech/type3_bcbl_03_11123_230313_block1_raw.fif.pysz",
            ),
        ],
    }

    # --- ЗАПУСК МНОГОФАЙЛОВОЙ ВАЛИДАЦИИ ---
    print("\n" + "#" * 75)
    print(" СТАРТ ГЛОБАЛЬНОЙ ПРОВЕРКИ ТОЧНОСТИ ПЕРЕСТАНОВОК ПО ВСЕМ ТИПАМ ДАННЫХ")
    print("#" * 75 + "\n")

    summary_stats = {}

    for data_type, pairs in test_cases.items():
        print(f"\n>>> ОБРАБОТКА КАТЕГОРИИ: {data_type} <<<\n")
        type_results = {(3, 1): [], (3, 5): [], (5, 5): []}
        files_count = 0

        for src_fif, pysz_file in pairs:
            res = verify_20_random_channels(src_fif, pysz_file, n_select=20)
            if res:
                files_count += 1
                for param, pcts in res.items():
                    type_results[param].extend(pcts)

        if files_count > 0:
            summary_stats[data_type] = {
                param: (np.mean(vals), np.min(vals))
                for param, vals in type_results.items()
            }

    # --- ИТОГОВЫЙ СВОДНЫЙ ОТЧЕТ ---
    print("\n" + "=" * 75)
    print(" ГЛОБАЛЬНЫЙ ИТОГОВЫЙ ОТЧЕТ СОХРАНЕНИЯ ПЕРЕСТАНОВОК ПО ВСЕМ ТИПАМ")
    print("=" * 75)

    for data_type, stats in summary_stats.items():
        print(f"\nКатегория: {data_type}:")
        for (m, tau), (avg_val, min_val) in stats.items():
            print(f"  * m={m}, tau={tau:2d} -> Среднее совпадение: {avg_val:6.2f}% (Худший канал: {min_val:6.2f}%)")

    print("\n" + "=" * 75)
    print(" ВАЛИДАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 75 + "\n")
