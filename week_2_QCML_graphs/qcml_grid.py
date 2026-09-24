#!/usr/bin/env python3
"""
run_qcml_grid.py
Диспетчер параллельных задач QCML для Slurm GPU Array.
Выбирает задачу по индексу SLURM_ARRAY_TASK_ID.
"""

import os
import sys
import subprocess
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_DIR / "01_raw_data"
DB_DIR = WORKSPACE_DIR / "04_processed_db"

# Создаем подпапки в базе признаков
(DB_DIR / "type1_features").mkdir(parents=True, exist_ok=True)
(DB_DIR / "type2_features").mkdir(parents=True, exist_ok=True)
(DB_DIR / "type3_features").mkdir(parents=True, exist_ok=True)
(DB_DIR / "resting_features").mkdir(parents=True, exist_ok=True)
(DB_DIR / "empty_room_features").mkdir(parents=True, exist_ok=True)

# 1. Формирование пула файлов
CORE_FILES = {
    # Type 1 Task
    "t1_sub00": (RAW_DIR / "type1_bigrams/type1_sub00_gromov_vasily_raw_preprocessed.fif", DB_DIR / "type1_features"),
    "t1_sub01": (RAW_DIR / "type1_bigrams/type1_sub01_isaeva_valeria_raw_preprocessed.fif", DB_DIR / "type1_features"),
    "t1_sub10": (RAW_DIR / "type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif", DB_DIR / "type1_features"),
    # Type 1 Resting
    "rest_sub00": (RAW_DIR / "resting_state/type1_sub00_resting_gromov_vasily_raw_preprocessed.fif", DB_DIR / "resting_features"),
    "rest_sub01": (RAW_DIR / "resting_state/type1_sub01_resting_isaeva_valeria_raw_preprocessed.fif", DB_DIR / "resting_features"),
    "rest_sub10": (RAW_DIR / "resting_state/type1_sub10_resting_chistova_alena_raw_preprocessed.fif", DB_DIR / "resting_features"),
    # Type 2 Drowsiness
    "t2_emily": (RAW_DIR / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif", DB_DIR / "type2_features"),
    "t2_diana": (RAW_DIR / "type2_drowsiness/type2_bakirova_diana_201210_raw.fif", DB_DIR / "type2_features"),
    "t2_kostya": (RAW_DIR / "type2_drowsiness/type2_chastkov_konstantin_220310_raw.fif", DB_DIR / "type2_features"),
    # Type 3 Inner Speech
    "t3_bcbl01": (RAW_DIR / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw_preprocessed.fif", DB_DIR / "type3_features"),
    "t3_bcbl02": (RAW_DIR / "type3_innerspeech/type3_bcbl_02_2810_220412_block1_raw_preprocessed.fif", DB_DIR / "type3_features"),
    "t3_bcbl03": (RAW_DIR / "type3_innerspeech/type3_bcbl_03_11123_230313_block1_raw_preprocessed.fif", DB_DIR / "type3_features"),
    # Empty Room Control
    "er_1510": (RAW_DIR / "shared_empty_rooms/er_151024_raw.fif", DB_DIR / "empty_room_features")
}

# 2. Построение списка всех задач (Task Grid)
GRID_TASKS = []

# Блок А: Сканирование кривой N для контрольной тройки (Task vs Resting vs Empty Room)
# N = [12, 16, 24, 32, 48, 64, 100]
N_SCAN_CURVE = [12, 16, 24, 32, 48, 64, 100]
SCAN_SUBJECTS = ["t1_sub10", "rest_sub10", "er_1510"]

for n in N_SCAN_CURVE:
    for s_key in SCAN_SUBJECTS:
        fif_p, out_p = CORE_FILES[s_key]
        GRID_TASKS.append({"fif": fif_p, "out": out_p, "N": n, "desc": f"N_SCAN_{s_key}_N{n}"})

# Блок Б: Прогон всех остальных испытуемых на двух опорных точках N = 24 и N = 48
OTHER_SUBJECTS = [k for k in CORE_FILES.keys() if k not in SCAN_SUBJECTS]
for n in [24, 48]:
    for s_key in OTHER_SUBJECTS:
        fif_p, out_p = CORE_FILES[s_key]
        GRID_TASKS.append({"fif": fif_p, "out": out_p, "N": n, "desc": f"CROSS_{s_key}_N{n}"})


def main():
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        if len(sys.argv) > 1:
            task_idx = int(sys.argv[1])
        else:
            print(f"Всего сконфигурировано задач: {len(GRID_TASKS)}")
            for i, t in enumerate(GRID_TASKS):
                print(f"  [{i:02d}] {t['desc']} -> {t['fif'].name} (N={t['N']})")
            return
    else:
        task_idx = int(task_id_str)

    if task_idx < 0 or task_idx >= len(GRID_TASKS):
        print(f"[Ошибка] Некорректный Task ID: {task_idx}. Доступно: 0..{len(GRID_TASKS)-1}")
        sys.exit(1)

    task = GRID_TASKS[task_idx]
    fif_file = task["fif"]
    out_dir = task["out"]
    n_hilbert = task["N"]

    print(f"\n{'='*70}")
    print(f"▶ [GPU ЗАДАЧА {task_idx:02d}/{len(GRID_TASKS)-1}] {task['desc']}")
    print(f"  * Файл: {fif_file}")
    print(f"  * Hilbert N: {n_hilbert}")
    print(f"  * Вывод: {out_dir}")
    print(f"{'='*70}\n")

    if not fif_file.exists():
        print(f"[Критическая ошибка] Файл не найден: {fif_file}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(WORKSPACE_DIR / "week_2_QCML_graphs/qcml_meg_engine_v2.py"),
        "--fif_path", str(fif_file),
        "--n_hilbert", str(n_hilbert),
        "--checkpoint_dir", str(out_dir),
        "--output_dir", str(out_dir)
    ]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
