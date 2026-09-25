#!/usr/bin/env python3

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT = WORKSPACE_ROOT / "01_raw_data"
PROCESSED_DATABASE_ROOT = WORKSPACE_ROOT / "04_processed_db"
ENGINE_SCRIPT_PATH = WORKSPACE_ROOT / "week_3_QCML_graphs" / "qcml_meg_engine_v3.py"

PHYSICAL_CHANNEL_COUNT = 306
SYNTHETIC_SEED = 42

BATCH_SIZE_FOR_ESTIMATE = 2048
TARGET_GRADIENT_STEPS = 40_000

DATASET_LOCATIONS = {
    "t1_sub00": ("type1_bigrams/type1_sub00_gromov_vasily_raw_preprocessed.fif", "type1_features", "fif"),
    "t1_sub01": ("type1_bigrams/type1_sub01_isaeva_valeria_raw_preprocessed.fif", "type1_features", "fif"),
    "t1_sub04": ("type1_bigrams/type1_sub04_kirillova_agatha_raw_preprocessed.fif", "type1_features", "fif"),
    "t1_sub07": ("type1_bigrams/type1_sub07_nikolaev_evgeniy_raw_preprocessed.fif", "type1_features", "fif"),
    "t1_sub10": ("type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif", "type1_features", "fif"),
    "rest_sub00": ("resting_state/type1_sub00_resting_gromov_vasily_raw_preprocessed.fif", "resting_features", "fif"),
    "rest_sub01": ("resting_state/type1_sub01_resting_isaeva_valeria_raw_preprocessed.fif", "resting_features", "fif"),
    "rest_sub04": ("resting_state/type1_sub04_resting_kirillova_agatha_raw_preprocessed.fif", "resting_features", "fif"),
    "rest_sub07": ("resting_state/type1_sub07_resting_nikolaev_evgeniy_raw_preprocessed.fif", "resting_features", "fif"),
    "rest_sub10": ("resting_state/type1_sub10_resting_chistova_alena_raw_preprocessed.fif", "resting_features", "fif"),
    "t2_emily": ("type2_drowsiness/type2_bainbridge_emily_220519_raw.fif", "type2_features", "fif"),
    "t2_kostya": ("type2_drowsiness/type2_chastkov_konstantin_220310_raw.fif", "type2_features", "fif"),
    "t2_diana": ("type2_drowsiness/type2_bakirova_diana_201210_raw.fif", "type2_features", "fif"),
    "t2_evgenia": ("type2_drowsiness/type2_gandina_evgenia_220428_raw.fif", "type2_features", "fif"),
    "t3_bcbl01": ("type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw_preprocessed.fif", "type3_features", "fif"),
    "er_1510": ("shared_empty_rooms/er_151024_raw.fif", "empty_room_features", "fif"),
    "er_2008": ("shared_empty_rooms/er_200824_raw.fif", "empty_room_features", "fif"),
    "syn_sphere_clean":    ("synthetic/syn_sphere_clean.npz",    "synthetic_features", "npz"),
    "syn_sphere_noise_05": ("synthetic/syn_sphere_noise_05.npz", "synthetic_features", "npz"),
    "syn_sphere_noise_10": ("synthetic/syn_sphere_noise_10.npz", "synthetic_features", "npz"),
    "syn_sphere_noise_20": ("synthetic/syn_sphere_noise_20.npz", "synthetic_features", "npz"),
    "syn_sphere_noise_50": ("synthetic/syn_sphere_noise_50.npz", "synthetic_features", "npz"),
    "syn_torus":           ("synthetic/syn_torus.npz",           "synthetic_features", "npz"),
    "syn_swiss_roll":      ("synthetic/syn_swiss_roll.npz",      "synthetic_features", "npz"),
    "syn_s_curve":         ("synthetic/syn_s_curve.npz",         "synthetic_features", "npz"),
    "syn_circle":          ("synthetic/syn_circle.npz",          "synthetic_features", "npz"),
    "syn_figure8":         ("synthetic/syn_figure8.npz",         "synthetic_features", "npz"),
}

REAL_DATASETS = tuple(key for key, entry in DATASET_LOCATIONS.items() if entry[2] == "fif")
SYNTHETIC_DATASETS = tuple(key for key, entry in DATASET_LOCATIONS.items() if entry[2] == "npz")

REFERENCE_DATASETS = ("t1_sub10", "rest_sub10", "er_1510", "t2_emily")
DETACH_COMPARISON_DATASETS = ("t1_sub10", "rest_sub10", "er_1510")

SYNTHETIC_REFERENCE_DATASETS = (
    "syn_sphere_clean",
    "syn_sphere_noise_20",
    "syn_torus",
    "syn_swiss_roll",
)

TAKENS_TAU_VALUES_MS = (10.0, 25.0, 50.0)
VARIANCE_WEIGHT_VALUES = (0.005, 0.01)
NORMALIZATION_MODES = ("block_rms", "channel_zscore")

HILBERT_DIMENSION_SCAN_M1 = (20, 24, 28, 32, 36, 40, 48, 56, 64)
HILBERT_DIMENSION_SCAN_M6 = (24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 80)
SYNTHETIC_HILBERT_SCAN = (8, 12, 16, 24, 32, 48)

BASELINE_HILBERT_DIMENSION = 48
BASELINE_TAKENS_EMBEDDING = 6
BASELINE_TAKENS_TAU_MS = 25.0
BASELINE_NORMALIZATION_MODE = "block_rms"
BASELINE_VARIANCE_WEIGHT = 0.005

SAMPLES_BY_DATASET = {
    "t1_sub00": 4_886_875,
    "t1_sub01": 1_867_875,
    "t1_sub04": 1_567_875,
    "t1_sub07": 1_221_875,
    "t1_sub10": 2_542_875,
    "rest_sub00": 237_875,
    "rest_sub01": 183_875,
    "rest_sub04": 179_875,
    "rest_sub07": 182_875,
    "rest_sub10": 191_875,
    "t2_emily": 720_175,
    "t2_kostya": 708_975,
    "t2_diana": 721_575,
    "t2_evgenia": 684_775,
    "t3_bcbl01": 1_513_245,
    "er_1510": 119_875,
    "er_2008": 319_875,
    "syn_sphere_clean": 2000,
    "syn_sphere_noise_05": 2000,
    "syn_sphere_noise_10": 2000,
    "syn_sphere_noise_20": 2000,
    "syn_sphere_noise_50": 2000,
    "syn_torus": 3000,
    "syn_swiss_roll": 3000,
    "syn_s_curve": 3000,
    "syn_circle": 2000,
    "syn_figure8": 2000,
}

SYNTHETIC_FEATURE_DIMENSION = {
    "syn_sphere_clean": 3,
    "syn_sphere_noise_05": 3,
    "syn_sphere_noise_10": 3,
    "syn_sphere_noise_20": 3,
    "syn_sphere_noise_50": 3,
    "syn_torus": 3,
    "syn_swiss_roll": 3,
    "syn_s_curve": 3,
    "syn_circle": 2,
    "syn_figure8": 2,
}

REFERENCE_GRADIENT_STEPS = 40_000
REFERENCE_TAKENS_M = 6
REFERENCE_HILBERT_DIMENSION = 48
REFERENCE_DURATION_SECONDS = 3000


def generate_sphere_s2(n_samples: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    points = rng.normal(size=(n_samples, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    if noise > 0.0:
        points = points + noise * rng.normal(size=points.shape)
    return points


def generate_torus_t2(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    v = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    major_radius = 1.0
    minor_radius = 0.3
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    return np.stack([x, y, z], axis=1)


def generate_swiss_roll(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(size=n_samples))
    height = 21.0 * rng.uniform(size=n_samples)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    return np.stack([x, y, z], axis=1)


def generate_s_curve(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = 3.0 * np.pi * (rng.uniform(size=n_samples) - 0.5)
    x = np.sin(t)
    y = 2.0 * rng.uniform(size=n_samples)
    z = np.sign(t) * (np.cos(t) - 1.0)
    return np.stack([x, y, z], axis=1)


def generate_circle_s1(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def generate_figure8(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    return np.stack([x, y], axis=1)


SYNTHETIC_GENERATORS = {
    "sphere_s2": generate_sphere_s2,
    "torus_t2": generate_torus_t2,
    "swiss_roll": generate_swiss_roll,
    "s_curve": generate_s_curve,
    "circle_s1": generate_circle_s1,
    "figure8": generate_figure8,
}

SYNTHETIC_RECIPES = {
    "syn_sphere_clean":    {"generator": "sphere_s2",  "n_samples": 2000, "noise": 0.0},
    "syn_sphere_noise_05": {"generator": "sphere_s2",  "n_samples": 2000, "noise": 0.05},
    "syn_sphere_noise_10": {"generator": "sphere_s2",  "n_samples": 2000, "noise": 0.10},
    "syn_sphere_noise_20": {"generator": "sphere_s2",  "n_samples": 2000, "noise": 0.20},
    "syn_sphere_noise_50": {"generator": "sphere_s2",  "n_samples": 2000, "noise": 0.50},
    "syn_torus":           {"generator": "torus_t2",   "n_samples": 3000},
    "syn_swiss_roll":      {"generator": "swiss_roll", "n_samples": 3000},
    "syn_s_curve":         {"generator": "s_curve",    "n_samples": 3000},
    "syn_circle":          {"generator": "circle_s1",  "n_samples": 2000},
    "syn_figure8":         {"generator": "figure8",    "n_samples": 2000},
}


def source_path_for_dataset(dataset_key: str) -> Path:
    relative_path, _, _ = DATASET_LOCATIONS[dataset_key]
    return RAW_DATA_ROOT / relative_path


def output_directory_for_dataset(dataset_key: str) -> Path:
    _, output_folder, _ = DATASET_LOCATIONS[dataset_key]
    return PROCESSED_DATABASE_ROOT / output_folder


def source_kind_for_dataset(dataset_key: str) -> str:
    _, _, kind = DATASET_LOCATIONS[dataset_key]
    return kind


def generate_synthetic_dataset(dataset_key: str) -> Path:
    recipe = SYNTHETIC_RECIPES[dataset_key]
    output_path = source_path_for_dataset(dataset_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    generator = SYNTHETIC_GENERATORS[recipe["generator"]]
    if "noise" in recipe:
        features = generator(recipe["n_samples"], rng, recipe["noise"])
    else:
        features = generator(recipe["n_samples"], rng)
    np.savez(output_path, features=features)
    print(f"[+] Синтетика сгенерирована: {output_path} | форма = {features.shape}")
    return output_path


def ensure_synthetic_source(dataset_key: str) -> Path:
    path = source_path_for_dataset(dataset_key)
    if path.exists():
        return path
    return generate_synthetic_dataset(dataset_key)


def feature_dimension_for_task(dataset_key: str, takens_embedding: int) -> int:
    if source_kind_for_dataset(dataset_key) == "npz":
        return SYNTHETIC_FEATURE_DIMENSION[dataset_key]
    return PHYSICAL_CHANNEL_COUNT * takens_embedding


def task_identifier(
    dataset_key: str,
    hilbert_dimension: int,
    takens_embedding: int,
    takens_tau_ms: float,
    normalization_mode: str,
    variance_weight: float,
    detach_eigenvectors: bool,
) -> tuple:
    return (
        dataset_key,
        hilbert_dimension,
        takens_embedding,
        takens_tau_ms,
        normalization_mode,
        variance_weight,
        detach_eigenvectors,
    )


def describe_task(
    dataset_key: str,
    hilbert_dimension: int,
    takens_embedding: int,
    takens_tau_ms: float,
    normalization_mode: str,
    variance_weight: float,
    detach_eigenvectors: bool,
) -> str:
    gradient_tag = "detach" if detach_eigenvectors else "grad"
    return (
        f"{dataset_key}_N{hilbert_dimension}_m{takens_embedding}"
        f"_tau{int(takens_tau_ms)}_{normalization_mode}_w{variance_weight}_{gradient_tag}"
    )


def append_task(
    tasks: list[dict[str, Any]],
    seen: set,
    dataset_key: str,
    hilbert_dimension: int,
    takens_embedding: int,
    takens_tau_ms: float,
    normalization_mode: str,
    variance_weight: float,
    detach_eigenvectors: bool,
) -> None:
    identifier = task_identifier(
        dataset_key,
        hilbert_dimension,
        takens_embedding,
        takens_tau_ms,
        normalization_mode,
        variance_weight,
        detach_eigenvectors,
    )
    if identifier in seen:
        return
    seen.add(identifier)
    kind = source_kind_for_dataset(dataset_key)
    synthetic_label = dataset_key if kind == "npz" else None
    tasks.append({
        "dataset_key": dataset_key,
        "source_kind": kind,
        "source_path": source_path_for_dataset(dataset_key),
        "output_directory": output_directory_for_dataset(dataset_key),
        "feature_dimension": feature_dimension_for_task(dataset_key, takens_embedding),
        "synthetic_label": synthetic_label,
        "n_hilbert": hilbert_dimension,
        "takens_m": takens_embedding,
        "takens_tau_ms": takens_tau_ms,
        "norm_mode": normalization_mode,
        "w_variance": variance_weight,
        "detach_eigenvectors": detach_eigenvectors,
        "description": describe_task(
            dataset_key,
            hilbert_dimension,
            takens_embedding,
            takens_tau_ms,
            normalization_mode,
            variance_weight,
            detach_eigenvectors,
        ),
    })


def add_baseline_cohort_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in REAL_DATASETS:
        append_task(
            tasks, seen,
            dataset_key=dataset_key,
            hilbert_dimension=BASELINE_HILBERT_DIMENSION,
            takens_embedding=BASELINE_TAKENS_EMBEDDING,
            takens_tau_ms=BASELINE_TAKENS_TAU_MS,
            normalization_mode=BASELINE_NORMALIZATION_MODE,
            variance_weight=BASELINE_VARIANCE_WEIGHT,
            detach_eigenvectors=False,
        )


def add_zscore_cohort_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in REAL_DATASETS:
        append_task(
            tasks, seen,
            dataset_key=dataset_key,
            hilbert_dimension=BASELINE_HILBERT_DIMENSION,
            takens_embedding=BASELINE_TAKENS_EMBEDDING,
            takens_tau_ms=BASELINE_TAKENS_TAU_MS,
            normalization_mode="channel_zscore",
            variance_weight=BASELINE_VARIANCE_WEIGHT,
            detach_eigenvectors=False,
        )


def add_synthetic_baseline_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in SYNTHETIC_DATASETS:
        append_task(
            tasks, seen,
            dataset_key=dataset_key,
            hilbert_dimension=BASELINE_HILBERT_DIMENSION,
            takens_embedding=1,
            takens_tau_ms=0.0,
            normalization_mode=BASELINE_NORMALIZATION_MODE,
            variance_weight=BASELINE_VARIANCE_WEIGHT,
            detach_eigenvectors=False,
        )


def add_synthetic_hilbert_scan_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in SYNTHETIC_REFERENCE_DATASETS:
        for hilbert_dimension in SYNTHETIC_HILBERT_SCAN:
            append_task(
                tasks, seen,
                dataset_key=dataset_key,
                hilbert_dimension=hilbert_dimension,
                takens_embedding=1,
                takens_tau_ms=0.0,
                normalization_mode=BASELINE_NORMALIZATION_MODE,
                variance_weight=BASELINE_VARIANCE_WEIGHT,
                detach_eigenvectors=False,
            )


def add_normalization_comparison_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in REFERENCE_DATASETS:
        for normalization_mode in NORMALIZATION_MODES:
            append_task(
                tasks, seen,
                dataset_key=dataset_key,
                hilbert_dimension=BASELINE_HILBERT_DIMENSION,
                takens_embedding=BASELINE_TAKENS_EMBEDDING,
                takens_tau_ms=BASELINE_TAKENS_TAU_MS,
                normalization_mode=normalization_mode,
                variance_weight=BASELINE_VARIANCE_WEIGHT,
                detach_eigenvectors=False,
            )


def add_takens_delay_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in REFERENCE_DATASETS:
        for takens_tau_ms in TAKENS_TAU_VALUES_MS:
            append_task(
                tasks, seen,
                dataset_key=dataset_key,
                hilbert_dimension=BASELINE_HILBERT_DIMENSION,
                takens_embedding=BASELINE_TAKENS_EMBEDDING,
                takens_tau_ms=takens_tau_ms,
                normalization_mode=BASELINE_NORMALIZATION_MODE,
                variance_weight=BASELINE_VARIANCE_WEIGHT,
                detach_eigenvectors=False,
            )


def add_variance_weight_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in REFERENCE_DATASETS:
        for variance_weight in VARIANCE_WEIGHT_VALUES:
            append_task(
                tasks, seen,
                dataset_key=dataset_key,
                hilbert_dimension=BASELINE_HILBERT_DIMENSION,
                takens_embedding=BASELINE_TAKENS_EMBEDDING,
                takens_tau_ms=BASELINE_TAKENS_TAU_MS,
                normalization_mode=BASELINE_NORMALIZATION_MODE,
                variance_weight=variance_weight,
                detach_eigenvectors=False,
            )


def add_hilbert_scan_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    scan_specifications = (
        (1, HILBERT_DIMENSION_SCAN_M1),
        (BASELINE_TAKENS_EMBEDDING, HILBERT_DIMENSION_SCAN_M6),
    )
    for dataset_key in REFERENCE_DATASETS:
        for takens_embedding, dimension_values in scan_specifications:
            for hilbert_dimension in dimension_values:
                append_task(
                    tasks, seen,
                    dataset_key=dataset_key,
                    hilbert_dimension=hilbert_dimension,
                    takens_embedding=takens_embedding,
                    takens_tau_ms=BASELINE_TAKENS_TAU_MS if takens_embedding > 1 else 0.0,
                    normalization_mode=BASELINE_NORMALIZATION_MODE,
                    variance_weight=BASELINE_VARIANCE_WEIGHT,
                    detach_eigenvectors=False,
                )


def add_detach_comparison_tasks(tasks: list[dict[str, Any]], seen: set) -> None:
    for dataset_key in DETACH_COMPARISON_DATASETS:
        for detach_eigenvectors in (False, True):
            append_task(
                tasks, seen,
                dataset_key=dataset_key,
                hilbert_dimension=BASELINE_HILBERT_DIMENSION,
                takens_embedding=BASELINE_TAKENS_EMBEDDING,
                takens_tau_ms=BASELINE_TAKENS_TAU_MS,
                normalization_mode=BASELINE_NORMALIZATION_MODE,
                variance_weight=BASELINE_VARIANCE_WEIGHT,
                detach_eigenvectors=detach_eigenvectors,
            )


def build_task_grid() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set = set()
    add_baseline_cohort_tasks(tasks, seen)
    add_zscore_cohort_tasks(tasks, seen)
    add_synthetic_baseline_tasks(tasks, seen)
    add_synthetic_hilbert_scan_tasks(tasks, seen)
    add_normalization_comparison_tasks(tasks, seen)
    add_takens_delay_tasks(tasks, seen)
    add_variance_weight_tasks(tasks, seen)
    add_hilbert_scan_tasks(tasks, seen)
    add_detach_comparison_tasks(tasks, seen)
    return tasks
BATCH_SIZE_DEFAULT = 2048
BASE_GRADIENT_STEPS = 40_000
SYNTHETIC_EPOCHS = 300


def gradient_steps_for_task(task: dict[str, Any]) -> int:
    if task["source_kind"] == "npz":
        samples = SAMPLES_BY_DATASET.get(task["dataset_key"], 2000)
        batches_per_epoch = max(1, math.ceil(samples / BATCH_SIZE_DEFAULT))
        return SYNTHETIC_EPOCHS * batches_per_epoch

    hilbert_dimension = task["n_hilbert"]
    takens_embedding = task["takens_m"]
    scale_hilbert = hilbert_dimension / float(BASELINE_HILBERT_DIMENSION)
    scale_takens = math.sqrt(takens_embedding / float(BASELINE_TAKENS_EMBEDDING))
    return int(BASE_GRADIENT_STEPS * scale_hilbert * scale_takens)

def estimate_task_duration_seconds(task: dict[str, Any]) -> int:
    steps = gradient_steps_for_task(task)
    reference_cost = REFERENCE_GRADIENT_STEPS * REFERENCE_TAKENS_M * REFERENCE_HILBERT_DIMENSION ** 2
    task_cost = steps * task["takens_m"] * task["n_hilbert"] ** 2
    return int(REFERENCE_DURATION_SECONDS * task_cost / reference_cost)


def build_grouped_task_assignments(
    tasks: list[dict[str, Any]], tasks_per_group: int
) -> list[list[int]]:
    if tasks_per_group < 1:
        tasks_per_group = 1
    sorted_indices = sorted(
        range(len(tasks)),
        key=lambda index: estimate_task_duration_seconds(tasks[index]),
        reverse=True,
    )
    group_count = math.ceil(len(tasks) / tasks_per_group)
    groups: list[list[int]] = [[] for _ in range(group_count)]
    for rank, task_index in enumerate(sorted_indices):
        groups[rank % group_count].append(task_index)
    return groups


def group_duration_seconds(tasks: list[dict[str, Any]], group: list[int]) -> int:
    return sum(estimate_task_duration_seconds(tasks[i]) for i in group)


def expected_suffix_tag(task: dict[str, Any]) -> str:
    gradient_tag = "detach" if task["detach_eigenvectors"] else "grad"
    return (
        f"D{task['feature_dimension']}"
        f"_N{task['n_hilbert']}_m{task['takens_m']}"
        f"_tau{int(task['takens_tau_ms'])}ms_{task['norm_mode']}"
        f"_w{task['w_variance']}_{gradient_tag}"
    )


def subject_stem_from_task(task: dict[str, Any]) -> str:
    if task["source_kind"] == "npz":
        return task["synthetic_label"]
    stem = task["source_path"].stem
    return stem.replace("_raw_preprocessed", "").replace("_raw", "")


def expected_parquet_path(task: dict[str, Any]) -> Path:
    return task["output_directory"] / f"{subject_stem_from_task(task)}_spectrum_{expected_suffix_tag(task)}.parquet"


def is_task_completed(task: dict[str, Any]) -> bool:
    parquet_path = expected_parquet_path(task)
    return parquet_path.exists() and parquet_path.stat().st_size > 1024


def engine_command_for_task(task: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(ENGINE_SCRIPT_PATH)]
    if task["source_kind"] == "npz":
        command.extend(["--synthetic_npz_path", str(task["source_path"])])
        command.extend(["--synthetic_label", str(task["synthetic_label"])])
        command.extend(["--n_epochs", str(SYNTHETIC_EPOCHS)])
    else:
        command.extend(["--fif_path", str(task["source_path"])])
        target_steps = gradient_steps_for_task(task)
        command.extend(["--target_steps", str(target_steps)])
    command.extend([
        "--n_hilbert", str(task["n_hilbert"]),
        "--takens_m", str(task["takens_m"]),
        "--takens_tau_ms", str(task["takens_tau_ms"]),
        "--norm_mode", str(task["norm_mode"]),
        "--w_variance", str(task["w_variance"]),
        "--batch_size", str(BATCH_SIZE_DEFAULT),
        "--checkpoint_dir", str(task["output_directory"]),
        "--output_dir", str(task["output_directory"]),
    ])
    if task["detach_eigenvectors"]:
        command.append("--detach_eigenvectors")
    return command

def format_duration(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}ч{minutes:02d}м"
    return f"{minutes}м"


def announce_group(
    group_index: int,
    total_groups: int,
    group: list[int],
    tasks: list[dict[str, Any]],
) -> None:
    duration = group_duration_seconds(tasks, group)
    print(f"\n{'#' * 75}")
    print(
        f"# [GROUP {group_index:03d}/{total_groups - 1:03d}] "
        f"задач: {len(group)}, оценка: {format_duration(duration)}"
    )
    print(f"{'#' * 75}")
    for task_index in group:
        task = tasks[task_index]
        task_duration = estimate_task_duration_seconds(task)
        steps = gradient_steps_for_task(task)
        print(
            f"  [{task_index:03d}] {format_duration(task_duration):>6} | "
            f"steps≈{steps:>6} | {task['description']}"
        )
    print()


def announce_task(task: dict[str, Any], task_index: int, total_tasks: int) -> None:
    print(f"\n{'=' * 75}")
    print(f"▶ [QCML v3 TASK {task_index:03d}/{total_tasks:03d}]")
    print(f"  * Датасет: {task['dataset_key']} -> {task['source_path'].name}")
    print(
        f"  * Конфигурация: m={task['takens_m']}, tau={task['takens_tau_ms']}мс, "
        f"N={task['n_hilbert']}, norm={task['norm_mode']}, "
        f"w={task['w_variance']}, detach={task['detach_eigenvectors']}"
    )
    if task["source_kind"] == "fif":
        print(f"  * Бюджет шагов: target_steps = {TARGET_GRADIENT_STEPS}")
    else:
        print(f"  * Эпох: {SYNTHETIC_EPOCHS} (синтетика, малый датасет)")
    print(f"{'=' * 75}\n")


def execute_task(task: dict[str, Any], task_index: int, total_tasks: int) -> int:
    if task["source_kind"] == "npz":
        ensure_synthetic_source(task["dataset_key"])

    if not task["source_path"].exists():
        print(f"[Ошибка] Исходный файл не найден: {task['source_path']}")
        return 1

    if is_task_completed(task):
        print(f"[Пропуск] Задача [{task_index:03d}/{total_tasks:03d}] уже рассчитана: {task['description']}")
        return 0

    announce_task(task, task_index, total_tasks)
    result = subprocess.run(engine_command_for_task(task))
    return result.returncode


def execute_group(
    group_index: int,
    group: list[int],
    tasks: list[dict[str, Any]],
    total_groups: int,
) -> int:
    announce_group(group_index, total_groups, group, tasks)
    last_exit_code = 0
    for task_index in group:
        exit_code = execute_task(tasks[task_index], task_index, len(tasks))
        if exit_code != 0:
            print(f"[!] Задача [{task_index:03d}] завершилась с кодом {exit_code}, продолжаем")
            last_exit_code = exit_code
    return last_exit_code


def print_task_list(tasks: list[dict[str, Any]], groups: list[list[int]]) -> None:
    print(f"Всего сконфигурировано задач: {len(tasks)} | групп: {len(groups)}")
    print()
    for group_index, group in enumerate(groups):
        duration = group_duration_seconds(tasks, group)
        print(
            f"[GROUP {group_index:03d}] задач: {len(group)}, "
            f"оценка: {format_duration(duration)}"
        )
        for task_index in group:
            task = tasks[task_index]
            status = "[ГОТОВО]" if is_task_completed(task) else "[ОЖИДАЕТ]"
            steps = gradient_steps_for_task(task)
            print(f"    [{task_index:03d}] {status} steps≈{steps:>6} {task['description']}")
        print()


def print_task_count(tasks: list[dict[str, Any]], groups: list[list[int]]) -> None:
    completed = sum(1 for task in tasks if is_task_completed(task))
    durations = [group_duration_seconds(tasks, group) for group in groups]
    total_duration = sum(durations)
    max_duration = max(durations) if durations else 0
    real_count = sum(1 for task in tasks if task["source_kind"] == "fif")
    synthetic_count = sum(1 for task in tasks if task["source_kind"] == "npz")
    print(
        f"Всего задач: {len(tasks)} (real: {real_count}, synthetic: {synthetic_count}) | "
        f"Завершено: {completed} | Осталось: {len(tasks) - completed}"
    )
    print(f"Групп: {len(groups)}")
    print(
        f"Суммарная оценка: {format_duration(total_duration)} | "
        f"Максимальная группа: {format_duration(max_duration)}"
    )
    print(
        f"Бюджет шагов: real = {TARGET_GRADIENT_STEPS} | "
        f"synthetic epochs = {SYNTHETIC_EPOCHS}"
    )


def generate_all_synthetic_datasets() -> None:
    for dataset_key in SYNTHETIC_DATASETS:
        ensure_synthetic_source(dataset_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCML v3 grid dispatcher")
    parser.add_argument(
        "group_index",
        nargs="?",
        type=int,
        default=None,
        help="Индекс группы задач в сетке",
    )
    parser.add_argument(
        "--tasks_per_group",
        type=int,
        default=3,
        help="Сколько задач выполняется на один номер SLURM array",
    )
    parser.add_argument("--list", action="store_true", help="Показать список задач и групп")
    parser.add_argument("--count", action="store_true", help="Показать счётчик задач и групп")
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Сгенерировать все синтетические датасеты и выйти",
    )
    return parser.parse_args()


def resolve_group_index(args: argparse.Namespace, total_groups: int) -> int | None:
    if args.group_index is not None:
        return args.group_index
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_array_task_id is not None:
        return int(slurm_array_task_id)
    print(f"Укажите индекс группы (0..{total_groups - 1}) или запустите через sbatch --array.")
    print("Справка по доступным задачам: python qcml_grid_v3.py --list")
    return None


def main() -> None:
    args = parse_args()


    tasks = build_task_grid()
    groups = build_grouped_task_assignments(tasks, args.tasks_per_group)

    if args.list:
        print_task_list(tasks, groups)
        return

    if args.count:
        print_task_count(tasks, groups)
        return

    group_index = resolve_group_index(args, len(groups))
    if group_index is None:
        return

    if group_index < 0 or group_index >= len(groups):
        print(f"[Критическая ошибка] Индекс группы {group_index} вне диапазона [0..{len(groups) - 1}]")
        sys.exit(1)

    exit_code = execute_group(group_index, groups[group_index], tasks, len(groups))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()