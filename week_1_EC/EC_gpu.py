#!/usr/bin/env python3
import os
import sys
import math
import logging
from itertools import permutations
from pathlib import Path

import mne
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

# Настройка подробного логирования для анализа в Slurm-логах
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MEG_ENTROPY_GPU_ANALYZER")

mne.set_log_level("ERROR")
torch.set_num_threads(1)

WORKSPACE_DIRECTORY = Path(__file__).resolve().parent.parent
RAW_DATA_DIRECTORY = WORKSPACE_DIRECTORY / "01_raw_data"
PROCESSED_DATABASE_DIRECTORY = WORKSPACE_DIRECTORY / "04_processed_db"

DATASET_CONFIGURATIONS = {
    0: {
        "description": "Type 1 (HSE Bigrams) - Chistova Alena",
        "raw_fif": RAW_DATA_DIRECTORY / "type1_bigrams/type1_sub10_chistova_alena_raw.fif",
        "prep_fif": RAW_DATA_DIRECTORY / "type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif",
        "output_parquet": PROCESSED_DATABASE_DIRECTORY / "type1_features/type1_sub10_chistova_alena_entropy_db.parquet",
        "window_mode": "type1_type3",
    },
    1: {
        "description": "Type 2 (Ushakov Drowsiness) - Bainbridge Emily",
        "raw_fif": RAW_DATA_DIRECTORY / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif",
        "prep_fif": RAW_DATA_DIRECTORY / "type2_drowsiness/type2_bainbridge_emily_220519_raw.fif",
        "output_parquet": PROCESSED_DATABASE_DIRECTORY / "type2_features/type2_bainbridge_emily_entropy_db.parquet",
        "window_mode": "type2",
    },
    2: {
        "description": "Type 3 (Spanish BCBL) - Subject 01 Block 1",
        "raw_fif": RAW_DATA_DIRECTORY / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw.fif",
        "prep_fif": RAW_DATA_DIRECTORY / "type3_innerspeech/type3_bcbl_01_9228_220404_block1_raw_preprocessed.fif",
        "output_parquet": PROCESSED_DATABASE_DIRECTORY / "type3_features/type3_bcbl_01_block1_entropy_db.parquet",
        "window_mode": "type1_type3",
    },
}

EPSILON_SMOOTHING = 1e-13


def log_gpu_memory_snapshot(stage_identifier: str, pytorch_device: torch.device):
    if not torch.cuda.is_available():
        return
    allocated_bytes = torch.cuda.memory_allocated(pytorch_device)
    reserved_bytes = torch.cuda.memory_reserved(pytorch_device)
    max_allocated_bytes = torch.cuda.max_memory_allocated(pytorch_device)
    total_device_memory = torch.cuda.get_device_properties(pytorch_device).total_memory

    logger.info(
        f"--- [GPU MEMORY ANALYSIS: {stage_identifier}] --- | "
        f"Allocated: {allocated_bytes / (1024**2):.2f} MB | "
        f"Reserved: {reserved_bytes / (1024**2):.2f} MB | "
        f"Peak So Far: {max_allocated_bytes / (1024**2):.2f} MB | "
        f"Total VRAM: {total_device_memory / (1024**2):.2f} MB"
    )


def select_permutation_order(window_samples_count: int, coverage_multiplier: int = 5) -> int:
    for candidate_permutation_order in (6, 5, 4, 3):
        if window_samples_count >= coverage_multiplier * math.factorial(candidate_permutation_order):
            return candidate_permutation_order
    return 3


def calculate_exact_normalization_constant_q0(number_of_patterns: int) -> float:
    n = float(number_of_patterns)
    term_first = ((n + 1.0) / n) * math.log2(n + 1.0)
    term_second = math.log2(n)
    term_third = 2.0 * math.log2(2.0 * n)
    denominator = term_first + term_second - term_third
    return float(-2.0 / denominator)


def reference_complexity_entropy_cpu(time_series_window: np.ndarray, permutation_order: int) -> tuple[float, float]:
    subsequences = [time_series_window[i : i + permutation_order] for i in range(len(time_series_window) - permutation_order + 1)]
    subsequences_array = np.array(subsequences)
    all_patterns = list(permutations(range(permutation_order)))
    number_of_patterns = len(all_patterns)

    patterns, counts = np.unique(np.argsort(subsequences_array, axis=1), return_counts=True, axis=0)
    frequencies = np.zeros(number_of_patterns, dtype=np.float64)
    for pattern, count in zip(patterns, counts):
        frequencies[np.all(all_patterns == pattern, axis=1)] = count

    probability_distribution = frequencies / np.sum(frequencies)
    equilibrium_distribution = np.full(number_of_patterns, 1.0 / number_of_patterns)

    def shannon_entropy_cpu(probability_vector):
        return -np.sum(probability_vector * np.log2(probability_vector + EPSILON_SMOOTHING))

    maximum_entropy = np.log2(number_of_patterns)
    normalized_entropy = shannon_entropy_cpu(probability_distribution) / maximum_entropy

    q0 = calculate_exact_normalization_constant_q0(number_of_patterns)
    jensen_shannon_divergence = (
        shannon_entropy_cpu((probability_distribution + equilibrium_distribution) / 2.0)
        - shannon_entropy_cpu(probability_distribution) / 2.0
        - shannon_entropy_cpu(equilibrium_distribution) / 2.0
    )
    statistical_complexity = jensen_shannon_divergence * normalized_entropy * q0
    return float(normalized_entropy), float(statistical_complexity)


def encode_single_channel_permutations(
    channel_signal_1d: torch.Tensor, 
    permutation_order: int
) -> torch.Tensor:
    subsequences = channel_signal_1d.unfold(dimension=-1, size=permutation_order, step=1)
    ranks = torch.argsort(subsequences, dim=-1).to(torch.int32)

    inversion_counts = torch.zeros_like(ranks)
    for index_i in range(permutation_order - 1):
        rank_current = ranks[:, index_i : index_i + 1]
        rank_future = ranks[:, index_i + 1 :]
        inversion_counts[:, index_i] = (rank_current > rank_future).sum(dim=-1)

    factorial_weights = torch.tensor(
        [math.factorial(permutation_order - 1 - index_i) for index_i in range(permutation_order)],
        dtype=torch.int32,
        device=channel_signal_1d.device
    )
    permutation_codes = (inversion_counts * factorial_weights).sum(dim=-1)
    return permutation_codes.to(torch.int32)


def compute_single_channel_multiscale_batched(
    channel_signal_1d: torch.Tensor,
    center_sample_indices_array: np.ndarray,
    window_samples_array: np.ndarray,
    selected_permutation_orders_list: list[int],
    stride_samples_count: int,
    pytorch_device: torch.device,
    center_batch_size_count: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    number_of_centers = len(center_sample_indices_array)
    number_of_windows = len(window_samples_array)

    permutation_entropy_left_matrix = np.empty((number_of_centers, number_of_windows), dtype=np.float32)
    permutation_entropy_right_matrix = np.empty((number_of_centers, number_of_windows), dtype=np.float32)
    statistical_complexity_left_matrix = np.empty((number_of_centers, number_of_windows), dtype=np.float32)
    statistical_complexity_right_matrix = np.empty((number_of_centers, number_of_windows), dtype=np.float32)

    unique_permutation_orders = sorted(list(set(selected_permutation_orders_list)))

    for order_m in unique_permutation_orders:
        logger.info(f"Начало кодирования паттернов для порядка m = {order_m}")
        log_gpu_memory_snapshot(f"Перед кодированием m={order_m}", pytorch_device)

        number_of_patterns = math.factorial(order_m)
        maximum_entropy = math.log2(number_of_patterns)
        uniform_probability = 1.0 / float(number_of_patterns)
        q0 = calculate_exact_normalization_constant_q0(number_of_patterns)
        entropy_equilibrium = -float(number_of_patterns) * (
            uniform_probability * math.log2(uniform_probability + EPSILON_SMOOTHING)
        )

        codes_1d = encode_single_channel_permutations(channel_signal_1d, order_m)
        number_of_codes = len(codes_1d)

        log_gpu_memory_snapshot(f"После кодирования m={order_m} (codes_1d: {number_of_codes * 4 / 1024**2:.2f} MB)", pytorch_device)

        cumulative_counts = torch.zeros((number_of_patterns, number_of_codes + 1), dtype=torch.int32, device=pytorch_device)
        for pattern_idx in range(number_of_patterns):
            mask = (codes_1d == pattern_idx).to(torch.int32)
            cumulative_counts[pattern_idx, 1:] = torch.cumsum(mask, dim=0)

        del codes_1d
        torch.cuda.empty_cache()
        log_gpu_memory_snapshot(f"После построения Prefix-Sum для m={order_m}", pytorch_device)

        for window_index, window_samples in enumerate(window_samples_array):
            if selected_permutation_orders_list[window_index] != order_m:
                continue

            subsequences_in_window = window_samples - order_m + 1

            for batch_start_index in range(0, number_of_centers, center_batch_size_count):
                batch_end_index = min(batch_start_index + center_batch_size_count, number_of_centers)

                left_starts = torch.from_numpy(center_sample_indices_array[batch_start_index:batch_end_index] - window_samples).to(pytorch_device)
                left_ends = torch.from_numpy(center_sample_indices_array[batch_start_index:batch_end_index] - order_m + 1).to(pytorch_device)

                right_starts = torch.from_numpy(center_sample_indices_array[batch_start_index:batch_end_index]).to(pytorch_device)
                right_ends = torch.from_numpy(center_sample_indices_array[batch_start_index:batch_end_index] + window_samples - order_m + 1).to(pytorch_device)

                left_counts = cumulative_counts[:, left_ends] - cumulative_counts[:, left_starts]
                right_counts = cumulative_counts[:, right_ends] - cumulative_counts[:, right_starts]

                def evaluate_counts_tensor(counts_tensor):
                    probability_distribution = counts_tensor.to(torch.float32) / float(subsequences_in_window)
                    shannon_p = -(
                        probability_distribution * torch.log2(probability_distribution + EPSILON_SMOOTHING)
                    ).sum(dim=0)
                    normalized_entropy = shannon_p / maximum_entropy

                    mixed_probabilities = 0.5 * (probability_distribution + uniform_probability)
                    shannon_mixed = -(
                        mixed_probabilities * torch.log2(mixed_probabilities + EPSILON_SMOOTHING)
                    ).sum(dim=0)
                    
                    jensen_shannon_divergence = (
                        shannon_mixed - 0.5 * shannon_p - 0.5 * entropy_equilibrium
                    )
                    statistical_complexity = jensen_shannon_divergence * normalized_entropy * q0
                    return normalized_entropy, statistical_complexity

                hl, cl = evaluate_counts_tensor(left_counts)
                hr, cr = evaluate_counts_tensor(right_counts)

                permutation_entropy_left_matrix[batch_start_index:batch_end_index, window_index] = hl.cpu().numpy()
                statistical_complexity_left_matrix[batch_start_index:batch_end_index, window_index] = cl.cpu().numpy()
                permutation_entropy_right_matrix[batch_start_index:batch_end_index, window_index] = hr.cpu().numpy()
                statistical_complexity_right_matrix[batch_start_index:batch_end_index, window_index] = cr.cpu().numpy()

        del cumulative_counts
        torch.cuda.empty_cache()

    return (
        permutation_entropy_left_matrix,
        permutation_entropy_right_matrix,
        statistical_complexity_left_matrix,
        statistical_complexity_right_matrix,
    )


def run_numerical_consistency_check(pytorch_device: torch.device):
    logger.info("Запуск самотеста: верификация математики с эталоном CPU...")
    np.random.seed(42)
    synthetic_signal = np.sin(np.linspace(0, 30, 1500)) + 0.5 * np.cos(np.linspace(0, 90, 1500))
    tensor_gpu = torch.from_numpy(synthetic_signal.astype(np.float32)).to(pytorch_device)

    test_centers = np.array([700, 750], dtype=int)
    test_windows = np.array([30, 125, 605], dtype=int)
    test_orders = [select_permutation_order(window_sample, coverage_multiplier=5) for window_sample in test_windows]

    gpu_pe_l, gpu_pe_r, gpu_c_l, gpu_c_r = compute_single_channel_multiscale_batched(
        channel_signal_1d=tensor_gpu,
        center_sample_indices_array=test_centers,
        window_samples_array=test_windows,
        selected_permutation_orders_list=test_orders,
        stride_samples_count=50,
        pytorch_device=pytorch_device,
        center_batch_size_count=500
    )

    tolerance = 1e-5
    for window_index, (window_samples, order_m) in enumerate(zip(test_windows, test_orders)):
        for center_index, center in enumerate(test_centers):
            ref_entropy_l, ref_complexity_l = reference_complexity_entropy_cpu(
                synthetic_signal[center - window_samples : center], permutation_order=order_m
            )
            ref_entropy_r, ref_complexity_r = reference_complexity_entropy_cpu(
                synthetic_signal[center : center + window_samples], permutation_order=order_m
            )

            current_entropy_l = float(gpu_pe_l[center_index, window_index])
            current_complexity_l = float(gpu_c_l[center_index, window_index])
            current_entropy_r = float(gpu_pe_r[center_index, window_index])
            current_complexity_r = float(gpu_c_r[center_index, window_index])

            if (
                not np.isclose(ref_entropy_l, current_entropy_l, atol=tolerance)
                or not np.isclose(ref_complexity_l, current_complexity_l, atol=tolerance)
                or not np.isclose(ref_entropy_r, current_entropy_r, atol=tolerance)
                or not np.isclose(ref_complexity_r, current_complexity_r, atol=tolerance)
            ):
                logger.error(f"КРИТИЧЕСКИЙ СБОЙ ВЕРИФИКАЦИИ: w={window_samples}, m={order_m}")
                sys.exit(1)

    logger.info("Самотест успешно пройден. Математика абсолютна точна.")


def execute_dataset_processing(dataset_filecode: int):
    config = DATASET_CONFIGURATIONS[dataset_filecode]
    target_fif_path = config["prep_fif"] if config["prep_fif"].exists() else config["raw_fif"]
    if not target_fif_path.exists():
        raise FileNotFoundError(f"Файл FIF не найден: {target_fif_path}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA CUDA не обнаружена. Если запуск производится в Docker/Singularity, "
            "убедитесь, что контейнер запущен с флагами проброса GPU: '--gpus all' (Docker) или '--nv' (Singularity/Apptainer)."
        )

    pytorch_device = torch.device("cuda:0")
    torch.cuda.set_device(pytorch_device)

    logger.info(f"Старт обработки датасета: {config['description']}")
    log_gpu_memory_snapshot("Инициализация процесса", pytorch_device)

    run_numerical_consistency_check(pytorch_device)

    raw_meg_data = mne.io.read_raw_fif(target_fif_path, preload=True, verbose=False)
    sampling_frequency_hz = float(raw_meg_data.info["sfreq"])
    total_number_of_samples = raw_meg_data.n_times  # Оптимизация памяти (вместо len(raw_meg_data.times))

    meg_channel_indices = mne.pick_types(raw_meg_data.info, meg=True, eeg=False, stim=False, exclude=[])
    channel_names_list = [raw_meg_data.ch_names[index] for index in meg_channel_indices]
    meg_signals_numpy_matrix = raw_meg_data.get_data(picks=meg_channel_indices).astype(np.float32)
    number_of_channels = len(channel_names_list)
    del raw_meg_data

    if config["window_mode"] == "type1_type3":
        window_durations_seconds_array = np.linspace(0.2, 5.0, 50)
    else:
        window_durations_seconds_array = np.arange(1.0, 25.5, 0.5)

    stride_samples_count = int(round(0.1 * sampling_frequency_hz))
    window_samples_array = np.round(window_durations_seconds_array * sampling_frequency_hz).astype(int)
    maximum_window_samples_count = int(np.max(window_samples_array))

    center_sample_indices_array = np.arange(
        maximum_window_samples_count,
        total_number_of_samples - maximum_window_samples_count,
        stride_samples_count
    )
    number_of_centers_count = len(center_sample_indices_array)
    number_of_windows_count = len(window_durations_seconds_array)

    selected_permutation_orders_per_window_list = [
        select_permutation_order(window_samples, coverage_multiplier=5) for window_samples in window_samples_array
    ]

    logger.info(
        f"Параметры сетки -> Каналов: {number_of_channels} | "
        f"Временных центров: {number_of_centers_count} | Окон: {number_of_windows_count}"
    )

    output_parquet_file_path = config["output_parquet"]
    output_parquet_file_path.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet_file_path.exists():
        output_parquet_file_path.unlink()

    time_seconds_array = np.round(center_sample_indices_array / sampling_frequency_hz, 4)
    parquet_file_writer = None

    for channel_index, channel_name in enumerate(channel_names_list):
        channel_signal_tensor = torch.from_numpy(meg_signals_numpy_matrix[channel_index]).to(pytorch_device)

        permutation_entropy_left, permutation_entropy_right, statistical_complexity_left, statistical_complexity_right = (
            compute_single_channel_multiscale_batched(
                channel_signal_1d=channel_signal_tensor,
                center_sample_indices_array=center_sample_indices_array,
                window_samples_array=window_samples_array,
                selected_permutation_orders_list=selected_permutation_orders_per_window_list,
                stride_samples_count=stride_samples_count,
                pytorch_device=pytorch_device,
                center_batch_size_count=2000
            )
        )
        del channel_signal_tensor
        torch.cuda.empty_cache()

        # Прямое округление и расчет дельт на CPU без лишнего PCIe-трансфера
        pe_l_numpy = np.round(permutation_entropy_left, 5)
        pe_r_numpy = np.round(permutation_entropy_right, 5)
        c_l_numpy = np.round(statistical_complexity_left, 5)
        c_r_numpy = np.round(statistical_complexity_right, 5)

        delta_pe_numpy = np.abs(pe_l_numpy - pe_r_numpy)
        delta_c_numpy = np.abs(c_l_numpy - c_r_numpy)
        delta_ec_numpy = delta_pe_numpy + delta_c_numpy

        channel_records_dictionary = {
            "sample_idx": center_sample_indices_array,
            "time_sec": time_seconds_array,
            "channel": channel_name,
        }

        for window_index, window_duration_seconds in enumerate(window_durations_seconds_array):
            order_used = selected_permutation_orders_per_window_list[window_index]
            tag = f"w_{window_duration_seconds:.3f}s_m{order_used}" if window_duration_seconds < 1.0 else f"w_{window_duration_seconds:.1f}s_m{order_used}"

            channel_records_dictionary[f"pe_left_{tag}"] = pe_l_numpy[:, window_index]
            channel_records_dictionary[f"pe_right_{tag}"] = pe_r_numpy[:, window_index]
            channel_records_dictionary[f"delta_pe_{tag}"] = delta_pe_numpy[:, window_index]

            channel_records_dictionary[f"c_left_{tag}"] = c_l_numpy[:, window_index]
            channel_records_dictionary[f"c_right_{tag}"] = c_r_numpy[:, window_index]
            channel_records_dictionary[f"delta_c_{tag}"] = delta_c_numpy[:, window_index]

            channel_records_dictionary[f"delta_ec_{tag}"] = delta_ec_numpy[:, window_index]

        # Создание табличного чанка через Polars и однократный экспорт в Arrow
        polars_dataframe_chunk = pl.DataFrame(channel_records_dictionary)
        arrow_table_chunk = polars_dataframe_chunk.to_arrow()

        if parquet_file_writer is None:
            parquet_file_writer = pq.ParquetWriter(
                output_parquet_file_path, 
                schema=arrow_table_chunk.schema,
                compression="snappy"
            )
        parquet_file_writer.write_table(arrow_table_chunk)

        if (channel_index + 1) % 50 == 0 or (channel_index + 1) == number_of_channels:
            logger.info(f"Прогресс записи: обработано и записано каналов {channel_index + 1} из {number_of_channels}")
            log_gpu_memory_snapshot(f"Канал {channel_index + 1}/{number_of_channels}", pytorch_device)

    if parquet_file_writer is not None:
        parquet_file_writer.close()

    logger.info(f"Успешно сформирован итоговый Parquet-файл: {output_parquet_file_path}")
    log_gpu_memory_snapshot("Завершение работы сессии", pytorch_device)


if __name__ == "__main__":
    filecode_environment_variable = os.environ.get("SLURM_ARRAY_TASK_ID")
    if filecode_environment_variable is not None:
        selected_filecode = int(filecode_environment_variable)
    elif len(sys.argv) > 1:
        selected_filecode = int(sys.argv[1])
    else:
        raise ValueError("filecode не передан ни через SLURM_ARRAY_TASK_ID, ни через аргумент командной строки")

    execute_dataset_processing(dataset_filecode=selected_filecode)
