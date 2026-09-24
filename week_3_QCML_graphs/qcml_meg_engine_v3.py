#!/usr/bin/env python3
"""
qcml_meg_engine_v3.py
Движок Quantum Cognition Machine Learning (QCML) для анализа динамических
многообразий МЭГ (Версия v3).
"""

import gc
import math
import time
import argparse
from pathlib import Path
from typing import Generator

import numpy as np
import mne
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from functools import lru_cache, partial
from scipy.integrate import quad
from scipy.optimize import brentq

mne.set_log_level("ERROR")


MILLISECONDS_PER_SECOND = 1000.0
MINIMUM_TAKENS_LAG_SAMPLES = 1
EIGENVALUE_ZERO_TOLERANCE_MULTIPLIER = 10.0
DIMENSION_THRESHOLD = 0.5
RMT_BULK_FRACTION = 0.25


def marchenko_pastur_eigenvalue_support(aspect_ratio: float) -> tuple[float, float]:
    sqrt_aspect_ratio = math.sqrt(aspect_ratio)
    lower_edge = (1.0 - sqrt_aspect_ratio) ** 2
    upper_edge = (1.0 + sqrt_aspect_ratio) ** 2
    return lower_edge, upper_edge


def marchenko_pastur_eigenvalue_density(eigenvalue: float, aspect_ratio: float) -> float:
    lower_edge, upper_edge = marchenko_pastur_eigenvalue_support(aspect_ratio)
    bulk_product = (upper_edge - eigenvalue) * (eigenvalue - lower_edge)
    bulk_amplitude = math.sqrt(max(bulk_product, 0.0))
    normalization = 2.0 * math.pi * aspect_ratio * eigenvalue
    return bulk_amplitude / normalization


def marchenko_pastur_cumulative_probability(eigenvalue: float, aspect_ratio: float) -> float:
    lower_edge, _ = marchenko_pastur_eigenvalue_support(aspect_ratio)
    integral_value, _ = quad(
        marchenko_pastur_eigenvalue_density, lower_edge, eigenvalue, args=(aspect_ratio,)
    )
    return integral_value


def marchenko_pastur_median_deviation(eigenvalue: float, aspect_ratio: float) -> float:
    cumulative_probability = marchenko_pastur_cumulative_probability(eigenvalue, aspect_ratio)
    return cumulative_probability - 0.5


@lru_cache(maxsize=128)
def marchenko_pastur_median(aspect_ratio: float) -> float:
    lower_edge, upper_edge = marchenko_pastur_eigenvalue_support(aspect_ratio)
    median_deviation = partial(marchenko_pastur_median_deviation, aspect_ratio=aspect_ratio)
    return brentq(median_deviation, lower_edge, upper_edge)


def least_favorable_amplitude_squared(aspect_ratio: float) -> float:
    shifted_aspect_ratio = aspect_ratio + 1.0
    quartic_beta_coefficient = 3.0
    discriminant = shifted_aspect_ratio ** 2 + 4.0 * quartic_beta_coefficient * aspect_ratio
    return (shifted_aspect_ratio + math.sqrt(discriminant)) / 2.0


def gavish_donoho_squared_coefficient(aspect_ratio: float) -> float:
    amplitude_squared = least_favorable_amplitude_squared(aspect_ratio)
    shifted_aspect_ratio = aspect_ratio + 1.0
    aspect_ratio_over_amplitude_squared = aspect_ratio / amplitude_squared
    return amplitude_squared + shifted_aspect_ratio + aspect_ratio_over_amplitude_squared


def gavish_donoho_coefficient(aspect_ratio: float) -> float:
    return math.sqrt(gavish_donoho_squared_coefficient(aspect_ratio))


@lru_cache(maxsize=128)
def unknown_noise_threshold_coefficient(aspect_ratio: float) -> float:
    median_under_null = marchenko_pastur_median(aspect_ratio)
    known_noise_coefficient = gavish_donoho_coefficient(aspect_ratio)
    return known_noise_coefficient / math.sqrt(median_under_null)


def load_preprocessed_raw_fif(fif_path: Path) -> mne.io.Raw:
    return mne.io.read_raw_fif(fif_path, preload=True, verbose=False)


def meg_sampling_frequency(recording: mne.io.Raw) -> float:
    return float(recording.info["sfreq"])


def magnetometer_channel_indices(recording: mne.io.Raw) -> list[int]:
    indices = mne.pick_types(
        recording.info, meg="mag", eeg=False, stim=False, eog=False, ecg=False
    )
    return indices.tolist()


def gradiometer_channel_indices(recording: mne.io.Raw) -> list[int]:
    indices = mne.pick_types(
        recording.info, meg="grad", eeg=False, stim=False, eog=False, ecg=False
    )
    return indices.tolist()


def extract_meg_samples(
    recording: mne.io.Raw,
    selected_channel_indices: np.ndarray,
) -> np.ndarray:
    return recording.get_data(picks=selected_channel_indices).T.astype(np.float64)


def remove_sensor_offsets(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sensor_offsets = np.mean(samples, axis=0, keepdims=True)
    centered_samples = samples - sensor_offsets
    return centered_samples, sensor_offsets.squeeze(0)


def normalize_magnetometers_and_gradiometers_by_block_rms(
    samples: np.ndarray, magnetometer_count: int
) -> tuple[np.ndarray, float, float]:
    centered_samples, _ = remove_sensor_offsets(samples)
    magnetometer_block = centered_samples[:, :magnetometer_count]
    gradiometer_block = centered_samples[:, magnetometer_count:]
    magnetometer_rms = float(np.sqrt(np.mean(magnetometer_block ** 2)))
    gradiometer_rms = float(np.sqrt(np.mean(gradiometer_block ** 2)))
    normalized_samples = centered_samples.copy()
    normalized_samples[:, :magnetometer_count] /= magnetometer_rms
    normalized_samples[:, magnetometer_count:] /= gradiometer_rms
    return normalized_samples, magnetometer_rms, gradiometer_rms


def normalize_channels_by_zscore(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered_samples, _ = remove_sensor_offsets(samples)
    channel_standard_deviations = np.std(centered_samples, axis=0, keepdims=True)
    normalized_samples = centered_samples / channel_standard_deviations
    return normalized_samples, channel_standard_deviations.squeeze(0)


def takens_lag_in_samples(delay_milliseconds: float, sampling_frequency: float) -> int:
    delay_seconds = delay_milliseconds / MILLISECONDS_PER_SECOND
    lag_samples = round(delay_seconds * sampling_frequency)
    return max(MINIMUM_TAKENS_LAG_SAMPLES, int(lag_samples))


def lagged_block(
    samples: np.ndarray, block_offset_samples: int, block_length_samples: int
) -> np.ndarray:
    return samples[block_offset_samples : block_offset_samples + block_length_samples]


def descending_block_offsets(
    highest_offset_samples: int, lag_samples: int, block_count: int
) -> Generator[int, None, None]:
    for block_index in range(block_count):
        yield highest_offset_samples - block_index * lag_samples


def build_takens_delay_embedding(
    samples: np.ndarray, embedding_dimension: int, lag_samples: int
) -> tuple[np.ndarray, int]:
    if embedding_dimension == 1:
        return samples, 0

    total_sample_count, _ = samples.shape
    dropped_leading_samples = (embedding_dimension - 1) * lag_samples
    kept_sample_count = total_sample_count - dropped_leading_samples

    lagged_blocks = [
        lagged_block(samples, block_offset, kept_sample_count)
        for block_offset in descending_block_offsets(
            dropped_leading_samples, lag_samples, embedding_dimension
        )
    ]
    embedded_samples = np.concatenate(lagged_blocks, axis=1)
    print(
        f"  [Такенс] Фазовое вложение: m = {embedding_dimension}, "
        f"lag = {lag_samples} сэмплов | Форма: {embedded_samples.shape}"
    )
    return embedded_samples, dropped_leading_samples


def sequential_batch_slices(
    sample_count: int, batch_size: int
) -> Generator[tuple[int, int], None, None]:
    for batch_start in range(0, sample_count, batch_size):
        batch_end = min(batch_start + batch_size, sample_count)
        yield batch_start, batch_end


def hermitian_part(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def operator_initialization_scale(feature_dimension: int, hilbert_dimension: int) -> float:
    return 1.0 / math.sqrt(feature_dimension * hilbert_dimension)


def initialize_operator_parameters(
    feature_dimension: int, hilbert_dimension: int, device: torch.device
) -> torch.Tensor:
    initialization_scale = operator_initialization_scale(feature_dimension, hilbert_dimension)
    real_part = torch.randn(feature_dimension, hilbert_dimension, hilbert_dimension, device=device)
    imaginary_part = torch.randn(feature_dimension, hilbert_dimension, hilbert_dimension, device=device)
    raw_operators = torch.complex(real_part * initialization_scale, imaginary_part * initialization_scale)
    raw_operators.requires_grad_(True)
    return raw_operators


def hermitian_operators_from_parameters(raw_operators: torch.Tensor) -> torch.Tensor:
    return hermitian_part(raw_operators)


def sum_of_squared_operators(hermitian_operators: torch.Tensor) -> torch.Tensor:
    operator_squares = torch.matmul(hermitian_operators, hermitian_operators)
    return 0.5 * torch.sum(operator_squares, dim=0)


def build_hermitian_and_squared_operators(
    raw_operators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hermitian_operators = hermitian_operators_from_parameters(raw_operators)
    sum_of_squared = sum_of_squared_operators(hermitian_operators)
    return hermitian_operators, sum_of_squared


def complex_identity_matrix(dimension: int, device: torch.device) -> torch.Tensor:
    return torch.eye(dimension, device=device, dtype=torch.complex64)


def assemble_error_hamiltonian(
    sum_of_squared_operators: torch.Tensor,
    linear_contraction: torch.Tensor,
    point_norms_squared: torch.Tensor,
) -> torch.Tensor:
    hilbert_dimension = sum_of_squared_operators.shape[0]
    identity = complex_identity_matrix(hilbert_dimension, sum_of_squared_operators.device)
    point_term = point_norms_squared.to(torch.complex64) * identity
    assembled = sum_of_squared_operators.unsqueeze(0) - linear_contraction + point_term
    return hermitian_part(assembled)

def complex_eigh(hamiltonian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = hermitian_part(hamiltonian)
    if not torch.is_grad_enabled():
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix.to(torch.complex128))
        return eigenvalues.to(torch.float32), eigenvectors.to(torch.complex64)
    return torch.linalg.eigh(matrix)



def lowest_eigenpair(hamiltonian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = complex_eigh(hamiltonian)
    return eigenvalues[:, 0].to(torch.float32), eigenvectors[:, :, 0].to(torch.complex64)


def ground_and_excited_eigenpairs(
    hamiltonian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = complex_eigh(hamiltonian)
    ground_eigenvalue = eigenvalues[:, 0].to(torch.float32)
    ground_state = eigenvectors[:, :, 0].to(torch.complex64)
    excited_eigenvalues = eigenvalues[:, 1:].to(torch.float32)
    excited_states = eigenvectors[:, :, 1:].to(torch.complex64)
    return ground_eigenvalue, ground_state, excited_eigenvalues, excited_states


def compute_linear_operator_contraction(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
) -> torch.Tensor:
    complex_feature_batch = feature_batch.to(torch.complex64)
    return torch.einsum("bd, dnm -> bnm", complex_feature_batch, hermitian_operators)


def compute_point_norms_squared(feature_batch: torch.Tensor) -> torch.Tensor:
    squared_norms = torch.sum(feature_batch ** 2, dim=-1, keepdim=True)
    return 0.5 * squared_norms.unsqueeze(-1)


def compute_coordinate_expectations(
    ground_state: torch.Tensor,
    hermitian_operators: torch.Tensor,
) -> torch.Tensor:
    symmetrized_operators = hermitian_part(hermitian_operators)
    expectations = torch.einsum(
        "bn, dnm, bm -> bd",
        ground_state.conj(),
        symmetrized_operators,
        ground_state,
    )
    return expectations.real



def compute_bias_loss(
    feature_batch: torch.Tensor,
    projected_batch: torch.Tensor,
) -> torch.Tensor:
    squared_errors = torch.sum((projected_batch - feature_batch) ** 2, dim=-1)
    return torch.mean(squared_errors)


def compute_local_variance(
    ground_state: torch.Tensor,
    projected_batch: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> torch.Tensor:
    expected_squared_sum = 2.0 * torch.einsum(
        "bn, nm, bm -> b",
        ground_state.conj(),
        sum_of_squared,
        ground_state,
    ).real
    projected_norm_squared = torch.sum(projected_batch ** 2, dim=-1)
    return torch.clamp(expected_squared_sum - projected_norm_squared, min=0.0)


def compute_projection_residuals(
    feature_batch: torch.Tensor,
    projected_batch: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.norm(projected_batch - feature_batch, dim=-1)


def compute_eigenvalue_gaps(
    ground_eigenvalue: torch.Tensor,
    excited_eigenvalues: torch.Tensor,
) -> torch.Tensor:
    return excited_eigenvalues - ground_eigenvalue.unsqueeze(-1)


def compute_transition_matrix_elements(
    ground_state: torch.Tensor,
    excited_states: torch.Tensor,
    hermitian_operators: torch.Tensor,
) -> torch.Tensor:
    excited_states_transposed = excited_states.permute(0, 2, 1)
    return torch.einsum(
        "bn, dnm, bkm -> bkd",
        ground_state.conj(),
        hermitian_operators,
        excited_states_transposed,
    )


def scale_transition_vectors_by_gap(
    transition_vectors: torch.Tensor,
    eigenvalue_gaps: torch.Tensor,
) -> torch.Tensor:
    gap_scales = torch.sqrt(2.0 / eigenvalue_gaps).unsqueeze(-1)
    return transition_vectors * gap_scales


def assemble_transition_feature_matrix(
    scaled_transitions: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([scaled_transitions.real, scaled_transitions.imag], dim=1)


def compute_analytic_metric_diagonal(
    transition_features: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(transition_features ** 2, dim=1)


def active_eigenvalues_from_transition_features(
    transition_features: torch.Tensor,
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(transition_features.to(torch.float64))
    return (singular_values ** 2).to(torch.float32)


def compute_ratio_gap_dimension(active_eigenvalues: torch.Tensor) -> torch.Tensor:
    tensor_epsilon = torch.finfo(active_eigenvalues.dtype).eps
    consecutive_ratios = active_eigenvalues[:, :-1] / (active_eigenvalues[:, 1:] + tensor_epsilon)
    ratio_gap = (torch.argmax(consecutive_ratios, dim=1) + 1).to(torch.float32)
    zero_tolerance = EIGENVALUE_ZERO_TOLERANCE_MULTIPLIER * tensor_epsilon
    return torch.where(
        active_eigenvalues[:, 0] > zero_tolerance,
        ratio_gap,
        torch.zeros_like(ratio_gap),
    )


def compute_rmt_dimension(
    active_eigenvalues: torch.Tensor,
    unknown_noise_threshold: float,
) -> torch.Tensor:
    singular_values = torch.sqrt(torch.clamp(active_eigenvalues, min=0.0))
    gramian_dimension = singular_values.shape[1]
    bulk_start = int(RMT_BULK_FRACTION * gramian_dimension)
    bulk_slice = singular_values[:, bulk_start:]
    median_singular_value = torch.median(bulk_slice, dim=-1).values
    threshold = unknown_noise_threshold * median_singular_value
    return (singular_values > threshold.unsqueeze(-1)).sum(dim=-1).to(torch.float32)


def compute_threshold_dimension(active_eigenvalues: torch.Tensor) -> torch.Tensor:
    return (active_eigenvalues > DIMENSION_THRESHOLD).sum(dim=-1).to(torch.float32)


def error_hamiltonian_for_batch(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> torch.Tensor:
    linear_contraction = compute_linear_operator_contraction(feature_batch, hermitian_operators)
    point_norms_squared = compute_point_norms_squared(feature_batch)
    return assemble_error_hamiltonian(sum_of_squared, linear_contraction, point_norms_squared)

def assemble_core_hamiltonian(
    sum_of_squared_operators: torch.Tensor,
    linear_contraction: torch.Tensor,
) -> torch.Tensor:
    assembled = sum_of_squared_operators.unsqueeze(0) - linear_contraction
    return hermitian_part(assembled)


def core_hamiltonian_for_batch(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> torch.Tensor:
    linear_contraction = compute_linear_operator_contraction(feature_batch, hermitian_operators)
    return assemble_core_hamiltonian(sum_of_squared, linear_contraction)

def project_batch_to_point_cloud(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> dict[str, torch.Tensor]:
    hamiltonian = core_hamiltonian_for_batch(feature_batch, hermitian_operators, sum_of_squared)
    ground_eigenvalue_core, ground_state, excited_eigenvalues_core, _ = ground_and_excited_eigenpairs(hamiltonian)
    point_norms_squared = 0.5 * torch.sum(feature_batch ** 2, dim=-1)
    ground_eigenvalue = ground_eigenvalue_core + point_norms_squared
    projected_batch = compute_coordinate_expectations(ground_state, hermitian_operators)
    return {
        "projected_batch": projected_batch,
        "ground_state": ground_state,
        "ground_eigenvalue": ground_eigenvalue,
        "first_eigenvalue_gap": excited_eigenvalues_core[:, 0] - ground_eigenvalue_core,
        "local_variance": compute_local_variance(ground_state, projected_batch, sum_of_squared),
    }


def compute_qcml_composite_loss(
    bias_loss: torch.Tensor,
    variance_batch: torch.Tensor | None,
    variance_weight: float,
) -> torch.Tensor:
    if variance_batch is None:
        return bias_loss
    return bias_loss + variance_weight * torch.mean(variance_batch)


def execute_gradient_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    raw_operators: torch.Tensor,
    grad_clip_norm: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([raw_operators], max_norm=grad_clip_norm)
    optimizer.step()
    return loss.item()


def differentiate_ground_state(
    hamiltonian: torch.Tensor,
    ground_state: torch.Tensor,
    ground_eigenvalue: torch.Tensor,
    excited_states: torch.Tensor,
    excited_eigenvalues: torch.Tensor,
) -> torch.Tensor:
    transition_elements = torch.einsum(
        "bnk, bnm, bm -> bk",
        excited_states.conj(),
        hamiltonian,
        ground_state,
    )
    energy_gaps = excited_eigenvalues - ground_eigenvalue.unsqueeze(-1)
    scaled_transitions = transition_elements / energy_gaps
    perturbation_vector = torch.einsum("bk, bnk -> bn", scaled_transitions, excited_states)
    return ground_state - perturbation_vector


def ground_state_for_batch(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
    detach_eigenvectors: bool,
) -> torch.Tensor:
    hamiltonian = core_hamiltonian_for_batch(feature_batch, hermitian_operators, sum_of_squared)
    with torch.no_grad():
        ground_eigenvalue, ground_state, excited_eigenvalues, excited_states = (
            ground_and_excited_eigenpairs(hamiltonian)
        )
    if detach_eigenvectors:
        return ground_state
    return differentiate_ground_state(
        hamiltonian, ground_state, ground_eigenvalue, excited_states, excited_eigenvalues
    )


def training_step(
    feature_batch: torch.Tensor,
    raw_operators: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    detach_eigenvectors: bool,
    variance_weight: float,
    grad_clip_norm: float,
) -> float:
    hermitian_operators, sum_of_squared = build_hermitian_and_squared_operators(raw_operators)
    ground_state = ground_state_for_batch(
        feature_batch, hermitian_operators, sum_of_squared, detach_eigenvectors
    )
    projected_batch = compute_coordinate_expectations(ground_state, hermitian_operators)
    bias_loss = compute_bias_loss(feature_batch, projected_batch)
    if variance_weight == 0.0:
        variance_batch = None
    else:
        variance_batch = compute_local_variance(ground_state, projected_batch, sum_of_squared)
    loss = compute_qcml_composite_loss(bias_loss, variance_batch, variance_weight)
    return execute_gradient_step(optimizer, loss, raw_operators, grad_clip_norm)


def run_training_epoch(
    meg_samples: np.ndarray,
    raw_operators: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    detach_eigenvectors: bool,
    variance_weight: float,
    grad_clip_norm: float,
) -> float:
    shuffled_indices = np.random.permutation(meg_samples.shape[0])
    accumulated_loss = 0.0
    processed_batch_count = 0
    for batch_start in range(0, meg_samples.shape[0], batch_size):
        batch_indices = shuffled_indices[batch_start : batch_start + batch_size]
        feature_batch = torch.from_numpy(meg_samples[batch_indices]).to(device)
        accumulated_loss += training_step(
            feature_batch,
            raw_operators,
            optimizer,
            detach_eigenvectors,
            variance_weight,
            grad_clip_norm,
        )
        processed_batch_count += 1
    return accumulated_loss / max(processed_batch_count, 1)


def active_eigenvalues_from_projected_batch(
    projected_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> torch.Tensor:
    hamiltonian = core_hamiltonian_for_batch(projected_batch, hermitian_operators, sum_of_squared)
    ground_eigenvalue, ground_state, excited_eigenvalues, excited_states = ground_and_excited_eigenpairs(hamiltonian)
    eigenvalue_gaps = compute_eigenvalue_gaps(ground_eigenvalue, excited_eigenvalues)
    raw_transitions = compute_transition_matrix_elements(ground_state, excited_states, hermitian_operators)
    scaled_transitions = scale_transition_vectors_by_gap(raw_transitions, eigenvalue_gaps)
    transition_features = assemble_transition_feature_matrix(scaled_transitions)
    return active_eigenvalues_from_transition_features(transition_features)


def validation_batch_metrics(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
    unknown_noise_threshold: float,
    variance_weight: float,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    projection = project_batch_to_point_cloud(
        feature_batch, hermitian_operators, sum_of_squared
    )
    projected_batch = projection["projected_batch"]
    ground_state = projection["ground_state"]

    bias_sum = torch.sum((projected_batch - feature_batch) ** 2, dim=-1).sum().item()
    if variance_weight == 0.0:
        variance_sum = 0.0
    else:
        variance_sum = compute_local_variance(
            ground_state, projected_batch, sum_of_squared
        ).sum().item()
    active_eigenvalues = active_eigenvalues_from_projected_batch(
        projected_batch, hermitian_operators, sum_of_squared
    )
    ratio_gap = compute_ratio_gap_dimension(active_eigenvalues)
    rmt_dimension = compute_rmt_dimension(active_eigenvalues, unknown_noise_threshold)
    threshold_dimension = compute_threshold_dimension(active_eigenvalues)
    return bias_sum, variance_sum, ratio_gap, rmt_dimension, threshold_dimension, active_eigenvalues


def validation_statistics(
    accumulated_bias: float,
    accumulated_variance: float,
    ratio_gaps: torch.Tensor,
    rmt_dimensions: torch.Tensor,
    threshold_dimensions: torch.Tensor,
    processed_sample_count: int,
    variance_weight: float,
) -> dict[str, float]:
    bias_mean = accumulated_bias / max(processed_sample_count, 1)
    variance_mean = accumulated_variance / max(processed_sample_count, 1)
    return {
        "val_loss": bias_mean + variance_weight * variance_mean,
        "val_bias": bias_mean,
        "d_ratio_mean": ratio_gaps.mean().item(),
        "d_ratio_std": ratio_gaps.std(unbiased=True).item() if ratio_gaps.numel() > 1 else 0.0,
        "d_rmt_mean": rmt_dimensions.mean().item(),
        "d_rmt_std": rmt_dimensions.std(unbiased=True).item() if rmt_dimensions.numel() > 1 else 0.0,
        "d_th_mean": threshold_dimensions.mean().item(),
        "d_th_std": threshold_dimensions.std(unbiased=True).item() if threshold_dimensions.numel() > 1 else 0.0,
    }


def print_spectrum_diagnostic(active_eigenvalues: torch.Tensor) -> None:
    first_sample = active_eigenvalues[0].detach().cpu().numpy()
    gramian_dimension = len(first_sample)
    probe_positions = sorted(set([
        0,
        gramian_dimension // 4,
        gramian_dimension // 2,
        (3 * gramian_dimension) // 4,
        gramian_dimension - 1,
    ]))
    formatted = " ".join(f"λ[{position}]={first_sample[position]:.4g}" for position in probe_positions)
    print(
        f"      [спектр] {formatted} | "
        f"min={float(first_sample.min()):.2e} max={float(first_sample.max()):.2e}"
    )


@torch.no_grad()
def evaluate_validation(
    validation_samples: np.ndarray,
    raw_operators: torch.Tensor,
    unknown_noise_threshold: float,
    batch_size: int,
    device: torch.device,
    variance_weight: float,
    print_spectrum: bool,
) -> dict[str, float]:
    hermitian_operators, sum_of_squared = build_hermitian_and_squared_operators(raw_operators)
    accumulated_bias = 0.0
    accumulated_variance = 0.0
    ratio_gap_chunks: list[torch.Tensor] = []
    rmt_dimension_chunks: list[torch.Tensor] = []
    threshold_dimension_chunks: list[torch.Tensor] = []
    processed_sample_count = 0
    first_batch_eigenvalues: torch.Tensor | None = None
    for batch_start in range(0, validation_samples.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, validation_samples.shape[0])
        feature_batch = torch.from_numpy(validation_samples[batch_start:batch_end]).to(device)
        bias_sum, variance_sum, ratio_gap, rmt_dimension, threshold_dimension, active_eigenvalues = (
            validation_batch_metrics(
                feature_batch,
                hermitian_operators,
                sum_of_squared,
                unknown_noise_threshold,
                variance_weight,
            )
        )
        accumulated_bias += bias_sum
        accumulated_variance += variance_sum
        ratio_gap_chunks.append(ratio_gap.cpu())
        rmt_dimension_chunks.append(rmt_dimension.cpu())
        threshold_dimension_chunks.append(threshold_dimension.cpu())
        processed_sample_count += feature_batch.shape[0]
        if first_batch_eigenvalues is None:
            first_batch_eigenvalues = active_eigenvalues

    if print_spectrum and first_batch_eigenvalues is not None:
        print_spectrum_diagnostic(first_batch_eigenvalues)

    return validation_statistics(
        accumulated_bias,
        accumulated_variance,
        torch.cat(ratio_gap_chunks),
        torch.cat(rmt_dimension_chunks),
        torch.cat(threshold_dimension_chunks),
        processed_sample_count,
        variance_weight,
    )


def split_train_validation(
    meg_samples: np.ndarray,
    validation_fraction: float,
    validation_max_samples: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_sample_count = meg_samples.shape[0]
    validation_sample_count = min(
        validation_max_samples,
        max(batch_size, int(total_sample_count * validation_fraction)),
    )
    training_samples = meg_samples[:-validation_sample_count]
    validation_samples = meg_samples[-validation_sample_count:]
    print_split_statistics(training_samples, validation_samples, total_sample_count)
    return training_samples, validation_samples


def print_split_statistics(
    training_samples: np.ndarray,
    validation_samples: np.ndarray,
    total_sample_count: int,
) -> None:
    print(
        f"[*] Сплит: train = {len(training_samples)} | val = {len(validation_samples)} "
        f"({100.0 * len(validation_samples) / total_sample_count:.1f}%)"
    )


def gramian_aspect_ratio(hilbert_dimension: int, feature_dimension: int) -> float:
    gramian_dimension = 2 * (hilbert_dimension - 1)
    return min(1.0, gramian_dimension / float(feature_dimension))


def print_optimization_start(
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    variance_weight: float,
    unknown_noise_threshold: float,
    detach_eigenvectors: bool,
    hilbert_dimension: int,
    feature_dimension: int,
) -> None:
    gradient_tag = "detach" if detach_eigenvectors else "grad"
    print(
        f"[*] QCML v3 | D={feature_dimension}, N={hilbert_dimension}, "
        f"{n_epochs} эпох, батч={batch_size}, LR={learning_rate}, "
        f"w={variance_weight}, {gradient_tag}, omega={unknown_noise_threshold:.4f}"
    )


def print_epoch_statistics(
    epoch_index: int,
    n_epochs: int,
    train_loss: float,
    statistics: dict[str, float],
    elapsed_seconds: float,
) -> None:
    print(
        f"    Эпоха [{epoch_index + 1:02d}/{n_epochs:02d}] | "
        f"Train: {train_loss:9.4f} | "
        f"Val: {statistics['val_loss']:9.4f} "
        f"(bias {statistics['val_bias']:9.4f}) | "
        f"d_rg: {statistics['d_ratio_mean']:5.2f}±{statistics['d_ratio_std']:.2f} | "
        f"d_rmt: {statistics['d_rmt_mean']:5.2f}±{statistics['d_rmt_std']:.2f} | "
        f"d_th: {statistics['d_th_mean']:5.2f}±{statistics['d_th_std']:.2f} | "
        f"{elapsed_seconds:5.1f}s"
    )


def empty_training_history() -> dict:
    return {
        "epoch_index": [],
        "train_loss": [],
        "val_loss": [],
        "val_bias": [],
        "d_ratio_mean": [],
        "d_ratio_std": [],
        "d_rmt_mean": [],
        "d_rmt_std": [],
        "d_th_mean": [],
        "d_th_std": [],
        "elapsed_seconds": [],
    }


def append_training_history_entry(
    history: dict,
    epoch_index: int,
    train_loss: float,
    statistics: dict[str, float],
    elapsed_seconds: float,
) -> None:
    history["epoch_index"].append(int(epoch_index))
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(statistics["val_loss"]))
    history["val_bias"].append(float(statistics["val_bias"]))
    history["d_ratio_mean"].append(float(statistics["d_ratio_mean"]))
    history["d_ratio_std"].append(float(statistics["d_ratio_std"]))
    history["d_rmt_mean"].append(float(statistics["d_rmt_mean"]))
    history["d_rmt_std"].append(float(statistics["d_rmt_std"]))
    history["d_th_mean"].append(float(statistics["d_th_mean"]))
    history["d_th_std"].append(float(statistics["d_th_std"]))
    history["elapsed_seconds"].append(float(elapsed_seconds))


def run_training_and_validation_epoch(
    training_samples: np.ndarray,
    validation_samples: np.ndarray,
    raw_operators: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    detach_eigenvectors: bool,
    variance_weight: float,
    grad_clip_norm: float,
    unknown_noise_threshold: float,
    epoch_index: int,
    n_epochs: int,
    history: dict,
) -> dict[str, float]:
    start_time = time.perf_counter()
    train_loss = run_training_epoch(
        training_samples,
        raw_operators,
        optimizer,
        batch_size,
        device,
        detach_eigenvectors,
        variance_weight,
        grad_clip_norm,
    )
    statistics = evaluate_validation(
        validation_samples,
        raw_operators,
        unknown_noise_threshold,
        batch_size,
        device,
        variance_weight,
        print_spectrum=True,
    )
    elapsed_seconds = time.perf_counter() - start_time
    print_epoch_statistics(epoch_index, n_epochs, train_loss, statistics, elapsed_seconds)
    append_training_history_entry(
        history, epoch_index + 1, train_loss, statistics, elapsed_seconds
    )
    return statistics


def fit_qcml_geometry(
    meg_samples: np.ndarray,
    feature_dimension: int,
    hilbert_dimension: int,
    batch_size: int,
    device: torch.device,
    n_epochs: int,
    learning_rate: float,
    variance_weight: float,
    grad_clip_norm: float,
    detach_eigenvectors: bool,
    validation_fraction: float,
    validation_max_samples: int,
) -> tuple[torch.Tensor, dict]:
    training_samples, validation_samples = split_train_validation(
        meg_samples, validation_fraction, validation_max_samples, batch_size
    )
    raw_operators = initialize_operator_parameters(feature_dimension, hilbert_dimension, device)
    optimizer = torch.optim.AdamW([raw_operators], lr=learning_rate)
    unknown_noise_threshold = unknown_noise_threshold_coefficient(
        gramian_aspect_ratio(hilbert_dimension, feature_dimension)
    )
    print_optimization_start(
        n_epochs,
        batch_size,
        learning_rate,
        variance_weight,
        unknown_noise_threshold,
        detach_eigenvectors,
        hilbert_dimension,
        feature_dimension,
    )
    history = empty_training_history()
    best_val_loss = float("inf")
    best_epoch_index = 0
    for epoch_index in range(n_epochs):
        statistics = run_training_and_validation_epoch(
            training_samples,
            validation_samples,
            raw_operators,
            optimizer,
            batch_size,
            device,
            detach_eigenvectors,
            variance_weight,
            grad_clip_norm,
            unknown_noise_threshold,
            epoch_index,
            n_epochs,
            history,
        )
        if statistics["val_loss"] < best_val_loss:
            best_val_loss = statistics["val_loss"]
            best_epoch_index = epoch_index + 1
    history["best_val_loss"] = float(best_val_loss)
    history["best_epoch_index"] = int(best_epoch_index)
    print(
        f"[*] Оптимизация завершена. Лучший val_loss: {best_val_loss:.4f} "
        f"(эпоха {best_epoch_index}/{n_epochs})"
    )
    return raw_operators.detach(), history


def save_qcml_checkpoint(
    checkpoint_path: Path,
    raw_operators: torch.Tensor,
    normalization_parameters: dict,
    feature_dimension: int,
    hilbert_dimension: int,
    takens_embedding_dimension: int,
    normalization_mode: str,
    variance_weight: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    hermitian_operators = hermitian_operators_from_parameters(raw_operators).detach().cpu()
    payload = {
        "raw_matrices": raw_operators.cpu(),
        "a_hermitian": hermitian_operators,
        "norm_params": normalization_parameters,
        "config": {
            "d_features": feature_dimension,
            "n_hilbert": hilbert_dimension,
            "takens_m": takens_embedding_dimension,
            "norm_mode": normalization_mode,
            "w_variance": variance_weight,
        },
    }
    torch.save(payload, str(checkpoint_path))
    print(f"[+] Чекпоинт квантовой геометрии сохранен: {checkpoint_path}")


def load_qcml_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return payload["raw_matrices"].to(device)


def save_training_history(history_path: Path, history: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(history, str(history_path))
    print(f"[+] История обучения сохранена: {history_path}")


def reduce_takens_channel_leverage(
    mean_metric_diagonal: np.ndarray,
    embedding_dimension: int,
    physical_channel_count: int,
) -> np.ndarray:
    if embedding_dimension == 1:
        return mean_metric_diagonal.copy()
    reshaped = mean_metric_diagonal.reshape(embedding_dimension, physical_channel_count)
    return np.mean(reshaped, axis=0)


def print_rmt_configuration(
    aspect_ratio: float,
    median_under_null: float,
    threshold_coefficient: float,
) -> None:
    print(
        f"  [RMT] beta = {aspect_ratio:.4f} | MP-медиана mu_beta = {median_under_null:.4f} | "
        f"omega(beta) = {threshold_coefficient:.4f}"
    )


def fixed_size_eigenvalue_array(
    eigenvalues: np.ndarray,
    gramian_dimension: int,
) -> pa.FixedSizeListArray:
    flattened = pa.array(eigenvalues.ravel(), type=pa.float64())
    return pa.FixedSizeListArray.from_arrays(flattened, list_size=gramian_dimension)


def spectrum_table(
    sample_indices: list[int],
    eigenvalues: np.ndarray,
    metric_traces: np.ndarray,
    residuals: np.ndarray,
    ratio_gaps: np.ndarray,
    rmt_dimensions: np.ndarray,
    ground_eigenvalues: np.ndarray,
    first_eigenvalue_gaps: np.ndarray,
    local_variances: np.ndarray,
    gramian_dimension: int,
) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(sample_indices, type=pa.int64()),
            fixed_size_eigenvalue_array(eigenvalues, gramian_dimension),
            pa.array(metric_traces, type=pa.float64()),
            pa.array(residuals, type=pa.float64()),
            pa.array(ratio_gaps, type=pa.float64()),
            pa.array(rmt_dimensions, type=pa.float64()),
            pa.array(ground_eigenvalues, type=pa.float64()),
            pa.array(first_eigenvalue_gaps, type=pa.float64()),
            pa.array(local_variances, type=pa.float64()),
        ],
        names=[
            "sample_index",
            "eigenvalues",
            "metric_trace",
            "reconstruction_residual",
            "d_ratio_gap",
            "d_rmt",
            "ground_eigenvalue",
            "first_eigenvalue_gap",
            "local_variance",
        ],
    )


def write_chunk_to_parquet(
    writer: pq.ParquetWriter,
    buffers: dict[str, list],
    gramian_dimension: int,
) -> None:
    table = spectrum_table(
        buffers["sample_indices"],
        np.concatenate(buffers["eigenvalue_chunks"], axis=0),
        np.asarray(buffers["metric_traces"], dtype=np.float64),
        np.asarray(buffers["residuals"], dtype=np.float64),
        np.asarray(buffers["ratio_gaps"], dtype=np.float64),
        np.asarray(buffers["rmt_dimensions"], dtype=np.float64),
        np.asarray(buffers["ground_eigenvalues"], dtype=np.float64),
        np.asarray(buffers["first_eigenvalue_gaps"], dtype=np.float64),
        np.asarray(buffers["local_variances"], dtype=np.float64),
        gramian_dimension,
    )
    writer.write_table(table)


def spectrum_parquet_schema(gramian_dimension: int) -> pa.Schema:
    eigenvalue_column = pa.list_(pa.float64(), list_size=gramian_dimension)
    fields = [
        ("sample_index", pa.int64()),
        ("eigenvalues", eigenvalue_column),
        ("metric_trace", pa.float64()),
        ("reconstruction_residual", pa.float64()),
        ("d_ratio_gap", pa.float64()),
        ("d_rmt", pa.float64()),
        ("ground_eigenvalue", pa.float64()),
        ("first_eigenvalue_gap", pa.float64()),
        ("local_variance", pa.float64()),
    ]
    return pa.schema(fields)


def init_parquet_writer(output_path: Path, gramian_dimension: int) -> pq.ParquetWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(
        output_path,
        spectrum_parquet_schema(gramian_dimension),
        compression="snappy",
    )


def close_parquet_writer(writer: pq.ParquetWriter) -> None:
    writer.close()


def empty_inference_buffers() -> dict[str, list]:
    return {
        "sample_indices": [],
        "eigenvalue_chunks": [],
        "metric_traces": [],
        "residuals": [],
        "ratio_gaps": [],
        "rmt_dimensions": [],
        "ground_eigenvalues": [],
        "first_eigenvalue_gaps": [],
        "local_variances": [],
    }


def extend_inference_buffers(
    buffers: dict[str, list],
    start_sample_offset: int,
    batch_start: int,
    batch_end: int,
    active_eigenvalues: torch.Tensor,
    metric_traces: torch.Tensor,
    residuals: torch.Tensor,
    ratio_gap: torch.Tensor,
    rmt_dimension: torch.Tensor,
    ground_eigenvalue: torch.Tensor,
    first_eigenvalue_gap: torch.Tensor,
    local_variance: torch.Tensor,
) -> None:
    buffers["sample_indices"].extend(
        range(start_sample_offset + batch_start, start_sample_offset + batch_end)
    )
    buffers["eigenvalue_chunks"].append(active_eigenvalues.cpu().numpy())
    buffers["metric_traces"].extend(metric_traces.cpu().numpy().tolist())
    buffers["residuals"].extend(residuals.cpu().numpy().tolist())
    buffers["ratio_gaps"].extend(ratio_gap.cpu().numpy().tolist())
    buffers["rmt_dimensions"].extend(rmt_dimension.cpu().numpy().tolist())
    buffers["ground_eigenvalues"].extend(ground_eigenvalue.cpu().numpy().tolist())
    buffers["first_eigenvalue_gaps"].extend(first_eigenvalue_gap.cpu().numpy().tolist())
    buffers["local_variances"].extend(local_variance.cpu().numpy().tolist())


def write_and_clear_inference_buffers(
    writer: pq.ParquetWriter,
    buffers: dict[str, list],
    gramian_dimension: int,
) -> None:
    write_chunk_to_parquet(writer, buffers, gramian_dimension)
    buffers["sample_indices"].clear()
    buffers["eigenvalue_chunks"].clear()
    buffers["metric_traces"].clear()
    buffers["residuals"].clear()
    buffers["ratio_gaps"].clear()
    buffers["rmt_dimensions"].clear()
    buffers["ground_eigenvalues"].clear()
    buffers["first_eigenvalue_gaps"].clear()
    buffers["local_variances"].clear()
    torch.cuda.empty_cache()


def flush_inference_buffers_if_due(
    writer: pq.ParquetWriter,
    buffers: dict[str, list],
    chunk_flush_size: int,
    is_last_batch: bool,
    gramian_dimension: int,
) -> int:
    if len(buffers["sample_indices"]) < chunk_flush_size and not is_last_batch:
        return 0
    flushed_count = len(buffers["sample_indices"])
    write_and_clear_inference_buffers(writer, buffers, gramian_dimension)
    return flushed_count


def print_inference_progress(
    batch_end: int,
    total_sample_count: int,
    flushed_count: int,
) -> None:
    print(
        f"    Прогресс: {batch_end}/{total_sample_count} "
        f"({100.0 * batch_end / total_sample_count:.1f}%) | Записано: {flushed_count}"
    )


def project_and_residuals(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> dict[str, torch.Tensor]:
    projection = project_batch_to_point_cloud(feature_batch, hermitian_operators, sum_of_squared)
    projection["residuals"] = compute_projection_residuals(feature_batch, projection["projected_batch"])
    return projection


def spectrum_and_metric_diagonal(
    projected_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hamiltonian = core_hamiltonian_for_batch(projected_batch, hermitian_operators, sum_of_squared)
    ground_eigenvalue, ground_state, excited_eigenvalues, excited_states = ground_and_excited_eigenpairs(hamiltonian)
    eigenvalue_gaps = compute_eigenvalue_gaps(ground_eigenvalue, excited_eigenvalues)
    raw_transitions = compute_transition_matrix_elements(ground_state, excited_states, hermitian_operators)
    scaled_transitions = scale_transition_vectors_by_gap(raw_transitions, eigenvalue_gaps)
    transition_features = assemble_transition_feature_matrix(scaled_transitions)
    metric_diagonal = compute_analytic_metric_diagonal(transition_features)
    metric_traces = torch.sum(metric_diagonal, dim=-1)
    active_eigenvalues = active_eigenvalues_from_transition_features(transition_features)
    return active_eigenvalues, metric_diagonal, metric_traces


def dimension_estimates(
    active_eigenvalues: torch.Tensor,
    unknown_noise_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio_gap = compute_ratio_gap_dimension(active_eigenvalues)
    rmt_dimension = compute_rmt_dimension(active_eigenvalues, unknown_noise_threshold)
    return ratio_gap, rmt_dimension


@torch.no_grad()
def process_single_inference_chunk(
    feature_batch: torch.Tensor,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
    unknown_noise_threshold: float,
) -> dict[str, torch.Tensor]:
    projection = project_and_residuals(feature_batch, hermitian_operators, sum_of_squared)
    active_eigenvalues, metric_diagonal, metric_traces = spectrum_and_metric_diagonal(
        projection["projected_batch"], hermitian_operators, sum_of_squared
    )
    ratio_gap, rmt_dimension = dimension_estimates(active_eigenvalues, unknown_noise_threshold)
    return {
        "active_eigenvalues": active_eigenvalues,
        "metric_diagonal": metric_diagonal,
        "metric_traces": metric_traces,
        "residuals": projection["residuals"],
        "ratio_gap": ratio_gap,
        "rmt_dimension": rmt_dimension,
        "ground_eigenvalue": projection["ground_eigenvalue"],
        "first_eigenvalue_gap": projection["first_eigenvalue_gap"],
        "local_variance": projection["local_variance"],
    }


def run_inference_batches(
    data: np.ndarray,
    start_sample_offset: int,
    writer: pq.ParquetWriter,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
    unknown_noise_threshold: float,
    gramian_dimension: int,
    batch_size: int,
    chunk_flush_size: int,
    device: torch.device,
) -> torch.Tensor:
    total_sample_count, feature_dimension = data.shape
    accumulated_metric_diagonal = torch.zeros(feature_dimension, device=device, dtype=torch.float64)
    buffers = empty_inference_buffers()
    flushed_count = 0

    with torch.no_grad():
        for batch_start, batch_end in sequential_batch_slices(total_sample_count, batch_size):
            feature_batch = torch.from_numpy(data[batch_start:batch_end]).to(device)

            inference_results = process_single_inference_chunk(
                feature_batch, hermitian_operators, sum_of_squared, unknown_noise_threshold
            )

            accumulated_metric_diagonal += (
                inference_results["metric_diagonal"]
                / inference_results["metric_traces"].unsqueeze(1)
            ).sum(dim=0)

            extend_inference_buffers(
                buffers,
                start_sample_offset,
                batch_start,
                batch_end,
                inference_results["active_eigenvalues"],
                inference_results["metric_traces"],
                inference_results["residuals"],
                inference_results["ratio_gap"],
                inference_results["rmt_dimension"],
                inference_results["ground_eigenvalue"],
                inference_results["first_eigenvalue_gap"],
                inference_results["local_variance"],
            )

            flushed_count += flush_inference_buffers_if_due(
                writer,
                buffers,
                chunk_flush_size,
                batch_end == total_sample_count,
                gramian_dimension,
            )

            print_inference_progress(batch_end, total_sample_count, flushed_count)

    return accumulated_metric_diagonal


def write_spectrum_to_parquet(
    data: np.ndarray,
    start_sample_offset: int,
    output_parquet_path: Path,
    hermitian_operators: torch.Tensor,
    sum_of_squared: torch.Tensor,
    unknown_noise_threshold: float,
    gramian_dimension: int,
    batch_size: int,
    chunk_flush_size: int,
    device: torch.device,
) -> torch.Tensor:
    writer = init_parquet_writer(output_parquet_path, gramian_dimension)
    accumulated_metric_diagonal = run_inference_batches(
        data,
        start_sample_offset,
        writer,
        hermitian_operators,
        sum_of_squared,
        unknown_noise_threshold,
        gramian_dimension,
        batch_size,
        chunk_flush_size,
        device,
    )
    close_parquet_writer(writer)
    return accumulated_metric_diagonal


def stream_dataset_inference(
    data: np.ndarray,
    start_sample_offset: int,
    raw_operators: torch.Tensor,
    output_parquet_path: Path,
    hilbert_dimension: int,
    feature_dimension: int,
    takens_embedding_dimension: int,
    batch_size: int,
    chunk_flush_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    physical_channel_count = feature_dimension // takens_embedding_dimension
    gramian_dimension = 2 * (hilbert_dimension - 1)
    aspect_ratio = gramian_aspect_ratio(hilbert_dimension, feature_dimension)
    unknown_noise_threshold = unknown_noise_threshold_coefficient(aspect_ratio)
    print_rmt_configuration(
        aspect_ratio,
        marchenko_pastur_median(aspect_ratio),
        unknown_noise_threshold,
    )

    hermitian_operators, sum_of_squared = build_hermitian_and_squared_operators(raw_operators)

    accumulated_metric_diagonal = write_spectrum_to_parquet(
        data,
        start_sample_offset,
        output_parquet_path,
        hermitian_operators,
        sum_of_squared,
        unknown_noise_threshold,
        gramian_dimension,
        batch_size,
        chunk_flush_size,
        device,
    )

    mean_metric_diagonal = (accumulated_metric_diagonal / float(data.shape[0])).cpu().numpy()
    physical_leverage = reduce_takens_channel_leverage(
        mean_metric_diagonal,
        takens_embedding_dimension,
        physical_channel_count,
    )
    return mean_metric_diagonal, physical_leverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCML MEG Engine v3")
    parser.add_argument("--fif_path", type=str, required=True, help="Путь к файлу МЭГ (.fif)")
    parser.add_argument("--n_hilbert", type=int, default=24, help="Размерность Гильбертова пространства N")
    parser.add_argument("--takens_m", type=int, default=1, help="Размерность вложения Такенса")
    parser.add_argument("--takens_tau_ms", type=float, default=25.0, help="Задержка Такенса в миллисекундах")
    parser.add_argument("--norm_mode", type=str, choices=["block_rms", "channel_zscore"], default="block_rms", help="Режим нормализации")
    parser.add_argument("--w_variance", type=float, default=0.0, help="Вес регуляризации локальной дисперсии")
    parser.add_argument("--batch_size", type=int, default=512, help="Размер батча")
    parser.add_argument("--n_epochs", type=int, default=20, help="Число эпох обучения")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Скорость обучения")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Норма отсечения градиентов")
    parser.add_argument("--chunk_flush_size", type=int, default=50_000, help="Размер буфера сброса в Parquet")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Доля валидации от числа сэмплов")
    parser.add_argument("--val_max_samples", type=int, default=100_000, help="Максимум сэмплов валидации")
    parser.add_argument("--seed", type=int, default=666, help="Сид генераторов случайных чисел")
    parser.add_argument("--device", type=str, default=None, help="Устройство: cuda или cpu (авто по умолчанию)")
    parser.add_argument("--detach_eigenvectors", action="store_true", help="Заморозить psi_0 от A (дешевле, менее устойчиво)")
    parser.add_argument("--checkpoint_dir", type=str, default="04_processed_db", help="Папка чекпоинтов")
    parser.add_argument("--output_dir", type=str, default="04_processed_db", help="Папка выходных файлов")
    return parser.parse_args()


def resolve_device(device_string: str | None) -> torch.device:
    if device_string is not None:
        return torch.device(device_string)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_random_seeds(args.seed)

    fif_path = Path(args.fif_path)
    recording = load_preprocessed_raw_fif(fif_path)
    sampling_frequency = meg_sampling_frequency(recording)
    magnetometer_indices = magnetometer_channel_indices(recording)
    gradiometer_indices = gradiometer_channel_indices(recording)
    selected_channel_indices = np.concatenate([magnetometer_indices, gradiometer_indices])
    raw_samples = extract_meg_samples(recording, selected_channel_indices)
    del recording
    gc.collect()

    if args.norm_mode == "channel_zscore":
        normalized_samples, channel_standard_deviations = normalize_channels_by_zscore(raw_samples)
        normalization_parameters = {
            "mode": "channel_zscore",
            "stds": channel_standard_deviations,
        }
    else:
        normalized_samples, magnetometer_rms, gradiometer_rms = (
            normalize_magnetometers_and_gradiometers_by_block_rms(
                raw_samples, len(magnetometer_indices)
            )
        )
        normalization_parameters = {
            "mode": "block_rms",
            "std_mag": magnetometer_rms,
            "std_grad": gradiometer_rms,
        }
    del raw_samples
    gc.collect()

    lag_samples = (
        takens_lag_in_samples(args.takens_tau_ms, sampling_frequency)
        if args.takens_m > 1
        else 0
    )
    embedded_samples, dropped_leading_samples = build_takens_delay_embedding(
        normalized_samples, args.takens_m, lag_samples
    )
    del normalized_samples
    gc.collect()

    feature_dimension = embedded_samples.shape[1]
    subject_stem = fif_path.stem.replace("_raw_preprocessed", "").replace("_raw", "")
    gradient_tag = "detach" if args.detach_eigenvectors else "grad"
    suffix_tag = (
        f"D{feature_dimension}_N{args.n_hilbert}_m{args.takens_m}"
        f"_tau{int(args.takens_tau_ms)}ms_{args.norm_mode}"
        f"_w{args.w_variance}_{gradient_tag}"
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"{subject_stem}_geometry_{suffix_tag}.pt"
    spectrum_file = output_dir / f"{subject_stem}_spectrum_{suffix_tag}.parquet"
    leverage_file = output_dir / f"{subject_stem}_leverage_{suffix_tag}.pt"
    history_file = output_dir / f"{subject_stem}_history_{suffix_tag}.pt"

    if checkpoint_file.exists():
        raw_operators = load_qcml_checkpoint(checkpoint_file, device)
    else:
        raw_operators, training_history = fit_qcml_geometry(
            embedded_samples,
            feature_dimension=feature_dimension,
            hilbert_dimension=args.n_hilbert,
            batch_size=args.batch_size,
            device=device,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            variance_weight=args.w_variance,
            grad_clip_norm=args.grad_clip_norm,
            detach_eigenvectors=args.detach_eigenvectors,
            validation_fraction=args.val_fraction,
            validation_max_samples=args.val_max_samples,
        )
        save_qcml_checkpoint(
            checkpoint_file,
            raw_operators,
            normalization_parameters,
            feature_dimension=feature_dimension,
            hilbert_dimension=args.n_hilbert,
            takens_embedding_dimension=args.takens_m,
            normalization_mode=args.norm_mode,
            variance_weight=args.w_variance,
        )
        save_training_history(history_file, training_history)

    mean_metric_diagonal, physical_leverage = stream_dataset_inference(
        data=embedded_samples,
        start_sample_offset=dropped_leading_samples,
        raw_operators=raw_operators,
        output_parquet_path=spectrum_file,
        hilbert_dimension=args.n_hilbert,
        feature_dimension=feature_dimension,
        takens_embedding_dimension=args.takens_m,
        batch_size=args.batch_size,
        chunk_flush_size=args.chunk_flush_size,
        device=device,
    )

    torch.save(
        {
            "channel_leverage": physical_leverage,
            "metric_diagonal_full": mean_metric_diagonal,
            "mag_picks": magnetometer_indices,
            "grad_picks": gradiometer_indices,
            "config": {
                "d_features": feature_dimension,
                "n_hilbert": args.n_hilbert,
                "takens_m": args.takens_m,
                "takens_tau_ms": args.takens_tau_ms,
                "norm_mode": args.norm_mode,
                "w_variance": args.w_variance,
            },
        },
        str(leverage_file),
    )
    print(f"[+] Топографический вклад сенсоров сохранен: {leverage_file}")
    print(f"[✓] Обработка записи {subject_stem} завершена успешно.")


if __name__ == "__main__":
    main()