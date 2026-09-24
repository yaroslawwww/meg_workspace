import os
import gc
import math
import argparse
from pathlib import Path
from typing import Tuple, Generator, Optional

import numpy as np
import mne
import torch
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ И КОНФИГУРАЦИЯ
# =============================================================================

N_FEATURES: int = 306
N_HILBERT: int = 24
K_ACTIVE: int = 2 * (N_HILBERT - 1)
BATCH_SIZE: int = 512
N_EPOCHS: int = 20
N_SAMPLES_TRAIN: int = -1
LEARNING_RATE: float = 1e-4
W_VARIANCE: float = 0.0
W_COMMUTATOR: float = 0.0
GRAD_CLIP_NORM: float = 1.0
CHUNK_FLUSH_SIZE: int = 50000
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_hyperparameters(
    n_features: int = 306,
    n_hilbert: int = 24,
    batch_size: int = 512,
    n_epochs: int = 20,
    n_samples_train: int = -1,
    learning_rate: float = 1e-4,
    w_variance: float = 0.0,
    w_commutator: float = 0.0,
    grad_clip_norm: float = 1.0,
    chunk_flush_size: int = 50000,
    device_str: Optional[str] = None
) -> None:
    """Конфигурация параметров QCML с автоматическим расчетом теоретического ранга K_ACTIVE."""
    global N_FEATURES, N_HILBERT, K_ACTIVE, BATCH_SIZE, N_EPOCHS, N_SAMPLES_TRAIN
    global LEARNING_RATE, W_VARIANCE, W_COMMUTATOR, GRAD_CLIP_NORM, CHUNK_FLUSH_SIZE, DEVICE

    N_FEATURES = int(n_features)
    N_HILBERT = int(n_hilbert)
    K_ACTIVE = 2 * (N_HILBERT - 1)
    BATCH_SIZE = int(batch_size)
    N_EPOCHS = int(n_epochs)
    N_SAMPLES_TRAIN = int(n_samples_train)
    LEARNING_RATE = float(learning_rate)
    W_VARIANCE = float(w_variance)
    W_COMMUTATOR = float(w_commutator)
    GRAD_CLIP_NORM = float(grad_clip_norm)
    CHUNK_FLUSH_SIZE = int(chunk_flush_size)

    if device_str is not None:
        DEVICE = torch.device(device_str)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_mode_str = "ПОЛНЫЙ ДАТАСЕТ (100%)" if N_SAMPLES_TRAIN <= 0 else f"{N_SAMPLES_TRAIN} точек"
    print(
        f"[Config] Параметры обновлены:\n"
        f"  * Сенсоры D = {N_FEATURES}, Hilbert N = {N_HILBERT} (Активных мод K = {K_ACTIVE})\n"
        f"  * Батч = {BATCH_SIZE}, Эпох = {N_EPOCHS}, Выборка = {train_mode_str}\n"
        f"  * LR = {LEARNING_RATE}, Device = {DEVICE}\n"
        f"  * Сброс в Parquet чанками по {CHUNK_FLUSH_SIZE} точек"
    )


# =============================================================================
# БЛОК 1. ЗАГРУЗКА И НОРМАЛИЗАЦИЯ ДАННЫХ
# =============================================================================

def load_preprocessed_raw_fif(fif_path: str, preload: bool = True) -> mne.io.Raw:
    """Чтение файла .fif без модификации сигналов."""
    return mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)


def extract_meg_channels_tensor(raw: mne.io.Raw) -> Tuple[np.ndarray, float, float]:
    """
    Извлечение 306 каналов МЭГ с центрированием и фиксацией RMS масштабов.
    Возвращает матрицу данных и коэффициенты нормировки.
    """
    mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False, ecg=False)
    grad_picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, ecg=False)
    all_picks = np.concatenate([mag_picks, grad_picks])

    data = raw.get_data(picks=all_picks).T.astype(np.float32)
    n_mags = len(mag_picks)

    data -= np.mean(data, axis=0, keepdims=True)

    std_mag = float(np.sqrt(np.mean(data[:, :n_mags] ** 2)))
    std_grad = float(np.sqrt(np.mean(data[:, n_mags:] ** 2)))

    data[:, :n_mags] /= (std_mag + 1e-15)
    data[:, n_mags:] /= (std_grad + 1e-15)

    print(
        f"[QCML Ingestion] Данные согласованы с требованиями алгоритма:\n"
        f"  * 102 Магнитометра: масштабированы на 1/{std_mag:.3e}\n"
        f"  * 204 Градиентометра: масштабированы на 1/{std_grad:.3e}\n"
        f"  * Итоговый диапазон x: [{data.min():.2f}, {data.max():.2f}]"
    )
    return data, std_mag, std_grad


def generate_sequential_batch_slices(n_samples: int) -> Generator[Tuple[int, int], None, None]:
    """Генератор последовательных временных срезов [start, end) для инференса."""
    for start_idx in range(0, n_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_samples)
        yield start_idx, end_idx


# =============================================================================
# БЛОК 2. ОПЕРАТОРЫ И АЛГЕБРАИЧЕСКИЕ ИНВАРИАНТЫ
# =============================================================================

def init_raw_matrix_parameters() -> torch.Tensor:
    """Инициализация комплексных обучаемых параметров (D, N, N) на GPU."""
    scale = 1.0 / math.sqrt(N_FEATURES * N_HILBERT)
    real_part = torch.randn(N_FEATURES, N_HILBERT, N_HILBERT, device=DEVICE) * scale
    imag_part = torch.randn(N_FEATURES, N_HILBERT, N_HILBERT, device=DEVICE) * scale
    raw_params = torch.complex(real_part, imag_part)
    raw_params.requires_grad_(True)
    return raw_params


def build_hermitian_matrices(raw_matrices: torch.Tensor) -> torch.Tensor:
    """Дифференцируемая сборка эрмитовых матриц A_k = 0.5 * (M_k + M_k^H)."""
    return 0.5 * (raw_matrices + raw_matrices.conj().transpose(-1, -2))


def precompute_sum_squared_operators(hermitian_matrices: torch.Tensor) -> torch.Tensor:
    """Операторный инвариант K = 0.5 * sum_{k=1}^D A_k^2 размера (N, N)."""
    A_squared = torch.matmul(hermitian_matrices, hermitian_matrices)
    return 0.5 * torch.sum(A_squared, dim=0)


def compute_commutator_penalty(hermitian_matrices: torch.Tensor) -> torch.Tensor:
    """Регуляризатор некоммутативности операторов."""
    idx_i = torch.randint(0, N_FEATURES, (N_FEATURES,), device=DEVICE)
    idx_j = torch.randint(0, N_FEATURES, (N_FEATURES,), device=DEVICE)
    mask = idx_i != idx_j
    i_s, j_s = idx_i[mask], idx_j[mask]

    A_i = hermitian_matrices[i_s]
    A_j = hermitian_matrices[j_s]
    comm = torch.matmul(A_i, A_j) - torch.matmul(A_j, A_i)
    comm_norm_sq = torch.sum(torch.abs(comm)**2)
    return -comm_norm_sq / float(len(i_s) + 1e-7)


# =============================================================================
# БЛОК 3. ГАМИЛЬТОНИАН ОШИБКИ И ВОЛНОВЫЕ ФУНКЦИИ
# =============================================================================

def compute_linear_operator_contraction(
    x_batch: torch.Tensor, hermitian_matrices: torch.Tensor
) -> torch.Tensor:
    """Линейный член sum_{k=1}^D x_{i,k} A_k формы (B, N, N)."""
    return torch.einsum('bd, dnm -> bnm', x_batch.to(torch.complex64), hermitian_matrices)


def compute_point_norms_squared(x_batch: torch.Tensor) -> torch.Tensor:
    """Полусумма квадратов координат 0.5 * ||x_i||^2 формы (B, 1, 1)."""
    return 0.5 * torch.sum(x_batch**2, dim=-1, keepdim=True).unsqueeze(-1)


def assemble_error_hamiltonian(
    K_operator: torch.Tensor,
    linear_contraction: torch.Tensor,
    point_norms_sq: torch.Tensor
) -> torch.Tensor:
    """Сборка Гамильтониана ошибки H(x) формы (B, N, N)."""
    eye = torch.eye(N_HILBERT, device=DEVICE, dtype=torch.complex64).unsqueeze(0)
    H = K_operator.unsqueeze(0) - linear_contraction + point_norms_sq.to(torch.complex64) * eye
    return 0.5 * (H + H.conj().transpose(-1, -2))


def solve_ground_state_only(H_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Вычисление энергии E_0 (B,) и вакуумного состояния psi_0 (B, N)."""
    evals, evecs = torch.linalg.eigh(H_batch)
    return evals[:, 0], evecs[:, :, 0]


def solve_full_hamiltonian_eigensystem(
    H_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Полное спектральное разложение Гамильтониана H(x)."""
    evals, evecs = torch.linalg.eigh(H_batch)
    return evals[:, 0], evecs[:, :, 0], evals[:, 1:], evecs[:, :, 1:]


# =============================================================================
# БЛОК 4. ФУНКЦИОНАЛ ПОТЕРЬ И ОПТИМИЗАЦИЯ
# =============================================================================

def compute_coordinate_expectations(
    psi_0_batch: torch.Tensor, hermitian_matrices: torch.Tensor
) -> torch.Tensor:
    """Квантовые координаты точки y_k(x) = <psi_0| A_k |psi_0> формы (B, D)."""
    return torch.einsum('bn, dnm, bm -> bd', psi_0_batch.conj(), hermitian_matrices, psi_0_batch).real


def compute_local_quantum_variance(
    psi_0_batch: torch.Tensor,
    y_coords: torch.Tensor,
    K_operator: torch.Tensor
) -> torch.Tensor:
    """Суммарная квантовая дисперсия sigma^2(x) = 2 <psi_0| K |psi_0> - ||y||^2."""
    exp_2K = 2.0 * torch.einsum('bn, nm, bm -> b', psi_0_batch.conj(), K_operator, psi_0_batch).real
    y_norm_sq = torch.sum(y_coords**2, dim=-1)
    return torch.clamp(exp_2K - y_norm_sq, min=0.0)


def compute_bias_loss(x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
    """Геометрическая невязка ||y(x) - x||^2."""
    return torch.mean(torch.sum((y_batch - x_batch)**2, dim=-1))


def compute_qcml_composite_loss(
    bias_loss: torch.Tensor,
    variance_batch: torch.Tensor,
    comm_penalty: torch.Tensor
) -> torch.Tensor:
    """Итоговый функционал потерь QCML."""
    total = bias_loss
    if W_VARIANCE > 0.0:
        total = total + W_VARIANCE * torch.mean(variance_batch)
    if W_COMMUTATOR > 0.0:
        total = total + W_COMMUTATOR * comm_penalty
    return total


def execute_gradient_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    raw_matrices: torch.Tensor
) -> float:
    """Обратное распространение ошибки с ограничением нормы градиента."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([raw_matrices], max_norm=GRAD_CLIP_NORM)
    optimizer.step()
    return loss.item()


# =============================================================================
# БЛОК 5. ПРОЕКЦИЯ НА КВАНТОВОЕ МНОГООБРАЗИЕ
# =============================================================================

@torch.no_grad()
def project_batch_to_point_cloud(
    x_batch: torch.Tensor,
    hermitian_matrices: torch.Tensor,
    K_operator: torch.Tensor
) -> torch.Tensor:
    """Проекция входных МЭГ точек на квантовое многообразие: x -> y(x)."""
    linear = compute_linear_operator_contraction(x_batch, hermitian_matrices)
    norms_sq = compute_point_norms_squared(x_batch)
    H = assemble_error_hamiltonian(K_operator, linear, norms_sq)
    _, psi_0 = solve_ground_state_only(H)
    return compute_coordinate_expectations(psi_0, hermitian_matrices)


def compute_projection_residuals(x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
    """Евклидова норма отсеченного шума ||y(x) - x|| формы (B,)."""
    return torch.linalg.norm(y_batch - x_batch, dim=-1)


# =============================================================================
# БЛОК 6. ТЕНЗОР КВАНТОВОЙ МЕТРИКИ g(y)
# =============================================================================

def compute_energy_gaps(E_0: torch.Tensor, E_excited: torch.Tensor) -> torch.Tensor:
    """Разности энергетических уровней Delta E_n = E_n - E_0 формы (B, N-1)."""
    return torch.clamp(E_excited - E_0.unsqueeze(-1), min=1e-7)


def compute_transition_matrix_elements(
    psi_0: torch.Tensor,
    psi_excited: torch.Tensor,
    hermitian_matrices: torch.Tensor
) -> torch.Tensor:
    """Матричные элементы перехода V_{n, mu} = <psi_0 | A_mu | psi_n> формы (B, N-1, D)."""
    psi_n_trans = psi_excited.permute(0, 2, 1)
    return torch.einsum('bn, dnm, bkm -> bkd', psi_0.conj(), hermitian_matrices, psi_n_trans)


def scale_transition_vectors_by_gap(
    transition_vectors: torch.Tensor, energy_gaps: torch.Tensor
) -> torch.Tensor:
    """Масштабирование элементов перехода на sqrt(2 / Delta E_n)."""
    scales = torch.sqrt(2.0 / energy_gaps).unsqueeze(-1)
    return transition_vectors * scales


def assemble_quantum_metric_tensor(scaled_transitions: torch.Tensor) -> torch.Tensor:
    """Сборка матрицы метрики g_{\mu\nu} формы (B, D, D)."""
    V_r = scaled_transitions.real
    V_i = scaled_transitions.imag
    return torch.bmm(V_r.transpose(1, 2), V_r) + torch.bmm(V_i.transpose(1, 2), V_i)


def enforce_metric_symmetry_inplace(metric_batch: torch.Tensor) -> torch.Tensor:
    """Симметризация матрицы метрики: 0.5 * (g + g^T)."""
    return 0.5 * (metric_batch + metric_batch.transpose(-1, -2))


def compute_metric_trace_vector(metric_batch: torch.Tensor) -> torch.Tensor:
    """Расчет следа квантовой метрики Tr(g(y)) формы (B,)."""
    return torch.diagonal(metric_batch, dim1=-2, dim2=-1).sum(dim=-1)


# =============================================================================
# БЛОК 7. ИЗВЛЕЧЕНИЕ АКТИВНОГО СПЕКТРА И ВЛИЯТЕЛЬНОСТИ КАНАЛОВ
# =============================================================================

def extract_active_metric_spectrum(metric_evals: torch.Tensor, k_modes: int) -> torch.Tensor:
    """
    Извлечение K = 2*(N-1) старших собственных значений метрики (по убыванию).
    Отсекает тривиальные нули размерности D - 2*(N-1). Форма: (B, K).
    """
    evals_desc = torch.flip(metric_evals, dims=[-1])
    return evals_desc[:, :k_modes]


def compute_continuous_participation_ratio(evals_active: torch.Tensor) -> torch.Tensor:
    """
    Непрерывная эффективная размерность (Participation Ratio): (sum lambda)^2 / sum lambda^2.
    Гладкая оценка размерности без дискретных порогов. Форма: (B,).
    """
    sum_evals = torch.sum(evals_active, dim=-1)
    sum_sq_evals = torch.sum(evals_active**2, dim=-1)
    return (sum_evals**2) / (sum_sq_evals + 1e-12)


def compute_batch_channel_leverage(evecs: torch.Tensor, k_modes: int) -> torch.Tensor:
    """
    Диагональ проектора касательного расслоения строго по формуле плана:
    w_k = sum_{m=1}^K |v_{m, k}|^2.
    Форма: (B, D). Не содержит эвристических весов и констант отсечения.
    """
    top_evecs = evecs[:, :, -k_modes:]
    return torch.sum(top_evecs**2, dim=-1)


# =============================================================================
# БЛОК 8. ИНФЕРЕНС-ШАГ И ОБУЧЕНИЕ
# =============================================================================

@torch.no_grad()
def process_single_inference_chunk(
    x_chunk: torch.Tensor,
    hermitian_matrices: torch.Tensor,
    K_operator: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Полный расчет активного спектра, следа, невязки, непрерывной размерности и влиятельности каналов."""
    y_chunk = project_batch_to_point_cloud(x_chunk, hermitian_matrices, K_operator)
    residuals = compute_projection_residuals(x_chunk, y_chunk)

    linear_y = compute_linear_operator_contraction(y_chunk, hermitian_matrices)
    norms_y = compute_point_norms_squared(y_chunk)
    H_y = assemble_error_hamiltonian(K_operator, linear_y, norms_y)

    E_0, psi_0, E_excited, psi_excited = solve_full_hamiltonian_eigensystem(H_y)
    delta_E = compute_energy_gaps(E_0, E_excited)
    V = compute_transition_matrix_elements(psi_0, psi_excited, hermitian_matrices)
    V_scaled = scale_transition_vectors_by_gap(V, delta_E)

    g = assemble_quantum_metric_tensor(V_scaled)
    g = enforce_metric_symmetry_inplace(g)

    traces = compute_metric_trace_vector(g)
    evals, evecs = torch.linalg.eigh(g)

    evals_active = extract_active_metric_spectrum(evals, k_modes=K_ACTIVE)
    pr_dimension = compute_continuous_participation_ratio(evals_active)
    channel_leverage = compute_batch_channel_leverage(evecs, k_modes=K_ACTIVE)

    return evals_active, traces, residuals, pr_dimension, channel_leverage


def run_training_epoch(
    meg_data: np.ndarray,
    raw_matrices: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """Выполнение одной эпохи стохастического обучения геометрии QCML."""
    T, _ = meg_data.shape

    if N_SAMPLES_TRAIN <= 0 or N_SAMPLES_TRAIN >= T:
        sample_indices = np.random.permutation(T)
    else:
        sample_indices = np.random.choice(T, size=N_SAMPLES_TRAIN, replace=False)

    n_samples_epoch = len(sample_indices)
    total_loss = 0.0
    n_batches = 0

    for start_idx in range(0, n_samples_epoch, BATCH_SIZE):
        batch_idx = sample_indices[start_idx : start_idx + BATCH_SIZE]
        x_batch = torch.from_numpy(meg_data[batch_idx]).to(DEVICE)

        A = build_hermitian_matrices(raw_matrices)
        K = precompute_sum_squared_operators(A)

        with torch.no_grad():
            linear = compute_linear_operator_contraction(x_batch, A)
            norms_sq = compute_point_norms_squared(x_batch)
            H = assemble_error_hamiltonian(K, linear, norms_sq)
            _, psi_0 = solve_ground_state_only(H)

        y = compute_coordinate_expectations(psi_0, A)
        var = compute_local_quantum_variance(psi_0, y, K)

        loss_bias = compute_bias_loss(x_batch, y)
        comm_pen = compute_commutator_penalty(A) if W_COMMUTATOR > 0.0 else torch.tensor(0.0, device=DEVICE)
        loss = compute_qcml_composite_loss(loss_bias, var, comm_pen)

        loss_val = execute_gradient_step(optimizer, loss, raw_matrices)
        total_loss += loss_val
        n_batches += 1

    return total_loss / max(n_batches, 1)


def fit_qcml_geometry(meg_data: np.ndarray) -> torch.Tensor:
    """Главный цикл оптимизации матричной конфигурации QCML."""
    raw_matrices = init_raw_matrix_parameters()
    optimizer = torch.optim.AdamW([raw_matrices], lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        epoch_loss = run_training_epoch(
            meg_data=meg_data,
            raw_matrices=raw_matrices,
            optimizer=optimizer
        )
        print(f"[*] Эпоха [{epoch+1:02d}/{N_EPOCHS:02d}] | QCML Loss: {epoch_loss:.6f}")

    return raw_matrices.detach()


# =============================================================================
# БЛОК 9. ХРАНЕНИЕ И СЕРИАЛИЗАЦИЯ (PYARROW PARQUET И ЧЕКПОИНТЫ)
# =============================================================================

def init_parquet_writer(output_path: Path) -> pq.ParquetWriter:
    """Инициализация потокового Parquet со спектром и Participation Ratio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_type = pa.list_(pa.float32(), list_size=K_ACTIVE)
    schema = pa.schema([
        ("sample_index", pa.int64()),
        ("eigenvalues", list_type),
        ("metric_trace", pa.float32()),
        ("reconstruction_residual", pa.float32()),
        ("participation_ratio", pa.float32())
    ])
    return pq.ParquetWriter(output_path, schema, compression="snappy")


def write_chunk_to_parquet(
    writer: pq.ParquetWriter,
    indices: np.ndarray,
    evals: np.ndarray,
    traces: np.ndarray,
    residuals: np.ndarray,
    pr_values: np.ndarray
) -> None:
    """Потоковая запись порции векторов собственных значений без Python-циклов."""
    evals_flat = pa.array(evals.ravel(), type=pa.float32())
    evals_fixed = pa.FixedSizeListArray.from_arrays(evals_flat, list_size=K_ACTIVE)

    table = pa.Table.from_arrays(
        [
            pa.array(indices, type=pa.int64()),
            evals_fixed,
            pa.array(traces, type=pa.float32()),
            pa.array(residuals, type=pa.float32()),
            pa.array(pr_values, type=pa.float32())
        ],
        names=["sample_index", "eigenvalues", "metric_trace", "reconstruction_residual", "participation_ratio"]
    )
    writer.write_table(table)


def close_parquet_writer(writer: pq.ParquetWriter) -> None:
    """Закрытие потока записи Parquet."""
    writer.close()


def save_qcml_checkpoint(
    checkpoint_path: str,
    raw_matrices: torch.Tensor,
    std_mag: float = 1.0,
    std_grad: float = 1.0
) -> None:
    """Сохранение состояния операторов и параметров нормализации."""
    A_herm = build_hermitian_matrices(raw_matrices).detach().cpu()
    payload = {
        "raw_matrices": raw_matrices.cpu(),
        "A_hermitian": A_herm,
        "normalization": {
            "std_mag": std_mag,
            "std_grad": std_grad
        }
    }
    torch.save(payload, checkpoint_path)


def load_qcml_checkpoint(checkpoint_path: str) -> torch.Tensor:
    """Загрузка матрицы геометрии из файла чекпоинта на целевой DEVICE."""
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    return payload["raw_matrices"].to(DEVICE)


# =============================================================================
# БЛОК 10. ПОТОКОВЫЙ КОНВЕЙЕР ИНФЕРЕНСА
# =============================================================================

def stream_dataset_inference(
    meg_data: np.ndarray,
    raw_matrices: torch.Tensor,
    output_parquet_path: str
) -> np.ndarray:
    """Потоковый расчет активного спектра и агрегация вектора влиятельности сенсоров."""
    T, _ = meg_data.shape
    out_path = Path(output_parquet_path)
    writer = init_parquet_writer(out_path)

    A = build_hermitian_matrices(raw_matrices)
    K = precompute_sum_squared_operators(A)

    buf_indices = []
    buf_evals = []
    buf_trace = []
    buf_res = []
    buf_pr = []
    total_flushed = 0

    accumulated_leverage = torch.zeros(N_FEATURES, device=DEVICE, dtype=torch.float32)

    print(f"[*] Старт потокового инференса: T={T} отсчетов. Активных мод K={K_ACTIVE}...")

    with torch.no_grad():
        for start_idx, end_idx in generate_sequential_batch_slices(T):
            x_batch = torch.from_numpy(meg_data[start_idx:end_idx]).to(DEVICE)
            evals_act, traces, residuals, pr, batch_leverage = process_single_inference_chunk(x_batch, A, K)

            accumulated_leverage += batch_leverage.sum(dim=0)

            buf_indices.extend(range(start_idx, end_idx))
            buf_evals.append(evals_act.cpu().numpy())
            buf_trace.extend(traces.cpu().numpy().tolist())
            buf_res.extend(residuals.cpu().numpy().tolist())
            buf_pr.extend(pr.cpu().numpy().tolist())

            if len(buf_indices) >= CHUNK_FLUSH_SIZE or end_idx == T:
                evals_matrix = np.concatenate(buf_evals, axis=0)

                write_chunk_to_parquet(
                    writer,
                    np.array(buf_indices, dtype=np.int64),
                    evals_matrix,
                    np.array(buf_trace, dtype=np.float32),
                    np.array(buf_res, dtype=np.float32),
                    np.array(buf_pr, dtype=np.float32)
                )
                total_flushed += len(buf_indices)
                buf_indices.clear()
                buf_evals.clear()
                buf_trace.clear()
                buf_res.clear()
                buf_pr.clear()

                torch.cuda.empty_cache()
                progress = (end_idx / T) * 100.0
                print(f"    Прогресс: {end_idx}/{T} ({progress:.2f}%) | Записано: {total_flushed}")

    close_parquet_writer(writer)
    print(f"[+] Активный спектр квантовой метрики сохранен: {out_path.resolve()}")

    global_leverage = (accumulated_leverage / float(T)).cpu().numpy()
    return global_leverage


# =============================================================================
# ТОЧКА ВХОДА (CLI ДЛЯ HPC SLURM)
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="QCML MEG Intrinsic Dimension Engine")
    parser.add_argument("--fif_path", type=str, default="01_raw_data/type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif")
    parser.add_argument("--n_hilbert", type=int, default=24, help="Размерность Гильбертова пространства N (12..100)")
    parser.add_argument("--checkpoint_dir", type=str, default="04_processed_db/type1_features")
    parser.add_argument("--output_dir", type=str, default="04_processed_db/type1_features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    configure_hyperparameters(
        n_features=306,
        n_hilbert=args.n_hilbert,
        batch_size=512,
        n_epochs=20,
        n_samples_train=-1,
        learning_rate=1e-4,
        w_variance=0.0,
        w_commutator=0.0,
        grad_clip_norm=1.0,
        chunk_flush_size=50000,
        device_str="cuda" if torch.cuda.is_available() else "cpu"
    )

    fif_p = Path(args.fif_path)
    subject_stem = fif_p.stem.replace("_raw_preprocessed", "")
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)

    checkpoint_file = ckpt_dir / f"{subject_stem}_qcml_geometry_N{args.n_hilbert}.pt"
    out_parquet = out_dir / f"{subject_stem}_metric_spectrum_N{args.n_hilbert}.parquet"
    out_leverage_file = out_dir / f"{subject_stem}_channel_leverage_N{args.n_hilbert}.pt"

    print(f"[*] Чтение предобработанных МЭГ сигналов: {fif_p}")
    raw = load_preprocessed_raw_fif(str(fif_p), preload=True)
    meg_signals, std_mag, std_grad = extract_meg_channels_tensor(raw)
    del raw
    gc.collect()

    if checkpoint_file.exists():
        print(f"[+] Найден готовый чекпоинт {checkpoint_file}. Загружаем геометрию...")
        raw_matrices = load_qcml_checkpoint(str(checkpoint_file))
    else:
        print(f"[*] Чекпоинт не найден. Запуск обучения QCML (N={args.n_hilbert})...")
        raw_matrices = fit_qcml_geometry(meg_signals)

    global_leverage = stream_dataset_inference(
        meg_data=meg_signals,
        raw_matrices=raw_matrices,
        output_parquet_path=str(out_parquet)
    )

    # 1. Сохранение чекпоинта матричной геометрии
    save_qcml_checkpoint(
        checkpoint_path=str(checkpoint_file),
        raw_matrices=raw_matrices,
        std_mag=std_mag,
        std_grad=std_grad
    )
    print(f"[+] Чекпоинт геометрии сохранен: {checkpoint_file}")

    # 2. Сохранение отдельного файла топографии сенсоров
    torch.save(
        {"channel_leverage": global_leverage, "n_hilbert": args.n_hilbert},
        str(out_leverage_file)
    )
    print(f"[+] Топографическая карта влиятельности сенсоров сохранена: {out_leverage_file}")
