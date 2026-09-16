import os
import gc
import math
from pathlib import Path
from typing import Tuple, Generator, Dict, Any, Optional

import numpy as np
import mne
import torch
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И ФУНКЦИЯ ИХ ПЕРЕОПРЕДЕЛЕНИЯ
# =============================================================================

N_FEATURES: int = 306
N_HILBERT: int = 24
BATCH_SIZE: int = 512
N_EPOCHS: int = 20
N_SAMPLES_TRAIN: int = -1
LEARNING_RATE: float = 1e-4
W_VARIANCE: float = 0.0
W_COMMUTATOR: float = 0.0
GRAD_CLIP_NORM: float = 1.0
GAP_EPSILON: float = 1e-5
CHUNK_FLUSH_SIZE: int = 50000
TARGET_VRAM_GB: float = 10.0
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_hyperparameters(
    n_features: int = 306,
    n_hilbert: int = 24,
    batch_size: int = 512,
    n_epochs: int = 20,
    n_samples_train: int = -1,     # -1 означает 100% данных (все 2 550 000 точек)
    learning_rate: float = 1e-4,
    w_variance: float = 0.0,
    w_commutator: float = 0.0,
    grad_clip_norm: float = 1.0,
    gap_epsilon: float = 1e-5,
    chunk_flush_size: int = 50000,
    target_vram_gb: float = 10.0,
    device_str: Optional[str] = None
) -> None:
    """
    Единая точка конфигурации пайплайна QCML.
    Переопределяет все параметры выполнения в глобальной области видимости.
    """
    global N_FEATURES, N_HILBERT, BATCH_SIZE, N_EPOCHS, N_SAMPLES_TRAIN
    global LEARNING_RATE, W_VARIANCE, W_COMMUTATOR, GRAD_CLIP_NORM
    global GAP_EPSILON, CHUNK_FLUSH_SIZE, TARGET_VRAM_GB, DEVICE

    N_FEATURES = int(n_features)
    N_HILBERT = int(n_hilbert)
    BATCH_SIZE = int(batch_size)
    N_EPOCHS = int(n_epochs)
    N_SAMPLES_TRAIN = int(n_samples_train)
    LEARNING_RATE = float(learning_rate)
    W_VARIANCE = float(w_variance)
    W_COMMUTATOR = float(w_commutator)
    GRAD_CLIP_NORM = float(grad_clip_norm)
    GAP_EPSILON = float(gap_epsilon)
    CHUNK_FLUSH_SIZE = int(chunk_flush_size)
    TARGET_VRAM_GB = float(target_vram_gb)

    if device_str is not None:
        DEVICE = torch.device(device_str)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_mode_str = "ПОЛНЫЙ ДАТАСЕТ (100%)" if N_SAMPLES_TRAIN <= 0 else f"{N_SAMPLES_TRAIN} точек"
    print(
        f"[Config] Параметры обновлены:\n"
        f"  * Сенсоры D = {N_FEATURES}, Hilbert N = {N_HILBERT} (Rank Bound <= {2*(N_HILBERT-1)})\n"
        f"  * Батч = {BATCH_SIZE}, Эпох = {N_EPOCHS}, Выборка = {train_mode_str}\n"
        f"  * LR = {LEARNING_RATE}, Device = {DEVICE}\n"
        f"  * Сброс в Parquet чанками по {CHUNK_FLUSH_SIZE} точек"
    )


# =============================================================================
# БЛОК 1. ЗАГРУЗКА, ИЗВЛЕЧЕНИЕ СЕНСОРОВ И БАТЧИНГ (DATA INGESTION)
# =============================================================================

def load_preprocessed_raw_fif(fif_path: str, preload: bool = True) -> mne.io.Raw:
    """Чтение файла .fif без модификации сигналов."""
    return mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)


def extract_meg_channels_tensor(raw: mne.io.Raw) -> np.ndarray:
    """
    Извлечение 306 каналов МЭГ в строгом соответствии с требованиями QCML:
    1. Центрирование сигналов (mean = 0).
    2. Приведение физических размерностей (Тл и Тл/м) к соразмерному безразмерному масштабу O(1)
       двумя глобальными коэффициентами RMS (по одному на тип сенсора).
    3. Полное сохранение пространственной топологии внутри магнитометров и градиентометров.
    """
    mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False, ecg=False)
    grad_picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, ecg=False)
    all_picks = np.concatenate([mag_picks, grad_picks])

    data = raw.get_data(picks=all_picks).T.astype(np.float32)  # Форма: (T, 306)
    n_mags = len(mag_picks)

    # 1. Центрирование
    data -= np.mean(data, axis=0, keepdims=True)

    # 2. Согласование масштабов по типам сенсоров
    std_mag = np.sqrt(np.mean(data[:, :n_mags] ** 2))
    std_grad = np.sqrt(np.mean(data[:, n_mags:] ** 2))

    data[:, :n_mags] /= (std_mag + 1e-15)
    data[:, n_mags:] /= (std_grad + 1e-15)

    print(
        f"[QCML Ingestion] Данные согласованы с требованиями алгоритма:\n"
        f"  * 102 Магнитометра: масштабированы на 1/{std_mag:.3e}\n"
        f"  * 204 Градиентометра: масштабированы на 1/{std_grad:.3e}\n"
        f"  * Итоговый диапазон x: [{data.min():.2f}, {data.max():.2f}] (дисперсия = 1.0)"
    )
    return data


def allocate_pinned_transfer_buffer() -> torch.Tensor:
    """Выделение pinned-буфера в RAM хоста под батч BATCH_SIZE x N_FEATURES."""
    return torch.empty((BATCH_SIZE, N_FEATURES), dtype=torch.float32, pin_memory=True)


def generate_random_batch_indices(n_samples: int) -> Generator[np.ndarray, None, None]:
    """Генератор случайных индексов для стохастического обучения."""
    indices = np.random.permutation(n_samples)
    for start_idx in range(0, n_samples, BATCH_SIZE):
        yield indices[start_idx : start_idx + BATCH_SIZE]


def generate_sequential_batch_slices(n_samples: int) -> Generator[Tuple[int, int], None, None]:
    """Генератор последовательных временных срезов [start, end) для инференса."""
    for start_idx in range(0, n_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_samples)
        yield start_idx, end_idx


# =============================================================================
# БЛОК 2. ПАРАМЕТРИЗАЦИЯ ОПЕРАТОРОВ И АЛГЕБРАИЧЕСКИЕ ИНВАРИАНТЫ
# =============================================================================

def init_raw_matrix_parameters() -> torch.Tensor:
    """Инициализация комплексных обучаемых параметров (D, N, N) на целевом GPU."""
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
# БЛОК 3. ГАМИЛЬТОНИАН ОШИБКИ И ПОИСК ОСНОВНОГО СОСТОЯНИЯ
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
    """Вычисление энергии вакуума E_0 (B,) и волновой функции psi_0 (B, N)."""
    evals, evecs = torch.linalg.eigh(H_batch)
    return evals[:, 0], evecs[:, :, 0]


def solve_full_hamiltonian_eigensystem(
    H_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Полное разложение Гамильтониана: E_0, psi_0, E_excited (B, N-1), psi_excited (B, N, N-1)."""
    evals, evecs = torch.linalg.eigh(H_batch)
    return evals[:, 0], evecs[:, :, 0], evals[:, 1:], evecs[:, :, 1:]


def check_ground_state_gap(
    E_0: torch.Tensor, E_excited: torch.Tensor, min_gap: float = 1e-6
) -> torch.Tensor:
    """Маска точек с невырожденным вакуумом (E_1 - E_0) > min_gap."""
    return (E_excited[:, 0] - E_0) > min_gap


# =============================================================================
# БЛОК 4. КВАНТОВЫЕ СРЕДНИЕ, ДИСПЕРСИЯ И ЦЕЛЕВОЙ ФУНКЦИОНАЛ
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
    """Итоговый функционал потерь QCML с учетом W_VARIANCE и W_COMMUTATOR."""
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
    """Обратное распространение ошибки и градиентный шаг с ограничением нормы."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([raw_matrices], max_norm=GRAD_CLIP_NORM)
    optimizer.step()
    return loss.item()


# =============================================================================
# БЛОК 5. ПРОЕКЦИЯ НА КВАНТОВОЕ ОБЛАКО ТОЧЕК X_A
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
    """Евклидова невязка реконструкции ||y(x) - x|| (амплитуда отсеченного шума)."""
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
    """Масштабирование V_{n, mu} на sqrt(2 / Delta E_n)."""
    scales = torch.sqrt(2.0 / energy_gaps).unsqueeze(-1)
    return transition_vectors * scales


def assemble_quantum_metric_tensor(scaled_transitions: torch.Tensor) -> torch.Tensor:
    """Сборка матрицы метрики g_{\mu\nu} = sum Re( V_n V_n^H ) формы (B, D, D)."""
    V_r = scaled_transitions.real
    V_i = scaled_transitions.imag
    return torch.bmm(V_r.transpose(1, 2), V_r) + torch.bmm(V_i.transpose(1, 2), V_i)


def enforce_metric_symmetry_inplace(metric_batch: torch.Tensor) -> torch.Tensor:
    """Устранение численной асимметрии: 0.5 * (g + g^T)."""
    return 0.5 * (metric_batch + metric_batch.transpose(-1, -2))


def compute_metric_trace_vector(metric_batch: torch.Tensor) -> torch.Tensor:
    """Быстрый расчет следа Tr(g(y)) формы (B,)."""
    return torch.diagonal(metric_batch, dim1=-2, dim2=-1).sum(dim=-1)


# =============================================================================
# БЛОК 7. ДИАГОНАЛИЗАЦИЯ И ЦЕЛОЧИСЛЕННАЯ РАЗМЕРНОСТЬ d(y)
# =============================================================================

def diagonalize_metric_spectra(metric_batch: torch.Tensor) -> torch.Tensor:
    """Спектральное разложение симметричной метрики. Выход: eigenvalues (B, D)."""
    return torch.linalg.eigvalsh(metric_batch)


def verify_spectral_rank_bound(metric_evals: torch.Tensor) -> torch.Tensor:
    """Проверка теорематического ограничения ранга rank(g) <= 2(N-1)."""
    active_modes = torch.sum(metric_evals > 1e-5, dim=-1)
    return active_modes <= 2 * (N_HILBERT - 1)


def extract_id_exact_geometry(metric_evals: torch.Tensor) -> torch.Tensor:
    """
    Математически строгая оценка внутренней размерности без эвристических констант.
    Опирается на тождество I = g(y) + Hess(E_0) и условие доминирования
    касательного проектора над нормальным:
    lambda_k > 1 - lambda_k  <=>  lambda_k > 0.5.
    """
    # Спектр по убыванию: lambda_1 >= lambda_2 >= ... >= lambda_D
    evals_desc = torch.flip(metric_evals, dims=[-1])

    # Теоретический ранг оператора Ли: не более 2*(N - 1) ненулевых мод
    max_rank = 2 * (N_HILBERT - 1)
    evals_active = evals_desc[:, :max_rank]

    # Строгий подсчет мод, принадлежащих касательному расслоению (lambda_k > 0.5)
    tangent_mask = evals_active > 0.5
    d_local = torch.sum(tangent_mask, dim=-1)

    # Защита от вырожденных точек: размерность не может быть меньше 1
    d_local = torch.clamp(d_local, min=1)

    return d_local.to(torch.int32)


def extract_tangent_basis_vectors(
    metric_batch: torch.Tensor, d_local: int
) -> torch.Tensor:
    """Извлечение d касательных собственных векторов метрики формы (B, D, d)."""
    _, evecs = torch.linalg.eigh(metric_batch)
    return evecs[:, :, -d_local:]


# =============================================================================
# БЛОК 8. ИНФЕРЕНС-ШАГ И СТОХАСТИЧЕСКОЕ ОБУЧЕНИЕ
# =============================================================================

def calculate_safe_batch_size() -> int:
    """Автоматический расчет предельного безопасного батча под лимит TARGET_VRAM_GB."""
    bytes_per_point = 4 * (N_FEATURES**2 + N_FEATURES * N_HILBERT + N_HILBERT**2 + 4 * N_FEATURES)
    safe_points = int((TARGET_VRAM_GB * 1024**3) / bytes_per_point)
    return min(max(safe_points, 128), 1024)


@torch.no_grad()
def process_single_inference_chunk(
    x_chunk: torch.Tensor,
    hermitian_matrices: torch.Tensor,
    K_operator: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Полный расчет размерностей и следов для одного батча отсчетов на GPU."""
    y_chunk = project_batch_to_point_cloud(x_chunk, hermitian_matrices, K_operator)
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
    evals = diagonalize_metric_spectra(g)
    
    # ВЫЗОВ СТРОГОЙ МАТЕМАТИЧЕСКОЙ ФУНКЦИИ (БЕЗ МАГИЧЕСКИХ ЧИСЕЛ)
    d_local = extract_id_exact_geometry(evals)

    return d_local, traces


def run_training_epoch(
    meg_data: np.ndarray,
    raw_matrices: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """Выполнение одной эпохи обучения (по всему массиву при N_SAMPLES_TRAIN <= 0)."""
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
    """Главный цикл обучения матричной геометрии QCML."""
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
# БЛОК 9. ПОТОКОВАЯ СЕРИАЛИЗАЦИЯ И ДИСКОВЫЙ I/O (PYARROW PARQUET)
# =============================================================================

def init_parquet_writer(output_path: Path) -> pq.ParquetWriter:
    """Инициализация потокового райтера Parquet с компрессией Snappy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("sample_index", pa.int64()),
        ("intrinsic_dimension", pa.int32()),
        ("metric_trace", pa.float32())
    ])
    return pq.ParquetWriter(output_path, schema, compression="snappy")


def write_chunk_to_parquet(
    writer: pq.ParquetWriter,
    indices: np.ndarray,
    d_values: np.ndarray,
    traces: np.ndarray
) -> None:
    """Потоковая дозапись порции строк без чтения файла в RAM."""
    table = pa.Table.from_arrays(
        [
            pa.array(indices, type=pa.int64()),
            pa.array(d_values, type=pa.int32()),
            pa.array(traces, type=pa.float32())
        ],
        names=["sample_index", "intrinsic_dimension", "metric_trace"]
    )
    writer.write_table(table)


def close_parquet_writer(writer: pq.ParquetWriter) -> None:
    """Корректное завершение потоковой записи Parquet."""
    writer.close()


def save_qcml_checkpoint(checkpoint_path: str, raw_matrices: torch.Tensor) -> None:
    """Сохранение состояния операторов и текущих глобальных гиперпараметров."""
    payload = {
        "raw_matrices": raw_matrices.cpu(),
        "config": {
            "N_FEATURES": N_FEATURES,
            "N_HILBERT": N_HILBERT,
            "BATCH_SIZE": BATCH_SIZE,
            "W_VARIANCE": W_VARIANCE
        }
    }
    torch.save(payload, checkpoint_path)


def load_qcml_checkpoint(checkpoint_path: str) -> torch.Tensor:
    """Загрузка матрицы геометрии из файла чекпоинта на целевой DEVICE."""
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    return payload["raw_matrices"].to(DEVICE)


# =============================================================================
# БЛОК 10. ГЛАВНЫЙ ПОТОКОВЫЙ КОНВЕЙЕР ПО ВСЕМ ТОЧКАМ (STREAMING PIPELINE)
# =============================================================================

def stream_dataset_inference(
    meg_data: np.ndarray,
    raw_matrices: torch.Tensor,
    output_parquet_path: str
) -> None:
    """Потоковый расчет целочисленной размерности d(t) по всем точкам."""
    T, _ = meg_data.shape
    out_path = Path(output_parquet_path)
    writer = init_parquet_writer(out_path)

    A = build_hermitian_matrices(raw_matrices)
    K = precompute_sum_squared_operators(A)

    buf_indices = []
    buf_d = []
    buf_trace = []
    total_flushed = 0

    print(f"[*] Старт потокового инференса: T={T} отсчетов. Сброс чанками по {CHUNK_FLUSH_SIZE}...")

    with torch.no_grad():
        for start_idx, end_idx in generate_sequential_batch_slices(T):
            x_batch = torch.from_numpy(meg_data[start_idx:end_idx]).to(DEVICE)
            d_local, traces = process_single_inference_chunk(x_batch, A, K)

            buf_indices.extend(range(start_idx, end_idx))
            buf_d.extend(d_local.cpu().numpy().tolist())
            buf_trace.extend(traces.cpu().numpy().tolist())

            if len(buf_d) >= CHUNK_FLUSH_SIZE or end_idx == T:
                write_chunk_to_parquet(
                    writer,
                    np.array(buf_indices, dtype=np.int64),
                    np.array(buf_d, dtype=np.int32),
                    np.array(buf_trace, dtype=np.float32)
                )
                total_flushed += len(buf_d)
                buf_indices.clear()
                buf_d.clear()
                buf_trace.clear()

                torch.cuda.empty_cache()
                progress = (end_idx / T) * 100.0
                print(f"    Прогресс: {end_idx}/{T} ({progress:.2f}%) | Записано: {total_flushed}")

    close_parquet_writer(writer)
    print(f"[+] Расчет завершен. Файл сохранен: {out_path.resolve()}")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    configure_hyperparameters(
        n_features=306,
        n_hilbert=24,
        batch_size=512,
        n_epochs=20,
        n_samples_train=-1,
        learning_rate=1e-4,
        w_variance=0.0,
        w_commutator=0.0,
        gap_epsilon=1e-5,
        chunk_flush_size=50000,
        target_vram_gb=10.0,
        device_str="cuda" if torch.cuda.is_available() else "cpu"
    )

    fif_path = "01_raw_data/type1_bigrams/type1_sub10_chistova_alena_raw_preprocessed.fif"
    checkpoint_file = "04_processed_db/type1_features/type1_sub10_qcml_geometry.pt"
    out_parquet = "04_processed_db/type1_features/type1_sub10_intrinsic_dimension.parquet"

    print(f"[*] Чтение предобработанных МЭГ сигналов: {fif_path}")
    raw = load_preprocessed_raw_fif(fif_path, preload=True)
    meg_signals = extract_meg_channels_tensor(raw)
    del raw
    gc.collect()

    # ЕСЛИ ЧЕКПОИНТ СУЩЕСТВУЕТ — НЕ ТРАТИМ ВРЕМЯ НА ОБУЧЕНИЕ, СРАЗУ ГРУЗИМ:
    if os.path.exists(checkpoint_file):
        print(f"[+] Найден готовый чекпоинт {checkpoint_file}. Загружаем геометрию...")
        raw_matrices = load_qcml_checkpoint(checkpoint_file)
    else:
        print("[*] Чекпоинт не найден. Запуск обучения матричной конфигурации QCML...")
        raw_matrices = fit_qcml_geometry(meg_signals)
        save_qcml_checkpoint(checkpoint_file, raw_matrices)
        print(f"[+] Чекпоинт сохранен: {checkpoint_file}")

    # ПОТОКОВЫЙ ИНФЕРЕНС (ЗАЙМЁТ ~1.5 МИНУТЫ)
    stream_dataset_inference(
        meg_data=meg_signals,
        raw_matrices=raw_matrices,
        output_parquet_path=out_parquet
    )
