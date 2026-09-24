import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# =============================================================================
# 1. ПАРАМЕТРЫ
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FEATURES: int = 3
N_HILBERT: int = 5           # для D=3 достаточно; 24 -- перебор
BATCH_SIZE: int = 256
N_EPOCHS: int = 80
LEARNING_RATE: float = 3e-3
W_VARIANCE: float = 0.01      # ВКЛЮЧЕНО
W_COMMUTATOR: float = 0.0
GRAD_CLIP_NORM: float = 1.0
N_POINTS_EACH: int = 50000

# =============================================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ
# =============================================================================
def generate_synthetic_data(n_points_each=N_POINTS_EACH):
    np.random.seed(42)

    # Сфера d=2
    phi = np.random.uniform(0, 2 * np.pi, n_points_each)
    costheta = np.random.uniform(-1, 1, n_points_each)
    theta = np.arccos(costheta)
    sphere = np.stack([np.sin(theta) * np.cos(phi),
                       np.sin(theta) * np.sin(phi),
                       np.cos(theta)], axis=1)

    # Прямая d=1 (ось Z)
    zl = np.linspace(-2.0, 2.0, n_points_each)
    line = np.stack([np.zeros(n_points_each),
                     np.zeros(n_points_each),
                     zl], axis=1)

    raw_data = np.vstack([sphere, line]).astype(np.float32)

    data = raw_data - np.mean(raw_data, axis=0, keepdims=True)
    std_scale = float(np.sqrt(np.mean(data ** 2)))
    data /= (std_scale + 1e-15)

    return raw_data, data, std_scale


# =============================================================================
# 3. ОПЕРАТОРЫ
# =============================================================================
def init_raw_matrix_parameters():
    scale = 1.0 / math.sqrt(N_FEATURES * N_HILBERT)
    real_part = torch.randn(N_FEATURES, N_HILBERT, N_HILBERT, device=DEVICE) * scale
    imag_part = torch.randn(N_FEATURES, N_HILBERT, N_HILBERT, device=DEVICE) * scale
    raw = torch.complex(real_part, imag_part)
    raw.requires_grad_(True)
    return raw


def build_hermitian_matrices(raw):
    return 0.5 * (raw + raw.conj().transpose(-1, -2))


def precompute_sum_squared_operators(A):
    return 0.5 * torch.sum(A @ A, dim=0)


def compute_linear_operator_contraction(x_batch, A):
    return torch.einsum('bd, dnm -> bnm', x_batch.to(torch.complex64), A)


def compute_point_norms_squared(x_batch):
    return 0.5 * torch.sum(x_batch ** 2, dim=-1, keepdim=True).unsqueeze(-1)


def assemble_error_hamiltonian(K, lin, norms_sq):
    eye = torch.eye(N_HILBERT, device=DEVICE, dtype=torch.complex64).unsqueeze(0)
    H = K.unsqueeze(0) - lin + norms_sq.to(torch.complex64) * eye
    return 0.5 * (H + H.conj().transpose(-1, -2))


def solve_ground_state(H_batch):
    """Дифференцируемый поиск основного состояния (без no_grad!)."""
    evals, evecs = torch.linalg.eigh(H_batch)
    return evals[:, 0], evecs[:, :, 0]


def solve_full_eigensystem(H_batch):
    evals, evecs = torch.linalg.eigh(H_batch)
    return evals[:, 0], evecs[:, :, 0], evals[:, 1:], evecs[:, :, 1:]


# =============================================================================
# 4. LOSS
# =============================================================================
def compute_coordinate_expectations(psi_0, A):
    return torch.einsum('bn, dnm, bm -> bd', psi_0.conj(), A, psi_0).real


def compute_bias_loss(x_batch, y_batch):
    return torch.mean(torch.sum((y_batch - x_batch) ** 2, dim=-1))


def compute_variance(psi_0, A):
    """σ²(ψ) = Σ_k [<A_k²> − <A_k>²]  — усредняется по батчу."""
    A_sq = A @ A                                   # [D, N, N]
    exp_A2 = torch.einsum('bn, dnm, bm -> bd', psi_0.conj(), A_sq, psi_0).real
    exp_A  = torch.einsum('bn, dnm, bm -> bd', psi_0.conj(), A,    psi_0).real
    return torch.sum(exp_A2 - exp_A ** 2, dim=-1)  # [B]


def execute_gradient_step(optimizer, loss, raw_matrices):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([raw_matrices], max_norm=GRAD_CLIP_NORM)
    optimizer.step()
    return loss.item()


# =============================================================================
# 5. КВАНТОВАЯ МЕТРИКА g(x)
# =============================================================================
def compute_energy_gaps(E_0, E_exc):
    return torch.clamp(E_exc - E_0.unsqueeze(-1), min=1e-6)


def compute_transition_matrix_elements(psi_0, psi_exc, A):
    # psi_0:   [B, N]
    # psi_exc: [B, N, K]   (eigh возвращает evecs в [B, N, N])
    # A:       [D, N, N]
    # out:     [B, K, D]   --  <psi_0| A_d |psi_k>
    return torch.einsum('bn, dnm, bmk -> bkd', psi_0.conj(), A, psi_exc)


def scale_transition_vectors_by_gap(V, gaps):
    scales = torch.sqrt(2.0 / gaps).unsqueeze(-1)   # [B, K, 1]
    return V * scales


def assemble_quantum_metric_tensor(V_scaled):
    V_r = V_scaled.real
    V_i = V_scaled.imag
    g = torch.bmm(V_r.transpose(1, 2), V_r) + torch.bmm(V_i.transpose(1, 2), V_i)
    return 0.5 * (g + g.transpose(-1, -2))


# =============================================================================
# 6. ИНФЕРЕНС
# =============================================================================
@torch.no_grad()
def evaluate_manifold_and_metric(data_tensor, raw_matrices):
    A = build_hermitian_matrices(raw_matrices)
    K = precompute_sum_squared_operators(A)

    # 1) H(x) и его полный спектр
    lin_x  = compute_linear_operator_contraction(data_tensor, A)
    norm_x = compute_point_norms_squared(data_tensor)
    H_x    = assemble_error_hamiltonian(K, lin_x, norm_x)

    E_0, psi_0, E_exc, psi_exc = solve_full_eigensystem(H_x)

    # 2) y(x) = <ψ₀|A|ψ₀>
    y_coords = compute_coordinate_expectations(psi_0, A)

    # 3) Метрика g(x) СТРОГО в спектре H(x)
    gaps = compute_energy_gaps(E_0, E_exc)
    V = compute_transition_matrix_elements(psi_0, psi_exc, A)
    V_scaled = scale_transition_vectors_by_gap(V, gaps)
    g = assemble_quantum_metric_tensor(V_scaled)

    # 4) Собственные значения метрики (по возрастанию)
    evals, _ = torch.linalg.eigh(g)
    g_evals = evals.cpu().numpy()

    # 5) Детекция спектрального зазора
    ratios = g_evals[:, 1:] / (g_evals[:, :-1] + 1e-6)
    gap_step = np.argmax(ratios, axis=1)
    detected_dim = 2 - gap_step   # 0 -> d=2, 1 -> d=1

    return y_coords.cpu().numpy(), g_evals, detected_dim


# =============================================================================
# 7. ОСНОВНОЙ ЦИКЛ
# =============================================================================
if __name__ == "__main__":
    raw_data, norm_data, std_scale = generate_synthetic_data(N_POINTS_EACH)
    T = norm_data.shape[0]

    raw_matrices = init_raw_matrix_parameters()
    optimizer = torch.optim.AdamW([raw_matrices], lr=LEARNING_RATE)

    print(f"[*] Старт обучения QCML: D={N_FEATURES}, N={N_HILBERT}, "
          f"W_VAR={W_VARIANCE}, W_COMM={W_COMMUTATOR}...")

    data_tensor = torch.from_numpy(norm_data).to(DEVICE)

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(T, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for start_idx in range(0, T, BATCH_SIZE):
            batch_idx = perm[start_idx : start_idx + BATCH_SIZE]
            x_b = data_tensor[batch_idx]

            A = build_hermitian_matrices(raw_matrices)
            K = precompute_sum_squared_operators(A)

            # ВАЖНО: без torch.no_grad() — градиент через eigh
            lin   = compute_linear_operator_contraction(x_b, A)
            norms = compute_point_norms_squared(x_b)
            H     = assemble_error_hamiltonian(K, lin, norms)
            _, psi_0 = solve_ground_state(H)

            y = compute_coordinate_expectations(psi_0, A)

            bias = compute_bias_loss(x_b, y)
            if W_VARIANCE > 0.0:
                var  = compute_variance(psi_0, A).mean()
                loss = bias + W_VARIANCE * var
            else:
                loss = bias

            loss_val = execute_gradient_step(optimizer, loss, raw_matrices)
            epoch_loss += loss_val
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Эпоха [{epoch+1:02d}/{N_EPOCHS:02d}] | Loss: "
                  f"{epoch_loss / n_batches:.6f}")

    # --- Инференс -----------------------------------------------------------
    print("\n[*] Инференс: проецирование и расчет g(x) в точке данных...")
    y_learned_norm, g_evals, detected_dim = evaluate_manifold_and_metric(
        data_tensor, raw_matrices)

    mean_offset = raw_data.mean(axis=0, keepdims=True)
    X_learned = (y_learned_norm * std_scale) + mean_offset

    # ИСПРАВЛЕНО: правильное разбиение сфера / прямая
    sphere_dims = detected_dim[:N_POINTS_EACH]
    line_dims   = detected_dim[N_POINTS_EACH:]

    print("\n" + "=" * 50)
    print("СПЕКТР g(x) НА СФЕРЕ (первые 3 точки):")
    print(np.round(g_evals[:3], 3))
    print("\nСПЕКТР g(x) НА ПРЯМОЙ (первые 3 точки прямой):")
    print(np.round(g_evals[N_POINTS_EACH:N_POINTS_EACH + 3], 3))

    print("\n" + "=" * 50)
    print(f"Точки сферы:  средняя размерность = {np.mean(sphere_dims):.2f} "
          f"(ожидалось 2)")
    print(f"Точки прямой: средняя размерность = {np.mean(line_dims):.2f} "
          f"(ожидалось 1)")
    print("=" * 50)

    # --- Графика ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(raw_data[:500, 0], raw_data[:500, 1], raw_data[:500, 2],
                c='gray', alpha=0.3, s=12, label='Сфера (d=2)')
    ax1.scatter(raw_data[N_POINTS_EACH:N_POINTS_EACH + 500, 0],
                raw_data[N_POINTS_EACH:N_POINTS_EACH + 500, 1],
                raw_data[N_POINTS_EACH:N_POINTS_EACH + 500, 2],
                c='black', alpha=0.7, s=12, label='Прямая (d=1)')
    ax1.set_title("Исходные данные X")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    colors = ['blue' if d == 1 else 'green' if d == 2 else 'red'
              for d in detected_dim]
    ax2.scatter(X_learned[:, 0], X_learned[:, 1], X_learned[:, 2],
                c=colors, s=18, alpha=0.8)
    ax2.set_title("Обученное квантовое многообразие $X_A = y(x)$\n"
                  "(Зеленый = d=2, Синий = d=1)")

    plt.tight_layout()
    plt.show()
