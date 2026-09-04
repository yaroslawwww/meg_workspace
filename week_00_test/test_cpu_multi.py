import os, time, torch, numpy as np

print("=" * 50)
print("=== ТЕСТ 3: CPU SINGLE-MULTI СТЕК ===")
print("=" * 50)
threads = int(os.environ.get("OMP_NUM_THREADS", 1))
print("OMP_NUM_THREADS из окружения:", threads)

torch.set_num_threads(threads)
print("PyTorch использует потоков CPU:", torch.get_num_threads())

t0 = time.time()
x = torch.randn(3000, 3000)
y = torch.matmul(x, x)
dt = time.time() - t0

print(f"Матрица 3000x3000 перемножена на {threads} ядрах за {dt:.2f} сек")
print("✅ [УСПЕХ] Многопоточный CPU стек работоспособен!")
