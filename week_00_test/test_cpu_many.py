import os, numpy as np, scipy

print("=" * 50)
print("=== ТЕСТ 2: CPU MANY-SINGLE СТЕК ===")
print("=" * 50)
print("Process PID:", os.getpid())
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
print("NumPy Version:", np.__version__)
print("SciPy Version:", scipy.__version__)

a = np.random.randn(1000, 1000)
b = np.dot(a, a)
print("Результат умножения матриц:", b.shape)
print("✅ [УСПЕХ] Одноядерный CPU стек работоспособен!")
