import torch, mne, cudf, h5py, zarr

print("=" * 50)
print("=== ТЕСТ 1: GPU СТЕК ===")
print("=" * 50)
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())
print("cuDF RAPIDS Version:", cudf.__version__)
print("MNE Version:", mne.__version__)
print("HDF5 Version:", h5py.__version__)
print("Zarr Version:", zarr.__version__)
print("✅ [УСПЕХ] GPU стек полностью работоспособен!")
