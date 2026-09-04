FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    build-essential \
    libhdf5-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# 2. Символическая ссылка для вызова python и pip по умолчанию
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 3. Обновление pip до свежей версии
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 4. Фиксированные релизы пакетов (NumPy согласован с SciPy 1.15)
RUN pip install --no-cache-dir \
    "numpy>=2.0.0,<2.5.0" \
    "scipy==1.15.0" \
    "h5py==3.12.0" \
    "zarr==3.0.0" \
    "numba>=0.60.0" \
    "joblib==1.5.0" \
    "mne==1.12.1"

# 5. PyTorch 2026 с поддержкой CUDA 12.6
RUN pip install --no-cache-dir "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu126
# 6. Расширенный стек аналитика (Визуализация, Big Data, классический ML и GPU)
# 6. Расширенный стек аналитики и ML
RUN pip install --no-cache-dir \
    "pandas>=3.0.5" \
    "polars>=1.44.1" \
    "pyarrow>=18.0.0" \
    "openpyxl>=3.1.5" \
    "statsmodels>=0.14.4" \
    "matplotlib>=3.11.1" \
    "seaborn>=0.13.2" \
    "plotly>=6.0.0" \
    "bokeh>=3.6.0" \
    "scikit-learn>=1.6.0" \
    "xgboost>=2.1.3" \
    "lightgbm>=4.5.0" \
    "catboost>=1.2.7" \
    "tqdm>=4.67.0" \
    "scikit-image>=0.25.0" \
    "mef3io" \
    "pysz"


# 7. Ускорение вычислений NVIDIA RAPIDS (ТРЕБУЕТСЯ --extra-index-url)
RUN pip install --no-cache-dir \
    "cudf-cu12" \
    "cuml-cu12" \
    "cugraph-cu12" \
    --extra-index-url https://pypi.nvidia.com


WORKDIR /workspace

