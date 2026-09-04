#!/bin/bash
#SBATCH --job-name=meg_gpu
#SBATCH --partition=rocky
#SBATCH --output=05_logs/gpu-%j.out
#SBATCH --error=05_logs/gpu-%j.err

module load apptainer/1.5.0

SCRIPT_TO_RUN="${1:-compute_engine.py}"
CPUS=${SLURM_CPUS_PER_TASK:-1}

echo "=== [GPU JOB] Узел: $(hostname) | Выделено CPU: ${CPUS} | Скрипт: ${SCRIPT_TO_RUN} ==="

apptainer exec --nv \
  --env OMP_NUM_THREADS=${CPUS} \
  --env MKL_NUM_THREADS=${CPUS} \
  --bind .:/workspace \
  --bind /scratch:/scratch \
  meg_pipeline_2026.sif \
  python "/workspace/${SCRIPT_TO_RUN}"
