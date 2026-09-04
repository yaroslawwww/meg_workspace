#!/bin/bash
#SBATCH --job-name=meg_cpu_multi
#SBATCH --partition=rocky
#SBATCH --output=05_logs/cpu_multi-%j.out
#SBATCH --error=05_logs/cpu_multi-%j.err

module load apptainer/1.5.0

SCRIPT_TO_RUN="${1:-compute_engine.py}"
CPUS=${SLURM_CPUS_PER_TASK:-1}

echo "=== [CPU SINGLE-MULTI JOB] Узел: $(hostname) | Выделено ядер: ${CPUS} | Скрипт: ${SCRIPT_TO_RUN} ==="

apptainer exec \
  --env OMP_NUM_THREADS=${CPUS} \
  --env MKL_NUM_THREADS=${CPUS} \
  --bind .:/workspace \
  --bind /scratch:/scratch \
  meg_pipeline_2026.sif \
  python "/workspace/${SCRIPT_TO_RUN}"
