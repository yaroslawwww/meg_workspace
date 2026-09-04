#!/bin/bash
#SBATCH --job-name=meg_cpu_many
#SBATCH --partition=rocky
#SBATCH --output=05_logs/cpu_many-%j.out
#SBATCH --error=05_logs/cpu_many-%j.err

module load apptainer/1.5.0

SCRIPT_TO_RUN="${1:-compute_engine.py}"

echo "=== [CPU MANY-SINGLE JOB] Узел: $(hostname) | Задач: ${SLURM_NTASKS} | Скрипт: ${SCRIPT_TO_RUN} ==="

apptainer exec \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --bind .:/workspace \
  --bind /scratch:/scratch \
  meg_pipeline_2026.sif \
  python "/workspace/${SCRIPT_TO_RUN}"
