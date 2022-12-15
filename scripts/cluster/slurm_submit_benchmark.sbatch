#!/bin/bash

#SBATCH --job-name=LowDataGeneralization_Benchmark

#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:1,VRAM:10G
#SBATCH --mem=24GB

#SBATCH --signal=2

#SBATCH --output=logs/experiment_status/%A-%a.log
#SBATCH --open-mode=append

#SBATCH --time=2-00:00:00

#SBATCH --array=0-4

echo Starting experiment $SLURM_ARRAY_TASK_ID with arguments: $SCRIPT_ARGUMENTS
python -u -m lowdataregime.active_learning.active_learning $SCRIPT_ARGUMENTS --run_id $SLURM_ARRAY_TASK_ID