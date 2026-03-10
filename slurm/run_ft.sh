#!/bin/bash

#SBATCH --job-name=ft_520_trt
#SBATCH --output=out/2026-03-09/%x.log
#SBATCH --error=out/2026-03-09/%x.log

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=4-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/finetune.jl
