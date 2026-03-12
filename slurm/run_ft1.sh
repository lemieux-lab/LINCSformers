#!/bin/bash

#SBATCH --job-name=ft_lvl1_11_trt_v1
#SBATCH --output=out/2026-03-12/%x.log
#SBATCH --error=out/2026-03-12/%x.log

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=1-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/finetuning/lvl1_ft.jl
