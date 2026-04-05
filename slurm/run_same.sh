#!/bin/bash

#SBATCH --job-name=SSS_lvl1_1_rtf_trt
#SBATCH --output=out/2026-04-02/%x.log
#SBATCH --error=out/2026-04-02/%x.log

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/finetune/ft_same.jl
