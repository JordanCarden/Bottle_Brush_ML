#!/bin/bash
#SBATCH -n 16
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -p gpu4
#SBATCH -A loni_poly_surf
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N
#SBATCH --gres=gpu:1

module purge
module load cuda

/work/jcarde7/NAMD_2.14_Linux-x86_64-multicore-CUDA/namd2 +p16 20_waterbox.conf > output.log
