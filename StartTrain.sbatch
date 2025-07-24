#!/bin/bash
#SBATCH --job-name=75PopEvolution
#SBATCH --output=TAI_%j.out
#SBATCH --error=TAI_%j.err
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -A trn039

cd $SLURM_SUBMIT_DIR
module load miniforge3/23.11.0

python3 -u AITrainNoPygame.py