#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --account=plgdiffusion-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_out/collect_activations/audioldm2.out
#SBATCH --job-name=collect_activations_audioldm2

module load ML-bundle/24.06a
cd /net/scratch/hscra/plgrid/plglukaszst/projects/music-diffsae
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 src/scripts/collect_activations.py