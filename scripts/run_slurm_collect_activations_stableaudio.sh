#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --account=plgdynamic2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_out/collect_activations/stableaudio_dpm_public2_10s_freqs.out
#SBATCH --job-name=collect_activations_stableaudio_dpm_public2_10s_freqs

module load ML-bundle/24.06a
cd /net/scratch/hscra/plgrid/plglukaszst/projects/music-diffsae
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 src/scripts/collect_activations_stableaudio.py