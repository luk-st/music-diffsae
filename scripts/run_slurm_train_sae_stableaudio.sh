#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --account=plgdynamic2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_out/train_sae/stableaudio/freq/musiccaps_tf11attn2_ef4_k32_lr8e-6_epochs2.out
#SBATCH --job-name=stableaudio_musiccaps_tf11attn2_ef4_k32_lr8e-6_epochs2

module load ML-bundle/24.06a
cd /net/scratch/hscra/plgrid/plglukaszst/projects/music-diffsae
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

python src/scripts/train_stableaudio.py --dataset_path activations/dpmscheduler/musiccaps_public2_10s_alongfreq/stable-audio-open-1.0 --hookpoints transformer_blocks.11.attn2 --expansion_factor 4 -k 32 --lr 8e-6 --num_epochs 2
