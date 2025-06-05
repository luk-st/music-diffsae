#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --account=plgdynamic2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_out/train_sae/stableaudio_musiccaps_tf11attn2_ef4_k64_lr8e-6_epoch2.out
#SBATCH --job-name=stableaudio_musiccaps_tf11attn2_ef4_k64_lr8e-6_epoch2

module load ML-bundle/24.06a
cd /net/scratch/hscra/plgrid/plglukaszst/projects/music-diffsae
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

python src/scripts/train_stableaudio.py --dataset_path activations/musiccaps_public_2_along_time/stable-audio-open-1.0 --hookpoints transformer_blocks.11.attn2 --expansion_factor 4 -k 64
