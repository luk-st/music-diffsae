#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --account=plgdynamic2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_out/train_sae/audioldm2_up1tf5at0.out
#SBATCH --job-name=train_sae_audioldm2_up1tf5at0

module load ML-bundle/24.06a
cd /net/scratch/hscra/plgrid/plglukaszst/projects/music-diffsae
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

python src/scripts/train.py --dataset_path activations/musiccaps_public_8/audioldm2-large --hookpoints up_blocks.1.attentions.5.transformer_blocks.0

# up_blocks.1.attentions.5.transformer_blocks.0
# up_blocks.1.attentions.5.transformer_blocks.1
# up_blocks.1.attentions.10.transformer_blocks.0
# up_blocks.1.attentions.10.transformer_blocks.1
