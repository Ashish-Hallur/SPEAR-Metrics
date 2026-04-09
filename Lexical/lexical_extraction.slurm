#!/bin/bash
#SBATCH --job-name=lexical_seamless
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0-430
#SBATCH --output=/home/ahallur1/spear/Seamless_Experiments/Lexical/logs/lexical_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/Seamless_Experiments/Lexical/logs/lexical_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

module purge
module load conda
conda deactivate
conda activate /home/ahallur1/.conda/envs/py310

NUM_SHARDS=431

python /home/ahallur1/spear/Seamless_Experiments/Lexical/run_lexical_extraction.py \
    ${SLURM_ARRAY_TASK_ID} ${NUM_SHARDS}