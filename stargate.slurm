#!/bin/bash

# --- Make sure preprocessed spectrograms are provided beforehand so compute nodes don't have to do it themselves in parallel

#SBATCH --job-name=AiTunes
#SBATCH --output=AiTunes_J%A_t%a.out
#SBATCH --error=AiTunes_J%A_t%a.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=byron.cholet@etud.univ-angers.fr
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=intel-E5-2695
#SBATCH --mem-per-cpu=1G

source ~/.venv/bin/activate

TASK_PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" task_params.txt)
python3 main.py $TASK_PARAMS

deactivate
