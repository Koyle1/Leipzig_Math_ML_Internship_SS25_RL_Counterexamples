#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=wagner_comb
#SBATCH --partition=paul
#SBATCH --mem=2G
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_80,
#SBATCH --output=/home/sc.uni-leipzig.de/kv99fuda/dev/Leipzig_Math_ML_Internship_SS25_RL_Counterexamples/wagner_combinatorics/out/%x-%j.out

cd /home/sc.uni-leipzig.de/kv99fuda/dev/Leipzig_Math_ML_Internship_SS25_RL_Counterexamples/wagner_combinatorics
source /home/sc.uni-leipzig.de/kv99fuda/dev/cross-entropy-for-combinatorics/venv/bin/activate

which python
python --version

python train.py -r $SLURM_ARRAY_TASK_ID -v
