#!/bin/bash
#SBATCH --job-name=exp_synthetic3_0       # create a short name for your job 
#SBATCH --partition gpu
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1               # number of gpus per node
#SBATCH --time=03:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=bdedhia@princeton.edu
module purge
module load anaconda3/2020.7
conda activate txf_design-space
cd ..
cd experiment 
python -u fitandpredict.py --exp_name synthetic3 --datapath ../datasets/synthetic_data3/  --config ../exp_configs/synthetic3/config.yaml --op_path ../logs_dir/synthetic3_lowrank/ --random_seed 0 --lowrank low_rank
