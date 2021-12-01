#!/bin/bash
#SBATCH --job-name=exp_basque_0       # create a short name for your job 
#SBATCH --partition gpu
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2               # number of gpus per node
#SBATCH --time=12:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=bdedhia@princeton.edu
module purge
module load anaconda3/2020.7
conda activate txf_design-space
cd ..
cd experiment/
python -u  pretrain.py --exp_name basque --datapath ../datasets/basque/ --config ../exp_configs/basque/config_pretrain.yaml --op_path ../logs_dir/basque_lowrank/pretrain/ --random_seed 0 --data_transform lowrank
