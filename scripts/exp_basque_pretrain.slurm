#!/bin/bash
#SBATCH --job-name=exp_basque_0       # create a short name for your job    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2               # number of gpus per node
#SBATCH --time=12:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=bdedhia@princeton.edu
module purge
module load anaconda3/2020.7
conda activate txf_design-space
cd ..
cd experiment/
python -u  pretrain.py --exp_name basque --datapath ../datasets/basque/ --config ../exp_configs/basque/config_pretrain.yaml --op_path ../logs_dir/basque/pretrain/ --random_seed 0
