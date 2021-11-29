#!/bin/bash

# Author : Bhishma Dedhia

cluster="della"
id="bdedhia"
exp="basque"
partition="gpu"
random_seed="0"
config="../exp_configs/basque/config_pretrain.yaml"
op_path="../logs_dir/basque/pretrain/"
data_path='../datasets/LunarLander/llander_file1.npz'

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo
   echo -e "Syntax: source ${CYAN}job_creator_script.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-e${ENDC} | ${YELLOW}--exp_name${ENDC} [default = ${GREEN}\"lunarlander\"${ENDC}] Selected Experiment"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"della\"${ENDC}]  Selected cluster - adroit or della or tiger"
   echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"bdedhia\"${ENDC}] Selected PU-NetID to email slurm updates"
   echo -e "${YELLOW}-p${ENDC} | ${YELLOW}--partition${ENDC} [default = ${GREEN}\"gpu\"${ENDC}] GPU partition"
   echo -e "${YELLOW}-l${ENDC} | ${YELLOW}--config${ENDC} [default = ${GREEN}\"../exp_configs/lunarlander/\"${ENDC}] Input directory"
   echo -e "${YELLOW}-o${ENDC} | ${YELLOW}--op_path${ENDC} [default = ${GREEN}\"../logs_dir/lunarlander/\"${ENDC}] Output directory"
   echo -e "${YELLOW}-s${ENDC} | ${YELLOW}--random_seed${ENDC} [default = ${GREEN}\"0\"${ENDC}] Random Seed"
   echo -e "${YELLOW}-d${ENDC} | ${YELLOW}--datapath${ENDC} [default = ${GREEN}\".../dataSets/LunarLander/llander_file1.npz\"${ENDC}] Datapath"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -e | --exp_name)
        shift
        exp=$1
        shift
        ;;
    -c | --cluster)
        shift
        cluster=$1
        shift
        ;;
    -i | --id)
        shift
        id=$1
        shift
        ;;
    -p | --partition)
        shift
        partition=$1
        shift
        ;;
    -l | --config)
        shift
        config=$1
        shift
       ;;
    -o | --op_path)
        shift
        op_path=$1
        shift
       ;;
    -r | --random_seed)
        shift
        random_seed=$1
        shift
       ;;
    -s | --sampling)
        shift
        sampling=$1
        shift
        ;;
    -k | --checkpoint)
        shift
        checkpoint=$1
        shift
       ;;
    -d | --datapath)
    		shift
    		datapath=$1
    		shift
  		;;
    -h | --help)
       Help
       return 1;
       ;;
    *)
       echo "Unrecognized flag $1"
       return 1;
       ;;
esac
done  

if [[ $cluster == "adroit" ]]
then
  cluster_gpu="gpu:tesla_v100:4"
elif [[ $cluster == "tiger" ]]
then
  cluster_gpu="gpu:4"
elif [[ $cluster == "della" ]]
then
  cluster_gpu="gpu:2"
else
	echo "Unrecognized cluster"
	return 1
fi


cd ..

cd scripts/

job_file="exp_${exp}_${random_seed}.slurm"

# Create SLURM job script to train surrogate model
echo "#!/bin/bash
#SBATCH --job-name=exp_${exp}_${random_seed}       # create a short name for your job 
#SBATCH --partition ${partition}
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=${cluster_gpu}               # number of gpus per node
#SBATCH --time=10:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=${id}@princeton.edu
module purge
module load anaconda3/2020.7
conda activate txf_design-space
cd ..
cd experiment/
python -u  pretrain.py --exp_name ${exp} --datapath ${datapath} --config ${config} --op_path ${op_path} --random_seed ${random_seed} 

sbatch $job_file

cd ..

cd utils/
