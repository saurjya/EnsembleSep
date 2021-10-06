#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=24:00:00
# set name of job
#SBATCH -J bbcso_close
#SBATCH --partition=big
# set number of GPUs
#SBATCH --gres=gpu:8
# load the anaconda module
#module load python3/anaconda
# if you need the custom conda environment:
#source activate custom
source activate asteroid2
which python
echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=saurjya.sarkar@qmul.ac.uk
# execute the program
./run.sh