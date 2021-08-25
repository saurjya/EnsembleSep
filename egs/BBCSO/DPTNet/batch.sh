#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=1:00:00
# set name of job
#SBATCH -J bbcso
#SBATCH --partition=devel
# set number of GPUs
#SBATCH --gres=gpu:1
# load the anaconda module
#module load python3/anaconda
# if you need the custom conda environment:
#source activate custom
source activate asteroid1
which python
echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=saurjya.sarkar@qmul.ac.uk
# execute the program
./run.sh