#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=80:00:00
# set number of GPUs
#SBATCH --gres=gpu:4
#SBATCH --mem=128000
# if you need the custom conda environment:
#SBATCH --qos turing
module purge
module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3
module load cuDNN/8.0.4.30-CUDA-11.1.1
module load libsndfile
source activate asteroid1
which python
echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
./run.sh
