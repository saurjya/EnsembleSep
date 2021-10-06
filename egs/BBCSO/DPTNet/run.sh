#!/bin/bash
set -e  # Exit on error

#if starting from stage 0
# Destination to save json files with list of track locations for instrument sets
train_json="/jmain02/home/J2AD002/jxm06/sxs01-jxm06/data/BBCSO/train_mono.json"
val_json="/jmain02/home/J2AD002/jxm06/sxs01-jxm06/data/BBCSO/val_mono.json"
test_json="/jmain02/home/J2AD002/jxm06/sxs01-jxm06/data/BBCSO/test_mono.json"
# Location for tracklist for all data dirs
tracklist=  # Directory containing tracklists for V1, V2, Bach10 and others

# Location for MedleyDB V1
V1_dir= # Directory containing MedleyDB V1 audio files

# Location for MedleyDB V2
V2_dir= # Directory containing MedleyDB V2 audio files

# Location for Bach10
Bach10_dir=  # Directory containing MedleyDB format Bach10 audio files

# Location for additional MedleyDB format multitracks
extra_dir= # Directory containing additional MedleyDB format audio files

# Location for MedleyDB format metadata files for all multitracks
#metadata_dir=/jmain01/home/JAD007/txk02/sxs01-txk02/metadata/medleydb # Directory containing MedleyDB github repository with metadata for all files

# Location for evaluation multitrack sourceFolders
#wav_dir=/jmain01/home/JAD007/txk02/sxs01-txk02/data/fix/split_2/tt/ # Directory containing MedleyDB github repository with metadata for all files



# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --loss_alpha 0.1 --id 0,1

# General
stage=3  # Controls from which stage to start
tag="BBCSO_new_2sep_6r_16f_violinviola1only"  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
# Data
#data_dir=data  # Local data directory (No disk space needed)
sample_rate=22050
n_src=2
segment=220500
# Training
batch_size=4
#num_workers=10
optimizer=adam
#lr=0.0005
#weight_decay=0.0
epochs=200
#loss_alpha=1.0  # DC loss weight : 1.0 => DC, <1.0 => Chimera
take_log=true  # Whether to input log mag spec to the NN

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=${n_inst}inst${n_poly}poly${sr_string}sr${segment}sec
dumpdir=data/$suffix  # directory to put generated json file

#json_dir=$dumpdir
is_raw=True

if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Downloading MedleyDB repo for tracklist"
  mkdir -p $metadata_dir
  git clone https://github.com/marl/medleydb.git  $metadata_dir
  fi

if [[ $stage -le  1 ]]; then
  echo "Stage 1: Download MedleyDB dataset, update dataset path, add custom tracklist if required"
fi

if [[ $stage -le  2 ]]; then
	# Make json files with wav paths for instrument set
	echo "Stage 2: Generating json files including wav path and activity info"
  $python_path local/preprocess_medleyDB.py --metadata_path $metadata_dir --json_dir $json_dir --v1_path "$V1_dir" --v2_path "$V2_dir"

fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${n_src}sep_${sr_string}k${mode}_${uuid}
fi
#expdir=/jmain01/home/JAD029/txl22/sxs01-txl22/workspace/asteroid/egs/BBCSO/DPTNet/exp/train_convtasnet_BBCSO_trial
if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  expdir=exp/train_convtasnet_${tag}
  mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
  echo "Results from the following experiment will be stored in $expdir"

  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_json $train_json \
		--val_json $val_json \
		--sample_rate $sample_rate \
		--epochs $epochs \
		--batch_size $batch_size \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	    --n_src $n_src \
		--json_dir $test_json \
		--use_gpu $eval_use_gpu \
		--batch_size $batch_size \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
