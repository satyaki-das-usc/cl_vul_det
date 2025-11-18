#!/bin/bash
# This script is used to run our Contrastive Learning based Vulnerability Detection technique
# Author: Satyaki Das
# Date: 2025-02-19

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error and exit immediately
set -o pipefail  # Consider a pipeline to fail if any command in the pipeline fails

# Your script starts here
# For Linux x86 architecture
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
# For ARM64 architecture
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash ~/Miniconda3-latest-Linux-aarch64.sh

git clone https://github.com/satyaki-das-usc/cl_vul_det.git
cd cl_vul_det

export PATH=/media/satyaki/160f047a-dacc-449d-9208-a75717e899a1/research/miniconda3/bin:$PATH

export TMPDIR=/media/satyaki/160f047a-dacc-449d-9208-a75717e899a1/pip_tmp

conda remove --name cl_vul_det --all
# For Linux x86 architecture
conda create -n cl_vul_det python=3.10 notebook ipython -y
# For ARM64 architecture
conda create -n cl_vul_det -c conda-forge pytorch-gpu python=3.10 -y
conda deactivate
conda activate cl_vul_det

pip install --cache-dir /media/satyaki/160f047a-dacc-449d-9208-a75717e899a1/pip_cache -r requirements.txt
pip install -r requirements.txt

PYTHONPATH="." python src/generate_vf_slices.py --use_temp_data --only_clear_slices &&
PYTHONPATH="." python src/generate_vf_slices.py --use_temp_data &&
PYTHONPATH="." python src/tokenize_slices.py --use_temp_data &&
PYTHONPATH="." python src/generate_file_slice_mapping.py --use_temp_data &&
PYTHONPATH="." python src/preprocess/remove_duplicates.py --use_temp_data &&
PYTHONPATH="." python src/preprocess/word_embedding.py --use_temp_data &&
PYTHONPATH="." python src/preprocess/split_dataset.py --use_temp_data

PYTHONPATH="." python src/generate_vf_slices.py --use_nvd --only_clear_slices &&
PYTHONPATH="." python src/generate_vf_slices.py --use_nvd &&
PYTHONPATH="." python src/tokenize_slices.py --use_nvd &&

PYTHONPATH="." python src/generate_vf_slices.py --only_clear_slices &&
PYTHONPATH="." python src/generate_vf_slices.py &&
PYTHONPATH="." python src/tokenize_slices.py &&
PYTHONPATH="." python src/generate_file_slice_mapping.py &&
PYTHONPATH="." python src/preprocess/remove_duplicates.py &&
xargs -d '\n' -a duplicate_slices.txt rm -- &&
PYTHONPATH="." python src/preprocess/split_dataset.py &&
PYTHONPATH="." python src/preprocess/word_embedding.py &&
PYTHONPATH="." python src/swav/main_swav_vul_det.py &&
PYTHONPATH="." python src/swav/main_swav_vul_det.py --skip_training &&
PYTHONPATH="." python src/preprocess/generate_instance_perturbation_mapping.py &&
PYTHONPATH="." python src/preprocess/generate_swav_batches.py &&
PYTHONPATH="." python src/preprocess/generate_multicrop_swav_batches.py &&
PYTHONPATH="." python src/preprocess/generate_multicrop_swav_batches.py --do_train
# PYTHONPATH="." python src/preprocess/split_large_batches.py


PYTHONPATH="." python src/preprocess/generate_custom_balanced_batches.py -s instance
PYTHONPATH="." python src/preprocess/generate_custom_balanced_batches.py -s vf
PYTHONPATH="." python src/preprocess/generate_custom_balanced_batches.py -s swav

# screen -S instance_representation -d -m bash -c "PYTHONPATH='.' python src/run_representation.py -s instance --exclude_NNs"
# screen -S vf_representation -d -m bash -c "PYTHONPATH='.' python src/run_representation.py -s vf --exclude_NNs"
# screen -S swav_representation -d -m bash -c "PYTHONPATH='.' python src/run_representation.py -s swav --exclude_NNs"

# screen -S instance_representation -d -m bash -c "PYTHONPATH='.' python src/run_representation.py -s instance"
# screen -S vf_representation -d -m bash -c "PYTHONPATH='.' python src/run_representation.py -s vf"
# screen -S swav_representation -d -m bash -c "PYTHONPATH='.' python src/run_representation.py -s swav"

screen -S multicrop_swav -d -m bash -c "PYTHONPATH='.' python src/preprocess/generate_multicrop_swav_batches.py"

screen -S instance -d -m bash -c "PYTHONPATH='.' python src/run_classification.py -s instance"
screen -S vf -d -m bash -c "PYTHONPATH='.' python src/run_classification.py -s vf"
screen -S swav -d -m bash -c "PYTHONPATH='.' python src/run_classification.py -s swav"

screen -S instance_no_cl -d -m bash -c "PYTHONPATH='.' python src/run_classification.py -s instance --no_cl"
screen -S vf_no_cl -d -m bash -c "PYTHONPATH='.' python src/run_classification.py -s vf --no_cl"
screen -S swav_no_cl -d -m bash -c "PYTHONPATH='.' python src/run_classification.py -s swav --no_cl"


PYTHONPATH="." python src/run_classification.py -s instance && PYTHONPATH="." python src/run_classification.py -s vf && PYTHONPATH="." python src/run_classification.py -s swav

PYTHONPATH="." python src/run_classification.py -s instance --exclude_NNs && PYTHONPATH="." python src/run_classification.py -s vf --exclude_NNs && PYTHONPATH="." python src/run_classification.py -s swav --exclude_NNs

PYTHONPATH="." python src/run_classification.py -s instance --no_cl_warmup && PYTHONPATH="." python src/run_classification.py -s vf --no_cl_warmup && PYTHONPATH="." python src/run_classification.py -s swav --no_cl_warmup

PYTHONPATH="." python src/run_classification.py -s instance --exclude_NNs --no_cl_warmup && PYTHONPATH="." python src/run_classification.py -s vf --exclude_NNs --no_cl_warmup && PYTHONPATH="." python src/run_classification.py -s swav --exclude_NNs --no_cl_warmup


PYTHONPATH="." python src/run_classification.py -s instance &&
PYTHONPATH="." python src/run_classification.py -s instance --no_cl &&
PYTHONPATH="." python src/run_classification.py -s vf &&
PYTHONPATH="." python src/run_classification.py -s vf --no_cl &&
PYTHONPATH="." python src/run_classification.py -s swav &&
PYTHONPATH="." python src/run_classification.py -s swav --no_cl


PYTHONPATH="." python src/evaluate_all_model_configurations.py -p saved_models/graph_swav_classification/VF_perts/GINE/IncludeNN/NoCLWarmup/InfoNCEGraphSwAVVD.ckpt
PYTHONPATH="." python src/generate_nvd_ground_truth.py


