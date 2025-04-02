#!/bin/bash
# This script is used to run our Contrastive Learning based Vulnerability Detection technique
# Author: Satyaki Das
# Date: 2025-02-19

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error and exit immediately
set -o pipefail  # Consider a pipeline to fail if any command in the pipeline fails

# Your script starts here

export PATH=/media/satyaki/160f047a-dacc-449d-9208-a75717e899a1/research/miniconda3/bin:$PATH

export TMPDIR=/media/satyaki/160f047a-dacc-449d-9208-a75717e899a1/pip_tmp

conda remove --name cl_vul_det --all
conda create -n cl_vul_det python=3.10
conda activate cl_vul_det

pip install --cache-dir /media/satyaki/160f047a-dacc-449d-9208-a75717e899a1/pip_cache -r requirements.txt

PYTHONPATH="." python src/generate_vf_slices.py --use_temp_data --only_clear_slices &&
PYTHONPATH="." python src/generate_vf_slices.py --use_temp_data &&
PYTHONPATH="." python src/tokenize_slices.py --use_temp_data &&
PYTHONPATH="." python src/generate_file_slice_mapping.py --use_temp_data

PYTHONPATH="." python src/generate_vf_slices.py --only_clear_slices &&
PYTHONPATH="." python src/generate_vf_slices.py &&
PYTHONPATH="." python src/tokenize_slices.py &&
PYTHONPATH="." python src/generate_file_slice_mapping.py