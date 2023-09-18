#!/bin/bash
source /home/bsft19/xuezhwang2/anaconda3/etc/profile.d/conda.sh
conda activate py37 
mkdir -p logs
# Define the path to the .ini file
ini_file="./configs.ini"


# Use awk to extract the value of the parameter from the .ini file
taxa=$(awk -F '=' '{if (! ($0 ~ /^;/) && $0 ~ /taxa/) print $2}' $ini_file)
seeds=$(awk -F '=' '{if (! ($0 ~ /^;/) && $0 ~ /seed/) print $2}' $ini_file)
# seeds=(4 13 42 71 265)
python3 baseline_inverse.py --dataset "hostg.${taxa}" --seeds $((seeds))
# for seed in "${seeds[@]}"; do
#     python3 main.py --dataset "hostg.${taxa}" --seeds $seed
# done 


