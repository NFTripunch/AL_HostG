#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1     # 32 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH TASK, i.e. 1x32x4=128 cores
#SBATCH --mem=440GB              # 440GB mem on EACH NODE
#SBATCH --time=0-12:00:00          # Time limit hrs:min:sec
#SBATCH --output=logs/hostg_training_%j.log      # Standard output
#SBATCH --error=logs/hostg_training_%j.err       # Standard error log

source /gpfs1/home/yixihuang9/anaconda3/etc/profile.d/conda.sh
conda activate py37 
mkdir -p logs
# python3 main.py --dataset citeseer --num_classes 6 --num_nodes 3327

# python main.py 
python main.py --dataset hostg.phylum --num_classes 169 --num_nodes 58697
# python main.py --dataset hostg.order --num_classes 227 --num_nodes 58697
# python main.py --dataset hostg.class --num_classes 108 --num_nodes 58697
# python main.py --dataset hostg.family --num_classes 494 --num_nodes 58697
# python main.py --dataset hostg.genus --num_classes 2573 --num_nodes 58697
