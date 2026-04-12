#!/bin/bash

#SBATCH --job-name=vpr_gem_test

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_access

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH --output=results/gem_training_%j.log



# Load environment

module load anaconda

conda activate vpr_env



# Single-line execution: Train on Dataset A, then immediately run evaluation

python train_example.py --data_root ./datasets/dataset_a --save_dir ./checkpoints/gem_v1 --loss contrastive --epochs 20 --batch_size 32 --lr 0.0001 && python train_example.py --predict --dataset_root ./datasets/dataset_a --dataset_name dataset_a --checkpoint ./checkpoints/gem_v1/best_model.pth --output ./results/gem_dataset_a.csv && python evaluate.py --student_id "Kerry_Jin" --prediction ./results/gem_dataset_a.csv --datasets dataset_a
