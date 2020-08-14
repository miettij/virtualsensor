#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'

cd $WRKDIR/virse/code/train_bdrnn/

module load anaconda3/latest

srun python main.py
