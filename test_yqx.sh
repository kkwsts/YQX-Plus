#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8        # 8 cores (8 cores per GPU)
#$ -l h_rt=1:0:0    
#$ -l h_vmem=7.5G    # 11 * 8 = 88G total RAM
# $ -l gpu=1         # request 1 GPU

# source ~/.bashrc
echo "Allocated GPU(s): $SGE_HGR_gpu"
source .venv/bin/activate

export WANDB_DISABLED=true

WANDB_DISABLED=True python yqx.py \
    train.enabled=false \
    test.enabled=true \
    model.type=flow \
    model.feature_experiment=long_context \