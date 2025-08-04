#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8        # 8 cores (8 cores per GPU)
#$ -l h_rt=240:0:0    
#$ -l h_vmem=7.5G    # 11 * 8 = 88G total RAM
# $ -l gpu=1         # request 1 GPU

source ~/.bashrc
echo "Allocated GPU(s): $SGE_HGR_gpu"
source .venv/bin/activate

export WANDB_API_KEY=47e8ce799bbaf0f8b5664b5d9db3792d7176e163

WANDB_DISABLED=false python yqx.py \
    train.enabled=true \
    model.type=gmm \
    data.use_vienna4x22=false \
    data.use_asap=true \
    data.use_atepp=true \
    model.feature_experiment=long_context \

