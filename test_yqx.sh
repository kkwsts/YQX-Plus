

export WANDB_DISABLED=true

WANDB_DISABLED=True python yqx.py \
    train.enabled=false \
    test.enabled=true \
    model.type=flow \
    model.feature_experiment=full_context \