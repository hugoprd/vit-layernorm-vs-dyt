#!/bin/bash

# stops the script if any command fails
set -e

# experiment configuration
EPOCHS=2
BATCH_SIZE=128
LR="3e-4" # learning rate (LR) -> 3e-4 = 0.0003

echo "================================================================="
echo "Inicializing Experiment Pipeline: ViT LayerNorm vs. DyT"
echo "================================================================="
echo "CONFIG: Epochs=$EPOCHS | Batch Size=$BATCH_SIZE | LR=$LR"
echo ""

echo "-----------------------------------------------------------------"
echo "1. Training Baseline (ViT LayerNorm):"
echo "-----------------------------------------------------------------"
uv run src/cmd/train.py --model layernorm --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR

echo "-----------------------------------------------------------------"
echo "2. Training Experimental Model (ViT DyT):"
echo "-----------------------------------------------------------------"
uv run src/cmd/train.py --model dyt --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR

echo ""
echo "================================================================="
echo "End of the pipeline. Verify directory experiments/"
echo "================================================================="