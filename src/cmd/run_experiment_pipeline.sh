#!/bin/bash

# stops the script if any command fails
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# experiment configuration
EPOCHS=2
BATCH_SIZE=128
LR="3e-4" # learning rate (LR) -> 3e-4 = 0.0003

echo "================================================================="
echo "Inicializing Experiment Pipeline: ViT LayerNorm vs. DyT"
echo "================================================================="
echo "CONFIG: Epochs=$EPOCHS | Batch Size=$BATCH_SIZE | LR=$LR"
echo ""

echo "-------------------------- 1. TRAINING --------------------------"

echo "-----------------------------------------------------------------"
echo "1.1. Training Baseline (ViT LayerNorm):"
echo "-----------------------------------------------------------------"
uv run src/cmd/train.py --model layernorm --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --timestamp $TIMESTAMP

echo "-----------------------------------------------------------------"
echo "1.2. Training Experimental Model (ViT DyT):"
echo "-----------------------------------------------------------------"
uv run src/cmd/train.py --model dyt --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --timestamp $TIMESTAMP


echo "-------------------------- 2. PREPARING TESTS -------------------"
LATEST_LAYERNORM_DIR=experiments/training/layernorm_$TIMESTAMP
LATEST_DYT_DIR=experiments/training/dyt_$TIMESTAMP

LAYERNORM_WEIGHTS="${LATEST_LAYERNORM_DIR}/layernorm_best_weights.pth"
DYT_WEIGHTS="${LATEST_DYT_DIR}/dyt_best_weights.pth"

echo "LayerNorm weights found in: $LAYERNORM_WEIGHTS"
echo "DyT weights found in: $DYT_WEIGHTS"
echo ""

echo "-------------------------- 3. TESTING ---------------------------"

echo "-----------------------------------------------------------------"
echo "3.1. Evaluating Baseline (ViT LayerNorm):"
echo "-----------------------------------------------------------------"
uv run src/cmd/evaluate.py --model layernorm --weights "$LAYERNORM_WEIGHTS" --timestamp $TIMESTAMP

echo "-----------------------------------------------------------------"
echo "3.2. Evaluating Experimental Model (ViT DyT):"
echo "-----------------------------------------------------------------"
uv run src/cmd/evaluate.py --model dyt --weights "$DYT_WEIGHTS" --timestamp $TIMESTAMP

echo ""
echo "================================================================="
echo "End of the pipeline. Verify directory experiments/"
echo "================================================================="