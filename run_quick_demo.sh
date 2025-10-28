#!/bin/bash
# ===============================================================
#  Script: run_quick_demo.sh
#  Purpose: Quick demo run for SGMSE (Score-based Generative Models for Speech Enhancement)
#           Train only 10 epochs and test on a small subset.
#  ===============================================================

# === [1] Global config ===
PROJECT_ROOT=$(pwd)
EXP_DIR="$PROJECT_ROOT/demo_experiments"
DATA_DIR="$PROJECT_ROOT/data"
BACKBONE="ncsnpp"
SDE_TYPE="ve"
BATCH_SIZE=2
LR=1e-4
EPOCHS=10
NOLOG="--nolog"

# === [2] Dataset (change this if you have your own small dataset) ===
# Structure required:
# data/
#   demo/
#     train/clean/, train/noisy/
#     valid/clean/, valid/noisy/
#     test/clean/,  test/noisy/
DEMO_DATA="$DATA_DIR/demo"

# === [3] Train a small model ===
echo "🚀 [TRAIN] Starting short training for 10 epochs ..."
python train.py \
    --base_dir $DEMO_DATA \
    --out_dir "$EXP_DIR/sgmse_demo" \
    --backbone $BACKBONE \
    --sde $SDE_TYPE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --epochs $EPOCHS \
    --n_fft 512 \
    --hop_length 128 \
    --spec_factor 0.15 \
    --spec_abs_exponent 0.5 \
    $NOLOG

# === [4] Enhance (inference) a few files ===
echo "🎧 [TEST] Enhancing sample noisy speech ..."
python enhancement.py \
    --ckpt "$EXP_DIR/sgmse_demo/checkpoints/epoch=326-step=408750.ckpt" \
    --input_dir "$DEMO_DATA/test/noisy" \
    --output_dir "$EXP_DIR/sgmse_demo/enhanced" \
    --sampler pc \
    --corrector_steps 1 \
    --num_steps 10

# === [5] Calculate metrics (quick check) ===
echo "📈 [EVAL] Calculating basic metrics (PESQ, STOI, SI-SDR) ..."
python calc_metrics.py \
    --ref_dir "$DEMO_DATA/test/clean" \
    --est_dir "$EXP_DIR/sgmse_demo/enhanced" \
    --metrics pesq stoi sisdr \
    --save_to "$EXP_DIR/results_demo.json"

echo "✅ Quick demo complete! Check results in $EXP_DIR"
