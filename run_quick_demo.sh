#!/bin/bash

# Script to reproduce experiments from SGMSE+ paper
# Tables II, III, IV from the paper

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base paths
BASE_DIR=$(pwd)
DATA_DIR="$BASE_DIR/data"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
RESULTS_DIR="$BASE_DIR/results"

# Create necessary directories
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0  # Modify based on available GPUs

# Training configuration
EPOCHS=3
BATCH_SIZE=8
NUM_GPUS=1  # Paper uses 4, adjust based on available resources

# Sampling configuration
N_STEPS=30
CORRECTOR_STEPS=1

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_python_deps() {
    log "Checking Python dependencies..."
    python -c "import torch; import pytorch_lightning; import pesq; import pystoi" 2>/dev/null || {
        log "ERROR: Missing dependencies. Please run: pip install -r requirements.txt"
        exit 1
    }
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

prepare_wsj0_chime3() {
    log "Preparing WSJ0-CHiME3 dataset..."

    if [ ! -d "$DATA_DIR/wsj0_chime3" ]; then
        log "WSJ0-CHiME3 not found. Please download:"
        log "  - WSJ0: https://catalog.ldc.upenn.edu/LDC93S6A"
        log "  - CHiME3 noise: https://catalog.ldc.upenn.edu/LDC2015S23"
        log "Then run the data preparation script"
        # Uncomment and adjust paths:
        # python create_dataset.py --wsj0_path /path/to/wsj0 --chime3_path /path/to/chime3 --output $DATA_DIR/wsj0_chime3
        return 1
    fi

    log "WSJ0-CHiME3 dataset ready"
    return 0
}

prepare_vb_dmd() {
    log "Preparing VoiceBank-DEMAND dataset..."

    if [ ! -d "$DATA_DIR/vb_dmd" ]; then
        log "Downloading VoiceBank-DEMAND..."
        # Download from official source
        # The dataset should be organized as per the paper
        log "Please download VB-DMD manually from:"
        log "https://datashare.ed.ac.uk/handle/10283/2791"
        return 1
    fi

    log "VB-DMD dataset ready"
    return 0
}

prepare_wsj0_reverb() {
    log "Preparing WSJ0-REVERB dataset..."

    if [ ! -d "$DATA_DIR/wsj0_reverb" ]; then
        log "Creating WSJ0-REVERB with simulated RIRs..."
        # This creates reverberant data using pyroomacoustics
        python scripts/create_reverb_dataset.py \
            --wsj0_path "$DATA_DIR/wsj0" \
            --output "$DATA_DIR/wsj0_reverb" \
            --t60_min 0.4 \
            --t60_max 1.0 \
            --absorption_coef 0.99
    fi

    log "WSJ0-REVERB dataset ready"
    return 0
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

train_speech_enhancement() {
    local DATASET=$1
    local MODEL_NAME=$2

    log "Training SGMSE+ on $DATASET for speech enhancement..."

    python train.py \
        --base_dir "$DATA_DIR/$DATASET" \
        --gpus $NUM_GPUS \
        --strategy ddp \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --num_workers 8 \
        --learning_rate 1e-4 \
        --ema_decay 0.999 \
        --sigma_min 0.05 \
        --sigma_max 0.5 \
        --gamma 1.5 \
        --t_eps 0.03 \
        --N $N_STEPS \
        --corrector_steps $CORRECTOR_STEPS \
        --snr_min 0 \
        --snr_max 20 \
        --default_root_dir "$CHECKPOINT_DIR/$MODEL_NAME" \
        --model_name "$MODEL_NAME"

    log "Training completed: $MODEL_NAME"
}

train_dereverberation() {
    local DATASET="wsj0_reverb"
    local MODEL_NAME="sgmse_plus_reverb"

    log "Training SGMSE+ on $DATASET for dereverberation..."

    python train.py \
        --base_dir "$DATA_DIR/$DATASET" \
        --gpus $NUM_GPUS \
        --strategy ddp \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --num_workers 8 \
        --learning_rate 1e-4 \
        --ema_decay 0.999 \
        --sigma_min 0.05 \
        --sigma_max 0.5 \
        --gamma 1.5 \
        --t_eps 0.03 \
        --N $N_STEPS \
        --corrector_steps $CORRECTOR_STEPS \
        --default_root_dir "$CHECKPOINT_DIR/$MODEL_NAME" \
        --model_name "$MODEL_NAME" \
        --task dereverberation

    log "Training completed: $MODEL_NAME"
}

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

evaluate_model() {
    local CHECKPOINT=$1
    local TEST_DIR=$2
    local OUTPUT_DIR=$3
    local TASK=$4

    log "Evaluating model: $CHECKPOINT on $TEST_DIR"

    python enhancement.py \
        --test_dir "$TEST_DIR" \
        --enhanced_dir "$OUTPUT_DIR/enhanced" \
        --ckpt "$CHECKPOINT" \
        --N $N_STEPS \
        --corrector_steps $CORRECTOR_STEPS \
        --task "$TASK"

    # Calculate metrics
    log "Calculating metrics..."
    python calculate_metrics.py \
        --clean_dir "$TEST_DIR/clean" \
        --noisy_dir "$TEST_DIR/noisy" \
        --enhanced_dir "$OUTPUT_DIR/enhanced" \
        --output_file "$OUTPUT_DIR/metrics.json" \
        --metrics pesq estoi sisdr polqa dnsmos

    log "Evaluation results saved to: $OUTPUT_DIR/metrics.json"
}

# ============================================================================
# EXPERIMENT 1: TABLE II - Speech Enhancement on WSJ0-CHiME3
# ============================================================================

experiment_table_ii_matched() {
    log "=========================================="
    log "EXPERIMENT: Table II - Matched Condition"
    log "=========================================="

    # Train on WSJ0-CHiME3
    if [ ! -f "$CHECKPOINT_DIR/sgmse_plus_wsj0_chime3/last.ckpt" ]; then
        train_speech_enhancement "wsj0_chime3" "sgmse_plus_wsj0_chime3"
    fi

    # Evaluate on WSJ0-CHiME3 test set (matched)
    evaluate_model \
        "$CHECKPOINT_DIR/sgmse_plus_wsj0_chime3/last.ckpt" \
        "$DATA_DIR/wsj0_chime3/test" \
        "$RESULTS_DIR/table_ii_matched" \
        "enhancement"
}

experiment_table_ii_mismatched() {
    log "============================================="
    log "EXPERIMENT: Table II - Mismatched Condition"
    log "============================================="

    # Train on VB-DMD
    if [ ! -f "$CHECKPOINT_DIR/sgmse_plus_vb_dmd/last.ckpt" ]; then
        train_speech_enhancement "vb_dmd" "sgmse_plus_vb_dmd"
    fi

    # Evaluate on WSJ0-CHiME3 test set (mismatched)
    evaluate_model \
        "$CHECKPOINT_DIR/sgmse_plus_vb_dmd/last.ckpt" \
        "$DATA_DIR/wsj0_chime3/test" \
        "$RESULTS_DIR/table_ii_mismatched" \
        "enhancement"
}

# ============================================================================
# EXPERIMENT 2: TABLE III - Speech Enhancement on VB-DMD
# ============================================================================

experiment_table_iii() {
    log "======================================="
    log "EXPERIMENT: Table III - VB-DMD Benchmark"
    log "======================================="

    # Use model trained on VB-DMD
    if [ ! -f "$CHECKPOINT_DIR/sgmse_plus_vb_dmd/last.ckpt" ]; then
        train_speech_enhancement "vb_dmd" "sgmse_plus_vb_dmd"
    fi

    # Evaluate on VB-DMD test set
    evaluate_model \
        "$CHECKPOINT_DIR/sgmse_plus_vb_dmd/last.ckpt" \
        "$DATA_DIR/vb_dmd/test" \
        "$RESULTS_DIR/table_iii" \
        "enhancement"
}

# ============================================================================
# EXPERIMENT 3: TABLE IV - Dereverberation on WSJ0-REVERB
# ============================================================================

experiment_table_iv() {
    log "==========================================="
    log "EXPERIMENT: Table IV - Dereverberation"
    log "==========================================="

    # Prepare reverberant dataset
    prepare_wsj0_reverb

    # Train on WSJ0-REVERB
    if [ ! -f "$CHECKPOINT_DIR/sgmse_plus_reverb/last.ckpt" ]; then
        train_dereverberation
    fi

    # Evaluate on WSJ0-REVERB test set
    evaluate_model \
        "$CHECKPOINT_DIR/sgmse_plus_reverb/last.ckpt" \
        "$DATA_DIR/wsj0_reverb/test" \
        "$RESULTS_DIR/table_iv" \
        "dereverberation"
}

# ============================================================================
# EXPERIMENT 4: Real-world evaluation (Table V)
# ============================================================================

experiment_table_v() {
    log "==========================================="
    log "EXPERIMENT: Table V - Real-world DNS Challenge"
    log "==========================================="

    # Download DNS Challenge 2020 test set
    if [ ! -d "$DATA_DIR/dns_challenge_2020" ]; then
        log "Downloading DNS Challenge 2020 test set..."
        # Download 300 files from DNS Challenge 2020
        # This requires manual download or Azure blob access
        log "Please download DNS Challenge 2020 test set manually"
        return 1
    fi

    # Evaluate using model trained on VB-DMD
    python enhancement.py \
        --test_dir "$DATA_DIR/dns_challenge_2020/test" \
        --enhanced_dir "$RESULTS_DIR/table_v/enhanced" \
        --ckpt "$CHECKPOINT_DIR/sgmse_plus_vb_dmd/last.ckpt" \
        --N $N_STEPS \
        --corrector_steps $CORRECTOR_STEPS

    # Calculate non-intrusive metrics only (no clean reference)
    python calculate_metrics.py \
        --noisy_dir "$DATA_DIR/dns_challenge_2020/test/noisy" \
        --enhanced_dir "$RESULTS_DIR/table_v/enhanced" \
        --output_file "$RESULTS_DIR/table_v/metrics.json" \
        --metrics dnsmos sig bak ovrl wvmos \
        --no_reference
}

# ============================================================================
# ABLATION STUDY: Different sampler configurations (Table I)
# ============================================================================

ablation_samplers() {
    log "==========================================="
    log "ABLATION: Sampler configurations (Table I)"
    log "==========================================="

    CHECKPOINT="$CHECKPOINT_DIR/sgmse_plus_vb_dmd/last.ckpt"
    TEST_DIR="$DATA_DIR/vb_dmd/test"

    # PC sampler with 0 corrector steps
    python enhancement.py --test_dir "$TEST_DIR" \
        --enhanced_dir "$RESULTS_DIR/ablation/pc_0_corrector" \
        --ckpt "$CHECKPOINT" --N 30 --corrector_steps 0

    # PC sampler with 1 corrector step
    python enhancement.py --test_dir "$TEST_DIR" \
        --enhanced_dir "$RESULTS_DIR/ablation/pc_1_corrector" \
        --ckpt "$CHECKPOINT" --N 30 --corrector_steps 1

    # PC sampler with 2 corrector steps
    python enhancement.py --test_dir "$TEST_DIR" \
        --enhanced_dir "$RESULTS_DIR/ablation/pc_2_corrector" \
        --ckpt "$CHECKPOINT" --N 30 --corrector_steps 2

    # ODE sampler
    python enhancement.py --test_dir "$TEST_DIR" \
        --enhanced_dir "$RESULTS_DIR/ablation/ode_sampler" \
        --ckpt "$CHECKPOINT" --N 30 --sampler "ode"

    log "Calculating metrics for all sampler configurations..."
    for config in pc_0_corrector pc_1_corrector pc_2_corrector ode_sampler; do
        python calculate_metrics.py \
            --clean_dir "$TEST_DIR/clean" \
            --noisy_dir "$TEST_DIR/noisy" \
            --enhanced_dir "$RESULTS_DIR/ablation/$config" \
            --output_file "$RESULTS_DIR/ablation/${config}_metrics.json" \
            --metrics pesq sisdr
    done
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log "Starting SGMSE+ experiments"
    log "=========================================="

    # Check dependencies
    check_python_deps

    # Parse command line arguments
    EXPERIMENT=${1:-"all"}

    case $EXPERIMENT in
        "table2_matched")
            experiment_table_ii_matched
            ;;
        "table2_mismatched")
            experiment_table_ii_mismatched
            ;;
        "table3")
            experiment_table_iii
            ;;
        "table4")
            experiment_table_iv
            ;;
        "table5")
            experiment_table_v
            ;;
        "ablation")
            ablation_samplers
            ;;
        "all")
            log "Running all experiments..."
            experiment_table_ii_matched
            experiment_table_ii_mismatched
            experiment_table_iii
            experiment_table_iv
            experiment_table_v
            ablation_samplers
            ;;
        *)
            echo "Usage: $0 [table2_matched|table2_mismatched|table3|table4|table5|ablation|all]"
            echo ""
            echo "Options:"
            echo "  table2_matched    - Table II matched condition (train/test on WSJ0-CHiME3)"
            echo "  table2_mismatched - Table II mismatched (train on VB-DMD, test on WSJ0-CHiME3)"
            echo "  table3            - Table III (VB-DMD benchmark)"
            echo "  table4            - Table IV (dereverberation on WSJ0-REVERB)"
            echo "  table5            - Table V (real-world DNS Challenge data)"
            echo "  ablation          - Table I (sampler configurations)"
            echo "  all               - Run all experiments"
            exit 1
            ;;
    esac

    log "=========================================="
    log "Experiments completed!"
    log "Results saved to: $RESULTS_DIR"
    log "=========================================="
}

# Run main function
main "$@"