#!/bin/bash
# Script to train a custom encoder-decoder model using the fine-tuned CNN model

# Set default values
CNN_MODEL_PATH="/notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned/model.pt"
OUTPUT_DIR="/notebooks/Jarvis_AI_Assistant/models/custom-encoder-decoder"
EPOCHS=3
BATCH_SIZE=4
MAX_SAMPLES=5000
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
HIDDEN_SIZE=768
NUM_ENCODER_LAYERS=3
NUM_DECODER_LAYERS=3
DROPOUT=0.1
FORCE_GPU=true
SAVE_CHECKPOINTS=true
LOG_EVERY=10

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --help                     Show this help message"
    echo "  --cnn-model-path PATH      Path to the fine-tuned CNN model (default: $CNN_MODEL_PATH)"
    echo "  --output-dir DIR           Directory to save the trained model (default: $OUTPUT_DIR)"
    echo "  --epochs N                 Number of training epochs (default: $EPOCHS)"
    echo "  --batch-size N             Batch size for training (default: $BATCH_SIZE)"
    echo "  --max-samples N            Maximum number of samples to use from each dataset (default: $MAX_SAMPLES)"
    echo "  --learning-rate RATE       Learning rate for training (default: $LEARNING_RATE)"
    echo "  --weight-decay RATE        Weight decay for training (default: $WEIGHT_DECAY)"
    echo "  --hidden-size N            Hidden size for the encoder-decoder model (default: $HIDDEN_SIZE)"
    echo "  --num-encoder-layers N     Number of encoder layers (default: $NUM_ENCODER_LAYERS)"
    echo "  --num-decoder-layers N     Number of decoder layers (default: $NUM_DECODER_LAYERS)"
    echo "  --dropout RATE             Dropout rate (default: $DROPOUT)"
    echo "  --no-force-gpu             Don't force GPU usage"
    echo "  --no-save-checkpoints      Don't save checkpoints during training"
    echo "  --log-every N              Log every N batches (default: $LOG_EVERY)"
    echo ""
    echo "Example: $0 --cnn-model-path /path/to/model.pt --output-dir /path/to/output --epochs 5"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --cnn-model-path)
            CNN_MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num-encoder-layers)
            NUM_ENCODER_LAYERS="$2"
            shift 2
            ;;
        --num-decoder-layers)
            NUM_DECODER_LAYERS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --no-force-gpu)
            FORCE_GPU=false
            shift
            ;;
        --no-save-checkpoints)
            SAVE_CHECKPOINTS=false
            shift
            ;;
        --log-every)
            LOG_EVERY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print banner
echo "===== Custom Encoder-Decoder Model Training ====="
echo "CNN Model Path: $CNN_MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Samples: $MAX_SAMPLES"
echo "Learning Rate: $LEARNING_RATE"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Hidden Size: $HIDDEN_SIZE"
echo "Encoder Layers: $NUM_ENCODER_LAYERS"
echo "Decoder Layers: $NUM_DECODER_LAYERS"
echo "Dropout: $DROPOUT"
echo "Force GPU: $FORCE_GPU"
echo "Save Checkpoints: $SAVE_CHECKPOINTS"
echo "Log Every: $LOG_EVERY"
echo "=============================================="

# Check if CNN model exists
if [ ! -f "$CNN_MODEL_PATH" ]; then
    echo "❌ ERROR: CNN model not found at $CNN_MODEL_PATH"
    echo "Please run train_jarvis.sh with --model-type cnn-text first"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set up force GPU flag
FORCE_GPU_FLAG=""
if [ "$FORCE_GPU" = true ]; then
    FORCE_GPU_FLAG="--force-gpu"
fi

# Set up save checkpoints flag
SAVE_CHECKPOINTS_FLAG=""
if [ "$SAVE_CHECKPOINTS" = true ]; then
    SAVE_CHECKPOINTS_FLAG="--save-checkpoints"
fi

# Run the training script
echo "Starting training..."
python -m src.generative_ai_module.train_with_cnn_model \
    --cnn-model-path "$CNN_MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$MAX_SAMPLES" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-encoder-layers "$NUM_ENCODER_LAYERS" \
    --num-decoder-layers "$NUM_DECODER_LAYERS" \
    --dropout "$DROPOUT" \
    --log-every "$LOG_EVERY" \
    $FORCE_GPU_FLAG \
    $SAVE_CHECKPOINTS_FLAG

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "❌ Training failed. See logs for details."
    exit 1
else
    echo "✓ Training completed successfully!"
    echo "Model saved to $OUTPUT_DIR/custom_encoder_decoder.pt"
fi

# Verify that the model was saved
if [ -f "$OUTPUT_DIR/custom_encoder_decoder.pt" ]; then
    echo "✓ Model file exists at $OUTPUT_DIR/custom_encoder_decoder.pt"
else
    echo "❌ WARNING: Model file not found at $OUTPUT_DIR/custom_encoder_decoder.pt"
fi

echo "✓ Done"
