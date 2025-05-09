# Jarvis AI Assistant

A comprehensive AI assistant framework featuring multiple model architectures, training pipelines, and dataset processing capabilities. This project includes code generation, text generation, and CNN-enhanced text generation models.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Models](#models)
- [Datasets](#datasets)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Original Contributors](#original-contributors)

## Overview

Jarvis AI Assistant is a modular framework designed to train and deploy various AI models for text generation, code generation, and custom encoder-decoder tasks. The system is optimized for different GPU environments, particularly Paperspace notebooks with A6000 GPUs.

Key features:

- Multiple model architectures (DeepSeek, FLAN-UL2, CNN-enhanced models)
- Comprehensive dataset processing for various data sources
- Memory-efficient training with quantization and optimization
- Progressive fallback mechanisms for OOM recovery
- Multi-scale feature extraction with CNN layers

## Architecture

The project is organized into several key components:

1. **Core Modules**:

   - `text_generator.py`: Implements text generation models including CNN-enhanced variants
   - `code_generator.py`: Implements code generation models based on DeepSeek
   - `dataset_processor.py`: Handles dataset loading, preprocessing, and batching

2. **Training Scripts**:

   - `train_jarvis.sh`: Main entry point for training all model types
   - `train_text_model.py`: Trains text generation models
   - `train_code_model.py`: Trains code generation models
   - `train_cnn_text_model.py`: Trains CNN-enhanced text models
   - `train_custom_model.py`: Trains custom encoder-decoder models

3. **Setup Scripts**:
   - `consolidated_unified_setup.sh`: Main setup script for environment configuration
   - Various fix scripts for handling specific issues

## Models

### Text Generation Models

1. **Base Text Generator**

   - Based on FLAN-UL2 model
   - Supports standard text generation tasks
   - Optimized for memory efficiency

2. **CNN-Enhanced Text Generator**

   - Extends the base text generator with CNN layers
   - Uses multi-scale feature extraction with different kernel sizes (3, 5, 7)
   - Implements grouped convolutions for parameter efficiency
   - Features progressive fallback for OOM recovery

3. **Custom Encoder-Decoder Model**
   - Built on top of the CNN-enhanced model
   - Implements a custom encoder-decoder architecture
   - Uses the CNN model as a feature extractor

### Code Generation Models

1. **DeepSeek Coder**
   - Based on DeepSeek-Coder-6.7B model
   - Fine-tuned on code datasets
   - Optimized for memory efficiency with 4-bit quantization
   - Supports Unsloth optimization when available

## Datasets

The framework supports multiple datasets for training:

1. **Text Generation Datasets**:

   - **Persona Chat**: Conversational dataset with persona information
   - **Writing Prompts**: Creative writing prompts and responses
   - **OpenAssistant**: Instruction-following dataset
   - **GPTeacher**: Instruction-tuning dataset
   - **Pile**: Large-scale text dataset with various subsets

2. **Code Generation Datasets**:
   - **CodeSearchNet**: Code snippets in multiple programming languages
   - Custom code datasets

Each dataset has specialized preprocessing pipelines to handle its unique structure and requirements.

## Setup and Installation

### Prerequisites

- CUDA-compatible GPU (optimized for A6000, A4000, RTX5000)
- Python 3.8+
- PyTorch 2.0+

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/noah-mclain/Jarvis-AI-Assistant.git
   cd Jarvis-AI-Assistant
   ```

2. Run the unified setup script:

   ```bash
   ./setup/consolidated_unified_setup.sh
   ```

This script will:

- Set up the Python environment
- Install all dependencies
- Configure GPU optimizations
- Set up minimal Unsloth implementation
- Apply necessary fixes for model compatibility

## Training

### Training a Text Generation Model

```bash
./setup/train_jarvis.sh --gpu-type A6000 --vram 48 --model-type text
```

### Training a CNN-Enhanced Text Model

```bash
./setup/train_jarvis.sh --gpu-type A6000 --vram 48 --model-type cnn-text
```

### Training a Code Generation Model

```bash
./setup/train_jarvis.sh --gpu-type A6000 --vram 48 --model-type code
```

### Training a Custom Encoder-Decoder Model

```bash
./setup/train_jarvis.sh --gpu-type A6000 --vram 48 --model-type custom-model
```

## Advanced Features

### Memory Optimization

The framework implements several memory optimization techniques:

1. **Quantization**: 4-bit and 8-bit quantization for model weights
2. **Gradient Checkpointing**: Reduces memory usage during backpropagation
3. **Progressive Fallback**: Automatically reduces CNN layers if OOM errors occur
4. **CUDA Cache Clearing**: Strategic cache clearing during training
5. **Grouped Convolutions**: Reduces parameter count in CNN layers

### Multi-Scale Feature Extraction

The CNN-enhanced models use multiple kernel sizes (3, 5, 7) to capture patterns at different scales, similar to how CNNs work in computer vision.

### Adaptive Training

The training process automatically adapts to the available hardware:

- Adjusts batch size based on GPU type and VRAM
- Configures gradient accumulation steps accordingly
- Implements progressive fallback for OOM recovery

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**

   - Reduce batch size or sequence length
   - Enable 4-bit quantization
   - Use gradient checkpointing
   - Reduce the number of CNN layers

2. **Training Instability**

   - Reduce learning rate
   - Increase gradient accumulation steps
   - Use mixed precision training

3. **Slow Training**
   - Enable Flash Attention 2 if supported
   - Use Unsloth optimization for DeepSeek models
   - Optimize dataset preprocessing

For more detailed troubleshooting, refer to the logs in the `logs` directory.

## Original Contributors

1. **Ahmed** as NLP builds **bag of words with neural network**.
2. **Hamza** as Speech builds **GRU/CTC with MFCC features**.
3. **Nada** as Generative text builds **character level LSTM**.
4. **Amr** as Generative image builds **simple GAN**

## License

This project is licensed under the MIT License - see the LICENSE file for details.
