#!/usr/bin/env python3
"""
Train text generation model for Jarvis AI Assistant.
"""

import sys
import os
import torch
from src.generative_ai_module.text_generator import create_cnn_text_generator

def train_text_model(gpu_type, vram_size):
    """
    Create and train the text generator with optimized parameters for GPU.
    
    Args:
        gpu_type (str): Type of GPU (e.g., 'A6000')
        vram_size (int): VRAM size in GiB
    """
    try:
        print('Starting text generation model training...')

        # Ensure we're using GPU
        if not torch.cuda.is_available():
            print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
            sys.exit(1)

        # Create and train the text generator with optimized parameters for GPU
        model = create_cnn_text_generator(
            model_name='google/flan-ul2',
            force_gpu=True,
            gpu_type=gpu_type,
            vram_size=int(vram_size),
            load_in_4bit=True,
            use_flash_attention_2=True,
            gradient_checkpointing=True,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.05,
            max_length=2048,  # Reduced from 4096 to ensure stability with FLAN-UL2
            batch_size=3,  # Explicitly set for stability
            gradient_accumulation_steps=8,  # Explicitly set for stability
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with CUDA
            warmup_ratio=0.03,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0
        )

        # Verify model is on GPU
        if not next(model.parameters()).is_cuda:
            print('❌ WARNING: Model is not on GPU. Moving model to GPU...')
            model = model.cuda()

        # Ensure datasets directory exists
        datasets_dir = 'notebooks/Jarvis_AI_Assistant/datasets'
        os.makedirs(datasets_dir, exist_ok=True)

        # Import dataset processor
        from src.generative_ai_module.dataset_processor import DatasetProcessor

        # Create processor
        processor = DatasetProcessor(model)

        # Define datasets to preprocess
        datasets = ['persona_chat', 'writing_prompts', 'pile', 'openassistant', 'gpteacher']
        preprocessed_paths = {}

        # Preprocess all datasets
        print('Preprocessing datasets...')
        for dataset_name in datasets:
            preprocessed_path = os.path.join(datasets_dir, f'preprocessed_{dataset_name}.pt')
            preprocessed_paths[dataset_name] = preprocessed_path

            # Check if dataset is already preprocessed
            if os.path.exists(preprocessed_path):
                print(f'Dataset {dataset_name} already preprocessed at {preprocessed_path}')
                continue

            print(f'Preprocessing {dataset_name}...')
            try:
                # Prepare dataset with appropriate parameters
                if dataset_name == 'persona_chat':
                    sequence_length = 512
                    batch_size = 16
                    raw_text = processor.load_persona_chat(split='train', max_samples=5000)
                elif dataset_name == 'writing_prompts':
                    sequence_length = 1024
                    batch_size = 8
                    raw_text = processor.load_writing_prompts(split='train', max_samples=5000)
                elif dataset_name == 'pile':
                    sequence_length = 1024
                    batch_size = 8
                    raw_text = processor.load_pile_dataset(split='train', max_samples=5000)
                elif dataset_name == 'openassistant':
                    sequence_length = 512
                    batch_size = 16
                    raw_text = processor.load_openassistant_dataset(split='train', max_samples=5000)
                elif dataset_name == 'gpteacher':
                    sequence_length = 768
                    batch_size = 12
                    raw_text = processor.load_gpteacher_dataset(split='train', max_samples=5000)

                # Create sequences and batches
                print(f'Creating sequences with length {sequence_length}...')
                sequences = processor.create_sequences(raw_text, sequence_length)

                print(f'Creating batches with batch size {batch_size}...')
                batches = processor.create_batches(sequences, batch_size=batch_size)

                # Create dataset dictionary
                dataset = {
                    'batches': batches,
                    'metadata': {
                        'dataset_name': dataset_name,
                        'split': 'train',
                        'sequence_length': sequence_length,
                        'batch_size': batch_size,
                        'sample_count': len(sequences),
                        'batch_count': len(batches)
                    }
                }

                # Save preprocessed data
                print(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
                torch.save(dataset, preprocessed_path)
                print(f'✓ Successfully preprocessed {dataset_name}')

            except Exception as e:
                print(f'❌ Error preprocessing {dataset_name}: {e}')
                import traceback
                traceback.print_exc()

        # Train the model with all available datasets
        print('Starting training...')

        # Check which datasets were successfully preprocessed
        available_datasets = []
        for dataset_name, path in preprocessed_paths.items():
            if os.path.exists(path):
                available_datasets.append(dataset_name)

        if len(available_datasets) > 1:
            print(f'Training with multiple datasets: {available_datasets}')
            model.train_from_multiple_datasets(
                dataset_names=available_datasets,
                epochs=3,
                dataset_paths=preprocessed_paths
            )
        else:
            # Fall back to single dataset training
            print(f'Training with single dataset: persona_chat')
            model.train_from_preprocessed(
                dataset_name='persona_chat',
                epochs=3,
                preprocessed_path=preprocessed_paths.get('persona_chat')
            )

        # Save the model
        output_dir = 'notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned'
        os.makedirs(output_dir, exist_ok=True)
        model.save_model(f'{output_dir}/model.pt')

        # Verify saved model
        if os.path.exists(f'{output_dir}/model.pt'):
            print(f'✓ Model successfully saved to {output_dir}/model.pt')
        else:
            print(f'❌ ERROR: Failed to save model to {output_dir}/model.pt')
            sys.exit(1)

        print('✓ Text generation model training completed successfully')
        return True

    except Exception as e:
        print(f'❌ ERROR during text model training: {e}')

        # Try to save partial results
        try:
            backup_dir = f'notebooks/Jarvis_AI_Assistant/models/backup_text_model_{int(torch.cuda.current_device())}'
            os.makedirs(backup_dir, exist_ok=True)

            if 'model' in locals():
                model.save_model(f'{backup_dir}/partial_model.pt')
                print(f'✓ Partial model saved to {backup_dir}/partial_model.pt')
        except Exception as save_error:
            print(f'❌ Failed to save partial results: {save_error}')

        sys.exit(1)

if __name__ == "__main__":
    # Get GPU type and VRAM size from command line arguments
    if len(sys.argv) >= 3:
        gpu_type = sys.argv[1]
        vram_size = sys.argv[2]
    else:
        # Default values
        gpu_type = "A6000"
        vram_size = 50
    
    train_text_model(gpu_type, vram_size)
