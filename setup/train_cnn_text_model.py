#!/usr/bin/env python3
"""
Train CNN-based text generation model for Jarvis AI Assistant.
"""

import sys
import os
import torch
from src.generative_ai_module.text_generator import create_cnn_text_generator

def train_cnn_text_model(gpu_type, vram_size, use_improved_preprocessor=False):
    """
    Create and train the CNN-enhanced text generator with optimized parameters for GPU.

    Args:
        gpu_type (str): Type of GPU (e.g., 'A6000')
        vram_size (int): VRAM size in GiB
        use_improved_preprocessor (bool): Whether to use the ImprovedPreprocessor
    """
    try:
        print('Starting CNN-based text generation model training...')

        # Ensure we're using GPU
        if not torch.cuda.is_available():
            print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
            sys.exit(1)

        # Clear CUDA cache before model creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✅ Cleared CUDA cache before model creation")

        # Create and train the CNN-enhanced text generator with multiple memory-efficient CNN layers
        model = create_cnn_text_generator(
            model_name='google/flan-ul2',  # Use the original model
            force_gpu=True,
            gpu_type=gpu_type,
            vram_size=int(vram_size),
            cnn_layers=3,  # Increased to 3 lightweight CNN layers for better learning
            cnn_kernel_sizes=[3, 5, 7],  # Use different kernel sizes for multi-scale feature extraction
            cnn_dropout=0.1,  # Keep dropout for regularization
            load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
            use_flash_attention_2=False,  # Disable Flash Attention
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
            lora_rank=8,  # Use moderate LoRA rank for better fine-tuning
            lora_alpha=16,  # Moderate LoRA alpha
            lora_dropout=0.1,  # Keep dropout for regularization
            max_length=128,  # Reduced sequence length to prevent OOM errors
            batch_size=1,  # Minimum batch size
            gradient_accumulation_steps=16,  # Use moderate gradient accumulation
            num_workers=0,  # No parallel workers to minimize memory usage
            warmup_ratio=0.03,  # Keep optimal warmup
            weight_decay=0.01,  # Keep weight decay for regularization
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0
        )

        # Set environment variables for extreme memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid deadlocks
        os.environ["FORCE_CPU_TOKENIZATION"] = "1"  # Force tokenization on CPU

        # Enable mixed precision training
        os.environ["MIXED_PRECISION_TRAINING"] = "1"

        # Print memory optimization message
        print("⚠️ Using memory-efficient CNN layers for improved learning")
        print("⚠️ Using original flan-ul2 model with 3 lightweight CNN layers")
        print("⚠️ Using multi-scale feature extraction with kernel sizes [3, 5, 7]")
        print("⚠️ Sequence length reduced to 128 tokens to prevent OOM errors")
        print("⚠️ Using 4-bit quantization and gradient checkpointing")
        print("⚠️ Using mixed precision training (FP16/BF16)")
        print("⚠️ Using grouped convolutions to reduce CNN parameters")
        print("⚠️ Using Adafactor optimizer for memory efficiency")
        print("⚠️ Clearing CUDA cache at strategic points")
        print("⚠️ Properly converting input batch types for datasets")
        print("⚠️ Progressive CNN layer fallback for OOM recovery")

        # Monitor GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"📊 Initial GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

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

        # Check if we should use the ImprovedPreprocessor with dataset-specific settings
        if use_improved_preprocessor:
            print('Using ImprovedPreprocessor with dataset-specific settings')
            from src.generative_ai_module.improved_preprocessing import ImprovedPreprocessor
            improved_processor = ImprovedPreprocessor()

            for dataset_name in datasets:
                preprocessed_path = os.path.join(datasets_dir, f'preprocessed_{dataset_name}.pt')
                preprocessed_paths[dataset_name] = preprocessed_path

                # Check if dataset is already preprocessed
                if os.path.exists(preprocessed_path):
                    print(f'Dataset {dataset_name} already preprocessed at {preprocessed_path}')
                    continue

                print(f'Preprocessing {dataset_name} with ImprovedPreprocessor...')
                try:
                    # Process dataset with dataset-specific settings
                    # writing_prompts will use special memory-optimized settings
                    data = improved_processor.process_dataset(dataset_name, max_samples=5000)

                    # Save preprocessed data
                    print(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
                    torch.save(data, preprocessed_path)
                    print(f'✓ Successfully preprocessed {dataset_name}')

                    # Log the dataset-specific parameters used
                    print(f'{dataset_name} parameters:')
                    print(f"  - Batch size: {data['params']['batch_size']}")
                    print(f"  - Sequence length: {data['params']['max_sequence_length']}")
                    print(f"  - Stride: {data['params']['stride']}")
                    print(f"  - Gradient accumulation steps: {data['params']['grad_accum_steps']}")
                    print(f"  - Number of batches: {len(data['batches'])}")

                except Exception as e:
                    print(f'❌ Error preprocessing {dataset_name}: {e}')
                    import traceback
                    traceback.print_exc()
        else:
            # Use the standard DatasetProcessor
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
        datasets_to_reprocess = []

        # First pass: Check which datasets exist
        for dataset_name, path in preprocessed_paths.items():
            if os.path.exists(path):
                try:
                    # Try to load the dataset to verify it has valid batches
                    data = torch.load(path)
                    if 'batches' in data and data['batches']:
                        # Validate batch tensor types
                        valid_batches = True
                        for batch in data['batches']:
                            if isinstance(batch, tuple) and len(batch) == 2:
                                input_batch, target_batch = batch
                                # Check if tensors have the correct dtype
                                if hasattr(input_batch, 'dtype') and input_batch.dtype != torch.long:
                                    print(f'Warning: Input batch in {dataset_name} has incorrect dtype: {input_batch.dtype}')
                                    # We'll fix this in train_from_multiple_datasets
                                if hasattr(target_batch, 'dtype') and target_batch.dtype != torch.long:
                                    print(f'Warning: Target batch in {dataset_name} has incorrect dtype: {target_batch.dtype}')
                                    # We'll fix this in train_from_multiple_datasets
                            else:
                                valid_batches = False
                                print(f'Warning: Invalid batch format in {dataset_name}')
                                break

                        if valid_batches:
                            available_datasets.append(dataset_name)
                            print(f'Dataset {dataset_name} has {len(data["batches"])} valid batches')
                        else:
                            print(f'Warning: Invalid batch format in {dataset_name}')
                            print(f'Will re-preprocess {dataset_name} dataset')
                            datasets_to_reprocess.append(dataset_name)
                    else:
                        print(f'Warning: No batches found in preprocessed data: {path}')
                        print(f'Will re-preprocess {dataset_name} dataset')
                        datasets_to_reprocess.append(dataset_name)
                except Exception as e:
                    print(f'Error loading dataset {dataset_name}: {e}')
                    print(f'Will re-preprocess {dataset_name} dataset')
                    datasets_to_reprocess.append(dataset_name)
            else:
                print(f'Dataset {dataset_name} not found at {path}, will preprocess it')
                datasets_to_reprocess.append(dataset_name)

        # Second pass: Re-preprocess datasets with missing or invalid batches
        for dataset_name in datasets_to_reprocess:
            print(f'Re-preprocessing {dataset_name} dataset...')
            try:
                # Check if we should use the ImprovedPreprocessor
                if use_improved_preprocessor:
                    print(f'Using ImprovedPreprocessor for {dataset_name}')
                    from src.generative_ai_module.improved_preprocessing import ImprovedPreprocessor
                    improved_processor = ImprovedPreprocessor()
                    data = improved_processor.process_dataset(dataset_name, max_samples=5000)
                else:
                    # Use standard preprocessing
                    print(f'Using standard preprocessing for {dataset_name}')
                    from src.generative_ai_module.dataset_processor import DatasetProcessor
                    processor = DatasetProcessor(model)

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
                    else:
                        print(f'Unknown dataset: {dataset_name}, skipping')
                        continue

                    # Create sequences and batches
                    print(f'Creating sequences with length {sequence_length}...')
                    sequences = processor.create_sequences(raw_text, sequence_length)

                    print(f'Creating batches with batch size {batch_size}...')
                    batches = processor.create_batches(sequences, batch_size=batch_size)

                    # Create dataset dictionary
                    data = {
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
                path = preprocessed_paths[dataset_name]
                print(f'Saving re-preprocessed {dataset_name} to {path}...')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(data, path)

                # Add to available datasets
                if 'batches' in data and data['batches']:
                    available_datasets.append(dataset_name)
                    print(f'Added {dataset_name} with {len(data["batches"])} batches to available datasets')
                else:
                    print(f'Warning: No valid batches in re-preprocessed {dataset_name}')
            except Exception as e:
                print(f'Error re-preprocessing {dataset_name}: {e}')
                import traceback
                traceback.print_exc()

        # Train with available datasets
        if len(available_datasets) > 1:
            print(f'Training with multiple datasets: {available_datasets}')
            model.train_from_multiple_datasets(
                dataset_names=available_datasets,
                epochs=3,
                dataset_paths=preprocessed_paths
            )
        elif len(available_datasets) == 1:
            # Train with the single available dataset
            dataset_name = available_datasets[0]
            print(f'Training with single dataset: {dataset_name}')
            model.train_from_preprocessed(
                dataset_name=dataset_name,
                epochs=3,
                preprocessed_path=preprocessed_paths.get(dataset_name)
            )
        else:
            # No valid datasets available, even after re-preprocessing
            raise ValueError("No valid datasets available for training, even after re-preprocessing")

        # Save the model
        output_dir = 'notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned'
        os.makedirs(output_dir, exist_ok=True)
        model.save_model(f'{output_dir}/model.pt')

        # Verify saved model
        if os.path.exists(f'{output_dir}/model.pt'):
            print(f'✓ Model successfully saved to {output_dir}/model.pt')
        else:
            print(f'❌ ERROR: Failed to save model to {output_dir}/model.pt')
            sys.exit(1)

        print('✓ CNN-enhanced text generation model training completed successfully')
        return True

    except Exception as e:
        print(f'❌ ERROR during CNN text model training: {e}')

        # Try to save partial results
        try:
            backup_dir = f'notebooks/Jarvis_AI_Assistant/models/backup_cnn_model_{int(torch.cuda.current_device())}'
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
        use_improved_preprocessor = len(sys.argv) >= 4 and sys.argv[3] == "1"
    else:
        # Default values
        gpu_type = "A6000"
        vram_size = 50
        use_improved_preprocessor = False

    train_cnn_text_model(gpu_type, vram_size, use_improved_preprocessor)
