#!/usr/bin/env python3
"""
Train custom encoder-decoder model for Jarvis AI Assistant.
"""

import sys
import os
import torch
import logging
import argparse
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class CustomEncoderDecoder(torch.nn.Module):
    """
    Custom encoder-decoder model that leverages the fine-tuned CNN-enhanced FLAN-UL2 model.

    This model uses the CNN-enhanced FLAN-UL2 model as a feature extractor and adds
    custom encoder-decoder layers on top.
    """

    def __init__(
        self,
        cnn_model_path: str,
        hidden_size: int = 768,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        force_gpu: bool = True
    ):
        super().__init__()

        # Load the CNN-enhanced FLAN-UL2 model
        logger.info(f'Loading CNN-enhanced FLAN-UL2 model from {cnn_model_path}')
        self.cnn_model = self._load_cnn_model(cnn_model_path)

        # Freeze the CNN model parameters
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # Get the hidden size from the CNN model
        self.feature_size = self.cnn_model.hidden_size

        # Create encoder layers
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.feature_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        # Create decoder layers
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.feature_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )

        # Output projection
        self.output_projection = torch.nn.Linear(self.feature_size, self.cnn_model.tokenizer.vocab_size)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and force_gpu else 'cpu')
        self.to(self.device)

        logger.info(f'Initialized custom encoder-decoder model on {self.device}')

    def _load_cnn_model(self, model_path: str):
        """Load the CNN-enhanced FLAN-UL2 model"""
        # Import required module
        from src.generative_ai_module.text_generator import CNNTextGenerator

        # Check if the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model path {model_path} does not exist')

        # Load the model
        model_dir = os.path.dirname(model_path)
        tokenizer_path = os.path.join(model_dir, 'tokenizer')

        # Create a new CNN model with enhanced attention mechanisms
        cnn_model = CNNTextGenerator(
            model_name_or_path='google/flan-ul2',
            force_gpu=True,
            cnn_layers=3,
            load_in_4bit=True,
            use_flash_attention_2=True,  # This will trigger our enhanced attention mechanisms for T5/FLAN models
            lora_dropout=0.1,  # Increased dropout for better regularization
            warmup_ratio=0.05  # Increased warmup for better convergence
        )

        # Load the saved model
        state_dict = torch.load(model_path, map_location='cpu')
        cnn_model.base_model.load_state_dict(state_dict['base_model'])

        # Load CNN layers
        for i, layer_state in enumerate(state_dict['cnn_layers']):
            if i < len(cnn_model.cnn_layers_list):
                cnn_model.cnn_layers_list[i].load_state_dict(layer_state)

        # Load adapter
        cnn_model.adapter.load_state_dict(state_dict['adapter'])

        return cnn_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input [batch_size, seq_len]
            decoder_input_ids: Decoder input token IDs [batch_size, target_len]
            decoder_attention_mask: Attention mask for decoder [batch_size, target_len]

        Returns:
            Logits for next token prediction [batch_size, target_len, vocab_size]
        """
        # Get embeddings from CNN model
        with torch.no_grad():
            # Get embeddings from base model
            if hasattr(self.cnn_model.base_model, 'transformer') and hasattr(self.cnn_model.base_model.transformer, 'wte'):
                # GPT-2 style models
                embeddings = self.cnn_model.base_model.transformer.wte(input_ids)
            elif hasattr(self.cnn_model.base_model, 'get_input_embeddings'):
                # Generic approach for most models
                embedding_layer = self.cnn_model.base_model.get_input_embeddings()
                embeddings = embedding_layer(input_ids)
            else:
                raise ValueError('Could not get embeddings from model')

            # Apply CNN layers for feature extraction
            # First, transpose for CNN (batch_size, hidden_size, seq_len)
            x = embeddings.transpose(1, 2)

            # Pass through each CNN layer
            for cnn_layer in self.cnn_model.cnn_layers_list:
                x = cnn_layer(x)

            # Transpose back to transformer format (batch_size, seq_len, hidden_size)
            x = x.transpose(1, 2)

            # Apply adapter to ensure compatibility with transformer
            enhanced_embeddings = self.cnn_model.adapter(x)

            # Add residual connection to preserve original embeddings
            enhanced_embeddings = enhanced_embeddings + embeddings

        # Pass through encoder
        encoder_output = self.encoder(enhanced_embeddings, src_key_padding_mask=~attention_mask.bool())

        # Get decoder embeddings
        if hasattr(self.cnn_model.base_model, 'transformer') and hasattr(self.cnn_model.base_model.transformer, 'wte'):
            # GPT-2 style models
            decoder_embeddings = self.cnn_model.base_model.transformer.wte(decoder_input_ids)
        elif hasattr(self.cnn_model.base_model, 'get_input_embeddings'):
            # Generic approach for most models
            embedding_layer = self.cnn_model.base_model.get_input_embeddings()
            decoder_embeddings = embedding_layer(decoder_input_ids)
        else:
            raise ValueError('Could not get embeddings from model')

        # Pass through decoder
        decoder_output = self.decoder(
            decoder_embeddings,
            encoder_output,
            tgt_key_padding_mask=~decoder_attention_mask.bool()
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

def train_custom_model(cnn_model_path, output_dir, epochs=3, batch_size=4, max_samples=5000,
                      learning_rate=5e-5, weight_decay=0.01, hidden_size=768,
                      num_encoder_layers=3, num_decoder_layers=3, dropout=0.1,
                      log_every=10, force_gpu=True, save_checkpoints=True,
                      use_improved_preprocessor=False):
    """Train the custom encoder-decoder model"""
    try:
        # Import required modules
        from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler

        # Initialize the dataset handler
        dataset_handler = UnifiedDatasetHandler()

        # Load all preprocessed datasets
        datasets_dir = 'notebooks/Jarvis_AI_Assistant/datasets'
        os.makedirs(datasets_dir, exist_ok=True)
        dataset_names = ['writing_prompts', 'persona_chat', 'pile', 'openassistant', 'gpteacher']
        preprocessed_paths = {name: os.path.join(datasets_dir, f'preprocessed_{name}.pt') for name in dataset_names}

        # Check which datasets are available
        available_datasets = []
        for dataset_name, path in preprocessed_paths.items():
            if os.path.exists(path):
                available_datasets.append(dataset_name)

        if not available_datasets:
            logger.warning("No preprocessed datasets found. Running preprocessing...")

            # Check if we should use the ImprovedPreprocessor with dataset-specific settings
            if use_improved_preprocessor:
                logger.info('Using ImprovedPreprocessor with dataset-specific settings')
                from src.generative_ai_module.improved_preprocessing import ImprovedPreprocessor

                # Create improved processor for preprocessing
                improved_processor = ImprovedPreprocessor()

                # Preprocess all datasets
                logger.info('Preprocessing datasets with ImprovedPreprocessor...')
                for dataset_name in dataset_names:
                    preprocessed_path = preprocessed_paths[dataset_name]

                    logger.info(f'Preprocessing {dataset_name} with ImprovedPreprocessor...')
                    try:
                        # Process dataset with dataset-specific settings
                        # writing_prompts will use special memory-optimized settings
                        data = improved_processor.process_dataset(dataset_name, max_samples=max_samples // 5)

                        # Save preprocessed data
                        logger.info(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
                        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
                        torch.save(data, preprocessed_path)
                        logger.info(f'Successfully preprocessed {dataset_name}')

                        # Log the dataset-specific parameters used
                        logger.info(f'{dataset_name} parameters:')
                        logger.info(f"  - Batch size: {data['params']['batch_size']}")
                        logger.info(f"  - Sequence length: {data['params']['max_sequence_length']}")
                        logger.info(f"  - Stride: {data['params']['stride']}")
                        logger.info(f"  - Gradient accumulation steps: {data['params']['grad_accum_steps']}")
                        logger.info(f"  - Number of batches: {len(data['batches'])}")

                        # Add to available datasets
                        available_datasets.append(dataset_name)

                    except Exception as e:
                        logger.error(f'Error preprocessing {dataset_name}: {e}')
                        import traceback
                        logger.error(traceback.format_exc())
            else:
                # Import standard dataset processor
                from src.generative_ai_module.dataset_processor import DatasetProcessor

                # Create processor for preprocessing
                processor = DatasetProcessor()

                # Preprocess all datasets
                logger.info('Preprocessing datasets...')
                for dataset_name in dataset_names:
                    preprocessed_path = preprocessed_paths[dataset_name]

                    logger.info(f'Preprocessing {dataset_name}...')
                    try:
                        # Prepare dataset with appropriate parameters
                        if dataset_name == 'persona_chat':
                            sequence_length = 512
                            batch_size = 16
                            raw_text = processor.load_persona_chat(split='train', max_samples=max_samples // 5)
                        elif dataset_name == 'writing_prompts':
                            sequence_length = 1024
                            batch_size = 8
                            raw_text = processor.load_writing_prompts(split='train', max_samples=max_samples // 5)
                        elif dataset_name == 'pile':
                            sequence_length = 1024
                            batch_size = 8
                            raw_text = processor.load_pile_dataset(split='train', max_samples=max_samples // 5)
                        elif dataset_name == 'openassistant':
                            sequence_length = 512
                            batch_size = 16
                            raw_text = processor.load_openassistant_dataset(split='train', max_samples=max_samples // 5)
                        elif dataset_name == 'gpteacher':
                            sequence_length = 768
                            batch_size = 12
                            raw_text = processor.load_gpteacher_dataset(split='train', max_samples=max_samples // 5)

                        # Create sequences and batches
                        logger.info(f'Creating sequences with length {sequence_length}...')
                        sequences = processor.create_sequences(raw_text, sequence_length)

                        logger.info(f'Creating batches with batch size {batch_size}...')
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
                        logger.info(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
                        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
                        torch.save(dataset, preprocessed_path)
                        logger.info(f'Successfully preprocessed {dataset_name}')

                        # Add to available datasets
                        available_datasets.append(dataset_name)

                    except Exception as e:
                        logger.error(f'Error preprocessing {dataset_name}: {e}')
                        import traceback
                        logger.error(traceback.format_exc())

        # Load the preprocessed datasets
        datasets = []
        datasets_to_reprocess = []

        # First pass: Try to load preprocessed datasets
        for dataset_name in available_datasets:
            logger.info(f'Loading preprocessed dataset: {dataset_name}')
            try:
                dataset = torch.load(preprocessed_paths[dataset_name])
                if 'batches' in dataset and dataset['batches']:
                    datasets.append(dataset)
                    logger.info(f'Successfully loaded {len(dataset["batches"])} batches from {dataset_name}')
                else:
                    logger.warning(f'Warning: No batches found in preprocessed data: {preprocessed_paths[dataset_name]}')
                    logger.info(f'Will re-preprocess {dataset_name} dataset')
                    datasets_to_reprocess.append(dataset_name)
            except Exception as e:
                logger.warning(f'Failed to load dataset {dataset_name}: {e}')
                logger.info(f'Will re-preprocess {dataset_name} dataset')
                datasets_to_reprocess.append(dataset_name)

        # Second pass: Re-preprocess datasets with missing or invalid batches
        for dataset_name in datasets_to_reprocess:
            logger.info(f'Re-preprocessing {dataset_name} dataset...')
            try:
                # Check if we should use the ImprovedPreprocessor
                if use_improved_preprocessor:
                    logger.info(f'Using ImprovedPreprocessor for {dataset_name}')
                    from src.generative_ai_module.improved_preprocessing import ImprovedPreprocessor
                    improved_processor = ImprovedPreprocessor()
                    data = improved_processor.process_dataset(dataset_name, max_samples=max_samples // 5)
                else:
                    # Use standard preprocessing
                    logger.info(f'Using standard preprocessing for {dataset_name}')
                    from src.generative_ai_module.dataset_processor import DatasetProcessor
                    processor = DatasetProcessor()

                    # Prepare dataset with appropriate parameters
                    if dataset_name == 'persona_chat':
                        sequence_length = 512
                        batch_size = 16
                        raw_text = processor.load_persona_chat(split='train', max_samples=max_samples // 5)
                    elif dataset_name == 'writing_prompts':
                        sequence_length = 1024
                        batch_size = 8
                        raw_text = processor.load_writing_prompts(split='train', max_samples=max_samples // 5)
                    elif dataset_name == 'pile':
                        sequence_length = 1024
                        batch_size = 8
                        raw_text = processor.load_pile_dataset(split='train', max_samples=max_samples // 5)
                    elif dataset_name == 'openassistant':
                        sequence_length = 512
                        batch_size = 16
                        raw_text = processor.load_openassistant_dataset(split='train', max_samples=max_samples // 5)
                    elif dataset_name == 'gpteacher':
                        sequence_length = 768
                        batch_size = 12
                        raw_text = processor.load_gpteacher_dataset(split='train', max_samples=max_samples // 5)
                    else:
                        logger.warning(f'Unknown dataset: {dataset_name}, skipping')
                        continue

                    # Create sequences and batches
                    logger.info(f'Creating sequences with length {sequence_length}...')
                    sequences = processor.create_sequences(raw_text, sequence_length)

                    logger.info(f'Creating batches with batch size {batch_size}...')
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
                preprocessed_path = preprocessed_paths[dataset_name]
                logger.info(f'Saving re-preprocessed {dataset_name} to {preprocessed_path}...')
                os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
                torch.save(data, preprocessed_path)

                # Add to datasets list
                if 'batches' in data and data['batches']:
                    datasets.append(data)
                    logger.info(f'Added {len(data["batches"])} batches from re-preprocessed {dataset_name}')
                else:
                    logger.warning(f'Warning: No valid batches in re-preprocessed {dataset_name}')
            except Exception as e:
                logger.error(f'Error re-preprocessing {dataset_name}: {e}')
                import traceback
                logger.error(traceback.format_exc())

        # Combine datasets
        combined_batches = []
        for dataset in datasets:
            combined_batches.extend(dataset.get('batches', []))

        logger.info(f'Combined {len(combined_batches)} batches from all datasets')

        if not combined_batches:
            raise ValueError("No valid batches found in any of the datasets, even after re-preprocessing")

        # Initialize the custom model
        model = CustomEncoderDecoder(
            cnn_model_path=cnn_model_path,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            force_gpu=force_gpu
        )

        # Set up optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Set up loss function
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Training loop
        logger.info(f'Starting training for {epochs} epochs')
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            # Track dataset-specific parameters
            current_dataset = None
            prev_dataset = None
            grad_accum_steps = 1  # Default
            step_count = 0

            # Process batches
            for i, (input_batch, target_batch) in enumerate(combined_batches):
                # Skip invalid batches
                if input_batch is None or target_batch is None:
                    continue

                # Determine which dataset this batch is from
                # This is a heuristic based on batch size
                batch_size = input_batch.shape[0]
                if batch_size <= 2:
                    # Likely writing_prompts with special settings
                    current_dataset = "writing_prompts"
                    grad_accum_steps = 4  # Use writing_prompts grad accumulation
                    use_mixed_precision = True
                else:
                    # Other datasets with default settings
                    current_dataset = "other"
                    grad_accum_steps = 1
                    use_mixed_precision = False

                # Log dataset change
                if i == 0 or (i > 0 and current_dataset != prev_dataset):
                    logger.info(f"Processing batch from {current_dataset} dataset")
                    logger.info(f"  - Batch size: {batch_size}")
                    logger.info(f"  - Gradient accumulation steps: {grad_accum_steps}")
                    prev_dataset = current_dataset

                # Move to device
                input_batch = input_batch.to(model.device)
                target_batch = target_batch.to(model.device)

                # Create attention masks
                input_attention_mask = (input_batch != model.cnn_model.tokenizer.pad_token_id).float()
                target_attention_mask = (target_batch != model.cnn_model.tokenizer.pad_token_id).float()

                # Create decoder input IDs (shift right)
                decoder_input_ids = torch.zeros_like(target_batch)
                decoder_input_ids[:, 1:] = target_batch[:, :-1]
                decoder_input_ids[:, 0] = model.cnn_model.tokenizer.bos_token_id if model.cnn_model.tokenizer.bos_token_id is not None else model.cnn_model.tokenizer.pad_token_id

                # Use mixed precision for writing_prompts dataset
                if use_mixed_precision and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Forward pass
                        logits = model(
                            input_ids=input_batch,
                            attention_mask=input_attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=target_attention_mask
                        )

                        # Calculate loss
                        loss = loss_fn(logits.view(-1, logits.size(-1)), target_batch.view(-1))

                        # Scale loss for gradient accumulation
                        if grad_accum_steps > 1:
                            loss = loss / grad_accum_steps
                else:
                    # Forward pass
                    logits = model(
                        input_ids=input_batch,
                        attention_mask=input_attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=target_attention_mask
                    )

                    # Calculate loss
                    loss = loss_fn(logits.view(-1, logits.size(-1)), target_batch.view(-1))

                    # Scale loss for gradient accumulation
                    if grad_accum_steps > 1:
                        loss = loss / grad_accum_steps

                # Backward pass
                loss.backward()

                # Update step counter
                step_count += 1

                # Step optimizer based on gradient accumulation
                if grad_accum_steps == 1 or step_count % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Update total loss (scale back for reporting)
                total_loss += loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)

                # Log progress
                if (i + 1) % log_every == 0:
                    logger.info(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(combined_batches)}, Loss: {loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1):.4f}')

                # Memory cleanup for writing_prompts dataset
                if current_dataset == "writing_prompts" and torch.cuda.is_available():
                    del logits
                    torch.cuda.empty_cache()

            # Log epoch results
            avg_loss = total_loss / len(combined_batches)
            logger.info(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # Save checkpoint
            if save_checkpoints:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, checkpoint_path)
                logger.info(f'Saved checkpoint to {checkpoint_path}')

        # Save final model
        final_model_path = os.path.join(output_dir, 'custom_encoder_decoder.pt')
        torch.save(model.state_dict(), final_model_path)
        logger.info(f'Saved final model to {final_model_path}')

        return model

    except Exception as e:
        logger.error(f'Error during custom model training: {e}')
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train custom encoder-decoder model')
    parser.add_argument('--cnn-model-path', type=str, required=True, help='Path to CNN model')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=5000, help='Maximum number of samples')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--hidden-size', type=int, default=768, help='Hidden size')
    parser.add_argument('--num-encoder-layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--log-every', type=int, default=10, help='Log every N batches')
    parser.add_argument('--use-improved-preprocessor', action='store_true', help='Use improved preprocessor')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Train the model
        model = train_custom_model(
            cnn_model_path=args.cnn_model_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_size=args.hidden_size,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            log_every=args.log_every,
            force_gpu=True,
            save_checkpoints=True,
            use_improved_preprocessor=args.use_improved_preprocessor
        )

        # Verify saved model
        final_model_path = os.path.join(args.output_dir, 'custom_encoder_decoder.pt')
        if os.path.exists(final_model_path):
            print(f'✓ Model successfully saved to {final_model_path}')
        else:
            print(f'❌ ERROR: Failed to save model to {final_model_path}')
            sys.exit(1)

        print('✓ Custom encoder-decoder model training completed successfully')

    except Exception as e:
        print(f'❌ ERROR during custom model training: {e}')

        # Try to save partial results
        try:
            backup_dir = f'notebooks/Jarvis_AI_Assistant/models/backup_custom_model_{int(torch.cuda.current_device())}'
            os.makedirs(backup_dir, exist_ok=True)

            if 'model' in locals():
                torch.save(model.state_dict(), f'{backup_dir}/partial_model.pt')
                print(f'✓ Partial model saved to {backup_dir}/partial_model.pt')
        except Exception as save_error:
            print(f'❌ Failed to save partial results: {save_error}')

        sys.exit(1)
