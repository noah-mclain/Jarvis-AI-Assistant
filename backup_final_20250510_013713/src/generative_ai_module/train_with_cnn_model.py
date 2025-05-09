#!/usr/bin/env python3
"""
Train a custom model using the fine-tuned CNN-enhanced FLAN-UL2 model.

This script loads the fine-tuned CNN-enhanced FLAN-UL2 model and uses it to help train
a custom model on all 5 available datasets.

Usage:
    python train_with_cnn_model.py --cnn-model-path /path/to/cnn/model
                                   --output-dir /path/to/output
                                   --epochs 3
                                   --batch-size 4
                                   --max-samples 1000
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import local modules
from .text_generator import CNNTextGenerator
from .unified_dataset_handler import UnifiedDatasetHandler

class CustomEncoderDecoder(nn.Module):
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
        logger.info(f"Loading CNN-enhanced FLAN-UL2 model from {cnn_model_path}")
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        
        # Freeze the CNN model parameters
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        
        # Get the hidden size from the CNN model
        self.feature_size = self.cnn_model.hidden_size
        
        # Create encoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        # Create decoder layers
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.feature_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.feature_size, self.cnn_model.tokenizer.vocab_size)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and force_gpu else "cpu")
        self.to(self.device)
        
        logger.info(f"Initialized custom encoder-decoder model on {self.device}")
    
    def _load_cnn_model(self, model_path: str) -> CNNTextGenerator:
        """Load the CNN-enhanced FLAN-UL2 model"""
        # Check if the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        # Load the model
        model_dir = os.path.dirname(model_path)
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        
        # Create a new CNN model
        cnn_model = CNNTextGenerator(
            model_name_or_path="google/flan-ul2",
            force_gpu=True,
            cnn_layers=3,
            load_in_4bit=True
        )
        
        # Load the saved model
        state_dict = torch.load(model_path, map_location="cpu")
        cnn_model.base_model.load_state_dict(state_dict["base_model"])
        
        # Load CNN layers
        for i, layer_state in enumerate(state_dict["cnn_layers"]):
            if i < len(cnn_model.cnn_layers_list):
                cnn_model.cnn_layers_list[i].load_state_dict(layer_state)
        
        # Load adapter
        cnn_model.adapter.load_state_dict(state_dict["adapter"])
        
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
            if hasattr(self.cnn_model.base_model, "transformer") and hasattr(self.cnn_model.base_model.transformer, "wte"):
                # GPT-2 style models
                embeddings = self.cnn_model.base_model.transformer.wte(input_ids)
            elif hasattr(self.cnn_model.base_model, "get_input_embeddings"):
                # Generic approach for most models
                embedding_layer = self.cnn_model.base_model.get_input_embeddings()
                embeddings = embedding_layer(input_ids)
            else:
                raise ValueError("Could not get embeddings from model")
            
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
        if hasattr(self.cnn_model.base_model, "transformer") and hasattr(self.cnn_model.base_model.transformer, "wte"):
            # GPT-2 style models
            decoder_embeddings = self.cnn_model.base_model.transformer.wte(decoder_input_ids)
        elif hasattr(self.cnn_model.base_model, "get_input_embeddings"):
            # Generic approach for most models
            embedding_layer = self.cnn_model.base_model.get_input_embeddings()
            decoder_embeddings = embedding_layer(decoder_input_ids)
        else:
            raise ValueError("Could not get embeddings from model")
        
        # Pass through decoder
        decoder_output = self.decoder(
            decoder_embeddings,
            encoder_output,
            tgt_key_padding_mask=~decoder_attention_mask.bool()
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate text using the encoder-decoder model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        batch_size = input_ids.shape[0]
        
        # Encode input
        with torch.no_grad():
            # Get embeddings from base model
            if hasattr(self.cnn_model.base_model, "transformer") and hasattr(self.cnn_model.base_model.transformer, "wte"):
                # GPT-2 style models
                embeddings = self.cnn_model.base_model.transformer.wte(input_ids)
            elif hasattr(self.cnn_model.base_model, "get_input_embeddings"):
                # Generic approach for most models
                embedding_layer = self.cnn_model.base_model.get_input_embeddings()
                embeddings = embedding_layer(input_ids)
            else:
                raise ValueError("Could not get embeddings from model")
            
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
        
        # Initialize decoder input with start token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.cnn_model.tokenizer.bos_token_id if self.cnn_model.tokenizer.bos_token_id is not None else self.cnn_model.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Generate tokens auto-regressively
        for _ in range(max_length - 1):
            # Get decoder embeddings
            if hasattr(self.cnn_model.base_model, "transformer") and hasattr(self.cnn_model.base_model.transformer, "wte"):
                # GPT-2 style models
                decoder_embeddings = self.cnn_model.base_model.transformer.wte(decoder_input_ids)
            elif hasattr(self.cnn_model.base_model, "get_input_embeddings"):
                # Generic approach for most models
                embedding_layer = self.cnn_model.base_model.get_input_embeddings()
                decoder_embeddings = embedding_layer(decoder_input_ids)
            else:
                raise ValueError("Could not get embeddings from model")
            
            # Create decoder attention mask
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
            
            # Pass through decoder
            decoder_output = self.decoder(
                decoder_embeddings,
                encoder_output,
                tgt_key_padding_mask=~decoder_attention_mask.bool()
            )
            
            # Project to vocabulary
            logits = self.output_projection(decoder_output[:, -1:, :])
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample from logits
            probs = torch.softmax(logits.squeeze(1), dim=-1)
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            
            # Check if all sequences have reached the end token
            if (next_token == self.cnn_model.tokenizer.eos_token_id).all():
                break
        
        return decoder_input_ids

def train_custom_model(args):
    """Train the custom encoder-decoder model"""
    # Initialize the dataset handler
    dataset_handler = UnifiedDatasetHandler()
    
    # Load all 5 datasets
    datasets = []
    for dataset_name in ["writing_prompts", "persona_chat", "pile", "openassistant", "gpteacher"]:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = dataset_handler.load_dataset(
            dataset_name=dataset_name,
            max_samples=args.max_samples // 5  # Divide max samples among 5 datasets
        )
        datasets.append(dataset)
    
    # Combine datasets
    combined_batches = []
    for dataset in datasets:
        combined_batches.extend(dataset.get("batches", []))
    
    logger.info(f"Combined {len(combined_batches)} batches from all datasets")
    
    # Initialize the custom model
    model = CustomEncoderDecoder(
        cnn_model_path=args.cnn_model_path,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        force_gpu=args.force_gpu
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Process batches
        for i, (input_batch, target_batch) in enumerate(combined_batches):
            # Skip invalid batches
            if input_batch is None or target_batch is None:
                continue
            
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
            
            # Forward pass
            logits = model(
                input_ids=input_batch,
                attention_mask=input_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=target_attention_mask
            )
            
            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Log progress
            if (i + 1) % args.log_every == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(combined_batches)}, Loss: {loss.item():.4f}")
        
        # Log epoch results
        avg_loss = total_loss / len(combined_batches)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if args.save_checkpoints:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "custom_encoder_decoder.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    return model

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train a custom encoder-decoder model using a fine-tuned CNN model")
    
    # Required arguments
    parser.add_argument("--cnn-model-path", type=str, required=True,
                        help="Path to the fine-tuned CNN model")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the trained model")
    
    # Optional arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum number of samples to use from each dataset")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for training")
    parser.add_argument("--hidden-size", type=int, default=768,
                        help="Hidden size for the encoder-decoder model")
    parser.add_argument("--num-encoder-layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--force-gpu", action="store_true",
                        help="Force GPU usage")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save checkpoints during training")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N batches")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    model = train_custom_model(args)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
