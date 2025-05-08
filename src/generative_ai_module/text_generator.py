import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import os
import tempfile
import shutil

from .utils import is_zipfile, process_zip

class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CombinedModel, self).__init__()

        # Add embedding layer for tokenized inputs
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Simple LSTM model
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        # Add safety checks for tensor types and shapes
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")

        # Handle different input shapes:
        # x shape can be either:
        # [batch_size, seq_len, input_size] (3D tensor) - one-hot encoded
        # [batch_size, seq_len] (2D tensor) - token indices
        # [batch_size, input_size] (2D tensor) - single time step one-hot
        # [seq_len] (1D tensor) - single sample token indices

        # Handle 1D tensor (single sequence of tokens)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension [1, seq_len]

        # Handle unexpected higher dimensions (> 3D)
        if x.dim() > 3:
            x = x.view(x.size(0), x.size(1), -1)  # Flatten extra dimensions to 3D

        # Process input based on dimensionality and dtype
        if x.dim() == 2:
            if x.dtype in [torch.long, torch.int64]:
                # Input is token indices [batch_size, seq_len]
                # Ensure values are within valid embedding range
                vocab_size = self.embedding.num_embeddings
                if x.max() >= vocab_size:
                    # Clip indices to prevent out-of-bounds errors
                    x = torch.clamp(x, 0, vocab_size - 1)
                # Pass through embedding layer
                x = self.embedding(x)
            else:
                # If input is [batch_size, features] one-hot, add sequence dimension
                x = x.unsqueeze(1)
                # And convert to float in case it's not
                x = x.float()
        elif x.dim() == 3:
            # 3D tensor inputs are assumed to be one-hot encodings
            # No embedding needed, just make sure it's float
            x = x.float()

        # Execute forward pass with error handling
        try:
            # Pass through LSTM
            lstm_out, hidden = self.lstm(x, hidden)

            # Get output from last time step only
            last_output = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out
            # Apply dropout and linear layer
            last_output = self.dropout(last_output)
            output = self.fc(last_output)

            return output, hidden
        except RuntimeError as e:
            # Handle specific runtime errors with more informative messages
            if "device-side assert triggered" in str(e):
                error_msg = (f"CUDA error: Device-side assert triggered. Input shape: {x.shape}, "
                           f"dtype: {x.dtype}. This might be caused by invalid indices or values.")
                raise RuntimeError(error_msg) from e
            elif "expected hidden[0] size" in str(e):
                error_msg = (f"LSTM hidden state size mismatch. Input shape: {x.shape}, "
                           f"hidden state shapes: {hidden[0].shape}, {hidden[1].shape} if hidden else 'None'")
                raise RuntimeError(error_msg) from e
            elif "CUDA out of memory" in str(e):
                # Try to free memory and provide a helpful message
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error_msg = "CUDA out of memory error. Try reducing batch size or sequence length."
                raise RuntimeError(error_msg) from e
            else:
                # Re-raise the original error
                raise

class TextGenerator(nn.Module):
    def __init__(self, force_gpu=False):
        super().__init__()
        self.all_chars = string.printable
        self.char_to_index = {char: i for i, char in enumerate(self.all_chars)}
        self.unknown_token = "<UNK>"

        # Add unknown token to character set and mapping
        self.char_to_index[self.unknown_token] = len(self.all_chars)
        self.all_chars += self.unknown_token

        # Now set n_chars to the actual vocabulary size
        self.n_chars = len(self.char_to_index)

        # Create index to character mapping
        self.index_to_char = {i: char for char, i in self.char_to_index.items()}

        # Initialize model with correct vocabulary size
        self.model = CombinedModel(self.n_chars, 128, self.n_chars, 2)

        self.force_gpu = force_gpu
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.context = None

    def _get_device(self):
        """Determine the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
        if self.force_gpu:
            # Try to use MPS (Metal Performance Shaders) for Apple Silicon
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU) for text generation")
                return torch.device("mps")
            # Fall back to CUDA if available
            elif torch.cuda.is_available():
                print(f"Using CUDA GPU for text generation: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            else:
                print("Warning: GPU requested but neither MPS nor CUDA is available. Falling back to CPU.")
                return torch.device("cpu")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU) for text generation")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print(f"Using CUDA GPU for text generation: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            print("Using CPU for text generation (no GPU available)")
            return torch.device("cpu")

    def char_indices_tensor(self, string):
        tensor = torch.zeros(len(string), dtype=torch.long)
        for i, char in enumerate(string):
            tensor[i] = self.char_to_index.get(char, self.char_to_index[self.unknown_token])

        return tensor.to(self.device)

    def handle_input(self, input_data):
        try:
            if isinstance(input_data, str):
                if not is_zipfile(input_data):
                    return self.char_indices_tensor(input_data)

                # Create a temporary directory to extract the ZIP file
                extract_dir = tempfile.mkdtemp()

                # Process the ZIP file
                if process_zip(input_data, extract_dir):
                    # Read the extracted files
                    all_text = []
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            try:
                                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                    all_text.append(f.read())
                            except UnicodeDecodeError:
                                # Skip binary files
                                pass

                    # Clean up the temporary directory
                    shutil.rmtree(extract_dir)

                    # Join all texts and return
                    combined_text = " ".join(all_text)
                    return self.char_indices_tensor(combined_text)
                else:
                    raise ValueError(f"Failed to process ZIP file: {input_data}")
            elif hasattr(input_data, 'read'):  # Handle file-like objects
                try:
                    content = input_data.read().decode('utf-8')
                    return self.char_indices_tensor(content)
                except UnicodeDecodeError as e:
                    raise ValueError("Failed to decode file content") from e
            else:
                raise ValueError("Unsupported input type")
        except Exception as e:
            raise ValueError(f"Error processing input: {str(e)}") from e


    def generate(self, initial_str="", pred_len=1000, temperature=0.8):
        """Generate text starting from initial_str"""
        self.model.eval()

        # Start with the initial string
        predicted = initial_str

        # Convert to input format
        char_to_idx = self.char_to_index
        idx_to_char = self.index_to_char
        n_chars = len(char_to_idx)
        unknown_token = self.unknown_token
        device = self.device

        # Initial hidden state
        hidden = None

        # Generation loop
        with torch.no_grad():
            for _ in range(pred_len):
                try:
                    # Get the last 'sequence_length' characters (or pad with spaces if not enough)
                    context_length = min(100, len(predicted))  # Use at most 100 chars for context
                    if len(predicted) >= context_length:
                        context = predicted[-context_length:]
                    else:
                        context = ' ' * (context_length - len(predicted)) + predicted

                    # Convert to token indices - this works with the embedding layer
                    token_indices = torch.zeros(1, len(context), dtype=torch.long, device=device)
                    for t, char in enumerate(context):
                        idx = char_to_idx.get(char, char_to_idx[unknown_token])
                        token_indices[0, t] = idx

                    # Get prediction using token indices
                    output, hidden = self.model(token_indices, hidden)

                    # Apply temperature scaling and sample
                    probs = F.softmax(output.squeeze() / temperature, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()

                    # Convert to character and append to result
                    next_char = idx_to_char.get(next_char_idx, unknown_token)
                    predicted += next_char

                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Try a simpler approach on error
                    try:
                        # Use just the last character as input
                        last_char = predicted[-1] if predicted else ' '
                        idx = char_to_idx.get(last_char, char_to_idx[unknown_token])

                        # Create a simpler input tensor (just one token)
                        simple_input = torch.tensor([[idx]], dtype=torch.long).to(device)

                        # Get prediction with simpler input
                        output, _ = self.model(simple_input)
                        probs = F.softmax(output.squeeze() / temperature, dim=-1)
                        next_char_idx = torch.multinomial(probs, 1).item()
                        next_char = idx_to_char.get(next_char_idx, unknown_token)
                        predicted += next_char
                    except Exception as inner_e:
                        print(f"Error during fallback generation: {inner_e}")
                        # If all else fails, just append a space
                        predicted += ' '

        return predicted

    def train(self, data, epochs=10, gradient_accumulation_steps=1):
        """
        Train the model on batched data

        Args:
            data: List of (input_batch, target_batch) tuples
            epochs: Number of epochs to train
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights

        Returns:
            List of losses
        """
        self.model.train()
        losses = []
        criterion = nn.CrossEntropyLoss()

        # If data is a string, use character-by-character training
        if isinstance(data, str):
            return self.train_character_model_step(data, [])

        # Otherwise, assume it's batched data
        try:
            # Free up GPU memory before starting training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                steps_since_update = 0

                for input_batch, target_batch in data:
                    try:
                        # Move to device
                        input_batch = input_batch.to(self.device)
                        target_batch = target_batch.to(self.device)

                        # Get vocabulary size
                        vocab_size = self.model.embedding.num_embeddings

                        # Safety check: Ensure target indices are within valid range
                        if target_batch.max() >= vocab_size:
                            target_batch = torch.clamp(target_batch, 0, vocab_size - 1)

                        # Forward pass - don't clear gradients yet for gradient accumulation
                        if steps_since_update == 0:
                            self.optimizer.zero_grad()

                        outputs, _ = self.model(input_batch)

                        # Calculate loss based on target shape
                        if target_batch.dim() == 1:
                            # For 1D targets (batch of single tokens)
                            loss = criterion(outputs, target_batch)
                        else:
                            # For 2D targets (batch of sequences)
                            loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))

                        # Scale loss by gradient accumulation steps to keep things balanced
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                        # Backward pass
                        loss.backward()

                        steps_since_update += 1

                        # Only update weights after accumulating gradients for specified steps
                        if steps_since_update >= gradient_accumulation_steps:
                            # Add gradient clipping to prevent explosive gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            steps_since_update = 0

                        epoch_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
                        batch_count += 1

                    except RuntimeError as e:
                        if "device-side assert triggered" in str(e):
                            print(f"CUDA Error in batch: {str(e)}")
                            print(f"Input shape: {input_batch.shape}, Target shape: {target_batch.shape}")
                            if hasattr(target_batch, 'min') and hasattr(target_batch, 'max'):
                                print(f"Target range: min={target_batch.min().item()}, max={target_batch.max().item()}")
                            # Free CUDA memory and continue
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            steps_since_update = 0  # Reset gradient accumulation counter
                        elif "CUDA out of memory" in str(e):
                            print(f"CUDA out of memory error. Clearing cache and continuing with next batch.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            steps_since_update = 0  # Reset gradient accumulation counter
                        else:
                            print(f"Error during batch training: {e}")
                        continue

                # Make sure to perform final optimization step if there are remaining gradients
                if steps_since_update > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    losses.append(avg_loss)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, No valid batches.")

            return losses

        except Exception as e:
            print(f"Error during training: {e}")
            return losses

    def train_character_model_step(self, text, losses):
        input_sequence = self.char_indices_tensor(text[:-1])
        target_sequence = self.char_indices_tensor(text[1:])

        criterion = torch.nn.CrossEntropyLoss()
        hidden = None

        # Train on the entire sequence
        self.optimizer.zero_grad()
        output, hidden = self.model(input_sequence.unsqueeze(0), hidden)
        loss = criterion(output, target_sequence.unsqueeze(0))
        loss.backward()
        self.optimizer.step()

        losses.append(loss.item())

    def reinitialize_for_sequence_length(self, sequence_length):
        """
        Reinitialize the model for a different sequence length

        Args:
            sequence_length: New sequence length to use
        """
        # Create a new model with the same parameters but potentially different architecture
        self.model = CombinedModel(self.n_chars, 128, self.n_chars, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.context = None

        # Move to the same device if it was on GPU
        for param in self.model.parameters():
            if param.device.type != 'cpu':
                self.model = self.model.to(param.device)
                break

    def train_from_preprocessed(self, dataset_name="persona_chat", epochs=5, preprocessed_path=None):
        """
        Train the model using preprocessed data

        Args:
            dataset_name: Name of the preprocessed dataset to use
            epochs: Number of training epochs
            preprocessed_path: Optional custom path to the preprocessed file

        Returns:
            Training loss history
        """
        # Import here to avoid circular imports
        from .dataset_processor import DatasetProcessor

        # Create a processor to load the data
        processor = DatasetProcessor(self)

        # Load preprocessed data
        data = processor.load_preprocessed_data(dataset_name, preprocessed_path)

        if 'batches' not in data or not data['batches']:
            raise ValueError(f"No valid batches found in preprocessed data: {dataset_name}")

        print(f"Training on {len(data['batches'])} batches from preprocessed {dataset_name} dataset")
        return self.train(data['batches'], epochs=epochs)

    def save_model(self, path="models/text_generator_model.pt"):
        """Save the trained model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="models/text_generator_model.pt"):
        """Load a trained model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load the model
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set to evaluation mode
        print(f"Model loaded from {path}")

    def adapt_to_tokenizer(self, tokenizer):
        """
        Adapt the generator to work with a custom tokenizer

        Args:
            tokenizer: A tokenizer object with encode/decode methods and vocab_size attribute
        """
        # Update the model with a new output layer matching the tokenizer's vocab size
        vocab_size = tokenizer.vocab_size

        # Create a new model with the right vocabulary size
        old_model = self.model
        hidden_size = old_model.fc.in_features  # Get hidden size from existing model

        # Create new model with updated vocab size
        self.model = CombinedModel(vocab_size, hidden_size, vocab_size,
                                  num_layers=old_model.lstm.num_layers)

        # Initialize optimizer for the new model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)

        print(f"Model adapted to tokenizer with vocabulary size: {vocab_size}")

class CNNTextGenerator(TextGenerator):
    """
    CNN-enhanced text generation model

    This class extends the TextGenerator with convolutional layers to enhance
    pattern recognition in text generation tasks. It combines the strengths of
    CNNs for local feature extraction with transformers for sequence modeling.

    Optimized for A6000 GPUs with 48 GiB VRAM and memory-efficient training options.
    """

    def __init__(self,
                model_name_or_path="google/flan-ul2-20b",
                cnn_layers=3,  # Increased for A6000 GPU
                cnn_kernel_sizes=None,
                cnn_dropout=0.1,
                force_gpu=True,
                gpu_type="A6000",
                vram_size=48,
                load_in_4bit=True,  # Use 4-bit quantization by default for memory efficiency
                load_in_8bit=False,  # 8-bit quantization alternative
                quantization_config=None,
                use_flash_attention_2=True,  # Enabled by default for A6000
                gradient_checkpointing=True,  # Enabled by default for memory efficiency
                lora_rank=32,  # Optimized for A6000 GPU
                lora_alpha=64,
                lora_dropout=0.05,
                batch_size=None,  # Will be set based on GPU type
                gradient_accumulation_steps=None,  # Will be set based on GPU type
                max_length=4096,  # Increased for A6000 GPU
                bf16=True,  # Use bfloat16 precision for A6000 GPUs
                num_workers=None):  # Number of workers for data loading
        super().__init__(force_gpu)
        """
        Initialize the CNN-enhanced text generator optimized for A6000 GPUs

        Args:
            model_name_or_path: Model name or path for the base transformer
            cnn_layers: Number of CNN layers to use (3 for A6000)
            cnn_kernel_sizes: List of kernel sizes for CNN layers (default: [3, 5, 7])
            cnn_dropout: Dropout rate for CNN layers
            force_gpu: Whether to force GPU usage
            gpu_type: GPU type (A6000, A4000, RTX5000)
            vram_size: GPU VRAM size in GiB
            load_in_4bit: Whether to load model in 4-bit precision (default: True for memory efficiency)
            load_in_8bit: Whether to load model in 8-bit precision (default: False)
            quantization_config: Custom quantization configuration (overrides load_in_4bit/8bit if provided)
            use_flash_attention_2: Whether to use Flash Attention 2 for faster training (default: True for A6000)
            gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency (default: True)
            lora_rank: LoRA rank parameter for fine-tuning (32 for A6000)
            lora_alpha: LoRA alpha parameter for fine-tuning (64 for A6000)
            lora_dropout: LoRA dropout parameter for fine-tuning
            batch_size: Batch size for training (if None, will be set based on GPU type)
            gradient_accumulation_steps: Steps to accumulate gradients (if None, will be set based on GPU type)
            max_length: Maximum sequence length for training (4096 for A6000)
            bf16: Whether to use bfloat16 precision (True for A6000 with Ampere or newer architecture)
            num_workers: Number of workers for data loading (if None, will be set based on GPU type)
        """
        # Store parameters
        self.cnn_layers = cnn_layers
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 5, 7]
        self.cnn_dropout = cnn_dropout
        self.quantization_config = quantization_config
        self.use_flash_attention_2 = use_flash_attention_2
        self.gradient_checkpointing = gradient_checkpointing
        self.gpu_type = gpu_type
        self.vram_size = vram_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.bf16 = bf16
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.num_workers = num_workers

        # Configure based on GPU type and VRAM
        self._configure_for_gpu()

        # Initialize CNN model
        self._initialize_model(model_name_or_path)

    def _configure_for_gpu(self):
        """Configure training parameters based on GPU type and VRAM size"""
        print(f"Configuring for GPU type {self.gpu_type} with {self.vram_size} GiB VRAM")

        # Set batch size and gradient accumulation steps based on GPU type if not specified
        if self.batch_size is None or self.gradient_accumulation_steps is None or self.num_workers is None:
            if self.gpu_type == "A6000" and self.vram_size >= 48:
                # A6000 with 48+ GiB VRAM - maximize parameters while staying within constraints
                print("Using optimized settings for A6000 with 48+ GiB VRAM")
                self.batch_size = self.batch_size or 3  # Reduced to ensure stability with large models
                self.gradient_accumulation_steps = self.gradient_accumulation_steps or 8
                self.max_length = 2048  # Reduced from 4096 to ensure stability with FLAN-UL2
                self.lora_rank = 32     # Increase LoRA rank for better quality
                self.lora_alpha = 64    # Increase LoRA alpha for better adaptation
                self.lora_dropout = 0.05  # Optimal dropout for stability
                self.num_workers = self.num_workers if self.num_workers is not None else 8  # Match your 8 CPU cores
                self.warmup_ratio = 0.03  # Optimal warmup for large models
                self.weight_decay = 0.01  # Prevent overfitting
                self.adam_beta1 = 0.9   # Standard beta1 for AdamW
                self.adam_beta2 = 0.999  # Standard beta2 for AdamW
                self.adam_epsilon = 1e-8  # Standard epsilon for AdamW
                self.max_grad_norm = 1.0  # Prevent gradient explosion
                # Memory optimization
                self.load_in_4bit = True  # Use 4-bit quantization for maximum memory efficiency

            elif self.gpu_type == "A6000" and self.vram_size >= 40:
                # A6000 with 40-48 GiB VRAM
                print("Using optimized settings for A6000 with 40-48 GiB VRAM")
                self.batch_size = self.batch_size or 3
                self.gradient_accumulation_steps = self.gradient_accumulation_steps or 8
                self.max_length = 3072
                self.lora_rank = 24
                self.lora_alpha = 48
                self.lora_dropout = 0.05
                self.num_workers = self.num_workers if self.num_workers is not None else 6
                self.warmup_ratio = 0.03
                self.weight_decay = 0.01
                self.adam_beta1 = 0.9
                self.adam_beta2 = 0.999
                self.adam_epsilon = 1e-8
                self.max_grad_norm = 1.0

            elif self.gpu_type == "A4000" or (self.gpu_type == "A6000" and self.vram_size < 40):
                # A4000 or A6000 with less VRAM
                print("Using optimized settings for A4000 or A6000 with <40 GiB VRAM")
                self.batch_size = self.batch_size or 2
                self.gradient_accumulation_steps = self.gradient_accumulation_steps or 16
                self.max_length = 2048
                self.lora_rank = 16
                self.lora_alpha = 32
                self.lora_dropout = 0.05
                self.num_workers = self.num_workers if self.num_workers is not None else 4
                self.warmup_ratio = 0.03
                self.weight_decay = 0.01
                self.adam_beta1 = 0.9
                self.adam_beta2 = 0.999
                self.adam_epsilon = 1e-8
                self.max_grad_norm = 1.0

            elif self.gpu_type == "RTX5000":
                # RTX5000 with limited VRAM
                print("Using optimized settings for RTX5000")
                self.batch_size = self.batch_size or 1
                self.gradient_accumulation_steps = self.gradient_accumulation_steps or 32
                self.max_length = 1024
                self.lora_rank = 8
                self.lora_alpha = 16
                self.lora_dropout = 0.05
                self.num_workers = self.num_workers if self.num_workers is not None else 2
                self.warmup_ratio = 0.03
                self.weight_decay = 0.01
                self.adam_beta1 = 0.9
                self.adam_beta2 = 0.999
                self.adam_epsilon = 1e-8
                self.max_grad_norm = 1.0

        # Calculate effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"Effective batch size: {self.effective_batch_size}")

        # Adjust learning rate based on batch size (linear scaling rule)
        base_lr = 2e-5
        base_batch_size = 32
        self.learning_rate = base_lr * (self.effective_batch_size / base_batch_size)
        print(f"Adjusted learning rate: {self.learning_rate}")

        # Calculate warmup steps based on warmup ratio
        self.warmup_steps = int(self.warmup_ratio * self.effective_batch_size * 100)  # Assuming ~100 steps per epoch
        print(f"Warmup steps: {self.warmup_steps}")

        # Memory optimization settings
        if torch.cuda.is_available():
            # Set environment variables for optimal memory usage
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.8"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))  # Use available CPU cores efficiently
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow logging

            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Print GPU information
            device_name = torch.cuda.get_device_name(0)
            device_capability = torch.cuda.get_device_capability(0)
            print(f"Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")

        return self

    def _initialize_model(self, model_name_or_path: str, **kwargs):
        """Initialize the hybrid CNN-Transformer model"""
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        # Determine model type
        is_gpt2_model = "gpt2" in model_name_or_path.lower()
        is_t5_model = any(name in model_name_or_path.lower() for name in ["t5", "flan", "ul2"])

        # Set model type attribute for use in forward pass
        self.model_type = "seq2seq" if is_t5_model else "causal"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with optimization options
        load_kwargs = {}

        # Configure quantization for memory efficiency
        if self.quantization_config:
            # Use provided quantization config
            load_kwargs['quantization_config'] = self.quantization_config
        elif self.load_in_4bit:
            # Configure 4-bit quantization for maximum memory efficiency
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("Using 4-bit quantization for maximum memory efficiency")
        elif self.load_in_8bit:
            # Configure 8-bit quantization
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("Using 8-bit quantization")

        # Add Flash Attention 2 if requested and possible
        if self.use_flash_attention_2:
            if is_gpt2_model:
                print("Flash Attention 2 is not supported for GPT2 models - disabling")
                self.use_flash_attention_2 = False
            else:
                # Check if flash attention is available
                try:
                    # Just check if it can be imported
                    import importlib.util
                    if importlib.util.find_spec("flash_attn") is not None:
                        print("Flash Attention 2 detected - enabling for transformer model")
                        load_kwargs['use_flash_attention_2'] = True
                    else:
                        raise ImportError("flash_attn module not found")
                except ImportError:
                    print("Flash Attention 2 requested but not installed - continuing without it")
                    print("To install: pip install flash-attn --no-build-isolation")

        # Load the appropriate model type with optimized settings
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and self.bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using {torch_dtype} precision for model loading")

        if is_t5_model:
            print(f"Loading T5-based model: {model_name_or_path}")
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                max_length=self.max_length,
                trust_remote_code=True,
                **load_kwargs
            )
        else:
            # Default to causal language model (GPT-style)
            print(f"Loading causal language model: {model_name_or_path}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                **load_kwargs
            )

        # Enable gradient checkpointing if requested (for memory efficiency)
        if self.gradient_checkpointing and hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for transformer model")

        # Apply LoRA adapters if using quantization
        if self.quantization_config or self.load_in_4bit or self.load_in_8bit:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

                # Prepare model for k-bit training
                self.base_model = prepare_model_for_kbit_training(
                    self.base_model,
                    use_gradient_checkpointing=self.gradient_checkpointing
                )

                # Define LoRA configuration
                lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    task_type="CAUSAL_LM" if not is_t5_model else "SEQ_2_SEQ_LM"
                )

                # Apply LoRA adapters
                self.base_model = get_peft_model(self.base_model, lora_config)
                print(f"Applied LoRA adapters with rank={self.lora_rank}, alpha={self.lora_alpha}")
            except ImportError:
                print("PEFT library not available. LoRA adapters not applied.")
                print("To install: pip install peft")

        # Get the model's config
        config = self.base_model.config

        # Get hidden size for dimensionality
        self.hidden_size = config.hidden_size

        # Create CNN layers for pattern extraction
        self.cnn_layers_list = nn.ModuleList()
        for i in range(self.cnn_layers):
            kernel_size = self.cnn_kernel_sizes[i] if i < len(self.cnn_kernel_sizes) else 3
            padding = kernel_size // 2  # Same padding to maintain sequence length

            # Create convolutional layer with batch normalization and dropout
            cnn_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=kernel_size,
                    padding=padding
                ),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.cnn_dropout)
            )

            self.cnn_layers_list.append(cnn_layer)

        # Create adapter to transform CNN outputs back to transformer format
        self.adapter = nn.Linear(self.hidden_size, self.hidden_size)

        # Move model to appropriate device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() and self.force_gpu
            else torch.device("mps") if torch.backends.mps.is_available() and self.force_gpu
            else torch.device("cpu")
        )

        # Initialize optimizer with configured learning rate and parameters
        from torch.optim import AdamW
        from transformers import get_scheduler

        # Use AdamW optimizer with optimized parameters
        self.optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon
        )

        # Create learning rate scheduler
        self.lr_scheduler = get_scheduler(
            name="cosine",  # Cosine scheduler with warmup for better convergence
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.effective_batch_size * 100  # Estimate total steps
        )

        # Print model information
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized CNN-enhanced text generator with {self.cnn_layers} CNN layers")
        print(f"Model has ~{num_params / 1_000_000:.2f}M parameters")
        print(f"Trainable parameters: ~{trainable_params / 1_000_000:.2f}M")
        print(f"Learning rate: {self.learning_rate}")

        # Move model to device
        self.to(self.device)

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_input_ids=None, **kwargs):
        """
        Forward pass through the hybrid CNN-Transformer model

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            decoder_input_ids: Decoder input IDs (for seq2seq models)

        Returns:
            Transformer model outputs with logits
        """
        # Get embeddings from base model
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            # GPT-2 style models
            embeddings = self.base_model.transformer.wte(input_ids)
        elif hasattr(self.base_model, "get_input_embeddings"):
            # Generic approach for most models
            embedding_layer = self.base_model.get_input_embeddings()
            embeddings = embedding_layer(input_ids)
        else:
            raise ValueError("Could not get embeddings from model")

        # Apply CNN layers for feature extraction
        # First, transpose for CNN (batch_size, hidden_size, seq_len)
        x = embeddings.transpose(1, 2)

        # Pass through each CNN layer
        for cnn_layer in self.cnn_layers_list:
            x = cnn_layer(x)

        # Transpose back to transformer format (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2)

        # Apply adapter to ensure compatibility with transformer
        enhanced_embeddings = self.adapter(x)

        # Add residual connection to preserve original embeddings
        enhanced_embeddings = enhanced_embeddings + embeddings

        # Handle different model types
        if self.model_type == "seq2seq":
            # For T5/FLAN models (encoder-decoder)
            if decoder_input_ids is None and labels is not None:
                # Use labels as decoder_input_ids if not provided (common practice)
                decoder_input_ids = labels

            # Pass through the seq2seq model with enhanced embeddings
            outputs = self.base_model(
                inputs_embeds=enhanced_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )
        else:
            # For causal language models (GPT-style)
            outputs = self.base_model(
                inputs_embeds=enhanced_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        return outputs

    def train(self, data, epochs=3, gradient_accumulation_steps=None, eval_steps=None, save_steps=None, checkpoint_dir=None):
        """
        Train the hybrid CNN-Transformer model with gradient accumulation

        Args:
            data: Training data
            epochs: Number of training epochs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            eval_steps: How often to evaluate (None = no evaluation)
            save_steps: How often to save checkpoints (None = no intermediate saves)
            checkpoint_dir: Directory to save checkpoints

        Returns:
            List of losses during training
        """
        # Use configured gradient_accumulation_steps if not provided
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
            print(f"Using configured gradient accumulation steps: {gradient_accumulation_steps}")

        # Move model to training mode
        self.base_model.train()
        for layer in self.cnn_layers_list:
            layer.train()
        self.adapter.train()

        losses = []

        # Configure progress bar
        try:
            from tqdm.auto import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("tqdm not installed. Progress bar disabled.")

        # Ensure data is in right format
        batches = data.get('batches', []) if isinstance(data, dict) else data

        # Validate that we have training data
        if not batches:
            raise ValueError("No training data available. Dataset processing failed.")

        if len(batches) == 0:
            print("No valid batches found in dataset")
            raise RuntimeError("Training data is empty. Check dataset processing.")

        # Print training configuration
        print(f"Starting training with:")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"- Effective batch size: {self.batch_size * gradient_accumulation_steps}")
        print(f"- Learning rate: {self.learning_rate}")
        print(f"- Max sequence length: {self.max_length}")
        print(f"- Evaluation steps: {eval_steps if eval_steps else 'None'}")
        print(f"- Save steps: {save_steps if save_steps else 'None'}")

        # Track global step for eval_steps
        global_step = 0

        # Start training
        for epoch in range(epochs):
            print(f"Beginning epoch {epoch+1}/{epochs}")

            epoch_loss = 0
            batch_count = 0
            steps_since_update = 0

            # Create iterator with or without progress bar
            if use_tqdm:
                iterator = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                iterator = batches

            # Clear CUDA cache at start of epoch if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training loop
            for input_batch, target_batch in iterator:
                if input_batch is None or target_batch is None:
                    self.logger.warning("Skipping invalid batch")
                    continue
                try:
                    # Move data to device
                    input_batch = input_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    # Forward pass through hybrid model
                    outputs = self.forward(input_batch)
                    logits = outputs.logits

                    # Calculate loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_batch.view(-1)
                    )

                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Update step counting
                    steps_since_update += 1
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    batch_count += 1
                    global_step += 1

                    # Update weights if we've accumulated enough gradients
                    if steps_since_update >= gradient_accumulation_steps:
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(
                            list(self.cnn_layers_list.parameters()) +
                            list(self.adapter.parameters()) +
                            list(self.base_model.parameters()),
                            max_norm=self.max_grad_norm
                        )

                        # Update weights
                        self.optimizer.step()

                        # Update learning rate with scheduler
                        self.lr_scheduler.step()

                        # Log current learning rate
                        if global_step % 10 == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            print(f"Step {global_step}: Learning rate = {current_lr:.6f}")

                        self.optimizer.zero_grad()
                        steps_since_update = 0

                    # Evaluation
                    if eval_steps and global_step % eval_steps == 0:
                        # Compute validation loss on a sample input
                        self.base_model.eval()
                        for layer in self.cnn_layers_list:
                            layer.eval()
                        self.adapter.eval()

                        with torch.no_grad():
                            val_loss = 0.0
                            val_count = 0

                            # Use first 10 batches for validation
                            for i, (val_input, val_target) in enumerate(batches[:10]):
                                val_input = val_input.to(self.device)
                                val_target = val_target.to(self.device)

                                val_outputs = self.forward(val_input)
                                val_logits = val_outputs.logits

                                batch_loss = torch.nn.functional.cross_entropy(
                                    val_logits.view(-1, val_logits.size(-1)),
                                    val_target.view(-1)
                                )

                                val_loss += batch_loss.item()
                                val_count += 1

                                if i >= 9:  # Only use 10 batches for eval
                                    break

                            avg_val_loss = val_loss / max(1, val_count)
                            losses.append(avg_val_loss)
                            print(f"Step {global_step}: Validation loss = {avg_val_loss:.4f}")

                        # Back to training mode
                        self.base_model.train()
                        for layer in self.cnn_layers_list:
                            layer.train()
                        self.adapter.train()

                    # Save checkpoint if requested
                    if checkpoint_dir and save_steps and global_step % save_steps == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        self.save_model(os.path.join(checkpoint_path, "model.pt"))
                        print(f"Checkpoint saved at step {global_step}")

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OOM error. Clearing cache and skipping batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        steps_since_update = 0
                        self.optimizer.zero_grad()
                    else:
                        print(f"Error in training: {e}")
                    continue

            # Perform final optimization step if needed
            if steps_since_update > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.cnn_layers_list.parameters()) +
                    list(self.adapter.parameters()) +
                    list(self.base_model.parameters()),
                    max_norm=self.max_grad_norm
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Log final learning rate
                final_lr = self.optimizer.param_groups[0]['lr']
                print(f"Final learning rate: {final_lr:.6f}")

            # Calculate average loss for epoch
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                losses.append(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, No valid batches.")

        return losses

    def save_model(self, path="models/cnn_text_model"):
        """Save the hybrid model to disk"""
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model components
        state_dict = {
            'base_model': self.base_model.state_dict(),
            'cnn_layers': [layer.state_dict() for layer in self.cnn_layers_list],
            'adapter': self.adapter.state_dict(),
            'cnn_config': {
                'cnn_layers': self.cnn_layers,
                'cnn_kernel_sizes': self.cnn_kernel_sizes,
                'cnn_dropout': self.cnn_dropout
            }
        }

        torch.save(state_dict, path)

        # Save tokenizer separately
        tokenizer_path = os.path.join(os.path.dirname(path), "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_path)

        print(f"Model saved to {path}")
        print(f"Tokenizer saved to {tokenizer_path}")

def create_cnn_text_generator(model_name="google/flan-ul2-20b", force_gpu=True, cnn_layers=3,
                             gpu_type="A6000", vram_size=48,
                             load_in_4bit=True, load_in_8bit=False,
                             quantization_config=None, use_flash_attention_2=True,
                             gradient_checkpointing=True, lora_rank=32, lora_alpha=64,
                             lora_dropout=0.05, batch_size=None, gradient_accumulation_steps=None,
                             max_length=4096, num_workers=None):
    """
    Helper function to create a CNN-enhanced text generator optimized for A6000 GPUs

    Args:
        model_name: Base model name or path
        force_gpu: Whether to force GPU usage
        cnn_layers: Number of CNN layers to use
        gpu_type: GPU type (A6000, A4000, RTX5000)
        vram_size: GPU VRAM size in GiB
        load_in_4bit: Whether to load model in 4-bit precision
        load_in_8bit: Whether to load model in 8-bit precision
        quantization_config: Configuration for 4-bit or 8-bit quantization
        use_flash_attention_2: Whether to use Flash Attention 2
        gradient_checkpointing: Whether to use gradient checkpointing
        lora_rank: LoRA rank parameter for fine-tuning
        lora_alpha: LoRA alpha parameter for fine-tuning
        lora_dropout: LoRA dropout parameter for fine-tuning
        batch_size: Batch size for training (if None, will be set based on GPU)
        gradient_accumulation_steps: Steps to accumulate gradients (if None, will be set based on GPU)
        max_length: Maximum sequence length for training
        num_workers: Number of workers for data loading (if None, will be set based on GPU)

    Returns:
        Initialized CNNTextGenerator optimized for the specified GPU
    """
    # Check if model is GPT2 - Flash Attention 2 is not supported for GPT2
    is_gpt2_model = "gpt2" in model_name.lower()

    # Disable Flash Attention 2 for GPT2 models
    if is_gpt2_model and use_flash_attention_2:
        print("Flash Attention 2 is not supported for GPT2 models - disabling")
        use_flash_attention_2 = False

    # Determine if we should use bfloat16 based on model type and GPU capabilities
    use_bf16 = False
    if torch.cuda.is_available():
        try:
            # Check if GPU supports bfloat16 (Ampere or newer architecture)
            if torch.cuda.get_device_capability()[0] >= 8:
                use_bf16 = not is_gpt2_model  # bfloat16 not well supported for GPT2
                print(f"GPU supports bfloat16: {'Yes' if use_bf16 else 'No'}")
            else:
                print("GPU does not support bfloat16, using float16 instead")
        except Exception as e:
            print(f"Error checking GPU capabilities: {e}. Using float16 instead.")

    return CNNTextGenerator(
        model_name_or_path=model_name,
        force_gpu=force_gpu,
        cnn_layers=cnn_layers,
        cnn_kernel_sizes=[3, 5, 7][:cnn_layers],
        cnn_dropout=0.1,
        gpu_type=gpu_type,
        vram_size=vram_size,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
        use_flash_attention_2=use_flash_attention_2,
        gradient_checkpointing=gradient_checkpointing,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        bf16=use_bf16,
        num_workers=num_workers
    )
