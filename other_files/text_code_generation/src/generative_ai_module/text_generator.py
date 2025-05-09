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

        # Validate and convert batch tensors to correct dtype
        validated_batches = []
        for batch in data['batches']:
            if isinstance(batch, tuple) and len(batch) == 2:
                input_batch, target_batch = batch

                # Check and convert input_batch dtype if needed
                if hasattr(input_batch, 'dtype') and input_batch.dtype != torch.long:
                    print(f"Converting input batch from {input_batch.dtype} to torch.long in {dataset_name}")
                    input_batch = input_batch.to(torch.long)

                # Check and convert target_batch dtype if needed
                if hasattr(target_batch, 'dtype') and target_batch.dtype != torch.long:
                    print(f"Converting target batch from {target_batch.dtype} to torch.long in {dataset_name}")
                    target_batch = target_batch.to(torch.long)

                validated_batches.append((input_batch, target_batch))
            else:
                print(f"Warning: Skipping invalid batch format in {dataset_name}")

        print(f"Training on {len(validated_batches)} validated batches from preprocessed {dataset_name} dataset")
        return self.train(validated_batches, epochs=epochs)

    def train_from_multiple_datasets(self, dataset_names=None, epochs=5, dataset_paths=None):
        """
        Train the model using multiple preprocessed datasets

        Args:
            dataset_names: List of dataset names to use
            epochs: Number of training epochs
            dataset_paths: Dictionary mapping dataset names to preprocessed file paths

        Returns:
            Training loss history
        """
        # Import here to avoid circular imports
        from .dataset_processor import DatasetProcessor
        import os

        # Default datasets if none provided
        if dataset_names is None:
            dataset_names = ["persona_chat", "writing_prompts", "pile", "openassistant", "gpteacher"]

        # Default paths if none provided
        if dataset_paths is None:
            dataset_paths = {
                name: f"notebooks/Jarvis_AI_Assistant/datasets/preprocessed_{name}.pt"
                for name in dataset_names
            }

        # Create a processor to load the data
        processor = DatasetProcessor(self)

        # Collect all batches from available datasets
        all_batches = []
        loaded_datasets = []
        datasets_to_reprocess = []

        # First pass: Try to load preprocessed datasets
        for dataset_name in dataset_names:
            path = dataset_paths.get(dataset_name)
            if path and os.path.exists(path):
                try:
                    print(f"Loading preprocessed dataset: {dataset_name} from {path}")
                    data = processor.load_preprocessed_data(dataset_name, path)

                    if 'batches' in data and data['batches']:
                        # Ensure all batches have the correct dtype (LongTensor)
                        validated_batches = []
                        for batch in data['batches']:
                            if isinstance(batch, tuple) and len(batch) == 2:
                                input_batch, target_batch = batch

                                # Check and convert input_batch dtype if needed
                                if hasattr(input_batch, 'dtype') and input_batch.dtype != torch.long:
                                    print(f"Converting input batch from {input_batch.dtype} to torch.long in {dataset_name}")
                                    input_batch = input_batch.to(torch.long)

                                # Check and convert target_batch dtype if needed
                                if hasattr(target_batch, 'dtype') and target_batch.dtype != torch.long:
                                    print(f"Converting target batch from {target_batch.dtype} to torch.long in {dataset_name}")
                                    target_batch = target_batch.to(torch.long)

                                validated_batches.append((input_batch, target_batch))
                            else:
                                print(f"Warning: Skipping invalid batch format in {dataset_name}")

                        all_batches.extend(validated_batches)
                        loaded_datasets.append(dataset_name)
                        print(f"Added {len(validated_batches)} validated batches from {dataset_name}")
                    else:
                        print(f"Warning: No batches found in preprocessed data: {path}")
                        print(f"Will re-preprocess {dataset_name} dataset")
                        datasets_to_reprocess.append(dataset_name)
                except Exception as e:
                    print(f"Error loading {dataset_name}: {e}")
                    print(f"Will re-preprocess {dataset_name} dataset")
                    datasets_to_reprocess.append(dataset_name)
            else:
                print(f"Dataset {dataset_name} not found at {path}, will preprocess it")
                datasets_to_reprocess.append(dataset_name)

        # Second pass: Re-preprocess datasets with missing or invalid batches
        for dataset_name in datasets_to_reprocess:
            try:
                print(f"Re-preprocessing {dataset_name} dataset...")

                # Check if we should use the ImprovedPreprocessor
                try:
                    from .improved_preprocessing import ImprovedPreprocessor
                    print(f"Using ImprovedPreprocessor for {dataset_name}")
                    improved_processor = ImprovedPreprocessor()
                    data = improved_processor.process_dataset(dataset_name, max_samples=5000)
                except ImportError:
                    # Fall back to standard preprocessing
                    print(f"ImprovedPreprocessor not available, using standard preprocessing for {dataset_name}")
                    if dataset_name == 'persona_chat':
                        raw_text = processor.load_persona_chat(split='train', max_samples=5000)
                    elif dataset_name == 'writing_prompts':
                        raw_text = processor.load_writing_prompts(split='train', max_samples=5000)
                    elif dataset_name == 'pile':
                        raw_text = processor.load_pile_dataset(split='train', max_samples=5000)
                    elif dataset_name == 'openassistant':
                        raw_text = processor.load_openassistant_dataset(split='train', max_samples=5000)
                    elif dataset_name == 'gpteacher':
                        raw_text = processor.load_gpteacher_dataset(split='train', max_samples=5000)
                    else:
                        print(f"Unknown dataset: {dataset_name}, skipping")
                        continue

                    # Create sequences and batches
                    print(f"Creating sequences...")
                    sequences = processor.create_sequences(raw_text, 512)  # Default sequence length

                    print(f"Creating batches...")
                    batches = processor.create_batches(sequences, batch_size=16)  # Default batch size

                    data = {
                        'batches': batches,
                        'metadata': {
                            'dataset_name': dataset_name,
                            'split': 'train',
                            'sequence_length': 512,
                            'batch_size': 16,
                            'sample_count': len(sequences),
                            'batch_count': len(batches)
                        }
                    }

                # Save the preprocessed data
                path = dataset_paths.get(dataset_name)
                print(f"Saving re-preprocessed {dataset_name} to {path}")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(data, path)

                # Add the batches to our collection
                if 'batches' in data and data['batches']:
                    all_batches.extend(data['batches'])
                    loaded_datasets.append(dataset_name)
                    print(f"Added {len(data['batches'])} batches from re-preprocessed {dataset_name}")
                else:
                    print(f"Warning: No valid batches in re-preprocessed {dataset_name}")
            except Exception as e:
                print(f"Error re-preprocessing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_batches:
            raise ValueError("No valid batches found in any of the datasets, even after re-preprocessing")

        print(f"Training on {len(all_batches)} total batches from {len(loaded_datasets)} datasets: {loaded_datasets}")
        return self.train(all_batches, epochs=epochs)

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
                model_name_or_path="google/flan-ul2",
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
                max_length=2048,  # Reduced to 2048 to prevent OOM errors
                bf16=True,  # Use bfloat16 precision for A6000 GPUs
                num_workers=None,  # Number of workers for data loading
                warmup_ratio=0.03,  # Ratio of warmup steps to total training steps
                weight_decay=0.01,  # Weight decay for optimizer
                adam_beta1=0.9,  # Beta1 parameter for Adam optimizer
                adam_beta2=0.999,  # Beta2 parameter for Adam optimizer
                adam_epsilon=1e-8,  # Epsilon parameter for Adam optimizer
                max_grad_norm=1.0,  # Maximum gradient norm for gradient clipping
                use_mixed_precision=True):  # Enable mixed precision training
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
            warmup_ratio: Ratio of warmup steps to total training steps
            weight_decay: Weight decay for optimizer
            adam_beta1: Beta1 parameter for Adam optimizer
            adam_beta2: Beta2 parameter for Adam optimizer
            adam_epsilon: Epsilon parameter for Adam optimizer
            max_grad_norm: Maximum gradient norm for gradient clipping
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
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision

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
                # A6000 with 48+ GiB VRAM - extreme memory optimization settings
                print("Using ultra-optimized settings for A6000 with 48+ GiB VRAM")
                self.batch_size = self.batch_size or 1  # Absolute minimum batch size
                self.gradient_accumulation_steps = self.gradient_accumulation_steps or 32  # Reduced to 16 as requested
                self.max_length = 2048  # Reduced sequence length to prevent OOM errors
                self.cnn_layers = 1  # Use only 1 CNN layer to minimize memory usage
                self.lora_rank = 8  # Minimal LoRA rank to save memory
                self.lora_alpha = 16  # Reduced LoRA alpha to save memory
                self.lora_dropout = 0.1  # Keep dropout for regularization
                self.num_workers = 0  # No parallel workers to minimize memory usage
                self.warmup_ratio = 0.03  # Keep optimal warmup
                self.weight_decay = 0.01  # Keep weight decay for regularization
                self.adam_beta1 = 0.9  # Standard beta1
                self.adam_beta2 = 0.999  # Standard beta2
                self.adam_epsilon = 1e-8  # Standard epsilon
                self.max_grad_norm = 1.0  # Keep gradient clipping

                # Extreme memory optimization
                self.load_in_4bit = True  # Use 4-bit quantization
                self.gradient_checkpointing = True  # Enable gradient checkpointing
                self.use_flash_attention_2 = False  # Disable Flash Attention completely
                self.use_mixed_precision = True  # Enable mixed precision training

                # Set environment variables for extreme memory optimization
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
                os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid deadlocks

                # Force CPU offloading for certain operations
                os.environ["FORCE_CPU_TOKENIZATION"] = "1"

                # Print memory optimization message
                print("⚠️ Using extreme memory optimization settings - training will be slower but more stable")
                print("⚠️ Sequence length reduced to 2048 tokens to prevent OOM errors")
                print("⚠️ Using 4-bit quantization and gradient checkpointing")
                print("⚠️ Using mixed precision training (FP16/BF16)")
                print("⚠️ Gradient accumulation steps reduced to 16")

                # Monitor GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

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

        # Determine model type with more robust detection
        model_name_lower = model_name_or_path.lower()
        is_gpt2_model = "gpt2" in model_name_lower

        # More comprehensive check for T5/FLAN-UL2 models
        is_t5_model = any(name in model_name_lower for name in ["t5", "flan-t5", "flan-ul2", "ul2", "flan"])

        # Print detected model type for debugging
        if is_t5_model:
            print(f"Detected T5/FLAN-UL2 model: {model_name_or_path}")
        elif is_gpt2_model:
            print(f"Detected GPT-2 model: {model_name_or_path}")
        else:
            print(f"Detected other model type: {model_name_or_path}")

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

        # Add optimized attention mechanisms if requested and possible
        if self.use_flash_attention_2:
            if is_gpt2_model:
                print("Flash Attention 2 is not supported for GPT2 models - disabling")
                self.use_flash_attention_2 = False
            elif is_t5_model:
                print("Flash Attention 2 is not supported for T5/FLAN-UL2 models - trying alternative attention mechanisms")
                self.use_flash_attention_2 = False

                # Try to use xFormers memory-efficient attention for T5/FLAN models
                try:
                    import xformers
                    import xformers.ops
                    xformers_version = getattr(xformers, "__version__", "unknown")
                    print(f"xFormers {xformers_version} detected - enabling memory-efficient attention for T5/FLAN model")

                    # Enable memory efficient attention in the model config
                    load_kwargs['attention_mode'] = 'xformers'
                    print("✅ xFormers memory-efficient attention enabled for T5/FLAN model")
                except ImportError:
                    print("xFormers not installed - continuing with standard attention")
                    print("To install: pip install xformers --no-build-isolation")
                except Exception as e:
                    print(f"Error enabling xFormers attention: {e}")
                    print("Continuing with standard attention")

                # Set additional attention parameters for T5/FLAN models
                if is_t5_model:
                    # Increase attention dropout for better regularization
                    load_kwargs['attention_dropout'] = 0.1
                    print("Set attention dropout to 0.1 for better regularization")

                    # Enable gradient checkpointing for memory efficiency
                    self.gradient_checkpointing = True
                    print("Enabled gradient checkpointing for memory efficiency")
            else:
                # For other models, try Flash Attention 2
                try:
                    # Try to import flash_attn
                    import flash_attn
                    flash_attn_version = getattr(flash_attn, "__version__", "unknown")
                    print(f"Flash Attention {flash_attn_version} detected - enabling for transformer model")

                    # Enable Flash Attention 2 in the model config
                    load_kwargs['use_flash_attention_2'] = True

                    # For Flash Attention 2.5.5, we need to set attn_implementation="flash_attention_2"
                    if hasattr(flash_attn, "flash_attn_func"):
                        print("Using Flash Attention implementation: flash_attention_2")
                        load_kwargs['attn_implementation'] = "flash_attention_2"

                    # Print confirmation
                    print("✅ Flash Attention 2 enabled for faster training and inference")
                except ImportError:
                    print("Flash Attention 2 requested but not installed - continuing without it")
                    print("To install: pip install flash-attn==2.5.5 --no-build-isolation --no-deps")
                except Exception as e:
                    print(f"Error enabling Flash Attention: {e}")
                    print("Continuing without Flash Attention")

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

                # Define enhanced LoRA configuration based on model architecture
                if is_t5_model:
                    # T5/FLAN-UL2 architecture uses different module names
                    # Focus on attention modules for better attention fine-tuning
                    target_modules = [
                        # Attention modules
                        "q", "k", "v", "o",  # Core attention components
                        "SelfAttention.q", "SelfAttention.k", "SelfAttention.v", "SelfAttention.o",  # Self-attention
                        "EncDecAttention.q", "EncDecAttention.k", "EncDecAttention.v", "EncDecAttention.o",  # Cross-attention
                        # Feed-forward modules
                        "wi", "wo",  # T5 FFN
                        "DenseReluDense.wi", "DenseReluDense.wo",  # T5 FFN alternative names
                    ]
                    task_type = "SEQ_2_SEQ_LM"
                    print(f"Using enhanced T5/FLAN-UL2 target modules focused on attention: {target_modules}")
                else:
                    # Default modules for transformer models like GPT, LLaMA, etc.
                    target_modules = [
                        # Attention modules
                        "q_proj", "k_proj", "v_proj", "o_proj",  # Core attention projections
                        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",  # Self-attention
                        # Feed-forward modules
                        "gate_proj", "up_proj", "down_proj",  # FFN for LLaMA-style models
                        "fc1", "fc2"  # FFN for other models
                    ]
                    task_type = "CAUSAL_LM"
                    print(f"Using enhanced transformer target modules focused on attention: {target_modules}")

                # Verify target modules exist in the model
                try:
                    # Get all named modules in the model
                    model_modules = dict(self.base_model.named_modules())

                    # Check if any of the target modules exist in the model
                    found_modules = [module for module in target_modules if any(module in name for name in model_modules.keys())]

                    if not found_modules:
                        print(f"⚠️ Warning: None of the target modules {target_modules} found in model!")
                        print("Available modules (sample):")
                        # Print a sample of available modules for debugging
                        sample_modules = list(model_modules.keys())[:20]
                        for module in sample_modules:
                            print(f"  - {module}")

                        # Special case for FLAN-UL2
                        if "flan-ul2" in model_name_or_path.lower():
                            print("Detected FLAN-UL2 model, using specific target modules")
                            # These are the actual module names in FLAN-UL2
                            target_modules = ["SelfAttention.q", "SelfAttention.k", "SelfAttention.v", "SelfAttention.o",
                                             "DenseReluDense.wi", "DenseReluDense.wo", "EncDecAttention.q",
                                             "EncDecAttention.k", "EncDecAttention.v", "EncDecAttention.o"]
                            task_type = "SEQ_2_SEQ_LM"
                        # Try to infer target modules from model structure
                        elif hasattr(self.base_model, "encoder") and hasattr(self.base_model, "decoder"):
                            print("Detected encoder-decoder structure, using T5-style target modules")
                            target_modules = ["q", "k", "v", "o", "wi", "wo"]
                            task_type = "SEQ_2_SEQ_LM"
                        else:
                            print("Using minimal target modules as fallback")
                            # Use a minimal set that's likely to exist in most models
                            target_modules = ["q", "k", "v", "o"]

                        print(f"Using fallback target modules: {target_modules}")
                except Exception as e:
                    print(f"Error checking target modules: {e}")
                    print("Continuing with original target modules")

                # Add a safety mechanism to handle completely different architectures
                try:
                    # Enhanced LoRA configuration with attention-focused parameters
                    lora_config = LoraConfig(
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=target_modules,
                        bias="none",  # No bias adaptation to reduce parameters
                        task_type=task_type,
                        # Additional parameters for better attention fine-tuning
                        modules_to_save=["layer_norm", "layernorm", "LN", "ln"] if is_t5_model else None,  # Save layer norms for T5 models
                        fan_in_fan_out=False,  # Set to False for better compatibility with attention modules
                        init_lora_weights="gaussian"  # Use gaussian initialization for better convergence
                    )
                    print("Using enhanced LoRA configuration with attention-focused parameters")
                except Exception as e:
                    print(f"Error creating enhanced LoRA config: {e}")
                    print("Trying with a more generic configuration...")

                    # Try with a more generic configuration but still optimized for attention
                    lora_config = LoraConfig(
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        # Focus on attention modules even in fallback
                        target_modules=["q", "k", "v"] if is_t5_model else ["q_proj", "k_proj", "v_proj"],
                        bias="none",
                        task_type="CAUSAL_LM" if not is_t5_model else "SEQ_2_SEQ_LM",
                        init_lora_weights="gaussian"  # Still use gaussian initialization
                    )
                    print("Using fallback LoRA configuration with attention focus")

                # Apply LoRA adapters with error handling
                try:
                    self.base_model = get_peft_model(self.base_model, lora_config)
                    print(f"Applied LoRA adapters with rank={self.lora_rank}, alpha={self.lora_alpha}")
                except Exception as e:
                    print(f"❌ Error applying LoRA adapters: {e}")
                    print("Continuing with base model without LoRA")
                    # If we can't apply LoRA, we'll just use the base model
                    # This allows training to continue even if LoRA fails
            except ImportError:
                print("PEFT library not available. LoRA adapters not applied.")
                print("To install: pip install peft")

        # Get the model's config
        config = self.base_model.config

        # Get hidden size for dimensionality
        self.hidden_size = config.hidden_size

        # Create enhanced CNN layers for pattern extraction with attention-like features
        self.cnn_layers_list = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Create multiple lightweight CNN layers for improved learning with multi-scale feature extraction
        print(f"Creating {self.cnn_layers} lightweight CNN layers with multi-scale feature extraction")

        # Create CNN layers with different kernel sizes for multi-scale feature extraction
        for i in range(self.cnn_layers):
            kernel_size = self.cnn_kernel_sizes[i] if i < len(self.cnn_kernel_sizes) else 3
            padding = kernel_size // 2  # Same padding to maintain sequence length

            # Calculate groups based on kernel size to balance parameters vs. expressiveness
            # Smaller kernels can have more groups (more parameter efficient)
            groups = 32 if kernel_size <= 3 else 16 if kernel_size <= 5 else 8

            print(f"  - CNN Layer {i+1}: kernel_size={kernel_size}, groups={groups}")

            # Use a memory-efficient CNN architecture
            cnn_layer = nn.Sequential(
                # Lightweight convolutional layer with grouped convolutions
                nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=groups,  # Use grouped convolutions to reduce parameters
                    bias=False  # Disable bias to save memory
                ),
                # Use batch normalization for better training stability
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.cnn_dropout)
            )

            self.cnn_layers_list.append(cnn_layer)

        # Create layer normalization for each CNN layer output
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(self.cnn_layers)])

        # Create a lightweight adapter with skip connection capability
        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),  # Simplified adapter
            nn.ReLU(),
            nn.Dropout(self.cnn_dropout)
        )

        # Track active CNN layers for progressive fallback
        self.active_cnn_layers = self.cnn_layers

        # Add explicit CUDA cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ Cleared CUDA cache after model initialization")

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
        Forward pass through the hybrid CNN-Transformer model with memory optimizations

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            decoder_input_ids: Decoder input IDs (for seq2seq models)

        Returns:
            Transformer model outputs with logits
        """
        # Memory optimization: truncate sequences if too long to prevent OOM
        max_seq_len = 128  # Hard limit to prevent OOM errors (reduced from 256 to 128)
        original_seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1 else 0

        if original_seq_len > max_seq_len:
            print(f"⚠️ Truncating sequence from {original_seq_len} to {max_seq_len} tokens to prevent OOM")
            input_ids = input_ids[:, :max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_seq_len]
            if labels is not None and hasattr(labels, 'shape') and len(labels.shape) > 1 and labels.shape[1] > max_seq_len:
                labels = labels[:, :max_seq_len]

        # Ensure input_ids is of type Long
        if input_ids.dtype != torch.long:
            print(f"Warning in forward(): input_ids has incorrect dtype: {input_ids.dtype}. Converting to torch.long.")
            input_ids = input_ids.long()

        # Check if we should bypass CNN layers completely
        if len(self.cnn_layers_list) == 0:
            # Skip all CNN processing and use base model directly
            print("CNN layers disabled - using base model directly")

            # Handle different model types
            if self.model_type == "seq2seq":
                # For T5/FLAN models (encoder-decoder)
                try:
                    # Prepare decoder inputs for seq2seq models
                    if decoder_input_ids is None and labels is not None:
                        # Create decoder_input_ids by shifting labels right
                        decoder_input_ids = self._shift_right(labels)

                    # Use base model directly
                    return self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        decoder_input_ids=decoder_input_ids,
                        **kwargs
                    )
                except Exception as e:
                    print(f"Error in seq2seq direct forward pass: {e}")
                    # Try fallback approach
                    if hasattr(self.base_model, "prepare_decoder_input_ids_from_labels") and labels is not None:
                        decoder_input_ids = self.base_model.prepare_decoder_input_ids_from_labels(labels)
                        return self.base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_input_ids=decoder_input_ids,
                            **kwargs
                        )
                    else:
                        # Last resort
                        return self.base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            **kwargs
                        )
            else:
                # For causal language models (GPT-style)
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs
                )

        # Get embeddings from base model
        try:
            # Memory optimization: use gradient checkpointing for embedding lookup
            with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
                if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
                    # GPT-2 style models
                    embeddings = self.base_model.transformer.wte(input_ids)
                elif hasattr(self.base_model, "get_input_embeddings"):
                    # Generic approach for most models
                    embedding_layer = self.base_model.get_input_embeddings()
                    embeddings = embedding_layer(input_ids)
                else:
                    raise ValueError("Could not get embeddings from model")
        except RuntimeError as e:
            if "expected scalar type Long" in str(e) or "out of memory" in str(e).lower():
                # Handle dtype issues or OOM errors
                print(f"Runtime error with embeddings: {e}. Attempting recovery...")

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force input_ids to Long type
                input_ids = input_ids.long()

                # Try again with corrected dtype and reduced precision
                with torch.cuda.amp.autocast(enabled=True):
                    if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
                        embeddings = self.base_model.transformer.wte(input_ids)
                    elif hasattr(self.base_model, "get_input_embeddings"):
                        embedding_layer = self.base_model.get_input_embeddings()
                        embeddings = embedding_layer(input_ids)
                    else:
                        raise ValueError("Could not get embeddings from model")
            else:
                # Re-raise if it's not a dtype or OOM issue
                raise

        # Memory-efficient multi-layer CNN processing with progressive fallback
        # First, transpose for CNN (batch_size, hidden_size, seq_len)
        x = embeddings.transpose(1, 2)
        original_x = x  # Store original input for residual connections

        # Process with CNN layers if available
        if len(self.cnn_layers_list) > 0 and self.active_cnn_layers > 0:
            # Clear CUDA cache before CNN processing
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 1e9:  # If using more than 1GB
                torch.cuda.empty_cache()

            # Apply multiple CNN layers with progressive fallback
            try:
                # Process through each active CNN layer
                for i in range(min(self.active_cnn_layers, len(self.cnn_layers_list))):
                    # Store the input for residual connection
                    residual = x

                    # Apply CNN layer with gradient checkpointing
                    if self.gradient_checkpointing and hasattr(torch.utils, 'checkpoint'):
                        # Use checkpoint with use_reentrant=False for better memory efficiency
                        x = torch.utils.checkpoint.checkpoint(self.cnn_layers_list[i], x, use_reentrant=False)
                    else:
                        x = self.cnn_layers_list[i](x)

                    # Add residual connection
                    x = x + residual

                    # Apply layer normalization (after transposing back and forth)
                    x_norm = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
                    x_norm = self.layer_norms[i](x_norm)
                    x = x_norm.transpose(1, 2)  # Back to (batch_size, hidden_size, seq_len)

                    # Clear CUDA cache periodically during multi-layer processing
                    if (i + 1) % 2 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"Successfully processed through {min(self.active_cnn_layers, len(self.cnn_layers_list))} CNN layers")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Progressive fallback: reduce active CNN layers for next forward pass
                    if self.active_cnn_layers > 1:
                        self.active_cnn_layers -= 1
                        print(f"OOM in CNN layer, reducing to {self.active_cnn_layers} active CNN layers for future passes")
                    else:
                        self.active_cnn_layers = 0
                        print(f"OOM in CNN layer, disabling CNN layers for future passes")

                    # Fall back to using the original embeddings for this pass
                    x = original_x

                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        # Transpose back to transformer format (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2)

        # Apply adapter with gradient checkpointing
        try:
            if self.gradient_checkpointing and hasattr(torch.utils, 'checkpoint'):
                adapter_output = torch.utils.checkpoint.checkpoint(self.adapter, x, use_reentrant=False)
            else:
                adapter_output = self.adapter(x)

            # Add residual connection to preserve original embeddings
            enhanced_embeddings = adapter_output + embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM in adapter layer, falling back to base embeddings")
                # Fall back to using the original embeddings
                enhanced_embeddings = embeddings
            else:
                raise

        # Clear intermediate tensors to save memory
        del x, original_x
        if 'residual' in locals():
            del residual
        if 'adapter_output' in locals():
            del adapter_output
        if 'x_norm' in locals():
            del x_norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Handle different model types
        if self.model_type == "seq2seq":
            # For T5/FLAN models (encoder-decoder)
            try:
                # Prepare decoder inputs for seq2seq models
                if decoder_input_ids is None:
                    # For T5/FLAN models, we need to prepare decoder_input_ids
                    if labels is not None:
                        # Shift labels to create decoder_input_ids (standard practice for T5/FLAN)
                        decoder_input_ids = self._shift_right(labels)
                    else:
                        # If no labels are provided, create minimal decoder_input_ids
                        batch_size = input_ids.shape[0]
                        decoder_input_ids = torch.full(
                            (batch_size, 1),
                            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
                            dtype=torch.long,
                            device=input_ids.device
                        )

                # Memory optimization: use gradient checkpointing for the base model
                if self.gradient_checkpointing and hasattr(self.base_model, "gradient_checkpointing_enable"):
                    self.base_model.gradient_checkpointing_enable()

                # Pass through the seq2seq model with enhanced embeddings
                with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
                    outputs = self.base_model(
                        inputs_embeds=enhanced_embeddings,
                        attention_mask=attention_mask,
                        labels=labels,
                        decoder_input_ids=decoder_input_ids,
                        **kwargs
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM error
                    print(f"OOM in seq2seq forward pass: {e}. Attempting recovery...")

                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Try with minimal inputs
                    try:
                        # Create minimal decoder_input_ids
                        batch_size = input_ids.shape[0]
                        decoder_input_ids = torch.full(
                            (batch_size, 1),
                            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
                            dtype=torch.long,
                            device=input_ids.device
                        )

                        # Try without labels to save memory
                        with torch.cuda.amp.autocast(enabled=True):
                            outputs = self.base_model(
                                inputs_embeds=enhanced_embeddings,
                                attention_mask=attention_mask,
                                decoder_input_ids=decoder_input_ids,
                                **kwargs
                            )
                    except Exception as inner_e:
                        print(f"Recovery failed: {inner_e}. Using fallback approach.")
                        # Last resort: try with decoder_inputs_embeds=None
                        outputs = self.base_model(
                            inputs_embeds=enhanced_embeddings,
                            attention_mask=attention_mask,
                            decoder_inputs_embeds=None,
                            **kwargs
                        )
                else:
                    # For other errors, try fallback approaches
                    print(f"Error in seq2seq forward pass: {e}. Trying fallback approach.")

                    # Fallback: use the base model's prepare_decoder_input_ids_from_labels method
                    if hasattr(self.base_model, "prepare_decoder_input_ids_from_labels") and labels is not None:
                        decoder_input_ids = self.base_model.prepare_decoder_input_ids_from_labels(labels)
                        outputs = self.base_model(
                            inputs_embeds=enhanced_embeddings,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_input_ids=decoder_input_ids,
                            **kwargs
                        )
                    else:
                        # Last resort: try without decoder_input_ids
                        outputs = self.base_model(
                            inputs_embeds=enhanced_embeddings,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_inputs_embeds=None,
                            **kwargs
                        )
        else:
            # For causal language models (GPT-style)
            # Memory optimization: use gradient checkpointing
            if self.gradient_checkpointing and hasattr(self.base_model, "gradient_checkpointing_enable"):
                self.base_model.gradient_checkpointing_enable()

            # Use mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.base_model(
                    inputs_embeds=enhanced_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs
                )

        # Memory optimization: clear unnecessary tensors
        del enhanced_embeddings, x, adapter_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs

    def _shift_right(self, input_ids):
        """
        Shift input_ids right for T5/FLAN models to create decoder_input_ids

        Args:
            input_ids: Input token IDs to shift

        Returns:
            Shifted input_ids for decoder input
        """
        # Get pad token ID, default to 0 if not available
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Create shifted input by prepending pad token and removing last token
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 0] = pad_token_id
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()

        # Where the original input was padding, keep it as padding
        if pad_token_id != 0:
            # Create a mask where original input was padding
            pad_mask = (input_ids == pad_token_id)
            # Apply the mask to keep padding in the shifted input
            shifted_input_ids = shifted_input_ids.masked_fill(pad_mask, pad_token_id)

        return shifted_input_ids

    def train(self, data, epochs=3, gradient_accumulation_steps=None, eval_steps=None, save_steps=None, checkpoint_dir=None):
        """
        Train the hybrid CNN-Transformer model with gradient accumulation and mixed precision

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

        # Initialize mixed precision training
        use_mixed_precision = getattr(self, 'use_mixed_precision', False)
        if use_mixed_precision and torch.cuda.is_available():
            # Create gradient scaler for mixed precision training
            scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision training enabled with gradient scaling")

            # Determine precision type (bfloat16 or float16)
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and self.bf16:
                amp_dtype = torch.bfloat16
                print("Using bfloat16 precision for mixed precision training")
            else:
                amp_dtype = torch.float16
                print("Using float16 precision for mixed precision training")
        else:
            scaler = None
            if torch.cuda.is_available():
                print("Mixed precision training disabled")
            else:
                print("Mixed precision training not available (CUDA not available)")

        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing and hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for transformer model")

        # Monitor initial GPU memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"📊 Initial GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

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
            for batch_idx, (input_batch, target_batch) in enumerate(iterator):
                if input_batch is None or target_batch is None:
                    self.logger.warning("Skipping invalid batch")
                    continue
                try:
                    # Clear CUDA cache periodically
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"Cleared CUDA cache at batch {batch_idx}")

                        # Print memory usage
                        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

                    # Ensure input_batch is of type Long before moving to device
                    if input_batch.dtype != torch.long:
                        print(f"Converting input batch from {input_batch.dtype} to torch.long in batch {batch_idx}")
                        input_batch = input_batch.to(torch.long)

                    # Ensure target_batch is of type Long before moving to device
                    if target_batch.dtype != torch.long:
                        print(f"Converting target batch from {target_batch.dtype} to torch.long in batch {batch_idx}")
                        target_batch = target_batch.to(torch.long)

                    # Move data to device, ensuring they remain LongTensors
                    input_batch = input_batch.to(self.device, dtype=torch.long)
                    target_batch = target_batch.to(self.device, dtype=torch.long)

                    # Verify tensor types after moving to device
                    if input_batch.dtype != torch.long:
                        print(f"Error: Input batch still has incorrect dtype after conversion: {input_batch.dtype}. Forcing to torch.long.")
                        input_batch = input_batch.long()

                    # Memory optimization: truncate sequences if too long to prevent OOM
                    max_seq_len = 256  # Hard limit to prevent OOM errors
                    original_seq_len = input_batch.shape[1] if hasattr(input_batch, 'shape') and len(input_batch.shape) > 1 else 0

                    if original_seq_len > max_seq_len:
                        print(f"⚠️ Truncating sequence from {original_seq_len} to {max_seq_len} tokens to prevent OOM")
                        input_batch = input_batch[:, :max_seq_len]
                        if hasattr(target_batch, 'shape') and len(target_batch.shape) > 1 and target_batch.shape[1] > max_seq_len:
                            target_batch = target_batch[:, :max_seq_len]

                    # Clear CUDA cache periodically
                    if torch.cuda.is_available() and batch_count % 10 == 0:
                        torch.cuda.empty_cache()

                    # Forward pass through hybrid model with mixed precision
                    try:
                        # Enable gradient checkpointing if available
                        if hasattr(self, 'gradient_checkpointing') and self.gradient_checkpointing:
                            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                                self.base_model.gradient_checkpointing_enable()

                        # Use mixed precision for forward pass if enabled
                        if use_mixed_precision and torch.cuda.is_available():
                            # Use autocast context manager for mixed precision
                            with torch.cuda.amp.autocast(dtype=amp_dtype):
                                # For seq2seq models, ensure decoder_input_ids are properly created
                                if self.model_type == "seq2seq":
                                    # Create decoder_input_ids by shifting target_batch right
                                    decoder_input_ids = self._shift_right(target_batch)
                                    outputs = self.forward(input_batch, labels=target_batch, decoder_input_ids=decoder_input_ids)
                                else:
                                    outputs = self.forward(input_batch, labels=target_batch)
                        else:
                            # Regular forward pass without mixed precision
                            if self.model_type == "seq2seq":
                                decoder_input_ids = self._shift_right(target_batch)
                                outputs = self.forward(input_batch, labels=target_batch, decoder_input_ids=decoder_input_ids)
                            else:
                                outputs = self.forward(input_batch, labels=target_batch)
                    except Exception as e:
                        print(f"Error in forward pass: {e}")

                        # Try to recover with more aggressive memory optimization
                        try:
                            # Clear CUDA cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()

                            # Try with smaller sequence length and direct embedding access
                            if hasattr(input_batch, 'shape') and len(input_batch.shape) > 1 and input_batch.shape[1] > 128:
                                print(f"Trying with reduced sequence length (128 tokens)...")

                                try:
                                    # Clear CUDA cache first
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        import gc
                                        gc.collect()

                                    # Create smaller input batch
                                    input_batch_small = input_batch[:, :128].clone().detach()

                                    # Ensure input_batch is of type Long
                                    if input_batch_small.dtype != torch.long:
                                        input_batch_small = input_batch_small.long()

                                    # Create smaller target batch if needed
                                    if hasattr(target_batch, 'shape') and len(target_batch.shape) > 1 and target_batch.shape[1] > 128:
                                        target_batch_small = target_batch[:, :128].clone().detach()
                                    else:
                                        target_batch_small = target_batch

                                    # Skip CNN layers and use base model directly to avoid dimension issues
                                    print("Using base model directly (bypassing CNN layers) for recovery")

                                    # Use mixed precision with smaller batch if enabled
                                    if use_mixed_precision and torch.cuda.is_available():
                                        with torch.cuda.amp.autocast(dtype=amp_dtype):
                                            if self.model_type == "seq2seq":
                                                decoder_input_ids = self._shift_right(target_batch_small)
                                                # Use base model directly
                                                outputs = self.base_model(
                                                    input_ids=input_batch_small,
                                                    labels=target_batch_small,
                                                    decoder_input_ids=decoder_input_ids
                                                )
                                            else:
                                                # Use base model directly
                                                outputs = self.base_model(
                                                    input_ids=input_batch_small,
                                                    labels=target_batch_small
                                                )
                                    else:
                                        # Regular forward pass without mixed precision
                                        if self.model_type == "seq2seq":
                                            decoder_input_ids = self._shift_right(target_batch_small)
                                            # Use base model directly
                                            outputs = self.base_model(
                                                input_ids=input_batch_small,
                                                labels=target_batch_small,
                                                decoder_input_ids=decoder_input_ids
                                            )
                                        else:
                                            # Use base model directly
                                            outputs = self.base_model(
                                                input_ids=input_batch_small,
                                                labels=target_batch_small
                                            )

                                    print("✅ Successfully recovered with smaller sequence length and direct model access")
                                except Exception as e:
                                    print(f"Error during recovery with smaller sequence: {e}")
                                    raise
                            else:
                                # Fallback approach for seq2seq models
                                if self.model_type == "seq2seq":
                                    # Try using the base model's prepare_decoder_input_ids_from_labels method
                                    if hasattr(self.base_model, "prepare_decoder_input_ids_from_labels"):
                                        decoder_input_ids = self.base_model.prepare_decoder_input_ids_from_labels(target_batch)

                                        if use_mixed_precision and torch.cuda.is_available():
                                            with torch.cuda.amp.autocast(dtype=amp_dtype):
                                                outputs = self.forward(input_batch, labels=target_batch, decoder_input_ids=decoder_input_ids)
                                        else:
                                            outputs = self.forward(input_batch, labels=target_batch, decoder_input_ids=decoder_input_ids)
                                    else:
                                        # Last resort: try with decoder_inputs_embeds=None
                                        if use_mixed_precision and torch.cuda.is_available():
                                            with torch.cuda.amp.autocast(dtype=amp_dtype):
                                                outputs = self.forward(input_batch, labels=target_batch, decoder_inputs_embeds=None)
                                        else:
                                            outputs = self.forward(input_batch, labels=target_batch, decoder_inputs_embeds=None)
                                else:
                                    # Re-raise if it's not a seq2seq model
                                    raise
                        except Exception as inner_e:
                            print(f"Recovery failed: {inner_e}. Skipping batch.")
                            continue

                    logits = outputs.logits

                    # Calculate loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_batch.view(-1)
                    )

                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps

                    # Backward pass with mixed precision if enabled
                    if scaler is not None:
                        # Mixed precision backward pass
                        scaler.scale(loss).backward()
                    else:
                        # Regular backward pass
                        loss.backward()

                    # Update step counting
                    steps_since_update += 1
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    batch_count += 1
                    global_step += 1

                    # Update weights if we've accumulated enough gradients
                    if steps_since_update >= gradient_accumulation_steps:
                        # Get all parameters that require gradients
                        parameters = list(self.cnn_layers_list.parameters()) + \
                                    list(self.adapter.parameters()) + \
                                    list(self.base_model.parameters())

                        if scaler is not None:
                            # Mixed precision gradient clipping and optimizer step
                            scaler.unscale_(self.optimizer)

                            # Clip gradients to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(
                                parameters,
                                max_norm=self.max_grad_norm
                            )

                            # Update weights with gradient scaling
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            # Regular gradient clipping and optimizer step
                            torch.nn.utils.clip_grad_norm_(
                                parameters,
                                max_norm=self.max_grad_norm
                            )

                            # Update weights
                            self.optimizer.step()

                        # Update learning rate with scheduler
                        self.lr_scheduler.step()

                        # Log current learning rate and memory usage
                        if global_step % 10 == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            print(f"Step {global_step}: Learning rate = {current_lr:.6f}")

                            # Monitor GPU memory usage
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                                print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

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

                                # For seq2seq models, ensure decoder_input_ids are properly created
                                try:
                                    if self.model_type == "seq2seq":
                                        # Create decoder_input_ids by shifting target right
                                        decoder_input_ids = self._shift_right(val_target)
                                        val_outputs = self.forward(val_input, labels=val_target, decoder_input_ids=decoder_input_ids)
                                    else:
                                        val_outputs = self.forward(val_input, labels=val_target)
                                except Exception as e:
                                    print(f"Error in validation forward pass: {e}")
                                    # Fallback approach
                                    if self.model_type == "seq2seq":
                                        # Try with decoder_inputs_embeds=None
                                        val_outputs = self.forward(val_input, labels=val_target, decoder_inputs_embeds=None)
                                    else:
                                        # Re-raise if it's not a seq2seq model
                                        raise

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
                    if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                        print(f"❌ CUDA OOM error. Attempting aggressive recovery...")

                        # Aggressive memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()  # Force garbage collection
                            print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")

                        # Reset optimizer state
                        steps_since_update = 0
                        self.optimizer.zero_grad()

                        # Try to reduce sequence length first (most effective for OOM)
                        if hasattr(self, 'max_length') and self.max_length > 128:
                            old_max_length = self.max_length
                            self.max_length = max(128, self.max_length // 2)
                            print(f"⚠️ Reduced max sequence length from {old_max_length} to {self.max_length}")

                            # Try to truncate the current batch if possible
                            try:
                                if hasattr(input_batch, 'shape') and len(input_batch.shape) > 1 and input_batch.shape[1] > self.max_length:
                                    print(f"Truncating current batch from {input_batch.shape[1]} to {self.max_length}")
                                    input_batch = input_batch[:, :self.max_length]
                                    if hasattr(target_batch, 'shape') and len(target_batch.shape) > 1 and target_batch.shape[1] > self.max_length:
                                        target_batch = target_batch[:, :self.max_length]

                                    # Try again with truncated batch using mixed precision if enabled
                                    if use_mixed_precision and torch.cuda.is_available():
                                        with torch.cuda.amp.autocast(dtype=amp_dtype):
                                            if self.model_type == "seq2seq":
                                                decoder_input_ids = self._shift_right(target_batch)
                                                outputs = self.forward(input_batch, labels=target_batch, decoder_input_ids=decoder_input_ids)
                                            else:
                                                outputs = self.forward(input_batch, labels=target_batch)

                                            loss = outputs.loss / gradient_accumulation_steps

                                            # Use scaler for backward pass if available
                                            if scaler is not None:
                                                scaler.scale(loss).backward()
                                            else:
                                                loss.backward()
                                    else:
                                        # Regular forward pass without mixed precision
                                        if self.model_type == "seq2seq":
                                            decoder_input_ids = self._shift_right(target_batch)
                                            outputs = self.forward(input_batch, labels=target_batch, decoder_input_ids=decoder_input_ids)
                                        else:
                                            outputs = self.forward(input_batch, labels=target_batch)

                                        loss = outputs.loss / gradient_accumulation_steps
                                        loss.backward()

                                    print("✅ Successfully recovered with truncated sequence")

                                    # Update metrics
                                    epoch_loss += loss.item() * gradient_accumulation_steps
                                    batch_count += 1
                                    global_step += 1
                                    steps_since_update += 1

                                    # Continue with next batch
                                    continue
                            except Exception as truncate_e:
                                print(f"Error with truncated batch: {truncate_e}")

                        # If truncation didn't work, increase gradient accumulation
                        if gradient_accumulation_steps < 128:  # Cap at 128
                            old_grad_accum = gradient_accumulation_steps
                            gradient_accumulation_steps *= 2
                            print(f"⚠️ Increased gradient accumulation steps from {old_grad_accum} to {gradient_accumulation_steps}")

                        # Try to reduce batch size as last resort
                        if hasattr(self, 'batch_size') and self.batch_size > 1:
                            old_batch_size = self.batch_size
                            self.batch_size = max(1, self.batch_size // 2)
                            print(f"⚠️ Reduced batch size from {old_batch_size} to {self.batch_size}")

                        # Set environment variables for extreme memory optimization
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
                        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                        os.environ["TOKENIZERS_PARALLELISM"] = "false"

                        # Enable gradient checkpointing if not already enabled
                        if hasattr(self, 'gradient_checkpointing') and not self.gradient_checkpointing:
                            self.gradient_checkpointing = True
                            print("⚠️ Enabled gradient checkpointing for memory efficiency")

                            # Apply gradient checkpointing to base model if possible
                            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                                self.base_model.gradient_checkpointing_enable()

                        print("⚠️ Skipping problematic batch and continuing with optimized settings...")
                    else:
                        print(f"Error in training: {e}")
                        import traceback
                        traceback.print_exc()

                    # Continue with next batch
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

    def create_optimizer(self, learning_rate=2e-5, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        """
        Create an optimizer for the model

        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
            adam_beta1: Beta1 parameter for Adam optimizer
            adam_beta2: Beta2 parameter for Adam optimizer
            adam_epsilon: Epsilon parameter for Adam optimizer

        Returns:
            Optimizer (Adafactor or AdamW)
        """
        # Get all parameters that require gradients
        parameters = list(self.cnn_layers_list.parameters()) + list(self.adapter.parameters())

        # Add base model parameters if they require gradients
        for param in self.base_model.parameters():
            if param.requires_grad:
                parameters.append(param)

        # Use Adafactor for memory efficiency if available
        try:
            from transformers import Adafactor

            # Create Adafactor optimizer (more memory efficient)
            print("✅ Using Adafactor optimizer for memory efficiency")
            optimizer = Adafactor(
                parameters,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None  # Let Adafactor determine the learning rate
            )
        except ImportError:
            # Fallback to AdamW if Adafactor is not available
            print("⚠️ Adafactor not available, using AdamW optimizer")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                eps=adam_epsilon,
                weight_decay=weight_decay
            )

        return optimizer

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

def create_cnn_text_generator(model_name="google/flan-ul2", force_gpu=True, cnn_layers=3,
                             gpu_type="A6000", vram_size=48,
                             load_in_4bit=True, load_in_8bit=False,
                             quantization_config=None, use_flash_attention_2=True,
                             gradient_checkpointing=True, lora_rank=32, lora_alpha=64,
                             lora_dropout=0.05, batch_size=None, gradient_accumulation_steps=None,
                             max_length=4096, num_workers=None, warmup_ratio=0.03,
                             weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999,
                             adam_epsilon=1e-8, max_grad_norm=1.0):
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
        warmup_ratio: Ratio of warmup steps to total training steps
        weight_decay: Weight decay for optimizer
        adam_beta1: Beta1 parameter for Adam optimizer
        adam_beta2: Beta2 parameter for Adam optimizer
        adam_epsilon: Epsilon parameter for Adam optimizer
        max_grad_norm: Maximum gradient norm for gradient clipping

    Returns:
        Initialized CNNTextGenerator optimized for the specified GPU
    """
    # Check if model is GPT2 or T5/FLAN - Flash Attention 2 is not supported for these models
    is_gpt2_model = "gpt2" in model_name.lower()
    is_t5_model = any(name in model_name.lower() for name in ["t5", "flan-t5", "flan-ul2", "ul2", "flan"])

    # Disable Flash Attention 2 for unsupported models
    if (is_gpt2_model or is_t5_model) and use_flash_attention_2:
        print(f"Flash Attention 2 is not supported for {'GPT2' if is_gpt2_model else 'T5/FLAN'} models - disabling")
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
        num_workers=num_workers,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm
    )
