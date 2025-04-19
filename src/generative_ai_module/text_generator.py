import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import os

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
        # Handle different input shapes:
        # x shape can be either:
        # [batch_size, seq_len, input_size] (3D tensor) - one-hot encoded
        # [batch_size, seq_len] (2D tensor) - token indices
        # [batch_size, input_size] (2D tensor) - single time step one-hot

        # Process input based on dimensionality and dtype
        if x.dim() == 2:
            if x.dtype in [torch.long, torch.int64]:
                # Input is token indices [batch_size, seq_len]
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

        # Pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Get output from last time step only
        last_output = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out
        # Apply dropout and linear layer
        last_output = self.dropout(last_output)
        output = self.fc(last_output)

        return output, hidden
    
class TextGenerator:
    def __init__(self, force_gpu=False):
        self.all_chars = string.printable
        self.n_chars = len(self.all_chars)
        self.char_to_index = {char: i for i, char in enumerate(self.all_chars)}
        self.index_to_char = {i: char for char, i in self.char_to_index.items()}
        self.unknown_token = "<UNK>"
        self.all_chars += self.unknown_token
        self.char_to_index[self.unknown_token] = len(self.all_chars) - 1
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
                zip_contents = process_zip(input_data)
                return " ".join(zip_contents)
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
                
    def train(self, data, epochs=10):
        """
        Train the model on batched data
        
        Args:
            data: List of (input_batch, target_batch) tuples
            epochs: Number of epochs to train
            
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
            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                
                for input_batch, target_batch in data:
                    try:
                        # Move to device
                        input_batch = input_batch.to(self.device)
                        target_batch = target_batch.to(self.device)
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        outputs, _ = self.model(input_batch)
                        
                        # Calculate loss
                        loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
                        
                        # Backward pass and optimize
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                    except Exception as e:
                        print(f"Error during batch training: {e}")
                        continue
                
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
                