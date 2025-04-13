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
            if x.dtype == torch.long or x.dtype == torch.int64:
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
        if lstm_out.dim() == 3:
            # For 3D output: [batch_size, seq_len, hidden_size]
            last_output = lstm_out[:, -1, :]
        else:
            # For 2D output: [batch_size, hidden_size]
            last_output = lstm_out
        
        # Apply dropout and linear layer
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output, hidden
    
class TextGenerator:
    def __init__(self):
        self.all_chars = string.printable
        self.n_chars = len(self.all_chars)
        self.char_to_index = {char: i for i, char in enumerate(self.all_chars)}
        self.index_to_char = {i: char for char, i in self.char_to_index.items()}
        self.unknown_token = "<UNK>"
        self.all_chars += self.unknown_token
        self.char_to_index[self.unknown_token] = len(self.all_chars) - 1
        self.model = CombinedModel(self.n_chars, 128, self.n_chars, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.context = None
        
    def char_indices_tensor(self, string):
        tensor = torch.zeros(len(string), dtype=torch.long)
        for i, char in enumerate(string):
            tensor[i] = self.char_to_index.get(char, self.char_to_index[self.unknown_token])
        
        return tensor
    
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
        device = next(self.model.parameters()).device
        
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
                    token_indices = torch.zeros(1, len(context), dtype=torch.long)
                    for t, char in enumerate(context):
                        idx = char_to_idx.get(char, char_to_idx[unknown_token])
                        token_indices[0, t] = idx
                    
                    # Move to device
                    token_indices = token_indices.to(device)
                    
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
                        device = next(self.model.parameters()).device
                        
                        # Check input dtype and format
                        if input_batch.dtype == torch.long or input_batch.dtype == torch.int64:
                            # Token indices - keep as is
                            input_batch = input_batch.to(device)
                            print(f"Batch {batch_count+1}: Using tokenized input {input_batch.shape}")
                        else:
                            # One-hot or other format - convert to float
                            # Ensure input is properly formatted
                            if input_batch.dim() == 2:
                                # If input is [batch_size, features], make sure it's float
                                input_batch = input_batch.float().to(device)
                            elif input_batch.dim() == 3:
                                # If input is [batch_size, seq_len, features], make sure it's float
                                input_batch = input_batch.float().to(device)
                            else:
                                raise ValueError(f"Unexpected input shape: {input_batch.shape}")
                            print(f"Batch {batch_count+1}: Using one-hot input {input_batch.shape}")
                        
                        # Ensure target is long for CrossEntropyLoss
                        target_batch = target_batch.long().to(device)
                        
                        # Print batch shapes for debugging (only for first few batches)
                        if batch_count < 3:
                            print(f"Batch {batch_count+1}: input {input_batch.shape}, target {target_batch.shape}")
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        output, _ = self.model(input_batch)
                        
                        # Calculate loss
                        loss = criterion(output, target_batch.squeeze())
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                        self.optimizer.step()
                        
                        # Log metrics
                        epoch_loss += loss.item()
                        batch_count += 1
                        
                    except Exception as e:
                        print(f"Error in batch: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Log epoch metrics
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    losses.append(avg_loss)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}: No valid batches processed")
            
            return losses
        
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
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
                