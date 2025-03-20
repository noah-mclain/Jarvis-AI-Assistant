import torch
import torch.nn as nn
import torch.nn.functional as F
import string

from .utils import is_zipfile, process_zip

class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CombinedModel, self).__init__()
        
        # Simple LSTM model
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get output from last time step only
        last_output = lstm_out[:, -1, :]
        
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
        n_chars = len(char_to_idx)
        unknown_token = self.unknown_token
        device = next(self.model.parameters()).device
        
        # Initial hidden state
        hidden = None
        
        # Generation loop
        with torch.no_grad():
            for _ in range(pred_len):
                # Get the last 'sequence_length' characters (or pad with spaces if not enough)
                context = predicted[-100:] if len(predicted) >= 100 else ' ' * (100 - len(predicted)) + predicted
                
                # Convert to one-hot input tensor
                input_tensor = torch.zeros(1, len(context), n_chars)
                for t, char in enumerate(context):
                    idx = char_to_idx.get(char, char_to_idx[unknown_token])
                    input_tensor[0, t, idx] = 1.0
                
                # Move to device
                input_tensor = input_tensor.to(device)
                
                # Get prediction
                output, hidden = self.model(input_tensor, hidden)
                
                # Apply temperature scaling and sample
                probs = F.softmax(output.squeeze() / temperature, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # Convert to character and append to result
                next_char = self.index_to_char.get(next_char_idx, unknown_token)
                predicted += next_char
        
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
                        input_batch = input_batch.float().to(device)  # Ensure it's float for one-hot
                        target_batch = target_batch.long().to(device)  # Ensure it's long for CE loss
                        
                        # Print batch shapes for debugging
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
                