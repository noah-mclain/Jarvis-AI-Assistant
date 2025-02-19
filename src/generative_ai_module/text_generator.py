import torch
import torch.nn as nn
import torch.nn.functional as F
import string

from .utils import is_zipfile, process_zip

class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CombinedModel, self).__init__()
        
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input, hidden):
        encoder_out, hidden = self.encoder(input, hidden)
        decoder_out, hidden = self.decoder(encoder_out, hidden)
        out = self.linear(decoder_out[:, -1, :])
        
        return out, hidden
    
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
        self.model.eval()
        
        hidden = self.context or (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
        initial_input = self.char_indices_tensor(initial_str).unsqueeze(0)
        device = next(self.model.parameters()).device
        initial_input = initial_input.to(device)
        predicted = initial_str
        
        with torch.no_grad():
            for _ in range(pred_len):
                output, hidden = self.model(initial_input, hidden)
                output_dist = F.softmax(output.squeeze() / temperature, dim=0)
                predicted_index = torch.multinomial(output_dist, 1)[0].item()
                predicted_char = self.index_to_char[predicted_index]
                predicted += predicted_char
                initial_input = self.char_indices_tensor(predicted_char).unsqueeze(0).to(device)
                
        self.context = hidden
        return predicted
                
    def train(self, text, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            hidden = None
            for char in text:
                input_tensor = self.char_indices_tensor(char).unsqueeze(0)
                output, hidden = self.model(input_tensor, hidden)
                # Add loss calculation and backprop here
                