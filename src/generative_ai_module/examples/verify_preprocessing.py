import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
from tqdm import tqdm

# Add the parent directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from generative_ai_module.text_generator import TextGenerator
from generative_ai_module.dataset_processor import DatasetProcessor

# Try to import datasets, but don't fail if not available
try:
    from datasets import load_dataset
except ImportError:
    print("Datasets library not available. Using sample data only.")
    load_dataset = None

# Simple LSTM model for character-level generation
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        lstm_out, hidden = self.lstm(embedded, hidden)
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        output = self.fc(lstm_out[:, -1, :])
        # output shape: [batch_size, vocab_size]
        return output, hidden

def get_sample_persona_chat():
    """Sample data for Persona Chat"""
    return """<PERSONA>
- I love hiking in the mountains.
- I have a dog named Max.
- I work as a software engineer.
<DIALOGUE>
USER: Hi there! Do you like outdoor activities?
ASSISTANT: Yes, I love hiking in the mountains. Do you enjoy hiking too?
USER: I do! What's your favorite hiking spot?
ASSISTANT: I really enjoy trails in the Rocky Mountains. I often take my dog Max with me.
<END>

<PERSONA>
- I am a chef at a restaurant.
- I enjoy classical music.
- I have visited 15 countries.
<DIALOGUE>
USER: What do you do for a living?
ASSISTANT: I'm a chef at a restaurant. I specialize in Italian cuisine. Do you enjoy cooking?
USER: That's cool! What's your favorite dish to prepare?
ASSISTANT: I love making homemade pasta with fresh ingredients. The process is as relaxing as listening to classical music.
<END>
"""

def get_sample_writing_prompts():
    """Sample data for Writing Prompts"""
    return """<PROMPT> You wake up one day with the ability to see 10 seconds into the future.
<STORY> I blinked rapidly, trying to make sense of what was happening. The world seemed to flicker between now and... something else. I watched as my coffee mug tipped over before it actually happened, giving me just enough time to catch it. This was going to be interesting.
<END>

<PROMPT> Aliens have been watching our TV shows for decades. They finally make contact.
<STORY> The massive ship hovered silently above New York. Everyone waited anxiously for first contact. The alien ambassador emerged and said, 'We come in peace. Also, we need to know if Ross and Rachel ever got back together. Our transmission cut out during season 8.'
<END>
"""

class CharTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self):
        # Start with just basic ASCII
        self.chars = sorted(list(string.ascii_letters + string.digits + string.punctuation + ' \t\n'))
        # Add <UNK> token
        self.chars.append('<UNK>')
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Last character index: {self.vocab_size - 1}")
    
    def encode(self, text):
        """Convert text to sequence of token IDs"""
        return [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in ids])

def load_dataset_text(dataset_name, tokenizer, split="train", max_samples=100):
    """Load and tokenize dataset text"""
    
    # Use sample data if datasets library not available
    if load_dataset is None:
        if dataset_name == "persona_chat":
            return tokenizer.encode(get_sample_persona_chat())
        else:
            return tokenizer.encode(get_sample_writing_prompts())
    
    try:
        if dataset_name == "persona_chat":
            try:
                dataset = load_dataset("google/Synthetic-Persona-Chat", split=split)
            except Exception as e:
                print(f"Error loading Persona Chat: {e}")
                return tokenizer.encode(get_sample_persona_chat())
                
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Convert to text format
            text = ""
            for item in tqdm(dataset, desc="Processing Persona Chat"):
                # Add persona
                text += "<PERSONA>\n"
                if 'user 1 personas' in item:
                    text += f"- {item['user 1 personas'].strip()}\n"
                else:
                    text += "- No persona available\n"
                    
                # Add conversation
                text += "<DIALOGUE>\n"
                if 'Best Generated Conversation' in item:
                    conv = item['Best Generated Conversation']
                    for line in conv.strip().split('\n'):
                        if line.startswith('User:'):
                            text += f"USER: {line[5:].strip()}\n"
                        elif line.startswith('Assistant:'):
                            text += f"ASSISTANT: {line[10:].strip()}\n"
                
                text += "<END>\n\n"
            
            return tokenizer.encode(text)
            
        elif dataset_name == "writing_prompts":
            try:
                dataset = load_dataset("euclaise/writingprompts", split=split)
            except Exception as e:
                print(f"Error loading Writing Prompts: {e}")
                return tokenizer.encode(get_sample_writing_prompts())
                
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Convert to text format
            text = ""
            for item in tqdm(dataset, desc="Processing Writing Prompts"):
                text += f"<PROMPT> {item['prompt'].strip()}\n"
                text += f"<STORY> {item['story'].strip()}\n"
                text += "<END>\n\n"
            
            return tokenizer.encode(text)
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return sample data as fallback
        if dataset_name == "persona_chat":
            return tokenizer.encode(get_sample_persona_chat())
        else:
            return tokenizer.encode(get_sample_writing_prompts())

def create_sequences(tokenized_text, seq_length=50):
    """Create input-target sequences for training"""
    sequences = []
    
    for i in range(0, len(tokenized_text) - seq_length):
        # Input is the current sequence
        seq_in = tokenized_text[i:i+seq_length]
        # Target is the next character
        seq_out = tokenized_text[i+seq_length]
        
        sequences.append((seq_in, seq_out))
    
    return sequences

def create_batches(sequences, batch_size=32):
    """Create batches from sequences"""
    random.shuffle(sequences)
    
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        if not batch:
            continue
            
        # Separate inputs and targets
        inputs, targets = zip(*batch)
        
        # Convert to tensors
        input_tensor = torch.tensor(inputs, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        
        batches.append((input_tensor, target_tensor))
    
    return batches

def train_model(model, data_batches, epochs=5, lr=0.001):
    """Train the model on batched data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in tqdm(data_batches, desc=f"Epoch {epoch+1}/{epochs}"):
            try:
                # Get inputs and targets
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Print epoch stats
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

def generate_text(model, tokenizer, seed_text, max_length=100, temperature=0.8):
    """Generate text using the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Tokenize the seed text
    with torch.no_grad():
        # Encode the seed text
        context_tokens = tokenizer.encode(seed_text)
        context_tensor = torch.tensor([context_tokens[-50:]], dtype=torch.long).to(device)
        
        # Generate text
        generated_text = seed_text
        
        for _ in range(max_length):
            # Get predictions
            outputs, _ = model(context_tensor)
            
            # Apply temperature and sample
            probs = F.softmax(outputs / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            next_char = tokenizer.idx_to_char[next_token]
            generated_text += next_char
            
            # Update context
            context_tokens.append(next_token)
            context_tensor = torch.tensor([context_tokens[-50:]], dtype=torch.long).to(device)
    
    return generated_text

def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Parameters
    seq_length = 50  # Shorter sequences for stability
    batch_size = 32
    epochs = 3
    
    # Train on Persona Chat
    try:
        print("\n===== Training on Persona Chat Dataset =====")
        
        # Load and tokenize data
        persona_tokens = load_dataset_text("persona_chat", tokenizer, max_samples=50)
        print(f"Loaded {len(persona_tokens)} tokens from Persona Chat dataset")
        
        # Create sequences and batches
        persona_sequences = create_sequences(persona_tokens, seq_length)
        persona_batches = create_batches(persona_sequences, batch_size)
        print(f"Created {len(persona_sequences)} sequences and {len(persona_batches)} batches")
        
        # Create and train model
        persona_model = CharLSTM(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        persona_model = train_model(persona_model, persona_batches, epochs=epochs)
        
        # Save model
        torch.save(persona_model.state_dict(), "models/persona_chat_model.pt")
        
        # Generate sample text
        seed = "<PERSONA>\n- I am a"
        generated = generate_text(persona_model, tokenizer, seed, max_length=100)
        print(f"\nGenerated Persona Chat text:\n{generated}")
        
    except Exception as e:
        print(f"Error training on Persona Chat: {e}")
        import traceback
        traceback.print_exc()
    
    # Train on Writing Prompts
    try:
        print("\n===== Training on Writing Prompts Dataset =====")
        
        # Load and tokenize data
        prompts_tokens = load_dataset_text("writing_prompts", tokenizer, max_samples=20)
        print(f"Loaded {len(prompts_tokens)} tokens from Writing Prompts dataset")
        
        # Create sequences and batches
        prompts_sequences = create_sequences(prompts_tokens, seq_length)
        prompts_batches = create_batches(prompts_sequences, batch_size)
        print(f"Created {len(prompts_sequences)} sequences and {len(prompts_batches)} batches")
        
        # Create and train model
        prompts_model = CharLSTM(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=64, 
            hidden_dim=128
        )
        
        prompts_model = train_model(prompts_model, prompts_batches, epochs=epochs)
        
        # Save model
        torch.save(prompts_model.state_dict(), "models/writing_prompts_model.pt")
        
        # Generate sample text
        seed = "<PROMPT> In a world where"
        generated = generate_text(prompts_model, tokenizer, seed, max_length=150)
        print(f"\nGenerated Writing Prompt text:\n{generated}")
        
    except Exception as e:
        print(f"Error training on Writing Prompts: {e}")
        import traceback
        traceback.print_exc()

def inspect_dataset_structure(self, dataset_name):
    """Print the structure of a dataset for debugging"""
    try:
        dataset = load_dataset(dataset_name, split="train[:1]")  # Just load one sample
        print(f"\nStructure of {dataset_name}:")
        
        # Print first example to see structure
        example = dataset[0]
        for key, value in example.items():
            print(f"  {key}: {type(value)}")
            
            # Show a sample of the value
            if isinstance(value, list) and len(value) > 0:
                print(f"    Sample: {value[0]}")
            elif not isinstance(value, (list, dict)):
                print(f"    Sample: {value}")
        
        print("\n")
        return dataset[0]  # Return first example for reference
        
    except Exception as e:
        print(f"Error inspecting dataset {dataset_name}: {e}")
        return None

if __name__ == "__main__":
    main() 