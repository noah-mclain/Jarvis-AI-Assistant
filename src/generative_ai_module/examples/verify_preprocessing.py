import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
from tqdm import tqdm
import gensim.downloader as api
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from collections import Counter

# Add the parent directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from generative_ai_module.text_generator import TextGenerator
from generative_ai_module.dataset_processor import DatasetProcessor

# Add this import to fix the "ImprovedCharTokenizer not defined" error
from improved_preprocessing import ImprovedCharTokenizer, create_improved_sequences, clean_and_normalize_text

# Try to import datasets, but don't fail if not available
try:
    from datasets import load_dataset
except ImportError:
    print("Datasets library not available. Using sample data only.")
    load_dataset = None

# Load pretrained word embeddings
word_vectors = None
try:
    word_vectors = api.load("glove-wiki-gigaword-100")
    print(f"Loaded word vectors with dimension: {word_vectors.vectors.shape[1]}")
except Exception as e:
    print(f"Could not load word vectors: {e}. Proceeding without pre-trained embeddings.")

# Simple LSTM model for character-level generation
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Only copy word vectors if dimensions match
        if embedding_dim == 100 and word_vectors is not None:
            try:
                self.embedding.weight.data.copy_(torch.from_numpy(word_vectors.vectors))
                print("Initialized with pre-trained embeddings")
            except Exception as e:
                print(f"Could not initialize with pre-trained embeddings: {e}")
        
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
        self.idx_to_char = dict(enumerate(self.chars))
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
    
    for i in range(len(tokenized_text) - seq_length):
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

def train_epoch(model, data_batches):
    """Train for one epoch and return average loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    batch_count = 0
    
    for batch in tqdm(data_batches, desc="Training"):
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
    
    return total_loss / max(1, batch_count)  # Avoid division by zero

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

def generate_improved_text(model, tokenizer, seed_text, max_length=200, temperature=0.7):
    """Generate text with better sampling strategy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Tokenize the seed text
    with torch.no_grad():
        # Encode the seed text
        context_tokens = tokenizer.encode(seed_text)
        
        # Use the last 100 tokens as context (or pad if needed)
        if len(context_tokens) < 100:
            # Pad with spaces
            pad_length = 100 - len(context_tokens)
            context_tokens = tokenizer.encode(' ' * pad_length) + context_tokens
        else:
            # Use the last 100 tokens
            context_tokens = context_tokens[-100:]
            
        context_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device)
        
        # Generate text
        generated_text = seed_text
        
        for _ in range(max_length):
            # Get predictions
            outputs, _ = model(context_tensor)
            
            # Apply temperature and sample
            probs = F.softmax(outputs / temperature, dim=-1)
            
            # Top-k sampling (restrict to most likely tokens)
            top_k = 40
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            
            # Renormalize probabilities for top-k
            top_k_probs = top_k_probs / top_k_probs.sum()
            
            # Sample from top-k
            next_token_idx = torch.multinomial(top_k_probs, 1).item()
            next_token = top_k_indices[0, next_token_idx].item()
            
            # Add to generated text
            next_char = tokenizer.idx_to_char[next_token]
            generated_text += next_char
            
            # Update context
            context_tokens.append(next_token)
            context_tokens = context_tokens[-100:]  # Keep last 100 tokens
            context_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device)
    
    return generated_text

def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Parameters
    seq_length = 100  # Longer sequences capture more context
    batch_size = 64   # Larger batches for more stable training
    learning_rate = 0.001  # Standard learning rate for Adam
    
    # Initialize optimizer (moved here from inside functions)
    optimizer = None  # Will be initialized for each model
    
    # Add learning rate scheduling function
    def get_scheduler(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=2, verbose=True
        )
    
    # Train on Persona Chat
    try:
        print("\n===== Training on Persona Chat Dataset =====")
        
        # Load and tokenize data
        persona_tokens = load_dataset_text("persona_chat", tokenizer, max_samples=500)
        print(f"Loaded {len(persona_tokens)} tokens from Persona Chat dataset")
        
        # Create sequences and batches
        persona_sequences = create_sequences(persona_tokens, seq_length)
        persona_batches = create_batches(persona_sequences, batch_size)
        print(f"Created {len(persona_sequences)} sequences and {len(persona_batches)} batches")
        
        # Create model
        persona_model = ImprovedCharLSTM(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=100,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(persona_model.parameters(), lr=learning_rate)
        
        # Initialize scheduler
        scheduler = get_scheduler(optimizer)
        
        # Train with early stopping
        persona_model = train_with_early_stopping(persona_model, persona_batches, epochs=10, patience=3)
        
        # Save model
        torch.save(persona_model.state_dict(), "models/persona_chat_model.pt")
        
        # Generate sample text
        seed = "<PERSONA>\n- I am a"
        generated = generate_improved_text(persona_model, tokenizer, seed, max_length=200)
        print(f"\nGenerated Persona Chat text:\n{generated}")
        
    except Exception as e:
        print(f"Error training on Persona Chat: {e}")
        import traceback
        traceback.print_exc()
    
    # Train on Writing Prompts
    try:
        print("\n===== Training on Writing Prompts Dataset =====")
        
        # Load and tokenize data
        prompts_tokens = load_dataset_text("writing_prompts", tokenizer, max_samples=200)
        print(f"Loaded {len(prompts_tokens)} tokens from Writing Prompts dataset")
        
        # Create sequences and batches
        prompts_sequences = create_sequences(prompts_tokens, seq_length)
        prompts_batches = create_batches(prompts_sequences, batch_size)
        print(f"Created {len(prompts_sequences)} sequences and {len(prompts_batches)} batches")
        
        # Create and train model
        prompts_model = CharLSTM(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=100,
            hidden_dim=128
        )
        
        prompts_model = train_with_early_stopping(prompts_model, prompts_batches, epochs=10, patience=3)
        
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

class ImprovedCharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(ImprovedCharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Only attempt to use pre-trained embeddings if dimensions match
        if embedding_dim == 100 and word_vectors is not None:
            try:
                self.embedding.weight.data.copy_(torch.from_numpy(word_vectors.vectors))
                print("Initialized with pre-trained embeddings")
            except Exception as e:
                print(f"Could not initialize with pre-trained embeddings: {e}")
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        return output, hidden

def train_with_early_stopping(model, train_batches, epochs=10, patience=3):
    best_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, train_batches)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve_epochs = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model  # Return the model after training

def clean_text(text):
    """Clean and normalize text"""
    # Remove excess whitespace
    text = ' '.join(text.split())
    
    # Normalize dialogue tags
    text = text.replace("USER:", "<USER>").replace("ASSISTANT:", "<ASSISTANT>")
    
    # Add space around special tokens for better tokenization
    for token in ['<PERSONA>', '<DIALOGUE>', '<END>', '<PROMPT>', '<STORY>']:
        text = text.replace(token, f" {token} ")
    
    return text

def evaluate_model(model, tokenizer, test_data, seq_length=50):
    """Evaluate model performance on test data"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_bleu = 0
    num_samples = 0
    
    with torch.no_grad():
        for input_seq, target_seq in test_data:
            # Generate prediction
            output, _ = model(input_seq)
            
            # Calculate metrics
            loss = criterion(output, target_seq)
            total_loss += loss.item()
            
            # Generate text for BLEU calculation
            seed = tokenizer.decode(input_seq[0].tolist())
            generated = generate_improved_text(model, tokenizer, seed, max_length=50)
            reference = tokenizer.decode(target_seq[0].tolist())
            
            # Calculate BLEU score
            bleu = sentence_bleu([reference], generated)
            total_bleu += bleu
            
            num_samples += 1
    
    avg_loss = total_loss / num_samples
    avg_bleu = total_bleu / num_samples
    
    return {
        'loss': avg_loss,
        'bleu': avg_bleu
    }

def verify_preprocessed_data(preprocessed_path, sample_text_path=None):
    """
    Verify preprocessed data saved by improved_preprocessing.py
    
    Args:
        preprocessed_path: Path to the .pt file with preprocessed data
        sample_text_path: Optional path to a sample text file for testing
    """
    print(f"\n===== Verifying Preprocessed Data: {preprocessed_path} =====")
    
    # Load preprocessed data
    try:
        data = torch.load(preprocessed_path)
        print(f"Successfully loaded data for {data['dataset_name']}")
        
        # Print basic statistics
        print(f"Vocabulary size: {data['vocab_size']}")
        print(f"Token count: {len(data['tokens'])}")
        print(f"Sequence count: {len(data['sequences'])}")
        print(f"Batch count: {len(data['batches'])}")
        
        # Reconstruct the tokenizer from the loaded data
        tokenizer = None
        if 'vocab_size' in data:
            # Create a compatible tokenizer for testing
            tokenizer = ImprovedCharTokenizer(add_special_tokens=True)
            
        # Test a sequence example
        if data['sequences'] and tokenizer:
            print("\nSample sequence:")
            seq_in, seq_out = data['sequences'][0]
            input_text = tokenizer.decode(seq_in)
            target_text = tokenizer.decode([seq_out])
            print(f"Input: {input_text[:50]}...")
            print(f"Target: '{target_text}'")
            
        # Check a batch example
        if data['batches']:
            inputs, targets = data['batches'][0]
            print(f"\nBatch shapes - Input: {inputs.shape}, Target: {targets.shape}")
            
        # Test with sample_text if provided
        if sample_text_path and tokenizer:
            test_with_sample_text(tokenizer, sample_text_path)
            
        return data, tokenizer
        
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None, None
    
def test_with_sample_text(tokenizer, sample_text_path):
    """Test tokenizer with a sample text file"""
    print(f"\n===== Testing with sample text: {sample_text_path} =====")
    
    try:
        # Load the sample text
        with open(sample_text_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()
            
        # Clean and tokenize
        cleaned_text = clean_and_normalize_text(sample_text)
        tokens = tokenizer.encode(cleaned_text)
        
        print(f"Sample text length: {len(sample_text)}")
        print(f"Cleaned text length: {len(cleaned_text)}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Show token distribution
        counter = Counter(tokens)
        print(f"Unique tokens: {len(counter)} out of {tokenizer.vocab_size}")
        
        # Show most common tokens
        print("\nMost common tokens:")
        for token, count in counter.most_common(10):
            char_repr = tokenizer.idx_to_char.get(token, '<UNK>')
            if char_repr in ['\n', '\t', ' ']:
                char_repr = f"'{repr(char_repr)[1:-1]}'"
            print(f"  {token}: '{char_repr}' ({count} occurrences, {count/len(tokens)*100:.2f}%)")
        
        # Create sequences for testing
        sequences = create_improved_sequences(tokens, seq_length=50, stride=25)
        print(f"\nCreated {len(sequences)} test sequences")
        
        # Show a sample sequence
        if sequences:
            seq_in, seq_out = sequences[0]
            input_text = tokenizer.decode(seq_in)
            target_text = tokenizer.decode([seq_out])
            print(f"\nSample sequence:")
            print(f"Input: {input_text[:50]}...")
            print(f"Target: '{target_text}'")
            
        # Recreate the original text to verify tokenization is reversible
        decoded_text = tokenizer.decode(tokens)
        match_percent = sum(1 for a, b in zip(cleaned_text, decoded_text) if a == b) / len(cleaned_text) * 100
        print(f"\nTokenization reversibility: {match_percent:.2f}% match")
        
        # Plot token distribution
        plot_token_distribution(tokens, tokenizer, "manual_test")
        
    except Exception as e:
        print(f"Error testing with sample text: {e}")

def plot_token_distribution(tokens, tokenizer, filename_prefix):
    """Plot the distribution of tokens"""
    try:
        # Count token frequencies
        counter = Counter(tokens)
        
        # Get most common for plotting
        most_common = counter.most_common(20)
        tokens_to_plot = [token for token, _ in most_common]
        frequencies = [count for _, count in most_common]
        
        # Convert token IDs to characters for labels
        labels = [tokenizer.idx_to_char.get(token, '<UNK>') for token in tokens_to_plot]
        # Replace special whitespace characters for display
        labels = [repr(label)[1:-1] if label in ['\n', '\t', ' '] else label for label in labels]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(tokens_to_plot)), frequencies)
        ax.set_xticks(range(len(tokens_to_plot)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('Token Frequency Distribution (Sample Text)')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Token')
        
        plt.tight_layout()
        
        # Create directory for plots
        os.makedirs("preprocessing_analysis", exist_ok=True)
        plt.savefig(f"preprocessing_analysis/{filename_prefix}_tokens.png")
        plt.close()
        
        print(f"Token distribution plot saved to preprocessing_analysis/{filename_prefix}_tokens.png")
        
    except Exception as e:
        print(f"Error creating token distribution plot: {e}")
        
def verify_model_compatibility(data, model_class=None):
    """Verify that the processed data is compatible with the model"""
    if not data or 'batches' not in data or not data['batches']:
        print("No batches available to test model compatibility")
        return
        
    try:
        # Create a model with the right vocabulary size
        if model_class is None:
            from verify_preprocessing import CharLSTM
            model = CharLSTM(
                vocab_size=data['vocab_size'],
                embedding_dim=128,
                hidden_dim=256
            )
        else:
            model = model_class(vocab_size=data['vocab_size'])
            
        # Test a forward pass with the first batch
        inputs, targets = data['batches'][0]
        outputs, _ = model(inputs)
        
        # Check output shape
        expected_shape = (inputs.shape[0], data['vocab_size'])
        print(f"\nModel output shape: {outputs.shape}, Expected: {expected_shape}")
        
        if outputs.shape == expected_shape:
            print("✓ Model compatibility verified: Output shape matches expectation")
        else:
            print("✗ Model compatibility issue: Output shape doesn't match expected shape")
            
    except Exception as e:
        print(f"Error verifying model compatibility: {e}")

# Add this to your main function
def manual_verification():
    """Manually verify the preprocessing with saved files"""
    print("\n====== Manual Preprocessing Verification ======")
    
    # Check if preprocessed data exists
    if not os.path.exists("preprocessed_data"):
        print("Preprocessed data directory not found. Run improved_preprocessing.py first.")
        return
        
    # Get paths to preprocessed files
    persona_path = os.path.join("preprocessed_data", "persona_chat_preprocessed.pt")
    prompts_path = os.path.join("preprocessed_data", "writing_prompts_preprocessed.pt")
    
    # Verify both datasets
    persona_data, persona_tokenizer = verify_preprocessed_data(persona_path)
    prompts_data, prompts_tokenizer = verify_preprocessed_data(prompts_path)
    
    # Check model compatibility
    if persona_data:
        verify_model_compatibility(persona_data)
        
    # Test with additional sample text if available
    sample_text_path = "data/sample_text.txt"
    if os.path.exists(sample_text_path) and persona_tokenizer:
        test_with_sample_text(persona_tokenizer, sample_text_path)
    else:
        print(f"\nSample text file not found at {sample_text_path}")
        print("To test with your own text, create a file at this location.")
        
    print("\n====== Manual Verification Complete ======")

# Add this to your main() function
if __name__ == "__main__":
    # Run the regular preprocessing steps if needed
    # main()
    
    # Or just run the manual verification
    manual_verification() 