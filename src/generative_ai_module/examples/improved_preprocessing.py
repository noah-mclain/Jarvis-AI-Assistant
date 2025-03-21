import os
import sys
import torch
import random
import string
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add the parent directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from datasets import load_dataset
except ImportError:
    print("Datasets library not available. Using sample data only.")
    load_dataset = None

class ImprovedCharTokenizer:
    """Improved character-level tokenizer with better special token handling"""
    def __init__(self, add_special_tokens=True):
        # Start with just basic ASCII
        self.chars = sorted(list(string.printable))
        
        # Add special tokens
        self.special_tokens = []
        if add_special_tokens:
            self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
            self.chars.extend(self.special_tokens)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Special token indices
        self.pad_idx = self.char_to_idx.get('<PAD>', 0)
        self.unk_idx = self.char_to_idx.get('<UNK>', 1)
        self.bos_idx = self.char_to_idx.get('<BOS>', 2)
        self.eos_idx = self.char_to_idx.get('<EOS>', 3)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens: {self.special_tokens}")
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Convert text to sequence of token IDs"""
        result = []
        
        if add_bos and '<BOS>' in self.special_tokens:
            result.append(self.bos_idx)
            
        for ch in text:
            result.append(self.char_to_idx.get(ch, self.unk_idx))
            
        if add_eos and '<EOS>' in self.special_tokens:
            result.append(self.eos_idx)
            
        return result
    
    def decode(self, ids, skip_special_tokens=True):
        """Convert token IDs back to text"""
        # Filter out special tokens if requested
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in 
                  [self.pad_idx, self.bos_idx, self.eos_idx]]
        
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in ids])

def clean_and_normalize_text(text):
    """Apply more extensive text cleaning and normalization"""
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Keep some structure with newlines for dialogue/formatting
    lines = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line.strip())
    
    # Rejoin with single newlines
    text = '\n'.join(lines)
    
    # Add spaces around special tokens for better tokenization
    special_tokens = ['<PERSONA>', '<DIALOGUE>', '<END>', '<PROMPT>', '<STORY>']
    for token in special_tokens:
        text = text.replace(token, f" {token} ")
    
    # Normalize dialogue markers
    text = text.replace("USER:", "<USER>").replace("ASSISTANT:", "<ASSISTANT>")
    
    return text

def load_and_preprocess_dataset(dataset_name, tokenizer, split="train", max_samples=100):
    """Load, clean, and tokenize dataset text"""
    # Use sample data if datasets library not available
    if load_dataset is None:
        return get_sample_data(dataset_name, tokenizer)
    
    try:
        # Different handling based on dataset type
        if dataset_name == "persona_chat":
            raw_text = load_persona_chat(split, max_samples)
        elif dataset_name == "writing_prompts":
            raw_text = load_writing_prompts(split, max_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Clean text
        cleaned_text = clean_and_normalize_text(raw_text)
        
        # Create character-level tokens
        tokens = tokenizer.encode(cleaned_text)
        
        return {
            'dataset_name': dataset_name,
            'raw_length': len(raw_text),
            'cleaned_length': len(cleaned_text),
            'tokens': tokens,
            'vocab_size': tokenizer.vocab_size,
            'sample_text': cleaned_text[:500] + "..."  # First 500 chars as sample
        }
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return sample data as fallback
        return get_sample_data(dataset_name, tokenizer)

def get_sample_data(dataset_name, tokenizer):
    """Get sample data for a dataset"""
    if dataset_name == "persona_chat":
        text = get_sample_persona_chat()
    else:
        text = get_sample_writing_prompts()
    
    cleaned_text = clean_and_normalize_text(text)
    tokens = tokenizer.encode(cleaned_text)
    
    return {
        'dataset_name': f"{dataset_name} (sample)",
        'raw_length': len(text),
        'cleaned_length': len(cleaned_text),
        'tokens': tokens,
        'vocab_size': tokenizer.vocab_size,
        'sample_text': cleaned_text[:500] + "..."
    }

def load_persona_chat(split='train', max_samples=None, cache_dir=None):
    """Load Persona Chat dataset with better error handling"""
    try:
        # Try the Google Synthetic Persona Chat dataset first
        try:
            dataset = load_dataset("google/Synthetic-Persona-Chat", split=split, cache_dir=cache_dir)
            print("Loaded google/Synthetic-Persona-Chat dataset")
        except Exception as e:
            print(f"Error loading Google Synthetic Persona Chat: {e}")
            # Fallback to Facebook dataset
            dataset = load_dataset("facebook/personachat", split=split, cache_dir=cache_dir)
            print("Loaded facebook/personachat dataset")
            
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Convert to text format
        text = ""
        for item in tqdm(dataset, desc="Processing Persona Chat"):
            # Add persona section
            text += "<PERSONA>\n"
            
            # Extract persona information
            if 'user 1 personas' in item:
                text += f"- {item['user 1 personas'].strip()}\n"
            elif 'personality' in item:
                for persona in item['personality']:
                    text += f"- {persona.strip()}\n"
            else:
                text += "- No persona available\n"
                
            # Add dialogue section
            text += "<DIALOGUE>\n"
            
            # Extract conversation
            if 'Best Generated Conversation' in item:
                conv = item['Best Generated Conversation']
                for line in conv.strip().split('\n'):
                    if line.startswith('User:'):
                        text += f"USER: {line[5:].strip()}\n"
                    elif line.startswith('Assistant:'):
                        text += f"ASSISTANT: {line[10:].strip()}\n"
            elif 'utterances' in item:
                # Handle Facebook dataset format
                for utterance in item['utterances'][-1]['history']:
                    if len(utterance.strip()) > 0:
                        # Alternate between user and assistant
                        if text.rstrip().endswith('USER:'):
                            text += f" {utterance.strip()}\n"
                            text += f"ASSISTANT: "
                        else:
                            text += f"USER: {utterance.strip()}\n"
            
            # End the sample
            text += "<END>\n\n"
        
        return text
            
    except Exception as e:
        print(f"Error loading Persona Chat dataset: {e}")
        return get_sample_persona_chat()

def load_writing_prompts(split='train', max_samples=None, cache_dir=None):
    """Load Writing Prompts dataset with better error handling"""
    try:
        # Load the dataset
        dataset = load_dataset("euclaise/writingprompts", split=split, cache_dir=cache_dir)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Convert to text format
        text = ""
        for item in tqdm(dataset, desc="Processing Writing Prompts"):
            # Format: <PROMPT> prompt text <STORY> story text <END>
            prompt = item['prompt'].strip()
            story = item['story'].strip()
            
            # Skip empty prompts or stories
            if not prompt or not story:
                continue
                
            text += f"<PROMPT> {prompt}\n"
            text += f"<STORY> {story}\n"
            text += "<END>\n\n"
            
        return text
            
    except Exception as e:
        print(f"Error loading Writing Prompts dataset: {e}")
        return get_sample_writing_prompts()

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

def create_improved_sequences(tokens, seq_length=100, stride=1):
    """Create sequences with stride for more efficient data usage"""
    sequences = []
    
    # Create sequences with stride
    for i in range(0, len(tokens) - seq_length - 1, stride):
        # Input is the current sequence
        seq_in = tokens[i:i+seq_length]
        # Target is the next token
        seq_out = tokens[i+seq_length]
        
        sequences.append((seq_in, seq_out))
    
    return sequences

def analyze_token_distribution(tokens, tokenizer):
    """Analyze token distribution and plot histogram"""
    # Count token frequencies
    counter = Counter(tokens)
    
    # Get most common tokens and their frequencies
    most_common = counter.most_common(20)
    
    # Print most common tokens
    print("\nMost common tokens:")
    for token, count in most_common:
        char_repr = tokenizer.idx_to_char.get(token, '<UNK>')
        if char_repr in ['\n', '\t', ' ']:
            char_repr = f"'{repr(char_repr)[1:-1]}'"
        print(f"  {token}: '{char_repr}' ({count} occurrences, {count/len(tokens)*100:.2f}%)")
    
    # Plot histogram of token frequencies
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get tokens for x-axis (limit to most frequent for visibility)
    tokens_to_plot = [token for token, _ in counter.most_common(30)]
    frequencies = [counter[token] for token in tokens_to_plot]
    
    # Convert token IDs to characters for labels
    labels = [tokenizer.idx_to_char.get(token, '<UNK>') for token in tokens_to_plot]
    # Replace special whitespace characters for display
    labels = [repr(label)[1:-1] if label in ['\n', '\t', ' '] else label for label in labels]
    
    ax.bar(range(len(tokens_to_plot)), frequencies)
    ax.set_xticks(range(len(tokens_to_plot)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Token Frequency Distribution')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Token')
    
    plt.tight_layout()
    
    # Create directory for plots
    os.makedirs("preprocessing_analysis", exist_ok=True)
    plt.savefig(f"preprocessing_analysis/token_distribution.png")
    plt.close()
    
    return {
        'unique_tokens': len(counter),
        'most_common': most_common,
        'total_tokens': len(tokens)
    }

def verify_sequence_creation(tokens, tokenizer, seq_length=100):
    """Verify sequence creation and show examples"""
    sequences = create_improved_sequences(tokens, seq_length, stride=seq_length)
    
    print(f"\nCreated {len(sequences)} sequences with length {seq_length}")
    
    # Show a few examples
    print("\nSequence examples:")
    for i in range(min(3, len(sequences))):
        input_seq, target = sequences[i]
        
        # Decode the sequence
        input_text = tokenizer.decode(input_seq)
        target_text = tokenizer.decode([target])
        
        # Print sample
        print(f"\nExample {i+1}:")
        print(f"Input (truncated): {input_text[:50]}...")
        print(f"Target: '{target_text}'")
    
    return sequences

def create_and_verify_batches(sequences, batch_size=64):
    """Create batches and verify shapes"""
    # Shuffle sequences
    random.shuffle(sequences)
    
    # Create mini-batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        
        if not batch_sequences:
            continue
            
        # Unzip the batch
        input_seqs, targets = zip(*batch_sequences)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_seqs, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        
        batches.append((input_tensor, target_tensor))
    
    print(f"\nCreated {len(batches)} batches with batch size {batch_size}")
    
    if batches:
        # Print shape information for the first batch
        inputs, targets = batches[0]
        print(f"Input batch shape: {inputs.shape}")
        print(f"Target batch shape: {targets.shape}")
    
    return batches

def save_preprocessed_data(data, output_dir="preprocessed_data"):
    """Save preprocessed data to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenized data
    torch.save(data, os.path.join(output_dir, f"{data['dataset_name']}_preprocessed.pt"))
    
    # Save a sample as text
    with open(os.path.join(output_dir, f"{data['dataset_name']}_sample.txt"), "w") as f:
        f.write(data['sample_text'])
    
    print(f"\nSaved preprocessed data to {output_dir}/{data['dataset_name']}_preprocessed.pt")
    print(f"Saved text sample to {output_dir}/{data['dataset_name']}_sample.txt")

def create_vocabulary(text_corpus, min_frequency=5):
    """Create a vocabulary from text corpus with frequency thresholding"""
    # Count all character occurrences
    counter = Counter(text_corpus)
    
    # Filter by frequency
    vocab = [char for char, count in counter.items() if count >= min_frequency]
    
    # Add special tokens at the beginning
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    vocab = special_tokens + sorted(vocab)
    
    return vocab

def segment_by_dialogue_turns(text):
    """Segment text into dialogue turns for better context modeling"""
    segments = []
    current_segment = []
    
    for line in text.split('\n'):
        if line.startswith('USER:') or line.startswith('ASSISTANT:'):
            # Start a new segment if buffer is not empty
            if current_segment:
                segments.append('\n'.join(current_segment))
                current_segment = []
        
        if line.strip():
            current_segment.append(line)
    
    # Add the last segment
    if current_segment:
        segments.append('\n'.join(current_segment))
    
    return segments

def main():
    print("====== Dataset Preprocessing Verification ======")
    
    # Create directories
    os.makedirs("preprocessed_data", exist_ok=True)
    
    # Create improved tokenizer
    tokenizer = ImprovedCharTokenizer(add_special_tokens=True)
    
    # Process Persona Chat dataset
    print("\n===== Processing Persona Chat Dataset =====")
    persona_data = load_and_preprocess_dataset("persona_chat", tokenizer, max_samples=100)
    
    print(f"Raw text length: {persona_data['raw_length']}")
    print(f"Cleaned text length: {persona_data['cleaned_length']}")
    print(f"Number of tokens: {len(persona_data['tokens'])}")
    
    # Analyze token distribution
    persona_analysis = analyze_token_distribution(persona_data['tokens'], tokenizer)
    print(f"Unique tokens: {persona_analysis['unique_tokens']} out of {tokenizer.vocab_size}")
    
    # Create and verify sequences
    persona_sequences = verify_sequence_creation(persona_data['tokens'], tokenizer, seq_length=100)
    
    # Create and verify batches
    persona_batches = create_and_verify_batches(persona_sequences, batch_size=32)
    
    # Save preprocessed data
    persona_data['sequences'] = persona_sequences
    persona_data['batches'] = persona_batches
    save_preprocessed_data(persona_data)
    
    # Process Writing Prompts dataset
    print("\n===== Processing Writing Prompts Dataset =====")
    prompts_data = load_and_preprocess_dataset("writing_prompts", tokenizer, max_samples=50)
    
    print(f"Raw text length: {prompts_data['raw_length']}")
    print(f"Cleaned text length: {prompts_data['cleaned_length']}")
    print(f"Number of tokens: {len(prompts_data['tokens'])}")
    
    # Analyze token distribution
    prompts_analysis = analyze_token_distribution(prompts_data['tokens'], tokenizer)
    print(f"Unique tokens: {prompts_analysis['unique_tokens']} out of {tokenizer.vocab_size}")
    
    # Create and verify sequences
    prompts_sequences = verify_sequence_creation(prompts_data['tokens'], tokenizer, seq_length=100)
    
    # Create and verify batches
    prompts_batches = create_and_verify_batches(prompts_sequences, batch_size=32)
    
    # Save preprocessed data
    prompts_data['sequences'] = prompts_sequences
    prompts_data['batches'] = prompts_batches
    save_preprocessed_data(prompts_data)
    
    print("\n====== Preprocessing Verification Complete ======")
    print(f"Saved all preprocessed data to 'preprocessed_data' directory")

if __name__ == "__main__":
    main() 