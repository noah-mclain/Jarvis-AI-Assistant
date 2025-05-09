import os
import sys
import torch
import random
import string
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple

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
        self.idx_to_char = dict(enumerate(self.chars))
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

        result.extend(self.char_to_idx.get(ch, self.unk_idx) for ch in text)
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
    lines.extend(line.strip() for line in text.splitlines() if line.strip())
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
            'sample_text': f"{cleaned_text[:500]}...",  # First 500 chars as sample
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
        'sample_text': f"{cleaned_text[:500]}...",    # First 500 chars as sample
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
                            text += "ASSISTANT: "
                        else:
                            text += f"USER: {utterance.strip()}\n"

            # End the sample
            text += "<END>\n\n"

        return text

    except Exception as e:
        print(f"Error loading Persona Chat dataset: {e}")
        return get_sample_persona_chat()

def load_writing_prompts(split='train', max_samples=None, cache_dir=None):
    """
    Load and preprocess the Writing Prompts dataset

    Args:
        split: Dataset split ('train', 'test', or 'validation')
        max_samples: Maximum number of samples to load (None for all)
        cache_dir: Optional directory to cache the downloaded dataset

    Returns:
        Preprocessed text ready for sequence creation
    """
    # Define a fallback sample in case loading fails
    if not get_sample_writing_prompts():
        print("Warning: Failed to generate sample writing prompts data")

    try:
        # Try loading the dataset with the correct path
        dataset = load_dataset("euclaise/writingprompts", split=split, cache_dir=cache_dir)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return "".join(
            f"<PROMPT>\n{item['prompt']}\n<STORY>\n{item['story']}\n<END>\n\n"
            for item in tqdm(dataset, desc="Processing Writing Prompts")
        )
    except Exception as e:
        print(f"Error loading Writing Prompts dataset: {e}")
        print("Falling back to sample data")
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

def create_sequences(self, text: str, seq_length: int):
    """Create sequences with proper dtype handling"""
    tokens = self.tokenizer.encode(text)

    # Ensure proper dtype handling
    input_ids = torch.tensor(
        tokens[:seq_length],
        dtype=torch.long,  # Force long type
        device='cpu'  # Keep on CPU initially
    )

    # Handle padding with correct dtype
    if len(input_ids) < seq_length:
        padding = torch.full((seq_length - len(input_ids),),
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                            device='cpu')
        input_ids = torch.cat([input_ids, padding])

    return input_ids

def create_improved_sequences(tokens, tokenizer, seq_length=256, stride=1):  # Reduced from 512 to 256
    """Create sequences with stride and padding"""
    sequences = []
    for i in range(0, len(tokens) - seq_length, stride):
        chunk = tokens[i:i + seq_length]
        # Pad if shorter than seq_length
        if len(chunk) < seq_length:
            chunk += [tokenizer.pad_idx] * (seq_length - len(chunk))
        next_token = tokens[i + seq_length] if (i + seq_length) < len(tokens) else tokenizer.pad_idx
        sequences.append((chunk, next_token))
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
    plt.savefig("preprocessing_analysis/token_distribution.png")
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

def create_batches(sequences: List[torch.Tensor], batch_size: int):
    """Create batches with memory optimization"""
    batches = []
    current_batch = []

    # Add memory monitoring
    for seq in sequences:
        # Keep tensors on CPU until needed
        if len(current_batch) >= batch_size:
            # Create tensor on CPU first
            batch_tensor = torch.stack(current_batch)

            # Only pin memory if we're on CPU and will be moving to CUDA later
            if batch_tensor.device.type == 'cpu' and torch.cuda.is_available():
                try:
                    batch_tensor = batch_tensor.pin_memory()
                except Exception as e:
                    print(f"Warning: Could not pin memory: {e}. Continuing without pinning.")

            batches.append(batch_tensor)
            current_batch = []
            # Clear memory aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        current_batch.append(seq.cpu())  # Keep on CPU initially

    # Handle the last batch if it exists
    if current_batch:
        # Create tensor on CPU first
        batch_tensor = torch.stack(current_batch)

        # Only pin memory if we're on CPU and will be moving to CUDA later
        if batch_tensor.device.type == 'cpu' and torch.cuda.is_available():
            try:
                batch_tensor = batch_tensor.pin_memory()
            except Exception as e:
                print(f"Warning: Could not pin memory: {e}. Continuing without pinning.")

        batches.append(batch_tensor)

    return batches

def create_and_verify_batches(sequences, batch_size=4):  # Reduced from 32 to 4
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
        if (line.startswith('USER:') or line.startswith('ASSISTANT:')) and current_segment:
            segments.append('\n'.join(current_segment))
            current_segment = []

        if line.strip():
            current_segment.append(line)

    # Add the last segment
    if current_segment:
        segments.append('\n'.join(current_segment))

    return segments

class ImprovedPreprocessor:
    """Enhanced with dataset-specific memory controls"""
    def __init__(self, min_length=10, max_length=100, analyze=False, batch_size=4):
        self.min_length = min_length
        self.max_length = max_length
        self.analyze = analyze
        self.tokenizer = ImprovedCharTokenizer(add_special_tokens=True)
        self.batch_size = batch_size

        # Dataset-specific configuration
        self.dataset_params = {
            "writing_prompts": {
                "max_sequence_length": 1024,
                "batch_size": 2,
                "max_text_length": 2048,
                "stride": 512,
                "grad_accum_steps": 4,
                "use_mixed_precision": True,
                "memory_efficient": True
            },
            "persona_chat": {
                "max_sequence_length": 512,
                "batch_size": 16,
                "max_text_length": None,
                "stride": 256,
                "grad_accum_steps": 1,
                "use_mixed_precision": False,
                "memory_efficient": False
            },
            "pile": {
                "max_sequence_length": 1024,
                "batch_size": 8,
                "max_text_length": None,
                "stride": 512,
                "grad_accum_steps": 2,
                "use_mixed_precision": False,
                "memory_efficient": True
            },
            "openassistant": {
                "max_sequence_length": 512,
                "batch_size": 16,
                "max_text_length": None,
                "stride": 256,
                "grad_accum_steps": 1,
                "use_mixed_precision": False,
                "memory_efficient": False
            },
            "gpteacher": {
                "max_sequence_length": 768,
                "batch_size": 12,
                "max_text_length": None,
                "stride": 384,
                "grad_accum_steps": 1,
                "use_mixed_precision": False,
                "memory_efficient": False
            },
            "default": {
                "max_sequence_length": 512,
                "batch_size": 8,
                "max_text_length": None,
                "stride": 256,
                "grad_accum_steps": 1,
                "use_mixed_precision": False,
                "memory_efficient": False
            }
        }

        self.config = {
            "max_sequence_length": 2048,
            "batch_size": batch_size,
            "use_mixed_precision": True,
            "cpu_offload": True
        }

        # Track current dataset for training
        self.current_dataset = "default"
        self.step_count = 0

    def process_dataset(self, dataset_name, max_samples=100):
        """Modified with dataset-specific processing"""
        # Get dataset parameters
        params = self.dataset_params.get(dataset_name, self.dataset_params["default"])

        # Load dataset with dataset-specific processing
        try:
            # Import dataset processor for loading
            from src.generative_ai_module.dataset_processor import DatasetProcessor
            processor = DatasetProcessor()

            # Load raw text based on dataset name
            print(f"Loading {dataset_name} dataset...")
            if dataset_name == 'persona_chat':
                raw_text = processor.load_persona_chat(split='train', max_samples=max_samples)
                # Apply persona chat specific preprocessing
                raw_text = self._preprocess_persona_chat(raw_text)
            elif dataset_name == 'writing_prompts':
                raw_text = processor.load_writing_prompts(split='train', max_samples=max_samples)
                # Apply writing prompts specific preprocessing
                raw_text = self._preprocess_writing_prompts(raw_text, max_length=params.get("max_text_length"))
            elif dataset_name == 'pile':
                raw_text = processor.load_pile_dataset(split='train', max_samples=max_samples)
                # Apply pile specific preprocessing
                raw_text = self._preprocess_pile(raw_text)
            elif dataset_name == 'openassistant':
                raw_text = processor.load_openassistant_dataset(split='train', max_samples=max_samples)
                # Apply openassistant specific preprocessing
                raw_text = self._preprocess_openassistant(raw_text)
            elif dataset_name == 'gpteacher':
                raw_text = processor.load_gpteacher_dataset(split='train', max_samples=max_samples)
                # Apply gpteacher specific preprocessing
                raw_text = self._preprocess_gpteacher(raw_text)
            else:
                print(f"Unknown dataset: {dataset_name}, using sample data")
                return get_sample_data(dataset_name, self.tokenizer)

            # Clean and normalize text
            cleaned_text = clean_and_normalize_text(raw_text)

            # Tokenize text
            tokens = self.tokenizer.encode(cleaned_text)

            # Create data dictionary
            data = {
                'dataset_name': dataset_name,
                'raw_text': raw_text,
                'tokens': tokens,
                'vocab_size': self.tokenizer.vocab_size
            }

            # Apply dataset-specific sequence creation
            sequences = create_improved_sequences(
                data['tokens'],
                self.tokenizer,
                seq_length=params["max_sequence_length"],
                stride=params["stride"]
            )

            # Create memory-optimized batches
            batches = self.create_dataset_batches(
                sequences,
                batch_size=params["batch_size"],
                dataset_name=dataset_name
            )

            # Update data with sequences and batches
            data.update({
                'sequences': sequences,
                'batches': batches,
                'params': params
            })

            # Set current dataset for training
            self.current_dataset = dataset_name

            return data

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            data = get_sample_data(dataset_name, self.tokenizer)
            return data

    def _preprocess_persona_chat(self, text):
        """Apply persona chat specific preprocessing."""
        # Ensure consistent formatting
        text = text.replace("USER:", "USER: ").replace("ASSISTANT:", "ASSISTANT: ")

        # Ensure proper spacing around special tokens
        for token in ["<PERSONA>", "<DIALOGUE>", "<END>"]:
            text = text.replace(token, f"\n{token}\n")

        return text

    def _preprocess_writing_prompts(self, text, max_length=2048):
        """Apply writing prompts specific preprocessing with length limits."""
        # Split into prompt-story pairs
        pairs = text.split("<END>")

        processed_pairs = []
        for pair in pairs:
            if "<PROMPT>" not in pair or "<STORY>" not in pair:
                continue

            # Split into prompt and story
            parts = pair.split("<STORY>")
            if len(parts) != 2:
                continue

            prompt_part = parts[0]
            story_part = parts[1]

            # Truncate very long stories to save memory
            if max_length and len(story_part) > max_length:
                story_part = story_part[:max_length] + "..."

            # Reassemble with proper formatting
            processed_pair = f"{prompt_part}<STORY>{story_part}<END>"
            processed_pairs.append(processed_pair)

        return "\n\n".join(processed_pairs)

    def _preprocess_pile(self, text):
        """Apply pile specific preprocessing."""
        # The Pile has diverse formats, normalize spacing and structure
        text = text.replace("\t", " ").replace("  ", " ")

        # Ensure consistent paragraph breaks
        text = text.replace("\n\n\n", "\n\n")

        return text

    def _preprocess_openassistant(self, text):
        """Apply openassistant specific preprocessing."""
        # Ensure consistent formatting for instruction-response pairs
        text = text.replace("USER:", "USER: ").replace("ASSISTANT:", "ASSISTANT: ")

        # Handle potential JSON formatting issues
        text = text.replace("\\n", "\n").replace('\\"', '"')

        return text

    def _preprocess_gpteacher(self, text):
        """Apply gpteacher specific preprocessing."""
        # Ensure consistent formatting for instruction-response pairs
        text = text.replace("User:", "USER: ").replace("Assistant:", "ASSISTANT: ")

        # Handle potential formatting issues
        text = text.replace("\\n", "\n").replace('\\"', '"')

        return text

    def create_dataset_batches(self, sequences, batch_size, dataset_name):
        """Memory-optimized batch creation"""
        params = self.dataset_params.get(dataset_name, self.dataset_params["default"])

        # Sort sequences by length for efficient packing
        sequences.sort(key=lambda x: len(x[0]))

        batches = []
        current_batch = []

        # Check if we're already on CUDA
        device = 'cpu'  # Always start on CPU

        for seq in sequences:
            current_batch.append(seq)
            if len(current_batch) >= params["batch_size"]:
                # Create tensors on CPU first
                inputs = torch.tensor([s[0] for s in current_batch], dtype=torch.long, device=device)
                targets = torch.tensor([s[1] for s in current_batch], dtype=torch.long, device=device)

                # Only pin memory if we're on CPU and will be moving to CUDA later
                if device == 'cpu' and torch.cuda.is_available():
                    try:
                        inputs = inputs.pin_memory()
                        targets = targets.pin_memory()
                    except Exception as e:
                        print(f"Warning: Could not pin memory: {e}. Continuing without pinning.")

                batches.append((inputs, targets))
                current_batch = []

        # Handle remaining sequences
        if current_batch:
            # Create tensors on CPU first
            inputs = torch.tensor([s[0] for s in current_batch], dtype=torch.long, device=device)
            targets = torch.tensor([s[1] for s in current_batch], dtype=torch.long, device=device)

            # Only pin memory if we're on CPU and will be moving to CUDA later
            if device == 'cpu' and torch.cuda.is_available():
                try:
                    inputs = inputs.pin_memory()
                    targets = targets.pin_memory()
                except Exception as e:
                    print(f"Warning: Could not pin memory: {e}. Continuing without pinning.")

            batches.append((inputs, targets))

        print(f"Created {len(batches)} {dataset_name} batches (size {params['batch_size']})")
        return batches

    def train_batch(self, batch, model=None, optimizer=None):
        """Memory-optimized training with dataset awareness"""
        # Initialize device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Store model and optimizer if provided
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer

        # Validate model/optimizer existence
        if not hasattr(self, 'model') or not hasattr(self, 'optimizer'):
            raise ValueError("Model and optimizer must be provided")

        # Unpack batch
        inputs, targets = batch

        # Move to GPU in chunks if writing_prompts
        if self.current_dataset == "writing_prompts":
            # Split large batches for memory-constrained datasets
            chunk_size = len(inputs) // 2
            losses = []

            for i in range(0, len(inputs), chunk_size):
                chunk_inputs = inputs[i:i+chunk_size].to(device, non_blocking=True)
                chunk_targets = targets[i:i+chunk_size].to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(chunk_inputs)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        chunk_targets.view(-1)
                    )
                    losses.append(loss)

                # Cleanup
                del chunk_inputs, chunk_targets, outputs
                torch.cuda.empty_cache()

            # Average losses from chunks
            loss = torch.mean(torch.stack(losses))
        else:
            # Standard training for other datasets
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1)
                )

        # Backpropagation
        loss.backward()

        # Step optimizer based on grad accumulation
        params = self.dataset_params.get(self.current_dataset, self.dataset_params["default"])
        self.step_count += 1
        if (self.step_count) % params["grad_accum_steps"] == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Memory cleanup
        loss_value = loss.item()
        del inputs, targets, outputs, loss
        torch.cuda.empty_cache()

        return loss_value

    def analyze_token_distribution(self, data):
        """Analyze token distribution in the dataset"""
        return analyze_token_distribution(data['tokens'], self.tokenizer)

    def save_tokenized_data(self, data, output_dir, dataset_name=None):
        """Save preprocessed data"""
        os.makedirs(output_dir, exist_ok=True)
        save_preprocessed_data(data, output_dir=output_dir)

    def save_analysis_results(self, analysis_results, output_dir, dataset_name):
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_analysis.txt")

        with open(output_file, 'w') as f:
            f.write("Token Distribution Analysis\n")
            f.write("=========================\n\n")

            # Write token counts
            f.write("Token Counts:\n")
            for token, count in analysis_results['most_common']:
                char_repr = self.tokenizer.idx_to_char.get(token, '<UNK>')
                if char_repr in ['\n', '\t', ' ']:
                    char_repr = f"'{repr(char_repr)[1:-1]}'"
                f.write(f"  {token}: '{char_repr}' ({count} occurrences, {count/analysis_results['total_tokens']*100:.2f}%)\n")

            # Write sequence statistics
            f.write("\nToken Statistics:\n")
            f.write(f"Total tokens: {analysis_results['total_tokens']}\n")
            f.write(f"Unique tokens: {analysis_results['unique_tokens']} out of {self.tokenizer.vocab_size}\n")

            # Add a note about the visualization
            f.write("\nNote: A visualization of the token distribution has been saved as 'token_distribution.png'\n")
            f.write("in the preprocessing_analysis directory.\n")

def safe_preprocess(dataset_name):
    """Safely preprocess a dataset with comprehensive error handling"""
    try:
        # Create preprocessor with memory-optimized settings
        preprocessor = ImprovedPreprocessor(batch_size=4)

        # Process dataset
        data = preprocessor.process_dataset(dataset_name)

        # Analyze token distribution
        analysis = preprocessor.analyze_token_distribution(data)

        # Save results
        preprocessor.save_tokenized_data(data, "preprocessed_data", dataset_name)
        preprocessor.save_analysis_results(analysis, "preprocessing_analysis", dataset_name)

        return True
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            # Automatic batch size reduction
            print(f"CUDA out of memory error detected. Reducing batch size and retrying...")

            # Try with smaller batch size
            try:
                # Create preprocessor with reduced batch size
                new_batch_size = 2  # Reduce batch size
                print(f"Reducing batch size to {new_batch_size}")

                preprocessor = ImprovedPreprocessor(batch_size=new_batch_size)
                data = preprocessor.process_dataset(dataset_name)
                analysis = preprocessor.analyze_token_distribution(data)
                preprocessor.save_tokenized_data(data, "preprocessed_data", dataset_name)
                preprocessor.save_analysis_results(analysis, "preprocessing_analysis", dataset_name)

                print(f"Successfully processed with reduced batch size")
                return True
            except Exception as inner_e:
                print(f"Error with reduced batch size: {str(inner_e)}")
                return False
        else:
            print(f"RuntimeError preprocessing {dataset_name}: {str(e)}")
            return False
    except Exception as e:
        print(f"Critical error processing {dataset_name}: {str(e)}")
        return None

def main():
    print("====== Dataset Preprocessing Verification ======")

    # Create directories
    os.makedirs("preprocessed_data", exist_ok=True)
    os.makedirs("preprocessing_analysis", exist_ok=True)

    # Create the ImprovedPreprocessor
    print("\n===== Creating ImprovedPreprocessor with Dataset-Specific Settings =====")
    preprocessor = ImprovedPreprocessor()

    # Process writing_prompts with special settings
    print("\n===== Processing Writing Prompts Dataset with Special Settings =====")
    writing_data = preprocessor.process_dataset("writing_prompts")

    # Log the dataset-specific parameters used
    print(f"Writing Prompts parameters:")
    print(f"  - Batch size: {writing_data['params']['batch_size']}")
    print(f"  - Sequence length: {writing_data['params']['max_sequence_length']}")
    print(f"  - Stride: {writing_data['params']['stride']}")
    print(f"  - Gradient accumulation steps: {writing_data['params']['grad_accum_steps']}")
    print(f"  - Number of batches: {len(writing_data['batches'])}")

    # Analyze token distribution for writing_prompts
    writing_analysis = preprocessor.analyze_token_distribution(writing_data)
    print(f"Unique tokens: {writing_analysis['unique_tokens']} out of {preprocessor.tokenizer.vocab_size}")

    # Save the preprocessed data
    preprocessor.save_tokenized_data(writing_data, "preprocessed_data", "writing_prompts")

    # Process persona_chat with default settings
    print("\n===== Processing Persona Chat Dataset with Default Settings =====")
    persona_data = preprocessor.process_dataset("persona_chat")

    # Log the dataset-specific parameters used
    print(f"Persona Chat parameters:")
    print(f"  - Batch size: {persona_data['params']['batch_size']}")
    print(f"  - Sequence length: {persona_data['params']['max_sequence_length']}")
    print(f"  - Stride: {persona_data['params']['stride']}")
    print(f"  - Gradient accumulation steps: {persona_data['params']['grad_accum_steps']}")
    print(f"  - Number of batches: {len(persona_data['batches'])}")

    # Analyze token distribution for persona_chat
    persona_analysis = preprocessor.analyze_token_distribution(persona_data)
    print(f"Unique tokens: {persona_analysis['unique_tokens']} out of {preprocessor.tokenizer.vocab_size}")

    # Save the preprocessed data
    preprocessor.save_tokenized_data(persona_data, "preprocessed_data", "persona_chat")

    # Compare the datasets
    print("\n===== Dataset Comparison Summary =====")
    print("Writing Prompts (Special Settings):")
    print(f"  - Batch size: {writing_data['params']['batch_size']}")
    print(f"  - Sequence length: {writing_data['params']['max_sequence_length']}")
    print(f"  - Stride: {writing_data['params']['stride']}")
    print(f"  - Gradient accumulation steps: {writing_data['params']['grad_accum_steps']}")

    print("\nPersona Chat (Default Settings):")
    print(f"  - Batch size: {persona_data['params']['batch_size']}")
    print(f"  - Sequence length: {persona_data['params']['max_sequence_length']}")
    print(f"  - Stride: {persona_data['params']['stride']}")
    print(f"  - Gradient accumulation steps: {persona_data['params']['grad_accum_steps']}")

    # Create a comparison visualization
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot batch sizes
        datasets = ['writing_prompts', 'persona_chat']
        batch_sizes = [
            writing_data['params']['batch_size'],
            persona_data['params']['batch_size']
        ]

        ax1.bar(datasets, batch_sizes)
        ax1.set_title('Batch Size Comparison')
        ax1.set_ylabel('Batch Size')

        # Plot sequence lengths
        seq_lengths = [
            writing_data['params']['max_sequence_length'],
            persona_data['params']['max_sequence_length']
        ]

        ax2.bar(datasets, seq_lengths)
        ax2.set_title('Sequence Length Comparison')
        ax2.set_ylabel('Sequence Length')

        plt.tight_layout()
        plt.savefig("preprocessing_analysis/dataset_comparison.png")
        plt.close()
        print("\nSaved dataset comparison visualization to 'preprocessing_analysis/dataset_comparison.png'")
    except Exception as e:
        print(f"Error creating visualization: {e}")

    print("\n====== Preprocessing Verification Complete ======")
    print("Saved all preprocessed data to 'preprocessed_data' directory")

if __name__ == "__main__":
    main()