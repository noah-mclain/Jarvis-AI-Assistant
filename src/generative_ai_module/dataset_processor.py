import os
import torch
import random
import glob
from typing import List, Tuple, Dict, Union, Optional, Any
from .text_generator import TextGenerator, CombinedModel
from .utils import is_zipfile, process_zip

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        "Missing required packages. Please install with: pip install datasets tqdm"
    ) from e

class DatasetProcessor:
    def __init__(self, text_generator: Optional[TextGenerator] = None):
        """
        Initialize the dataset processor
        
        Args:
            text_generator: Optional TextGenerator instance to use for character mappings
        """
        self.text_generator = text_generator or TextGenerator()
        self.sequence_length = 100  # Default sequence length
        
    def load_data(self, source: Union[str, List[str]]) -> str:
        """
        Load text data from various sources (files, directories, zip files)
        
        Args:
            source: Path to file, directory, or list of paths
            
        Returns:
            Combined text data
        """
        combined_text = ""
        
        if isinstance(source, str):
            # Single source
            if os.path.isdir(source):
                # Process directory
                text_files = glob.glob(os.path.join(source, "*.txt"))
                code_files = glob.glob(os.path.join(source, "*.py")) + \
                             glob.glob(os.path.join(source, "*.js")) + \
                             glob.glob(os.path.join(source, "*.java")) + \
                             glob.glob(os.path.join(source, "*.cpp"))
                
                all_files = text_files + code_files
                for file_path in all_files:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        combined_text += f.read() + "\n\n"
            
            elif os.path.isfile(source):
                # Process single file
                if is_zipfile(source):
                    # Handle zip file
                    texts = process_zip(source)
                    combined_text = "\n\n".join(texts)
                else:
                    # Handle regular text file
                    with open(source, 'r', encoding='utf-8', errors='replace') as f:
                        combined_text = f.read()
            else:
                # Treat as raw text
                combined_text = source
                
        elif isinstance(source, list):
            # List of sources
            for item in source:
                combined_text += self.load_data(item) + "\n\n"
                
        return combined_text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        
        Args:
            text: Raw text data
            
        Returns:
            Cleaned text data
        """
        # Remove excessive newlines while preserving important separators
        lines = []
        for line in text.splitlines():
            if line := line.strip():
                # Preserve special tokens and formatting
                if any(token in line for token in ['<PROMPT>', '<STORY>', '<PERSONA>', '<DIALOGUE>', 'USER:', 'ASSISTANT:', '<END>']):
                    lines.append('\n' + line)  # Add extra newline before special tokens
                else:
                    lines.append(line)

        text = '\n'.join(lines)

        return text.replace('\t', '    ')
    
    def create_sequences(self, text: str, sequence_length: int = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create input-target sequences for training
        
        Args:
            text: Text data
            sequence_length: Length of sequences (default: self.sequence_length)
            
        Returns:
            List of (input, target) tensor pairs
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        char_to_idx = self.text_generator.char_to_index
        unknown_token = self.text_generator.unknown_token
        n_chars = len(char_to_idx)
        
        # Create character-level sequences
        sequences = []
        for i in range(len(text) - sequence_length - 1):
            # Input is sequence_length characters
            input_seq = text[i:i+sequence_length]
            # Target is the next character
            target_char = text[i+sequence_length]
            
            # Convert to indices
            input_tensor = torch.zeros(sequence_length, n_chars)
            for t, char in enumerate(input_seq):
                idx = char_to_idx.get(char, char_to_idx[unknown_token])
                input_tensor[t, idx] = 1.0
                
            target_idx = char_to_idx.get(target_char, char_to_idx[unknown_token])
            target_tensor = torch.tensor([target_idx])
            
            sequences.append((input_tensor, target_tensor))
            
        return sequences
    
    def create_batches(self, sequences: List[Tuple[torch.Tensor, torch.Tensor]], 
                      batch_size: int = 64, shuffle: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create batches from sequences
        
        Args:
            sequences: List of (input, target) tensor pairs
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            List of batched (inputs, targets) tensor pairs
        """
        if shuffle:
            random.shuffle(sequences)
            
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            if not batch_sequences:
                continue
                
            # Unzip the batch sequences
            batch_inputs, batch_targets = zip(*batch_sequences)
            
            # Stack tensors into batches
            input_batch = torch.stack(batch_inputs)
            target_batch = torch.stack(batch_targets)
            
            batches.append((input_batch, target_batch))
            
        return batches
    
    def prepare_text_batches(self, raw_text, sequence_length, batch_size):
        """
        Prepare text into batches for training
        
        Args:
            raw_text: Raw text data
            sequence_length: Length of sequences
            batch_size: Batch size
            
        Returns:
            Batched dataset ready for training
        """
        cleaned_text = self.clean_text(raw_text)
        sequences = self.create_sequences(cleaned_text, sequence_length)
        return self.create_batches(sequences, batch_size)
    
    def load_writing_prompts(self, split='train', max_samples=None, cache_dir=None):
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
        if not self._generate_sample_writing_prompts():
            print("Warning: Failed to generate sample writing prompts data")

        try:
            # Try loading the dataset
            dataset = load_dataset("writingprompts", split=split, cache_dir=cache_dir)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            return "".join(
                f"<PROMPT>\n{item['prompt']}\n<STORY>\n{item['story']}\n<END>\n\n"
                for item in tqdm(dataset, desc="Processing Writing Prompts")
            )
        except Exception as e:
            print(f"Error loading Writing Prompts dataset: {e}")
            print("Falling back to sample data")
            return self._generate_sample_writing_prompts()
    
    def _generate_sample_writing_prompts(self):
        """Generate a small sample of writing prompts data"""
        prompts = [
            "The world ended five minutes ago. You're the only one who knows.",
            "You discover that your everyday life is actually a virtual reality game.",
            "You wake up in a world where everyone can read minds, except you."
        ]
        
        stories = [
            "I stared at my watch in disbelief. Five minutes ago, everything changed...",
            "The loading screen appeared before my eyes as I tried to reach for my coffee...",
            "They all looked at me strangely, as if they knew something I didn't..."
        ]
        
        text = ""
        for p, s in zip(prompts, stories):
            text += f"<PROMPT>\n{p}\n<STORY>\n{s}\n<END>\n\n"
            
        return text
            
    def load_persona_chat(self, split='train', max_samples=None, cache_dir=None):
        """
        Load and preprocess the Persona Chat dataset
        
        Args:
            split: Dataset split ('train', 'test', or 'validation')
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Preprocessed text ready for sequence creation
        """
        # Load dataset
        try:
            dataset = self._load_persona_chat_dataset(split, cache_dir)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            return self._process_persona_chat_items(dataset)
        except Exception as e:
            print(f"Error loading Persona Chat dataset: {e}")
            print("Falling back to sample data")
            return self._generate_sample_persona_chat()
        
    def _load_persona_chat_dataset(self, split, cache_dir):
        """Helper to load the right persona chat dataset"""
        try:
            return load_dataset("bavard/personachat_truecased", split=split, cache_dir=cache_dir)
        except Exception:
            return load_dataset("persona_chat", split=split, cache_dir=cache_dir)
    
    def _print_dataset_structure(self, item):
        """Debug helper to print dataset structure"""
        print(f"Dataset keys: {item.keys()}")
        for k, v in item.items():
            print(f"{k}: {type(v)}")
            if isinstance(v, list) and v:
                print(f"  First item type: {type(v[0])}")
                print(f"  Example: {v[0]}")
    
    def _process_persona_chat_items(self, dataset):
        """Process persona chat dataset items into text format"""
        compiled_text = ""
        
        # Look at first item to determine format
        first_item = dataset[0]
        
        if 'personas' in first_item or 'personality' in first_item:
            # Use standardized extraction
            for item in tqdm(dataset, desc="Processing Persona Chat"):
                # Extract persona and dialogue
                persona_text = self._extract_persona_text(item)
                dialogue_text = self._extract_dialogue_text(item)
                
                # Compile the complete text
                text = f"<PERSONA>\n{persona_text}<DIALOGUE>\n{dialogue_text}<END>\n\n"
                compiled_text += text
                
        return compiled_text
    
    def _extract_persona_text(self, item):
        """Extract persona information from different dataset formats"""
        persona_text = ""
        
        # Handle different persona formats
        if 'personas' in item:
            # Bavard dataset format
            for persona in item['personas']['persona_1']:
                persona_text += f"- {persona}\n"
        elif 'personality' in item:
            # Original persona chat format
            for persona in item['personality']:
                persona_text += f"- {persona}\n"
        
        return persona_text
    
    def _extract_dialogue_text(self, item):
        """Extract dialogue from different dataset formats"""
        dialogue_text = ""
        
        # Handle different dialogue formats
        if 'utterances' in item:
            # Original personachat format
            utterances = item['utterances'][-1]['history']
            dialogue_turns = self._process_dialogue_turns(utterances)
            dialogue_text = dialogue_turns
        elif 'dialog' in item:
            # Bavard format
            dialog = item['dialog']
            turns = []
            for i, utt in enumerate(dialog):
                prefix = "USER: " if i % 2 == 0 else "ASSISTANT: "
                turns.append(f"{prefix}{utt}")
            dialogue_text = "\n".join(turns)
        elif 'history' in item:
            # Another possible format
            history = item['history'] 
            dialogue_turns = self._process_dialogue_turns(history)
            dialogue_text = dialogue_turns
            
        return dialogue_text + "\n"
    
    def _process_dialogue_turns(self, dialogue):
        """Convert dialogue list to alternating user/assistant turns"""
        result = []
        for i, utterance in enumerate(dialogue):
            # Skip empty lines
            if not utterance.strip():
                continue
                
            # Add appropriate speaker prefix
            if i % 2 == 0:
                result.append(f"USER: {utterance}")
            else:
                result.append(f"ASSISTANT: {utterance}")
        
        return "\n".join(result)
    
    def _generate_sample_persona_chat(self):
        """Generate a small sample of persona chat data"""
        return """<PERSONA>
- I am a teacher.
- I have two dogs named Max and Ruby.
- I love to cook Italian food.
- I've been to seven different countries.
<DIALOGUE>
USER: Hi there! How's your day going?
ASSISTANT: Hello! My day is going well. I just got home from teaching and took my dogs Max and Ruby for a walk. How about you?
USER: I'm doing fine. What do you teach?
ASSISTANT: I teach high school mathematics. I really enjoy it, especially when students have those "aha" moments. When I'm not teaching, I love cooking Italian dishes.
USER: That's cool! I love Italian food too. Have you ever been to Italy?
ASSISTANT: Yes! Italy was one of my favorite countries to visit. I've been to seven countries in total, and Italy was definitely a highlight. I learned some amazing pasta recipes there that I still make.
<END>

<PERSONA>
- I work as a software engineer.
- I play guitar in a band on weekends.
- I am training for a marathon.
- I prefer tea over coffee.
<DIALOGUE>
USER: Hey, how's it going?
ASSISTANT: Hi there! Just got back from my morning run, training for a marathon. Now having some tea before I start work. How are you?
USER: I'm good. What kind of work do you do?
ASSISTANT: I'm a software engineer during the week. But on weekends, I play guitar in a small band. It's a nice balance between technical and creative work.
<END>
"""
    
    def prepare_dialogue_dataset(self, source='persona_chat', split='train', 
                               sequence_length=100, batch_size=64, max_samples=None, 
                               cache_dir=None):
        """
        Prepare dialogue dataset for training
        
        Args:
            source: Source dataset ('persona_chat' or 'writing_prompts')
            split: Dataset split ('train', 'test', or 'validation')
            sequence_length: Length of sequences
            batch_size: Batch size
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Batched dataset ready for training
        """
        # Load and preprocess text data
        if source == 'persona_chat':
            raw_text = self.load_persona_chat(split, max_samples, cache_dir)
        elif source == 'writing_prompts':
            raw_text = self.load_writing_prompts(split, max_samples, cache_dir)
        else:
            # Treat as path to local file or directory
            raw_text = self.load_data(source)
        
        # Create batches
        return self.prepare_text_batches(raw_text, sequence_length, batch_size)
    
    def prepare_code_dataset(self, source, sequence_length=100, batch_size=64):
        """
        Prepare code dataset for training
        
        Args:
            source: Path to file, directory, or list of paths containing code
            sequence_length: Length of sequences
            batch_size: Batch size
            
        Returns:
            Batched dataset ready for training
        """
        # Load code text
        raw_text = self.load_data(source)
        
        # Prepare batches
        return self.prepare_text_batches(raw_text, sequence_length, batch_size)
    
    def prepare_local_dataset(self, data_dir, sequence_length=100, batch_size=64):
        """
        Prepare dataset from local files
        
        Args:
            data_dir: Directory containing text files
            sequence_length: Length of sequences
            batch_size: Batch size
            
        Returns:
            Batched dataset ready for training
        """
        # Check if path exists
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory {data_dir} does not exist")
            return []
        
        # Load data
        raw_text = ""
        
        if os.path.isdir(data_dir):
            # Compile all .txt files in the directory
            file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
            
            if not file_paths:
                print(f"Warning: No .txt files found in {data_dir}")
                return []
                
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        raw_text += f.read() + "\n\n"
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
        elif os.path.isfile(data_dir) and data_dir.endswith('.txt'):
            # Single text file
            with open(data_dir, 'r', encoding='utf-8', errors='replace') as f:
                raw_text = f.read()
        else:
            print(f"Warning: {data_dir} is not a valid directory or .txt file")
            return []
            
        # Prepare batches
        print(f"Loaded {len(raw_text)} characters from {data_dir}")
        return self.prepare_text_batches(raw_text, sequence_length, batch_size)
    
    def load_preprocessed_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load preprocessed data for a specific dataset"""
        # Define the correct path for preprocessed data
        preprocessed_path = os.path.join(
            "src", "generative_ai_module", "examples", "preprocessed_data",
            f"{dataset_name}_preprocessed.pt"
        )
        
        # Check if the file exists
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_path}")
        
        # Load the data
        try:
            data = torch.load(preprocessed_path)
            print(f"Loaded preprocessed data from {preprocessed_path}")
            return data
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return None

    def save_tokenized_data(self, data: Dict[str, Any], output_dir: str, dataset_name: str):
        """Save tokenized data to disk"""
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the preprocessed data
        output_path = os.path.join(output_dir, f"{dataset_name}_preprocessed.pt")
        torch.save(data, output_path)
        print(f"Saved preprocessed data to {output_path}")
        
        # Save a sample of the text for inspection
        sample_path = os.path.join(output_dir, f"{dataset_name}_sample.txt")
        with open(sample_path, 'w') as f:
            if 'batches' in data and len(data['batches']) > 0:
                sample_batch = data['batches'][0]
                if isinstance(sample_batch, tuple) and len(sample_batch) > 0:
                    sample_text = self.decode_tokens(sample_batch[0][0].tolist())
                    f.write(sample_text)
        print(f"Saved text sample to {sample_path}")
    
    def prepare_from_preprocessed(self, dataset_name="persona_chat", preprocessed_path=None,
                                batch_size=None):
        """
        Prepare batches from preprocessed data
        
        Args:
            dataset_name: Name of the preprocessed dataset
            preprocessed_path: Optional specific path to the preprocessed file
            batch_size: Optional batch size to reshape batches (None to keep original)
            
        Returns:
            List of (input_batch, target_batch) tuples
        """
        # Load the preprocessed data
        try:
            data = self.load_preprocessed_data(dataset_name)

            # Check if data contains batches
            if 'batches' not in data or not data['batches']:
                print("Warning: No batches found in preprocessed data")
                return []

            # If no reshaping is needed, return the batches directly
            if batch_size is None:
                return data['batches']

            # Otherwise, reshape the batches
            print(f"Reshaping batches to size {batch_size}")

            # Create a flat list of samples
            flat_samples = []
            for input_batch, target_batch in data['batches']:
                flat_samples.extend(
                    (input_batch[i], target_batch[i])
                    for i in range(input_batch.shape[0])
                )
            # Create new batches
            return self.create_batches(flat_samples, batch_size)

        except Exception as e:
            print(f"Error preparing from preprocessed data: {e}")
            return []
    
    def adapt_preprocessed_data(self, data):
        """
        Adapt preprocessed data format to match what the model expects
        
        Args:
            data: Dictionary containing preprocessed data
            
        Returns:
            Dictionary with adapted data batches
        """
        if 'batches' not in data or not data['batches']:
            print("WARNING: No batches found in preprocessed data")
            return data
        
        adapted_batches = []
        
        for i, (inputs, targets) in enumerate(data['batches']):
            # Print the shape and structure of a few batches for debugging
            if i < 3:
                print(f"Original batch {i} shapes: inputs {inputs.shape}, targets {targets.shape}")
                print(f"Inputs data type: {inputs.dtype}, Targets data type: {targets.dtype}")
            
            # Ensure inputs are in the right shape
            if inputs.dim() == 2:
                # If [batch_size, features], keep as is - already correct format
                adapted_inputs = inputs
            elif inputs.dim() == 3 and inputs.shape[1] == 1:
                # If [batch_size, 1, features], remove the sequence dimension
                adapted_inputs = inputs.squeeze(1)
            else:
                # Keep as is
                adapted_inputs = inputs
                
            # Ensure targets are in the right shape
            if targets.dim() > 1 and targets.shape[1] == 1:
                # If [batch_size, 1], squeeze to [batch_size]
                adapted_targets = targets.squeeze(1)
            else:
                # Keep as is
                adapted_targets = targets
                
            adapted_batches.append((adapted_inputs, adapted_targets))
            
            # Print adapted shape for a few batches for debugging
            if i < 3:
                print(f"Adapted batch {i} shapes: inputs {adapted_inputs.shape}, targets {adapted_targets.shape}")
        
        # Create a new data dictionary with adapted batches
        adapted_data = data.copy()
        adapted_data['batches'] = adapted_batches
        
        print(f"Adapted {len(adapted_batches)} batches")
        return adapted_data
    
    def initialize_with_tokenizer(self, tokenizer, vocab_size=None):
        """
        Initialize text generator with tokenizer
        
        Args:
            tokenizer: Tokenizer object with encode/decode methods
            vocab_size: Vocabulary size (if None, will be determined from tokenizer)
            
        Returns:
            None (modifies the text_generator in place)
        """
        if not hasattr(self.text_generator, 'model'):
            print("Warning: TextGenerator does not have a model attribute")
            return

        # Determine vocabulary size
        if vocab_size is None:
            vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 128
        # Create new model with the correct vocabulary size
        input_size = vocab_size
        hidden_size = 128  # Default value
        output_size = vocab_size
        num_layers = 2  # Default value

        print(f"Reinitializing model with vocabulary size: {vocab_size}")
        self.text_generator.model = CombinedModel(input_size, hidden_size, output_size, num_layers)
        self.text_generator.optimizer = torch.optim.Adam(self.text_generator.model.parameters(), lr=0.002)

        # Create a dummy char index mapping for integer tokens
        self.text_generator.char_to_index = {str(i): i for i in range(vocab_size)}
        self.text_generator.index_to_char = {i: str(i) for i in range(vocab_size)}

        # Special handling for common tokens if we know what they represent
        common_tokens = {
            0: '<PAD>',
            1: '<START>',
            2: '<END>',
            5: ' ',  # Assuming 5 is the space character
        }
        self.text_generator.index_to_char |= common_tokens

        # Set up token adaptation
        self.text_generator.adapt_to_tokenizer(tokenizer)
        print("TextGenerator initialized with tokenizer")