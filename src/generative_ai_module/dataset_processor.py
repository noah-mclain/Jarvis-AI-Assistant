import os
import torch
import random
import glob
from typing import List, Tuple, Dict, Union, Optional
from .text_generator import TextGenerator
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
        for i in range(0, len(text) - sequence_length - 1):
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
        try:
            # Use the correct identifier provided by the user
            dataset = load_dataset("euclaise/writingprompts", split=split, cache_dir=cache_dir)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Combine prompts and stories with clear separators
            combined_text = ""
            for item in tqdm(dataset, desc="Processing Writing Prompts"):
                # Format: <PROMPT> prompt text <STORY> story text <END>
                combined_text += f"<PROMPT> {item['prompt'].strip()}\n"
                combined_text += f"<STORY> {item['story'].strip()}\n"
                combined_text += "<END>\n\n"
                
            return combined_text
            
        except Exception as e:
            print(f"Error loading Writing Prompts dataset: {e}")
            # Fallback to sample data if needed
            return self._generate_sample_writing_prompts()

    def _generate_sample_writing_prompts(self):
        """Generate a small sample dataset for testing when the real dataset can't be loaded"""
        samples = [
            {
                "prompt": "You wake up one day with the ability to see 10 seconds into the future.",
                "story": "I blinked rapidly, trying to make sense of what was happening. The world seemed to flicker between now and... something else. I watched as my coffee mug tipped over before it actually happened, giving me just enough time to catch it. This was going to be interesting."
            },
            {
                "prompt": "Aliens have been watching our TV shows for decades. They finally make contact.",
                "story": "The massive ship hovered silently above New York. Everyone waited anxiously for first contact. The alien ambassador emerged and said, 'We come in peace. Also, we need to know if Ross and Rachel ever got back together. Our transmission cut out during season 8.'"
            }
        ]
        
        combined_text = ""
        for sample in samples:
            combined_text += f"<PROMPT> {sample['prompt']}\n"
            combined_text += f"<STORY> {sample['story']}\n"
            combined_text += "<END>\n\n"
            
        return combined_text
    
    def load_persona_chat(self, split='train', max_samples=None, cache_dir=None):
        """
        Load and preprocess the Synthetic Persona Chat dataset
        
        Args:
            split: Dataset split ('train', 'test', or 'validation')
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Preprocessed text ready for sequence creation
        """
        try:
            # Load the dataset
            dataset = self._load_persona_chat_dataset(split, cache_dir)
            
            # Print dataset structure for debugging
            self._print_dataset_structure(dataset[0])
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Process the dataset
            combined_text = self._process_persona_chat_items(dataset)
            return combined_text
            
        except Exception as e:
            print(f"Error loading Persona Chat dataset: {e}")
            print("Trying to generate fallback sample data...")
            return self._generate_sample_persona_chat()

    def _load_persona_chat_dataset(self, split, cache_dir):
        """Load the persona chat dataset with error handling"""
        try:
            return load_dataset("google/Synthetic-Persona-Chat", split=split, cache_dir=cache_dir)
        except Exception as e:
            print(f"Primary dataset source failed: {e}")
            return load_dataset("facebook/personachat", split=split, cache_dir=cache_dir)

    def _print_dataset_structure(self, item):
        """Print the structure of a dataset item for debugging"""
        print("Dataset structure example:")
        for key in item.keys():
            print(f"- Field: {key}, Type: {type(item[key])}")

    def _process_persona_chat_items(self, dataset):
        """Process all items in the persona chat dataset"""
        combined_text = ""
        for item in tqdm(dataset, desc="Processing Persona Chat"):
            # Add persona section
            combined_text += "<PERSONA>\n"
            combined_text += self._extract_persona_text(item)
            
            # Add dialogue section
            combined_text += "<DIALOGUE>\n"
            combined_text += self._extract_dialogue_text(item)
            combined_text += "<END>\n\n"
            
        return combined_text

    def _extract_persona_text(self, item):
        """Extract persona information from a dataset item"""
        # Handle Google Synthetic Persona Chat format
        if 'user 1 personas' in item:
            return f"- {item['user 1 personas'].strip()}\n"
        
        # Handle other formats (keep existing code)
        if 'persona' in item:
            personas = item['persona'] if isinstance(item['persona'], list) else [item['persona']]
            return "\n".join(f"- {persona.strip()}" for persona in personas) + "\n"
        
        if 'context' in item:
            return f"- {item['context'].strip()}\n"
        
        return "- No persona information available\n"

    def _extract_dialogue_text(self, item):
        """Extract dialogue information from a dataset item"""
        result = ""
        
        # Handle Google Synthetic Persona Chat format
        if 'Best Generated Conversation' in item:
            # Parse the conversation text into turns
            conversation = item['Best Generated Conversation']
            lines = conversation.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('User:'):
                    result += f"USER: {line[5:].strip()}\n"
                elif line.startswith('Assistant:'):
                    result += f"ASSISTANT: {line[10:].strip()}\n"
            
            return result
        
        # Handle other formats (keep existing code)
        if 'dialogue' in item:
            result += self._process_dialogue_turns(item['dialogue'])
        elif 'input' in item and 'output' in item:
            result += f"USER: {item['input'].strip()}\n"
            result += f"ASSISTANT: {item['output'].strip()}\n"
        
        return result

    def _process_dialogue_turns(self, dialogue):
        """Process dialogue turns in various formats"""
        result = ""
        turns = dialogue if isinstance(dialogue, list) else [dialogue]
        
        for i, turn in enumerate(turns):
            if isinstance(turn, dict):
                if 'user' in turn:
                    result += f"USER: {turn['user'].strip()}\n"
                if 'assistant' in turn:
                    result += f"ASSISTANT: {turn['assistant'].strip()}\n"
            elif i % 2 == 0:  # Assume even indices are user, odd are assistant
                result += f"USER: {turn.strip()}\n"
            else:
                result += f"ASSISTANT: {turn.strip()}\n"
        
        return result

    def _generate_sample_persona_chat(self):
        """Generate a small sample dataset for testing when the real dataset can't be loaded"""
        samples = [
            {
                "persona": ["I love hiking in the mountains.", "I have a dog named Max.", "I work as a software engineer."],
                "dialogue": [
                    {"user": "Hi there! Do you like outdoor activities?", 
                     "assistant": "Yes, I love hiking in the mountains. Do you enjoy hiking too?"},
                    {"user": "I do! What's your favorite hiking spot?",
                     "assistant": "I really enjoy trails in the Rocky Mountains. I often take my dog Max with me."}
                ]
            },
            {
                "persona": ["I am a chef at a restaurant.", "I enjoy classical music.", "I have visited 15 countries."],
                "dialogue": [
                    {"user": "What do you do for a living?",
                     "assistant": "I'm a chef at a restaurant. I specialize in Italian cuisine. Do you enjoy cooking?"},
                    {"user": "That's cool! What's your favorite dish to prepare?",
                     "assistant": "I love making homemade pasta with fresh ingredients. The process is as relaxing as listening to classical music."}
                ]
            }
        ]
        
        combined_text = ""
        for sample in samples:
            combined_text += "<PERSONA>\n"
            for persona in sample["persona"]:
                combined_text += f"- {persona}\n"
            
            combined_text += "<DIALOGUE>\n"
            for turn in sample["dialogue"]:
                combined_text += f"USER: {turn['user']}\n"
                combined_text += f"ASSISTANT: {turn['assistant']}\n"
            
            combined_text += "<END>\n\n"
            
        return combined_text
    
    def prepare_dialogue_dataset(self, source='persona_chat', split='train', 
                               sequence_length=100, batch_size=64, max_samples=None, 
                               cache_dir=None):
        """
        Prepare dialogue datasets (Persona Chat or Writing Prompts) for training
        
        Args:
            source: Dataset source ('persona_chat' or 'writing_prompts')
            split: Dataset split
            sequence_length: Length of sequences
            batch_size: Batch size
            max_samples: Maximum number of samples to load
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Batched dataset ready for training
        """
        # Reinitialize the text generator model for this sequence length
        # This ensures the model can handle the requested sequence length
        if hasattr(self.text_generator, 'reinitialize_for_sequence_length'):
            self.text_generator.reinitialize_for_sequence_length(sequence_length)
        
        self.sequence_length = sequence_length
        
        if source == 'persona_chat':
            raw_text = self.load_persona_chat(split, max_samples, cache_dir)
        elif source == 'writing_prompts':
            raw_text = self.load_writing_prompts(split, max_samples, cache_dir)
        else:
            raise ValueError(f"Unsupported dataset source: {source}. Choose 'persona_chat' or 'writing_prompts'")
        
        # Clean and prepare sequences
        cleaned_text = self.clean_text(raw_text)
        
        # Check if we have enough text data
        if len(cleaned_text) <= sequence_length:
            raise ValueError(f"Text length ({len(cleaned_text)}) must be greater than sequence length ({sequence_length})")
            
        sequences = self.create_sequences(cleaned_text, sequence_length)
        
        if not sequences:
            raise ValueError("No sequences were created. Check your dataset and sequence length.")
            
        batches = self.create_batches(sequences, batch_size)
        
        if not batches:
            raise ValueError("No batches were created. Check your batch size and dataset size.")
            
        print(f"Created {len(batches)} batches from {len(sequences)} sequences")
        return batches
    
    def prepare_local_dataset(self, data_dir, sequence_length=100, batch_size=64):
        """
        Prepare a dataset from local text files
        
        Args:
            data_dir: Directory containing text files
            sequence_length: Length of sequences
            batch_size: Batch size
            
        Returns:
            Batched dataset ready for training
        """
        self.sequence_length = sequence_length
        
        # Load all text files from the directory
        text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        if not text_files:
            raise ValueError(f"No text files found in {data_dir}")
        
        print(f"Found {len(text_files)} text files in {data_dir}")
        
        # Load and combine text
        raw_text = ""
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_text += f.read() + "\n\n"
        
        # Clean and prepare sequences
        cleaned_text = self.clean_text(raw_text)
        
        # Check if we have enough text data
        if len(cleaned_text) <= sequence_length:
            raise ValueError(f"Text length ({len(cleaned_text)}) must be greater than sequence length ({sequence_length})")
            
        sequences = self.create_sequences(cleaned_text, sequence_length)
        batches = self.create_batches(sequences, batch_size)
        
        print(f"Created {len(batches)} batches from {len(sequences)} sequences")
        return batches