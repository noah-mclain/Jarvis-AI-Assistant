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
                # Ensure the index is within bounds of the tensor dimension
                if idx >= n_chars:
                    idx = char_to_idx[unknown_token]  # Default to unknown token if out of range
                input_tensor[t, idx] = 1.0
                
            target_idx = char_to_idx.get(target_char, char_to_idx[unknown_token])
            # Ensure target index is within bounds
            if target_idx >= n_chars:
                target_idx = char_to_idx[unknown_token]
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
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        preprocessed_path = os.path.join(
            root_dir, "preprocessed_data", f"{dataset_name}_preprocessed.pt"
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

    def load_pile_dataset(self, subset=None, split='train', max_samples=None, cache_dir=None):
        """
        Load and preprocess data from The Pile dataset
        
        Args:
            subset: Specific subset of The Pile (None for the default mix)
            split: Dataset split ('train', 'test', or 'validation')
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Preprocessed text ready for sequence creation
        """
        try:
            # Load the dataset with the specified subset
            if subset:
                dataset = load_dataset("EleutherAI/pile", subset, split=split, cache_dir=cache_dir)
            else:
                dataset = load_dataset("EleutherAI/pile", split=split, cache_dir=cache_dir)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Process the samples
            compiled_text = ""
            for item in tqdm(dataset, desc=f"Processing Pile{' ('+subset+')' if subset else ''}"):
                text = item['text']
                meta = item.get('meta', {})
                
                # Add source information as metadata comment
                source_info = meta.get('pile_set_name', 'Unknown')
                compiled_text += f"<SOURCE>\n{source_info}\n<TEXT>\n{text}\n<END>\n\n"
            
            return compiled_text
                
        except Exception as e:
            print(f"Error loading The Pile dataset: {e}")
            return self._generate_sample_pile_data()
    
    def _generate_sample_pile_data(self):
        """Generate a small sample of Pile-like data"""
        return """<SOURCE>
Pile-CC
<TEXT>
The concept of artificial intelligence has fascinated humanity for decades. From the early works of science fiction to the development of machine learning algorithms, we have sought to create machines that can think and learn like humans.
<END>

<SOURCE>
PubMed Central
<TEXT>
Abstract
The human microbiome plays a crucial role in health and disease. Recent studies have shown that the gut microbiota affects numerous physiological processes, including digestion, immune function, and even neurological development.
<END>

<SOURCE>
GitHub
<TEXT>
def preprocess_text(text):
    '''
    Clean and normalize text data
    
    Args:
        text: Raw text data
        
    Returns:
        Cleaned text data
    '''
    return text.lower().strip()
<END>
"""
    
    def load_openassistant_dataset(self, split='train', max_samples=None, cache_dir=None):
        """
        Load and preprocess data from the OpenAssistant dataset
        
        Args:
            split: Dataset split ('train', 'test', or 'validation')
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Preprocessed text ready for sequence creation
        """
        try:
            dataset = load_dataset("agie-ai/OpenAssistant-oasst1", split=split, cache_dir=cache_dir)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Process the samples
            compiled_text = ""
            for item in tqdm(dataset, desc="Processing OpenAssistant"):
                if 'text' in item and 'role' in item:
                    # Simple format with just text and role
                    role = item['role'].upper()
                    text = item['text']
                    
                    compiled_text += f"<{role}>\n{text}\n"
                elif 'messages' in item:
                    # Format with conversation messages
                    compiled_text += "<CONVERSATION>\n"
                    
                    for msg in item['messages']:
                        role = msg.get('role', 'unknown').upper()
                        content = msg.get('content', '')
                        compiled_text += f"{role}: {content}\n"
                    
                    compiled_text += "<END>\n\n"
                else:
                    # Try to handle format with message pairs
                    instruction = item.get('instruction', item.get('prompt', ''))
                    response = item.get('response', item.get('completion', ''))
                    
                    if instruction and response:
                        compiled_text += f"USER: {instruction}\nASSISTANT: {response}\n<END>\n\n"
            
            return compiled_text
            
        except Exception as e:
            print(f"Error loading OpenAssistant dataset: {e}")
            return self._generate_sample_assistant_data()
    
    def _generate_sample_assistant_data(self):
        """Generate a small sample of assistant data"""
        return """<CONVERSATION>
USER: How does photosynthesis work?
ASSISTANT: Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose (sugar) and oxygen. The process occurs in the chloroplasts of plant cells, specifically using the green pigment chlorophyll to capture light energy. This energy is used to split water molecules, releasing oxygen as a byproduct. The hydrogen from water and carbon dioxide are then used to create glucose, which serves as food for the plant.
<END>

<CONVERSATION>
USER: What are some effective ways to manage time?
ASSISTANT: Effective time management strategies include:
1. Prioritizing tasks using methods like the Eisenhower Matrix (urgent/important)
2. Setting SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)
3. Using the Pomodoro Technique (25-minute focused work sessions with short breaks)
4. Batching similar tasks together
5. Planning your day the night before
6. Using digital tools like calendar apps and task managers
7. Learning to delegate and say no when appropriate
8. Identifying and eliminating time-wasting activities

The key is finding what works for your specific situation and consistently applying those methods.
<END>
"""
    
    def load_gpteacher_dataset(self, split='train', max_samples=None, cache_dir=None):
        """
        Load and preprocess data from the GPTeacher dataset
        
        Args:
            split: Dataset split ('train', 'test', or 'validation')
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            
        Returns:
            Preprocessed text ready for sequence creation
        """
        try:
            dataset = load_dataset("teknium/GPTeacher-General-Instruct", split=split, cache_dir=cache_dir)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Process the samples
            compiled_text = ""
            for item in tqdm(dataset, desc="Processing GPTeacher"):
                # Extract instruction and response
                instruction = item.get('instruction', '')
                response = item.get('response', '')
                context = item.get('context', '')
                
                if context:
                    compiled_text += f"<CONTEXT>\n{context}\n"
                
                compiled_text += f"<INSTRUCTION>\n{instruction}\n<RESPONSE>\n{response}\n<END>\n\n"
            
            return compiled_text
            
        except Exception as e:
            print(f"Error loading GPTeacher dataset: {e}")
            return self._generate_sample_instruction_data()
    
    def _generate_sample_instruction_data(self):
        """Generate a small sample of instruction data"""
        return """<INSTRUCTION>
Write a short poem about autumn leaves.
<RESPONSE>
Crimson and gold, they dance and sway,
Autumn leaves on a crisp, cool day.
Floating gently to the ground below,
Nature's confetti in the wind's gentle blow.
Crunching softly beneath passing feet,
Earth's carpet, vibrant and sweet.
<END>

<CONTEXT>
The user is trying to understand how to approach a math problem.
<INSTRUCTION>
I'm struggling with calculus derivatives. Can you explain the chain rule?
<RESPONSE>
The chain rule is a formula for computing the derivative of a composite function. If you have a function f(g(x)), the chain rule states that:

d/dx[f(g(x))] = f'(g(x)) · g'(x)

In words: the derivative of the composite function equals the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function.

Example:
Let's find the derivative of h(x) = sin(x²)
- Here f(x) = sin(x) and g(x) = x²
- We know that f'(x) = cos(x) and g'(x) = 2x
- Using the chain rule: h'(x) = cos(x²) · 2x = 2x·cos(x²)

The chain rule is powerful because it allows you to break down complex derivatives into smaller, manageable steps.
<END>
"""
    
    def prepare_dataset(self, source='persona_chat', split='train', 
                       sequence_length=100, batch_size=64, max_samples=None, 
                       cache_dir=None, subset=None, output_dir=None):
        """
        Prepare a dataset for training
        
        Args:
            source: Dataset name/path, can be a standard dataset or a HuggingFace dataset name (e.g., 'agie-ai/OpenAssistant-oasst1')
            split: Dataset split (train, validation, test)
            sequence_length: Length of sequences
            batch_size: Batch size
            max_samples: Maximum number of samples to load
            cache_dir: Optional directory to cache the dataset
            subset: Optional subset name for datasets with subsets
            output_dir: Optional directory to save processed datasets
            
        Returns:
            Dictionary with dataset information
        """
        self.sequence_length = sequence_length
        
        # Check if source is a HuggingFace dataset (containing a slash)
        is_huggingface_dataset = '/' in source
        
        # Output directory for processed datasets
        if output_dir is None:
            output_dir = os.path.join('datasets', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # First, check if we already have a preprocessed version
        dataset_name = source.replace('/', '_') if is_huggingface_dataset else source
        output_path = os.path.join(output_dir, f"{dataset_name}_{split}_preprocessed.pt")
        
        if os.path.exists(output_path):
            print(f"Loading preprocessed dataset from {output_path}...")
            return self.load_preprocessed_data(output_path)
        
        print(f"Preparing dataset: {source} (split: {split})")
        
        # Special handling for standard datasets
        if not is_huggingface_dataset:
            if source == 'writing_prompts':
                raw_text = self.load_writing_prompts(split=split, max_samples=max_samples, cache_dir=cache_dir)
            elif source == 'persona_chat':
                raw_text = self.load_persona_chat(split=split, max_samples=max_samples, cache_dir=cache_dir)
            elif source == 'pile':
                raw_text = self.load_pile_dataset(subset=subset, split=split, max_samples=max_samples, cache_dir=cache_dir)
            elif source == 'openassistant':
                raw_text = self.load_openassistant_dataset(split=split, max_samples=max_samples, cache_dir=cache_dir)
            elif source == 'gpteacher':
                raw_text = self.load_gpteacher_dataset(split=split, max_samples=max_samples, cache_dir=cache_dir)
            elif os.path.exists(source):
                # Local file or directory
                raw_text = self.load_data(source)
            else:
                raise ValueError(f"Unknown dataset source: {source}")
        else:
            # Handle HuggingFace datasets directly
            try:
                print(f"Loading HuggingFace dataset: {source}")
                hf_dataset = load_dataset(source, split=split, cache_dir=cache_dir)
                
                # Limit samples if specified
                if max_samples is not None and max_samples < len(hf_dataset):
                    hf_dataset = hf_dataset.select(range(max_samples))
                    
                # Extract text based on dataset format
                raw_text = self._process_huggingface_dataset(hf_dataset, source)
                
            except Exception as e:
                print(f"Error loading HuggingFace dataset {source}: {str(e)}")
                raise
        
        # Create sequences and batches
        print(f"Creating sequences with length {sequence_length}...")
        sequences = self.create_sequences(raw_text, sequence_length)
        
        print(f"Creating batches with batch size {batch_size}...")
        batches = self.create_batches(sequences, batch_size=batch_size)
        
        dataset = {
            'batches': batches,
            'metadata': {
                'source': source,
                'split': split,
                'sequence_length': sequence_length,
                'batch_size': batch_size,
                'sample_count': len(sequences),
                'batch_count': len(batches)
            }
        }
        
        # Save preprocessed data for future use
        print(f"Saving preprocessed dataset to {output_path}...")
        self.save_tokenized_data(dataset, os.path.dirname(output_path), os.path.basename(output_path))
        
        return dataset

    def _process_huggingface_dataset(self, dataset, source_name):
        """
        Process a HuggingFace dataset into a format suitable for training
        
        Args:
            dataset: HuggingFace dataset object
            source_name: Name of the dataset source
            
        Returns:
            Processed text ready for sequence creation
        """
        print(f"Processing HuggingFace dataset with {len(dataset)} samples")
        
        # Try to find text fields based on common field names
        text_fields = []
        sample_item = dataset[0] if len(dataset) > 0 else {}
        
        # Check for known text fields
        potential_fields = ['text', 'content', 'dialogue', 'prompt', 'completion', 
                            'input', 'output', 'question', 'answer', 'instruction',
                            'response', 'conversation', 'source']
        
        for field in potential_fields:
            if field in sample_item and isinstance(sample_item[field], str):
                text_fields.append(field)
        
        # Special handling for specific datasets
        if 'OpenAssistant-oasst1' in source_name:
            return self._process_openassistant_huggingface(dataset)
        elif 'GPTeacher' in source_name:
            return self._process_gpteacher_huggingface(dataset)
        elif 'Synthetic-Persona-Chat' in source_name:
            return self._process_persona_chat_huggingface(dataset)
        elif 'writingprompts' in source_name.lower():
            return self._process_writing_prompts_huggingface(dataset)
        elif 'pile' in source_name.lower():
            return self._process_pile_huggingface(dataset)
        elif 'code_search_net' in source_name.lower():
            return self._process_code_search_net_huggingface(dataset)
        
        # Generic processing if no special handling
        if not text_fields:
            # If no text fields found, try to use the first string field
            for key, value in sample_item.items():
                if isinstance(value, str) and len(value) > 10:  # Require some minimum length
                    text_fields.append(key)
                    break
        
        if not text_fields:
            raise ValueError(f"Could not identify text fields in dataset {source_name}")
        
        print(f"Using text fields: {text_fields}")
        
        # Combine text from identified fields
        combined_texts = []
        for item in tqdm(dataset, desc="Processing samples"):
            item_texts = []
            for field in text_fields:
                if field in item and item[field]:
                    item_texts.append(str(item[field]))
            
            if item_texts:
                combined_texts.append("\n".join(item_texts))
        
        return "\n\n".join(combined_texts)

    def _process_openassistant_huggingface(self, dataset):
        """Process OpenAssistant dataset from HuggingFace"""
        conversations = []
        
        # Group by message_tree_id to reconstruct conversations
        conversation_map = {}
        
        for item in tqdm(dataset, desc="Processing OpenAssistant"):
            message_id = item.get('message_id')
            parent_id = item.get('parent_id')
            text = item.get('text', '')
            role = item.get('role', '')
            message_tree_id = item.get('message_tree_id')
            
            if not message_tree_id or not text:
                continue
            
            if message_tree_id not in conversation_map:
                conversation_map[message_tree_id] = []
            
            conversation_map[message_tree_id].append({
                'id': message_id,
                'parent_id': parent_id,
                'text': text,
                'role': role
            })
        
        # Convert to formatted conversations
        for tree_id, messages in conversation_map.items():
            # Create a map for quick parent lookup
            id_to_message = {msg['id']: msg for msg in messages if msg['id']}
            
            # Find root messages (no parent)
            roots = [msg for msg in messages if not msg['parent_id']]
            
            if not roots:
                continue
            
            # Process each conversation tree
            for root in roots:
                conversation = []
                conversation.append(f"USER: {root['text']}")
                
                # Find direct children (responses)
                children = [msg for msg in messages if msg.get('parent_id') == root['id']]
                
                for child in children:
                    if child['role'] == 'assistant':
                        conversation.append(f"ASSISTANT: {child['text']}")
                
                conversations.append("\n".join(conversation))
        
        return "\n\n".join(conversations)

    def _process_gpteacher_huggingface(self, dataset):
        """Process GPTeacher dataset from HuggingFace"""
        conversations = []
        
        for item in tqdm(dataset, desc="Processing GPTeacher"):
            if 'instruction' in item and 'response' in item:
                conversation = []
                conversation.append(f"USER: {item['instruction']}")
                conversation.append(f"ASSISTANT: {item['response']}")
                conversations.append("\n".join(conversation))
        
        return "\n\n".join(conversations)

    def _process_persona_chat_huggingface(self, dataset):
        """Process Persona Chat dataset from HuggingFace"""
        conversations = []
        
        for item in tqdm(dataset, desc="Processing Persona Chat"):
            if 'personas' in item and 'dialogue' in item:
                # Extract persona information
                persona_text = "\n".join([f"PERSONA: {p}" for p in item['personas']])
                
                # Extract dialogue
                dialogue_turns = []
                
                if isinstance(item['dialogue'], list):
                    for i, turn in enumerate(item['dialogue']):
                        speaker = "USER" if i % 2 == 0 else "ASSISTANT"
                        dialogue_turns.append(f"{speaker}: {turn}")
                
                dialogue_text = "\n".join(dialogue_turns)
                
                # Combine persona and dialogue
                full_text = f"{persona_text}\n\n{dialogue_text}"
                conversations.append(full_text)
        
        return "\n\n".join(conversations)

    def _process_writing_prompts_huggingface(self, dataset):
        """Process Writing Prompts dataset from HuggingFace"""
        prompt_story_pairs = []
        
        for item in tqdm(dataset, desc="Processing Writing Prompts"):
            if 'prompt' in item and 'story' in item:
                prompt = item['prompt']
                story = item['story']
                
                formatted_text = f"<PROMPT>\n{prompt}\n<STORY>\n{story}\n<END>"
                prompt_story_pairs.append(formatted_text)
        
        return "\n\n".join(prompt_story_pairs)

    def _process_pile_huggingface(self, dataset):
        """Process Pile dataset from HuggingFace"""
        texts = []
        
        for item in tqdm(dataset, desc="Processing Pile"):
            if 'text' in item:
                texts.append(item['text'])
        
        return "\n\n".join(texts)

    def _process_code_search_net_huggingface(self, dataset):
        """Process Code Search Net dataset from HuggingFace"""
        code_texts = []
        
        for item in tqdm(dataset, desc="Processing Code Search Net"):
            if 'code' in item and 'docstring' in item:
                code = item['code']
                docstring = item['docstring']
                
                formatted_text = f"DOCSTRING:\n{docstring}\n\nCODE:\n{code}"
                code_texts.append(formatted_text)
        
        return "\n\n".join(code_texts)