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

        # Check if source is a HuggingFace dataset (contains a slash)
        if isinstance(source, str) and '/' in source:
            # This is a HuggingFace dataset identifier, handle differently
            try:
                print(f"Loading HuggingFace dataset: {source}")
                return self._load_huggingface_dataset(source)
            except Exception as e:
                print(f"Error loading HuggingFace dataset {source}: {str(e)}")
                # Fall back to treating as a regular path
                pass

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

    def _load_huggingface_dataset(self, dataset_name: str, split="train", max_samples=None):
        """
        Load data directly from a HuggingFace dataset

        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to load
            max_samples: Maximum number of samples to load

        Returns:
            Processed text from the dataset
        """
        from datasets import load_dataset

        # Load the dataset
        dataset = load_dataset(dataset_name, split=split)

        # Limit samples if specified
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        # Process the dataset based on common field patterns
        return self._process_huggingface_dataset(dataset, dataset_name)

    def _process_huggingface_dataset(self, dataset, dataset_name):
        """
        Process HuggingFace dataset based on its structure

        Args:
            dataset: HuggingFace dataset
            dataset_name: Name of the dataset for format detection

        Returns:
            Processed text from the dataset
        """
        # Print dataset structure to help with debugging
        if len(dataset) > 0:
            first_example = dataset[0]
            print(f"Dataset structure for {dataset_name}: {list(first_example.keys())}")

        # Process dataset in a memory-efficient way
        combined_texts = []
        batch_size = 1000  # Process 1000 examples at a time

        # Use the right field based on dataset structure
        if "OpenAssistant" in dataset_name:
            # Special handling for OpenAssistant
            desc = f"Processing OpenAssistant"
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
                batch = dataset[i:min(i+batch_size, len(dataset))]
                batch_texts = []

                for item in batch:
                    # Format as dialogue pairs
                    if item['role'] == 'assistant' and 'text' in item:
                        batch_texts.append(f"User: [Previous message]\nAssistant: {item['text']}")
                    elif item['role'] == 'prompter' and 'text' in item:
                        batch_texts.append(f"User: {item['text']}\nAssistant:")

                combined_texts.extend(batch_texts)

                # Free memory
                del batch
                del batch_texts

        elif "GPTeacher" in dataset_name:
            # GPTeacher format
            desc = f"Processing GPTeacher"
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
                batch = dataset[i:min(i+batch_size, len(dataset))]
                batch_texts = []

                for item in batch:
                    if 'instruction' in item and 'response' in item:
                        batch_texts.append(f"User: {item['instruction']}\nAssistant: {item['response']}")

                combined_texts.extend(batch_texts)

                # Free memory
                del batch
                del batch_texts

        elif "Persona-Chat" in dataset_name:
            # Persona Chat format
            desc = f"Processing Persona Chat"
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
                batch = dataset[i:min(i+batch_size, len(dataset))]
                batch_texts = []

                for item in batch:
                    if 'personas' in item and 'utterances' in item:
                        personas = "\n".join(item['personas'])
                        for utterance in item['utterances']:
                            if isinstance(utterance, list) and len(utterance) >= 2:
                                batch_texts.append(f"Persona: {personas}\nUser: {utterance[0]}\nAssistant: {utterance[1]}")

                combined_texts.extend(batch_texts)

                # Free memory
                del batch
                del batch_texts

        elif "writingprompts" in dataset_name.lower():
            # Writing prompts format
            desc = f"Processing Writing Prompts"
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
                batch = dataset[i:min(i+batch_size, len(dataset))]
                batch_texts = []

                for item in batch:
                    if 'prompt' in item and 'story' in item:
                        batch_texts.append(f"Prompt: {item['prompt']}\nStory: {item['story']}")

                combined_texts.extend(batch_texts)

                # Free memory
                del batch
                del batch_texts

        else:
            # Generic format - try to extract text based on common field names
            desc = f"Processing dataset: {dataset_name}"
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
                batch = dataset[i:min(i+batch_size, len(dataset))]
                batch_texts = []

                for item in batch:
                    # Try different field combinations
                    if 'text' in item:
                        batch_texts.append(item['text'])
                    elif 'content' in item:
                        batch_texts.append(item['content'])
                    elif 'input' in item and 'output' in item:
                        batch_texts.append(f"Input: {item['input']}\nOutput: {item['output']}")
                    elif 'question' in item and 'answer' in item:
                        batch_texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")
                    elif 'prompt' in item and 'completion' in item:
                        batch_texts.append(f"Prompt: {item['prompt']}\nCompletion: {item['completion']}")

                combined_texts.extend(batch_texts)

                # Free memory
                del batch
                del batch_texts

        # Join all texts with newlines between examples
        return "\n\n".join(combined_texts)

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

        # Safety check for empty text
        if not text:
            print("Warning: Empty text provided to create_sequences")
            return []

        # Get character mapping and vocabulary size
        char_to_idx = self.text_generator.char_to_index
        unknown_token = self.text_generator.unknown_token

        # Verify unknown token is in the mapping
        if unknown_token not in char_to_idx:
            print(f"Warning: Unknown token '{unknown_token}' not in char_to_index, adding it")
            char_to_idx[unknown_token] = len(char_to_idx)

        # Determine the actual vocabulary size from the char_to_index mapping
        n_chars = max(char_to_idx.values()) + 1

        # Create character-level sequences
        sequences = []

        # Skip sequences if text is too short
        if len(text) <= sequence_length + 1:
            print(f"Warning: Text length ({len(text)}) is too short for sequence length ({sequence_length})")
            return []

        try:
            for i in range(0, len(text) - sequence_length - 1, sequence_length // 2):  # Use stride of half the sequence length
                # Input is sequence_length characters
                input_seq = text[i:i+sequence_length]
                # Target is the next character
                target_char = text[i+sequence_length]

                # Two options for input representation:
                # 1. One-hot encoding (matrix approach)
                # 2. Index-based (embedding approach)

                # Approach 1: One-hot encoding
                try:
                    input_tensor = torch.zeros(sequence_length, n_chars)
                    for t, char in enumerate(input_seq):
                        # Get character index or use unknown token index if not found
                        idx = char_to_idx.get(char, char_to_idx.get(unknown_token, 0))

                        # Double-check index is within bounds
                        if idx >= n_chars:
                            print(f"Warning: Index {idx} for character '{char}' exceeds vocabulary size {n_chars}, using unknown token")
                            idx = char_to_idx.get(unknown_token, 0)

                        input_tensor[t, idx] = 1.0

                    # Get target index
                    target_idx = char_to_idx.get(target_char, char_to_idx.get(unknown_token, 0))

                    # Ensure target index is within bounds
                    if target_idx >= n_chars:
                        print(f"Warning: Target index {target_idx} for character '{target_char}' exceeds vocabulary size {n_chars}, using unknown token")
                        target_idx = char_to_idx.get(unknown_token, 0)

                    target_tensor = torch.tensor([target_idx])
                    sequences.append((input_tensor, target_tensor))
                except Exception as e:
                    print(f"Error creating sequence at position {i}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error in create_sequences: {str(e)}")
            # Fall back to a simpler approach with fewer sequences
            print("Falling back to simplified sequence creation")

            try:
                # Create a very small number of sequences as a fallback
                for i in range(min(10, len(text) - sequence_length - 1)):
                    input_seq = text[i:i+sequence_length]
                    target_char = text[i+sequence_length]

                    # Use index-based approach for simplicity
                    input_indices = [char_to_idx.get(char, char_to_idx.get(unknown_token, 0)) for char in input_seq]
                    target_idx = char_to_idx.get(target_char, char_to_idx.get(unknown_token, 0))

                    # Create tensors
                    input_tensor = torch.tensor(input_indices).view(1, -1)  # Add batch dimension
                    target_tensor = torch.tensor([target_idx])

                    sequences.append((input_tensor, target_tensor))
            except Exception as inner_e:
                print(f"Error in fallback sequence creation: {str(inner_e)}")
                # Return empty list if all else fails
                return []

        return sequences

    def create_batches(self, sequences: List[Tuple[torch.Tensor, torch.Tensor]],
                      batch_size: int = 64, shuffle: bool = True,
                      dataset_name: str = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create batches from sequences

        Args:
            sequences: List of (input, target) tensor pairs
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            dataset_name: Optional name of the dataset for special handling

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

            # Special handling for OpenAssistant dataset to ensure correct dtype
            if dataset_name == "openassistant" or (
                len(batch_sequences) > 0 and
                hasattr(batch_sequences[0][0], 'dtype') and
                batch_sequences[0][0].dtype != torch.long
            ):
                # Convert to long dtype explicitly
                print(f"Converting batch tensors to torch.long dtype")

                # Stack tensors into batches with explicit dtype
                try:
                    # First convert individual tensors if needed
                    converted_inputs = []
                    for tensor in batch_inputs:
                        if tensor.dtype != torch.long:
                            converted_inputs.append(tensor.to(dtype=torch.long))
                        else:
                            converted_inputs.append(tensor)

                    converted_targets = []
                    for tensor in batch_targets:
                        if tensor.dtype != torch.long:
                            converted_targets.append(tensor.to(dtype=torch.long))
                        else:
                            converted_targets.append(tensor)

                    # Then stack them
                    input_batch = torch.stack(converted_inputs)
                    target_batch = torch.stack(converted_targets)

                    # Verify dtype
                    if input_batch.dtype != torch.long:
                        print(f"Warning: Input batch still has incorrect dtype: {input_batch.dtype}. Forcing to torch.long.")
                        input_batch = input_batch.long()

                    if target_batch.dtype != torch.long:
                        print(f"Warning: Target batch still has incorrect dtype: {target_batch.dtype}. Forcing to torch.long.")
                        target_batch = target_batch.long()
                except Exception as e:
                    print(f"Error converting batch tensors: {e}")
                    # Fall back to standard stacking
                    input_batch = torch.stack(batch_inputs).long()
                    target_batch = torch.stack(batch_targets).long()
            else:
                # Standard stacking for other datasets
                try:
                    input_batch = torch.stack(batch_inputs)
                    target_batch = torch.stack(batch_targets)

                    # Safety check for dtype
                    if input_batch.dtype != torch.long:
                        print(f"Warning: Input batch has incorrect dtype: {input_batch.dtype}. Converting to torch.long.")
                        input_batch = input_batch.long()

                    if target_batch.dtype != torch.long:
                        print(f"Warning: Target batch has incorrect dtype: {target_batch.dtype}. Converting to torch.long.")
                        target_batch = target_batch.long()
                except Exception as e:
                    print(f"Error stacking tensors: {e}")
                    continue

            batches.append((input_batch, target_batch))

        return batches

    def prepare_text_batches(self, raw_text, sequence_length, batch_size, dataset_name=None):
        """
        Prepare text into batches for training

        Args:
            raw_text: Raw text data
            sequence_length: Length of sequences
            batch_size: Batch size
            dataset_name: Optional name of the dataset for special handling

        Returns:
            Batched dataset ready for training
        """
        cleaned_text = self.clean_text(raw_text)
        sequences = self.create_sequences(cleaned_text, sequence_length)
        return self.create_batches(sequences, batch_size, dataset_name=dataset_name)

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
            raw_text = self.load_writing_prompts(split=split, max_samples=max_samples, cache_dir=cache_dir)
        elif source == 'openassistant':
            raw_text = self.load_openassistant_dataset(split=split, max_samples=max_samples, cache_dir=cache_dir)
        else:
            # Treat as path to local file or directory
            raw_text = self.load_data(source)

        # Create batches with dataset name for special handling
        return self.prepare_text_batches(raw_text, sequence_length, batch_size, dataset_name=source.lower())

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

        # Prepare batches (code datasets don't need special handling)
        return self.prepare_text_batches(raw_text, sequence_length, batch_size, dataset_name="code")

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
        # Use a generic dataset name for local files
        return self.prepare_text_batches(raw_text, sequence_length, batch_size, dataset_name="local_text")

    def load_preprocessed_data(self, dataset_name: str, custom_path: str = None) -> Dict[str, Any]:
        """
        Load preprocessed dataset from disk

        Args:
            dataset_name: Name of the dataset
            custom_path: Optional custom path to the preprocessed file

        Returns:
            Dictionary containing preprocessed data
        """
        try:
            # Determine path
            if custom_path is not None:
                path = custom_path
            else:
                # Check in standard locations
                cache_dir = os.environ.get('JARVIS_CACHE_DIR', 'datasets/cache')
                path = os.path.join(cache_dir, f"{dataset_name}_preprocessed.pt")

                # Check if file exists
                if not os.path.exists(path):
                    # Also check for _preprocessed.pt_preprocessed.pt pattern (double extension)
                    alternative_path = path + "_preprocessed.pt"
                    if os.path.exists(alternative_path):
                        path = alternative_path
                    else:
                        raise FileNotFoundError(f"Preprocessed file not found: {path}")

            # Load the data
            data = torch.load(path)

            # Extra checks
            if not isinstance(data, dict):
                raise ValueError(f"Preprocessed data is not a dictionary: {path}")

            if 'batches' not in data or not data['batches']:
                print(f"Warning: No batches found in preprocessed data: {path}")

            return data
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return {'batches': []}

    def decode_tokens(self, tokens):
        """
        Convert token IDs back to text

        Args:
            tokens: List of token IDs (may be nested)

        Returns:
            Decoded text string
        """
        # Handle nested lists by flattening
        flat_tokens = []
        def flatten(items):
            for item in items:
                if isinstance(item, list):
                    flatten(item)
                else:
                    flat_tokens.append(item)

        # Flatten tokens if it's a nested list
        if isinstance(tokens, list):
            if tokens and isinstance(tokens[0], list):
                flatten(tokens)
            else:
                flat_tokens = tokens
        else:
            flat_tokens = [tokens]  # Handle single token case

        # Now decode the flattened tokens
        if hasattr(self, 'text_generator') and hasattr(self.text_generator, 'index_to_char'):
            # Use the text generator's mapping
            return ''.join(self.text_generator.index_to_char.get(token, "<UNK>") for token in flat_tokens)

        # If we have a tokenizer with a decode method (HuggingFace style)
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'decode'):
            try:
                return self.tokenizer.decode(flat_tokens)
            except Exception as e:
                return f"<Error decoding tokens: {str(e)}>"

        # Fallback: try to interpret as character codes
        try:
            return ''.join(chr(token) if 0 <= token <= 0x10FFFF else "<UNK>" for token in flat_tokens)
        except Exception as e:
            return f"<Unable to decode tokens: {flat_tokens[:10]}... Error: {str(e)}>"

    def save_tokenized_data(self, data: Dict[str, Any], output_dir: str, dataset_name: str):
        """
        Save tokenized dataset and a sample of the text

        Args:
            data: Dictionary containing dataset
            output_dir: Directory to save the data
            dataset_name: Name of the dataset
        """
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
            # Use the provided path if specified
            data = self.load_preprocessed_data(dataset_name, custom_path=preprocessed_path)

            # Check if data contains batches
            if 'batches' not in data or not data['batches']:
                print(f"Warning: No batches found in preprocessed data for {dataset_name}")
                return []

            # If no reshaping is needed, return the batches directly
            if batch_size is None:
                # Check if we need to convert tensor dtypes
                if dataset_name.lower() == "openassistant":
                    print(f"Checking and converting tensor dtypes for {dataset_name} dataset")
                    converted_batches = []
                    for input_batch, target_batch in data['batches']:
                        # Convert to long dtype if needed
                        if hasattr(input_batch, 'dtype') and input_batch.dtype != torch.long:
                            input_batch = input_batch.to(dtype=torch.long)
                        if hasattr(target_batch, 'dtype') and target_batch.dtype != torch.long:
                            target_batch = target_batch.to(dtype=torch.long)
                        converted_batches.append((input_batch, target_batch))
                    return converted_batches
                else:
                    return data['batches']

            # Otherwise, reshape the batches
            print(f"Reshaping batches to size {batch_size}")

            # Create a flat list of samples
            flat_samples = []
            for input_batch, target_batch in data['batches']:
                # Convert to long dtype if needed
                if hasattr(input_batch, 'dtype') and input_batch.dtype != torch.long:
                    input_batch = input_batch.to(dtype=torch.long)
                if hasattr(target_batch, 'dtype') and target_batch.dtype != torch.long:
                    target_batch = target_batch.to(dtype=torch.long)

                flat_samples.extend(
                    (input_batch[i], target_batch[i])
                    for i in range(input_batch.shape[0])
                )

            # Create new batches with dataset name for special handling
            return self.create_batches(flat_samples, batch_size, dataset_name=dataset_name.lower())

        except Exception as e:
            print(f"Error preparing from preprocessed data: {e}")
            import traceback
            traceback.print_exc()
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
            print("Loading OpenAssistant dataset with explicit dtype control...")
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

            # Add a note about the special handling for OpenAssistant
            print("Note: OpenAssistant dataset will be processed with special handling to ensure correct tensor dtype (torch.long)")

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

    def load_dataset_with_custom_id(self, dataset_id, split='train', max_samples=None, cache_dir=None,
                                   offset=0, extract_method=None):
        """
        Load and preprocess data from a custom dataset ID with flexible extraction

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "teknium/GPTeacher-General-Instruct")
            split: Dataset split ('train', 'test', or 'validation')
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Optional directory to cache the downloaded dataset
            offset: Offset to start loading from
            extract_method: Optional custom extraction method for the dataset

        Returns:
            Dictionary with batches and metadata
        """
        try:
            print(f"Loading dataset with custom ID: {dataset_id}")
            dataset = load_dataset(dataset_id, split=split, cache_dir=cache_dir)

            # Apply offset and limit
            if offset > 0:
                if offset >= len(dataset):
                    print(f"Offset {offset} exceeds dataset size {len(dataset)}")
                    return {'batches': [], 'metadata': {'sample_count': 0, 'source': dataset_id}}

                dataset = dataset.select(range(offset, len(dataset)))

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            # Use custom extraction method if provided
            if extract_method and callable(extract_method):
                # Extract texts using the provided method
                texts = extract_method(dataset)

                # Create batches from the extracted texts
                return {
                    'batches': texts,
                    'metadata': {
                        'sample_count': len(texts),
                        'source': dataset_id,
                        'offset': offset,
                        'max_samples': max_samples
                    }
                }
            else:
                # Default processing based on dataset ID
                if "GPTeacher" in dataset_id:
                    # Process as GPTeacher format
                    return self._process_gpteacher_format(dataset, dataset_id)
                elif "OpenAssistant" in dataset_id or "oasst" in dataset_id:
                    # Process as OpenAssistant format
                    return self._process_openassistant_format(dataset, dataset_id)
                else:
                    # Generic processing
                    return self._process_generic_dataset(dataset, dataset_id)

        except Exception as e:
            print(f"Error loading dataset with custom ID {dataset_id}: {e}")
            import traceback
            traceback.print_exc()

            # Return empty result
            return {'batches': [], 'metadata': {'sample_count': 0, 'source': dataset_id}}

    def _process_gpteacher_format(self, dataset, dataset_id):
        """Process dataset in GPTeacher format"""
        batches = []

        for item in tqdm(dataset, desc=f"Processing {dataset_id}"):
            # Extract instruction and response
            instruction = item.get('instruction', '')
            response = item.get('response', '')

            if instruction and response:
                batches.append(f"USER: {instruction}\nASSISTANT: {response}")

        return {
            'batches': batches,
            'metadata': {
                'sample_count': len(batches),
                'source': dataset_id
            }
        }

    def _process_openassistant_format(self, dataset, dataset_id):
        """Process dataset in OpenAssistant format"""
        text_batches = []

        # First, collect all text samples
        for item in tqdm(dataset, desc=f"Processing {dataset_id}"):
            if 'text' in item and 'role' in item:
                if item['role'] == 'assistant':
                    text_batches.append(f"USER: [Previous question]\nASSISTANT: {item['text']}")
                elif item['role'] == 'prompter':
                    text_batches.append(f"USER: {item['text']}\nASSISTANT:")

        # Now create sequences and batches with explicit dtype control
        print(f"Creating sequences for OpenAssistant dataset...")
        sequences = []

        # Use a simplified approach for OpenAssistant to ensure correct dtype
        for text in text_batches:
            # Create a simple sequence with explicit Long dtype
            input_tensor = torch.tensor([ord(c) % 256 for c in text[:100]], dtype=torch.long)
            target_tensor = torch.tensor([ord(c) % 256 for c in text[1:101]], dtype=torch.long)

            # Pad if needed
            if len(input_tensor) < 100:
                padding = torch.zeros(100 - len(input_tensor), dtype=torch.long)
                input_tensor = torch.cat([input_tensor, padding])

            if len(target_tensor) < 100:
                padding = torch.zeros(100 - len(target_tensor), dtype=torch.long)
                target_tensor = torch.cat([target_tensor, padding])

            sequences.append((input_tensor, target_tensor))

        # Create batches with explicit dtype control
        print(f"Creating batches for OpenAssistant dataset...")
        batches = []
        batch_size = 16  # Default batch size

        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]

            if not batch_sequences:
                continue

            # Unzip the batch sequences
            batch_inputs, batch_targets = zip(*batch_sequences)

            # Stack tensors into batches with explicit dtype
            try:
                input_batch = torch.stack(batch_inputs).to(dtype=torch.long)
                target_batch = torch.stack(batch_targets).to(dtype=torch.long)

                # Verify dtype
                assert input_batch.dtype == torch.long, f"Input batch dtype is {input_batch.dtype}, not torch.long"
                assert target_batch.dtype == torch.long, f"Target batch dtype is {target_batch.dtype}, not torch.long"

                batches.append((input_batch, target_batch))
            except Exception as e:
                print(f"Error creating batch: {e}")
                continue

        print(f"Created {len(batches)} batches for OpenAssistant dataset with dtype torch.long")

        return {
            'batches': batches,
            'metadata': {
                'sample_count': len(text_batches),
                'batch_count': len(batches),
                'source': dataset_id,
                'dtype': 'torch.long'  # Explicitly note the dtype
            }
        }

    def _process_generic_dataset(self, dataset, dataset_id):
        """Process generic dataset format"""
        batches = []

        for item in tqdm(dataset, desc=f"Processing {dataset_id}"):
            # Try different field combinations
            if 'instruction' in item and 'response' in item:
                batches.append(f"USER: {item['instruction']}\nASSISTANT: {item['response']}")
            elif 'prompt' in item and 'completion' in item:
                batches.append(f"USER: {item['prompt']}\nASSISTANT: {item['completion']}")
            elif 'input' in item and 'output' in item:
                batches.append(f"USER: {item['input']}\nASSISTANT: {item['output']}")
            elif 'question' in item and 'answer' in item:
                batches.append(f"USER: {item['question']}\nASSISTANT: {item['answer']}")
            elif 'text' in item:
                batches.append(item['text'])

        return {
            'batches': batches,
            'metadata': {
                'sample_count': len(batches),
                'source': dataset_id
            }
        }

    def prepare_dataset(self, source='persona_chat', split='train',
                       sequence_length=100, batch_size=64, max_samples=None,
                       cache_dir=None, subset=None, output_dir=None):
        """
        Prepare a dataset for training

        Args:
            source: Dataset name/path, can be a standard dataset or a HuggingFace dataset name
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
        # Pass the dataset name to create_batches for special handling
        dataset_name_for_batches = source.lower() if not is_huggingface_dataset else None
        batches = self.create_batches(sequences, batch_size=batch_size, dataset_name=dataset_name_for_batches)

        dataset = {
            'batches': batches,
            'metadata': {
                'source': source,
                'split': split,
                'sequence_length': sequence_length,
                'batch_size': batch_size,
                'sample_count': len(sequences),
                'batch_count': len(batches),
                'dtype': 'torch.long'  # Explicitly note the dtype
            }
        }

        # Save preprocessed data for future use
        print(f"Saving preprocessed dataset to {output_path}...")
        self.save_tokenized_data(dataset, os.path.dirname(output_path), os.path.basename(output_path))

        return dataset