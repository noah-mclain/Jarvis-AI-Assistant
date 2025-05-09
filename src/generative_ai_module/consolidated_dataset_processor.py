"""
Consolidated Dataset Processor

This module provides a unified interface for handling all supported datasets:
- writing_prompts
- persona_chat
- pile
- openassistant
- gpteacher

Features:
- Consistent API across all datasets
- Automatic dataset loading and preprocessing
- Smart dataset selection based on prompt content
- Common batch format for training
- Validation capabilities for dataset quality
- Context handling for conversations
- Memory-efficient processing
- Improved tokenization
"""

# Standard library imports
import os
import json
import logging
import datetime
import random
import glob
import string
import sys
from collections import deque, Counter
from typing import Dict, List, Any, Optional, Union, Tuple

# Third-party imports
import torch
import numpy as np
from tqdm import tqdm

# Try to import optional dependencies with graceful fallbacks
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("HuggingFace datasets library not found. Some features will be limited.")

# Local imports
from .utils import (
    get_storage_path,
    sync_to_gdrive,
    sync_from_gdrive,
    ensure_directory_exists,
    is_paperspace_environment,
    is_zipfile,
    process_zip
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConsolidatedDatasetProcessor")

class ImprovedTokenizer:
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

class ConversationContext:
    """
    Class for managing conversation context and history.

    This class provides methods for adding, formatting, saving, and loading
    conversation history with token management.
    """

    def __init__(self, max_history: int = 5, max_tokens: int = 1000):
        """
        Initialize the conversation context.

        Args:
            max_history: Maximum number of exchanges to keep in history
            max_tokens: Maximum number of tokens to include in formatted history
        """
        self.history = deque(maxlen=max_history)
        self.max_tokens = max_tokens
        self.metadata = {}

    def add_exchange(self, user_input: str, assistant_response: str) -> None:
        """
        Add a conversation exchange to history.

        Args:
            user_input: The user's input message
            assistant_response: The assistant's response message
        """
        self.history.append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.datetime.now().isoformat()
        })

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation - in production, use actual tokenizer
        return len(text.split())

    def _format_exchange(self, exchange: Dict[str, str]) -> str:
        """
        Format a single exchange for output.

        Args:
            exchange: Dictionary containing user and assistant messages

        Returns:
            Formatted exchange text
        """
        return f"USER: {exchange['user']}\nASSISTANT: {exchange['assistant']}\n\n"

    def get_formatted_history(self, include_current: bool = False, current_input: str = None) -> str:
        """
        Format conversation history for use in generation with token management.

        Args:
            include_current: Whether to include the current input
            current_input: The current user input to include

        Returns:
            Formatted conversation history
        """
        total_tokens = 0
        formatted_parts = []

        # Start from the most recent exchanges to prioritize recent context
        for exchange in reversed(list(self.history)):
            exchange_text = self._format_exchange(exchange)
            exchange_tokens = self._estimate_tokens(exchange_text)

            if total_tokens + exchange_tokens <= self.max_tokens:
                formatted_parts.insert(0, exchange_text)  # Insert at beginning to maintain order
                total_tokens += exchange_tokens
            else:
                break  # Too many tokens, stop adding more exchanges

        formatted = "".join(formatted_parts)

        # Handle current input if requested
        if include_current and current_input:
            current_text = f"USER: {current_input}\nASSISTANT: "
            current_tokens = self._estimate_tokens(current_text)

            # If adding current input would exceed limit, remove oldest exchanges
            while formatted_parts and (total_tokens + current_tokens > self.max_tokens):
                oldest_exchange = formatted_parts.pop(0)
                total_tokens -= self._estimate_tokens(oldest_exchange)

            # Rebuild formatted text and append current input
            formatted = "".join(formatted_parts) + current_text

        return formatted

    def save_to_file(self, filepath: str) -> None:
        """
        Save conversation context to a file.

        Args:
            filepath: Path to save the conversation context
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'history': list(self.history),
                'metadata': self.metadata,
                'max_tokens': self.max_tokens,
                'max_history': self.history.maxlen
            }, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationContext':
        """
        Load conversation context from a file.

        Args:
            filepath: Path to load the conversation context from

        Returns:
            Loaded ConversationContext object
        """
        if not os.path.exists(filepath):
            return cls()

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Get max_history and max_tokens if present, otherwise use defaults
            max_history = data.get('max_history', 5)
            max_tokens = data.get('max_tokens', 1000)

            context = cls(max_history=max_history, max_tokens=max_tokens)

            for exchange in data.get('history', []):
                # Ensure we have the required fields
                if 'user' in exchange and 'assistant' in exchange:
                    context.history.append(exchange)

            context.metadata = data.get('metadata', {})
            return context
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading conversation context: {e}")
            return cls()  # Return a new instance if loading fails

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the conversation context.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def clear(self) -> None:
        """Clear the conversation history and metadata."""
        self.history.clear()
        self.metadata = {}

    def truncate_history(self, max_tokens: int = None) -> None:
        """
        Truncate history to fit within token limit.

        Args:
            max_tokens: Maximum number of tokens (defaults to self.max_tokens)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        total_tokens = 0
        truncated_history = deque(maxlen=self.history.maxlen)

        # Start from most recent exchanges
        for exchange in reversed(list(self.history)):
            exchange_text = self._format_exchange(exchange)
            exchange_tokens = self._estimate_tokens(exchange_text)

            if total_tokens + exchange_tokens <= max_tokens:
                truncated_history.appendleft(exchange)
                total_tokens += exchange_tokens
            else:
                break

        self.history = truncated_history


class ConsolidatedDatasetProcessor:
    """
    A unified processor for handling all supported datasets.

    This class combines functionality from DatasetProcessor, ImprovedPreprocessor,
    and UnifiedDatasetHandler to provide a comprehensive solution for dataset
    management, preprocessing, and batching.
    """

    # Supported datasets with descriptions
    DATASET_INFO = {
        "writing_prompts": "Creative writing prompts and stories from Reddit's r/WritingPrompts",
        "persona_chat": "Dialogue dataset with persona-conditioned conversations",
        "pile": "Large-scale, diverse dataset of text from the internet",
        "openassistant": "Assistant-style conversations with helpful responses",
        "gpteacher": "Instruction-following dataset with educational content"
    }

    # Default configuration values
    DEFAULT_CONFIG = {
        "max_length": 1024,
        "batch_size": 32,
        "max_history": 5,
        "max_tokens": 1000
    }

    def __init__(self, text_generator=None, dataset_name=None, dataset_path=None,
                 output_dir=None, cache_dir=None):
        """
        Initialize the dataset processor.

        Args:
            text_generator: Optional text generator for character mappings
            dataset_name: Name of the dataset to load from HuggingFace Hub
            dataset_path: Path to a local dataset
            output_dir: Directory to save processed datasets
            cache_dir: Directory to cache downloaded datasets
        """
        # Set up basic attributes
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.text_generator = text_generator
        self.sequence_length = 100  # Default sequence length
        self.logger = logging.getLogger("ConsolidatedDatasetProcessor")

        # Set up directories
        self._setup_directories(output_dir, cache_dir)

        # Initialize dataset and model attributes
        self.dataset = None
        self.tokenizer = ImprovedTokenizer()
        self.max_length = self.DEFAULT_CONFIG["max_length"]
        self.batch_size = self.DEFAULT_CONFIG["batch_size"]

        # Sync datasets if in Paperspace environment
        self._sync_datasets_if_needed()

        # Create conversation context management
        self.conversation_context = ConversationContext(
            max_history=self.DEFAULT_CONFIG["max_history"],
            max_tokens=self.DEFAULT_CONFIG["max_tokens"]
        )

        # Track metadata about available models and datasets
        self.metadata = {
            "available_datasets": list(self.DATASET_INFO.keys()),
            "available_models": []
        }

        self.logger.info("Consolidated Dataset Processor initialized")

    def _setup_directories(self, output_dir, cache_dir):
        """Set up the necessary directories for datasets."""
        self.output_dir = output_dir or ensure_directory_exists("datasets", "processed")
        self.cache_dir = cache_dir or ensure_directory_exists("datasets", "cache")
        self.data_dir = os.path.dirname(self.cache_dir)

    def _sync_datasets_if_needed(self):
        """Sync datasets from Google Drive if in Paperspace environment."""
        if (self.dataset_path is None and
            self.dataset_name is None and
            is_paperspace_environment()):
            try:
                sync_from_gdrive("datasets")
                self.logger.info("Synced latest datasets from Google Drive")
            except Exception as e:
                self.logger.warning(f"Failed to sync datasets from Google Drive: {str(e)}")

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
        if not DATASETS_AVAILABLE:
            self.logger.warning("HuggingFace datasets library not available")
            return "HuggingFace datasets library not available"

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
            self.logger.info(f"Dataset structure for {dataset_name}: {list(first_example.keys())}")

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
            self.logger.warning("Empty text provided to create_sequences")
            return []

        # Get character mapping and vocabulary size
        if hasattr(self, 'text_generator') and self.text_generator:
            char_to_idx = self.text_generator.char_to_index
            unknown_token = self.text_generator.unknown_token

            # Verify unknown token is in the mapping
            if unknown_token not in char_to_idx:
                self.logger.warning(f"Unknown token '{unknown_token}' not in char_to_index, adding it")
                char_to_idx[unknown_token] = len(char_to_idx)

            # Determine the actual vocabulary size from the char_to_index mapping
            n_chars = max(char_to_idx.values()) + 1
        else:
            # Use the tokenizer's mapping
            char_to_idx = self.tokenizer.char_to_idx
            unknown_token = '<UNK>'
            n_chars = self.tokenizer.vocab_size

        # Create character-level sequences
        sequences = []

        # Skip sequences if text is too short
        if len(text) <= sequence_length + 1:
            self.logger.warning(f"Text length ({len(text)}) is too short for sequence length ({sequence_length})")
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
                            self.logger.warning(f"Index {idx} for character '{char}' exceeds vocabulary size {n_chars}, using unknown token")
                            idx = char_to_idx.get(unknown_token, 0)

                        input_tensor[t, idx] = 1.0

                    # Get target index
                    target_idx = char_to_idx.get(target_char, char_to_idx.get(unknown_token, 0))

                    # Ensure target index is within bounds
                    if target_idx >= n_chars:
                        self.logger.warning(f"Target index {target_idx} for character '{target_char}' exceeds vocabulary size {n_chars}, using unknown token")
                        target_idx = char_to_idx.get(unknown_token, 0)

                    target_tensor = torch.tensor([target_idx])
                    sequences.append((input_tensor, target_tensor))
                except Exception as e:
                    self.logger.error(f"Error creating sequence at position {i}: {str(e)}")
                    continue
        except Exception as e:
            self.logger.error(f"Error in create_sequences: {str(e)}")
            # Fall back to a simpler approach with fewer sequences
            self.logger.info("Falling back to simplified sequence creation")

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
                self.logger.error(f"Error in fallback sequence creation: {str(inner_e)}")
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
                self.logger.info(f"Converting batch tensors to torch.long dtype")

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
                        self.logger.warning(f"Input batch still has incorrect dtype: {input_batch.dtype}. Forcing to torch.long.")
                        input_batch = input_batch.long()

                    if target_batch.dtype != torch.long:
                        self.logger.warning(f"Target batch still has incorrect dtype: {target_batch.dtype}. Forcing to torch.long.")
                        target_batch = target_batch.long()
                except Exception as e:
                    self.logger.error(f"Error converting batch tensors: {e}")
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
                        self.logger.warning(f"Input batch has incorrect dtype: {input_batch.dtype}. Converting to torch.long.")
                        input_batch = input_batch.long()

                    if target_batch.dtype != torch.long:
                        self.logger.warning(f"Target batch has incorrect dtype: {target_batch.dtype}. Converting to torch.long.")
                        target_batch = target_batch.long()
                except Exception as e:
                    self.logger.error(f"Error stacking tensors: {e}")
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
                self.logger.warning(f"No batches found in preprocessed data: {path}")

            return data
        except Exception as e:
            self.logger.error(f"Error loading preprocessed data: {e}")
            return {'batches': []}

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
        self.logger.info(f"Saved preprocessed data to {output_path}")

        # Save a sample of the text for inspection
        sample_path = os.path.join(output_dir, f"{dataset_name}_sample.txt")
        with open(sample_path, 'w') as f:
            if 'batches' in data and len(data['batches']) > 0:
                sample_batch = data['batches'][0]
                if isinstance(sample_batch, tuple) and len(sample_batch) > 0:
                    sample_text = self.decode_tokens(sample_batch[0][0].tolist())
                    f.write(sample_text)
        self.logger.info(f"Saved text sample to {sample_path}")

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

        # If we have a tokenizer with a decode method
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
                self.logger.warning(f"No batches found in preprocessed data for {dataset_name}")
                return []

            # If no reshaping is needed, return the batches directly
            if batch_size is None:
                # Check if we need to convert tensor dtypes
                if dataset_name.lower() == "openassistant":
                    self.logger.info(f"Checking and converting tensor dtypes for {dataset_name} dataset")
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
            self.logger.info(f"Reshaping batches to size {batch_size}")

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
            self.logger.error(f"Error preparing from preprocessed data: {e}")
            import traceback
            traceback.print_exc()
            return []

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
        if not DATASETS_AVAILABLE:
            self.logger.warning("HuggingFace datasets library not available")
            return self._generate_sample_persona_chat()

        # Load dataset
        try:
            dataset = self._load_persona_chat_dataset(split, cache_dir)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            return self._process_persona_chat_items(dataset)
        except Exception as e:
            self.logger.error(f"Error loading Persona Chat dataset: {e}")
            self.logger.info("Falling back to sample data")
            return self._generate_sample_persona_chat()

    def _load_persona_chat_dataset(self, split, cache_dir):
        """Helper to load the right persona chat dataset"""
        try:
            return load_dataset("bavard/personachat_truecased", split=split, cache_dir=cache_dir)
        except Exception:
            return load_dataset("persona_chat", split=split, cache_dir=cache_dir)

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
        if not DATASETS_AVAILABLE:
            self.logger.warning("HuggingFace datasets library not available")
            return self._generate_sample_writing_prompts()

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
            self.logger.error(f"Error loading Writing Prompts dataset: {e}")
            self.logger.info("Falling back to sample data")
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
        if not DATASETS_AVAILABLE:
            self.logger.warning("HuggingFace datasets library not available")
            return self._generate_sample_assistant_data()

        try:
            self.logger.info("Loading OpenAssistant dataset with explicit dtype control...")
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
            self.logger.info("Note: OpenAssistant dataset will be processed with special handling to ensure correct tensor dtype (torch.long)")

            return compiled_text

        except Exception as e:
            self.logger.error(f"Error loading OpenAssistant dataset: {e}")
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

    def analyze_token_distribution(self, data):
        """
        Analyze the distribution of tokens in a dataset

        Args:
            data: Dictionary containing dataset with batches

        Returns:
            Dictionary with token distribution analysis
        """
        if 'batches' not in data or not data['batches']:
            self.logger.warning("No batches found in data for token distribution analysis")
            return {'token_counts': {}, 'total_tokens': 0, 'unique_tokens': 0}

        # Initialize token counter
        token_counts = Counter()
        total_tokens = 0

        # Process each batch
        for input_batch, target_batch in data['batches']:
            # Count tokens in input batch
            if hasattr(input_batch, 'view'):
                # Flatten the batch and count tokens
                flat_tokens = input_batch.view(-1).tolist()
                token_counts.update(flat_tokens)
                total_tokens += len(flat_tokens)

            # Count tokens in target batch
            if hasattr(target_batch, 'view'):
                # Flatten the batch and count tokens
                flat_tokens = target_batch.view(-1).tolist()
                token_counts.update(flat_tokens)
                total_tokens += len(flat_tokens)

        # Calculate statistics
        unique_tokens = len(token_counts)
        most_common = token_counts.most_common(10)

        # Convert token IDs to characters if possible
        if hasattr(self, 'index_to_char'):
            most_common_chars = [(self.index_to_char.get(token, f"<UNK-{token}>"), count)
                               for token, count in most_common]
        else:
            most_common_chars = most_common

        # Calculate frequency distribution
        frequency_distribution = {
            'most_common': most_common_chars,
            'least_common': [(self.index_to_char.get(token, f"<UNK-{token}>"), count)
                           for token, count in token_counts.most_common()[-10:]] if hasattr(self, 'index_to_char') else token_counts.most_common()[-10:],
            'distribution': {
                'top_10%': sum(count for _, count in token_counts.most_common(unique_tokens // 10)) / total_tokens if unique_tokens > 10 else 1.0,
                'top_50%': sum(count for _, count in token_counts.most_common(unique_tokens // 2)) / total_tokens if unique_tokens > 2 else 1.0
            }
        }

        return {
            'token_counts': dict(token_counts),
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'most_common': most_common_chars,
            'frequency_distribution': frequency_distribution
        }

    def process_dataset(self, dataset_name, max_samples=None, sequence_length=None, batch_size=None):
        """
        Process a dataset with dataset-specific settings

        Args:
            dataset_name: Name of the dataset to process
            max_samples: Maximum number of samples to process
            sequence_length: Length of sequences (default: from config)
            batch_size: Batch size (default: from config)

        Returns:
            Dictionary containing processed dataset
        """
        # Use provided parameters or defaults from config
        sequence_length = sequence_length or self.config['sequence_length']
        batch_size = batch_size or self.config['batch_size']

        # Set dataset-specific parameters
        params = self._get_dataset_specific_params(dataset_name)

        # Load and preprocess the dataset
        self.logger.info(f"Processing {dataset_name} dataset with params: {params}")

        # Load raw text based on dataset name
        if dataset_name == 'persona_chat':
            raw_text = self.load_persona_chat(split='train', max_samples=max_samples)
        elif dataset_name == 'writing_prompts':
            raw_text = self.load_writing_prompts(split='train', max_samples=max_samples)
        elif dataset_name == 'openassistant':
            raw_text = self.load_openassistant_dataset(split='train', max_samples=max_samples)
        else:
            # Try to load as a path
            raw_text = self.load_data(dataset_name)

        # Clean and prepare text
        cleaned_text = self.clean_text(raw_text)

        # Create sequences and batches
        sequences = self.create_sequences(cleaned_text, sequence_length)
        batches = self.create_batches(sequences, batch_size, dataset_name=dataset_name)

        # Return processed data
        return {
            'batches': batches,
            'params': params,
            'dataset_name': dataset_name,
            'sequence_length': sequence_length,
            'batch_size': batch_size,
            'num_sequences': len(sequences),
            'num_batches': len(batches)
        }

    def _get_dataset_specific_params(self, dataset_name):
        """Get dataset-specific parameters for processing"""
        # Default parameters
        params = {
            'sequence_length': self.config['sequence_length'],
            'batch_size': self.config['batch_size'],
            'learning_rate': 0.001,
            'dropout': 0.2
        }

        # Dataset-specific overrides
        if dataset_name == 'persona_chat':
            # Persona chat needs smaller sequences but larger batch size
            params.update({
                'sequence_length': 128,
                'batch_size': 64,
                'learning_rate': 0.0005,
                'dropout': 0.1
            })
        elif dataset_name == 'writing_prompts':
            # Writing prompts needs longer sequences
            params.update({
                'sequence_length': 256,
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout': 0.3
            })
        elif dataset_name == 'openassistant':
            # OpenAssistant needs special handling
            params.update({
                'sequence_length': 512,
                'batch_size': 16,
                'learning_rate': 0.0003,
                'dropout': 0.2
            })

        return params
