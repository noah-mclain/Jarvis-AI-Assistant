"""
Unified Dataset Handler

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
"""

# Standard library imports
import os
import json
import logging
import datetime
from collections import deque
from typing import Dict, List, Any, Optional, Union, Tuple

# Third-party imports
import torch
from tqdm import tqdm
import glob

# Try to import optional dependencies with graceful fallbacks
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("HuggingFace datasets library not found. Some features will be limited.")

# Local imports
from .dataset_processor import DatasetProcessor
from .improved_preprocessing import ImprovedPreprocessor
from .prompt_enhancer import analyze_prompt
from .utils import (
    get_storage_path,
    sync_to_gdrive,
    sync_from_gdrive,
    ensure_directory_exists,
    is_paperspace_environment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnifiedDatasetHandler")

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

class UnifiedDatasetHandler:
    """
    A unified handler for processing and managing datasets for generative AI tasks.

    This class implements dataset loading, preprocessing, and saving functionality
    with support for multiple dataset formats and conversation context management.
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

    def __init__(self, dataset_name=None, dataset_path=None, output_dir=None, cache_dir=None):
        """
        Initialize the dataset handler.

        Args:
            dataset_name: Name of the dataset to load from HuggingFace Hub
            dataset_path: Path to a local dataset
            output_dir: Directory to save processed datasets
            cache_dir: Directory to cache downloaded datasets
        """
        # Set up basic attributes
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.logger = logging.getLogger("UnifiedDatasetHandler")

        # Set up directories
        self._setup_directories(output_dir, cache_dir)

        # Initialize dataset and model attributes
        self.dataset = None
        self.tokenizer = None
        self.max_length = self.DEFAULT_CONFIG["max_length"]
        self.batch_size = self.DEFAULT_CONFIG["batch_size"]

        # Sync datasets if in Paperspace environment
        self._sync_datasets_if_needed()

        # Initialize processors and formatters
        self._initialize_processors()

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

        self.logger.info("Unified Dataset Handler initialized")

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

    def _initialize_processors(self):
        """Initialize dataset processors and formatters."""
        # Initialize processors
        self.processor = DatasetProcessor()
        self.improved_processor = ImprovedPreprocessor()

        # Create dataset-specific formatters
        self.formatters = {
            "writing_prompts": self._format_writing_prompts,
            "persona_chat": self._format_persona_chat,
            "pile": self._format_pile,
            "openassistant": self._format_openassistant,
            "gpteacher": self._format_gpteacher
        }

        # Create dataset-specific validators
        self.validators = {
            "writing_prompts": self._validate_writing_prompts,
            "persona_chat": self._validate_persona_chat,
            "pile": self._validate_pile,
            "openassistant": self._validate_openassistant,
            "gpteacher": self._validate_gpteacher
        }

    def load_dataset(self, dataset_name, split='train', max_samples=None,
                   subset=None, cache_dir=None, use_cache=True):
        """
        Load and process a dataset

        Args:
            dataset_name: Name of the dataset or path to the dataset
            split: Dataset split (train, validation, test)
            max_samples: Maximum number of samples to load
            subset: Optional subset name for datasets with subsets
            cache_dir: Optional directory to cache preprocessed datasets
            use_cache: Whether to use cached versions

        Returns:
            Processed dataset as DatasetDict
        """
        cache_dir = cache_dir or os.path.join(self.data_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Generate a cache file name that includes dataset name, split, and sample count
        cache_params = f"{dataset_name.replace('/', '_')}_{split}"
        if max_samples:
            cache_params += f"_{max_samples}samples"
        if subset:
            cache_params += f"_{subset}"

        cache_file = os.path.join(cache_dir, f"{cache_params}_cache.pt")

        # Load from cache if available
        if use_cache and os.path.exists(cache_file):
            self.logger.info(f"Loading cached dataset from {cache_file}")
            try:
                return torch.load(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to load cached dataset: {e}")
                # Continue to load from scratch

        # Check if the dataset is a HuggingFace dataset (contains '/')
        if '/' in dataset_name:
            # It's a HuggingFace dataset
            self.logger.info(f"Loading HuggingFace dataset: {dataset_name} (split: {split}, max_samples: {max_samples}, subset: {subset})")

            try:
                from datasets import load_dataset

                # Prepare cache directory for HuggingFace datasets
                hf_cache_dir = os.path.join(self.data_dir, 'huggingface_cache')
                os.makedirs(hf_cache_dir, exist_ok=True)

                # Load the dataset with caching
                dataset = load_dataset(
                    dataset_name,
                    name=subset,
                    split=split,
                    cache_dir=hf_cache_dir
                )

                # Limit the number of samples if specified
                if max_samples is not None and max_samples < len(dataset):
                    self.logger.info(f"Limiting dataset to {max_samples} samples (from {len(dataset)})")
                    dataset = dataset.select(range(min(max_samples, len(dataset))))

                # Process the dataset based on its structure
                return self._process_huggingface_dataset(dataset, dataset_name, cache_file)

            except Exception as e:
                self.logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
                raise

        # Handle local datasets or standard datasets
        if os.path.exists(dataset_name):
            # It's a local path
            return self._load_local_dataset(dataset_name, split, max_samples, cache_file)
        else:
            # Try loading as a standard dataset
            return self._load_standard_dataset(dataset_name, split, max_samples, subset, cache_file)

    def _process_huggingface_dataset(self, dataset, dataset_name, cache_file=None):
        """
        Process a HuggingFace dataset with memory-efficient batching.

        This method processes datasets in batches to avoid memory issues, extracts
        text based on dataset format, and tokenizes the text if a tokenizer is available.

        Args:
            dataset: The HuggingFace dataset object
            dataset_name: Name of the dataset (for identification)
            cache_file: Optional path to save the processed dataset

        Returns:
            Dictionary with 'train' key containing the processed Dataset object
        """
        from datasets import Dataset
        import gc
        from tqdm import tqdm

        self.logger.info(f"Processing HuggingFace dataset ({len(dataset)} samples)")

        # Special handling for OpenAssistant datasets
        if "OpenAssistant" in dataset_name or "oasst" in dataset_name:
            self.logger.info("Detected OpenAssistant dataset format")
            # Log the dataset structure for debugging
            if len(dataset) > 0:
                sample_item = dataset[0]
                self.logger.info(f"OpenAssistant sample item structure: {sample_item}")

                # Check if we need to convert the dataset format
                if 'message_tree' in sample_item and isinstance(sample_item['message_tree'], str):
                    self.logger.warning("OpenAssistant message_tree is a string, not a dictionary. This format is not directly supported.")
                    self.logger.info("Attempting to process with alternative method...")

                    # Try to extract using direct role/text fields
                    if 'role' in sample_item and 'text' in sample_item:
                        self.logger.info("Using role/text fields directly for extraction")

        # Check for empty dataset
        if not self.validate_dataset_structure(dataset, dataset_name):
            return self._create_empty_dataset()

        # Initialize collection containers
        all_texts, all_input_ids, all_attention_masks = [], [], []
        batch_size = getattr(self, 'processing_batch_size', 1000)  # Use class attribute or default to 1000

        # Process dataset in batches
        for batch_data in self._generate_batches(dataset, batch_size, dataset_name):
            batch, batch_index = batch_data

            try:
                # Extract texts from batch based on dataset format
                batch_texts = self._extract_texts_from_batch(batch, dataset_name)

                # Log extraction results
                self._log_batch_extraction_results(batch_texts, batch, dataset_name)

                # Skip empty batches
                if not batch_texts:
                    self.logger.warning(f"No texts extracted from batch {batch_index} in dataset {dataset_name}")
                    continue

                # Tokenize texts if tokenizer is available
                if self.tokenizer:
                    if tokenized_data := self._tokenize_batch(
                        batch_texts, dataset_name
                    ):
                        all_input_ids.extend(tokenized_data['input_ids'])
                        all_attention_masks.extend(tokenized_data['attention_mask'])

                # Always keep texts for future use or debug
                all_texts.extend(batch_texts)

            except Exception as e:
                # Log the error but continue processing other batches
                self.logger.error(f"Error processing batch {batch_index} in dataset {dataset_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
            finally:
                # Force garbage collection after each batch
                del batch
                gc.collect()

        # Create and return the final dataset
        return self._create_final_dataset(all_texts, all_input_ids, all_attention_masks, cache_file)

    def validate_dataset_structure(self, dataset, dataset_name):
        """Validate that the dataset is not empty and log its structure."""
        try:
            # Check if dataset has length
            if not hasattr(dataset, '__len__'):
                self.logger.warning(f"Dataset {dataset_name} doesn't have a length attribute")
                # Try to convert to list
                try:
                    dataset_list = list(dataset)
                    if len(dataset_list) == 0:
                        self.logger.warning(f"Dataset {dataset_name} is empty after conversion to list")
                        return False
                    # Use the list for further validation
                    dataset = dataset_list
                except Exception as e:
                    self.logger.error(f"Cannot convert dataset to list: {e}")
                    # Can't validate further
                    return True  # Assume it's valid and let processing handle it

            # Check if dataset is empty
            if len(dataset) == 0:
                self.logger.warning(f"Dataset {dataset_name} is empty")
                return False

            # Try to access the first item
            try:
                first_item = dataset[0]

                # Check if first item is a dictionary
                if isinstance(first_item, dict):
                    keys = list(first_item.keys())
                    self.logger.info(f"Dataset keys: {keys}")
                    self.logger.info(f"Sample item (first 5 fields): {dict(list(first_item.items())[:5])}")
                else:
                    self.logger.info(f"Dataset first item is not a dictionary: {type(first_item)}")
                    # If it's a string, show a sample
                    if isinstance(first_item, str):
                        self.logger.info(f"Sample text: {first_item[:100]}...")

            except Exception as e:
                self.logger.warning(f"Cannot access first item in dataset: {e}")
                # Can't validate further, but don't fail
                return True  # Assume it's valid and let processing handle it

            return True

        except Exception as e:
            self.logger.error(f"Error validating dataset structure: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return True to allow processing to continue and handle errors there
            return True

    def _create_empty_dataset(self):
        """Create an empty dataset with the required structure."""
        from datasets import Dataset
        return {"train": Dataset.from_dict({"text": [], "input_ids": [], "attention_mask": []})}

    def _generate_batches(self, dataset, batch_size, dataset_name):
        """Generate batches from the dataset with progress tracking."""
        from tqdm import tqdm

        # Handle different dataset types
        try:
            # Check if dataset is a list-like object that supports slicing
            if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
                # Standard case - dataset is a list-like object
                for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {dataset_name}"):
                    end_idx = min(i + batch_size, len(dataset))
                    try:
                        batch = dataset[i:end_idx]
                        yield batch, i // batch_size
                    except Exception as e:
                        self.logger.error(f"Error slicing dataset at indices {i}:{end_idx}: {e}")
                        # Try getting items one by one
                        try:
                            batch = [dataset[j] for j in range(i, end_idx)]
                            yield batch, i // batch_size
                        except Exception as e2:
                            self.logger.error(f"Error getting individual items: {e2}")
                            # As a last resort, yield the entire dataset as one batch
                            if i == 0:  # Only do this once
                                self.logger.warning(f"Falling back to processing entire dataset as one batch")
                                yield dataset, 0
            else:
                # Dataset doesn't support slicing - try to convert to list or process as is
                self.logger.warning(f"Dataset of type {type(dataset)} doesn't support standard slicing")

                # Try to convert to list
                try:
                    dataset_list = list(dataset)
                    self.logger.info(f"Converted dataset to list with {len(dataset_list)} items")

                    for i in tqdm(range(0, len(dataset_list), batch_size), desc=f"Processing {dataset_name}"):
                        end_idx = min(i + batch_size, len(dataset_list))
                        yield dataset_list[i:end_idx], i // batch_size

                except Exception as e:
                    self.logger.error(f"Cannot convert dataset to list: {e}")
                    # Process as a single batch
                    self.logger.warning(f"Processing entire dataset as one batch")
                    yield dataset, 0

        except Exception as e:
            self.logger.error(f"Error generating batches: {e}")
            # As a last resort, yield the entire dataset as one batch
            self.logger.warning(f"Falling back to processing entire dataset as one batch due to error")
            yield dataset, 0

    def _extract_texts_from_batch(self, batch, dataset_name):
        """Extract texts from a batch based on dataset format."""
        if "OpenAssistant" in dataset_name or "oasst" in dataset_name:
            return self._extract_openassistant_batch(batch)
        elif "GPTeacher" in dataset_name:
            return self._extract_gpteacher_batch(batch)
        elif "Persona-Chat" in dataset_name or "Synthetic-Persona-Chat" in dataset_name:
            return self._extract_persona_chat_batch(batch)
        elif "writingprompts" in dataset_name:
            return self._extract_writing_prompts_batch(batch)
        else:
            return self._extract_generic_batch(batch)

    def _extract_openassistant_batch(self, batch):
        """Extract texts from OpenAssistant format batch."""
        batch_texts = []

        # Log batch type for debugging
        self.logger.info(f"OpenAssistant batch type: {type(batch)}")

        # Handle different batch types
        if isinstance(batch, dict):
            # Process a single item (dictionary)
            items = [batch]
            self.logger.info("Processing batch as a single dictionary item")
        elif isinstance(batch, list):
            # Process a list of items
            items = batch
            # Log the first item for debugging if available
            if items and len(items) > 0:
                self.logger.info(f"OpenAssistant sample item: {items[0]}")
        else:
            # Try to convert to list or handle as iterable
            try:
                items = list(batch)
                self.logger.info(f"Converted batch to list with {len(items)} items")
            except Exception as e:
                self.logger.error(f"Cannot process batch of type {type(batch)}: {e}")
                return batch_texts

        # Process each item
        for item in items:
            try:
                # Skip non-dictionary items
                if not isinstance(item, dict):
                    self.logger.warning(f"Skipping non-dictionary item: {type(item)}")
                    continue

                # Handle OpenAssistant v2 structure with message trees
                if 'message_tree' in item and isinstance(item['message_tree'], dict) and 'messages' in item['message_tree']:
                    prompt = None
                    messages = item['message_tree']['messages']

                    # Ensure messages is a list
                    if not isinstance(messages, list):
                        continue

                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue

                        if 'role' in msg and 'text' in msg:
                            if msg['role'] == 'prompter':
                                prompt = msg['text']
                            elif msg['role'] == 'assistant' and prompt:
                                response = msg['text']
                                batch_texts.append(f"USER: {prompt}\nASSISTANT: {response}")
                                prompt = None  # Reset for next pair

                # Handle OpenAssistant v1 structure with message_tree_id
                elif 'message_tree_id' in item and 'text' in item and 'role' in item:
                    if item['role'] == 'assistant':
                        batch_texts.append(f"USER: [Previous question]\nASSISTANT: {item['text']}")
                    elif item['role'] == 'prompter':
                        batch_texts.append(f"USER: {item['text']}\nASSISTANT:")

                # Handle simpler structure with direct role and text
                elif 'text' in item and 'role' in item:
                    if item['role'] == 'assistant':
                        batch_texts.append(f"USER: [Previous question]\nASSISTANT: {item['text']}")
                    elif item['role'] == 'prompter':
                        batch_texts.append(f"USER: {item['text']}\nASSISTANT:")

            except Exception as e:
                # Log the error and continue with the next item
                self.logger.warning(f"Error processing OpenAssistant item: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
                continue

        self.logger.info(f"Extracted {len(batch_texts)} texts from OpenAssistant batch")
        return batch_texts

    def _extract_gpteacher_batch(self, batch):
        """Extract texts from GPTeacher format batch."""
        batch_texts = []
        for item in batch:
            if 'instruction' in item and 'response' in item:
                batch_texts.append(f"User: {item['instruction']}\nAssistant: {item['response']}")
        return batch_texts

    def _extract_persona_chat_batch(self, batch):
        """Extract texts from Persona-Chat format batch."""
        batch_texts = []
        for item in batch:
            if 'user 1 personas' in item and 'Best Generated Conversation' in item:
                # Handle Synthetic-Persona-Chat format
                persona = self._format_persona(item['user 1 personas'])
                conversation = item['Best Generated Conversation']
                if conversation:
                    batch_texts.append(f"Persona: {persona}\nConversation: {conversation}")
            elif 'personas' in item and 'utterances' in item:
                # Handle regular Persona-Chat format
                persona = self._format_persona(item['personas'])
                if isinstance(item['utterances'], list) and len(item['utterances']) > 0:
                    # Handle different structure variations
                    utterance = item['utterances'][-1]  # Take last utterance pair
                    if isinstance(utterance, list) and len(utterance) >= 2:
                        batch_texts.append(f"Persona: {persona}\nUser: {utterance[0]}\nAssistant: {utterance[1]}")
        return batch_texts

    def _format_persona(self, personas):
        """Format persona data consistently."""
        return "\n".join(personas) if isinstance(personas, list) else str(personas)

    def _extract_writing_prompts_batch(self, batch):
        """Extract texts from WritingPrompts format batch."""
        batch_texts = []
        for item in batch:
            if 'prompt' in item and 'story' in item:
                batch_texts.append(f"Prompt: {item['prompt']}\nStory: {item['story']}")
        return batch_texts

    def _extract_generic_batch(self, batch):
        """Extract texts from generic format batch using common patterns."""
        batch_texts = []
        for item in batch:
            if 'input' in item and 'output' in item:
                batch_texts.append(f"Input: {item['input']}\nOutput: {item['output']}")
            elif 'question' in item and 'answer' in item:
                batch_texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")
            elif 'prompt' in item and 'completion' in item:
                batch_texts.append(f"Prompt: {item['prompt']}\nCompletion: {item['completion']}")
            elif 'text' in item:
                batch_texts.append(item['text'])
            elif 'content' in item:
                batch_texts.append(item['content'])
        return batch_texts

    def _log_batch_extraction_results(self, batch_texts, batch, dataset_name):
        """Log the results of text extraction from a batch."""
        self.logger.info(f"Extracted {len(batch_texts)} texts from batch")
        if not batch_texts and batch and isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
            self.logger.warning(f"No texts extracted from batch. Sample item: {dict(list(batch[0].items())[:3])}")

    def _tokenize_batch(self, batch_texts, dataset_name):
        """Tokenize a batch of texts."""
        try:
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            # CRITICAL: Keep tensors on CPU - this is essential for DataLoader compatibility
            # They'll be moved to GPU during the training loop
            return {
                'input_ids': encodings['input_ids'].cpu().tolist(),
                'attention_mask': encodings['attention_mask'].cpu().tolist()
            }
        except Exception as e:
            self.logger.error(f"Error tokenizing batch in dataset {dataset_name}: {e}")
            return None

    def _create_final_dataset(self, all_texts, all_input_ids, all_attention_masks, cache_file):
        """Create the final dataset object and optionally cache it."""
        from datasets import Dataset

        # Create the dataset dictionary
        processed_dataset = {
            "text": all_texts,
            "input_ids": all_input_ids if self.tokenizer else [],
            "attention_mask": all_attention_masks if self.tokenizer else []
        }

        # Convert to Dataset object
        dataset_dict = {"train": Dataset.from_dict(processed_dataset)}

        # Cache the processed dataset if cache_file is provided
        if cache_file:
            self._cache_dataset(dataset_dict, cache_file)

        return dataset_dict

    def _cache_dataset(self, dataset_dict, cache_file):
        """Cache the processed dataset to a file."""
        self.logger.info(f"Saving processed dataset to {cache_file}")
        try:
            torch.save(dataset_dict, cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to cache dataset: {e}")

    def prepare_for_training(self,
                        dataset: Dict[str, Any],
                        batch_size: int = 32,
                        validation_split: float = 0.2,
                        test_split: float = 0.1,
                        max_target_idx: int = 100) -> Dict[str, Dict[str, List]]:
        """
        Prepare dataset for training by creating train/validation/test splits

        Args:
            dataset: Dataset dictionary with batches
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            max_target_idx: Maximum allowed target index to prevent out-of-bounds errors

        Returns:
            Dictionary of train/validation/test splits with batches
        """
        if 'batches' not in dataset or not dataset['batches']:
            raise ValueError("No batches found in dataset")

        # Get all batches
        all_batches = dataset['batches']

        # For safety, ensure all target indices are within bounds
        # This prevents CUDA device-side assert errors when using CrossEntropyLoss
        safe_batches = []
        for input_batch, target_batch in all_batches:
            # Check if target has any out-of-bounds indices
            if hasattr(target_batch, 'max') and target_batch.max() > max_target_idx:
                # Clamp target indices to prevent CUDA errors later
                safe_target = torch.clamp(target_batch, 0, max_target_idx)
                safe_batches.append((input_batch, safe_target))
            else:
                safe_batches.append((input_batch, target_batch))

        # Calculate splits
        total_batches = len(safe_batches)
        val_size = int(total_batches * validation_split)
        test_size = int(total_batches * test_split)
        train_size = total_batches - val_size - test_size

        # Create splits
        train_batches = safe_batches[:train_size]
        val_batches = safe_batches[train_size:train_size+val_size] if val_size > 0 else []
        test_batches = safe_batches[train_size+val_size:] if test_size > 0 else []

        logger.info(f"Splitting dataset into {len(train_batches)} train, {len(val_batches)} validation, and {len(test_batches)} test batches")

        # Create result dictionary
        result = {
            'train': {'batches': train_batches},
            'validation': {'batches': val_batches},
            'test': {'batches': test_batches}
        }

        # Add metadata
        for split_name, split_data in result.items():
            if dataset.get('vocab_size'):
                split_data['vocab_size'] = dataset['vocab_size']
            if dataset.get('metadata'):
                split_data['metadata'] = dataset['metadata']

        return result

    def get_batch_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about dataset batches.

        Args:
            dataset: Dataset dictionary

        Returns:
            Dictionary with statistics
        """
        if not dataset or 'batches' not in dataset or not dataset['batches']:
            logger.error("Invalid dataset or no batches found")
            return {}

        # Get all batches
        all_batches = dataset['batches']
        total_batches = len(all_batches)

        # Calculate statistics
        input_lengths = []
        target_lengths = []

        for input_batch, target_batch in all_batches:
            input_lengths.append(input_batch.size(1))
            target_lengths.append(target_batch.size(1))

        return {
            "total_batches": total_batches,
            "input_length_min": min(input_lengths, default=0),
            "input_length_max": max(input_lengths, default=0),
            "input_length_avg": (
                sum(input_lengths) / len(input_lengths) if input_lengths else 0
            ),
            "target_length_min": min(target_lengths, default=0),
            "target_length_max": max(target_lengths, default=0),
            "target_length_avg": (
                sum(target_lengths) / len(target_lengths) if target_lengths else 0
            ),
        }

    def determine_best_dataset(self, prompt: str) -> str:
        """
        Determine the best dataset for a given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Name of the best dataset
        """
        return analyze_prompt(prompt)

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.DATASET_INFO:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            return {"error": f"Unsupported dataset: {dataset_name}"}

        dataset_info = {
            "name": dataset_name,
            "description": self.DATASET_INFO.get(dataset_name, ""),
            "processor_type": "ImprovedPreprocessor" if dataset_name in ["writing_prompts", "persona_chat"]
                                                else "DatasetProcessor",
            "format": self._get_dataset_format(dataset_name),
            "typical_usage": self._get_dataset_usage(dataset_name),
            "available": self._check_dataset_availability(dataset_name)
        }

        return dataset_info

    def list_all_datasets(self) -> List[Dict[str, Any]]:
        """
        List information about all supported datasets.

        Returns:
            List of dictionaries with dataset information
        """
        return [self.get_dataset_info(dataset) for dataset in self.DATASET_INFO.keys()]

    def extract_prompt_response_pairs(self, dataset_name: str, dataset: Dict[str, Any],
                                    max_pairs: int = 10) -> List[Dict[str, str]]:
        """
        Extract human-readable prompt-response pairs from a processed dataset.

        Args:
            dataset_name: Name of the dataset
            dataset: Processed dataset
            max_pairs: Maximum number of pairs to extract

        Returns:
            List of dictionaries with prompt-response pairs
        """
        if dataset_name not in self.DATASET_INFO:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            return []

        # Use the appropriate formatter function
        formatter = self.formatters.get(dataset_name)
        if formatter and 'batches' in dataset and dataset['batches']:
            # Get raw data if available
            dataset_loaders = {
                "writing_prompts": self.processor.load_writing_prompts,
                "persona_chat": self.processor.load_persona_chat,
                "pile": self.processor.load_pile_dataset,
                "openassistant": self.processor.load_openassistant_dataset,
                "gpteacher": self.processor.load_gpteacher_dataset
            }

            # Get the appropriate loader function
            loader = dataset_loaders.get(dataset_name)
            raw_data = loader(max_samples=max_pairs) if loader else None

            # Format the data into prompt-response pairs
            return formatter(raw_data, max_pairs) if raw_data else []

        return []

    def validate_dataset_content(self, dataset_name: str, data: Dict[str, Any]) -> bool:
        """
        Validate a dataset for correctness and completeness.

        Args:
            dataset_name: Name of the dataset
            data: Dataset to validate

        Returns:
            True if validation passed, False otherwise
        """
        # Check for batches
        if 'batches' not in data or not data['batches']:
            self.logger.error(f"Dataset {dataset_name} has no batches")
            return False

        # Check batch format
        for i, batch in enumerate(data['batches'][:10]):  # Check first 10 batches
            if not isinstance(batch, tuple) or len(batch) != 2:
                self.logger.error(f"Dataset {dataset_name}, batch {i} has invalid format")
                return False

            input_batch, target_batch = batch
            if not isinstance(input_batch, torch.Tensor) or not isinstance(target_batch, torch.Tensor):
                self.logger.error(f"Dataset {dataset_name}, batch {i} contains non-tensor data")
                return False

        # Use the dataset-specific validator
        validator = self.validators.get(dataset_name)
        return validator(data) if validator else True

    # Alias for backward compatibility
    _validate_dataset = validate_dataset_content

    # Dataset-specific formatters
    def _format_writing_prompts(self, raw_data: str, max_pairs: int) -> List[Dict[str, str]]:
        """Format writing_prompts data into prompt-response pairs"""
        if not raw_data:
            return []

        pairs = []
        examples = raw_data.split('<END>\n\n')[:max_pairs]

        for example in examples:
            if not example.strip():
                continue

            # Extract prompt and story parts
            prompt_parts = example.split('<STORY>')
            if len(prompt_parts) != 2:
                continue

            prompt_text = prompt_parts[0].replace('<PROMPT>\n', '').strip()
            story_text = prompt_parts[1].strip()

            pairs.append({"prompt": prompt_text, "response": story_text})

        return pairs

    def _format_persona_chat(self, raw_data: str, max_pairs: int) -> List[Dict[str, str]]:
        """Format persona_chat data into prompt-response pairs"""
        if not raw_data:
            return []

        pairs = []
        examples = raw_data.split('<END>\n\n')[:max_pairs]

        for example in examples:
            if not example.strip():
                continue

            # Extract persona and dialogue parts
            parts = example.split('<DIALOGUE>')
            if len(parts) != 2:
                continue

            persona_text = parts[0].replace('<PERSONA>\n', '').strip()
            dialogue_text = parts[1].strip()

            # Extract user and assistant exchanges
            dialogue_lines = dialogue_text.split('\n')
            for i in range(0, len(dialogue_lines) - 1, 2):
                if i + 1 < len(dialogue_lines):
                    user_line = dialogue_lines[i].replace('USER: ', '').strip()
                    assistant_line = dialogue_lines[i + 1].replace('ASSISTANT: ', '').strip()

                    # Create prompt with persona context
                    prompt = f"Persona:\n{persona_text}\n\nConversation:\n{user_line}"

                    pairs.append({"prompt": prompt, "response": assistant_line})

                    if len(pairs) >= max_pairs:
                        break

            if len(pairs) >= max_pairs:
                break

        return pairs

    def _format_pile(self, raw_data: str, max_pairs: int) -> List[Dict[str, str]]:
        """Format pile data into prompt-response pairs"""
        if not raw_data:
            return []

        pairs = []
        # The Pile is not naturally prompt/response, so we'll split text into chunks
        text_chunks = [raw_data[i:i+500] for i in range(0, len(raw_data), 500)][:max_pairs*2]

        for i in range(0, len(text_chunks) - 1, 2):
            if i + 1 < len(text_chunks):
                prompt = text_chunks[i].strip()
                response = text_chunks[i + 1].strip()

                pairs.append({"prompt": prompt, "response": response})

                if len(pairs) >= max_pairs:
                    break

        return pairs

    def _format_openassistant(self, raw_data: Union[str, List[Dict[str, Any]]], max_pairs: int) -> List[Dict[str, str]]:
        """
        Format OpenAssistant data into prompt-response pairs.

        This method handles two formats:
        1. String format with "USER:" and "ASSISTANT:" markers
        2. HuggingFace dataset format with message trees

        Args:
            raw_data: Either a string of raw text or a list of dictionaries from the dataset
            max_pairs: Maximum number of pairs to extract

        Returns:
            List of dictionaries with prompt-response pairs
        """
        if isinstance(raw_data, str):
            return self._extract_pairs_from_text(raw_data, max_pairs)
        else:
            return self._extract_pairs_from_message_tree(raw_data, max_pairs)

    def _extract_pairs_from_text(self, text: str, max_pairs: int) -> List[Dict[str, str]]:
        """Extract prompt-response pairs from text format."""
        pairs = []
        parts = text.split("USER: ")

        # Skip the first split which might be empty
        for part in parts[1:]:
            if "ASSISTANT: " not in part:
                continue

            user_text, assistant_text = part.split("ASSISTANT: ", 1)

            # Handle multi-turn conversations by taking just the first response
            if "USER: " in assistant_text:
                assistant_text = assistant_text.split("USER: ")[0]

            pairs.append({
                "prompt": user_text.strip(),
                "response": assistant_text.strip()
            })

            if len(pairs) >= max_pairs:
                break

        return pairs

    def _extract_pairs_from_message_tree(self, messages: List[Dict[str, Any]], max_pairs: int) -> List[Dict[str, str]]:
        """Extract prompt-response pairs from HuggingFace dataset format."""
        pairs = []
        message_tree = self._build_message_tree(messages)

        # Extract conversation pairs (prompt-response)
        for msg_id, msg in message_tree.items():
            # Skip if we've reached the maximum number of pairs
            if len(pairs) >= max_pairs:
                break

            # Skip if not a prompter message or has no children
            if msg['role'] != 'prompter' or not msg['children']:
                continue

            # Find assistant responses among children
            for child_id in msg['children']:
                child = message_tree.get(child_id)
                if not child or child['role'] != 'assistant':
                    continue

                pairs.append({
                    "prompt": msg['text'].strip(),
                    "response": child['text'].strip()
                })

                if len(pairs) >= max_pairs:
                    break

        return pairs

    def _build_message_tree(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build a message tree from a list of messages."""
        message_tree = {}

        # First pass: organize messages by message_id
        for item in messages:
            if not isinstance(item, dict):
                continue

            # Skip items missing required fields
            if not all(key in item for key in ('message_id', 'role', 'text')):
                continue

            message_tree[item['message_id']] = {
                'role': item['role'],
                'text': item['text'],
                'parent_id': item.get('parent_id'),
                'children': []
            }

        # Second pass: build the tree structure
        for msg_id, msg in message_tree.items():
            parent_id = msg['parent_id']
            if parent_id and parent_id in message_tree:
                # Add this message as a child of its parent
                message_tree[parent_id]['children'].append(msg_id)

        return message_tree

    def _format_gpteacher(self, raw_data: str, max_pairs: int) -> List[Dict[str, str]]:
        """Format gpteacher data into prompt-response pairs"""
        if not raw_data:
            return []

        pairs = []
        # Split by obvious markers
        parts = raw_data.split("USER: ")

        for part in parts[1:]:  # Skip the first split which might be empty
            if "ASSISTANT: " in part:
                user_text, assistant_text = part.split("ASSISTANT: ", 1)

                # Handle multi-turn conversations by taking just the first response
                if "USER: " in assistant_text:
                    assistant_text = assistant_text.split("USER: ")[0]

                pairs.append({
                    "prompt": user_text.strip(),
                    "response": assistant_text.strip()
                })

                if len(pairs) >= max_pairs:
                    break

        return pairs

    # Dataset-specific validators
    def _validate_writing_prompts(self, data: Dict[str, Any]) -> bool:
        """Validate writing_prompts dataset"""
        # Check that at least some batches have a decent length
        if 'batches' in data and data['batches']:
            for input_batch, target_batch in data['batches'][:10]:
                # Check input batch has sufficient length
                if input_batch.size(1) < 5:
                    logger.warning("Writing prompts dataset has suspiciously short input sequences")
                    return False

                # Check target batch shape - if it's 2D check size(1), otherwise check size(0)
                if target_batch.dim() > 1:
                    if target_batch.size(1) < 10:
                        logger.warning("Writing prompts dataset has suspiciously short target sequences")
                        return False
                elif len(target_batch) < 10:
                    logger.warning("Writing prompts dataset has suspiciously few target elements")
                    return False
        return True

    def _validate_persona_chat(self, data: Dict[str, Any]) -> bool:
        """Validate persona_chat dataset"""
        # Check that at least some batches have a decent length
        if 'batches' in data and data['batches']:
            for input_batch, target_batch in data['batches'][:10]:
                # Check input batch has sufficient length
                if input_batch.size(1) < 5:
                    logger.warning("Persona chat dataset has suspiciously short input sequences")
                    return False

                # Check target batch shape - if it's 2D check size(1), otherwise check size(0)
                if target_batch.dim() > 1:
                    if target_batch.size(1) < 5:
                        logger.warning("Persona chat dataset has suspiciously short target sequences")
                        return False
                elif len(target_batch) < 5:
                    logger.warning("Persona chat dataset has suspiciously few target elements")
                    return False
        return True

    def _validate_pile(self, data: Dict[str, Any]) -> bool:
        """Validate pile dataset"""
        # The Pile should have substantial text content
        if 'batches' in data and data['batches']:
            # Calculate total length across input batches
            total_length = 0
            for input_batch, _ in data['batches'][:10]:
                # Add the sequence length for each batch
                if input_batch.dim() > 1:
                    total_length += input_batch.size(1)
                else:
                    total_length += len(input_batch)

            if total_length < 500:  # Arbitrary threshold
                logger.warning("Pile dataset has suspiciously little text content")
                return False
        return True

    def _validate_openassistant(self, data: Dict[str, Any]) -> bool:
        """Validate openassistant dataset"""
        # Check for question-answer pattern
        return True  # Basic validation already done in _validate_dataset

    def _validate_gpteacher(self, data: Dict[str, Any]) -> bool:
        """Validate gpteacher dataset"""
        return True  # Basic validation already done in _validate_dataset

    def _get_dataset_format(self, dataset_name: str) -> str:
        """Get the format description for a dataset"""
        formats = {
            "writing_prompts": "<PROMPT>\\n[prompt]\\n<STORY>\\n[story]\\n<END>",
            "persona_chat": "<PERSONA>\\n[traits]\\n<DIALOGUE>\\n[USER/ASSISTANT exchanges]\\n<END>",
            "pile": "Raw text from various sources (academic papers, books, websites)",
            "openassistant": "USER: [question]\\nASSISTANT: [response]",
            "gpteacher": "USER: [instruction]\\nASSISTANT: [instruction following]"
        }
        return formats.get(dataset_name, "Unknown format")

    def _get_dataset_usage(self, dataset_name: str) -> str:
        """Get the typical usage description for a dataset"""
        usages = {
            "writing_prompts": "Creative writing, story generation, fictional content",
            "persona_chat": "Dialogue systems, chatbots, persona-consistent responses",
            "pile": "General knowledge, fact-based Q&A, academic content",
            "openassistant": "Helpful assistant responses, task completion",
            "gpteacher": "How-to guides, tutorials, step-by-step instructions"
        }
        return usages.get(dataset_name, "Unknown usage")

    def _check_dataset_availability(self, dataset_name: str) -> bool:
        """Check if a dataset is available for download"""
        if not DATASETS_AVAILABLE:
            logger.warning("Cannot check dataset availability as datasets library is not installed")
            return False

        try:
            # Try to load a tiny sample to verify availability
            if dataset_name == "writing_prompts":
                # Check if HuggingFace has the dataset
                dataset = load_dataset("euclaise/writingprompts", split="train", cache_dir=self.cache_dir)
                return dataset is not None and len(dataset) > 0
            elif dataset_name == "persona_chat":
                try:
                    dataset = load_dataset("bavard/personachat_truecased", split="train", cache_dir=self.cache_dir)
                except Exception:
                    dataset = load_dataset("persona_chat", split="train", cache_dir=self.cache_dir)
                return dataset is not None and len(dataset) > 0
            elif dataset_name == "pile":
                # The Pile is available via other processors
                return True
            elif dataset_name == "openassistant":
                # Check if HuggingFace has the dataset
                dataset = load_dataset("OpenAssistant/oasst1", split="train", cache_dir=self.cache_dir)
                return dataset is not None and len(dataset) > 0
            elif dataset_name == "gpteacher":
                # Check if HuggingFace has the dataset
                dataset = load_dataset("teknium/GPTeacher-General-Instruct", split="train", cache_dir=self.cache_dir)
                return dataset is not None and len(dataset) > 0

            return False
        except Exception as e:
            logger.warning(f"Error checking availability of {dataset_name}: {e}")
            return False

    def generate_with_context(self, generator, prompt: str, temperature: float = 0.7,
                             max_length: int = 200) -> str:
        """
        Generate text with conversation context.

        Args:
            generator: Text generator model
            prompt: The user prompt
            temperature: Temperature for generation
            max_length: Maximum length of generated text

        Returns:
            Generated text
        """
        # Format the context with the current prompt
        context_text = self.conversation_context.get_formatted_history(
            include_current=True,
            current_input=prompt
        )

        # Generate text using the context
        if context_text:
            generated = generator.generate(
                initial_str=context_text,
                pred_len=max_length,
                temperature=temperature
            )

            # Extract just the assistant's response
            if "ASSISTANT: " in generated:
                response = generated.split("ASSISTANT: ")[-1].strip()
            else:
                response = generated
        else:
            # No context yet, just use the prompt
            generated = generator.generate(
                initial_str=f"USER: {prompt}\nASSISTANT: ",
                pred_len=max_length,
                temperature=temperature
            )

            # Extract just the assistant's response
            if "ASSISTANT: " in generated:
                response = generated.split("ASSISTANT: ")[-1].strip()
            else:
                response = generated

        # Add the exchange to history
        self.conversation_context.add_exchange(prompt, response)

        return response

    def save_conversation(self, filepath: str) -> None:
        """
        Save the current conversation context to a file.

        Args:
            filepath: Path to save the conversation context
        """
        self.conversation_context.save_to_file(filepath)

    def load_conversation(self, filepath: str) -> None:
        """
        Load a conversation context from a file.

        Args:
            filepath: Path to the conversation context file
        """
        self.conversation_context = ConversationContext.load_from_file(filepath)

    def clear_conversation(self) -> None:
        """Clear the current conversation context"""
        self.conversation_context.clear()

    def get_best_dataset_for_prompt(self, prompt: str) -> str:
        """
        Determine the best dataset for a given prompt.
        Uses the prompt analyzer to make the decision.

        Args:
            prompt: The user prompt

        Returns:
            Name of the best dataset
        """
        return analyze_prompt(prompt)

    def save_processed_dataset(self, dataset, output_path=None, dataset_name=None):
        """
        Save a processed dataset to disk and sync to Google Drive.

        Args:
            dataset: The dataset to save
            output_path (str, optional): Path to save the dataset
            dataset_name (str, optional): Name to use for the dataset file

        Returns:
            str: Path where the dataset was saved
        """
        if output_path is None:
            dataset_name = dataset_name or f"processed_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_path = os.path.join(self.output_dir, dataset_name)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the dataset
        logger.info(f"Saving processed dataset to {output_path}")
        dataset.save_to_disk(output_path)

        # Sync to Google Drive if in Paperspace
        if is_paperspace_environment():
            sync_to_gdrive("datasets")
            logger.info("Synced datasets to Google Drive")

        return output_path

    @staticmethod
    def list_available_datasets(directory=None):
        """
        List all available processed datasets.

        Args:
            directory (str, optional): Directory to look for datasets. Defaults to the datasets directory.

        Returns:
            list: List of dataset paths
        """
        # Try to sync from Google Drive first to get the latest datasets if in Paperspace
        if is_paperspace_environment():
            try:
                sync_from_gdrive("datasets")
                logger.info("Synced latest datasets from Google Drive")
            except Exception as e:
                logger.warning(f"Failed to sync datasets from Google Drive: {str(e)}")

        # Get and ensure the directory exists
        directory = directory or ensure_directory_exists("datasets")

        # Find directories that contain dataset files
        dataset_dirs = []
        for root, dirs, files in os.walk(directory):
            if any(f == "dataset_info.json" for f in files):
                dataset_dirs.append(root)

        return dataset_dirs

    def determine_best_model(self, prompt: str) -> str:
        """
        Select the best model based on prompt analysis

        Args:
            prompt: The user prompt

        Returns:
            The name of the model to use
        """
        # Determine input type (code, text, conversation)
        is_code_related = any(keyword in prompt.lower() for keyword in [
            "code", "function", "program", "script", "algorithm",
            "python", "javascript", "java", "c++", "html", "css",
            "ruby", "php", "swift", "kotlin", "sql", "git", "function",
            "class", "variable", "method", "api", "code snippet"
        ])

        is_creative = any(keyword in prompt.lower() for keyword in [
            "story", "write", "creative", "imagine", "fiction", "narrate",
            "poem", "poetry", "novel", "tale", "fantasy", "create a scenario",
            "describe a scene", "essay", "article", "blog post"
        ])

        is_analytical = any(keyword in prompt.lower() for keyword in [
            "analysis", "explain", "research", "compare", "contrast",
            "pros and cons", "advantages", "disadvantages", "evaluate",
            "examine", "investigate", "study", "review", "summarize"
        ])

        is_conversation = len(self.conversation_context.history) > 0

        # If we have conversation context, check if it's specialized
        if is_conversation:
            # Analyze conversation history for context
            history_text = " ".join([
                f"{exchange['user']} {exchange['assistant']}"
                for exchange in self.conversation_context.history
            ])

            # Check if history contains code-related content
            history_is_code = any(keyword in history_text.lower() for keyword in [
                "code", "function", "python", "javascript", "programming"
            ])

            # Check if history contains creative content
            history_is_creative = any(keyword in history_text.lower() for keyword in [
                "story", "creative", "fiction", "imagine", "poem"
            ])

            # Use history to influence decision
            if history_is_code:
                is_code_related = True
            if history_is_creative:
                is_creative = True

        # Select model based on analysis, with priorities
        if is_code_related:
            # Code-related prompts get highest priority
            if "deepseek" in self.metadata.get("available_models", []):
                return "deepseek_coder"
            else:
                return "code_generator"
        elif is_creative:
            # Creative tasks
            return "text_generator_creative"
        elif is_analytical:
            # Analytical tasks
            return "text_generator_analytical"
        elif is_conversation:
            # General conversation tasks
            return "text_generator_chat"
        else:
            # Default model for general queries
            return "text_generator_default"

    def get_best_dataset_for_model(self, model_name: str) -> str:
        """
        Get the best dataset to use for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Name of the best dataset to use
        """
        dataset_mapping = {
            "code_generator": "code_search_net",
            "deepseek_coder": "code_search_net",
            "text_generator_creative": "writing_prompts",
            "text_generator_analytical": "pile",
            "text_generator_chat": "persona_chat",
            "text_generator_default": "openassistant"
        }

        return dataset_mapping.get(model_name, "openassistant")

    def load_dataset_with_pagination(self, dataset_name: str, batch_size: int = 1000,
                                max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Load a dataset with pagination to prevent memory issues

        Args:
            dataset_name: Name of the dataset to load
            batch_size: Number of samples to load in each batch
            max_samples: Maximum number of samples to load in total

        Returns:
            Dictionary with processed dataset
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            logger.error(f"Unsupported dataset: {dataset_name}")
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                          f"Supported datasets: {', '.join(self.SUPPORTED_DATASETS)}")

        logger.info(f"Loading dataset with pagination: {dataset_name} "
                  f"(batch_size: {batch_size}, max_samples: {max_samples})")

        # Initialize data collection
        all_batches = []
        total_samples = 0
        metadata = {}

        # Determine which loader to use
        if dataset_name in {"writing_prompts", "persona_chat"}:
            # For these datasets, we can use the improved processor
            # Modify the processors to support pagination
            batches = []

            # Use a generator to paginate through the dataset
            for i, batch in enumerate(self._paginated_dataset_generator(dataset_name, batch_size)):
                batches.extend(batch)
                total_samples += len(batch)

                logger.info(f"Loaded {total_samples} samples so far")

                # Check if we've reached the maximum samples
                if max_samples and total_samples >= max_samples:
                    batches = batches[:max_samples]
                    break

            # Create a data dictionary
            data = {
                'batches': batches,
                'metadata': {
                    'dataset_name': dataset_name,
                    'total_samples': len(batches)
                }
            }

        else:
            # For other datasets, load in batches
            offset = 0
            while True:
                # Load a batch of data
                batch_data = self._load_dataset_batch(dataset_name, offset, batch_size)

                if not batch_data or 'batches' not in batch_data or not batch_data['batches']:
                    # No more data, break out of the loop
                    break

                # Add to collection
                all_batches.extend(batch_data['batches'])
                total_samples += len(batch_data['batches'])

                # Get metadata from the first batch
                if not metadata and 'metadata' in batch_data:
                    metadata = batch_data['metadata']

                logger.info(f"Loaded {total_samples} samples from {dataset_name}")

                # Check if we've reached the maximum samples
                if max_samples and total_samples >= max_samples:
                    all_batches = all_batches[:max_samples]
                    break

                # Update offset for next batch
                offset += batch_size

            # Create the final dataset
            data = {
                'batches': all_batches,
                'metadata': metadata
            }

            # Add total samples to metadata
            data['metadata']['total_samples'] = len(all_batches)

        # Validate dataset
        if data:
            if self.validate_dataset_content(dataset_name, data):
                logger.info(f"Successfully loaded and validated {dataset_name} dataset with pagination")
            else:
                logger.warning(f"Dataset {dataset_name} loaded with pagination but failed validation")
        else:
            logger.error(f"Failed to load dataset with pagination: {dataset_name}")

        return data

    def _load_dataset_batch(self, dataset_name: str, offset: int, batch_size: int) -> Dict[str, Any]:
        """
        Load a batch of data from a dataset

        Args:
            dataset_name: Name of the dataset
            offset: Offset to start loading from
            batch_size: Number of samples to load

        Returns:
            Dictionary with batch data
        """
        # Different handling for different datasets
        if dataset_name == "writing_prompts":
            return self.processor.load_writing_prompts(max_samples=batch_size, offset=offset)
        elif dataset_name == "persona_chat":
            return self.processor.load_persona_chat(max_samples=batch_size, offset=offset)
        elif dataset_name == "pile":
            return self.processor.load_pile_dataset(max_samples=batch_size, offset=offset)
        elif dataset_name == "openassistant":
            return self.processor.load_openassistant_dataset(max_samples=batch_size, offset=offset)
        elif dataset_name == "gpteacher":
            return self.processor.load_gpteacher_dataset(max_samples=batch_size, offset=offset)
        else:
            logger.error(f"Unsupported dataset for batch loading: {dataset_name}")
            return {}

    def _paginated_dataset_generator(self, dataset_name: str, batch_size: int):
        """
        Generator that yields batches of data from a dataset

        Args:
            dataset_name: Name of the dataset
            batch_size: Number of samples in each batch

        Yields:
            Batches of data
        """
        if dataset_name == "writing_prompts":
            processor_method = self.improved_processor.process_writing_prompts_dataset
        elif dataset_name == "persona_chat":
            processor_method = self.improved_processor.process_persona_chat_dataset
        else:
            logger.error(f"Unsupported dataset for pagination: {dataset_name}")
            return

        # Use a generator to process the dataset in batches
        offset = 0
        while True:
            try:
                # Process a batch of data
                batch_data = processor_method(max_samples=batch_size, offset=offset)

                if not batch_data or not batch_data.get('batches'):
                    # No more data, stop the generator
                    break

                # Yield the batch
                yield batch_data['batches']

                # Update offset for next batch
                offset += batch_size

            except Exception as e:
                logger.error(f"Error processing batch at offset {offset}: {e}")
                break

    def _load_local_dataset(self, dataset_path, split='train', max_samples=None, cache_file=None):
        """
        Load a dataset from a local file or directory

        Args:
            dataset_path: Path to the dataset file or directory
            split: Dataset split (train, validation, test)
            max_samples: Maximum number of samples to load
            cache_file: Optional path to save the processed dataset

        Returns:
            Processed dataset
        """
        from datasets import Dataset

        self.logger.info(f"Loading local dataset from {dataset_path}")

        # Check if we're loading a directory or a file
        if os.path.isdir(dataset_path):
            # It's a directory - gather all text files
            text_files = []
            for ext in ['.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.cpp', '.c', '.java']:
                text_files.extend(glob.glob(os.path.join(dataset_path, f"**/*{ext}"), recursive=True))

            self.logger.info(f"Found {len(text_files)} text files in directory")

            # Limit if max_samples is set
            if max_samples is not None and max_samples < len(text_files):
                text_files = text_files[:max_samples]

            # Read all files
            texts = []
            for file_path in tqdm(text_files, desc="Reading files"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        texts.append(f.read())
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")

        else:
            # It's a single file
            try:
                with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                # Split into chunks for large files
                if max_samples is not None:
                    # Split by paragraphs
                    paragraphs = content.split('\n\n')
                    # Ensure we don't exceed max_samples
                    paragraphs = paragraphs[:max_samples]
                    texts = paragraphs
                else:
                    texts = [content]
            except Exception as e:
                self.logger.error(f"Error reading file {dataset_path}: {e}")
                raise

        # Process the texts into dataset format
        return self._process_text_list(texts, cache_file)

    def _load_standard_dataset(self, dataset_name, split='train', max_samples=None, subset=None, cache_file=None):
        """
        Load a standard dataset from the supported list

        Args:
            dataset_name: Name of the standard dataset
            split: Dataset split (train, validation, test)
            max_samples: Maximum number of samples to load
            subset: Optional subset name
            cache_file: Optional path to save the processed dataset

        Returns:
            Processed dataset
        """
        self.logger.info(f"Loading standard dataset: {dataset_name} (split: {split}, subset: {subset})")

        # Use the existing dataset processor methods
        if hasattr(self, 'processor') and self.processor:
            # Pass the tokenizer to the processor if we have one
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.processor.tokenizer = self.tokenizer

            # Use the dataset_processor to prepare the dataset
            try:
                data = self.processor.prepare_dataset(
                    source=dataset_name,
                    split=split,
                    max_samples=max_samples,
                    batch_size=self.batch_size,
                    subset=subset,
                    cache_dir=os.path.dirname(cache_file) if cache_file else None,
                    output_dir=os.path.dirname(cache_file) if cache_file else None
                )
                return data
            except Exception as e:
                self.logger.error(f"Error loading standard dataset {dataset_name}: {e}")
                raise
        else:
            self.logger.error("Dataset processor not initialized")
            raise ValueError("Dataset processor not initialized")

    def _process_text_list(self, texts, cache_file=None):
        """
        Process a list of texts into a dataset format

        Args:
            texts: List of text strings
            cache_file: Optional path to save the processed dataset

        Returns:
            Processed dataset
        """
        from datasets import Dataset

        self.logger.info(f"Processing {len(texts)} text samples")

        # Tokenize the texts if tokenizer is available
        input_ids = []
        attention_masks = []

        if self.tokenizer:
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids.extend(encodings['input_ids'].tolist())
                attention_masks.extend(encodings['attention_mask'].tolist())

        # Create dataset dictionary
        processed_dataset = {
            "text": texts,
            "input_ids": input_ids,
            "attention_mask": attention_masks
        }

        # Create dataset object
        dataset_dict = {"train": Dataset.from_dict(processed_dataset)}

        # Cache the processed dataset if cache_file is provided
        if cache_file:
            self.logger.info(f"Saving processed dataset to {cache_file}")
            try:
                torch.save(dataset_dict, cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to cache dataset: {e}")

        return dataset_dict

    def _create_batches_from_texts(self, texts, dataset_name):
        """
        Create batches from a list of texts using tokenizer

        Args:
            texts: List of text strings
            dataset_name: Name of the dataset for metadata

        Returns:
            Dictionary with batches and metadata
        """
        if not texts:
            return {"batches": [], "metadata": {"sample_count": 0, "source": dataset_name}}

        try:
            batch_size = getattr(self, 'batch_size', 4)
            batches = []

            # Process in smaller batches to avoid memory issues
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:min(i+batch_size, len(texts))]

                try:
                    # Tokenize text for input
                    encodings = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )

                    # Get input and target tensors (for language modeling, target is input shifted)
                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']

                    # For causal language modeling, target is the input shifted right
                    # Create targets by shifting inputs
                    target_ids = input_ids.clone()

                    # Add batch to batches list
                    batches.append((input_ids, target_ids))

                except Exception as e:
                    self.logger.warning(f"Error creating batch: {e}")
                    continue

            return {
                "batches": batches,
                "metadata": {
                    "sample_count": len(texts),
                    "batch_count": len(batches),
                    "source": dataset_name
                }
            }
        except Exception as e:
            self.logger.error(f"Error creating batches from texts: {e}")
            return {"batches": [], "metadata": {"sample_count": 0, "source": dataset_name}}

    def process_huggingface_dataset(self, dataset, dataset_name, return_batches=True):
        """
        Process HuggingFace dataset into model-compatible format.

        Args:
            dataset: HuggingFace dataset object or dictionary
            dataset_name: Name of the dataset for logging and metadata
            return_batches: Whether to return batched data or just texts

        Returns:
            Dictionary containing either batches or texts, with metadata
        """
        # Handle empty dataset
        if not dataset:
            self.logger.warning(f"Empty dataset provided for {dataset_name}")
            return self._create_empty_result(dataset_name)

        self.logger.info(f"Processing HuggingFace dataset ({len(dataset)} samples)")

        try:
            # Handle empty dataset after processing
            if len(dataset) == 0:
                self.logger.warning("Dataset is empty after processing!")
                return self._create_empty_result(dataset_name)

            # Extract training split if dataset is a dictionary
            dataset = self._extract_training_split(dataset)

            # Log dataset structure for debugging
            self._log_dataset_structure(dataset)

            # Extract texts based on dataset format
            texts = self._extract_texts_from_dataset(dataset, dataset_name)

            # Log extraction results
            self.logger.info(f"Extracted {len(texts)} texts out of {len(dataset)} items")
            if texts:
                self.logger.info(f"Sample text: {texts[0][:200]}...")

        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_empty_result(dataset_name)

        # Filter out empty or very short texts
        texts = [text for text in texts if text and len(text) > 10]  # Min 10 chars

        if not texts:
            self.logger.warning(f"No valid texts extracted from dataset {dataset_name}")
            return self._create_empty_result(dataset_name)

        self.logger.info(f"Final extraction: {len(texts)} valid texts from dataset {dataset_name}")

        # Return results in requested format
        return self._create_result(texts, dataset_name, return_batches)

    def _create_empty_result(self, dataset_name):
        """Create a result dictionary for empty datasets."""
        return {"batches": [], "metadata": {"sample_count": 0, "source": dataset_name}}

    def _extract_training_split(self, dataset):
        """Extract training split if dataset is a dictionary."""
        if isinstance(dataset, dict) and 'train' in dataset:
            return dataset['train']
        return dataset

    def _log_dataset_structure(self, dataset):
        """Log the structure of the dataset for debugging purposes."""
        if len(dataset) > 0:
            first_item = dataset[0]
            keys = list(first_item.keys())
            self.logger.info(f"Dataset keys: {keys}")
            self.logger.info(f"Sample item (first 5 fields): {dict(list(first_item.items())[:5])}")

    def _extract_texts_from_dataset(self, dataset, dataset_name):
        """
        Extract texts from dataset based on its format and name.

        This is a wrapper around the batch extraction methods to handle full datasets.

        Args:
            dataset: The dataset to extract texts from
            dataset_name: Name of the dataset to determine extraction method

        Returns:
            List of extracted texts
        """
        try:
            # Log dataset structure for debugging
            if len(dataset) > 0:
                sample_item = dataset[0]
                self.logger.info(f"Dataset sample item: {sample_item}")

            # Use the same extraction logic as for batches
            return self._extract_texts_from_batch(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"Error extracting texts from dataset {dataset_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def _create_result(self, texts, dataset_name, return_batches):
        """
        Create the final result dictionary based on the extracted texts.

        Args:
            texts: List of extracted texts
            dataset_name: Name of the dataset for metadata
            return_batches: Whether to return batched data or just texts

        Returns:
            Dictionary with either batches or texts and metadata
        """
        if not return_batches:
            # Return texts directly
            return {
                "texts": texts,
                "metadata": {
                    "sample_count": len(texts),
                    "source": dataset_name
                }
            }

        # Determine batching method based on tokenizer availability
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return self._create_simple_batches(texts, dataset_name)
        else:
            return self._create_batches_from_texts(texts, dataset_name)

    def _create_simple_batches(self, texts, dataset_name):
        """Create simple batches without tokenization when no tokenizer is available."""
        self.logger.warning("No tokenizer available, using simple batching")

        batch_size = getattr(self, 'batch_size', 4)
        batches = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # Create dummy tensors
            batch_inputs = torch.zeros((len(batch_texts), 10), dtype=torch.long)
            batch_targets = torch.zeros((len(batch_texts), 10), dtype=torch.long)
            batches.append((batch_inputs, batch_targets))

        return {
            "batches": batches,
            "metadata": {
                "sample_count": len(texts),
                "batch_count": len(batches),
                "source": dataset_name
            }
        }


def main():
    """Main function to demonstrate the UnifiedDatasetHandler functionality."""
    # Initialize the handler
    handler = UnifiedDatasetHandler()

    # List all available datasets
    print("Available datasets:")
    for info in handler.list_all_datasets():
        print(f"- {info['name']}: {info['description']}")
        print(f"  Format: {info['format']}")
        print(f"  Usage: {info['typical_usage']}")
        print(f"  Available: {info['available']}")
        print()

    # Demonstrate dataset selection based on prompt
    prompt = "Write a story about a robot who discovers emotions"
    best_dataset = handler.determine_best_dataset(prompt)
    print(f"Best dataset for prompt '{prompt}': {best_dataset}")

    # Load and process a sample dataset
    try:
        dataset = handler.load_dataset("writing_prompts", max_samples=5)
        if dataset and 'batches' in dataset:
            print(f"Successfully loaded writing_prompts dataset with {len(dataset['batches'])} batches")

            # Extract and display sample prompt-response pairs
            if pairs := handler.extract_prompt_response_pairs(
                "writing_prompts", dataset, max_pairs=2
            ):
                print("\nSample prompt-response pairs:")
                for i, pair in enumerate(pairs):
                    print(f"\nPair {i+1}:")
                    print(f"Prompt: {pair['prompt'][:100]}...")
                    print(f"Response: {pair['response'][:100]}...")
    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()
