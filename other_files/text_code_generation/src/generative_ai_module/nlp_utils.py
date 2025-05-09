"""
Natural Language Processing Utilities

This module provides a unified interface for NLP functionality across the Jarvis AI Assistant.
It handles compatibility issues with spaCy, especially in environments like Paperspace,
and provides fallbacks when necessary.

Key features:
1. Safe initialization of spaCy with fallbacks
2. Minimal tokenizer for Paperspace compatibility
3. Text processing utilities
"""

import os
import sys
import logging
import re
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

# Import minimal tokenizer if available
try:
    from .minimal_spacy_tokenizer import tokenize as minimal_tokenize
    from .minimal_spacy_tokenizer import tokenizer as minimal_tokenizer
    MINIMAL_TOKENIZER_AVAILABLE = True
except ImportError:
    MINIMAL_TOKENIZER_AVAILABLE = False

# Check for Paperspace environment
def is_paperspace_environment() -> bool:
    """Detect if we're running in Paperspace Gradient environment"""
    return os.environ.get('PAPERSPACE') == 'true' or os.environ.get('PAPERSPACE_ENVIRONMENT') == 'true'

def is_spacy_available() -> Tuple[bool, Optional[str]]:
    """
    Check if spaCy is available in the current environment
    
    Returns:
        Tuple of (is_available, version/reason)
    """
    try:
        # Check if minimal tokenizer is available first
        if MINIMAL_TOKENIZER_AVAILABLE:
            return True, "minimal-tokenizer"
        
        # Try importing regular spaCy
        import spacy
        return True, spacy.__version__
    except ImportError:
        return False, None
    except Exception as e:
        # Handle ParametricAttention_v2 error or other issues
        if "ParametricAttention_v2" in str(e):
            # Try to use minimal tokenizer instead
            if MINIMAL_TOKENIZER_AVAILABLE:
                return True, "minimal-tokenizer"
        return False, str(e)

def is_spacy_model_loaded(model_name: str = "en_core_web_sm") -> Tuple[bool, str]:
    """
    Check if a specific spaCy model is available and can be loaded
    
    Args:
        model_name: Name of the spaCy model to check
        
    Returns:
        Tuple of (is_loaded, message)
    """
    # First check if minimal tokenizer is available
    if MINIMAL_TOKENIZER_AVAILABLE:
        if minimal_tokenizer.is_available:
            return True, f"Minimal tokenizer loaded successfully (Paperspace-safe)"
    
    # Try regular spaCy model
    try:
        import spacy
        # Only try this on non-Paperspace environments to avoid segfaults
        if not is_paperspace_environment():
            nlp = spacy.load(model_name)
            # Test the model with a simple sentence
            doc = nlp("This is a test sentence.")
            return True, f"Model {model_name} loaded successfully"
        else:
            # On Paperspace, we don't want to load the full pipeline
            try:
                # Just try a very minimal test to see if the model can be found
                nlp = spacy.load(model_name, disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
                # Only test tokenization which is safe
                tokens = [t.text for t in nlp.tokenizer("Test sentence")]
                return True, f"Model {model_name} tokenizer available (Paperspace-safe mode)"
            except Exception as e:
                return False, f"Error loading model in Paperspace-safe mode: {str(e)}"
    except ImportError:
        return False, "spaCy not installed"
    except OSError:
        return False, f"Model {model_name} not found"
    except Exception as e:
        if "ParametricAttention_v2" in str(e):
            return False, "Paperspace compatibility issue: ParametricAttention_v2 error"
        return False, f"Error loading model: {str(e)}"

def initialize_spacy(fallback_to_basic: bool = True, 
                    log_errors: bool = True, 
                    paperspace_safe: Optional[bool] = None) -> Any:
    """
    Initialize spaCy with the en_core_web_sm model if available.
    
    Args:
        fallback_to_basic: If True, will not raise errors but return None if spaCy is unavailable
        log_errors: If True, will log errors and warnings
        paperspace_safe: Override for Paperspace detection - if None, will auto-detect
        
    Returns:
        nlp: The spaCy NLP object if available, None otherwise
    """
    # Auto-detect Paperspace if not specified
    if paperspace_safe is None:
        paperspace_safe = is_paperspace_environment()
    
    # Try using minimal tokenizer first if in Paperspace
    if paperspace_safe and MINIMAL_TOKENIZER_AVAILABLE:
        if minimal_tokenizer.is_available:
            if log_errors:
                logger.info("Using minimal spaCy tokenizer (Paperspace-safe)")
            
            # Return a simplified object with just tokenizer functionality
            class MinimalNLP:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                
                def __call__(self, text):
                    tokens = self.tokenizer.tokenize(text)
                    # Return a dummy doc object with just the tokens
                    class DummyDoc:
                        def __init__(self, tokens):
                            self.tokens = tokens
                            self.ents = []
                        def __iter__(self):
                            for t in self.tokens:
                                yield type('DummyToken', (), {'text': t, 'pos_': "UNKNOWN", 'dep_': "UNKNOWN", 'head': type('DummyHead', (), {'text': ""})})
                        def __getitem__(self, i):
                            if isinstance(i, slice):
                                return [type('DummyToken', (), {'text': t}) for t in self.tokens[i]]
                            return type('DummyToken', (), {'text': self.tokens[i]})
                    return DummyDoc(tokens)
            
            return MinimalNLP(minimal_tokenizer)
    
    # For non-Paperspace or if minimal tokenizer failed, try regular spaCy
    try:
        import spacy
        try:
            # In Paperspace, load with minimal components to avoid segfaults
            if paperspace_safe:
                if log_errors:
                    logger.info("Loading spaCy in Paperspace-safe mode")
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
            else:
                nlp = spacy.load("en_core_web_sm")
            
            if log_errors:
                logger.info("spaCy initialized with en_core_web_sm model")
            return nlp
        except OSError as e:
            if log_errors:
                logger.warning(f"spaCy model not found: {str(e)}")
                logger.warning("To install the model, run: python -m spacy download en_core_web_sm")
            if not fallback_to_basic:
                raise e
            return None
        except Exception as e:
            if "ParametricAttention_v2" in str(e):
                if log_errors:
                    logger.error(f"Paperspace compatibility issue with spaCy: {str(e)}")
                try:
                    # Final attempt - create blank model which should work
                    if log_errors:
                        logger.info("Trying to create blank model as last resort")
                    nlp = spacy.blank("en")
                    return nlp
                except Exception:
                    pass
            if log_errors:
                logger.error(f"Error initializing spaCy: {str(e)}")
            if not fallback_to_basic:
                raise e
            return None
    except ImportError:
        if log_errors:
            logger.warning("spaCy not installed")
            logger.warning("For better text processing, install spaCy with: pip install spacy")
        if not fallback_to_basic:
            raise ImportError("spaCy not installed")
        return None

def process_text_with_spacy_or_fallback(text: str, nlp: Any = None) -> Dict[str, Any]:
    """
    Process text with spaCy if available, or use a simple fallback method.
    
    Args:
        text: The text to process
        nlp: Optional spaCy NLP object. If None, will try to initialize
        
    Returns:
        dict: Processed text information (entities, tokens, etc.)
    """
    # Check if we're in Paperspace
    paperspace_mode = is_paperspace_environment()
    
    # Check for minimal tokenizer first
    if MINIMAL_TOKENIZER_AVAILABLE and paperspace_mode:
        tokens = minimal_tokenize(text)
        return {
            "entities": [],  # Empty since we can't detect entities in minimal mode
            "tokens": tokens,
            "pos_tags": [],  # Empty in minimal mode
            "nouns": [t for t in tokens if t[0].isupper() and len(t) > 3],  # Simple heuristic
            "verbs": [],  # Empty in minimal mode
            "adjectives": [],  # Empty in minimal mode
            "dependency_tree": [],  # Empty in minimal mode
            "sentences": [s.strip() for s in text.split('.') if s.strip()],  # Simple period splitting
            "minimal_mode": True
        }
    
    # If nlp is not provided, try to initialize spaCy
    if nlp is None:
        nlp = initialize_spacy(fallback_to_basic=True, log_errors=False, paperspace_safe=paperspace_mode)
    
    # If spaCy is available, use it for processing
    if nlp is not None:
        try:
            doc = nlp(text)
            
            # In Paperspace/minimal mode, only return tokens to avoid segfaults
            if paperspace_mode or hasattr(nlp, 'tokenizer') and not hasattr(doc, 'ents'):
                tokens = [token.text for token in doc]
                return {
                    "entities": [],  # Empty since we can't detect entities in minimal mode
                    "tokens": tokens,
                    "pos_tags": [],  # Empty in minimal mode
                    "nouns": [t for t in tokens if t[0].isupper() and len(t) > 3],  # Simple heuristic
                    "verbs": [],  # Empty in minimal mode
                    "adjectives": [],  # Empty in minimal mode
                    "dependency_tree": [],  # Empty in minimal mode
                    "sentences": [s.strip() for s in text.split('.') if s.strip()],  # Simple period splitting
                    "minimal_mode": True
                }
            
            # Full mode for non-Paperspace environments
            return {
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "tokens": [token.text for token in doc],
                "pos_tags": [(token.text, token.pos_) for token in doc],
                "nouns": [token.text for token in doc if token.pos_ == "NOUN"],
                "verbs": [token.text for token in doc if token.pos_ == "VERB"],
                "adjectives": [token.text for token in doc if token.pos_ == "ADJ"],
                "dependency_tree": [(token.text, token.dep_, token.head.text) for token in doc],
                "sentences": [sent.text for sent in doc.sents],
                "minimal_mode": False
            }
        except Exception as e:
            logger.warning(f"Error using spaCy to process text: {e}. Falling back to basic processing.")
            # Fall back to basic processing if spaCy fails
            pass
    
    # Basic fallback processing
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic POS guesses - very rudimentary!
    nouns = [word for word in words if word[0].isupper() and len(word) > 3]
    potential_verbs = [word for word in words if word.lower() in {
        'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
        'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
        'make', 'create', 'generate', 'write', 'build', 'design', 'develop'
    }]
    
    return {
        "entities": [],  # Empty since we can't detect entities
        "tokens": words,
        "pos_tags": [],  # Empty since we can't determine POS
        "nouns": nouns,
        "verbs": potential_verbs,
        "adjectives": [],  # Empty since we can't determine adjectives
        "dependency_tree": [],  # Empty since we can't determine dependencies
        "sentences": sentences,
        "minimal_mode": True  # Using basic mode
    }

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization function that works in any environment
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # Try minimal tokenizer first
    if MINIMAL_TOKENIZER_AVAILABLE:
        try:
            return minimal_tokenize(text)
        except Exception:
            pass
    
    # Try spaCy
    try:
        nlp = initialize_spacy(fallback_to_basic=True, log_errors=False)
        if nlp:
            doc = nlp(text)
            return [token.text for token in doc]
    except Exception:
        pass
    
    # Basic fallback
    # Split on whitespace and punctuation
    tokens = []
    current_token = ""
    
    for char in text:
        if char.isalnum() or char == "'" or char == "-":
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if not char.isspace():
                tokens.append(char)
    
    if current_token:
        tokens.append(current_token)
    
    return tokens

# Minimal tokenizer implementation in case the imported one is not available
class MinimalFallbackTokenizer:
    """Minimal tokenizer implementation for environments where spaCy is not available"""
    
    def __init__(self):
        self.is_available = True
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using simple regex rules"""
        # Split on whitespace and punctuation
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char == "'" or char == "-":
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if not char.isspace():
                    tokens.append(char)
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

# Create a global minimal tokenizer if the imported one is not available
if not MINIMAL_TOKENIZER_AVAILABLE:
    minimal_tokenizer = MinimalFallbackTokenizer()
    
    def minimal_tokenize(text: str) -> List[str]:
        """Tokenize text using the minimal fallback tokenizer"""
        return minimal_tokenizer.tokenize(text)

# Function to verify spaCy is working correctly
def verify_spacy() -> Tuple[bool, str]:
    """
    Verify spaCy is installed and working correctly.
    
    Returns:
        Tuple of (is_working, message)
    """
    spacy_available, version = is_spacy_available()
    if not spacy_available:
        return False, "spaCy is not installed"
    
    spacy_model_loaded, model_message = is_spacy_model_loaded()
    if not spacy_model_loaded:
        return False, f"spaCy model 'en_core_web_sm' is not loaded: {model_message}"
    
    try:
        # Try a simple test
        nlp = initialize_spacy()
        if not nlp:
            if MINIMAL_TOKENIZER_AVAILABLE:
                return True, "Using minimal tokenizer in Paperspace-safe mode"
            return False, "Failed to initialize spaCy"
            
        doc = nlp("This is a test sentence for spaCy.")
        tokens = [token.text for token in doc]
        return True, f"spaCy is working correctly. Tokenized: {tokens[:3]}..."
    except Exception as e:
        return False, f"Error using spaCy: {str(e)}"

# Install spaCy in a Paperspace-safe way
def install_spacy_with_model(model_name: str = "en_core_web_sm") -> Tuple[bool, str]:
    """
    Install spaCy and a model in a Paperspace-safe way
    
    Args:
        model_name: Name of the spaCy model to install
        
    Returns:
        Tuple of (success, message)
    """
    import subprocess
    
    # Check if we're in Paperspace
    paperspace_mode = is_paperspace_environment()
    
    try:
        # Install spaCy
        logger.info("Installing spaCy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        
        # Install the model
        logger.info(f"Installing spaCy model {model_name}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        
        # Verify installation
        spacy_available, _ = is_spacy_available()
        if not spacy_available:
            return False, "Failed to install spaCy"
        
        model_loaded, model_message = is_spacy_model_loaded(model_name)
        if not model_loaded:
            return False, f"Failed to load model: {model_message}"
        
        if paperspace_mode:
            logger.info("Running in Paperspace environment, installing minimal tokenizer for compatibility")
            
            # Create minimal tokenizer
            create_minimal_tokenizer()
            
        return True, f"Successfully installed spaCy and model {model_name}"
    except Exception as e:
        return False, f"Installation failed: {str(e)}"

def create_minimal_tokenizer():
    """Create a minimal_spacy_tokenizer.py file for Paperspace compatibility"""
    import inspect
    
    # Get the code for the MinimalFallbackTokenizer class
    tokenizer_code = inspect.getsource(MinimalFallbackTokenizer)
    
    # Add the minimal_tokenize function
    tokenize_code = inspect.getsource(minimal_tokenize) if 'minimal_tokenize' in globals() else """
def minimal_tokenize(text):
    \"\"\"Tokenize text using the minimal tokenizer\"\"\"
    return tokenizer.tokenize(text)
"""
    
    # Create the file content
    file_content = f'''"""
Minimal spaCy Tokenizer

This module provides a minimal tokenizer that can be used as a fallback
when spaCy is not available or encounters compatibility issues in environments
like Paperspace Gradient.
"""

import re
from typing import List

{tokenizer_code}

# Create a global tokenizer instance
tokenizer = MinimalFallbackTokenizer()

{tokenize_code}

# Verify that the tokenizer is working
if __name__ == "__main__":
    test_text = "This is a test sentence. It has punctuation, numbers (123), and symbols @#$!"
    tokens = minimal_tokenize(test_text)
    print(f"Tokenized result: {{tokens}}")
    print("Minimal tokenizer is working correctly!")
'''
    
    # Determine the module directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Write the file
    file_path = os.path.join(module_dir, "minimal_spacy_tokenizer.py")
    with open(file_path, 'w') as f:
        f.write(file_content)
    
    logger.info(f"Created minimal tokenizer at {file_path}")
    return file_path 