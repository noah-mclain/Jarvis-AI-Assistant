import io
import os
import logging
from PIL import Image
from typing import Union
import zipfile
import re
import sys
import importlib.util
from pathlib import Path

from .utils import is_zipfile

# Import our minimal tokenizer - will always work even if spaCy fails
try:
    from .minimal_spacy_tokenizer import tokenize as minimal_tokenize
    MINIMAL_TOKENIZER_AVAILABLE = True
except ImportError:
    MINIMAL_TOKENIZER_AVAILABLE = False
    logging.warning("Minimal tokenizer not available, will use basic fallback")

# Try to import spaCy with better error handling and Paperspace compatibility
SPACY_AVAILABLE = False
SPACY_MODEL_LOADED = False
TOKENIZER_ONLY_MODE = False
NLP = None

# Check if we're in Paperspace
IS_PAPERSPACE = (
    os.environ.get('PAPERSPACE', '').lower() == 'true' or 
    'gradient' in os.environ.get('HOSTNAME', '').lower() or
    os.path.exists('/paperspace')
)

def fix_paperspace_spacy_imports():
    """Fix spaCy imports for Paperspace environments by creating dummy imports"""
    try:
        # Create dummy modules to prevent problematic imports
        class DummyModule:
            """Dummy module to replace problematic imports"""
            def __init__(self, name):
                self.__name__ = name
                # Add ParametricAttention_v2 directly to the __dict__ to ensure it's available
                self.__dict__["ParametricAttention_v2"] = type("ParametricAttention_v2", (), {})
            
            def __getattr__(self, attr):
                # Return a dummy object for any attribute
                return type(attr, (), {})()
        
        # We'll manipulate sys.modules to prevent problematic imports
        for module_name in ['thinc.api', 'thinc.layers', 'thinc.model', 'thinc.config']:
            if module_name in sys.modules:
                del sys.modules[module_name]
            sys.modules[module_name] = DummyModule(module_name)
        
        logging.info("Paperspace compatibility fixed applied to spaCy imports")
        return True
    except Exception as e:
        logging.error(f"Error fixing Paperspace spaCy imports: {e}")
        return False

# If in Paperspace, use minimal tokenizer approach instead of full spaCy
if IS_PAPERSPACE and MINIMAL_TOKENIZER_AVAILABLE:
    logging.info("Paperspace environment detected, using minimal tokenizer")
    TOKENIZER_ONLY_MODE = True
else:
    # Try different approaches to load spaCy
    try:
        # First attempt normal import
        import spacy
        SPACY_AVAILABLE = True
        # Check if the model is also available
        try:
            # Try to load the model
            NLP = spacy.load("en_core_web_sm")
            SPACY_MODEL_LOADED = True
        except OSError:
            # Model not found, provide instructions
            logging.warning("spaCy model 'en_core_web_sm' not found. For optimal text processing, install it with:")
            logging.warning("python -m spacy download en_core_web_sm")
            logging.warning("Or run setup/setup_spacy.py to install spaCy and the model correctly.")
            logging.warning("Using basic text processing fallback for now.")
        except Exception as e:
            if "ParametricAttention_v2" in str(e) or IS_PAPERSPACE:
                logging.warning("Detected Paperspace environment with compatibility issues")
                # Try to fix the imports for Paperspace
                if fix_paperspace_spacy_imports():
                    try:
                        # Try loading again with minimal features
                        NLP = spacy.load("en_core_web_sm")
                        SPACY_MODEL_LOADED = True
                        TOKENIZER_ONLY_MODE = True
                        logging.warning("spaCy loaded in tokenizer-only mode for Paperspace compatibility")
                    except Exception as inner_e:
                        logging.warning(f"Failed to load spaCy even with Paperspace fixes: {inner_e}")
                else:
                    logging.warning("Failed to apply Paperspace-specific fixes")
            else:
                logging.warning(f"Error loading spaCy model: {str(e)}. Using basic text processing fallback.")
    except ImportError:
        logging.warning("spaCy not available, using basic text processing fallback")
        logging.warning("For optimal text processing, install spaCy by running setup/setup_spacy.py")

class PromptEnhancer:
    def __init__(self):
        self.logger = logging.getLogger("PromptEnhancer")
        
        # Set up the NLP pipeline based on what's available
        self.nlp = NLP  # This will be None if spaCy or model is not available
        self.tokenizer_only_mode = TOKENIZER_ONLY_MODE
        self.minimal_tokenizer_available = MINIMAL_TOKENIZER_AVAILABLE
        self.is_paperspace = IS_PAPERSPACE
        
        if self.is_paperspace and self.minimal_tokenizer_available:
            self.logger.info("Using minimal spaCy tokenizer for Paperspace compatibility")
        elif self.nlp:
            if self.tokenizer_only_mode:
                self.logger.info("Using spaCy in tokenizer-only mode for Paperspace compatibility")
            else:
                self.logger.info("Using spaCy for enhanced NLP processing")
        else:
            self.logger.info("Using basic text processing (no spaCy)")
        
    def enhance_prompt(self, input_data: Union[str, bytes, os.PathLike]) -> str:
        """
        Enhance prompts for generative AI based on input type (text, image, zip, file)
        
        Args:
            input_data: Can be text string, image bytes, file path, or zip file path
            
        Returns:
            Enhanced prompt with structured context and metadata
        """
        try:
            if isinstance(input_data, str):
                if os.path.isfile(input_data):
                    return self._enhance_file(input_data)
                return self._enhance_text(input_data)
                
            elif isinstance(input_data, bytes):
                return self._enhance_image(input_data)
                
            elif is_zipfile(input_data):
                return self._enhance_zip(input_data)
                
            else:
                raise ValueError("Unsupported input type")
                
        except Exception as e:
            self.logger.error(f"Error enhancing prompt: {str(e)}")
            return f"Original input could not be enhanced: {input_data}"

    def _enhance_text(self, text: str) -> str:
        """Enhance text prompts using NLP techniques or fallback method"""
        # For Paperspace, try minimal tokenizer first (most reliable)
        if self.is_paperspace and self.minimal_tokenizer_available:
            return self._enhance_text_minimal_tokenizer(text)
        
        # For regular environments, use full spaCy if available
        if self.nlp:
            try:
                if self.tokenizer_only_mode:
                    # Paperspace-safe mode: only use tokenizer
                    return self._enhance_text_tokenizer_only(text)
                
                # Full spaCy mode
                doc = self.nlp(text)
                
                # Extract entities and syntactic features
                entities = [ent.text for ent in doc.ents]
                verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
                nouns = [token.text for token in doc if token.pos_ == "NOUN"]
                adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

                # Build a more comprehensive enhanced prompt
                return (
                    "Generate high-quality output based on these requirements:\n"
                    f"Main subjects: {', '.join(entities) if entities else ', '.join(nouns[:3]) if nouns else 'general request'}\n"
                    f"Actions needed: {', '.join(verbs) if verbs else 'create content'}\n"
                    f"Attributes: {', '.join(adjectives[:5]) if adjectives else 'professional'}\n"
                    f"Detailed description: {text}\n"
                    "Include appropriate technical specifications and artistic style based on the context."
                )
            except Exception as e:
                self.logger.warning(f"spaCy processing failed, using fallback: {str(e)}")
                # Fall through to fallback method
        
        # Fallback method using basic text processing (no spaCy)
        return self._enhance_text_fallback(text)
    
    def _enhance_text_minimal_tokenizer(self, text: str) -> str:
        """Enhanced text processing using our minimal tokenizer module (most reliable in Paperspace)"""
        try:
            # Use minimal_tokenize imported from minimal_spacy_tokenizer.py
            tokens = minimal_tokenize(text)
            
            # Get word frequencies
            word_freq = {}
            for token in tokens:
                if len(token) > 3 and token.isalpha():
                    token = token.lower()
                    word_freq[token] = word_freq.get(token, 0) + 1
            
            # Extract top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:7]
            keywords = [word for word, _ in keywords]
            
            # Extract potential subjects (capitalized words)
            potential_subjects = [token for token in tokens if token and token[0].isupper() and len(token) > 3]
            
            return (
                "Generate high-quality output based on these requirements:\n"
                f"Main focus: {', '.join(potential_subjects[:3]) if potential_subjects else 'general request'}\n"
                f"Keywords: {', '.join(keywords) if keywords else 'general content'}\n"
                f"Word count: {len(tokens)} words\n"
                f"Detailed description: {text}\n"
                "Include appropriate specifications based on the context."
            )
        except Exception as e:
            self.logger.warning(f"Minimal tokenizer processing failed: {str(e)}")
            return self._enhance_text_fallback(text)
    
    def _enhance_text_tokenizer_only(self, text: str) -> str:
        """Enhanced text processing using only the tokenizer (Paperspace-safe)"""
        try:
            # Use only the tokenizer component which is safe in Paperspace
            tokens = [t.text for t in self.nlp.tokenizer(text)]
            
            # Get word frequencies
            word_freq = {}
            for token in tokens:
                if len(token) > 3 and token.isalpha():
                    token = token.lower()
                    word_freq[token] = word_freq.get(token, 0) + 1
            
            # Extract top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:7]
            keywords = [word for word, _ in keywords]
            
            # Extract potential subjects (capitalized words)
            potential_subjects = [token for token in tokens if token and token[0].isupper() and len(token) > 3]
            
            return (
                "Generate high-quality output based on these requirements:\n"
                f"Main focus: {', '.join(potential_subjects[:3]) if potential_subjects else 'general request'}\n"
                f"Keywords: {', '.join(keywords) if keywords else 'general content'}\n"
                f"Word count: {len(tokens)} words\n"
                f"Detailed description: {text}\n"
                "Include appropriate specifications based on the context."
            )
        except Exception as e:
            self.logger.warning(f"Tokenizer-only processing failed: {str(e)}")
            return self._enhance_text_fallback(text)
    
    def _enhance_text_fallback(self, text: str) -> str:
        """Basic text processing fallback when spaCy is not available"""
        words = text.split()
        # Extract potential subjects (nouns) using capitalization as a heuristic
        potential_subjects = [word.strip('.,?!()[]{}\'"`') for word in words if word and word[0].isupper() and len(word) > 3]
        
        # Extract potential verbs using simple regex
        action_verbs = set([
            'create', 'make', 'generate', 'write', 'build', 'design', 'develop', 
            'implement', 'analyze', 'explain', 'describe', 'summarize', 'produce',
            'compose', 'construct', 'prepare', 'provide', 'craft', 'form', 'arrange'
        ])
        potential_verbs = [word for word in words if word.lower() in action_verbs]
        
        # Extract keywords by frequency
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,?!()[]{}\'"`')
            if len(word) > 3 and word not in action_verbs:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = [word for word, _ in top_keywords]
        
        return (
            "Generate high-quality output based on these requirements:\n"
            f"Main focus: {', '.join(potential_subjects[:3]) if potential_subjects else 'general request'}\n"
            f"Actions: {', '.join(potential_verbs) if potential_verbs else 'create content'}\n"
            f"Keywords: {', '.join(top_keywords) if top_keywords else 'general content'}\n"
            f"Detailed description: {text}\n"
            "Include appropriate specifications based on the context."
        )

    def _enhance_image(self, image_data: bytes) -> str:
        """Enhance prompts from image inputs"""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Extract basic metadata
            metadata = {
                "format": img.format,
                "size": img.size,
                "mode": img.mode,
            }
            
            # Create structured prompt
            return (
                "Generate content matching this visual input:\n"
                f"Image characteristics: {metadata}\n"
                "Maintain visual consistency with the provided reference image "
                "in terms of color palette, composition, and style."
            )
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            return "Enhanced image prompt: [Could not process image]"

    def _enhance_zip(self, zip_path: str) -> str:
        """Process zip files containing multiple inputs"""
        enhanced_prompts = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with zip_ref.open(file) as img_file:
                            enhanced_prompts.append(self._enhance_image(img_file.read()))
                    else:
                        with zip_ref.open(file) as text_file:
                            content = text_file.read().decode('utf-8')
                            enhanced_prompts.append(self._enhance_text(content))
                            
            return "Combined requirements from archive:\n" + "\n".join(enhanced_prompts)
            
        except Exception as e:
            self.logger.error(f"Zip processing error: {str(e)}")
            return "Enhanced archive prompt: [Could not process zip contents]"

    def _enhance_file(self, file_path: str) -> str:
        """Handle various file types"""
        if file_path.lower().endswith(('.txt', '.md')):
            with open(file_path, 'r') as f:
                return self._enhance_text(f.read())
                
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(file_path, 'rb') as f:
                return self._enhance_image(f.read())
                
        else:
            return (
                "Enhanced file prompt: "
                f"Process contents of {os.path.basename(file_path)} "
                "with appropriate technical specifications based on file type."
            )

def analyze_prompt(prompt: str) -> str:
    """
    Analyze the prompt to determine which model is most appropriate.
    Returns the most suitable dataset name based on content analysis.
    """
    # Convert to lowercase for analysis
    prompt_lower = prompt.lower()

    # Keywords that suggest dialogue/conversation (persona_chat)
    dialogue_keywords = [
        'hello', 'hi', 'hey', 'how are you', 'what do you', 'can you',
        'would you', 'tell me', 'explain', 'help me', 'i need', 'i want',
        'question', 'ask', 'answer', 'chat', 'talk', 'conversation',
        'discuss', 'advice', 'suggest', 'recommend', 'opinion', 'think about',
        'personal', 'feeling', 'experience', 'dialogue', 'interview', 'response',
        'assistant', 'help'
    ]

    # Keywords that suggest story generation (writing_prompts)
    story_keywords = [
        'story', 'write', 'create', 'imagine', 'world where', 'what if',
        'once upon', 'beginning', 'end', 'plot', 'character', 'setting',
        'scene', 'describe', 'narrative', 'tale', 'fable', 'myth',
        'adventure', 'journey', 'quest', 'tale', 'legend', 'fiction',
        'fantasy', 'sci-fi', 'creative', 'novel', 'chapter', 'book',
        'story about', 'screenplay', 'drama', 'comedy'
    ]
    
    # Keywords that suggest factual knowledge (pile)
    knowledge_keywords = [
        'what is', 'how does', 'explain', 'definition', 'define',
        'research', 'science', 'history', 'fact', 'information',
        'data', 'study', 'analysis', 'report', 'paper', 'article',
        'statistics', 'evidence', 'prove', 'demonstration', 'example',
        'reference', 'citation', 'source', 'documented'
    ]
    
    # Keywords that suggest instruction following (gpteacher)
    instruction_keywords = [
        'how to', 'steps to', 'guide', 'tutorial', 'walkthrough',
        'instructions', 'procedure', 'method', 'technique', 'approach',
        'implement', 'build', 'create', 'design', 'develop', 'setup',
        'configure', 'install', 'debugging', 'solve', 'fix', 'repair'
    ]
    
    # Keywords that suggest assistant-like responses (openassistant)
    assistant_keywords = [
        'assistant', 'help me with', 'support', 'solve', 'give me',
        'provide', 'generate', 'analyze', 'summarize', 'compare',
        'evaluate', 'optimize', 'improve', 'enhance', 'correct'
    ]

    # Count matches for each type
    dialogue_matches = sum(keyword in prompt_lower for keyword in dialogue_keywords)
    story_matches = sum(keyword in prompt_lower for keyword in story_keywords)
    knowledge_matches = sum(keyword in prompt_lower for keyword in knowledge_keywords)
    instruction_matches = sum(keyword in prompt_lower for keyword in instruction_keywords)
    assistant_matches = sum(keyword in prompt_lower for keyword in assistant_keywords)

    # Check for question marks (more likely to be dialogue or knowledge)
    question_mark_count = prompt_lower.count('?')

    # Check for specific formats
    story_format = any([
        prompt_lower.startswith(('write', 'create', 'imagine')),
        'story about' in prompt_lower,
        'tale of' in prompt_lower,
        'write a' in prompt_lower and any(word in prompt_lower for word in ['story', 'novel', 'narrative'])
    ])
    
    knowledge_format = any([
        prompt_lower.startswith(('what is', 'how does', 'why is', 'when did')),
        'definition of' in prompt_lower,
        'meaning of' in prompt_lower
    ])
    
    instruction_format = any([
        prompt_lower.startswith(('how to', 'steps to', 'guide to')),
        'instructions for' in prompt_lower,
        'show me how to' in prompt_lower
    ])

    # Decision logic with weighted scoring
    scores = {
        'persona_chat': dialogue_matches * 1.2 + (question_mark_count * 0.8),
        'writing_prompts': story_matches * 1.5 + (1.5 if story_format else 0),
        'pile': knowledge_matches * 1.2 + (1.2 if knowledge_format else 0),
        'gpteacher': instruction_matches * 1.3 + (1.5 if instruction_format else 0),
        'openassistant': assistant_matches * 1.1
    }
    
    # Get the dataset with the highest score
    best_dataset = max(scores, key=scores.get)
    
    # If the best score is too low, default to a reasonable choice
    if scores[best_dataset] < 1.0:
        # Default to writing_prompts for creative/ambiguous prompts
        # or persona_chat for question-like prompts
        return 'persona_chat' if question_mark_count > 0 else 'writing_prompts'
    
    return best_dataset

# Function to verify spaCy is working correctly
def verify_spacy():
    """
    Verify spaCy is installed and working correctly.
    Returns a tuple of (is_working, message)
    """
    if not SPACY_AVAILABLE:
        return False, "spaCy is not installed"
    
    if not SPACY_MODEL_LOADED:
        return False, "spaCy model 'en_core_web_sm' is not loaded"
    
    try:
        # Try a simple test
        doc = NLP("This is a test sentence for spaCy.")
        tokens = [token.text for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        return True, f"spaCy is working correctly. Tokenized: {tokens[:3]}..."
    except Exception as e:
        return False, f"Error using spaCy: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Print spaCy status
    is_working, message = verify_spacy()
    if is_working:
        logging.info(f"spaCy status: {message}")
    else:
        logging.warning(f"spaCy status: {message}")
        logging.info("Run setup/setup_spacy.py to fix spaCy installation")
    
    # Create the enhancer
    enhancer = PromptEnhancer()
    
    # Test with a sample prompt
    sample_prompt = "Write a story about a journey through space and time with aliens and robots."
    enhanced = enhancer.enhance_prompt(sample_prompt)
    print("\nSample Prompt:", sample_prompt)
    print("\nEnhanced Prompt:")
    print(enhanced)
    
    # Test dataset analysis
    dataset = analyze_prompt(sample_prompt)
    print(f"\nRecommended dataset: {dataset}")
