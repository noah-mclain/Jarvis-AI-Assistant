import io
import os
import logging
from PIL import Image
import spacy
from typing import Union
import zipfile

from .utils import is_zipfile

class PromptEnhancer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = logging.getLogger("PromptEnhancer")
        
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
        """Enhance text prompts using NLP techniques"""
        doc = self.nlp(text)

        # Extract entities and syntactic features
        entities = [ent.text for ent in doc.ents]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        return f"Generate high-quality output based on these requirements:\nMain subject: {', '.join(entities) if entities else 'general request'}\nActions needed: {', '.join(verbs) if verbs else 'create content'}\nDetailed description: {text}\nInclude appropriate technical specifications and artistic style based on the context."

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
    Returns either 'persona_chat' or 'writing_prompts'.
    """
    # Convert to lowercase for analysis
    prompt_lower = prompt.lower()

    # Keywords that suggest dialogue/conversation
    dialogue_keywords = [
        'hello', 'hi', 'hey', 'how are you', 'what do you', 'can you',
        'would you', 'tell me', 'explain', 'help me', 'i need', 'i want',
        'question', 'ask', 'answer', 'chat', 'talk', 'conversation',
        'discuss', 'advice', 'suggest', 'recommend'
    ]

    # Keywords that suggest story generation
    story_keywords = [
        'story', 'write', 'create', 'imagine', 'world where', 'what if',
        'once upon', 'beginning', 'end', 'plot', 'character', 'setting',
        'scene', 'describe', 'narrative', 'tale', 'fable', 'myth',
        'adventure', 'journey', 'quest', 'tale', 'legend'
    ]

    # Count matches for each type
    dialogue_matches = sum(
        keyword in prompt_lower for keyword in dialogue_keywords
    )
    story_matches = sum(keyword in prompt_lower for keyword in story_keywords)

    # Check for question marks (more likely to be dialogue)
    question_mark_count = prompt_lower.count('?')

    # Check for story-like formatting
    story_format = any([
        prompt_lower.startswith(('write', 'create', 'imagine')),
        'story about' in prompt_lower,
        'tale of' in prompt_lower
    ])

    # Decision logic
    if story_format or story_matches > dialogue_matches + 2:
        return 'writing_prompts'
    elif question_mark_count > 0 or dialogue_matches > story_matches + 2:
        return 'persona_chat'
    else:
        # Default to writing_prompts for creative/ambiguous prompts
        return 'writing_prompts'

# Example usage
if __name__ == "__main__":
    enhancer = PromptEnhancer()
    
    # Text enhancement
    print(enhancer.enhance_prompt("Create a futuristic cityscape"))
    
    # File enhancement
    print(enhancer.enhance_prompt("example.jpg"))
    
    # Zip enhancement
    print(enhancer.enhance_prompt("inputs.zip"))
