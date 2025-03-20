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

# Example usage
if __name__ == "__main__":
    enhancer = PromptEnhancer()
    
    # Text enhancement
    print(enhancer.enhance_prompt("Create a futuristic cityscape"))
    
    # File enhancement
    print(enhancer.enhance_prompt("example.jpg"))
    
    # Zip enhancement
    print(enhancer.enhance_prompt("inputs.zip"))
