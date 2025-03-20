# Test that imports work correctly
from src.generative_ai_module.text_generator import TextGenerator
from src.generative_ai_module.dataset_processor import DatasetProcessor

print("Successfully imported modules!")

# Initialize the text generator
text_generator = TextGenerator()
print("TextGenerator initialized successfully")

# Initialize the dataset processor
dataset_processor = DatasetProcessor(text_generator)
print("DatasetProcessor initialized successfully") 