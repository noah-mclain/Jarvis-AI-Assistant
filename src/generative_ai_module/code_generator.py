from .text_generator import TextGenerator

class CodeGenerator:
    def __init__(self):
        self.text_generator = TextGenerator()

    def generate_code(self, prompt, length=100):
        enhanced_prompt = self.text_generator.handle_input(prompt)
        return self.text_generator.generate(
            initial_str=enhanced_prompt, pred_len=length
        )