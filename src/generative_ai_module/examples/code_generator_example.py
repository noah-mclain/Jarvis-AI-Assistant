from src.generative_ai_module.code_generator import CodeGenerator

# Initialize the code generator
generator = CodeGenerator()

# Train on a directory of code files
generator.train_on_codebase("path/to/codebase", epochs=10)

# Generate code with a prompt
generated_code = generator.generate_code("Create a function to calculate fibonacci numbers")