# Jarvis AI Assistant - Generative AI Module

A comprehensive framework for training, evaluating, and using generative AI models across multiple datasets. This module provides unified interfaces for dataset handling, model training, and generation evaluation with a focus on preventing hallucinations and ensuring high-quality outputs.

## Features

- **Unified Dataset Handling**: Consistent processing for multiple datasets (writing_prompts, persona_chat, pile, openassistant, gpteacher)
- **Advanced Training Pipeline**: Train models on any combination of datasets with automatic validation and evaluation
- **Comprehensive Evaluation Metrics**: BERTScore, ROUGE, BLEU, perplexity, and hallucination detection
- **Human Feedback Framework**: Collect and incorporate human feedback to improve model outputs
- **Smart Dataset Selection**: Automatically determine the best dataset/model for a given prompt

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Jarvis-AI-Assistant.git
cd Jarvis-AI-Assistant

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for evaluation metrics
python -m src.generative_ai_module.evaluation_metrics
```

## Quick Start Guide

### 1. Dataset Exploration

Explore available datasets and their characteristics:

```bash
python src/generative_ai_module/dataset_demo.py --action compare_all
```

View examples from a specific dataset:

```bash
python src/generative_ai_module/dataset_demo.py --action show_examples --dataset writing_prompts
```

Test the prompt analyzer to determine the best dataset for a given prompt:

```bash
python src/generative_ai_module/dataset_demo.py --action test_analyzer --prompt "Write a story about dragons"
```

### 2. Training Models

Train on all datasets:

```bash
python src/generative_ai_module/train_unified_models.py --datasets all
```

Train on specific datasets:

```bash
python src/generative_ai_module/train_unified_models.py --datasets writing_prompts persona_chat
```

Train with customized parameters:

```bash
python src/generative_ai_module/train_unified_models.py --datasets pile --max-samples 1000 --epochs 20 --batch-size 64 --validation-split 0.15 --test-split 0.1 --early-stopping 5
```

### 3. Interactive Generation

Run the interactive generation interface:

```bash
python src/generative_ai_module/unified_generation_pipeline.py --mode interactive
```

Generate text with a specific dataset:

```bash
python src/generative_ai_module/unified_generation_pipeline.py --mode generate --dataset writing_prompts --prompt "A world where dragons and humans live together"
```

### 4. Evaluation

Evaluate generated text with comprehensive metrics:

```bash
python src/generative_ai_module/evaluate_generation.py --generated-file output.txt --reference-file reference.txt --dataset-name writing_prompts --collect-human-feedback
```

Run batch evaluation on test sets:

```bash
python src/generative_ai_module/evaluate_generation.py --batch-evaluate --dataset-name pile
```

## Command Reference

### Unified Dataset Handler

```bash
# Load a specific dataset with customized parameters
python -c "from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler; handler = UnifiedDatasetHandler(); data = handler.load_dataset('writing_prompts', max_samples=100); print(f'Loaded {len(data.get(\"batches\", []))} batches')"
```

### Training Pipeline

```bash
# Train all models
python src/generative_ai_module/train_unified_models.py

# Train text models only on specific datasets
python src/generative_ai_module/train_all_models.py --text-models-only --dataset persona_chat

# Train code model only
python src/generative_ai_module/train_all_models.py --code-model-only
```

### Generation Pipeline

```bash
# Interactive session
python src/generative_ai_module/unified_generation_pipeline.py --mode interactive

# Generate from specific model
python src/generative_ai_module/unified_generation_pipeline.py --mode generate --dataset writing_prompts --prompt "Your prompt here"

# Evaluate model performance
python src/generative_ai_module/unified_generation_pipeline.py --mode evaluate --dataset persona_chat
```

### Evaluation Metrics

```bash
# Evaluate with BERTScore, ROUGE, BLEU, and perplexity
python src/generative_ai_module/evaluation_metrics.py --reference "Reference text" --generated "Generated text" --dataset-name writing_prompts

# Batch evaluation with hallucination detection
python src/generative_ai_module/evaluation_metrics.py --batch-evaluate --dataset-name pile --reference-file references.txt --generated-file generations.txt
```

## Dataset Details

The system supports multiple datasets, each with unique characteristics:

1. **Writing Prompts**: Creative writing prompts and stories from Reddit's r/WritingPrompts

   - Format: `<PROMPT>\n[prompt]\n<STORY>\n[story]\n<END>`
   - Best for: Creative writing, story generation, fictional content

2. **Persona Chat**: Dialogue dataset with persona-conditioned conversations

   - Format: `<PERSONA>\n[traits]\n<DIALOGUE>\n[USER/ASSISTANT exchanges]\n<END>`
   - Best for: Dialogue systems, chatbots, persona-consistent responses

3. **The Pile**: Large-scale, diverse dataset of text from the internet

   - Format: Raw text from various sources (academic papers, books, websites)
   - Best for: General knowledge, fact-based Q&A, academic content

4. **OpenAssistant**: Assistant-style conversations with helpful responses

   - Format: `USER: [question]\nASSISTANT: [response]`
   - Best for: Helpful assistant responses, task completion

5. **GPTeacher**: Instruction-following dataset with educational content
   - Format: `USER: [instruction]\nASSISTANT: [instruction following]`
   - Best for: How-to guides, tutorials, step-by-step instructions

## Evaluation Metrics

The system incorporates multiple metrics to ensure high-quality generations:

1. **BERTScore**: Semantic similarity to reference text
2. **ROUGE/BLEU**: N-gram overlap metrics
3. **Perplexity**: Fluency of generated text
4. **Hallucination Detection**: Identifies factual inconsistencies
5. **Human Feedback**: Interactive collection of human assessments

## Advanced Usage

### Custom Dataset Processing

```python
from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler

# Initialize handler
handler = UnifiedDatasetHandler(cache_dir="cached_datasets")

# Load and prepare dataset
dataset = handler.load_dataset("writing_prompts", max_samples=1000)
splits = handler.prepare_for_training(
    dataset,
    batch_size=64,
    validation_split=0.15,
    test_split=0.1
)

# Get statistics
stats = handler.get_batch_statistics(splits["train"])
print(f"Training set: {stats['total_batches']} batches, avg length: {stats['input_length_avg']:.1f}")
```

### Comprehensive Evaluation

```python
from src.generative_ai_module.evaluation_metrics import EvaluationMetrics

# Initialize metrics
metrics = EvaluationMetrics(metrics_dir="evaluation_results")

# Evaluate a generation
results = metrics.evaluate_generation(
    prompt="Write a story about a robot who discovers emotions",
    generated_text="Unit-7 had never understood why humans smiled...",
    reference_text="RB-9 had been programmed to serve, but not to feel...",
    reference_facts=[
        "The robot is referred to as a unit with a number designation",
        "The robot interacts with a human child",
        "The robot doesn't initially understand emotions",
        "The robot experiences an emotional revelation"
    ],
    collect_human_feedback=True,
    dataset_name="story_generation"
)

# Print key metrics
print(f"BERTScore F1: {results['metrics']['bert_score']['f1'][0]:.4f}")
print(f"ROUGE-L F-measure: {results['metrics']['rouge']['rougeL_fmeasure']:.4f}")
print(f"Hallucination risk: {results['metrics']['hallucination']['hallucination_risk']:.4f}")
```

## Project Structure

```
src/generative_ai_module/
├── train_unified_models.py     # Unified training script for all datasets
├── unified_generation_pipeline.py  # Text generation pipeline
├── unified_dataset_handler.py  # Consistent dataset handling
├── evaluation_metrics.py       # Comprehensive evaluation metrics
├── dataset_processor.py        # Base dataset processing
├── improved_preprocessing.py   # Enhanced preprocessing techniques
├── text_generator.py           # Core text generation model
├── prompt_enhancer.py          # Prompt analysis and improvement
├── dataset_demo.py             # Dataset exploration tools
└── basic_tokenizer.py          # Character-level tokenization
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
