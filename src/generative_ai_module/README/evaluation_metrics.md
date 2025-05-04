# Enhanced Evaluation Metrics

This module provides comprehensive evaluation metrics for generative AI models in the Jarvis AI Assistant. It includes rich evaluation capabilities for text generation quality assessment.

## Features

- **Semantic Similarity**: Uses BERTScore with advanced models like DeBERTa for better semantic understanding
- **Text Overlap Metrics**: ROUGE and BLEU scores for n-gram overlap assessment
- **Perplexity Calculation**: Measures fluency of generated text
- **Hallucination Detection**: Identifies factual errors and contradictions
- **Human Feedback Collection**: Framework for collecting structured feedback
- **Visualization Tools**: Generate plots and charts of evaluation results
- **Comprehensive Reports**: Aggregate metrics across tasks and datasets

## Usage Examples

### Basic Evaluation

```python
from src.generative_ai_module.evaluation_metrics import EvaluationMetrics

# Initialize metrics
metrics = EvaluationMetrics()

# Evaluate a single generation
result = metrics.evaluate_generation(
    prompt="Write a story about a robot discovering emotions.",
    generated_text="Unit-7 had never understood why humans smiled...",
    reference_text="RB-9 had been programmed to serve, but not to feel...",
    dataset_name="creative_writing"
)

# Access metrics
bert_score = result["metrics"]["bert_score"]
rouge_scores = result["metrics"]["rouge"]
```

### Batch Evaluation

```python
# Evaluate multiple samples
results = metrics.batch_evaluate(
    prompts=["Write a story...", "Explain quantum physics..."],
    generated_texts=["Unit-7 had...", "Quantum physics is..."],
    reference_texts=["RB-9 had...", "Quantum physics describes..."],
    dataset_name="mixed_content"
)
```

### Creating Reports

```python
# After evaluating multiple samples
report = metrics.create_evaluation_report(
    evaluation_results=all_results,
    report_name="model_evaluation",
    include_samples=True
)

# Generate visualizations
vis_files = metrics.visualize_metrics(
    report,
    output_dir="visualizations/",
    plot_type="comprehensive"
)
```

### Using the Evaluation Script

```bash
# Evaluate a model on a dataset
python src/generative_ai_module/evaluate_model.py \
    --model_path models/my_fine_tuned_model \
    --dataset data/test_examples.json \
    --visualize \
    --task_type creative
```

## Metrics Overview

### BERTScore

BERTScore leverages contextual embeddings from pre-trained language models to compute the similarity between candidate and reference sentences. This metric correlates better with human judgments than traditional n-gram overlap metrics.

### ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the n-gram overlap between generated and reference texts:

- ROUGE-1: Unigram overlap (individual words)
- ROUGE-2: Bigram overlap (word pairs)
- ROUGE-L: Longest common subsequence

### BLEU

BLEU (Bilingual Evaluation Understudy) measures the precision of n-grams in the generated text compared to reference texts. It applies a brevity penalty to penalize short translations.

### Hallucination Detection

The hallucination detection framework identifies:

- **Fact Coverage**: Percentage of reference facts present in the generated text
- **Contradiction Rate**: Percentage of facts contradicted in the generated text
- **Semantic Similarity**: Uses BERTScore to detect semantic contradictions

### Perplexity

Perplexity measures how well a model predicts a sample. Lower perplexity indicates better language modeling quality.

## Integration with Training

The evaluation metrics can be integrated with model training:

```python
# In train_models.py
trainer, results = train_model(
    model_name="gpt2",
    dataset_path="data/my_dataset",
    use_enhanced_eval=True,
    eval_metrics_dir="evaluation_results"
)
```

## Visualization Types

- **Simple**: Basic bar charts of key metrics
- **Comprehensive**: Multiple plots including correlations between metrics
- **Comparison**: Side-by-side comparisons between datasets or models

## Requirements

- torch
- transformers
- bert_score
- rouge_score
- nltk
- matplotlib
- seaborn

## Contributing

To extend the evaluation metrics:

1. Add new metric computations to the `EvaluationMetrics` class
2. Update the visualization tools to include new metrics
3. Ensure proper error handling and logging
4. Add tests for your new functionality

## Future Improvements

- Add support for multi-reference evaluation
- Implement reference-free metrics
- Integrate with popular benchmark datasets
- Add model-based factuality checking
