import json
import os

# Create the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Optimizing DeepSeek with Unsloth\n",
                "\n",
                "This notebook demonstrates how to optimize and fine-tune the DeepSeek model using Unsloth's optimization techniques. Unsloth provides significant improvements in memory usage and training speed for LLM fine-tuning.\n",
                "\n",
                "## What You Will Learn\n",
                "- How to set up Unsloth with DeepSeek\n",
                "- Memory-efficient fine-tuning techniques\n",
                "- Speeding up training with optimized kernels\n",
                "- Comparing performance between standard and Unsloth-optimized training"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Environment Setup\n",
                "\n",
                "First, let's install the necessary packages. We'll need the Unsloth library, DeepSeek, and other related dependencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Install required packages\n",
                "!pip install unsloth unsloth_zoo transformers accelerate bitsandbytes trl peft datasets scipy -q\n",
                "\n",
                "# Verify CUDA is available\n",
                "import torch\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"CUDA version: {torch.version.cuda}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Import Libraries\n",
                "\n",
                "Now let's import all the necessary libraries for our fine-tuning process. Note that we import `unsloth` first, before other libraries, to ensure all optimizations are properly applied."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Import unsloth first\n",
                "import unsloth\n",
                "from unsloth import FastLanguageModel\n",
                "\n",
                "# Then import other libraries\n",
                "import torch\n",
                "from datasets import load_dataset\n",
                "import transformers\n",
                "import time\n",
                "from transformers import AutoTokenizer\n",
                "from peft import LoraConfig\n",
                "from trl import SFTTrainer\n",
                "import gc"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Loading DeepSeek with Unsloth\n",
                "\n",
                "Unsloth provides optimized loading of models. Let's load the DeepSeek model and see how it compares to standard loading procedures."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Model configuration\n",
                "model_name = \"deepseek-ai/deepseek-coder-6.7b-base\"\n",
                "max_seq_length = 2048  # Adjust based on your GPU memory\n",
                "\n",
                "# Memory usage before loading\n",
                "torch.cuda.empty_cache()\n",
                "gc.collect()\n",
                "mem_before = torch.cuda.memory_allocated()/1024**3\n",
                "print(f\"Memory before loading: {mem_before:.2f} GB\")\n",
                "\n",
                "# Load model and tokenizer with Unsloth\n",
                "start_time = time.time()\n",
                "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                "    model_name=model_name,\n",
                "    max_seq_length=max_seq_length,\n",
                "    dtype=torch.bfloat16,\n",
                "    load_in_4bit=True,  # Quantize for memory efficiency\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "loading_time = time.time() - start_time\n",
                "\n",
                "# Memory usage after loading\n",
                "mem_after = torch.cuda.memory_allocated()/1024**3\n",
                "print(f\"Memory after loading: {mem_after:.2f} GB\")\n",
                "print(f\"Memory usage for model: {mem_after - mem_before:.2f} GB\")\n",
                "print(f\"Model loading time: {loading_time:.2f} seconds\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setting up LoRA for Fine-tuning\n",
                "\n",
                "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. Unsloth makes this even more efficient with its optimized implementation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Configure LoRA\n",
                "lora_config = LoraConfig(\n",
                "    r=16,  # Rank of the update matrices\n",
                "    lora_alpha=32,  # Alpha parameter for LoRA scaling\n",
                "    target_modules=[\n",
                "        \"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \n",
                "        \"gate_proj\", \"up_proj\", \"down_proj\"\n",
                "    ],\n",
                "    bias=\"none\",\n",
                "    lora_dropout=0.05,  # Dropout probability for LoRA layers\n",
                "    task_type=\"CAUSAL_LM\"\n",
                ")\n",
                "\n",
                "# Apply LoRA to the model\n",
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    lora_config,\n",
                "    use_gradient_checkpointing=True  # Further reduce memory usage\n",
                ")\n",
                "\n",
                "print(\"LoRA configuration applied to the model.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Prepare Dataset for Fine-tuning\n",
                "\n",
                "We'll use a small code completion dataset for this demonstration."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Load a sample dataset\n",
                "dataset = load_dataset(\"sahil2801/CodeAlpaca-20k\", split=\"train\")\n",
                "print(f\"Dataset loaded with {len(dataset)} examples\")\n",
                "print(\"Sample from dataset:\")\n",
                "print(dataset[0])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Format the Dataset\n",
                "\n",
                "We need to format our dataset into prompt-response pairs suitable for the DeepSeek model's format."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Define the chat template for DeepSeek\n",
                "def format_prompt(sample):\n",
                "    instruction = sample[\"instruction\"]\n",
                "    output = sample[\"output\"]\n",
                "    \n",
                "    # Format according to DeepSeek's instruction format\n",
                "    formatted_text = f\"### Instruction: {instruction}\\n\\n### Response: {output}\"\n",
                "    return {\"text\": formatted_text}\n",
                "\n",
                "# Apply formatting to dataset\n",
                "formatted_dataset = dataset.map(format_prompt)\n",
                "print(\"Dataset formatted. Sample:\")\n",
                "print(formatted_dataset[0][\"text\"][:500], \"...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Set up the Trainer\n",
                "\n",
                "Now we'll configure the SFTTrainer with our model, tokenizer, and dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Training arguments\n",
                "training_args = transformers.TrainingArguments(\n",
                "    per_device_train_batch_size=4,\n",
                "    gradient_accumulation_steps=4,\n",
                "    warmup_steps=5,\n",
                "    max_steps=50,\n",
                "    learning_rate=2e-4,\n",
                "    fp16=True,\n",
                "    logging_steps=1,\n",
                "    output_dir=\"results\",\n",
                "    optim=\"adamw_torch\"\n",
                ")\n",
                "\n",
                "# Set up the trainer\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=formatted_dataset,\n",
                "    dataset_text_field=\"text\",\n",
                "    max_seq_length=max_seq_length,\n",
                "    tokenizer=tokenizer,\n",
                "    args=training_args,\n",
                "    packing=True  # Enables input packing for efficiency\n",
                ")\n",
                "\n",
                "print(\"Trainer configured and ready for fine-tuning.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Fine-tune the Model\n",
                "\n",
                "Now let's start the fine-tuning process and measure performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Track memory usage and time\n",
                "torch.cuda.empty_cache()\n",
                "gc.collect()\n",
                "mem_before_training = torch.cuda.memory_allocated()/1024**3\n",
                "print(f\"Memory before training: {mem_before_training:.2f} GB\")\n",
                "\n",
                "# Start training\n",
                "start_training_time = time.time()\n",
                "trainer.train()\n",
                "training_time = time.time() - start_training_time\n",
                "\n",
                "# Memory after training\n",
                "mem_after_training = torch.cuda.memory_allocated()/1024**3\n",
                "print(f\"Memory after training: {mem_after_training:.2f} GB\")\n",
                "print(f\"Peak memory during training: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB\")\n",
                "print(f\"Training time: {training_time:.2f} seconds\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Save the Fine-tuned Model\n",
                "\n",
                "Let's save our fine-tuned model for later use."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Save the model\n",
                "output_dir = \"deepseek-unsloth-finetuned\"\n",
                "model.save_pretrained(output_dir)\n",
                "tokenizer.save_pretrained(output_dir)\n",
                "print(f\"Model saved to {output_dir}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test Generation with Fine-tuned Model\n",
                "\n",
                "Now let's test the generation capabilities of our fine-tuned model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Prepare the model for inference\n",
                "model.eval()\n",
                "\n",
                "# Define a function for generation\n",
                "def generate_response(prompt, max_new_tokens=128):\n",
                "    # Format prompt according to DeepSeek's instruction format\n",
                "    formatted_prompt = f\"### Instruction: {prompt}\\n\\n### Response: \"\n",
                "    \n",
                "    # Tokenize the prompt\n",
                "    inputs = tokenizer(formatted_prompt, return_tensors=\"pt\").to(model.device)\n",
                "    \n",
                "    # Track generation time\n",
                "    start_time = time.time()\n",
                "    \n",
                "    # Generate response\n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=max_new_tokens,\n",
                "            temperature=0.7,\n",
                "            top_p=0.9,\n",
                "            do_sample=True\n",
                "        )\n",
                "    \n",
                "    generation_time = time.time() - start_time\n",
                "    \n",
                "    # Decode and clean the response\n",
                "    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
                "    response = generated_text.split(\"### Response: \")[-1].strip()\n",
                "    \n",
                "    return response, generation_time\n",
                "\n",
                "# Test with a coding prompt\n",
                "test_prompt = \"Write a Python function to find all prime numbers less than n using the Sieve of Eratosthenes algorithm.\"\n",
                "response, gen_time = generate_response(test_prompt)\n",
                "\n",
                "print(f\"Prompt: {test_prompt}\")\n",
                "print(f\"\\nResponse (generated in {gen_time:.2f} seconds):\")\n",
                "print(response)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Performance Comparison\n",
                "\n",
                "Let's compare the performance of the Unsloth-optimized model with the standard approach."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create a table for comparison\n",
                "import pandas as pd\n",
                "\n",
                "# These values should be replaced with actual measurements if running both approaches\n",
                "comparison_data = {\n",
                "    \"Metric\": [\"Model Loading Time (s)\", \"Peak Memory Usage (GB)\", \"Training Time (s)\", \"Generation Time (s)\"],\n",
                "    \"Standard DeepSeek\": [\"~300\", \"~24\", \"~600\", \"~1.5\"],  # Example values - replace with actual measurements\n",
                "    \"Unsloth-Optimized\": [f\"{loading_time:.2f}\", f\"{torch.cuda.max_memory_allocated()/1024**3:.2f}\", f\"{training_time:.2f}\", f\"{gen_time:.2f}\"],\n",
                "    \"Improvement\": [\"~60%\", \"~40%\", \"~65%\", \"~40%\"]  # Example improvements - replace with actual calculations\n",
                "}\n",
                "\n",
                "comparison_df = pd.DataFrame(comparison_data)\n",
                "comparison_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "This notebook demonstrated how to optimize the DeepSeek model using Unsloth. Key benefits include:\n",
                "\n",
                "1. **Reduced Memory Usage**: Unsloth's optimized 4-bit quantization and efficient LoRA implementation significantly reduce memory requirements.\n",
                "2. **Faster Training**: The optimized kernels in Unsloth accelerate training by minimizing computational overhead.\n",
                "3. **Efficient Inference**: Even inference can be faster with Unsloth's optimizations.\n",
                "4. **Simple Integration**: Unsloth is designed to be a drop-in replacement that works seamlessly with the Hugging Face ecosystem.\n",
                "\n",
                "By using Unsloth with DeepSeek, you can fine-tune larger models on less powerful hardware, or train models faster on the same hardware."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Make sure the notebooks directory exists
os.makedirs("notebooks", exist_ok=True)

# Write the notebook to a file
notebook_path = "notebooks/unsloth_deepseek_demo.ipynb"
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook created at {notebook_path}") 