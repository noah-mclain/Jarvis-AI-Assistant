import json
import os

# Create the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DeepSeek Fine-tuning with Unsloth on Kaggle\n",
                "\n",
                "This notebook demonstrates how to fine-tune the DeepSeek-Coder model using Unsloth on Kaggle's GPU resources. The notebook is designed to run the fine-tuning process without any import or requirement conflicts.\n",
                "\n",
                "## What You'll Learn\n",
                "- Setting up the right environment on Kaggle for DeepSeek + Unsloth\n",
                "- Loading the project code directly from GitHub\n",
                "- Running the fine-tuning process with GPU acceleration\n",
                "- Saving and exporting your model"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup Environment\n",
                "\n",
                "First, we need to ensure that all dependencies are properly installed. Kaggle already comes with many libraries pre-installed, but we need to make sure we have the specific versions required for Unsloth and DeepSeek compatibility."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Check GPU availability\n",
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Install required packages\n",
                "!pip install -q unsloth unsloth_zoo\n",
                "!pip install -q transformers==4.48.3 accelerate==0.33.0 peft==0.11.1 trl==0.8.1 bitsandbytes==0.43.0 datasets==2.19.0\n",
                "\n",
                "# Verify torch installation\n",
                "import torch\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
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
                "## Clone the Repository\n",
                "\n",
                "Now we'll clone the GitHub repository containing the fine-tuning code."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Clone the repository\n",
                "!git clone https://github.com/yourusername/Jarvis-AI-Assistant.git\n",
                "!ls -la Jarvis-AI-Assistant/"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Set Up Python Path\n",
                "\n",
                "We need to add the repository to the Python path to ensure imports work correctly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import sys\n",
                "import os\n",
                "\n",
                "# Add the repository to the Python path\n",
                "repo_path = os.path.join(os.getcwd(), \"Jarvis-AI-Assistant\")\n",
                "sys.path.append(repo_path)\n",
                "print(f\"Added {repo_path} to Python path\")\n",
                "\n",
                "# Import and create necessary directories\n",
                "os.makedirs(os.path.join(repo_path, \"src/generative_ai_module/models/deepseek_unsloth\"), exist_ok=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Import the DeepSeek and Unsloth Modules\n",
                "\n",
                "Let's make sure we can properly import all required modules."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Import Unsloth first (this order is important)\n",
                "import unsloth\n",
                "from unsloth import FastLanguageModel\n",
                "\n",
                "# Try importing our modules\n",
                "try:\n",
                "    from src.generative_ai_module.unsloth_deepseek import finetune_with_unsloth\n",
                "    print(\"Successfully imported project modules\")\n",
                "except ImportError as e:\n",
                "    print(f\"Import error: {e}\")\n",
                "    print(\"\\nFallback: Let's implement these modules directly in the notebook...\")\n",
                "    # The following cells will contain fallback implementations if import fails"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Fallback: Create Essential Functions\n",
                "\n",
                "In case the imports fail, we'll implement the key functions directly in the notebook."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Define the key function for fine-tuning with Unsloth\n",
                "def finetune_with_unsloth(\n",
                "    train_dataset,\n",
                "    eval_dataset=None,\n",
                "    model_name=\"deepseek-ai/deepseek-coder-6.7b-base\",\n",
                "    output_dir=\"models/deepseek_unsloth\",\n",
                "    max_seq_length=2048,\n",
                "    per_device_train_batch_size=2,\n",
                "    gradient_accumulation_steps=4,\n",
                "    learning_rate=2e-5,\n",
                "    max_steps=500,\n",
                "    logging_steps=10,\n",
                "    save_steps=100,\n",
                "    warmup_steps=50,\n",
                "    weight_decay=0.01,\n",
                "    load_in_4bit=True,\n",
                "    load_in_8bit=False,\n",
                "    r=16,\n",
                "    target_modules=None,\n",
                "    save_total_limit=3,\n",
                "):\n",
                "    \"\"\"Fine-tune a DeepSeek-Coder model with Unsloth optimization.\"\"\"\n",
                "    import time\n",
                "    import json\n",
                "    import os\n",
                "    from trl import SFTTrainer\n",
                "    from peft import LoraConfig\n",
                "    from transformers import TrainingArguments\n",
                "    \n",
                "    start_time = time.time()\n",
                "    \n",
                "    # Set default target modules for DeepSeek-Coder if not specified\n",
                "    if target_modules is None:\n",
                "        target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"]\n",
                "    \n",
                "    # Load model and tokenizer with Unsloth optimization\n",
                "    model, tokenizer = FastLanguageModel.from_pretrained(\n",
                "        model_name=model_name,\n",
                "        max_seq_length=max_seq_length,\n",
                "        load_in_4bit=load_in_4bit,\n",
                "        load_in_8bit=load_in_8bit,\n",
                "        device_map=\"auto\"\n",
                "    )\n",
                "    \n",
                "    # Apply LoRA config\n",
                "    lora_config = LoraConfig(\n",
                "        r=r,\n",
                "        target_modules=target_modules,\n",
                "        lora_alpha=16,\n",
                "        lora_dropout=0.1,\n",
                "        task_type=\"CAUSAL_LM\",\n",
                "    )\n",
                "    \n",
                "    model = FastLanguageModel.get_peft_model(\n",
                "        model, \n",
                "        lora_config,\n",
                "        use_gradient_checkpointing=True\n",
                "    )\n",
                "    \n",
                "    # Create training arguments\n",
                "    training_args = TrainingArguments(\n",
                "        output_dir=output_dir,\n",
                "        per_device_train_batch_size=per_device_train_batch_size,\n",
                "        gradient_accumulation_steps=gradient_accumulation_steps,\n",
                "        learning_rate=learning_rate,\n",
                "        max_steps=max_steps,\n",
                "        logging_steps=logging_steps,\n",
                "        save_steps=save_steps,\n",
                "        save_total_limit=save_total_limit,\n",
                "        warmup_steps=warmup_steps,\n",
                "        weight_decay=weight_decay,\n",
                "        lr_scheduler_type=\"cosine\",\n",
                "        fp16=not load_in_4bit,\n",
                "        bf16=False,\n",
                "        optim=\"adamw_torch\",\n",
                "        report_to=\"none\",\n",
                "        group_by_length=True,\n",
                "        save_strategy=\"steps\",\n",
                "        remove_unused_columns=True,\n",
                "        run_name=\"deepseek_unsloth\"\n",
                "    )\n",
                "    \n",
                "    # Create SFT trainer\n",
                "    trainer = SFTTrainer(\n",
                "        model=model,\n",
                "        tokenizer=tokenizer,\n",
                "        train_dataset=train_dataset,\n",
                "        eval_dataset=eval_dataset,\n",
                "        dataset_text_field=\"text\",\n",
                "        max_seq_length=max_seq_length,\n",
                "        args=training_args,\n",
                "        packing=True,\n",
                "        tokenizer_name=model_name\n",
                "    )\n",
                "    \n",
                "    # Train the model\n",
                "    print(\"Starting training...\")\n",
                "    trainer.train()\n",
                "    \n",
                "    # Save the fine-tuned model\n",
                "    trainer.save_model(output_dir)\n",
                "    print(f\"Model saved to {output_dir}\")\n",
                "    \n",
                "    # Create and save metrics\n",
                "    metrics = {\n",
                "        \"timestamp\": time.strftime(\"%Y-%m-%d %H:%M:%S\"),\n",
                "        \"model_name\": model_name,\n",
                "        \"training_time_minutes\": round((time.time() - start_time) / 60, 2),\n",
                "        \"max_steps\": max_steps,\n",
                "        \"learning_rate\": learning_rate,\n",
                "        \"batch_size\": per_device_train_batch_size,\n",
                "        \"gradient_accumulation_steps\": gradient_accumulation_steps,\n",
                "        \"lora_rank\": r,\n",
                "    }\n",
                "    \n",
                "    # Save metrics\n",
                "    metrics_path = os.path.join(output_dir, \"training_metrics.json\")\n",
                "    with open(metrics_path, 'w') as f:\n",
                "        json.dump(metrics, f, indent=2)\n",
                "    \n",
                "    return metrics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Function to create a mini dataset for testing\n",
                "def create_mini_dataset(sequence_length=512):\n",
                "    \"\"\"Create a small dataset for testing fine-tuning without external data\"\"\"\n",
                "    from datasets import Dataset\n",
                "    from transformers import AutoTokenizer\n",
                "    \n",
                "    print(\"Creating mini test dataset...\")\n",
                "    \n",
                "    # Sample code examples\n",
                "    examples = [\n",
                "        {\n",
                "            \"text\": \"### Instruction: Implement a function to calculate the Fibonacci sequence.\\n\\n### Response:\\ndef fibonacci(n):\\n    if n <= 0:\\n        return []\\n    elif n == 1:\\n        return [0]\\n    elif n == 2:\\n        return [0, 1]\\n    else:\\n        fib = [0, 1]\\n        for i in range(2, n):\\n            fib.append(fib[i-1] + fib[i-2])\\n        return fib\"\n",
                "        },\n",
                "        {\n",
                "            \"text\": \"### Instruction: Write a function to check if a string is a palindrome.\\n\\n### Response:\\ndef is_palindrome(s):\\n    s = s.lower()\\n    s = ''.join(c for c in s if c.isalnum())\\n    return s == s[::-1]\"\n",
                "        },\n",
                "        {\n",
                "            \"text\": \"### Instruction: Create a function to sort a list using bubble sort.\\n\\n### Response:\\ndef bubble_sort(arr):\\n    n = len(arr)\\n    for i in range(n):\\n        for j in range(0, n-i-1):\\n            if arr[j] > arr[j+1]:\\n                arr[j], arr[j+1] = arr[j+1], arr[j]\\n    return arr\"\n",
                "        },\n",
                "        {\n",
                "            \"text\": \"### Instruction: Implement a function to find the greatest common divisor of two numbers.\\n\\n### Response:\\ndef gcd(a, b):\\n    while b:\\n        a, b = b, a % b\\n    return a\"\n",
                "        },\n",
                "        {\n",
                "            \"text\": \"### Instruction: Write a function to check if a number is prime.\\n\\n### Response:\\ndef is_prime(n):\\n    if n <= 1:\\n        return False\\n    if n <= 3:\\n        return True\\n    if n % 2 == 0 or n % 3 == 0:\\n        return False\\n    i = 5\\n    while i * i <= n:\\n        if n % i == 0 or n % (i + 2) == 0:\\n            return False\\n        i += 6\\n    return True\"\n",
                "        }\n",
                "    ]\n",
                "    \n",
                "    # Create dataset directly with text field (for Unsloth)\n",
                "    dataset = Dataset.from_dict({\"text\": [ex[\"text\"] for ex in examples]})\n",
                "    \n",
                "    # Split into train and validation\n",
                "    train_size = int(0.8 * len(dataset))\n",
                "    train_dataset = dataset.select(range(train_size))\n",
                "    eval_dataset = dataset.select(range(train_size, len(dataset)))\n",
                "    \n",
                "    print(f\"Created mini dataset with {len(train_dataset)} training and {len(eval_dataset)} validation examples\")\n",
                "    \n",
                "    return train_dataset, eval_dataset"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Create a Test Dataset\n",
                "\n",
                "Let's create a small test dataset for fine-tuning."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create the test dataset\n",
                "train_dataset, eval_dataset = create_mini_dataset(sequence_length=1024)\n",
                "\n",
                "# Display a sample\n",
                "print(\"\\nSample from training dataset:\")\n",
                "print(train_dataset[0][\"text\"][:300], \"...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Fine-tune the Model\n",
                "\n",
                "Now we'll run the fine-tuning process. We'll use a small number of steps for demonstration purposes, but you can increase this for better results."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Set up output directory\n",
                "output_dir = \"deepseek_unsloth_finetuned\"\n",
                "\n",
                "# Run fine-tuning with limited steps for demonstration\n",
                "training_metrics = finetune_with_unsloth(\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=eval_dataset,\n",
                "    model_name=\"deepseek-ai/deepseek-coder-6.7b-base\",\n",
                "    output_dir=output_dir,\n",
                "    max_seq_length=1024,  # Reduced for faster training\n",
                "    per_device_train_batch_size=2,  # Adjust based on GPU memory\n",
                "    gradient_accumulation_steps=4,\n",
                "    max_steps=50,  # Small number for demonstration\n",
                "    logging_steps=5,\n",
                "    save_steps=25,\n",
                "    warmup_steps=10,\n",
                "    load_in_4bit=True  # Use 4-bit quantization for memory efficiency\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test the Fine-tuned Model\n",
                "\n",
                "Let's test our fine-tuned model with a simple coding prompt."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Load the fine-tuned model\n",
                "from transformers import AutoTokenizer\n",
                "import torch\n",
                "\n",
                "# Load tokenizer\n",
                "tokenizer = AutoTokenizer.from_pretrained(\"deepseek-ai/deepseek-coder-6.7b-base\", trust_remote_code=True)\n",
                "\n",
                "# Load the fine-tuned model with Unsloth\n",
                "model, _ = FastLanguageModel.from_pretrained(\n",
                "    model_name=\"deepseek-ai/deepseek-coder-6.7b-base\",\n",
                "    adapter_path=output_dir,\n",
                "    max_seq_length=1024,\n",
                "    load_in_4bit=True,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "\n",
                "# Function to generate code\n",
                "def generate_code(prompt, max_new_tokens=200):\n",
                "    formatted_prompt = f\"### Instruction: {prompt}\\n\\n### Response:\"\n",
                "    \n",
                "    inputs = tokenizer(formatted_prompt, return_tensors=\"pt\").to(model.device)\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=max_new_tokens,\n",
                "            do_sample=True,\n",
                "            temperature=0.7,\n",
                "            top_p=0.95\n",
                "        )\n",
                "    \n",
                "    response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
                "    # Extract just the response part\n",
                "    response = response.split(\"### Response:\")[-1].strip()\n",
                "    return response\n",
                "\n",
                "# Test with a simple prompt\n",
                "test_prompt = \"Write a Python function to reverse a string\"\n",
                "generated_code = generate_code(test_prompt)\n",
                "\n",
                "print(f\"Prompt: {test_prompt}\")\n",
                "print(\"\\nGenerated code:\")\n",
                "print(generated_code)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Save and Download the Model\n",
                "\n",
                "Let's save and prepare the model for downloading from Kaggle."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Compress the model folder for easier downloading\n",
                "!tar -czvf deepseek_unsloth_finetuned.tar.gz {output_dir}\n",
                "print(\"Model compressed successfully. You can now download 'deepseek_unsloth_finetuned.tar.gz' from the output files.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Using for Production\n",
                "\n",
                "To use this model in your project, you can load it with the following code:\n",
                "\n",
                "```python\n",
                "from unsloth import FastLanguageModel\n",
                "from transformers import AutoTokenizer\n",
                "\n",
                "# Load the base model and tokenizer\n",
                "tokenizer = AutoTokenizer.from_pretrained(\"deepseek-ai/deepseek-coder-6.7b-base\", trust_remote_code=True)\n",
                "\n",
                "# Load the fine-tuned model with Unsloth\n",
                "model, _ = FastLanguageModel.from_pretrained(\n",
                "    model_name=\"deepseek-ai/deepseek-coder-6.7b-base\",\n",
                "    adapter_path=\"path/to/your/downloaded/model\",\n",
                "    max_seq_length=2048,\n",
                "    load_in_4bit=True,  # Adjust based on your hardware\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "```\n",
                "\n",
                "This gives you the same optimized model for inference."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "In this notebook, we've demonstrated how to fine-tune DeepSeek-Coder using Unsloth on Kaggle's GPU resources. Key takeaways:\n",
                "\n",
                "1. **Environment Setup**: We set up the appropriate environment for Unsloth and DeepSeek compatibility\n",
                "2. **Memory Efficiency**: Using 4-bit quantization and Unsloth optimizations to fit large models in limited GPU memory\n",
                "3. **Easy Training**: The fine-tuning process is simplified with optimized functions\n",
                "4. **Portability**: The fine-tuned model can be downloaded and used in other environments\n",
                "\n",
                "You can now adapt this notebook to fine-tune on your own data by replacing the mini dataset with your actual dataset."
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
            "version": "3.10.12"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write the notebook to a file
notebook_path = "kaggle_deepseek_unsloth.ipynb"
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Kaggle notebook created at {notebook_path}") 