import json
import os

# Create the notebook structure
notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# DeepSeek-Coder Fine-tuning with Unsloth on Google Colab"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# %% [code]\n",
        "#@title **CHANGED** Install Exact Dependencies\n",
        "!pip uninstall -y torch torchvision torchaudio protobuf thinc accelerate\n",
        "!pip install -q \"unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git\"\n",
        "!pip install -q torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121\n",
        "!pip install -q transformers==4.36.2 datasets==2.14.6 peft==0.8.2 trl==0.7.10 bitsandbytes==0.41.1\n",
        "!pip install -q accelerate==0.27.2 protobuf==3.20.3"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# %% [code]\n",
        "#@title **CHANGED** Setup File Structure\n",
        "import os\n",
        "os.makedirs(\"src/generative_ai_module\", exist_ok=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# %% [code]\n",
        "#@title **CHANGED** Upload Files Directly\n",
        "from google.colab import files\n",
        "uploaded = files.upload()\n",
        "\n",
        "# Move files to correct locations\n",
        "!mv run_finetune.py src/\n",
        "!mv code_preprocessing.py src/generative_ai_module/\n",
        "!mv finetune_deepseek.py src/generative_ai_module/\n",
        "!mv code_generator.py src/generative_ai_module/"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# %% [code]\n",
        "#@title **CHANGED** Run with Colab-Tuned Settings\n",
        "!python src/run_finetune.py \\\n",
        "    --max-samples 1000 \\  # Reduced from 8000\n",
        "    --batch-size 4 \\\n",
        "    --load-in-4bit \\\n",
        "    --subset python \\\n",
        "    --sequence-length 1024  # Reduced from 2048"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# %% [code]\n",
        "#@title **REMOVED** Clone Repository Section\n",
        "# (No longer needed since we're uploading files directly)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# %% [code]\n",
        "#@title **REMOVED** Conflicting Requirements Install\n",
        "# (We handle dependencies explicitly above)"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  }
}
# Write the notebook to a file
notebook_path = "colab_deepseek_unsloth.ipynb"
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Kaggle notebook created at {notebook_path}") 