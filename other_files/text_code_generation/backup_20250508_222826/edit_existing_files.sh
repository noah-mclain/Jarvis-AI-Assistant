#!/bin/bash
# Edit existing files to fix Unsloth issues
# This script directly modifies the unified_deepseek_training.py file and custom_unsloth implementation

echo "===== Fixing Unsloth Issues in Existing Files ====="

# Check if unified_deepseek_training.py exists
if [ -f "/notebooks/unified_deepseek_training.py" ]; then
    echo "Fixing /notebooks/unified_deepseek_training.py..."
    
    # Create a backup
    cp /notebooks/unified_deepseek_training.py /notebooks/unified_deepseek_training.py.bak
    
    # Fix the FastLanguageModel.from_pretrained call
    sed -i 's/model, tokenizer = FastLanguageModel\.from_pretrained(\s*model_name=args\.model_name,\s*max_seq_length=args\.max_length,\s*load_in_4bit=args\.load_in_4bit,\s*load_in_8bit=args\.load_in_8bit\(,\s*device_map="auto"\)\?/# Note: FastLanguageModel.from_pretrained already handles device_map internally\n    model, tokenizer = FastLanguageModel.from_pretrained(\n        model_name=args.model_name,\n        load_in_4bit=args.load_in_4bit,\n        load_in_8bit=args.load_in_8bit/g' /notebooks/unified_deepseek_training.py
    
    # Add code to set max sequence length after model is loaded
    sed -i '/model, tokenizer = FastLanguageModel\.from_pretrained(/,/)/a\
    \n    # Set max sequence length after model is loaded\n    model.config.max_position_embeddings = args.max_length\n    tokenizer.model_max_length = args.max_length' /notebooks/unified_deepseek_training.py
    
    # Fix the FastLanguageModel.get_peft_model call
    sed -i 's/model = FastLanguageModel\.get_peft_model(\s*model,\s*r=16,\s*lora_alpha=32,\s*lora_dropout=0\.05,\s*target_modules=\[\s*"q_proj", "k_proj", "v_proj", "o_proj",\s*"gate_proj", "up_proj", "down_proj"\s*\],\s*\(use_gradient_checkpointing=True,\s*\)\?\(random_state=42,\?\s*\)\?/# Note: use_gradient_checkpointing and random_state are not valid parameters for LoraConfig\n    model = FastLanguageModel.get_peft_model(\n        model,\n        r=16,  # LoRA rank\n        lora_alpha=32,\n        lora_dropout=0.05,\n        target_modules=[\n            "q_proj", "k_proj", "v_proj", "o_proj",\n            "gate_proj", "up_proj", "down_proj"\n        ]/g' /notebooks/unified_deepseek_training.py
    
    # Add gradient_checkpointing to TrainingArguments
    sed -i '/training_args = TrainingArguments(/,/)/s/group_by_length=True,\?/group_by_length=True,\n        # Enable gradient checkpointing for memory efficiency\n        gradient_checkpointing=True/g' /notebooks/unified_deepseek_training.py
    
    echo "✅ Successfully fixed /notebooks/unified_deepseek_training.py"
else
    echo "❌ /notebooks/unified_deepseek_training.py not found"
fi

# Check if custom_unsloth exists
if [ -d "/notebooks/custom_unsloth" ]; then
    echo "Fixing custom_unsloth implementation..."
    
    # Find the models/__init__.py file
    MODELS_INIT="/notebooks/custom_unsloth/unsloth/models/__init__.py"
    
    if [ -f "$MODELS_INIT" ]; then
        # Create a backup
        cp "$MODELS_INIT" "${MODELS_INIT}.bak"
        
        # Fix the AutoModelForCausalLM.from_pretrained call
        sed -i '/model = AutoModelForCausalLM\.from_pretrained(/,/)/s/device_map=.*,/# device_map is handled by FastLanguageModel\n/g' "$MODELS_INIT"
        
        # Fix the get_peft_model function
        sed -i 's/def get_peft_model(model, \(.*\)use_gradient_checkpointing=False, \(.*\)random_state=None\(.*\)):/def get_peft_model(model, \1 \2\3):\n    # use_gradient_checkpointing and random_state are not valid parameters for LoraConfig/g' "$MODELS_INIT"
        
        # Fix the LoraConfig call
        sed -i '/peft_config = LoraConfig(/,/)/s/use_gradient_checkpointing=use_gradient_checkpointing,/# use_gradient_checkpointing is handled by TrainingArguments\n/g' "$MODELS_INIT"
        sed -i '/peft_config = LoraConfig(/,/)/s/random_state=random_state,\?/# random_state is not a valid parameter\n/g' "$MODELS_INIT"
        
        echo "✅ Successfully fixed $MODELS_INIT"
    else
        echo "❌ $MODELS_INIT not found"
    fi
else
    echo "❌ /notebooks/custom_unsloth directory not found"
fi

echo "===== Fix Complete ====="
echo "You can now run the training script again."
