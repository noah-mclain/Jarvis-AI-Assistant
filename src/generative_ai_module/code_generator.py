from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from .code_preprocessing import load_and_preprocess_dataset, save_preprocessing_metrics
from .text_generator import TextGenerator
from .prompt_enhancer import PromptEnhancer
from .dataset_processor import DatasetProcessor
import os
import torch
import json
import datetime
import time
from tqdm import tqdm
import numpy as np
from .utils import setup_gpu_for_training, force_cuda_device, is_paperspace_environment

class CodeGenerator:
    def __init__(self, use_deepseek=False, load_in_8bit=True, load_in_4bit=False, force_gpu=False):
        self.use_deepseek = use_deepseek
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.force_gpu = force_gpu
        self.device = self._get_device()

        if self.use_deepseek:
            self.model_name = "deepseek-ai/deepseek-coder-6.7b-base"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Configure tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate quantization
            self.load_model()
        else:
            self.text_generator = TextGenerator()
            self.prompt_enhancer = PromptEnhancer()
            self.dataset_processor = DatasetProcessor(self.text_generator)
            
    def _get_device(self):
        """Determine the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
        # Always force GPU usage regardless of initialization parameter
        self.force_gpu = True
        
        print("Setting up GPU for all operations...")
        
        # Use the GPU utilities from utils.py to get the device with forced GPU mode
        try:
            # Use setup_gpu_for_training for detailed configuration
            device, gpu_config = setup_gpu_for_training(force_gpu=True)
            
            # Save GPU configuration for later model loading optimizations
            self.gpu_config = gpu_config
            
            # If we have a CUDA device, ensure it's properly configured
            if device.type == "cuda":
                # Apply any RTX5000-specific configurations
                if is_paperspace_environment() and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    if "RTX5000" in gpu_name or "RTX 5000" in gpu_name:
                        print(f"RTX 5000 GPU detected - applying optimized settings for model loading")
                        # Force 4-bit quantization for RTX5000 to maximize available memory
                        self.load_in_4bit = True
                        self.load_in_8bit = False
                
                # Set CUDA tensor as default type for all operations
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                
                # In PyTorch 2.0+, also set the default device
                if hasattr(torch, 'set_default_device'):
                    torch.set_default_device('cuda')
                
                print(f"GPU enforcement successful: Using CUDA device {device}")
                return device
            
            # For MPS (Apple Silicon) device
            elif device.type == "mps":
                print(f"GPU enforcement successful: Using Apple Silicon MPS device")
                # Optimize MPS settings
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                return device
            
            # For CPU fallback (only if both setup_gpu_for_training and the code below fail)
            else:
                print("Warning: setup_gpu_for_training returned CPU device despite force_gpu=True")
                # Fall through to try alternative methods for finding a GPU
        except Exception as e:
            print(f"Error in primary GPU setup: {e}")
            print("Attempting alternative GPU detection methods...")
        
        # Alternative GPU detection paths
        
        # First try CUDA
        if torch.cuda.is_available():
            # We have CUDA, so use it
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Alternative GPU enforcement: Using CUDA GPU: {gpu_name}")
            
            # Set CUDA tensor as default type
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # Apply RTX5000-specific configurations if detected
            if is_paperspace_environment() and ("RTX5000" in gpu_name or "RTX 5000" in gpu_name):
                print("RTX 5000 GPU detected - applying optimized settings")
                # Force GPU visibility
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                
                # Force 4-bit quantization for RTX5000 to maximize available memory
                self.load_in_4bit = True
                self.load_in_8bit = False
            
            return device
        # Then try MPS for Apple Silicon
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Alternative GPU enforcement: Using MPS (Apple Silicon GPU)")
            # Optimize MPS settings
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            return torch.device("mps")
        # Last resort - CPU with warning
        else:
            print("Warning: No GPU available despite forced GPU mode. Performance will be significantly slower on CPU.")
            return torch.device("cpu")
    
    def load_model(self):
        """Load the deepseek model with appropriate quantization settings"""
        print(f"Loading {self.model_name}...")
        
        # Special case for Apple Silicon (M1/M2/M3)
        if self.device.type == "mps":
            print("Loading model for Apple Silicon GPU with memory-efficient settings")
            
            # First load model with a device_map that spreads across CPU and GPU
            try:
                # Set environment variables to optimize MPS memory usage
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Avoid offloading to disk
                os.environ["PYTORCH_MPS_ACTIVE_MEMORY_MANAGER"] = "1"   # Better memory management
                
                print("Using 'auto' device mapping to optimize memory usage...")
                # Use device_map="auto" to let HuggingFace optimize placement
                from transformers.utils.quantization_config import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Try to load with optimized settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # Use float16 for better memory efficiency
                    device_map="auto",          # Let transformers decide placement
                    offload_folder="offload_folder", # Enable CPU offloading
                    offload_state_dict=True,    # Allow state dict offloading
                    max_memory={0: "4GiB", "cpu": "8GiB"}, # Limit GPU memory usage
                    # Apply 8-bit quantization for memory efficiency
                    quantization_config=quantization_config
                )
                print("Successfully loaded model with optimized memory settings")
                return
            except Exception as e:
                print(f"Error loading with optimized settings: {e}")
                print("Trying alternative loading approach...")
                
                try:
                    print("Loading in smaller segments...")
                    # Use transformers' loading in smaller segments
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,  # More stable on MPS
                        device_map="auto",          # Let transformers decide placement
                        low_cpu_mem_usage=True,     # Reduce CPU memory usage
                        offload_state_dict=True     # Enable state dict offloading
                    )
                    print("Successfully loaded model with segment approach")
                    return
                except Exception as e2:
                    print(f"Error with alternative loading: {e2}")
                    print("Falling back to basic CPU loading then MPS transfer...")
                    
                    # Last resort: load on CPU first then move to MPS
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
                    
                    print("Moving model to MPS device (without LoRA)")
                    try:
                        # Move to MPS in chunks to avoid OOM
                        for param in self.model.parameters():
                            param.data = param.data.to("mps")
                        print("Successfully moved model to MPS")
                    except Exception as e:
                        print(f"Error moving model to MPS, will use CPU instead: {e}")
                        self.device = torch.device("cpu")
            return
            
        # Set quantization parameters based on user settings (for CUDA devices)
        elif self.load_in_4bit and self.device.type == "cuda":
            print("Loading model in 4-bit quantization for extreme memory saving")
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Try loading with 4-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                print("Successfully loaded model with 4-bit quantization")
            except Exception as e:
                print(f"Error loading with 4-bit quantization: {e}")
                print("Falling back to 8-bit quantization...")
                try:
                    # Try 8-bit quantization as fallback
                    self.load_in_8bit = True
                    self.load_in_4bit = False
                    return self.load_model()  # Recursively call with new settings
                except Exception as e2:
                    print(f"Error with 8-bit quantization: {e2}")
                    print("Falling back to 16-bit precision...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
        elif self.load_in_8bit and self.device.type == "cuda":
            print("Loading model in 8-bit quantization")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True,
                )
            except Exception as e:
                print(f"Error loading with 8-bit quantization: {e}")
                print("Falling back to 16-bit precision...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            print("Loading model in full precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        
        print(f"\n=== Model Load Verification ===")
        print(f"Model type: {type(self.model)}")
        
        # Apply LoRA - only for non-MPS devices
        if self.device.type != "mps":
            lora_config = LoraConfig(
                r=16,  # Increased rank for better fine-tuning
                lora_alpha=32,  # Increased alpha
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Added more target modules
                lora_dropout=0.05,  # Lower dropout
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            print("Applying LoRA adapters to model...")
            try:
                self.model = get_peft_model(self.model, lora_config)
                print("Model ready with LoRA adapters")
            except Exception as e:
                print(f"Error applying LoRA adapters: {e}")
                print("Continuing with base model without LoRA")
    
    def generate_code(self, prompt, length=100, temperature=0.7, top_p=0.95):
        """
        Generate code from a prompt using either deepseek or text_generator
        
        Args:
            prompt: Text prompt for code generation
            length: Maximum length of generated code (used only for non-deepseek)
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter (only for deepseek)
            
        Returns:
            Generated code
        """
        if self.use_deepseek:
            # Format the prompt for deepseek model
            formatted_prompt = f"### Instruction: Write code for this task:\n{prompt}\n\n### Response:"
            # Ensure return_tensors="pt" is set to get PyTorch tensors
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
                    
            # Make sure the model and inputs are on the same device
            current_device = next(self.model.parameters()).device
            inputs = {k: v.to(current_device) for k, v in inputs.items()}
            
            # Generate with deepseek model
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return generated code (excluding the prompt)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the generated response part
            response_marker = "### Response:"
            if response_marker in generated_text:
                return generated_text.split(response_marker, 1)[1].strip()
            return generated_text
        else:
            # Use the text generator for code generation
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(prompt)
            return self.text_generator.generate(
                initial_str=enhanced_prompt, pred_len=length, temperature=temperature
            )
    
    def train_on_codebase(self, source_path, epochs=50, sequence_length=100, batch_size=64):
        """
        Train the code generator on a specific codebase
        
        Args:
            source_path: Path to codebase (file, directory, or zip)
            epochs: Number of training epochs
            sequence_length: Sequence length for training
            batch_size: Batch size for training
            
        Returns:
            Training loss history
        """
        # Prepare code-specific dataset
        batched_data = self.dataset_processor.prepare_code_dataset(
            source_path, 
            sequence_length=sequence_length, 
            batch_size=batch_size
        )

        return self.text_generator.train(batched_data, epochs=epochs)
    
    def freeze_model_layers(self, freeze_proportion=0.7):
        """
        Freeze a portion of lower transformer layers for efficient fine-tuning
        
        Args:
            freeze_proportion: Proportion of layers to freeze (0-1)
        """
        if not hasattr(self.model, 'base_model'):
            print("Model doesn't have a base_model attribute, skipping layer freezing")
            return
            
        base_model = self.model.base_model.model
        
        # Get layers based on model architecture
        if hasattr(base_model, 'layers'):
            # For models that have direct access to layers
            transformer_layers = base_model.layers
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            # For nested model structures
            transformer_layers = base_model.model.layers
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'layers'):
            # For Llama-type models that use transformer.layers structure
            transformer_layers = base_model.transformer.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Direct access via model.model.layers
            transformer_layers = self.model.model.layers
        elif hasattr(self.model.model, 'transformer') and hasattr(self.model.model.transformer, 'h'):
            # For GPT-2 style models
            transformer_layers = self.model.model.transformer.h
        else:
            # Check if we can find layers directly in base_model
            found_layers_attr = None
            for attr_name in dir(base_model):
                if 'layer' in attr_name.lower() and isinstance(getattr(base_model, attr_name), (list, torch.nn.ModuleList)):
                    found_layers_attr = attr_name
                    break
            
            if found_layers_attr:
                transformer_layers = getattr(base_model, found_layers_attr)
            else:
                print("Could not find transformer layers in model, skipping layer freezing")
                return
        
        num_layers = len(transformer_layers)
        
        # Calculate how many layers to freeze
        num_to_freeze = int(num_layers * freeze_proportion)
        
        print(f"Freezing {num_to_freeze} out of {num_layers} transformer layers")
        
        # Freeze embedding layer if available
        if hasattr(base_model, 'embed_tokens'):
            for param in base_model.embed_tokens.parameters():
                param.requires_grad = False
        elif hasattr(base_model, 'embeddings'):
            for param in base_model.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze lower transformer layers
        for i in range(num_to_freeze):
            for param in transformer_layers[i].parameters():
                param.requires_grad = False
        
        # Count trainable vs frozen parameters
        trainable_params = 0
        frozen_params = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
                
        total_params = trainable_params + frozen_params
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})")
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.1%})")
        print(f"Total parameters: {total_params:,}")

    def fine_tune_deepseek(self, train_dataset=None, eval_dataset=None, output_dir="deepseek_fine-tuned", 
                          epochs=50, batch_size=2, sequence_length=2048, learning_rate=2e-5,
                          warmup_steps=100, max_samples=None, subset="all", all_subsets=True):
        """
        Fine-tune the deepseek-coder model on code snippets
        
        Args:
            train_dataset: Optional pre-loaded training dataset
            eval_dataset: Optional pre-loaded evaluation dataset
            output_dir: Directory to save fine-tuned model
            epochs: Number of fine-tuning epochs
            batch_size: Batch size for training (per device)
            sequence_length: Maximum sequence length
            learning_rate: Learning rate for fine-tuning
            warmup_steps: Number of warmup steps
            max_samples: Maximum number of samples to use (if loading dataset)
            subset: Language subset of code_search_net (if loading dataset)
            all_subsets: Whether to use all language subsets (default: True)
            
        Returns:
            Dictionary with training metrics
        """
        if not self.use_deepseek:
            raise ValueError("This method requires use_deepseek=True. Please initialize CodeGenerator with use_deepseek=True")

        # Set up checkpointing directories
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create a checkpoint log file
        checkpoint_log = os.path.join(output_dir, "checkpoint_log.txt")
        with open(checkpoint_log, 'w') as f:
            f.write(f"Fine-tuning started at: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Batch size: {batch_size}, Sequence length: {sequence_length}, Learning rate: {learning_rate}\n")
            f.write(f"Gradient accumulation steps: 8\n\n")

        # Checkpoint function to log progress
        def log_checkpoint(message):
            with open(checkpoint_log, 'a') as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"[{timestamp}] {message}\n")
            print(f"CHECKPOINT: {message}")

        start_time = time.time()
        log_checkpoint("Starting fine-tuning process")

        # Load dataset if not provided
        if train_dataset is None or eval_dataset is None:
            log_checkpoint("Loading and preprocessing code dataset...")
            train_dataset, eval_dataset = load_and_preprocess_dataset(
                max_samples=max_samples,
                sequence_length=sequence_length,
                subset=subset,
                all_subsets=all_subsets
            )
            log_checkpoint(f"Dataset loaded - Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

        if train_dataset is None or eval_dataset is None:
            log_checkpoint("FAILED: Could not load datasets for fine-tuning")
            raise ValueError("Failed to load datasets for fine-tuning")

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # For Apple Silicon (MPS), use a simplified training approach without PEFT
        if self.device.type == "mps":
            log_checkpoint("Using simplified training for Apple Silicon...")
            return self._fine_tune_mps(
                train_dataset, eval_dataset, output_dir, 
                epochs, batch_size, learning_rate, warmup_steps
            )
        
        # Freeze lower layers for more efficient training
        log_checkpoint("Freezing lower layers for more efficient training...")
        self.freeze_model_layers(freeze_proportion=0.7)  # Freeze 70% of the layers
        
        # Prepare training arguments
        log_checkpoint("Configuring training arguments...")
        # Check if we're on MPS (Apple Silicon) - fp16 is not compatible with MPS
        use_fp16 = self.device.type != "mps"
        if not use_fp16:
            log_checkpoint("Disabling mixed precision training (fp16) as it's not supported on MPS devices")
        
        # On MPS, use even smaller batch size and disable gradient checkpointing
        if self.device.type == "mps":
            batch_size = min(2, batch_size)  # Respect 2 as the specified batch size
            use_gradient_checkpointing = False
            log_checkpoint(f"Using batch size of {batch_size} for MPS device")
            log_checkpoint("Disabling gradient checkpointing for MPS device")
        else:
            use_gradient_checkpointing = True
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            fp16=use_fp16,
            report_to="none",
            gradient_accumulation_steps=8,
            gradient_checkpointing=use_gradient_checkpointing,
            local_rank=-1,
            use_cpu=self.device.type == "cpu",
            save_safetensors=True,
            dataloader_pin_memory=True,  # Requires data to be on CPU
            dataloader_num_workers=2,    # For multiprocessing
            group_by_length=True,        # Keep existing parameter
        )

        # Initialize trainer
        log_checkpoint("Initializing Trainer...")
        
        # Special handling for MPS device
        if self.device.type == "mps":
            log_checkpoint("Using Apple Silicon GPU. Ensuring model is properly configured...")
            # Ensure the model is on the MPS device
            if not next(self.model.parameters()).is_meta:
                log_checkpoint("Moving model to MPS before training...")
                self.model = self.model.to("mps")
                
            # Use smaller batch size for MPS if needed
            if batch_size > 32:
                log_checkpoint(f"Reducing batch size from {batch_size} to 32 for MPS device")
                training_args.per_device_train_batch_size = 32
                training_args.per_device_eval_batch_size = 32
        
        # Initialize the trainer with our model and datasets
        try:
            # Define a custom callback for additional checkpointing
            class CheckpointCallback(TrainerCallback):
                def __init__(self, log_func):
                    self.log_func = log_func
                
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step % 50 == 0:  # Log every 50 steps
                        self.log_func(f"Completed step {state.global_step}, Loss: {state.log_history[-1]['loss'] if state.log_history else 'N/A'}")
                
                def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                    if metrics:
                        self.log_func(f"Evaluation - Step: {state.global_step}, Loss: {metrics.get('eval_loss', 'N/A')}")
                
                def on_save(self, args, state, control, **kwargs):
                    self.log_func(f"Saved checkpoint at step {state.global_step}")
                    
                def on_epoch_end(self, args, state, control, **kwargs):
                    self.log_func(f"Completed epoch {state.epoch}")
                    
                def on_train_begin(self, args, state, control, **kwargs):
                    self.log_func("Training started")
                    
                def on_train_end(self, args, state, control, **kwargs):
                    self.log_func(f"Training completed after {state.global_step} steps")
            
            checkpoint_callback = CheckpointCallback(log_checkpoint)
                        
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[checkpoint_callback],
            )
        
            # Fine-tune the model and track metrics
            log_checkpoint(f"Starting fine-tuning for {epochs} epochs...")
            training_metrics = {}
            training_output = trainer.train()

            # Calculate training time
            training_time = time.time() - start_time
            log_checkpoint(f"Training completed in {training_time:.2f} seconds")

            # Collect metrics
            dataset_name = "code_search_net_all_languages" if all_subsets else f"code_search_net_{subset}"
            if max_samples:
                dataset_name += f"_{max_samples}"
                
            training_metrics = {
                'model_name': self.model_name,
                'dataset': dataset_name,
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
                'epochs': epochs,
                'batch_size': batch_size,
                'sequence_length': sequence_length,
                'gradient_accumulation_steps': 8,
                'effective_batch_size': batch_size * 8,  # batch_size * gradient_accumulation_steps
                'training_time': training_time,
                'learning_rate': learning_rate,
                'final_loss': training_output.training_loss,
                'timestamp': datetime.datetime.now().isoformat(),
                'checkpoints': trainer.state.log_history,
                'lora_config': {
                    'r': self.model.peft_config.r,
                    'lora_alpha': self.model.peft_config.lora_alpha,
                    'target_modules': self.model.peft_config.target_modules,
                }
            }

            log_checkpoint(f"Final loss: {training_output.training_loss:.4f}")

            # Evaluate the model
            log_checkpoint("Evaluating fine-tuned model...")
            eval_metrics = trainer.evaluate()
            training_metrics['eval_loss'] = eval_metrics.get('eval_loss')
            training_metrics['perplexity'] = np.exp(eval_metrics.get('eval_loss', 0))

            log_checkpoint(f"Evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")
            log_checkpoint(f"Perplexity: {training_metrics['perplexity']:.4f}")

            # Save the fine-tuned model
            log_checkpoint(f"Saving fine-tuned model to {output_dir}...")
            
            # Move model to CPU for saving if it's on MPS
            if self.device.type == "mps":
                log_checkpoint("Moving model to CPU for saving...")
                model_to_save = self.model.to("cpu")
                model_to_save.save_pretrained(output_dir)
                self.model = self.model.to("mps")  # Move back to MPS
            else:
                self.model.save_pretrained(output_dir)
                
            self.tokenizer.save_pretrained(output_dir)

            # Save training metrics
            metrics_file = self._save_training_metrics(training_metrics, "deepseek")
            log_checkpoint(f"Training metrics saved to {metrics_file}")
            
            # Create a completion marker file to indicate successful training
            with open(os.path.join(output_dir, "TRAINING_COMPLETE"), 'w') as f:
                f.write(f"Training completed successfully at {datetime.datetime.now().isoformat()}\n")
                f.write(f"Final loss: {training_output.training_loss:.4f}\n")
                f.write(f"Evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}\n")
                f.write(f"Perplexity: {training_metrics['perplexity']:.4f}\n")
            
            return training_metrics
            
        except Exception as e:
            error_msg = f"Error during fine-tuning: {e}"
            log_checkpoint(error_msg)
            import traceback
            traceback_str = traceback.format_exc()
            log_checkpoint(f"Traceback: {traceback_str}")

            # Still try to save partial metrics
            training_metrics = {
                'model_name': self.model_name,
                'error': str(e),
                'traceback': traceback_str,
                'timestamp': datetime.datetime.now().isoformat()
            }
            metrics_file = self._save_training_metrics(training_metrics, "deepseek_error")
            log_checkpoint(f"Error metrics saved to {metrics_file}")
            
            return training_metrics
    
    def _save_training_metrics(self, metrics, model_type="deepseek"):
        """Save training metrics to a JSON file"""
        metrics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(metrics_dir, f"{model_type}_training_{timestamp}.json")
        
        # Ensure all numeric values are converted to float for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                serializable_metrics[key] = [float(x) for x in value]
            elif isinstance(value, dict):
                serializable_metrics[key] = {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in value.items()
                }
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        print(f"Training metrics saved to {metrics_path}")
        return metrics_path
        
    def _fine_tune_mps(self, train_dataset, eval_dataset, output_dir, 
                      epochs, batch_size, learning_rate, warmup_steps):
        """Simplified fine-tuning approach for Apple Silicon MPS devices
        
        This method bypasses PEFT/LoRA and uses a simpler training approach that's compatible
        with Apple Silicon's MPS backend.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import os
        
        print("Setting up memory-efficient training for Apple Silicon MPS...")
        
        # Use extremely small batch size for MPS to avoid memory issues
        batch_size = 1  # Always use batch size of 1 for MPS to avoid OOM
        print(f"Using batch size of {batch_size} for MPS training (enforced to avoid memory issues)")
        
        # Create model save directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check current device
        print(f"Current device: {next(self.model.parameters()).device}")
        
        # Empty cache before starting
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Create data loaders with very small batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True  # Pin memory for faster transfers
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True  # Pin memory for faster transfers
        )
        
        # Initialize optimizer with lower learning rate for stability
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate * 0.1,  # Lower learning rate for stability
            weight_decay=0.01,
            eps=1e-8  # Increase epsilon for numerical stability
        )
        
        # Shorter training for quick completion
        actual_epochs = min(epochs, 3)
        if actual_epochs < epochs:
            print(f"Reducing epochs from {epochs} to {actual_epochs} for Apple Silicon")
        
        total_steps = len(train_loader) * actual_epochs
        warmup_steps = min(warmup_steps, int(total_steps * 0.1))
        
        # Learning rate scheduler with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function - causal language modeling loss
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initial evaluation metrics
        best_eval_loss = float('inf')
        training_metrics = {
            'model_name': self.model_name,
            'train_samples': len(train_dataset),
            'eval_samples': len(eval_dataset),
            'epochs': actual_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate * 0.1,
            'device': 'mps',
            'training_loss': [],
            'eval_loss': [],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        try:
            for epoch in range(actual_epochs):
                # Training
                self.model.train()
                total_train_loss = 0
                step = 0
                
                device = "mps"
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{actual_epochs}"):
                    # Empty cache before each batch
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        
                    # Move data to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Calculate loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Cast both to same type to avoid mismatch
                    shift_logits = shift_logits.to(torch.float32)
                    
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to prevent instability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_train_loss += loss.item()
                    step += 1
                    
                    # Empty cache after each batch
                    if step % 2 == 0 and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                    # Log every 10 steps
                    if step % 10 == 0:
                        print(f"Epoch {epoch+1}, Step {step}/{len(train_loader)}: Loss = {loss.item():.4f}")
                
                avg_train_loss = total_train_loss / len(train_loader)
                training_metrics['training_loss'].append(avg_train_loss)
                print(f"Epoch {epoch+1} completed. Average training loss: {avg_train_loss:.4f}")
                
                # Empty cache before evaluation
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                # Evaluation
                self.model.eval()
                total_eval_loss = 0
                
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc="Evaluating"):
                        # Process on appropriate device
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        
                        # Forward pass
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        
                        # Calculate loss
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        
                        # Cast to same type
                        shift_logits = shift_logits.to(torch.float32)
                        
                        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        total_eval_loss += loss.item()
                
                avg_eval_loss = total_eval_loss / len(eval_loader)
                training_metrics['eval_loss'].append(avg_eval_loss)
                print(f"Evaluation loss: {avg_eval_loss:.4f}")
                
                # Empty cache after evaluation
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                # Save model if it's the best so far
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    print(f"New best model found! Saving to {output_dir}")
                    
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create full paths for model files
                    model_path = os.path.join(output_dir, "best_model.pt")
                    
                    # Save model to output directory - first move to CPU for reliable saving
                    save_device = "cpu"
                    print("Moving model to CPU for saving...")
                    model_to_save = self.model.to(save_device)
                    torch.save(model_to_save.state_dict(), model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Move back to original device
                    print("Moving model back to MPS...")
                    self.model = self.model.to(device)
                    
                    # Also save tokenizer
                    self.tokenizer.save_pretrained(output_dir)
            
            # Training completed
            training_time = time.time() - start_time
            
            # Update metrics with final values
            training_metrics['training_time'] = training_time
            training_metrics['final_loss'] = training_metrics['training_loss'][-1]
            training_metrics['best_eval_loss'] = best_eval_loss
            training_metrics['perplexity'] = np.exp(best_eval_loss)
            
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best evaluation loss: {best_eval_loss:.4f}")
            print(f"Perplexity: {np.exp(best_eval_loss):.4f}")
            
            # Save training metrics
            self._save_training_metrics(training_metrics, "deepseek_mps")
            
            # Empty cache for good measure
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            return training_metrics
            
        except Exception as e:
            print(f"Error during MPS training: {e}")
            import traceback
            traceback.print_exc()
            
            # Save partial metrics
            training_metrics['error'] = str(e)
            self._save_training_metrics(training_metrics, "deepseek_mps_error")
            
            return training_metrics
        
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-Coder on code datasets")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to use (None for all)")
    parser.add_argument("--output-dir", type=str, default="models/deepseek_finetuned", help="Output directory")
    parser.add_argument("--sequence-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--subset", type=str, default="python", help="Code dataset subset")
    
    args = parser.parse_args()

    # Load and preprocess dataset
    train_data, valid_data = load_and_preprocess_dataset(
        max_samples=args.max_samples,
        sequence_length=args.sequence_length,
        subset=args.subset
    )

    # Initialize code generator with deepseek
    code_gen = CodeGenerator(use_deepseek=True)
    
    # Fine-tune the model
    code_gen.fine_tune_deepseek(
        train_dataset=train_data, 
        eval_dataset=valid_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate
    )
