from email.headerregistry import DateHeader
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
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
from torch.utils.data import DataLoader
from .utils import setup_gpu_for_training, force_cuda_device, is_paperspace_environment

# Environment variable check
import os
if os.environ.get('FORCE_CPU_DATA_PIPELINE', '0') == '1':
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')

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
            self.text_generator = TextGenerator(force_gpu=self.force_gpu)
            self.prompt_enhancer = PromptEnhancer()
            self.dataset_processor = DatasetProcessor(self.text_generator)

    def _get_device(self):
        """Determine the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
        # Check for CPU-only mode for initial loading
        if os.environ.get('FORCE_CPU_ONLY_FOR_INITIAL_LOAD') == '1':
            print("FORCE_CPU_ONLY_FOR_INITIAL_LOAD is set - using CPU for initial model loading")
            # Store the actual target device for later use
            if torch.cuda.is_available():
                self._target_device = torch.device("cuda")
                print("Target device for training will be CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._target_device = torch.device("mps")
                print("Target device for training will be MPS")
            else:
                self._target_device = torch.device("cpu")
                print("Target device for training will be CPU")
            return torch.device("cpu")

        if os.environ.get('FORCE_CPU_DATA_PIPELINE') == '1':
            return torch.device("cpu")

        # Force GPU usage only when not in CPU mode
        self.force_gpu = True

        print("Setting up GPU for all operations...")

        try:
            device, gpu_config = setup_gpu_for_training(force_gpu=True)
            self.gpu_config = gpu_config

            if device.type == "cuda":
                # RTX5000 configurations
                if is_paperspace_environment() and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    if "RTX5000" in gpu_name or "RTX 5000" in gpu_name:
                        print("RTX 5000 GPU detected - applying optimized settings")
                        self.load_in_4bit = True
                        self.load_in_8bit = False

                # Conditional CUDA setup
                if os.environ.get('FORCE_CPU_DATA_PIPELINE') != '1':
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')
                    if hasattr(torch, 'set_default_device'):
                        torch.set_default_device('cuda')
                    print(f"GPU enforcement successful: Using CUDA device {device}")

                return device

            elif device.type == "mps":
                print("Using Apple Silicon MPS device")
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                return device

            else:
                print("Warning: setup_gpu_for_training returned CPU despite force_gpu=True")

        except Exception as e:
            print(f"GPU setup error: {e}, trying fallback detection")

        # Fallback GPU detection
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using MPS device")
            return torch.device("mps")
        else:
            print("Warning: Falling back to CPU")
            return torch.device("cpu")

    def load_model(self):
        """Load the deepseek model with appropriate quantization settings"""
        if getattr(self, '_loading', False):  # Prevent recursion
            return
        self._loading = True
        try:
            print(f"Loading {self.model_name}...")

            # Force garbage collection and clear CUDA cache before loading model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Set memory fraction to avoid OOM
                torch.cuda.set_per_process_memory_fraction(0.9)
                print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
                print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

            # Check for CPU-only initial loading mode
            if os.environ.get('FORCE_CPU_ONLY_FOR_INITIAL_LOAD') == '1':
                print("Using CPU-only for initial model loading (FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1)")
                try:
                    # Load model on CPU first
                    print("Loading model on CPU first...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        device_map="cpu",           # Force CPU
                        low_cpu_mem_usage=True      # Reduce CPU memory usage
                    )

                    print("Model loaded on CPU successfully")

                    # If GPU is available, we'll transfer parts of the model later
                    # after training setup is complete
                    if torch.cuda.is_available():
                        print("GPU is available - will transfer critical parts to GPU during training")
                        # We'll keep the model on CPU for now and transfer parts during training
                        return
                    else:
                        print("No GPU available - keeping model on CPU")
                        return
                except Exception as e:
                    print(f"Error loading model on CPU: {e}")
                    print("Falling back to standard loading approach")

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
                    # For CUDA devices (replace existing 4-bit block)
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,  # Force FP16 compute
                        bnb_4bit_use_double_quant=True,        # Second quantization for 4-bit
                        bnb_4bit_quant_type="nf4",             # Optimal quantization type
                        llm_int8_skip_modules=["lm_head"],     # Keep lm_head in FP16
                    )

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        device_map="auto",
                        max_memory={0: "13GiB", "cpu": "12GiB"},  # Reduced GPU allocation
                        quantization_config=quantization_config,
                        use_flash_attention_2=True,  # Must be enabled
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16  # Match compute dtype
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
                print("Optimized 4-bit loading for RTX A4000")
                try:
                    # More aggressive memory management for RTX A4000
                    from transformers import BitsAndBytesConfig

                    # Check available GPU memory and adjust settings accordingly
                    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"Total GPU memory: {total_gpu_memory:.2f} GB")

                    # More aggressive quantization for limited memory
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,  # Double quantization for more memory savings
                        bnb_4bit_quant_type="nf4",       # More efficient quantization type
                        llm_int8_skip_modules=["lm_head"],
                        bnb_4bit_quant_storage=torch.float16  # Use float16 for storage
                    )

                    # Set environment variables for better memory management
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

                    # Offload more aggressively to CPU
                    device_map = {
                        "model.embed_tokens": 0,
                        "model.norm": 0,
                        "lm_head": 0,
                        # Offload some layers to CPU to save GPU memory
                        "model.layers.0": 0,
                        "model.layers.1": 0,
                        "model.layers.2": "cpu",
                        "model.layers.3": "cpu",
                        "model.layers.4": 0,
                        "model.layers.5": 0,
                        "model.layers.6": "cpu",
                        "model.layers.7": "cpu",
                        "model.layers.8": 0,
                        "model.layers.9": 0,
                        "model.layers.10": "cpu",
                        "model.layers.11": "cpu",
                        "model.layers.12": 0,
                        "model.layers.13": 0,
                        "model.layers.14": "cpu",
                        "model.layers.15": "cpu",
                        "model.layers.16": 0,
                        "model.layers.17": 0,
                        "model.layers.18": "cpu",
                        "model.layers.19": "cpu",
                        "model.layers.20": 0,
                        "model.layers.21": 0,
                        "model.layers.22": "cpu",
                        "model.layers.23": "cpu",
                        "model.layers.24": 0,
                        "model.layers.25": 0,
                        "model.layers.26": "cpu",
                        "model.layers.27": "cpu",
                        "model.layers.28": 0,
                        "model.layers.29": 0,
                        "model.layers.30": "cpu",
                        "model.layers.31": "cpu",
                    }

                    print("Loading model with custom device map and aggressive memory optimization...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device_map,  # Use custom device map instead of "auto"
                        max_memory={0: "12GiB", "cpu": "24GiB"},  # Reduced GPU allocation, more CPU
                        quantization_config=quantization_config,
                        offload_folder="offload_folder",  # Enable disk offloading if needed
                        offload_state_dict=True,  # Offload state dict during loading
                        low_cpu_mem_usage=True
                    )

                    # Memory verification
                    mem_used = torch.cuda.memory_allocated() / (1024**3)
                    print(f"Model loaded using {mem_used:.2f} GB / {total_gpu_memory:.2f} GB VRAM")

                except Exception as e:
                    print(f"4-bit load with custom device map failed: {e}")
                    print("Attempting simplified 4-bit loading...")
                    try:
                        # Try with simpler settings but still with aggressive memory optimization
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )

                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            device_map="auto",
                            max_memory={0: "12GiB", "cpu": "24GiB"},
                            quantization_config=quantization_config,
                            offload_state_dict=True,
                            low_cpu_mem_usage=True
                        )

                        print("Successfully loaded model with simplified 4-bit settings")
                    except Exception as e2:
                        print(f"Simplified 4-bit loading failed: {e2}")
                        print("Attempting basic 4-bit loading...")
                        try:
                            # Last resort: basic 4-bit loading
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                trust_remote_code=True,
                                load_in_4bit=True,
                                device_map="auto"
                            )
                            print("Successfully loaded model with basic 4-bit settings")
                        except Exception as e3:
                            print(f"Basic 4-bit loading failed: {e3}")
                            print("Falling back to 8-bit quantization...")
                            try:
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    self.model_name,
                                    trust_remote_code=True,
                                    load_in_8bit=True,
                                    device_map="auto"
                                )
                                print("Successfully loaded model with 8-bit quantization")
                            except Exception as e4:
                                print(f"8-bit loading failed: {e4}")
                                raise RuntimeError("Failed to load model with any quantization method")
            elif self.load_in_8bit and self.device.type == "cuda":
                print("Loading model in 8-bit quantization")
                try:
                    # Try with custom device map for 8-bit quantization
                    device_map = {
                        "model.embed_tokens": 0,
                        "model.norm": 0,
                        "lm_head": 0,
                        # Alternate between GPU and CPU for layers
                        "model.layers.0": 0,
                        "model.layers.1": 0,
                        "model.layers.2": "cpu",
                        "model.layers.3": "cpu",
                        "model.layers.4": 0,
                        "model.layers.5": 0,
                        "model.layers.6": "cpu",
                        "model.layers.7": "cpu",
                        "model.layers.8": 0,
                        "model.layers.9": 0,
                        "model.layers.10": "cpu",
                        "model.layers.11": "cpu",
                        "model.layers.12": 0,
                        "model.layers.13": 0,
                        "model.layers.14": "cpu",
                        "model.layers.15": "cpu",
                        "model.layers.16": 0,
                        "model.layers.17": 0,
                        "model.layers.18": "cpu",
                        "model.layers.19": "cpu",
                        "model.layers.20": 0,
                        "model.layers.21": 0,
                        "model.layers.22": "cpu",
                        "model.layers.23": "cpu",
                        "model.layers.24": 0,
                        "model.layers.25": 0,
                        "model.layers.26": "cpu",
                        "model.layers.27": "cpu",
                        "model.layers.28": 0,
                        "model.layers.29": 0,
                        "model.layers.30": "cpu",
                        "model.layers.31": "cpu",
                    }

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        load_in_8bit=True,
                        max_memory={0: "12GiB", "cpu": "24GiB"},
                        offload_state_dict=True,
                        low_cpu_mem_usage=True
                    )
                    print("Successfully loaded model with 8-bit quantization and custom device map")
                except Exception as e:
                    print(f"8-bit loading with custom device map failed: {e}")
                    print("Attempting basic 8-bit loading...")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            load_in_8bit=True,
                            max_memory={0: "12GiB", "cpu": "24GiB"},
                            offload_state_dict=True
                        )
                        print("Successfully loaded model with basic 8-bit quantization")
                    except Exception as e2:
                        print(f"Basic 8-bit loading failed: {e2}")
                        print("Falling back to CPU loading...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            device_map="cpu",
                            torch_dtype=torch.float32
                        )
                        print("Model loaded on CPU. Performance will be slow.")
                        self.device = torch.device("cpu")
            else:
                print("Loading model with auto device map and memory optimization")
                try:
                    # Try to load with auto device map and memory optimization
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        max_memory={0: "12GiB", "cpu": "24GiB"},
                        offload_state_dict=True,
                        low_cpu_mem_usage=True
                    )
                    print("Successfully loaded model with auto device map")
                except Exception as e:
                    print(f"Auto device map loading failed: {e}")
                    print("Falling back to CPU loading...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        device_map="cpu",
                        torch_dtype=torch.float32
                    )
                    print("Model loaded on CPU. Performance will be slow.")
                    self.device = torch.device("cpu")

            print(f"\n=== Model Load Verification ===")
            print(f"Model type: {type(self.model)}")

            # Apply LoRA - only for non-MPS devices
            if self.device.type != "mps":
                lora_config = LoraConfig(
                    r=64,  # Increased from 16 → better adaptation for code tasks
                    lora_alpha=64,  # Keep alpha=r for simplicity
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"  # Added FFN layers
                    ],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    modules_to_save=["lm_head"]  # Keep lm_head trainable
                )

                print("Applying LoRA adapters to model...")
                try:
                    self.model = get_peft_model(self.model, lora_config)
                    print("Model ready with LoRA adapters")
                except Exception as e:
                    print(f"Error applying LoRA adapters: {e}")
                    print("Continuing with base model without LoRA")

            if self.device.type == "cuda":
                # Verify memory usage
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                if mem_alloc > 14.5:
                    raise RuntimeError(
                        f"Model requires {mem_alloc:.1f}GB/16GB VRAM. "
                        "Reduce --max_length or enable --gradient_checkpointing"
                    )

                # Warm up GPU memory
                try:
                    dummy_input = torch.zeros((1, 64), dtype=torch.long, device="cuda")
                    _ = self.model(dummy_input)
                    # Force release unused memory
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Limit PyTorch's reserved memory
                    torch.cuda.set_per_process_memory_fraction(0.85)  # 85% of 16GB = 13.6GB
                except Exception as e:
                    print(f"GPU warmup failed: {e}")
        finally:
            self._loading = False

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
        print(f"Attempting to freeze {freeze_proportion:.1%} of model layers...")

        # First, try to identify the model structure
        transformer_layers = None

        # Print model structure for debugging
        print(f"Model type: {type(self.model).__name__}")

        # For LLaMA models
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            print("Found layers in model.model.layers (LLaMA structure)")
            transformer_layers = self.model.model.layers

        # For models with base_model attribute
        elif hasattr(self.model, 'base_model'):
            print("Model has base_model attribute")
            base_model = self.model.base_model

            # Try different paths to find layers
            if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                print("Found layers in base_model.model.layers")
                transformer_layers = base_model.model.layers

            elif hasattr(base_model, 'layers'):
                print("Found layers in base_model.layers")
                transformer_layers = base_model.layers

            elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'layers'):
                print("Found layers in base_model.transformer.layers")
                transformer_layers = base_model.transformer.layers

        # For GPT-2 style models
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            print("Found layers in model.transformer.h (GPT-2 style)")
            transformer_layers = self.model.transformer.h

        # Direct layers
        elif hasattr(self.model, 'layers'):
            print("Found layers directly in model.layers")
            transformer_layers = self.model.layers

        # If we still haven't found layers, skip freezing
        if transformer_layers is None:
            print("Could not identify transformer layers in model, skipping layer freezing")
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

    def move_model_to_target_device(self):
        """
        Move the model from CPU to the target device (GPU/MPS) for training.
        This should be called right before training starts to ensure all model parts
        are on the same device.
        """
        # If we're already using the target device, no need to move
        if not hasattr(self, '_target_device'):
            print("No target device specified, using current device")
            if torch.cuda.is_available():
                self._target_device = torch.device("cuda")
                print(f"Setting target device to CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._target_device = torch.device("mps")
                print(f"Setting target device to MPS")
            else:
                self._target_device = torch.device("cpu")
                print(f"Setting target device to CPU")

        # Get current device of model
        current_device = next(self.model.parameters()).device

        # If already on target device, no need to move
        if current_device == self._target_device:
            print(f"Model already on target device: {self._target_device}")
            return

        print(f"Moving model from {current_device} to {self._target_device} for training...")

        # CRITICAL FIX: Patch the problematic function in transformers library
        # Do this regardless of target device to ensure consistent behavior
        self._patch_transformers_attention_mask()

        # Check available GPU memory if moving to CUDA
        if self._target_device.type == "cuda" and torch.cuda.is_available():
            # Aggressively clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

            # Check available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            free_memory = total_memory - allocated_memory

            print(f"GPU memory: Total={total_memory:.2f}GB, Used={allocated_memory:.2f}GB, Free={free_memory:.2f}GB")

            # If free memory is less than 2GB, don't try to move to GPU
            if free_memory < 2.0:
                print(f"WARNING: Not enough GPU memory available ({free_memory:.2f}GB). Keeping model on CPU.")
                self._target_device = torch.device("cpu")
                self.device = torch.device("cpu")
                return

        # For MPS device (Apple Silicon)
        if self._target_device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            gc.collect()

        # Move model to target device
        try:
            # For PEFT models with LoRA adapters, we can keep the base model on CPU
            # and only move the LoRA adapters to GPU to save memory
            if hasattr(self.model, 'base_model') and hasattr(self.model, 'peft_config'):
                print("Using memory-efficient approach: keeping base model on CPU and moving only LoRA adapters to GPU")

                # Only move LoRA adapter parameters to GPU
                lora_params_moved = 0
                for name, param in self.model.named_parameters():
                    if 'lora' in name and not param.data.is_meta:
                        try:
                            param.data = param.data.to(self._target_device)
                            lora_params_moved += 1
                        except Exception as e:
                            print(f"Warning: Could not move {name} to {self._target_device}: {e}")

                print(f"Moved {lora_params_moved} LoRA parameters to {self._target_device}")

                # Keep track of the fact that we're using a mixed device approach
                self._using_mixed_devices = True

                # Update device attribute but note that it's a mixed setup
                self.device = self._target_device
                print(f"Using mixed device setup: LoRA adapters on {self._target_device}, base model on CPU")
            else:
                # For non-PEFT models, try to move the entire model
                # This might fail due to memory constraints
                try:
                    print(f"Attempting to move entire model to {self._target_device}")
                    self.model = self.model.to(self._target_device)
                    self.device = self._target_device
                    print(f"Successfully moved entire model to {self._target_device}")
                except Exception as e:
                    print(f"Error moving entire model: {e}")
                    print(f"Falling back to CPU for model")
                    self._target_device = torch.device("cpu")
                    self.device = torch.device("cpu")

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        except Exception as e:
            print(f"Error during device transition: {e}")
            print("Falling back to CPU for all operations")
            self._target_device = torch.device("cpu")
            self.device = torch.device("cpu")
            import traceback
            traceback.print_exc()

    def _patch_transformers_attention_mask(self):
        """
        Patch the problematic function in transformers library that's causing device mismatch.
        This specifically targets the _unmask_unattended function that's using .cpu()
        """
        try:
            import transformers.modeling_attn_mask_utils as attn_utils

            # Store the original function
            original_unmask_unattended = attn_utils.AttentionMaskConverter._unmask_unattended

            # Define our patched version that doesn't use .cpu()
            # The issue is with the function signature - we need to match the original function's signature exactly
            # Check the original function signature first
            import inspect
            sig = inspect.signature(original_unmask_unattended)
            print(f"Original function signature: {sig}")

            # Define our patched version with the exact same signature
            def patched_unmask_unattended(self, attention_mask, unmasked_value=0.0):
                """Patched version that doesn't force CPU conversion"""
                # Get the device of the attention mask
                device = attention_mask.device

                # Create a temporary tensor on the same device
                tmp = torch.ones_like(attention_mask) * unmasked_value

                # Use argmax without forcing CPU
                indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

                # Create a range tensor on the same device
                range_tensor = torch.arange(attention_mask.shape[1], device=device).expand_as(attention_mask)

                # Create the expanded mask on the same device
                expanded_mask = (range_tensor <= indices).to(attention_mask.dtype)

                return expanded_mask

            # Apply the patch
            attn_utils.AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
            print("Successfully patched transformers attention mask function")

        except Exception as e:
            print(f"Error patching transformers attention mask function: {e}")
            print("Will continue without patching")

    def fine_tune_deepseek(self, train_dataset=None, eval_dataset=None, output_dir="deepseek_fine-tuned",
                          epochs=50, batch_size=2, sequence_length=2048, learning_rate=2e-5,
                          warmup_steps=100, max_samples=None, subset="all", all_subsets=True,
                          skip_layer_freezing=False):
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
        # Import necessary modules to ensure they're available in this function's scope
        import os
        import datetime
        import time
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

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

        if train_dataset is not None:
            log_checkpoint("Validating dataset device placement...")
            # Force dataset to CPU before creating loader
            try:
                # First try to move tensors to CPU if they're on GPU
                if hasattr(train_dataset, 'set_format'):
                    log_checkpoint("Moving dataset tensors to CPU...")
                    train_dataset.set_format(type="torch", device="cpu")
                if hasattr(eval_dataset, 'set_format'):
                    eval_dataset.set_format(type="torch", device="cpu")

                # Create a small loader to check
                temp_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                sample = next(iter(temp_loader))

                # Check and move tensors to CPU if needed
                for key in sample:
                    if isinstance(sample[key], torch.Tensor) and sample[key].device.type != 'cpu':
                        log_checkpoint(f"Moving {key} tensors from {sample[key].device} to CPU")
                        # This is just for logging - we'll handle the actual conversion in the DataLoader
            except Exception as e:
                log_checkpoint(f"Error during dataset device validation: {e}")
                log_checkpoint("Will force CPU tensors during training")

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

        # Try to freeze lower layers for more efficient training if not skipped
        if skip_layer_freezing:
            log_checkpoint("Skipping layer freezing as requested")
        else:
            log_checkpoint("Attempting to freeze lower layers for more efficient training...")
            try:
                self.freeze_model_layers(freeze_proportion=0.7)  # Freeze 70% of the layers
            except Exception as e:
                log_checkpoint(f"Error freezing layers: {e}. Continuing without layer freezing.")
                # Continue without layer freezing

        # Prepare training arguments
        log_checkpoint("Configuring training arguments...")
        # Check if we're on CUDA - fp16/bf16 is only compatible with CUDA, NPU, or certain XPU devices
        use_fp16 = self.device.type == "cuda"
        if not use_fp16:
            log_checkpoint("Disabling mixed precision training (fp16/bf16) as it's only supported on CUDA, NPU, or certain XPU devices")

        # On MPS, use even smaller batch size and disable gradient checkpointing
        if self.device.type == "mps":
            batch_size = min(2, batch_size)  # Respect 2 as the specified batch size
            use_gradient_checkpointing = False
            log_checkpoint(f"Using batch size of {batch_size} for MPS device")
            log_checkpoint("Disabling gradient checkpointing for MPS device")
        else:
            use_gradient_checkpointing = True

        # Set up checkpoint directory in the specified location
        notebooks_dir = "/notebooks/Jarvis_AI_Assistant"
        checkpoint_dir = os.path.join(notebooks_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Check for existing checkpoints
        resume_from_checkpoint = None
        if os.path.exists(checkpoint_dir):
            log_checkpoint("Checking for existing checkpoints...")
            checkpoint_folders = [
                folder for folder in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, folder)) and folder.startswith("checkpoint-")
            ]

            if checkpoint_folders:
                # Sort by checkpoint number (extract number from folder name)
                checkpoint_folders.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_folders[0])
                log_checkpoint(f"Found existing checkpoint: {latest_checkpoint}")
                resume_from_checkpoint = latest_checkpoint

        # Set up training arguments with the checkpoint directory
        # Create a dictionary of arguments first
        training_args_dict = {
            "output_dir": checkpoint_dir,  # Use the checkpoint directory for saving during training
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "logging_dir": os.path.join(notebooks_dir, "logs"),
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 5,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
            "report_to": "none",
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": use_gradient_checkpointing,
            "local_rank": -1,
            "use_cpu": self.device.type == "cpu",
            "save_safetensors": True,
            "dataloader_pin_memory": True,  # Requires data to be on CPU
            "dataloader_num_workers": 2,    # For multiprocessing
            "group_by_length": True,        # Keep existing parameter
            "remove_unused_columns": False,
            "hub_model_id": None,           # Disable hub pushing
        }

        # Only add fp16/bf16 if we're on a CUDA device
        if use_fp16:
            training_args_dict["fp16"] = True
            # Check if bf16 is requested via environment variable
            if os.environ.get("USE_BF16", "").lower() in ["true", "1", "yes"]:
                training_args_dict["bf16"] = True
                log_checkpoint("Using BF16 mixed precision training")

        # Create the TrainingArguments object
        training_args = TrainingArguments(**training_args_dict)

        # Move model to target device before training
        log_checkpoint("Moving model to target device before training...")
        self.move_model_to_target_device()

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

            # Define DeviceSafeDataLoader class with filtering for unexpected keys
            class DeviceSafeDataLoader(DataLoader):
                def __init__(self, *args, target_device=None, model=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.model = model or self.model  # Use provided model or default
                    # Use the specified target device or default to the model's device
                    self.target_device = target_device
                    log_checkpoint(f"DeviceSafeDataLoader initialized with target device: {self.target_device}")

                def __iter__(self):
                    # List of allowed keys for the model's forward pass
                    allowed_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids', 'position_ids']

                    for batch in super().__iter__():
                        # Filter out unexpected keys and move tensors to the target device
                        filtered_batch = {
                            k: v.to(self.target_device) if isinstance(v, torch.Tensor) and self.target_device is not None else v
                            for k, v in batch.items() if k in allowed_keys
                        }
                        yield filtered_batch

            # Ensure datasets are on CPU
            if hasattr(train_dataset, 'set_format'):
                train_dataset = train_dataset.with_format("torch", device='cpu')
            if hasattr(eval_dataset, 'set_format'):
                eval_dataset = eval_dataset.with_format("torch", device='cpu')

            # Create a simple data collator that works with CPU or GPU
            def memory_efficient_collator(features):
                batch = {}
                # List of allowed keys for the model's forward pass
                allowed_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids', 'position_ids']

                # Get current device - use CPU for collation to save GPU memory
                # The dataloader will move tensors to the right device later
                device = torch.device('cpu')

                for key in features[0].keys():
                    # Skip keys that aren't in the allowed list
                    if key not in allowed_keys:
                        continue

                    if isinstance(features[0][key], torch.Tensor):
                        try:
                            # Stack tensors on CPU
                            batch[key] = torch.stack([f[key] for f in features])
                        except Exception as e:
                            log_checkpoint(f"Error stacking tensors for {key}: {e}")
                            # Try a different approach if stacking fails
                            batch[key] = torch.cat([f[key].unsqueeze(0) for f in features])
                    else:
                        batch[key] = [f[key] for f in features]

                return batch

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[checkpoint_callback],
                data_collator=memory_efficient_collator,
            )

            # Fine-tune the model and track metrics
            log_checkpoint(f"Starting fine-tuning for {epochs} epochs...")

            # Get the current model device for the dataloader
            current_device = next(self.model.parameters()).device
            log_checkpoint(f"Using device for training: {current_device}")

            # Create a custom dataloader that ensures tensors are on the right device
            class MemorySafeDataLoader(DataLoader):
                def __init__(self, dataset, device, **kwargs):
                    super().__init__(dataset, **kwargs)
                    self.device = device
                    log_checkpoint(f"MemorySafeDataLoader initialized with device: {self.device}")

                def __iter__(self):
                    for batch in super().__iter__():
                        # Filter out unexpected keys and move tensors to the right device
                        allowed_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids', 'position_ids']
                        filtered_batch = {}

                        for k, v in batch.items():
                            if k in allowed_keys:
                                if isinstance(v, torch.Tensor):
                                    filtered_batch[k] = v.to(self.device)
                                else:
                                    filtered_batch[k] = v

                        yield filtered_batch

            # Use our memory-safe dataloader
            trainer.train_dataloader = MemorySafeDataLoader(
                trainer.train_dataset,
                device=current_device,
                batch_size=training_args.per_device_train_batch_size,
                shuffle=True,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory
            )
            training_metrics = {}

            # Apply attention mask fix before training
            log_checkpoint("Applying attention mask fix...")
            try:
                # Import and apply the fix
                import sys
                import os

                # Add the current directory to the path to find the setup.fix_attention_mask module
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                if project_root not in sys.path:
                    sys.path.append(project_root)

                # Import and apply the fix
                from setup.fix_attention_mask import patch_llama_model_forward, patch_attention_mask_in_dataset_collator
                patch_llama_model_forward()
                patch_attention_mask_in_dataset_collator()
                log_checkpoint("Successfully applied attention mask fix")
            except Exception as e:
                log_checkpoint(f"Warning: Failed to apply attention mask fix: {e}")
                log_checkpoint("Will attempt training anyway, but may encounter attention mask errors")

            # Resume from checkpoint if available
            if resume_from_checkpoint:
                log_checkpoint(f"Resuming training from checkpoint: {resume_from_checkpoint}")
                training_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                log_checkpoint("Starting training from scratch")
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
                    'r': self.model.peft_config.r if hasattr(self.model, 'peft_config') else None,
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
                try:
                    log_checkpoint("Moving model to CPU for saving...")
                    model_to_save = self.model.to("cpu")
                    model_to_save.save_pretrained(output_dir)
                finally:
                    self.model = self.model.to("mps")  # Move back to MPS
            else:
                self.model.save_pretrained(output_dir)

            self.tokenizer.save_pretrained(output_dir)

            # Save training metrics
            metrics_file = self._save_training_metrics(training_metrics, "deepseek_coder")
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
            metrics_file = self._save_training_metrics(training_metrics, "deepseek_coder_error")
            log_checkpoint(f"Error metrics saved to {metrics_file}")

            return training_metrics

    def _save_training_metrics(self, metrics, model_type="deepseek_coder"):
        """Save training metrics to a JSON file"""
        # Use the specified Jarvis_AI_Assistant directory
        notebooks_dir = "/notebooks/Jarvis_AI_Assistant"
        metrics_dir = os.path.join(notebooks_dir, "metrics")
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
        # Import necessary modules to ensure they're available in this function's scope
        import os
        import datetime
        import time
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import numpy as np

        print("Setting up memory-efficient training for Apple Silicon MPS...")

        self.model.gradient_checkpointing_enable()

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

        # Validate that the model is on the CPU
        sample = next(iter(train_loader))
        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                assert sample[key].device.type == 'cpu', \
                    f"Dataset contains GPU tensors in {key} column!"

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

                    # Fix attention mask shape if needed
                    if attention_mask.dim() == 2:
                        # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                        seq_length = attention_mask.size(1)
                        batch_size = attention_mask.size(0)

                        # First, expand to [batch_size, 1, 1, seq_length]
                        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                        # Then expand to [batch_size, 1, seq_length, seq_length]
                        attention_mask = attention_mask.expand(-1, -1, seq_length, -1)

                        print(f"Fixed attention mask shape: {attention_mask.shape}")

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

                        # Fix attention mask shape if needed
                        if attention_mask.dim() == 2:
                            # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                            seq_length = attention_mask.size(1)
                            batch_size = attention_mask.size(0)

                            # First, expand to [batch_size, 1, 1, seq_length]
                            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                            # Then expand to [batch_size, 1, seq_length, seq_length]
                            attention_mask = attention_mask.expand(-1, -1, seq_length, -1)

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
