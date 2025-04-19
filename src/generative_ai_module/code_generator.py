from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
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
        if self.force_gpu:
            # Try to use MPS (Metal Performance Shaders) for Apple Silicon
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                return torch.device("mps")
            # Fall back to CUDA if available
            elif torch.cuda.is_available():
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            else:
                print("Warning: GPU requested but neither MPS nor CUDA is available. Falling back to CPU.")
                return torch.device("cpu")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            print("Using CPU (no GPU available)")
            return torch.device("cpu")
    
    def load_model(self):
        """Load the deepseek model with appropriate quantization settings"""
        print(f"Loading {self.model_name}...")
        
        # Special case for Apple Silicon (M1/M2/M3)
        if self.device.type == "mps":
            print("Loading model for Apple Silicon GPU")
            # On MPS, we need to first load to CPU, then apply LoRA, then move to MPS
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu"  # First load to CPU to avoid meta tensor issues
            )
        # Set quantization parameters based on user settings (for CUDA devices)
        elif self.load_in_4bit and self.device.type == "cuda":
            print("Loading model in 4-bit quantization for extreme memory saving")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config
            )
        elif self.load_in_8bit and self.device.type == "cuda":
            print("Loading model in 8-bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
            )
        else:
            print("Loading model in full precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        
        # Apply LoRA
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
            
            # Now move to MPS device if needed (after LoRA is applied)
            if self.device.type == "mps":
                print("Moving model to MPS device")
                # Move model to MPS device after LoRA application
                self.model = self.model.to(self.device)
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
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate with deepseek model
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
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
    
    def fine_tune_deepseek(self, train_dataset=None, eval_dataset=None, output_dir="deepseek_fine-tuned", 
                          epochs=50, batch_size=64, sequence_length=512, learning_rate=2e-5,
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

        start_time = time.time()

        # Load dataset if not provided
        if train_dataset is None or eval_dataset is None:
            print("Loading and preprocessing code dataset...")
            train_dataset, eval_dataset = load_and_preprocess_dataset(
                max_samples=max_samples,
                sequence_length=sequence_length,
                subset=subset,
                all_subsets=all_subsets
            )

        if train_dataset is None or eval_dataset is None:
            raise ValueError("Failed to load datasets for fine-tuning")

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare training arguments
        print("Configuring training arguments...")
        # Check if we're on MPS (Apple Silicon) - fp16 is not compatible with MPS
        use_fp16 = self.device.type != "mps"
        if not use_fp16:
            print("Disabling mixed precision training (fp16) as it's not supported on MPS devices")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,  # Keep only the 3 best models
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            fp16=use_fp16,  # Only use fp16 when not on MPS
            report_to="tensorboard",
            gradient_accumulation_steps=4,  # To handle larger effective batch sizes
            gradient_checkpointing=True,  # Memory optimization
            # Disable distributed training on MPS 
            local_rank=-1 if self.device.type == "mps" else training_args.local_rank if hasattr(training_args, 'local_rank') else -1,
            # Special settings for MPS (Apple Silicon)
            no_cuda=self.device.type == "mps",  # Disable CUDA detection when using MPS
        )

        # Initialize trainer
        print("Initializing Trainer...")
        
        # Special handling for MPS device
        if self.device.type == "mps":
            print("Using Apple Silicon GPU. Ensuring model is properly configured...")
            # Ensure the model is on the MPS device
            if not next(self.model.parameters()).is_meta:
                print("Moving model to MPS before training...")
                self.model = self.model.to("mps")
                
            # Use smaller batch size for MPS if needed
            if batch_size > 32:
                print(f"Reducing batch size from {batch_size} to 32 for MPS device")
                training_args.per_device_train_batch_size = 32
                training_args.per_device_eval_batch_size = 32
        
        # Initialize the trainer with our model and datasets
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        
            # Fine-tune the model and track metrics
            print(f"Starting fine-tuning for {epochs} epochs...")
            training_metrics = {}
            training_output = trainer.train()

            # Calculate training time
            training_time = time.time() - start_time

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
                'training_time': training_time,
                'learning_rate': learning_rate,
                'final_loss': training_output.training_loss,
                'timestamp': datetime.datetime.now().isoformat(),
                'lora_config': {
                    'r': self.model.peft_config.r,
                    'lora_alpha': self.model.peft_config.lora_alpha,
                    'target_modules': self.model.peft_config.target_modules,
                }
            }

            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Final loss: {training_output.training_loss:.4f}")

            # Evaluate the model
            print("Evaluating fine-tuned model...")
            eval_metrics = trainer.evaluate()
            training_metrics['eval_loss'] = eval_metrics.get('eval_loss')
            training_metrics['perplexity'] = np.exp(eval_metrics.get('eval_loss', 0))

            print(f"Evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")
            print(f"Perplexity: {training_metrics['perplexity']:.4f}")

            # Save the fine-tuned model
            print(f"Saving fine-tuned model to {output_dir}...")
            
            # Move model to CPU for saving if it's on MPS
            if self.device.type == "mps":
                print("Moving model to CPU for saving...")
                model_to_save = self.model.to("cpu")
                model_to_save.save_pretrained(output_dir)
                self.model = self.model.to("mps")  # Move back to MPS
            else:
                self.model.save_pretrained(output_dir)
                
            self.tokenizer.save_pretrained(output_dir)

            # Save training metrics
            self._save_training_metrics(training_metrics, "deepseek")
            
            return training_metrics
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            import traceback
            traceback.print_exc()

            # Still try to save partial metrics
            training_metrics = {
                'model_name': self.model_name,
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }
            self._save_training_metrics(training_metrics, "deepseek_error")
            
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
