#!/bin/bash

echo "======================================================================"
echo "🔧 Fixing train_models.py for Paperspace"
echo "======================================================================"

# Function to check if a command succeeded
check_success() {
  if [ $? -ne 0 ]; then
    echo "❌ Error: $1 failed"
    exit 1
  else
    echo "✅ $1 successful"
  fi
}

# Create a backup of the original file
echo "Creating backup of train_models.py..."
cp -f /notebooks/src/generative_ai_module/train_models.py /notebooks/src/generative_ai_module/train_models.py.bak
check_success "Backup creation"

# Fix the import issue
echo "Fixing import issue in train_models.py..."
sed -i '1,50s/from .utils import get_storage_path, sync_to_gdrive, sync_logs, setup_logging, ensure_directory_exists, sync_from_gdrive/from .utils import get_storage_path, sync_to_gdrive, sync_logs, setup_logging, ensure_directory_exists, sync_from_gdrive, is_paperspace_environment/' /notebooks/src/generative_ai_module/train_models.py
check_success "Import fix"

# Fix any hardcoded paths
echo "Fixing any hardcoded paths..."
sed -i 's|/Users/[a-zA-Z0-9_]*/Documents/GitHub/Jarvis-AI-Assistant|/notebooks|g' /notebooks/src/generative_ai_module/train_models.py
check_success "Path fix"

# Make sure CUDA is available
echo "Setting up CUDA environment..."
export CUDA_VISIBLE_DEVICES=0
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
check_success "CUDA setup"

# Test if the fix was successful
echo "Testing fix..."
cd /notebooks
python -c "
import sys
try:
    sys.path.insert(0, '/notebooks')
    from src.generative_ai_module.utils import is_paperspace_environment
    print(f'is_paperspace_environment(): {is_paperspace_environment()}')
    print('✅ Utils import successful')
    
    # Test train_models imports
    from src.generative_ai_module.train_models import main
    print('✅ train_models imports successful')
    
    print('\\n✅ All fixes applied successfully!')
except Exception as e:
    print(f'❌ Error: {str(e)}')
    sys.exit(1)
"

check_success "Import test"

echo ""
echo "======================================================================"
echo "✅ train_models.py has been fixed for Paperspace!"
echo ""
echo "You can now run your training command:"
echo ""
echo "python src/generative_ai_module/train_models.py \\"
echo "    --model-type code \\"
echo "    --use-deepseek \\"
echo "    --code-subset \$CODE_SUBSET \\"
echo "    --batch-size 4 \\"
echo "    --epochs 3 \\"
echo "    --learning-rate 2e-5 \\"
echo "    --warmup-steps 100 \\"
echo "    --load-in-4bit"
echo ""
echo "======================================================================"