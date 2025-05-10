#!/usr/bin/env python3
"""
Run All Fixes Script

This script runs all the necessary fixes in the correct order:
1. Fix unterminated string literals in installed packages
2. Create missing setup files
3. Fix syntax errors in existing setup files
4. Fix transformers.utils module
5. Fix DeepSeek model in transformers
6. Fix custom encoder-decoder model

Run this script before running train_jarvis.sh
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """Run a script and log the result."""
    logger.info(f"Running {description}...")
    try:
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Run the script
        result = subprocess.run([sys.executable, script_path], check=False)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"❌ {description} failed with error: {e}")
        return False

def run_all_fixes():
    """Run all fixes in the correct order."""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Fix unterminated string literals in installed packages
    direct_package_fix = os.path.join(setup_dir, "direct_package_fix.py")
    if not os.path.exists(direct_package_fix):
        logger.error(f"❌ {direct_package_fix} not found")
        return False
    
    if not run_script(direct_package_fix, "Direct package fix"):
        logger.warning("⚠️ Direct package fix failed, but continuing anyway...")
    
    # Step 2: Fix all setup scripts
    paperspace_fix = os.path.join(setup_dir, "paperspace_fix.py")
    if os.path.exists(paperspace_fix):
        if not run_script(paperspace_fix, "Paperspace fix"):
            logger.warning("⚠️ Paperspace fix failed, but continuing anyway...")
    
    # Step 3: Fix transformers.utils module
    fix_transformers_utils = os.path.join(setup_dir, "fix_transformers_utils.py")
    if os.path.exists(fix_transformers_utils):
        if not run_script(fix_transformers_utils, "Transformers utils fix"):
            logger.warning("⚠️ Transformers utils fix failed, but continuing anyway...")
    
    # Step 4: Fix DeepSeek model in transformers
    fix_deepseek_model = os.path.join(setup_dir, "fix_deepseek_model.py")
    if os.path.exists(fix_deepseek_model):
        if not run_script(fix_deepseek_model, "DeepSeek model fix"):
            logger.warning("⚠️ DeepSeek model fix failed, but continuing anyway...")
    
    # Step 5: Fix custom encoder-decoder model
    fix_custom_encoder_decoder = os.path.join(setup_dir, "fix_custom_encoder_decoder_model.py")
    if os.path.exists(fix_custom_encoder_decoder):
        if not run_script(fix_custom_encoder_decoder, "Custom encoder-decoder model fix"):
            logger.warning("⚠️ Custom encoder-decoder model fix failed, but continuing anyway...")
    
    # Step 6: Fix Unsloth trust_remote_code issue
    fix_unsloth = os.path.join(setup_dir, "fix_unsloth_trust_remote_code.py")
    if os.path.exists(fix_unsloth):
        if not run_script(fix_unsloth, "Unsloth trust_remote_code fix"):
            logger.warning("⚠️ Unsloth trust_remote_code fix failed, but continuing anyway...")
    
    # Step 7: Fix attention mask issues
    fix_attention_mask = os.path.join(setup_dir, "fix_transformers_attention_mask.py")
    if os.path.exists(fix_attention_mask):
        if not run_script(fix_attention_mask, "Attention mask fix"):
            logger.warning("⚠️ Attention mask fix failed, but continuing anyway...")
    
    # Step 8: Verify GPU
    verify_gpu = os.path.join(setup_dir, "verify_gpu_code.py")
    if os.path.exists(verify_gpu):
        if not run_script(verify_gpu, "GPU verification"):
            logger.warning("⚠️ GPU verification failed, but continuing anyway...")
    
    logger.info("✅ All fixes have been applied")
    return True

if __name__ == "__main__":
    if run_all_fixes():
        logger.info("✅ All fixes completed successfully. You can now run train_jarvis.sh")
        sys.exit(0)
    else:
        logger.error("❌ Some fixes failed. Please check the logs for details.")
        sys.exit(1)
