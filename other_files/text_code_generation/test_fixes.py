#!/usr/bin/env python3
"""
Test script to verify that our fixes for TRL/PEFT and spaCy issues are working.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_trl_peft_imports():
    """Test TRL/PEFT imports."""
    logger.info("Testing TRL/PEFT imports...")
    
    try:
        # First, try to import top_k_top_p_filtering from transformers
        from transformers import top_k_top_p_filtering
        logger.info("✅ Successfully imported top_k_top_p_filtering from transformers")
        
        # Try to import TRL
        try:
            import trl
            from trl import SFTTrainer
            logger.info(f"✅ Successfully imported TRL (version: {getattr(trl, '__version__', 'unknown')})")
            
            # Try to import SFTConfig
            try:
                from trl import SFTConfig
                logger.info("✅ Successfully imported SFTConfig from TRL")
            except ImportError as e:
                logger.warning(f"⚠️ Could not import SFTConfig from TRL: {e}")
                logger.info("This is expected for TRL 0.7.x")
            
            # Try to import PEFT
            try:
                from peft import LoraConfig
                logger.info("✅ Successfully imported LoraConfig from PEFT")
                return True
            except ImportError as e:
                logger.error(f"❌ Could not import LoraConfig from PEFT: {e}")
                return False
        except ImportError as e:
            logger.error(f"❌ Could not import TRL: {e}")
            return False
    except ImportError as e:
        logger.error(f"❌ Could not import top_k_top_p_filtering from transformers: {e}")
        return False

def test_spacy_imports():
    """Test spaCy imports."""
    logger.info("Testing spaCy imports...")
    
    try:
        # Try to import spaCy
        import spacy
        logger.info(f"✅ Successfully imported spaCy (version: {getattr(spacy, '__version__', 'unknown')})")
        
        # Try to import thinc.api
        try:
            from thinc.api import ParametricAttention_v2
            logger.info("✅ Successfully imported ParametricAttention_v2 from thinc.api")
        except ImportError as e:
            logger.info(f"⚠️ Could not import ParametricAttention_v2 from thinc.api: {e}")
            logger.info("This is expected if our dummy module is working")
            
            # Check if our dummy module is working
            import thinc.api
            if hasattr(thinc.api, 'ParametricAttention_v2'):
                logger.info("✅ Successfully created dummy ParametricAttention_v2 in thinc.api")
                return True
            else:
                logger.error("❌ Dummy ParametricAttention_v2 not found in thinc.api")
                return False
        
        return True
    except ImportError as e:
        logger.error(f"❌ Could not import spaCy: {e}")
        return False

def test_minimal_spacy_tokenizer():
    """Test minimal spaCy tokenizer."""
    logger.info("Testing minimal spaCy tokenizer...")
    
    try:
        # Try to import minimal_spacy_tokenizer
        from src.generative_ai_module.minimal_spacy_tokenizer import tokenize
        
        # Test tokenization
        test_text = "This is a test of the minimal tokenizer for Jarvis AI Assistant!"
        tokens = tokenize(test_text)
        
        logger.info(f"Input: {test_text}")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Count: {len(tokens)} tokens")
        
        if len(tokens) > 0:
            logger.info("✅ Minimal spaCy tokenizer is working")
            return True
        else:
            logger.error("❌ Minimal spaCy tokenizer returned empty tokens")
            return False
    except ImportError as e:
        logger.error(f"❌ Could not import minimal_spacy_tokenizer: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing minimal spaCy tokenizer: {e}")
        return False

def test_unsloth_deepseek():
    """Test unsloth_deepseek module."""
    logger.info("Testing unsloth_deepseek module...")
    
    try:
        # Try to import unsloth_deepseek
        from src.generative_ai_module import unsloth_deepseek
        
        # Check if the module has the necessary attributes
        if hasattr(unsloth_deepseek, 'get_unsloth_model'):
            logger.info("✅ Successfully imported unsloth_deepseek module")
            
            # Check if the TRL/PEFT warning is gone
            if not hasattr(unsloth_deepseek, 'TRL_HAS_SFT_CONFIG'):
                logger.warning("⚠️ TRL_HAS_SFT_CONFIG attribute not found in unsloth_deepseek")
            else:
                logger.info(f"✅ TRL_HAS_SFT_CONFIG is set to {unsloth_deepseek.TRL_HAS_SFT_CONFIG}")
            
            return True
        else:
            logger.error("❌ unsloth_deepseek module does not have get_unsloth_model attribute")
            return False
    except ImportError as e:
        logger.error(f"❌ Could not import unsloth_deepseek: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing unsloth_deepseek: {e}")
        return False

def main():
    """Main function to run all tests."""
    logger.info("Starting tests for TRL/PEFT and spaCy fixes...")
    
    # Test TRL/PEFT imports
    trl_peft_success = test_trl_peft_imports()
    
    # Test spaCy imports
    spacy_success = test_spacy_imports()
    
    # Test minimal spaCy tokenizer
    tokenizer_success = test_minimal_spacy_tokenizer()
    
    # Test unsloth_deepseek module
    unsloth_success = test_unsloth_deepseek()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"TRL/PEFT imports: {'✅ PASSED' if trl_peft_success else '❌ FAILED'}")
    logger.info(f"spaCy imports: {'✅ PASSED' if spacy_success else '❌ FAILED'}")
    logger.info(f"Minimal spaCy tokenizer: {'✅ PASSED' if tokenizer_success else '❌ FAILED'}")
    logger.info(f"unsloth_deepseek module: {'✅ PASSED' if unsloth_success else '❌ FAILED'}")
    
    # Return overall success
    return trl_peft_success and spacy_success and tokenizer_success and unsloth_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
