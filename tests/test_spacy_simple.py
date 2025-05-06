#!/usr/bin/env python3
"""
Simple spaCy test script for Jarvis AI Assistant

This script checks if spaCy is installed and working correctly without importing
other Jarvis AI modules that might have dependencies on GPU.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("spacy_test")

def check_spacy_availability():
    """Check if spaCy is available and which version is installed"""
    try:
        import spacy
        logger.info(f"✅ spaCy version {spacy.__version__} is installed")
        return True, spacy.__version__
    except ImportError:
        logger.error("❌ spaCy is not installed")
        return False, None

def check_model_availability(model_name="en_core_web_sm"):
    """Check if the specified spaCy model is available and can be loaded"""
    try:
        import spacy
        nlp = spacy.load(model_name)
        logger.info(f"✅ Model {model_name} loaded successfully")
        return True, nlp
    except ImportError:
        logger.error("❌ spaCy is not installed")
        return False, None
    except OSError as e:
        logger.error(f"❌ Model {model_name} not found: {e}")
        logger.info(f"To install: python -m spacy download {model_name}")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False, None

def test_nlp_pipeline(nlp):
    """Test the spaCy NLP pipeline with a sample text"""
    if nlp is None:
        logger.error("❌ No NLP pipeline available to test")
        return False
    
    try:
        # Process a sample text
        text = "Jarvis AI Assistant is using spaCy for advanced natural language processing tasks."
        doc = nlp(text)
        
        # Print basic information
        logger.info("\nBasic tokenization:")
        for i, token in enumerate(doc):
            if i < 10:  # Just show the first 10 tokens
                logger.info(f"  {token.text} (POS: {token.pos_}, DEP: {token.dep_})")
        
        # Named entities
        if doc.ents:
            logger.info("\nNamed entities:")
            for ent in doc.ents:
                logger.info(f"  {ent.text} (Type: {ent.label_})")
        else:
            logger.info("\nNo named entities found in the sample text")
        
        # Sentences
        logger.info("\nSentence segmentation:")
        for sent in doc.sents:
            logger.info(f"  {sent.text}")
        
        logger.info("\n✅ NLP pipeline is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing NLP pipeline: {e}")
        return False

def main():
    """Main function to run spaCy tests"""
    logger.info("🔍 Testing spaCy installation for Jarvis AI Assistant")
    
    # Check if spaCy is available
    spacy_available, version = check_spacy_availability()
    if not spacy_available:
        logger.info("To install spaCy: pip install spacy==3.7.4")
        return 1
    
    # Check if the model is available
    model_available, nlp = check_model_availability()
    if not model_available:
        logger.info("To install the model: python -m spacy download en_core_web_sm")
        logger.info("Or: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz")
        return 1
    
    # Test the NLP pipeline
    if not test_nlp_pipeline(nlp):
        return 1
    
    logger.info("\n✅ All spaCy tests passed! spaCy is properly installed and working.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 