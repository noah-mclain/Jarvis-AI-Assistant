#!/usr/bin/env python3
"""
Test Imports for Generative AI Module

This script tests that all the consolidated functionality imports correctly,
ensuring that the code cleanup and consolidation was successful.
"""

import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

def test_imports():
    """Test all important imports"""
    print("\n=== Testing Imports ===\n")
    
    try:
        # Import the main module
        import src.generative_ai_module
        print("✅ Imported src.generative_ai_module")
    except ImportError as e:
        print(f"❌ Failed to import src.generative_ai_module: {e}")
        return False
    
    # List of important components to import
    components = [
        ("TextGenerator", "text_generator"),
        ("CodeGenerator", "code_generator"),
        ("DatasetProcessor", "dataset_processor"),
        ("UnifiedDatasetHandler", "unified_dataset_handler"),
        ("ConversationContext", "unified_dataset_handler"),
        ("PromptEnhancer", "prompt_enhancer"),
        ("ImprovedPreprocessor", "improved_preprocessing"),
        ("TrainingVisualizer", "unified_generation_pipeline"),
        ("train_text_generator", "unified_generation_pipeline"),
        ("calculate_metrics", "unified_generation_pipeline")
    ]
    
    success = True
    for component, module in components:
        try:
            # Try to import directly
            exec(f"from src.generative_ai_module.{module} import {component}")
            print(f"✅ Imported {component} from {module}")
        except ImportError as e:
            print(f"❌ Failed to import {component} from {module}: {e}")
            success = False
    
    # Test importing from the consolidated __init__.py
    print("\n=== Testing Imports from __init__.py ===\n")
    try:
        from src.generative_ai_module import (
            TextGenerator, CodeGenerator, DatasetProcessor,
            UnifiedDatasetHandler, ConversationContext, PromptEnhancer,
            ImprovedPreprocessor, TrainingVisualizer, train_text_generator,
            calculate_metrics, preprocess_data
        )
        print("✅ Successfully imported all components from __init__.py")
    except ImportError as e:
        print(f"❌ Failed to import from __init__.py: {e}")
        success = False
    
    return success

def test_functionality():
    """Test basic functionality of imported components"""
    print("\n=== Testing Basic Functionality ===\n")
    
    try:
        from src.generative_ai_module import UnifiedDatasetHandler, ConversationContext
        
        # Test UnifiedDatasetHandler
        handler = UnifiedDatasetHandler()
        print(f"✅ Initialized UnifiedDatasetHandler")
        print(f"  Supported datasets: {handler.SUPPORTED_DATASETS}")
        
        # Test ConversationContext
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there!")
        formatted = context.get_formatted_history()
        print(f"✅ ConversationContext working")
        print(f"  Formatted history: {formatted}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔍 Testing Generative AI Module Imports and Functionality")
    print("=" * 60)

    imports_ok = test_imports()
    functionality_ok = test_functionality() if imports_ok else False
    print("\n=== Test Summary ===")
    print(f"Imports: {'✅ Success' if imports_ok else '❌ Failed'}")
    print(f"Functionality: {'✅ Success' if functionality_ok else '❌ Failed'}")

    if imports_ok and functionality_ok:
        print("\n✨ All tests passed! The code cleanup was successful.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1) 