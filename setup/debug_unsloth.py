#!/usr/bin/env python3

import sys
import traceback

def debug_import():
    print("Python version:", sys.version)
    print("Python path:", sys.path)
    
    print("\n--- Testing unsloth import ---")
    try:
        import unsloth
        print("✅ Unsloth imported successfully!")
        print("Unsloth version:", getattr(unsloth, "__version__", "unknown"))
        print("Unsloth path:", unsloth.__file__)
    except Exception as e:
        print("❌ Error importing unsloth:", str(e))
        print("\n--- Error traceback ---")
        traceback.print_exc()
    
    print("\n--- Checking transformers version ---")
    try:
        import transformers
        print("Transformers version:", transformers.__version__)
        print("Transformers path:", transformers.__file__)
        
        # Check for the specific Gemma module
        print("\nChecking for transformers.models.gemma module:")
        try:
            import transformers.models.gemma
            print("✅ transformers.models.gemma exists")
        except ImportError:
            print("❌ transformers.models.gemma does not exist")
            
    except Exception as e:
        print("Error importing transformers:", str(e))
    
    print("\n--- Checking accelerate version ---")
    try:
        import accelerate
        print("Accelerate version:", accelerate.__version__)
    except Exception as e:
        print("Error importing accelerate:", str(e))
    
    print("\n--- Checking peft version ---")
    try:
        import peft
        print("PEFT version:", peft.__version__)
    except Exception as e:
        print("Error importing peft:", str(e))
    
    print("\n--- Available unsloth versions on PyPI ---")
    try:
        import json
        import urllib.request
        url = "https://pypi.org/pypi/unsloth/json"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        releases = sorted(list(data["releases"].keys()))
        print("Latest 10 versions:", releases[-10:])
    except Exception as e:
        print("Error fetching versions:", str(e))

if __name__ == "__main__":
    debug_import() 