#!/usr/bin/env python3
"""
PC Assistant - Main Entry Point

This script serves as the main entry point for the PC Assistant application.
It initializes and runs the assistant, allowing users to interact with it via command line.
"""

import logging
import sys
from assistant import Assistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("pc_assistant")


def main():
    """Main entry point for the PC Assistant."""
    logger.info("Starting PC Assistant...")
    
    # Print welcome message
    print("""
    ╔════════════════════════════════════════════╗
    ║               PC ASSISTANT                 ║
    ╚════════════════════════════════════════════╝
    
    Your comprehensive PC control assistant is ready!
    Type 'help' to see available commands.
    Type 'exit' or 'quit' to exit.
    """)
    
    # Initialize and run the assistant
    assistant = Assistant()
    assistant.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting PC Assistant. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)