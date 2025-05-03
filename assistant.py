#!/usr/bin/env python3
"""
Personal Assistant Application

A command-line based personal assistant that can execute various commands
like opening/closing applications, setting alarms, controlling Spotify,
and searching YouTube.
"""

import importlib
import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("assistant")


class Command(ABC):
    """Abstract base class for all commands."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> bool:
        """Execute the command with the given arguments."""
        pass
    
    @classmethod
    def get_command_name(cls) -> str:
        """Get the name of the command."""
        return cls.__name__.lower().replace('command', '')
    
    @classmethod
    def get_help(cls) -> str:
        """Get help information for the command."""
        return cls.__doc__ or "No help available for this command."


class CommandRegistry:
    """Registry for all available commands."""
    
    def __init__(self):
        self.commands: Dict[str, type] = {}
    
    def register(self, command_class: type) -> None:
        """Register a command class."""
        if not issubclass(command_class, Command):
            raise TypeError(f"{command_class.__name__} is not a subclass of Command")
        
        command_name = command_class.get_command_name()
        self.commands[command_name] = command_class
        logger.debug(f"Registered command: {command_name}")
    
    def get_command(self, command_name: str) -> Optional[type]:
        """Get a command class by name."""
        return self.commands.get(command_name.lower())
    
    def list_commands(self) -> List[str]:
        """List all available commands."""
        return sorted(self.commands.keys())


class Assistant:
    """Main assistant class that processes user commands."""
    
    def __init__(self):
        self.registry = CommandRegistry()
        self.load_commands()
    
    def load_commands(self) -> None:
        """Load all command modules from the commands directory."""
        commands_dir = os.path.join(os.path.dirname(__file__), "commands")
        
        # Create commands directory if it doesn't exist
        if not os.path.exists(commands_dir):
            os.makedirs(commands_dir)
            # Create an __init__.py file to make it a proper package
            with open(os.path.join(commands_dir, "__init__.py"), "w") as f:
                f.write("# Commands package\n")
        
        # Add commands directory to Python path
        if commands_dir not in sys.path:
            sys.path.append(os.path.dirname(__file__))
        
        # Import all command modules
        for filename in os.listdir(commands_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"commands.{module_name}")
                    
                    # Register all Command subclasses in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, Command) and attr != Command:
                            self.registry.register(attr)
                            
                except ImportError as e:
                    logger.error(f"Failed to import command module {module_name}: {e}")
    
    def process_command(self, user_input: str) -> bool:
        """Process a user command."""
        if not user_input.strip():
            return True
        
        # Split the input into command and arguments
        parts = user_input.split(maxsplit=1)
        command_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Handle built-in commands
        if command_name == "exit" or command_name == "quit":
            logger.info("Exiting assistant...")
            return False
        elif command_name == "help":
            self.show_help(args)
            return True
        
        # Get the command class
        command_class = self.registry.get_command(command_name)
        if not command_class:
            logger.error(f"Unknown command: {command_name}")
            logger.info("Type 'help' to see available commands.")
            return True
        
        # Execute the command
        try:
            command = command_class()
            command.execute(args)
        except Exception as e:
            logger.error(f"Error executing command {command_name}: {e}")
        
        return True
    
    def show_help(self, command_name: str = "") -> None:
        """Show help information for commands."""
        if command_name:
            command_class = self.registry.get_command(command_name)
            if command_class:
                logger.info(f"Help for '{command_name}':\n{command_class.get_help()}")
            else:
                logger.error(f"Unknown command: {command_name}")
        else:
            commands = self.registry.list_commands()
            logger.info("Available commands:")
            for cmd in commands:
                logger.info(f"  {cmd}")
            logger.info("\nType 'help <command>' for more information about a specific command.")
            logger.info("Type 'exit' or 'quit' to exit the assistant.")
    
    def run(self) -> None:
        """Run the assistant in an interactive loop."""
        logger.info("Personal Assistant started. Type 'help' for available commands.")
        
        running = True
        while running:
            try:
                user_input = input("\nAssistant> ").strip()
                running = self.process_command(user_input)
            except KeyboardInterrupt:
                logger.info("\nExiting assistant...")
                running = False
            except Exception as e:
                logger.error(f"Unexpected error: {e}")


def main():
    """Main entry point for the assistant."""
    assistant = Assistant()
    assistant.run()


if __name__ == "__main__":
    main()