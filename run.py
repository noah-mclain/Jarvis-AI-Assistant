#!/usr/bin/env python3
"""
PC Assistant - Unified Entry Point

This script serves as the main entry point for the PC Assistant application.
It properly imports all components and runs the assistant.
"""

import logging
import sys
import os
import importlib
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("pc_assistant")


class Command(ABC):
    """Abstract base class for all commands."""
    
    @abstractmethod
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        """Execute the command with the given arguments."""
        pass
    
    def get_command_name(self) -> str:
        """Get the name of the command (instance method)."""
        class_name = self.__class__.__name__.lower()
        if class_name.endswith('command'):
            return class_name[:-7]  # Remove 'command' suffix
        return class_name
    
    def get_help(self) -> str:
        """Get help information for the command."""
        return self.__doc__ or "No help available for this command."


class CommandRegistry:
    """Registry for all available commands."""
    
    def __init__(self):
        self.commands: Dict[str, Command] = {}
    
    def register(self, command: Command) -> None:
        """Register a command instance."""
        # Get the command name from the class name or a specified attribute
        if hasattr(command, 'get_command_name'):
            if callable(command.get_command_name):
                command_name = command.get_command_name()
            else:
                command_name = command.get_command_name
        else:
            # Use the class name without 'Command' suffix and lowercase
            command_name = command.__class__.__name__.lower()
            if command_name.endswith('command'):
                command_name = command_name[:-7]  # Remove 'command' suffix
                
        # Store the command in the registry
        self.commands[command_name] = command
        logger.debug(f"Registered command: {command_name}")
    
    def get_command(self, command_name: str) -> Optional[Command]:
        """Get a command instance by name."""
        # Try direct lookup
        if command_name.lower() in self.commands:
            return self.commands[command_name.lower()]
        
        # For debugging, print what commands we have
        logger.debug(f"Available commands: {list(self.commands.keys())}")
        logger.debug(f"Looking for command: {command_name}")
        
        return None
    
    def list_commands(self) -> List[str]:
        """List all available commands."""
        return sorted(self.commands.keys())


class Assistant:
    """Main assistant class that processes user commands."""
    
    def __init__(self, load_commands_immediately=False):
        self.registry = CommandRegistry()
        if load_commands_immediately:
            self.load_commands()
    
    def load_commands(self) -> None:
        """Load all command modules and register command instances."""
        # Import commands from the commands directory
        commands_dir = os.path.join(os.path.dirname(__file__), "commands")
        
        # Import command modules directly
        from commands.system_control import volume, brightness, power, screenshot
        from commands.spotify import spotify
        from commands.youtube import youtube
        from commands.google import google
        from commands.open_website import web
        from commands.netflix import NetflixCommand
        from commands.alarm import AlarmCommand
        from commands.app_control import OpenCommand, CloseCommand
        from commands.file_operations import SearchFileCommand, OpenFileCommand, ExploreCommand
        from commands.maps_search import maps
        
        # Register command instances
        self.registry.register(volume)
        self.registry.register(brightness)
        self.registry.register(power)
        self.registry.register(screenshot)
        self.registry.register(spotify)
        self.registry.register(youtube)
        self.registry.register(google)
        self.registry.register(web)
        self.registry.register(NetflixCommand())
        self.registry.register(AlarmCommand())
        self.registry.register(OpenCommand())
        self.registry.register(CloseCommand())
        self.registry.register(SearchFileCommand())
        self.registry.register(OpenFileCommand())
        self.registry.register(ExploreCommand())
        self.registry.register(maps)
    
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
            # Load commands to be able to show help
            self.load_commands()
            self.show_help(args)
            return True
        
        # Load commands if not already loaded
        self.load_commands()
        
        # Get the command
        logger.info(f"Executing command: {command_name} with args: {args}")
        command = self.registry.get_command(command_name)
        
        if not command:
            logger.error(f"Unknown command: {command_name}")
            logger.info("Type 'help' to see available commands.")
            print(f"Available commands: {self.registry.list_commands()}")
            return True
        
        # Execute the command
        try:
            logger.info(f"Executing {command_name} command with args: {args}")
            result = command.execute(args)
            if isinstance(result, dict) and result.get('message'):
                logger.info(result['message'])
        except Exception as e:
            logger.error(f"Error executing command {command_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def show_help(self, command_name: str = "") -> None:
        """Show help information for commands."""
        if command_name:
            command = self.registry.get_command(command_name)
            if command:
                logger.info(f"Help for '{command_name}':\n{command.get_help()}")
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


def execute_command(command_text: str) -> dict:
    """
    Execute a command from external applications.
    
    This function can be imported and used by other applications like AI assistants
    or chat applications to execute commands without using the interactive CLI.
    
    Args:
        command_text: The full command text (e.g., "volume up 10" or "youtube play song name")
        
    Returns:
        dict: A response dictionary with the following keys:
            - success: Whether the command executed successfully
            - message: A user-friendly message about the result
            - error: Error message if any
            - action: The action that was performed
            - additional_data: Any additional data returned by the command
    """
    # Initialize a default response
    response = {
        'success': False,
        'message': '',
        'error': '',
        'action': '',
        'additional_data': {}
    }
    
    try:
        # Initialize the assistant (without loading commands automatically)
        assistant = Assistant(load_commands_immediately=False)
        # Now load the commands when we actually need them
        assistant.load_commands()
        
        # Split the input into command and arguments
        parts = command_text.split(maxsplit=1)
        if not parts:
            response['error'] = "No command provided"
            return response
            
        command_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Handle built-in commands
        if command_name in ("exit", "quit", "help"):
            response['error'] = f"Command '{command_name}' is only available in interactive mode"
            return response
        
        # Get the command
        command = assistant.registry.get_command(command_name)
        if not command:
            response['error'] = f"Unknown command: {command_name}"
            return response
        
        # Execute the command
        cmd_response = command.execute(args)
        
        # Return the response
        return cmd_response
    except Exception as e:
        response['error'] = str(e)
        logger.error(f"Error executing command: {e}")
        return response


def structured_execute_command(command_name: str, query: str = "") -> dict:
    """
    Execute a command using structured input format (command_name, query).
    
    This function is ideal for AI assistants or applications that want to call 
    commands with separate command name and query parameters.
    
    Args:
        command_name: The name of the command to execute (e.g., "youtube", "volume")
        query: The query/parameters for the command (e.g., "play music", "up 10")
        
    Returns:
        dict: A response dictionary with the same format as execute_command
    """
    # Initialize a default response
    response = {
        'success': False,
        'message': '',
        'error': '',
        'action': '',
        'additional_data': {}
    }
    
    if not command_name:
        response['error'] = "No command name provided"
        return response
    
    # Create the full command text by combining command name and query
    command_text = f"{command_name} {query}".strip()
    
    # Use the existing execute_command function
    return execute_command(command_text)


def main():
    """Main entry point for the PC Assistant."""
    logger.info("Starting PC Assistant...")
    
    # Initialize the assistant without automatically loading commands
    assistant = Assistant(load_commands_immediately=False)
    
    # Print welcome message
    print("""
    ╔════════════════════════════════════════════╗
    ║               PC ASSISTANT                 ║
    ╚════════════════════════════════════════════╝
    
    Your comprehensive PC control assistant is ready!
    Type 'help' to see available commands.
    Type 'exit' or 'quit' to exit.
    """)
    
    # Run the assistant
    assistant.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting PC Assistant. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 