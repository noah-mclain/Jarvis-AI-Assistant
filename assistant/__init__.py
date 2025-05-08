"""Assistant module for command handling."""

class Command:
    """Base class for all commands.
    
    All command implementations should inherit from this class
    and implement the execute method.
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        """Execute the command with the given arguments.
        
        Args:
            args: Command arguments as a string
            *_args: Additional positional arguments
            **_kwargs: Additional keyword arguments
            
        Returns:
            dict: Response with these keys:
                - success: Whether the command was successful
                - message: User-friendly message about what happened
                - error: Error message if anything went wrong
                - action: The action that was performed
        """
        raise NotImplementedError("Command classes must implement execute()")
    
    def get_command_name(self) -> str:
        """Get the name of the command."""
        class_name = self.__class__.__name__.lower()
        if class_name.endswith('command'):
            return class_name[:-7]  # Remove 'command' suffix
        return class_name
    
    def get_help(self) -> str:
        """Get help information for the command."""
        return self.__doc__ or "No help available for this command."