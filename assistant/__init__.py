"""Assistant module for command handling."""

class Command:
    """Base class for all commands.
    
    All command implementations should inherit from this class
    and implement the execute method.
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> bool:
        """Execute the command with the given arguments.
        
        Args:
            args: Command arguments as a string
            *_args: Additional positional arguments
            **_kwargs: Additional keyword arguments
            
        Returns:
            bool: True if command executed successfully, False otherwise
        """
        raise NotImplementedError("Command classes must implement execute()")