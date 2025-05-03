#!/usr/bin/env python3
"""
YouTube Commands

Commands for controlling YouTube playback and search.
"""

import subprocess
import platform
import logging
import webbrowser
import urllib.parse
from youtubesearchpython import VideosSearch
from pytube import Search
from pathlib import Path
import keyboard
import sys

sys.path.append(str(Path(__file__).parent.parent))
from assistant import Command

logger = logging.getLogger("assistant.media.youtube")


class YoutubeCommand(Command):
    """Control YouTube playback and search for videos.
    
    This class provides a structured interface for YouTube control that can be easily
    integrated with voice assistants and LLM models. It returns standardized response
    objects that can be parsed by other systems.
    
    Integration Points:
    - Voice Assistant: Use the command_map for intent mapping
    - LLM: Use parse_command() for natural language processing
    
    Usage: youtube <action> [query]
    
    Actions:
        search <query>: Search for videos on YouTube
        play <query>: Search and play the first result
        pause: Pause the current video (if browser supports it)
        resume: Resume the current video (if browser supports it)
        fullscreen: Toggle fullscreen mode (if browser supports it)
    
    Examples:
        youtube search how to make pasta
        youtube play latest news
        youtube pause
        youtube resume
        youtube fullscreen
    """
    
    # Command mapping for easy integration
    command_map = {
        'search': {'requires_query': True, 'method': '_youtube_search'},
        'play': {'requires_query': True, 'method': '_youtube_play'},
        'pause': {'requires_query': False, 'method': '_youtube_send_key', 'args': ['k']},
        'resume': {'requires_query': False, 'method': '_youtube_send_key', 'args': ['k']},
        'fullscreen': {'requires_query': False, 'method': '_youtube_send_key', 'args': ['f']},
        'open': {'requires_query': False, 'method': '_youtube_open'}
    }
    
    def parse_command(self, command_str: str) -> tuple[str, str, list]:
        """Parse a command string into action, query, and additional arguments.
        
        This method can be extended for natural language processing integration.
        
        Args:
            command_str: The command string to parse
            
        Returns:
            tuple: (action, query, additional_args)
        """
        if not command_str:
            return '', '', []
            
        parts = command_str.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ''
        
        return action, query, []
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        """Execute a YouTube command and return a structured response.
        
        Returns:
            dict: A structured response containing:
                - success (bool): Whether the command was successful
                - action (str): The action that was performed
                - message (str): A human-readable message
                - error (str): Error message if any
        """
        response = {
            'success': False,
            'action': '',
            'message': '',
            'error': ''
        }
        
        # Parse the command
        action, query, _ = self.parse_command(args)

        if not action:
            response['error'] = "No action provided. Usage: youtube <action> [query]"
            logger.error(response['error'])
            return response
        
        # Check if the action exists
        if action not in self.command_map:
            response['error'] = f"Unknown YouTube action: {action}"
            logger.error(response['error'])
            return response
        
        # Get command details
        command = self.command_map[action]

        # Check if query is required but missing
        if command['requires_query'] and not query:
            response['error'] = f"No search query provided. Usage: youtube {action} <query>"
            logger.error(response['error'])
            return response
        
        # Execute the command
        try:
            method = getattr(self, command['method'])
            args = command.get('args', [query] if query else [])
            success = method(*args)
            
            print(f"method: {method}")
            print(f"args: {args}")
            print(f"success: {success}")
            
            response.update({
                'success': success,
                'action': action,
                'message': f"Successfully executed {action}" if success else "Command failed"
            })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error executing {action}"
            })
            logger.error(f"Error executing {action}: {e}")
        
        return response
    
    def _youtube_open(self) -> bool:
        """Open YouTube homepage."""
        try:
            webbrowser.open("https://www.youtube.com")
            logger.info("Opened YouTube homepage")
            return True
        except Exception as e:
            logger.error(f"Open failed: {e}")
            return False
    
    def _youtube_search(self, query: str) -> bool:
        """Search YouTube for the specified query."""
        
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)            
            
            # Create the YouTube search URL
            youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}"
            
            # Open the URL in the default browser
            webbrowser.open(youtube_url)
            
            logger.info(f"Opened YouTube search for: {query}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to search YouTube: {e}")
            return False
    
    def _youtube_play(self, query: str) -> bool:
        if not query or not query.strip():
            logger.error("Empty search query")
            return False

        try:
            s = Search(query.strip())
            if not s.results:
                logger.error("No results")
                return False

            url = s.results[0].watch_url + "&autoplay=1"
            logger.info(f"Playing: {url}")
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.error(f"Play failed: {e}")
            return False
    
    def _youtube_send_key(self, key: str) -> bool:
        """Send a keyboard shortcut to YouTube (if browser tab is active)."""
        system = platform.system().lower()
        
        try:
            if system == "windows":
                # Use PowerShell to send key
                powershell_cmd = f'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys(\'{key}\')"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":  # macOS
                # This is a simplified approach and may not work in all cases
                applescript = f'tell application "System Events" to keystroke "{key}"'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                # This is a simplified approach and may not work in all cases
                subprocess.run(["xdotool", "key", key])
            
            logger.info(f"YouTube: Sent keyboard shortcut '{key}'")
            return True
        except Exception as e:
            logger.error(f"Failed to send keyboard shortcut to YouTube: {e}")
            return False


# Create an instance
youtube = YoutubeCommand()

# Search for videos
# youtube.execute("open")

# # Play a video
youtube.execute("search bertrand russell")

# # Control playback
# youtube.execute("pause")  # Pause current videok
# youtube.execute("Resume") # Resume playback
# youtube.execute("fullscreen") # Toggle fullscreen