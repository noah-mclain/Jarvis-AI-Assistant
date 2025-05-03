#!/usr/bin/env python3
"""
Netflix Commands

Commands for controlling Netflix playback and search.
"""

import subprocess
import platform
import logging
import webbrowser
import urllib.parse
from assistant import Command

logger = logging.getLogger("assistant.media.netflix")


class NetflixCommand(Command):
    """Search and control Netflix.
    
    Usage: netflix <action> [query]
    
    Actions:
        search <query>: Search for shows/movies on Netflix
        open: Open Netflix homepage
        pause: Pause the current video (if browser supports it)
        play: Resume the current video (if browser supports it)
        fullscreen: Toggle fullscreen mode (if browser supports it)
    
    Examples:
        netflix search Stranger Things
        netflix open
        netflix pause
        netflix play
        netflix fullscreen
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> bool:
        """Execute a Netflix command."""
        if not args:
            logger.error("No action provided. Usage: netflix <action> [query]")
            return False
        
        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""
        
        # Execute the requested action
        if action == "search":
            if not query:
                logger.error("No search query provided. Usage: netflix search <query>")
                return False
            return self._netflix_search(query)
        elif action == "open":
            return self._netflix_open()
        elif action == "pause":
            return self._netflix_send_key("space")
        elif action == "play":
            return self._netflix_send_key("space")
        elif action == "fullscreen":
            return self._netflix_send_key("f")
        else:
            logger.error(f"Unknown Netflix action: {action}")
            return False
    
    def _netflix_search(self, query: str) -> bool:
        """Search Netflix for the specified query."""
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)
            
            # Create the Netflix search URL
            netflix_url = f"https://www.netflix.com/search?q={encoded_query}"
            
            # Open the URL in the default browser
            webbrowser.open(netflix_url)
            
            logger.info(f"Opened Netflix search for: {query}")
            return True
        except Exception as e:
            logger.error(f"Failed to search Netflix: {e}")
            return False
    
    def _netflix_open(self) -> bool:
        """Open Netflix homepage."""
        try:
            # Open Netflix homepage
            webbrowser.open("https://www.netflix.com")
            
            logger.info("Opened Netflix homepage")
            return True
        except Exception as e:
            logger.error(f"Failed to open Netflix: {e}")
            return False
    
    def _netflix_send_key(self, key: str) -> bool:
        """Send a keyboard shortcut to Netflix (if browser tab is active)."""
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
            
            logger.info(f"Netflix: Sent keyboard shortcut '{key}'")
            return True
        except Exception as e:
            logger.error(f"Failed to send keyboard shortcut to Netflix: {e}")
            return False