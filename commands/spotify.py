#!/usr/bin/env python3
"""
Spotify Commands

Commands for controlling Spotify application.
"""

import subprocess
import platform
import logging
import urllib.parse
from assistant import Command

logger = logging.getLogger("assistant.media.spotify")


class SpotifyCommand(Command):
    """Control Spotify.
    
    Usage: spotify <action> [query]
    
    Actions:
        play: Start playback
        pause: Pause playback
        next: Skip to next track
        previous: Go to previous track
        search <query>: Search for a song/artist/album
        volume <0-100>: Set volume level
    
    Examples:
        spotify play
        spotify pause
        spotify next
        spotify search Bohemian Rhapsody
        spotify volume 50
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> bool:
        """Execute a Spotify command."""
        if not args:
            logger.error("No action provided. Usage: spotify <action> [query]")
            return False
        
        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""
        
        system = platform.system().lower()
        
        # First, try to open Spotify if it's not already running
        if action != "search":
            self._ensure_spotify_running(system)
        
        # Execute the requested action
        if action == "play":
            return self._spotify_play(system)
        elif action == "pause":
            return self._spotify_pause(system)
        elif action == "next":
            return self._spotify_next(system)
        elif action == "previous":
            return self._spotify_previous(system)
        elif action == "search":
            if not query:
                logger.error("No search query provided. Usage: spotify search <query>")
                return False
            return self._spotify_search(system, query)
        elif action == "volume":
            if not query or not query.isdigit() or int(query) < 0 or int(query) > 100:
                logger.error("Invalid volume level. Usage: spotify volume <0-100>")
                return False
            return self._spotify_volume(system, int(query))
        else:
            logger.error(f"Unknown Spotify action: {action}")
            return False
    
    def _ensure_spotify_running(self, system: str) -> bool:
        """Make sure Spotify is running."""
        try:
            if system == "windows":
                # Check if Spotify is running
                result = subprocess.run("tasklist /FI \"IMAGENAME eq Spotify.exe\" /NH", shell=True, capture_output=True, text=True)
                if "Spotify.exe" not in result.stdout:
                    # Start Spotify
                    subprocess.Popen("start spotify:", shell=True)
                    logger.info("Started Spotify")
            elif system == "darwin":  # macOS
                # Check if Spotify is running
                result = subprocess.run("pgrep -x Spotify", shell=True, capture_output=True, text=True)
                if not result.stdout.strip():
                    # Start Spotify
                    subprocess.Popen(["open", "-a", "Spotify"])
                    logger.info("Started Spotify")
            elif system == "linux":
                # Check if Spotify is running
                result = subprocess.run("pgrep -x spotify", shell=True, capture_output=True, text=True)
                if not result.stdout.strip():
                    # Start Spotify
                    subprocess.Popen(["spotify"])
                    logger.info("Started Spotify")
            
            return True
        except Exception as e:
            logger.error(f"Failed to ensure Spotify is running: {e}")
            return False
    
    def _spotify_play(self, system: str) -> bool:
        """Play Spotify."""
        try:
            if system == "windows":
                # Use PowerShell to send media key
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]179)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":  # macOS
                applescript = 'tell application "Spotify" to play'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Play"])
            
            logger.info("Spotify: Playing")
            return True
        except Exception as e:
            logger.error(f"Failed to play Spotify: {e}")
            return False
    
    def _spotify_pause(self, system: str) -> bool:
        """Pause Spotify."""
        try:
            if system == "windows":
                # Use PowerShell to send media key
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]179)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":  # macOS
                applescript = 'tell application "Spotify" to pause'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Pause"])
            
            logger.info("Spotify: Paused")
            return True
        except Exception as e:
            logger.error(f"Failed to pause Spotify: {e}")
            return False
    
    def _spotify_next(self, system: str) -> bool:
        """Skip to next track in Spotify."""
        try:
            if system == "windows":
                # Use PowerShell to send media key
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]176)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":  # macOS
                applescript = 'tell application "Spotify" to next track'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Next"])
            
            logger.info("Spotify: Next track")
            return True
        except Exception as e:
            logger.error(f"Failed to skip to next track in Spotify: {e}")
            return False
    
    def _spotify_previous(self, system: str) -> bool:
        """Go to previous track in Spotify."""
        try:
            if system == "windows":
                # Use PowerShell to send media key
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]177)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":  # macOS
                applescript = 'tell application "Spotify" to previous track'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Previous"])
            
            logger.info("Spotify: Previous track")
            return True
        except Exception as e:
            logger.error(f"Failed to go to previous track in Spotify: {e}")
            return False
    
    def _spotify_search(self, system: str, query: str) -> bool:
        """Search for a song/artist/album in Spotify."""
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)
            
            # Open Spotify search URL
            spotify_url = f"spotify:search:{encoded_query}"
            
            if system == "windows":
                subprocess.Popen(f"start {spotify_url}", shell=True)
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", spotify_url])
            elif system == "linux":
                subprocess.Popen(["xdg-open", spotify_url])
            
            logger.info(f"Spotify: Searching for '{query}'")
            return True
        except Exception as e:
            logger.error(f"Failed to search Spotify: {e}")
            return False
    
    def _spotify_volume(self, system: str, volume: int) -> bool:
        """Set Spotify volume level."""
        try:
            if system == "windows":
                # Windows doesn't have a direct way to control Spotify volume
                logger.warning("Setting Spotify volume is not supported on Windows")
                return False
            elif system == "darwin":  # macOS
                # Volume should be between 0 and 100
                volume_decimal = volume / 100.0
                applescript = f'tell application "Spotify" to set sound volume to {volume_decimal}'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                # Volume should be between 0 and 1
                volume_decimal = volume / 100.0
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Volume", 
                              f"double:{volume_decimal}"])
            
            logger.info(f"Spotify: Volume set to {volume}%")
            return True
        except Exception as e:
            logger.error(f"Failed to set Spotify volume: {e}")
            return False


# Create an instance
spotify = SpotifyCommand()

# Control playback
spotify.execute("play")  # Start playback
spotify.execute("pause") # Pause playback
spotify.execute("next")  # Skip to next track
spotify.execute("previous") # Go to previous track

# Search for music
spotify.execute("search Bohemian Rhapsody")

# Control volume (not supported on Windows)
spotify.execute("volume 50") # Set volume to 50%