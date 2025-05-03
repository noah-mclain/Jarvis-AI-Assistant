#!/usr/bin/env python3
"""
Spotify Commands

Commands for controlling Spotify application.
"""

import subprocess
import platform
import logging
import urllib.parse
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from assistant import Command

logger = logging.getLogger("assistant.media.spotify")


class SpotifyCommand(Command):
    """Control Spotify playback and search for music."""
    
    # Command mapping for easy integration
    command_map = {
        'search':      {'requires_query': True,  'method': '_spotify_search'},
        'play':        {'requires_query': True,  'method': '_spotify_play_search'},
        'resume':      {'requires_query': False, 'method': '_spotify_resume'},
        'pause':       {'requires_query': False, 'method': '_spotify_pause'},
        'next':        {'requires_query': False, 'method': '_spotify_next'},
        'previous':    {'requires_query': False, 'method': '_spotify_previous'},
        'open':        {'requires_query': False, 'method': '_spotify_open'}
    }

    def parse_command(self, command_str: str) -> tuple[str, str, list]:
        if not command_str:
            return '', '', []
        parts = command_str.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ''
        return action, query, []

    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        action, query, _ = self.parse_command(args)

        if not action:
            response['error'] = "No action provided. Usage: spotify <action> [query]"
            logger.error(response['error'])
            return response
        if action not in self.command_map:
            response['error'] = f"Unknown Spotify action: {action}"
            logger.error(response['error'])
            return response

        command = self.command_map[action]
        if command['requires_query'] and not query:
            response['error'] = f"No query provided. Usage: spotify {action} <query>"
            logger.error(response['error'])
            return response

        try:
            method = getattr(self, command['method'])
            args = command.get('args', [query] if query else [])
            success = method(*args)
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

    def _ensure_spotify_running(self, system: str) -> bool:
        """Make sure Spotify is running."""
        try:
            if system == "windows":
                result = subprocess.run("tasklist /FI \"IMAGENAME eq Spotify.exe\" /NH", shell=True, capture_output=True, text=True)
                if "Spotify.exe" not in result.stdout:
                    subprocess.Popen("start spotify:", shell=True)
                    logger.info("Started Spotify")
            elif system == "darwin":  # macOS
                result = subprocess.run("pgrep -x Spotify", shell=True, capture_output=True, text=True)
                if not result.stdout.strip():
                    subprocess.Popen(["open", "-a", "Spotify"])
                    logger.info("Started Spotify")
            elif system == "linux":
                result = subprocess.run("pgrep -x spotify", shell=True, capture_output=True, text=True)
                if not result.stdout.strip():
                    subprocess.Popen(["spotify"])
                    logger.info("Started Spotify")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure Spotify is running: {e}")
            return False

    def _spotify_open(self) -> bool:
        """Open Spotify."""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("start spotify:", shell=True)
            elif system == "darwin":
                subprocess.Popen(["open", "-a", "Spotify"])
            elif system == "linux":
                subprocess.Popen(["spotify"])
            logger.info("Opened Spotify")
            return True
        except Exception as e:
            logger.error(f"Failed to open Spotify: {e}")
            return False

    def _spotify_play_search(self, query: str) -> bool:
        """Search and play a track directly."""
        try:
            encoded_query = urllib.parse.quote(query)
            spotify_url = f"spotify:search:{encoded_query}"
            system = platform.system().lower()
            
            # First ensure Spotify is running
            self._ensure_spotify_running(system)
            
            # Open the search URL which will start playing the top result
            if system == "windows":
                subprocess.Popen(f"start {spotify_url}", shell=True)
                print(f"Spotify url: {spotify_url}")
                print(f"Spotify: Searching for: {query}")
            # Wait for search then send play command
                time.sleep(3)
                # self._spotify_resume()
            elif system == "darwin":
                subprocess.Popen(["open", spotify_url])
                # Play first result using AppleScript
                applescript = f'''
                tell application "Spotify"
                    activate
                    delay 1
                    play track "spotify:search:{encoded_query}"
                end tell
                '''
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.Popen(["xdg-open", spotify_url])
                # Use DBus to trigger playback
            time.sleep(2)
            subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify",
                            "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Play"], check=True)
            
            logger.info(f"Playing search result for: {query}")
            return True
        except Exception as e:
            logger.error(f"Failed to play search result: {e}")
            return False

    def _spotify_search(self, query: str) -> bool:
        """Search for a song/artist/album in Spotify."""
        try:
            encoded_query = urllib.parse.quote(query)
            spotify_url = f"spotify:search:{encoded_query}"
            system = platform.system().lower()
            
            if system == "windows":
                subprocess.Popen(f"start {spotify_url}", shell=True)
            elif system == "darwin":
                subprocess.Popen(["open", spotify_url])
            elif system == "linux":
                subprocess.Popen(["xdg-open", spotify_url])
            
            logger.info(f"Searching for: {query}")
            return True
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return False

    def _spotify_resume(self) -> bool:
        """Resume playback."""
        system = platform.system().lower()
        self._ensure_spotify_running(system)
        try:
            if system == "windows":
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]179)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":
                applescript = 'tell application "Spotify" to play'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                time.sleep(2)
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify",
                                "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Play"], check=True)
                logger.info("Spotify: Playing")
            return True
        except Exception as e:
            logger.error(f"Failed to resume: {e}")
            return False

    def _spotify_pause(self) -> bool:
        """Pause playback."""
        system = platform.system().lower()
        self._ensure_spotify_running(system)
        try:
            if system == "windows":
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]179)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":
                applescript = 'tell application "Spotify" to pause'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Pause"])
            logger.info("Spotify: Paused")
            return True
        except Exception as e:
            logger.error(f"Failed to pause: {e}")
            return False

    def _spotify_next(self) -> bool:
        """Skip to next track."""
        system = platform.system().lower()
        self._ensure_spotify_running(system)
        try:
            if system == "windows":
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]176)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":
                applescript = 'tell application "Spotify" to next track'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Next"])
            logger.info("Spotify: Next track")
            return True
        except Exception as e:
            logger.error(f"Failed to skip to next track: {e}")
            return False

    def _spotify_previous(self) -> bool:
        """Go to previous track."""
        system = platform.system().lower()
        self._ensure_spotify_running(system)
        try:
            if system == "windows":
                powershell_cmd = 'powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]177)"'
                subprocess.run(powershell_cmd, shell=True)
            elif system == "darwin":
                applescript = 'tell application "Spotify" to previous track'
                subprocess.run(["osascript", "-e", applescript])
            elif system == "linux":
                subprocess.run(["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify", 
                              "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Previous"])
            logger.info("Spotify: Previous track")
            return True
        except Exception as e:
            logger.error(f"Failed to go to previous track: {e}")
            return False

# Create an instance
spotify = SpotifyCommand()

# spotify.execute('pause')
# spotify.execute('resume')
# spotify.execute('next')
# spotify.execute('previous')
# spotify.execute('search hydrogen')
spotify.execute('open')
# spotify.execute('play')
