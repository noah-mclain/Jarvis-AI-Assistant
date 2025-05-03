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
from pytube import Search
from pathlib import Path
import time
import keyboard
import sys

sys.path.append(str(Path(__file__).parent.parent))
from assistant import Command

logger = logging.getLogger("assistant.media.youtube")


class YoutubeCommand(Command):
    """Control YouTube playback and search for videos."""
    
    # Command mapping for easy integration
    command_map = {
        'search':      {'requires_query': True,  'method': '_youtube_search'},
        'play':        {'requires_query': True,  'method': '_youtube_play'},
        'pause':       {'requires_query': False, 'method': '_youtube_send_key', 'args': ['k']},
        'resume':      {'requires_query': False, 'method': '_youtube_send_key', 'args': ['k']},
        'fullscreen':  {'requires_query': False, 'method': '_youtube_send_key', 'args': ['f']},
        'open':        {'requires_query': False, 'method': '_youtube_open'},
        'rewind':      {'requires_query': False, 'method': '_youtube_send_key', 'args': ['j']},
        'forward':     {'requires_query': False, 'method': '_youtube_send_key', 'args': ['l']},
        'volume_up':   {'requires_query': False, 'method': '_youtube_send_key', 'args': ['up']},
        'volume_down': {'requires_query': False, 'method': '_youtube_send_key', 'args': ['down']},
        'mute':        {'requires_query': False, 'method': '_youtube_send_key', 'args': ['m']},
        'captions':    {'requires_query': False, 'method': '_youtube_send_key', 'args': ['c']},
        'theater':     {'requires_query': False, 'method': '_youtube_send_key', 'args': ['t']},
        'miniplayer':  {'requires_query': False, 'method': '_youtube_send_key', 'args': ['i']},
        'skip_next':   {'requires_query': False, 'method': '_youtube_send_key', 'args': ['shift+right']},
        'skip_prev':   {'requires_query': False, 'method': '_youtube_send_key', 'args': ['shift+left']},
        'start':       {'requires_query': False, 'method': '_youtube_send_key', 'args': ['0']},
        'end':         {'requires_query': False, 'method': '_youtube_send_key', 'args': ['end']},
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
            response['error'] = "No action provided. Usage: youtube <action> [query]"
            logger.error(response['error'])
            return response
        if action not in self.command_map:
            response['error'] = f"Unknown YouTube action: {action}"
            logger.error(response['error'])
            return response

        command = self.command_map[action]
        if command['requires_query'] and not query:
            response['error'] = f"No search query provided. Usage: youtube {action} <query>"
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

    def _youtube_open(self) -> bool:
        try:
            webbrowser.open("https://www.youtube.com")
            logger.info("Opened YouTube homepage")
            return True
        except Exception as e:
            logger.error(f"Open failed: {e}")
            return False

    def _youtube_search(self, query: str) -> bool:
        try:
            encoded = urllib.parse.quote(query)
            webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
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
                keyboard.send(key)
            elif system == "darwin":
                applescript = f'tell application "System Events" to keystroke "{key}"'
                subprocess.run(["osascript", "-e", applescript], check=True)
            elif system == "linux":
                subprocess.run(["xdotool", "key", key], check=True)
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
youtube.execute("play bertrand russell")

# # Control playback
# youtube.execute("pause")  # Pause current videok
# youtube.execute("mute") # Resume playback
# youtube.execute("fullscreen") # Toggle fullscreen