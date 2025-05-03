#!/usr/bin/env python3
"""
Media Commands

Commands for controlling media applications like Spotify, YouTube, and Netflix.
This module serves as a placeholder and imports the actual command implementations.
"""

from .spotify import SpotifyCommand
from .youtube import YoutubeCommand
from .netflix import NetflixCommand

__all__ = ['SpotifyCommand', 'YoutubeCommand', 'NetflixCommand']