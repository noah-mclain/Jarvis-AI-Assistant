#!/usr/bin/env python3
"""
Web Search Commands

Commands for searching the web, opening websites, and other web-related operations.
"""

import webbrowser
import urllib.parse
import logging
import platform
import subprocess
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from assistant import Command

logger = logging.getLogger("assistant.web_search")


class MapsSearchCommand(Command):
    """Search Google Maps for a location or directions.
    
    Usage: maps <query>
    
    Examples:
        maps coffee shops near me
        maps directions to airport
        maps New York to Boston
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No search query provided. Usage: maps <query>"
            logger.error(response['error'])
            return response
        
        query = args.strip()
        
        try:
            success = self._search_maps(query)
            
            if success:
                response.update({
                    'success': True,
                    'action': 'maps',
                    'message': f"Searched Google Maps for: {query}"
                })
            else:
                response.update({
                    'success': False,
                    'action': 'maps',
                    'message': f"Failed to search Google Maps for: {query}"
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error searching Google Maps for: {query}"
            })
            logger.error(f"Error searching Google Maps for '{query}': {e}")
        
        return response
    
    def _search_maps(self, query: str) -> bool:
        """Search Google Maps for the specified query."""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.google.com/maps/search/{encoded_query}"
            webbrowser.open(url)
            logger.info(f"Searched Google Maps for: {query}")
            return True
        except Exception as e:
            logger.error(f"Failed to search Google Maps: {e}")
            return False


# Create instances of the commands
maps = MapsSearchCommand()
maps.execute("coffee shops near me")