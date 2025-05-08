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


class GoogleSearchCommand(Command):
    """Search Google for a query.
    
    Usage: google <search_query>
    
    Examples:
        google how to make pancakes
        google weather in New York
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No search query provided. Usage: google <search_query>"
            logger.error(response['error'])
            return response
        
        query = args.strip()
        
        try:
            success = self._search_google(query)
            
            if success:
                response.update({
                    'success': True,
                    'action': 'google',
                    'message': f"Searched Google for: {query}"
                })
            else:
                response.update({
                    'success': False,
                    'action': 'google',
                    'message': f"Failed to search Google for: {query}"
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error searching Google for: {query}"
            })
            logger.error(f"Error searching Google for '{query}': {e}")
        
        return response
    
    def _search_google(self, query: str) -> bool:
        """Search Google for the specified query."""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            webbrowser.open(url)
            logger.info(f"Searched Google for: {query}")
            return True
        except Exception as e:
            logger.error(f"Failed to search Google: {e}")
            return False

google = GoogleSearchCommand()
# Test code - commented out to prevent automatic execution on import
# google.execute("how to make pancakes")