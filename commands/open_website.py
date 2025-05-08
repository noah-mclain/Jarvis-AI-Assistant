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

class OpenWebsiteCommand(Command):
    """Open a website in the default browser.
    
    Usage: web <url>
    
    If the URL doesn't start with http:// or https://, https:// will be added automatically.
    
    Examples:
        web google.com
        web https://github.com
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No URL provided. Usage: web <url>"
            logger.error(response['error'])
            return response
        
        url = args.strip()
        
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        try:
            success = self._open_website(url)
            
            if success:
                response.update({
                    'success': True,
                    'action': 'web',
                    'message': f"Opened website: {url}"
                })
            else:
                response.update({
                    'success': False,
                    'action': 'web',
                    'message': f"Failed to open website: {url}"
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error opening website: {url}"
            })
            logger.error(f"Error opening website {url}: {e}")
        
        return response
    
    def _open_website(self, url: str) -> bool:
        """Open a website in the default browser."""
        try:
            webbrowser.open(url)
            logger.info(f"Opened website: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to open website: {e}")
            return False


web = OpenWebsiteCommand()
# web.execute("google.com")