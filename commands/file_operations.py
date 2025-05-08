#!/usr/bin/env python3
"""
File Operations Commands

Commands for searching files, navigating directories, and managing files.
"""

import os
import glob
import shutil
import logging
import platform
import subprocess
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from assistant import Command

logger = logging.getLogger("assistant.file_operations")



class SearchFileCommand(Command):
    # search project C:\Users\username\Projects
    """
    Search for files or directories.
    If location is not provided, the search will be performed in the user's home directory.
        Examples:
            search document.txt
            search report.pdf
            search images/*
    """
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': '', 'results': []}
        
        if not args:
            response['error'] = "No search query provided. Usage: search <query> [location]"
            logger.error(response['error'])
            return response
        
        parts = args.split(maxsplit=1)
        query = parts[0]
        
        # Determine search location
        location = None
        if len(parts) > 1:
            location = parts[1]
        
        if not location:
            # Default to user's home directory
            location = os.path.expanduser("~")
        
        try:
            results = self._search_files(query, location)
            
            if results:
                response.update({
                    'success': True,
                    'action': 'search',
                    'message': f"Found {len(results)} results for '{query}' in {location}",
                    'results': results
                })
            else:
                response.update({
                    'success': True,
                    'action': 'search',
                    'message': f"No results found for '{query}' in {location}",
                    'results': []
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error searching for '{query}'"
            })
            logger.error(f"Error searching for '{query}': {e}")
        
        return response
    
    def _search_files(self, query: str, location: str) -> list:
        """Search for files matching the query in the specified location."""
        try:
            # Ensure the location exists
            if not os.path.exists(location):
                logger.error(f"Search location does not exist: {location}")
                return []
            
            # Construct the search pattern
            if os.path.isabs(query):
                # If query is an absolute path, use it directly
                pattern = query
            else:
                # Otherwise, join with the location
                pattern = os.path.join(location, "**", query)
            
            # Perform the search
            results = []
            for file_path in glob.glob(pattern, recursive=True):
                results.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'type': 'directory' if os.path.isdir(file_path) else 'file',
                    'size': os.path.getsize(file_path) if os.path.isfile(file_path) else None
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in _search_files: {e}")
            return []


class OpenFileCommand(Command):
    # openfile C:\Users\username\Documents\report.pdf

    """Open a file with the default application.
    
    Usage: openfile <file_path>
    
    Examples:
        openfile document.txt
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No file path provided. Usage: openfile <file_path>"
            logger.error(response['error'])
            return response
        
        file_path = args.strip()
        
        # If the path is not absolute, assume it's relative to the current directory
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        try:
            success = self._open_file(file_path)
            
            if success:
                response.update({
                    'success': True,
                    'action': 'openfile',
                    'message': f"Opened file: {file_path}"
                })
            else:
                response.update({
                    'success': False,
                    'action': 'openfile',
                    'message': f"Failed to open file: {file_path}"
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error opening file: {file_path}"
            })
            logger.error(f"Error opening file {file_path}: {e}")
        
        return response
    
    def _open_file(self, file_path: str) -> bool:
        """Open a file with the default application."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
            
            system = platform.system().lower()
            
            if system == "windows":
                os.startfile(file_path)
            elif system == "darwin":  # macOS
                subprocess.run(["open", file_path])
            elif system == "linux":
                subprocess.run(["xdg-open", file_path])
            
            logger.info(f"Opened file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to open file: {e}")
            return False


class ExploreCommand(Command):
    # explore C:\Users\username\Downloads
    
    """Open file explorer in a specific directory.
    
    Usage: explore [directory_path]
    
    If no directory is provided, opens the file explorer in the current directory.
    
    Examples:
        explore
        explore Documents
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        # If no args, use current directory
        directory = args.strip() if args.strip() else os.getcwd()
        
        # If the path is not absolute, assume it's relative to the current directory
        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)
        
        try:
            success = self._open_explorer(directory)
            
            if success:
                response.update({
                    'success': True,
                    'action': 'explore',
                    'message': f"Opened file explorer in: {directory}"
                })
            else:
                response.update({
                    'success': False,
                    'action': 'explore',
                    'message': f"Failed to open file explorer in: {directory}"
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error opening file explorer in: {directory}"
            })
            logger.error(f"Error opening file explorer in {directory}: {e}")
        
        return response
    
    def _open_explorer(self, directory: str) -> bool:
        """Open file explorer in the specified directory."""
        try:
            if not os.path.exists(directory) or not os.path.isdir(directory):
                logger.error(f"Directory does not exist: {directory}")
                return False
            
            system = platform.system().lower()
            
            if system == "windows":
                subprocess.Popen(["explorer", directory])
            elif system == "darwin":  # macOS
                subprocess.run(["open", directory])
            elif system == "linux":
                subprocess.run(["xdg-open", directory])
            
            logger.info(f"Opened file explorer in: {directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to open file explorer: {e}")
            return False


# Create instances of the commands
search_file =   ()
open_file = OpenFileCommand()
explore = ExploreCommand()