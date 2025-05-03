#!/usr/bin/env python3
"""
Application Control Commands

Commands for opening and closing applications on the system.
"""

import os
import subprocess
import platform
import logging
from typing import Dict, List, Optional
from assistant import Command

logger = logging.getLogger("assistant.app_control")


class OpenCommand(Command):
    """Open an application.
    
    Usage: open <application_name>
    
    Examples:
        open chrome
        open notepad
        open spotify
    """
    
    # Common applications and their executable paths/commands
    APP_PATHS: Dict[str, Dict[str, str]] = {
        "windows": {
            "chrome": "start chrome",
            "firefox": "start firefox",
            "edge": "start msedge",
            "notepad": "notepad",
            "calculator": "calc",
            "explorer": "explorer",
            "spotify": "start spotify:",
            "word": "start winword",
            "excel": "start excel",
            "powerpoint": "start powerpnt",
            "cmd": "start cmd",
            "powershell": "start powershell",
            "vscode": "code",
        },
        "darwin": {  # macOS
            "chrome": "open -a 'Google Chrome'",
            "firefox": "open -a Firefox",
            "safari": "open -a Safari",
            "notes": "open -a Notes",
            "calculator": "open -a Calculator",
            "finder": "open .",
            "spotify": "open -a Spotify",
            "terminal": "open -a Terminal",
            "vscode": "open -a 'Visual Studio Code'",
        },
        "linux": {
            "chrome": "google-chrome",
            "firefox": "firefox",
            "calculator": "gnome-calculator",
            "files": "nautilus",
            "spotify": "spotify",
            "terminal": "gnome-terminal",
            "vscode": "code",
        }
    }
    
    def execute(self, app_name: str, *args, **kwargs) -> bool:
        """Open the specified application."""
        if not app_name:
            logger.error("No application name provided. Usage: open <application_name>")
            return False
        
        system = platform.system().lower()
        if system == "windows":
            system_key = "windows"
        elif system == "darwin":
            system_key = "darwin"
        elif system == "linux":
            system_key = "linux"
        else:
            logger.error(f"Unsupported operating system: {system}")
            return False
        
        # Get the app command
        app_name = app_name.lower()
        app_command = self.APP_PATHS.get(system_key, {}).get(app_name)
        
        if not app_command:
            # Try to open the app directly if it's not in our predefined list
            if system_key == "windows":
                app_command = f"start {app_name}"
            elif system_key == "darwin":
                app_command = f"open -a '{app_name}'"
            else:  # Linux
                app_command = app_name
        
        try:
            if system_key == "windows":
                subprocess.Popen(app_command, shell=True)
            else:
                subprocess.Popen(app_command.split())
            
            logger.info(f"Opened application: {app_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to open application {app_name}: {e}")
            return False


class CloseCommand(Command):
    """Close an application.
    
    Usage: close <application_name>
    
    Examples:
        close chrome
        close notepad
        close spotify
    """
    
    # Process names for common applications
    APP_PROCESSES: Dict[str, Dict[str, List[str]]] = {
        "windows": {
            "chrome": ["chrome.exe"],
            "firefox": ["firefox.exe"],
            "edge": ["msedge.exe"],
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "explorer": ["explorer.exe"],
            "spotify": ["Spotify.exe"],
            "word": ["WINWORD.EXE"],
            "excel": ["EXCEL.EXE"],
            "powerpoint": ["POWERPNT.EXE"],
            "vscode": ["Code.exe"],
        },
        "darwin": {  # macOS
            "chrome": ["Google Chrome"],
            "firefox": ["Firefox"],
            "safari": ["Safari"],
            "notes": ["Notes"],
            "calculator": ["Calculator"],
            "finder": ["Finder"],
            "spotify": ["Spotify"],
            "vscode": ["Code"],
        },
        "linux": {
            "chrome": ["chrome", "google-chrome"],
            "firefox": ["firefox"],
            "calculator": ["gnome-calculator"],
            "files": ["nautilus"],
            "spotify": ["spotify"],
            "vscode": ["code"],
        }
    }
    
    def execute(self, app_name: str, *args, **kwargs) -> bool:
        """Close the specified application."""
        if not app_name:
            logger.error("No application name provided. Usage: close <application_name>")
            return False
        
        system = platform.system().lower()
        if system == "windows":
            system_key = "windows"
            kill_command = "taskkill /F /IM"
        elif system == "darwin":
            system_key = "darwin"
            kill_command = "pkill -f"
        elif system == "linux":
            system_key = "linux"
            kill_command = "pkill -f"
        else:
            logger.error(f"Unsupported operating system: {system}")
            return False
        
        # Get the app process names
        app_name = app_name.lower()
        process_names = self.APP_PROCESSES.get(system_key, {}).get(app_name)
        
        if not process_names:
            # Try to close the app directly if it's not in our predefined list
            process_names = [app_name]
        
        success = False
        for process in process_names:
            try:
                if system_key == "windows":
                    subprocess.run(f"{kill_command} {process}", shell=True, check=False)
                else:
                    subprocess.run(f"{kill_command} {process}", shell=True, check=False)
                
                logger.info(f"Closed application: {app_name} (process: {process})")
                success = True
            except Exception as e:
                logger.error(f"Failed to close process {process}: {e}")
        
        return success