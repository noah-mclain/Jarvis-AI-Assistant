#!/usr/bin/env python3
"""
System Control Commands

Commands for controlling system operations like volume, brightness, shutdown, restart, etc.
"""

import subprocess
import platform
import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from assistant import Command

logger = logging.getLogger("assistant.system_control")


class VolumeCommand(Command):
    """Control system volume.
    
    Usage: volume <action> [level]
    
    Actions:
        up [level] - Increase volume by level (default: 10)
        down [level] - Decrease volume by level (default: 10)
        set <level> - Set volume to specific level (0-100)
        mute - Toggle mute
        unmute - Unmute
    
    Examples:
        volume up
        volume down 5
        volume set 50
        volume mute
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No action provided. Usage: volume <action> [level]"
            logger.error(response['error'])
            return response
        
        parts = args.split()
        action = parts[0].lower()
        level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        
        system = platform.system().lower()
        
        try:
            if action == "up":
                step = level if level is not None else 10
                success = self._volume_up(system, step)
                action_desc = f"increase volume by {step}"
            elif action == "down":
                step = level if level is not None else 10
                success = self._volume_down(system, step)
                action_desc = f"decrease volume by {step}"
            elif action == "set" and level is not None:
                success = self._volume_set(system, level)
                action_desc = f"set volume to {level}"
            elif action == "mute":
                success = self._volume_mute(system)
                action_desc = "mute volume"
            elif action == "unmute":
                success = self._volume_unmute(system)
                action_desc = "unmute volume"
            else:
                response['error'] = f"Unknown volume action: {action}"
                logger.error(response['error'])
                return response
            
            response.update({
                'success': success,
                'action': action,
                'message': f"Successfully {action_desc}" if success else f"Failed to {action_desc}"
            })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error executing {action}"
            })
            logger.error(f"Error executing {action}: {e}")
        
        return response
    
    def _volume_up(self, system: str, step: int = 10) -> bool:
        try:
            if system == "windows":
                for _ in range(step // 2):  # Windows volume steps are typically 2%
                    subprocess.run('powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"', shell=True)
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) + {step}) --100%"])
            elif system == "linux":
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{step}%+"])
            logger.info(f"Increased volume by {step}%")
            return True
        except Exception as e:
            logger.error(f"Failed to increase volume: {e}")
            return False
    
    def _volume_down(self, system: str, step: int = 10) -> bool:
        try:
            if system == "windows":
                for _ in range(step // 2):  # Windows volume steps are typically 2%
                    subprocess.run('powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"', shell=True)
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) - {step}) --100%"])
            elif system == "linux":
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{step}%-"])
            logger.info(f"Decreased volume by {step}%")
            return True
        except Exception as e:
            logger.error(f"Failed to decrease volume: {e}")
            return False
    
    def _volume_set(self, system: str, level: int) -> bool:
        if not 0 <= level <= 100:
            logger.error(f"Volume level must be between 0 and 100, got {level}")
            return False
        
        try:
            if system == "windows":
                # Using nircmd utility (needs to be installed)
                subprocess.run(f"nircmd.exe setsysvolume {int(655.35 * level)}", shell=True)
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", f"set volume output volume {level} --100%"])
            elif system == "linux":
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level}%"])
            logger.info(f"Set volume to {level}%")
            return True
        except Exception as e:
            logger.error(f"Failed to set volume: {e}")
            return False
    
    def _volume_mute(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run('powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"', shell=True)
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "set volume with output muted"])
            elif system == "linux":
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "mute"])
            logger.info("Muted volume")
            return True
        except Exception as e:
            logger.error(f"Failed to mute volume: {e}")
            return False
    
    def _volume_unmute(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run('powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"', shell=True)
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "set volume without output muted"])
            elif system == "linux":
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "unmute"])
            logger.info("Unmuted volume")
            return True
        except Exception as e:
            logger.error(f"Failed to unmute volume: {e}")
            return False


class BrightnessCommand(Command):
    """Control screen brightness.
    
    Usage: brightness <action> [level]
    
    Actions:
        up [level] - Increase brightness by level (default: 10)
        down [level] - Decrease brightness by level (default: 10)
        set <level> - Set brightness to specific level (0-100)
    
    Examples:
        brightness up
        brightness down 5
        brightness set 50
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No action provided. Usage: brightness <action> [level]"
            logger.error(response['error'])
            return response
        
        parts = args.split()
        action = parts[0].lower()
        level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        
        system = platform.system().lower()
        
        try:
            if action == "up":
                step = level if level is not None else 10
                success = self._brightness_up(system, step)
                action_desc = f"increase brightness by {step}"
            elif action == "down":
                step = level if level is not None else 10
                success = self._brightness_down(system, step)
                action_desc = f"decrease brightness by {step}"
            elif action == "set" and level is not None:
                success = self._brightness_set(system, level)
                action_desc = f"set brightness to {level}"
            else:
                response['error'] = f"Unknown brightness action: {action}"
                logger.error(response['error'])
                return response
            
            response.update({
                'success': success,
                'action': action,
                'message': f"Successfully {action_desc}" if success else f"Failed to {action_desc}"
            })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error executing {action}"
            })
            logger.error(f"Error executing {action}: {e}")
        
        return response
    
    def _brightness_up(self, system: str, step: int = 10) -> bool:
        try:
            if system == "windows":
                # Using powershell to increase brightness
                script = """
                $brightness = (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness
                $brightness = [Math]::Min($brightness + {0}, 100)
                $monitors = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods
                $monitors.WmiSetBrightness(1, $brightness)
                """.format(step)
                subprocess.run(["powershell", "-Command", script])
            elif system == "darwin":  # macOS
                # macOS doesn't have a simple command-line brightness control
                logger.error("Brightness control not implemented for macOS")
                return False
            elif system == "linux":
                # Using xbacklight for Linux
                current = int(subprocess.check_output(["xbacklight", "-get"]).strip())
                new_level = min(current + step, 100)
                subprocess.run(["xbacklight", "-set", str(new_level)])
            logger.info(f"Increased brightness by {step}%")
            return True
        except Exception as e:
            logger.error(f"Failed to increase brightness: {e}")
            return False
    
    def _brightness_down(self, system: str, step: int = 10) -> bool:
        try:
            if system == "windows":
                # Using powershell to decrease brightness
                script = """
                $brightness = (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness
                $brightness = [Math]::Max($brightness - {0}, 0)
                $monitors = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods
                $monitors.WmiSetBrightness(1, $brightness)
                """.format(step)
                subprocess.run(["powershell", "-Command", script])
            elif system == "darwin":  # macOS
                # macOS doesn't have a simple command-line brightness control
                logger.error("Brightness control not implemented for macOS")
                return False
            elif system == "linux":
                # Using xbacklight for Linux
                current = int(subprocess.check_output(["xbacklight", "-get"]).strip())
                new_level = max(current - step, 0)
                subprocess.run(["xbacklight", "-set", str(new_level)])
            logger.info(f"Decreased brightness by {step}%")
            return True
        except Exception as e:
            logger.error(f"Failed to decrease brightness: {e}")
            return False
    
    def _brightness_set(self, system: str, level: int) -> bool:
        if not 0 <= level <= 100:
            logger.error(f"Brightness level must be between 0 and 100, got {level}")
            return False
        
        try:
            if system == "windows":
                # Using powershell to set brightness
                script = """
                $monitors = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods
                $monitors.WmiSetBrightness(1, {0})
                """.format(level)
                subprocess.run(["powershell", "-Command", script])
            elif system == "darwin":  # macOS
                # macOS doesn't have a simple command-line brightness control
                logger.error("Brightness control not implemented for macOS")
                return False
            elif system == "linux":
                # Using xbacklight for Linux
                subprocess.run(["xbacklight", "-set", str(level)])
            logger.info(f"Set brightness to {level}%")
            return True
        except Exception as e:
            logger.error(f"Failed to set brightness: {e}")
            return False


class PowerCommand(Command):
    """Control system power operations.
    
    Usage: power <action>
    
    Actions:
        shutdown - Shut down the system
        restart - Restart the system
        logout - Log out the current user
        sleep - Put the system to sleep
        hibernate - Hibernate the system
    
    Examples:
        power shutdown
        power restart
        power logout
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        if not args:
            response['error'] = "No action provided. Usage: power <action>"
            logger.error(response['error'])
            return response
        
        action = args.strip().lower()
        system = platform.system().lower()
        
        try:
            if action == "shutdown":
                success = self._shutdown(system)
                action_desc = "shut down the system"
            elif action == "restart":
                success = self._restart(system)
                action_desc = "restart the system"
            elif action == "logout":
                success = self._logout(system)
                action_desc = "log out the current user"
            elif action == "sleep":
                success = self._sleep(system)
                action_desc = "put the system to sleep"
            elif action == "hibernate":
                success = self._hibernate(system)
                action_desc = "hibernate the system"
            else:
                response['error'] = f"Unknown power action: {action}"
                logger.error(response['error'])
                return response
            
            response.update({
                'success': success,
                'action': action,
                'message': f"Successfully initiated {action_desc}" if success else f"Failed to {action_desc}"
            })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': f"Error executing {action}"
            })
            logger.error(f"Error executing {action}: {e}")
        
        return response
    
    def _shutdown(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run(["shutdown", "/s", "/t", "0"])
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "tell app \"System Events\" to shut down"])
            elif system == "linux":
                subprocess.run(["shutdown", "-h", "now"])
            logger.info("Shutting down the system")
            return True
        except Exception as e:
            logger.error(f"Failed to shut down: {e}")
            return False
    
    def _restart(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run(["shutdown", "/r", "/t", "0"])
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "tell app \"System Events\" to restart"])
            elif system == "linux":
                subprocess.run(["shutdown", "-r", "now"])
            logger.info("Restarting the system")
            return True
        except Exception as e:
            logger.error(f"Failed to restart: {e}")
            return False
    
    def _logout(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run(["shutdown", "/l"])
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "tell app \"System Events\" to log out"])
            elif system == "linux":
                subprocess.run(["gnome-session-quit", "--logout", "--no-prompt"])
            logger.info("Logging out the current user")
            return True
        except Exception as e:
            logger.error(f"Failed to log out: {e}")
            return False
    
    def _sleep(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "tell app \"System Events\" to sleep"])
            elif system == "linux":
                subprocess.run(["systemctl", "suspend"])
            logger.info("Putting the system to sleep")
            return True
        except Exception as e:
            logger.error(f"Failed to sleep: {e}")
            return False
    
    def _hibernate(self, system: str) -> bool:
        try:
            if system == "windows":
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "Hibernate"])
            elif system == "darwin":  # macOS
                # macOS doesn't support hibernate directly
                logger.error("Hibernate not supported on macOS")
                return False
            elif system == "linux":
                subprocess.run(["systemctl", "hibernate"])
            logger.info("Hibernating the system")
            return True
        except Exception as e:
            logger.error(f"Failed to hibernate: {e}")
            return False


class ScreenshotCommand(Command):
    """Take a screenshot.
    
    Usage: screenshot [filename]
    
    If no filename is provided, the screenshot will be saved with a timestamp.
    
    Examples:
        screenshot
        screenshot my_screenshot
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> dict:
        response = {'success': False, 'action': '', 'message': '', 'error': ''}
        
        filename = args.strip() if args.strip() else f"screenshot_{int(time.time())}"
        if not filename.endswith(".png"):
            filename += ".png"
        
        # Use the user's Pictures directory or current directory
        if platform.system().lower() == "windows":
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        else:
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        
        if not os.path.exists(pictures_dir):
            pictures_dir = os.getcwd()
        
        filepath = os.path.join(pictures_dir, filename)
        system = platform.system().lower()
        
        try:
            success = self._take_screenshot(system, filepath)
            
            if success:
                response.update({
                    'success': True,
                    'action': 'screenshot',
                    'message': f"Screenshot saved to {filepath}"
                })
            else:
                response.update({
                    'success': False,
                    'action': 'screenshot',
                    'message': "Failed to take screenshot"
                })
        except Exception as e:
            response.update({
                'success': False,
                'error': str(e),
                'message': "Error taking screenshot"
            })
            logger.error(f"Error taking screenshot: {e}")
        
        return response
    
    def _take_screenshot(self, system: str, filepath: str) -> bool:
        try:
            if system == "windows":
                # Using powershell to take screenshot
                script = """
                Add-Type -AssemblyName System.Windows.Forms
                Add-Type -AssemblyName System.Drawing
                $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bitmap = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($screen.X, $screen.Y, 0, 0, $screen.Size)
                $bitmap.Save('{0}')
                $graphics.Dispose()
                $bitmap.Dispose()
                """.format(filepath.replace('\\', '\\\\'))
                subprocess.run(["powershell", "-Command", script])
            elif system == "darwin":  # macOS
                subprocess.run(["screencapture", filepath])
            elif system == "linux":
                subprocess.run(["import", "-window", "root", filepath])
            logger.info(f"Screenshot saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return False


# Create instances of the commands
volume = VolumeCommand()
brightness = BrightnessCommand()
power = PowerCommand()
screenshot = ScreenshotCommand()
# Test code - commented out to prevent automatic execution on import
# screenshot.execute("screenshot")