#!/usr/bin/env python3
"""
Alarm Commands

Commands for setting and managing alarms.
"""

import datetime
import threading
import time
import logging
import platform
import subprocess
from typing import Dict, List, Optional
from assistant import Command

logger = logging.getLogger("assistant.alarm")

# Global dictionary to store active alarms
ACTIVE_ALARMS = {}


class AlarmCommand(Command):
    """Set an alarm for a specific time.
    
    Usage: alarm <time> [message]
    
    Examples:
        alarm 14:30 Take a break
        alarm 08:00 Wake up
        alarm 5m Quick reminder
        alarm 1h30m Check project status
    """
    
    def execute(self, args: str, *_args, **_kwargs) -> bool:
        """Set an alarm for the specified time."""
        if not args:
            logger.error("No time provided. Usage: alarm <time> [message]")
            return False
        
        parts = args.split(maxsplit=1)
        time_str = parts[0]
        message = parts[1] if len(parts) > 1 else "Alarm!"
        
        try:
            # Parse the time string
            alarm_time = self._parse_time(time_str)
            if not alarm_time:
                return False
            
            # Create a unique ID for the alarm
            alarm_id = str(int(time.time()))
            
            # Calculate seconds until the alarm
            now = datetime.datetime.now()
            if isinstance(alarm_time, datetime.datetime):
                delta = (alarm_time - now).total_seconds()
            else:  # It's a timedelta
                delta = alarm_time.total_seconds()
            
            if delta <= 0:
                logger.error("Cannot set alarm in the past.")
                return False
            
            # Format the alarm time for display
            if isinstance(alarm_time, datetime.datetime):
                display_time = alarm_time.strftime("%H:%M:%S")
            else:
                hours, remainder = divmod(int(delta), 3600)
                minutes, seconds = divmod(remainder, 60)
                display_time = f"{hours}h {minutes}m {seconds}s from now"
            
            # Start a timer thread for the alarm
            timer = threading.Timer(delta, self._trigger_alarm, args=[alarm_id, message])
            timer.daemon = True
            timer.start()
            
            # Store the alarm
            ACTIVE_ALARMS[alarm_id] = {
                "time": alarm_time,
                "message": message,
                "timer": timer
            }
            
            logger.info(f"Alarm set for {display_time}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set alarm: {e}")
            return False
    
    def _parse_time(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse the time string into a datetime object."""
        # Check if it's a relative time (e.g., 5m, 1h30m)
        if any(unit in time_str.lower() for unit in ['h', 'm', 's']):
            return self._parse_relative_time(time_str)
        
        # Check if it's a specific time (e.g., 14:30)
        try:
            # Parse time in HH:MM or HH:MM:SS format
            if ':' in time_str:
                time_parts = time_str.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
                
                now = datetime.datetime.now()
                alarm_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
                
                # If the time is already past, set it for tomorrow
                if alarm_time <= now:
                    alarm_time += datetime.timedelta(days=1)
                
                return alarm_time
            else:
                # Assume it's just hours
                hour = int(time_str)
                now = datetime.datetime.now()
                alarm_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # If the time is already past, set it for tomorrow
                if alarm_time <= now:
                    alarm_time += datetime.timedelta(days=1)
                
                return alarm_time
        except ValueError:
            logger.error(f"Invalid time format: {time_str}")
            return None
    
    def _parse_relative_time(self, time_str: str) -> datetime.timedelta:
        """Parse a relative time string (e.g., 5m, 1h30m) into a timedelta."""
        total_seconds = 0
        current_num = ""
        
        for char in time_str.lower():
            if char.isdigit():
                current_num += char
            elif char == 'h' and current_num:
                total_seconds += int(current_num) * 3600  # hours to seconds
                current_num = ""
            elif char == 'm' and current_num:
                total_seconds += int(current_num) * 60  # minutes to seconds
                current_num = ""
            elif char == 's' and current_num:
                total_seconds += int(current_num)  # seconds
                current_num = ""
        
        # Handle any remaining digits without a unit (assume minutes)
        if current_num:
            total_seconds += int(current_num) * 60
        
        return datetime.timedelta(seconds=total_seconds)
    
    def _trigger_alarm(self, alarm_id: str, message: str) -> None:
        """Trigger the alarm with a notification and sound."""
        logger.info(f"ALARM: {message}")
        
        # Play a sound based on the platform
        system = platform.system().lower()
        try:
            if system == "windows":
                # Use PowerShell to play a beep sound and show a notification
                powershell_cmd = f'powershell -Command "[Console]::Beep(800,1000); Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show(\"{message}\", \"Alarm\", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information)"'
                subprocess.Popen(powershell_cmd, shell=True)
            elif system == "darwin":  # macOS
                # Use AppleScript to show a notification
                applescript = f'display notification "{message}" with title "Alarm"'
                subprocess.Popen(["osascript", "-e", applescript])
                # Play a sound
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
            else:  # Linux
                # Use notify-send for notification
                subprocess.Popen(["notify-send", "Alarm", message])
                # Play a sound if available
                subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"], stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"Failed to trigger alarm notification: {e}")
        
        # Remove the alarm from active alarms
        if alarm_id in ACTIVE_ALARMS:
            del ACTIVE_ALARMS[alarm_id]


class AlarmsCommand(Command):
    """List all active alarms.
    
    Usage: alarms
    """
    
    def execute(self, *args, **kwargs) -> bool:
        """List all active alarms."""
        if not ACTIVE_ALARMS:
            logger.info("No active alarms.")
            return True
        
        logger.info("Active alarms:")
        now = datetime.datetime.now()
        
        for alarm_id, alarm in ACTIVE_ALARMS.items():
            alarm_time = alarm["time"]
            message = alarm["message"]
            
            if isinstance(alarm_time, datetime.datetime):
                time_str = alarm_time.strftime("%H:%M:%S")
                delta = (alarm_time - now).total_seconds()
            else:  # It's a timedelta
                delta = alarm_time.total_seconds()
                hours, remainder = divmod(int(delta), 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours}h {minutes}m {seconds}s from now"
            
            # Calculate remaining time
            hours, remainder = divmod(int(delta), 3600)
            minutes, seconds = divmod(remainder, 60)
            remaining = f"{hours}h {minutes}m {seconds}s remaining"
            
            logger.info(f"  - {time_str}: {message} ({remaining})")
        
        return True


class CancelAlarmCommand(Command):
    """Cancel an active alarm.
    
    Usage: cancelalarm <alarm_message>
    
    Examples:
        cancelalarm Wake up
    """
    
    def execute(self, message: str, *args, **kwargs) -> bool:
        """Cancel an alarm with the specified message."""
        if not message:
            logger.error("No alarm message provided. Usage: cancelalarm <alarm_message>")
            return False
        
        if not ACTIVE_ALARMS:
            logger.info("No active alarms to cancel.")
            return False
        
        # Find alarms with matching messages
        matching_alarms = []
        for alarm_id, alarm in ACTIVE_ALARMS.items():
            if message.lower() in alarm["message"].lower():
                matching_alarms.append(alarm_id)
        
        if not matching_alarms:
            logger.error(f"No active alarms found with message containing '{message}'.")
            return False
        
        # Cancel the matching alarms
        for alarm_id in matching_alarms:
            alarm = ACTIVE_ALARMS[alarm_id]
            alarm["timer"].cancel()
            del ACTIVE_ALARMS[alarm_id]
            logger.info(f"Canceled alarm: {alarm['message']}")
        
        return True