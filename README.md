# PC Assistant

A comprehensive PC control assistant that can execute various commands like controlling system volume, brightness, taking screenshots, playing music on Spotify, searching YouTube, and more.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pc-assistant.git
cd pc-assistant

# Install the package
pip install -e .
```

## Running the Assistant

You can run the assistant directly from the command line:

```bash
python run.py
```

Or using the installed console script:

```bash
pc-assistant
```

"""
COMMAND LIST FOR PC ASSISTANT

1. YOUTUBE COMMANDS
   - youtube play "metallica nothing else matters"    # Search and play a video
   - youtube search "python tutorial"                 # Search for videos
   - youtube pause                                    # Pause current video
   - youtube resume                                   # Resume playback
   - youtube mute                                     # Toggle mute
   - youtube fullscreen                               # Toggle fullscreen
   - youtube volume_up                                # Increase YouTube volume
   - youtube volume_down                              # Decrease YouTube volume
   - youtube captions                                 # Toggle captions
   - youtube theater                                  # Toggle theater mode
   - youtube forward                                  # Forward 5 seconds
   - youtube rewind                                   # Rewind 5 seconds

2. VOLUME CONTROL
   - volume up 10                                     # Increase system volume by 10%
   - volume down 5                                    # Decrease system volume by 5%
   - volume set 50                                    # Set system volume to 50%
   - volume mute                                      # Toggle mute
   - volume unmute                                    # Unmute system volume

3. BRIGHTNESS CONTROL
   - brightness up 10                                 # Increase brightness by 10%
   - brightness down 5                                # Decrease brightness by 5%
   - brightness set 70                                # Set brightness to 70%

4. POWER CONTROL
   - power sleep                                      # Put computer to sleep
   - power hibernate                                  # Hibernate computer
   - power shutdown                                   # Shut down computer
   - power restart                                    # Restart computer
   - power logout                                     # Log out current user

5. SCREENSHOT
   - screenshot                                       # Take a screenshot with timestamp
   - screenshot meeting_notes                         # Take screenshot named "meeting_notes.png"

6. SPOTIFY
   - spotify play "coldplay yellow"                   # Search and play a song
   - spotify pause                                    # Pause playback
   - spotify resume                                   # Resume playback
   - spotify next                                     # Skip to next track
   - spotify previous                                 # Go to previous track
   - spotify search "rock playlist"                   # Search for music
   - spotify open                                     # Open Spotify app

7. WEB BROWSING
   - google "weather in new york"                     # Search Google
   - web "github.com"                                 # Open website in browser
   - web "https://stackoverflow.com"                  # Open website with https

8. APP CONTROL
   - open chrome                                      # Open Chrome browser
   - open notepad                                     # Open Notepad
   - open spotify                                     # Open Spotify
   - close chrome                                     # Close Chrome browser
   - close notepad                                    # Close Notepad

9. FILE OPERATIONS
   - file search "budget 2023"                        # Search for files
   - file open "report.docx"                          # Open a file
   - file explore "Downloads"                         # Open file explorer at location

10. MAPS
    - maps "coffee shops near me"                     # Search maps for locations
    - maps "directions to airport"                    # Get directions

11. NETFLIX
    - netflix open                                    # Open Netflix
    - netflix search "stranger things"                # Search for shows
    - netflix play                                    # Play/pause toggle

12. ALARM
    - alarm set "14:30 Meeting with client"           # Set alarm for 2:30 PM
    - alarm 5m "Check oven"                           # Set alarm for 5 minutes from now
    - alarm list                                      # List all alarms
    - alarm cancel "Meeting"                          # Cancel alarm containing "Meeting"
"""

## Integrating with AI Assistants or Chat Applications

The PC Assistant can be easily integrated with AI assistants or chat applications. Here's how:

### Basic Integration

```python
from run import execute_command

# Execute a command and get the result
result = execute_command("volume up 10")

# Check if the command was successful
if result.get('success'):
    print(f"Command succeeded: {result.get('message')}")
else:
    print(f"Command failed: {result.get('error')}")
```

### Structured Command Approach (Recommended)

```python
from run import structured_execute_command

# Execute a command using the structured approach
result = structured_execute_command("volume", "up 10")
# Or with YouTube
result = structured_execute_command("youtube", "play music")

# Check if the command was successful
if result.get('success'):
    print(f"Command succeeded: {result.get('message')}")
else:
    print(f"Command failed: {result.get('error')}")
```

### Complete Example

See `assistant_integration.py` for a complete example of how to integrate with an AI assistant.

```bash
python assistant_integration.py
```

### Response Format

Commands return a standardized response dictionary with these keys:

- `success`: Boolean indicating whether the command executed successfully
- `message`: User-friendly message about the result
- `error`: Error message if any
- `action`: The action that was performed
- `additional_data`: Any additional data returned by the command

## Available Commands

- **volume**: Control system volume
  - `volume up [level]` - Increase volume
  - `volume down [level]` - Decrease volume
  - `volume mute` - Toggle mute
- **brightness**: Control screen brightness
  - `brightness up [level]` - Increase brightness
  - `brightness down [level]` - Decrease brightness
- **power**: Control system power
  - `power shutdown` - Shut down the system
  - `power restart` - Restart the system
  - `power sleep` - Put system to sleep
- **screenshot**: Take a screenshot
  - `screenshot [filename]` - Take screenshot and save it
- **spotify**: Control Spotify
  - `spotify play [song]` - Play a song
  - `spotify pause` - Pause playback
- **youtube**: Control YouTube
  - `youtube play [query]` - Search and play a video
  - `youtube search [query]` - Search for videos
- **google**: Search Google
  - `google [query]` - Search Google
- And many more!

## Creating Custom Commands

You can create your own commands by extending the `Command` base class:

```python
from assistant import Command

class MyCustomCommand(Command):
    """Help text for my custom command."""

    def execute(self, args: str, *_args, **_kwargs) -> dict:
        # Process the command
        # ...

        # Return a standardized response
        return {
            'success': True,
            'action': 'mycustom',
            'message': 'Command executed successfully'
        }
```

## License

MIT License
