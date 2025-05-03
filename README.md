# PC Assistant

A comprehensive command-line based personal assistant for controlling your PC. This assistant can perform various tasks including system control, media playback, file operations, web searches, and more.

## Features

### System Control
- **Volume Control**: Adjust system volume, mute/unmute
- **Brightness Control**: Adjust screen brightness
- **Power Management**: Shutdown, restart, logout, sleep, hibernate
- **Screenshots**: Take and save screenshots

### Application Management
- **Open Applications**: Launch any installed application
- **Close Applications**: Close running applications

### Media Control
- **Spotify**: Search and play music, control playback
- **YouTube**: Search and play videos, control playback
- **Netflix**: Basic Netflix controls

### File Operations
- **Search Files**: Find files and folders on your system
- **Open Files**: Open files with their default applications
- **Explore Directories**: Open file explorer in specific locations

### Web Interaction
- **Google Search**: Search the web directly
- **Open Websites**: Open any website in your default browser
- **Maps Search**: Search locations on Google Maps

### Time Management
- **Set Alarms**: Schedule alarms with custom messages

## Project Structure

```
Commands/
├── assistant.py          # Core assistant framework
├── requirements.txt     # Dependencies
├── assistant/           # Assistant core modules
│   └── __init__.py
├── commands/            # Command implementations
│   ├── __init__.py
│   ├── alarm.py         # Alarm commands
│   ├── app_control.py   # Application control commands
│   ├── file_operations.py # File management commands
│   ├── media.py         # Media module (imports media controls)
│   ├── netflix.py       # Netflix control commands
│   ├── spotify.py       # Spotify control commands
│   ├── system_control.py # System control commands
│   ├── web_search.py    # Web search commands
│   └── youtube.py       # YouTube control commands
└── tests/               # Test modules
    └── test_spotify.py
```

## Installation

1. Clone this repository or download the source code
2. Make sure you have Python 3.6+ installed
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the assistant by executing the main script:

```bash
python assistant.py
```

Once the assistant is running, you can enter commands at the prompt:

```
Assistant> help
```

### Available Commands

#### App Control

```
Assistant> open chrome
Assistant> close notepad
```

#### Alarms

```
Assistant> alarm 14:30 Take a break
Assistant> alarm 5m Quick reminder
Assistant> alarms
Assistant> cancelalarm Take a break
```

#### Spotify

```
Assistant> spotify play
Assistant> spotify pause
Assistant> spotify next
Assistant> spotify previous
Assistant> spotify search Bohemian Rhapsody
Assistant> spotify volume 50
```

#### YouTube

```
Assistant> youtube how to make pasta
```

## Extending the Assistant

You can add new commands by creating new Python files in the `commands` directory. Each command should be a class that inherits from the `Command` base class and implements the `execute` method.

Example:

```python
from assistant import Command

class MyNewCommand(Command):
    """Description of what the command does.

    Usage: mynew <arguments>

    Examples:
        mynew example1
        mynew example2
    """

    def execute(self, args: str, *_args, **_kwargs) -> bool:
        # Implement your command logic here
        return True
```

## Architecture

The assistant uses a command pattern architecture:

- `assistant.py`: Main application with command registry and processing logic
- `commands/`: Directory containing all command implementations
  - `app_control.py`: Commands for opening and closing applications
  - `alarm.py`: Commands for setting and managing alarms
  - `media.py`: Commands for controlling media applications

## License

MIT
