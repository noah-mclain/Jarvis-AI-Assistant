# Personal Assistant Application

A command-line based personal assistant that can execute various commands like opening/closing applications, setting alarms, controlling Spotify, and searching YouTube.

## Features

- **App Control**: Open and close applications on your system
- **Alarms**: Set alarms with custom messages and get notifications when they trigger
- **Spotify Integration**: Control Spotify playback, search for songs, and adjust volume
- **YouTube Search**: Quickly search for videos on YouTube

## Installation

1. Clone this repository or download the source code
2. Make sure you have Python 3.6+ installed
3. No additional dependencies are required for basic functionality

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
