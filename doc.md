# PC Assistant Command Documentation

This document provides a comprehensive list of all available commands and their usage in the PC Assistant application.

## Table of Contents

- [YouTube Commands](#youtube-commands)
- [Volume Control](#volume-control)
- [Brightness Control](#brightness-control)
- [Power Control](#power-control)
- [Screenshot](#screenshot)
- [Spotify Commands](#spotify-commands)
- [Web Browsing](#web-browsing)
- [Application Control](#application-control)
- [File Operations](#file-operations)
- [Maps Search](#maps-search)
- [Netflix](#netflix)


## YouTube Commands

Control YouTube playback and navigation.

```bash
youtube open                     # Open YouTube homepage
youtube search <query>          # Search for videos
youtube play <query>            # Play the top result for the query
youtube pause                   # Pause current video playback
youtube fullscreen              # Toggle fullscreen mode
```

## Volume Control

Control system audio volume.

```bash
volume up <percentage>          # Increase volume by specified percentage
volume down <percentage>        # Decrease volume by specified percentage
volume set <percentage>         # Set volume to specific percentage
volume mute                     # Mute system audio
volume unmute                   # Unmute system audio
```

## Brightness Control

Control screen brightness.

```bash
brightness up <percentage>      # Increase brightness by specified percentage
brightness down <percentage>    # Decrease brightness by specified percentage
brightness set <percentage>     # Set brightness to specific percentage
```

## Power Control

Control system power states.

```bash
power sleep                     # Put system to sleep
power hibernate                 # Hibernate system
power shutdown                  # Shutdown the system
power restart                   # Restart the system
```

## Screenshot

Take screenshots of the screen.

```bash
screenshot                      # Take a screenshot with timestamp name
screenshot <name>               # Take screenshot with custom name
```

## Spotify Commands

Control Spotify playback and navigation.

```bash
spotify open                    # Open Spotify application
spotify search <query>          # Search for music on Spotify
spotify play <query>            # Search and play music
spotify pause                   # Pause Spotify playback
spotify resume                  # Resume Spotify playback
spotify next                    # Skip to next track
spotify previous                # Go to previous track
```

## Web Browsing

Web search and navigation commands.

```bash
google <query>                  # Search Google for the query
web <url>                      # Open specified website
```

## Application Control

Open and close applications.

```bash
open <application>             # Open specified application
close <application>            # Close specified application
```

## File Operations

File and directory management.

```bash
searchfile <filename>          # Search for a specific file
openfile <filepath>            # Open file with default application
explore                        # Open file explorer in current directory
explore <directory>            # Open file explorer in specified directory
```

## Maps Search

Google Maps search and navigation.

```bash
maps <query>                   # Search Google Maps for location
maps <from> to <to>            # Get directions between locations
```

## Netflix

Netflix control commands.

```bash
netflix open                   # Open Netflix website
netflix search <query>         # Search for content on Netflix
```



## Notes

- All commands are case-insensitive
- Use 'help' to see available commands
- Use 'help <command>' for specific command help
- Type 'exit' or 'quit' to exit the assistant
