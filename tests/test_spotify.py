import pytest
from unittest.mock import patch, MagicMock
from commands.spotify import SpotifyCommand

@pytest.fixture
def spotify_command():
    return SpotifyCommand()

@pytest.fixture
def mock_subprocess():
    with patch('commands.spotify.subprocess') as mock:
        mock.run = MagicMock(return_value=MagicMock(stdout=''))
        mock.Popen = MagicMock()
        yield mock

@pytest.fixture
def mock_platform():
    with patch('commands.spotify.platform') as mock:
        mock.system = MagicMock(return_value='Windows')
        yield mock

@pytest.fixture
def mock_logger():
    with patch('commands.spotify.logger') as mock:
        yield mock

def test_execute_no_args(spotify_command, mock_logger):
    result = spotify_command.execute('')
    assert result is False
    mock_logger.error.assert_called_once_with('No action provided. Usage: spotify <action> [query]')

def test_execute_unknown_action(spotify_command, mock_logger):
    result = spotify_command.execute('unknown')
    assert result is False
    mock_logger.error.assert_called_once_with('Unknown Spotify action: unknown')

def test_play_command(spotify_command, mock_subprocess, mock_platform, mock_logger):
    result = spotify_command.execute('play')
    assert result is True
    mock_subprocess.run.assert_called_with('powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]179)"', shell=True)
    mock_logger.info.assert_called_with('Spotify: Playing')

def test_pause_command(spotify_command, mock_subprocess, mock_platform, mock_logger):
    result = spotify_command.execute('pause')
    assert result is True
    mock_subprocess.run.assert_called_with('powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]179)"', shell=True)
    mock_logger.info.assert_called_with('Spotify: Paused')

def test_next_command(spotify_command, mock_subprocess, mock_platform, mock_logger):
    result = spotify_command.execute('next')
    assert result is True
    mock_subprocess.run.assert_called_with('powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]176)"', shell=True)
    mock_logger.info.assert_called_with('Spotify: Next track')

def test_previous_command(spotify_command, mock_subprocess, mock_platform, mock_logger):
    result = spotify_command.execute('previous')
    assert result is True
    mock_subprocess.run.assert_called_with('powershell -Command "$wshShell = New-Object -ComObject WScript.Shell; $wshShell.SendKeys([char]177)"', shell=True)
    mock_logger.info.assert_called_with('Spotify: Previous track')

def test_search_command_no_query(spotify_command, mock_logger):
    result = spotify_command.execute('search')
    assert result is False
    mock_logger.error.assert_called_once_with('No search query provided. Usage: spotify search <query>')

def test_search_command(spotify_command, mock_subprocess, mock_platform, mock_logger):
    result = spotify_command.execute('search Bohemian Rhapsody')
    assert result is True
    mock_subprocess.Popen.assert_called_with('start spotify:search:Bohemian%20Rhapsody', shell=True)
    mock_logger.info.assert_called_with("Spotify: Searching for 'Bohemian Rhapsody'")

def test_volume_command_invalid(spotify_command, mock_logger):
    result = spotify_command.execute('volume invalid')
    assert result is False
    mock_logger.error.assert_called_once_with('Invalid volume level. Usage: spotify volume <0-100>')

def test_volume_command_windows(spotify_command, mock_subprocess, mock_platform, mock_logger):
    result = spotify_command.execute('volume 50')
    assert result is False
    mock_logger.warning.assert_called_once_with('Setting Spotify volume is not supported on Windows')

def test_ensure_spotify_running_windows(spotify_command, mock_subprocess, mock_platform):
    mock_subprocess.run.return_value.stdout = ''
    spotify_command._ensure_spotify_running('windows')
    mock_subprocess.run.assert_called_with('tasklist /FI "IMAGENAME eq Spotify.exe" /NH', shell=True, capture_output=True, text=True)
    mock_subprocess.Popen.assert_called_with('start spotify:', shell=True)

def test_ensure_spotify_running_exception(spotify_command, mock_subprocess, mock_platform, mock_logger):
    mock_subprocess.run.side_effect = Exception('Test error')
    result = spotify_command._ensure_spotify_running('windows')
    assert result is False
    mock_logger.error.assert_called_once_with('Failed to ensure Spotify is running: Test error')