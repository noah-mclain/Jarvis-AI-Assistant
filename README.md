# Jarvis AI Assistant Desktop Application

A standalone desktop application for the Jarvis AI Assistant that runs without requiring an internet connection.

## Features

- Modern, responsive UI built with React and Tailwind CSS
- Offline functionality - no internet connection required
- Native desktop application experience
- Chat history and conversation management
- Customizable themes and settings

## Technologies Used

- **Frontend**: React, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Flask (Python)
- **Desktop Wrapper**: PyWebView
- **Build Tools**: Vite, Python

## Prerequisites

- Python 3.7 or higher
- Node.js 14 or higher
- npm

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/noah-mclain/fluid-jarvis-whisper.git
   cd fluid-jarvis-whisper
   ```

2. Install JavaScript dependencies:

   ```sh
   npm install
   ```

3. Install Python dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

### Development Mode

For macOS/Linux:

```sh
npm run start:dev
```

For Windows:

```sh
npm run start:win:dev
```

This will start both the React development server and the Flask backend.

### Production Mode

For macOS/Linux:

```sh
npm run start
```

For Windows:

```sh
npm run start:win
```

This will build the React application and then start the Flask server to serve the built files.

## Development

### Frontend Development Only

To work on the frontend in development mode without the backend:

```sh
npm run dev
```

This will start the Vite development server with hot reloading.

### Backend Development Only

To work on the Flask backend separately:

```sh
cd server
flask run --debug
```

## Project Structure

- `/src` - React frontend code
- `/server` - Flask backend code
- `/dist` - Built frontend assets (created after build)
- `main.py` - PyWebView application entry point
- `build.py` - Build script

## Integrating with Your Jarvis AI Assistant

This application is designed to work fully offline with your Jarvis AI assistant. There are two ways to integrate:

### Automatic Integration (During Build)

When you run the build script (`python build.py`), you'll be prompted to provide the path to your Jarvis AI project. The script will:

1. Copy the necessary Python files from your Jarvis project
2. Copy the model files to the appropriate location
3. Set up the integration automatically

### Manual Integration

If you prefer to set up the integration manually:

1. Create a directory structure:

   ```sh
   server/jarvis/
   data/models/
   ```

2. Copy your Jarvis AI implementation files to `server/jarvis/`:

   - `__init__.py`
   - `assistant.py` - This should contain a `JarvisAI` class with a `process_query(query)` method
   - Any other necessary files

3. Copy your model files to `data/models/`

4. The application will automatically detect and use your Jarvis AI implementation

### Integration Requirements

Your Jarvis AI implementation should:

1. Have a `JarvisAI` class in `server/jarvis/assistant.py`
2. Implement a `process_query(query)` method that takes a string and returns a string
3. Work completely offline without requiring internet access

## License

MIT
