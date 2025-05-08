import webview

html = """
<!DOCTYPE html>
<html>
<head>
    <title>PyWebView Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PyWebView Test</h1>
        <p>If you can see this page, PyWebView is working correctly!</p>
        <button onclick="alert('Button clicked!')">Click Me</button>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    # Create a window with PyWebView
    window = webview.create_window(
        title="PyWebView Test",
        html=html,
        width=800,
        height=600,
        resizable=True,
        text_select=True,
        confirm_close=False,
        background_color='#ffffff'
    )

    # Start the PyWebView event loop
    webview.start(debug=True)
