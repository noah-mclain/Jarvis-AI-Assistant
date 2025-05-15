import requests
from PIL import Image
from io import BytesIO

query = "skyscraper"

print(query)

json_query = {
    "prompt": query,
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "guidance_scale": 9.0,
    "seed": 12345
}
response = requests.post("https://f20d-34-125-240-149.ngrok-free.app/generate", json=json_query)

if response.status_code == 200:

    img = Image.open(BytesIO(response.content))
    img.show()  # Opens the image in your default viewer
    img.save("generated_image.png")
    print("Image generated successfully and saved as 'generated_image.png'.")