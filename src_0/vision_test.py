import base64
import requests
import os
from openai import OpenAI

# OpenAI API Key
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path_0 = "test_imgs/test_0.png"
image_path_1 = "test_imgs/test_1.png"

# Getting the base64 string
base64_image_0 = encode_image(image_path_0)
base64_image_1 = encode_image(image_path_1)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "The robot exectued an action called pickup(Apple). The two images are egocentric observation of the robot before and after the execution. Can you tell which one is before and which one is after execution?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_1}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_0}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())