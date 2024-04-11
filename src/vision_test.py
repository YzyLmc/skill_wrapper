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
image_path_0 = "test_0.png"
image_path_1 = "test_0.png"

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
          "text": "The robot executes the skill: pickup(apple). What are the change of truth values regarding the predicates: hand_is_empty, picked_up(Apple)?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_0}",
            "url": f"data:image/jpeg;base64,{base64_image_1}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())