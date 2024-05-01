import openai
import numpy as np



import google.generativeai as genai
#Using Gemini Guide: https://ai.google.dev/gemini-api/docs/get-started/python?_gl=1*1tyxd27*_up*MQ..&gclid=Cj0KCQjw0MexBhD3ARIsAEI3WHJWMzD8_zedKR_LoV2Zc0e23VzI7kMDhS_cHWDTOfrv-ROgfZa5W7waAlwnEALw_wcB

import clip
import clipCAP

from IPython.display import display
from IPython.display import Markdown


'''
OpenAI GPT4
Google Gemini
Instruct BLIP (with + without fine-tuning)
LLaVa (with + without fine-tuning)
OpenFlamingo
CLIP With Pair
Trained Binary Classifier (for each task)
'''


'''
"""Example usage of GPT4-V API.

Usage:

    OPENAI_API_KEY=<your_api_key> python3 gpt4v.py \
        [<path/to/image1.png>] [<path/to/image2.jpg>] [...] "text prompt"

Example:

    OPENAI_API_KEY=xxx python3 gpt4v.py photo.png "What's in this photo?"
"""

from pprint import pprint
import base64
import json
import mimetypes
import os
import requests
import sys


api_key = os.getenv("OPENAI_API_KEY")


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"


def create_payload(images: list[str], prompt: str, model="gpt-4-vision-preview", max_tokens=100, detail="high"):
    """Creates the payload for the API request."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    for image in images:
        base64_image = encode_image(image)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": base64_image,
                "detail": detail,
            }
        })

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }


def query_openai(payload):
    """Sends a request to the OpenAI API and prints the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py [image1.jpg] [image2.png] ... \"Text Prompt\"")
        sys.exit(1)

    # All arguments except the last one are image paths
    image_paths = sys.argv[1:-1]

    # The last argument is the text prompt
    prompt = sys.argv[-1]

    payload = create_payload(image_paths, prompt)
    response = query_openai(payload)
    pprint(response)


if __name__ == "__main__":
    main()


'''


'''

openai.api_key = os.getenv("OPENAI_API_KEY")
image = b"..."  # binary image stream
completion = openai.ChatCompletion.create(
    model="gpt-4-0xxx",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant and can describe images.",
        },
        {
            "role": "user",
            "content": ["What's in this screenshot?", {"image": image}],
        },
    ],
)
print(completion["choices"][0]["message"]["content"])
'''


class FoundationModel():

    def __init__(self, model_type):

        assert model_type in set(['openai', 'gemini', 'clip']), "Error: model type for FoundationModel should be either openai, gemini or clip"


        if model_type == 'openai':
            openai.api_key = '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'

        elif model_type == 'gemini':
            genai.configure(api_key = GOOGLE_API_KEY)

        elif model_type == 'clip'
            self.clip_model = ??

        
        foundation_models = {'openai': run_openai_api, 'gemini': run_gemini_api, 'clip': run_clip_api}
        self.run_model = foundation_models[model_type]


    
    def run_clip_api(self, **kwargs):
    
        




    def run_openai_api(self, **kwargs, max_iters = 100, engine = 'gpt-4-turbo', with_vision = False):

        def create_payload(images: list[str], prompt: str, engine='gpt-4-turbo', detail="high", **kwargs):
            """Creates the payload for the API request."""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ]

            for image in images:
                base64_image = encode_image(image)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": detail,
                    }
                })


            payload = { "model": engine, "messages": messages}
            for (k,v) in kwargs.items():

                if k == 'images' or k=='prompt':
                    continue

                payload[k] = v

            return payload



        curr_iter = 0
        response = None #populate with response if API call does not throw an error

        while curr_iter < max_iters:

            try:

                if with_vision:
                    payload = create_payload_vision(kwargs['images'], kwargs['prompt'], engine, kwargs)
                else:
                    payload = create_payload()
                response = openai.Completion.create(payload) #provide payload in the form of a kwargs
                break

            except (openai.error.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as err:

                curr_iter += 1

                print(f'ERROR: [{curr_iter}/{max_iters}] API returns {err.__class__.__name__}: {err}')

                sleep_time = np.random.randint(low=10, high=30)
                time.sleep(sleep_time)
                continue

        
        return response

    
    def run_gemini_api(self, **kwargs, engine = 'gemini-pro-vision'):

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        response = model.generate_content([kwargs['text_prompt'], kwargs['image_prompt']], stream=True)
        response.resolve()
        to_markdown(response.text)