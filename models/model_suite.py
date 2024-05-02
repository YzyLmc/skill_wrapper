import openai
import numpy as np



import google.generativeai as genai
#Using Gemini Guide: https://ai.google.dev/gemini-api/docs/get-started/python?_gl=1*1tyxd27*_up*MQ..&gclid=Cj0KCQjw0MexBhD3ARIsAEI3WHJWMzD8_zedKR_LoV2Zc0e23VzI7kMDhS_cHWDTOfrv-ROgfZa5W7waAlwnEALw_wcB

import clip
from PIL import Image
import torch

from IPython.display import display
from IPython.display import Markdown

from lavis.models import load_model_and_preprocess
#Using LAVIS InstructBLIP: https://github.com/salesforce/LAVIS/tree/main

#https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md

#Using LlaVa: https://llava-vl.github.io/ | https://github.com/haotian-liu/LLaVA
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
            self.model = genai.GenerativeModel('gemini-pro-vision')

            genai.configure(api_key = GOOGLE_API_KEY)

        elif model_type == 'clip':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        
        foundation_models = {'openai': run_openai_api, 'gemini': run_gemini_api, 'clip': run_clip_api, 'clipcap': run_clipcap_api}
        self.run_model = foundation_models[model_type]


    
    def run_clip_api(self, **kwargs):

        image = self.preprocess(Image.open(kwargs['image_prompt']).unsqueeze(0).to(self.device))

        #NOTE: 'text_prompts' need to be a list of potential labels to consider with image classification
        text = clip.tokenize(kwargs['text_prompts']).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            label_features = self.model.encode_text(text)
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        label_features /= label_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ label_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)


        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{kwargs['text_prompts'][index]:>16s}: {100 * value.item():.2f}%")
                    
        




    def run_openai_api(self, **kwargs, max_iters = 100, engine = 'gpt-4-turbo', with_vision = False):

        def create_payload_vision(images: list[str], prompt: str, engine='gpt-4-turbo', detail="high", **kwargs):
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


        def create_payload(prompt: str, engine='gpt-4-turbo', **kwargs):

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

            payload = {"model": engine, "messages": messages}

            for (k,v) in kwargs.items():

                if k=='images'

            return payload

        curr_iter = 0
        response = None #populate with response if API call does not throw an error

        while curr_iter < max_iters:

            try:

                if with_vision:
                    payload = create_payload_vision(kwargs['images'], kwargs['prompt'], engine, kwargs)
                else:
                    payload = create_payload(kwargs['prompt'], engine)
                response = openai.Completion.create(payload) #provide payload in the form of a kwargs
                break

            except (openai.error.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as err:

                curr_iter += 1

                print(f'ERROR: [{curr_iter}/{max_iters}] API returns {err.__class__.__name__}: {err}')

                sleep_time = np.random.randint(low=10, high=30)
                time.sleep(sleep_time)
                continue

        
        return response

    
    def run_gemini_api(self, **kwargs, engine = 'gemini-pro-vision', with_vision=False):

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
        
        if with_vision:
            #NOTE: 'image_prompt' is a PIL Image loaded with img = PIL.Image.open('image.jpg')
            response = self.model.generate_content([kwargs['text_prompt'], kwargs['image_prompt']], stream=True)
        else:
            response = model.generate_content(kwargs['text_prompt'], stream=True)

        response.resolve()
        to_markdown(response.text)
        return response.text


class OpensourceModels():

    def __init__ (self, model_type, model_name, finetuned=False):


        '''
        we expand the positive and negative labels into a slightly broader set of verbalizers to exploit word frequencies in natural text (e.g., yes and true for the positive class; no and false for the negative class
        '''

        if model_type == 'instruct_BLIP':
            self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

            if finetuned:
                #TODO: load the trained path for finetuned opensource models
            
            '''
            model names: (blip2_vicuna_instruct, vicuna7b) (blip2_vicuna_instruct, vicuna13b) (blip2_t5, pretrain_flant5xxl)
            '''
            self.model, self.image_preprocessor, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)

        elif model_type == 'LLaVa':

        elif model_type == 'Llama':

        opensource_models = {'instruct_BLIP': run_instruct_blip, 'LLaVa': run_llava, 'Llama': run_llama}
        self.run_model = opensource_models[model_type]

    
    def run_instruct_blip(self, **kwargs):

        image = self.image_preprocessor["eval"](kwargs['image_prompt']).unsqueeze(0).to(self.device)
        response = self.model.generate({"image": kwargs['image_prompt'], "prompt": kwargs['text_prompt'], 
            length_penalty=kwargs['length_penalty'], 
            repetition_penalty = kwargs['repetition_penalty']
            num_beams = kwargs['num_beams'],
            max_length = kwargs['max_len'],
            min_length = kwargs['min_len'],
            top_p = kwargs['top_p'],
            use_nucleus_sampling = kwargs['use_nucleus_sampling']})
           

        return response
        

    def run_llava(self, **kwargs, finetuned=False):
    

    def run_open_flamingo(self, **kwargs, finetuned=False):

        

'''
[DONE] OpenAI GPT4 
[DONE] Google Gemini
Instruct BLIP (with + without fine-tuning)
LLaVa (with + without fine-tuning)
OpenFlamingo
[DONE] CLIP With Pair
Trained Binary Classifier (for each task)
'''