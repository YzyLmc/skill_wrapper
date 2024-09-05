from openai import OpenAI
import base64
import requests
from time import sleep
import logging
import json
import csv
import os
from pathlib import Path
import string
from string import ascii_lowercase
from collections import defaultdict
import numpy as np
import dill
import random
import sys
import copy
from copy import deepcopy
import itertools
from PIL import Image

# OpenAI API Key
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# General utils

def load_from_file(fpath, noheader=True):
    ftype = os.path.splitext(fpath)[-1][1:]
    if ftype == 'pkl':
        with open(fpath, 'rb') as rfile:
            out = dill.load(rfile)
    elif ftype == 'txt':
        with open(fpath, 'r') as rfile:
            if 'prompt' in fpath:
                out = "".join(rfile.readlines())
            else:
                out = [line.strip() for line in rfile.readlines()]
    elif ftype == 'json':
        with open(fpath, 'r') as rfile:
            out = json.load(rfile)
    elif ftype == 'csv':
        with open(fpath, 'r') as rfile:
            csvreader = csv.reader(rfile)
            if noheader:
                fileds = next(csvreader)
            out = [row for row in csvreader]
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
    return out

def save_to_file(data, fpth, mode=None):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, mode if mode else 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, mode if mode else 'w') as wfile:
            wfile.write(data)
    elif ftype == 'json':
        with open(fpth, mode if mode else 'w') as wfile:
            json.dump(data, wfile, sort_keys=True,  indent=4)
    elif ftype == 'csv':
        with open(fpth, mode if mode else 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            writer.writerows(data)
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")

# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def prompt2msg(query_prompt, vision=False):
    """
    Make prompts for GPT-3 compatible with GPT-3.5 and GPT-4.
    Support prompts for
        RER: e.g., data/osm/rer_prompt_16.txt
        symbolic translation: e.g., data/prompt_symbolic_batch12_perm/prompt_nexamples1_symbolic_batch12_perm_ltl_formula_9_42_fold0.txt
        end-to-end translation: e.g., data/osm/osm_full_e2e_prompt_boston_0.txt
    :param query_prompt: prompt used by text completion API (text-davinci-003).
    :return: message used by chat completion API (gpt-3, gpt-3.5-turbo).
    """
    prompt_splits = query_prompt.split("\n\n") if type(query_prompt) == str else query_prompt
    # breakpoint()
    task_description = prompt_splits[0]
    examples = prompt_splits[1: -1]
    query = prompt_splits[-1]

    tag = "text" if vision else "content"
    msg = [{"role": "system", tag: task_description}]
    if len(prompt_splits) > 1:
        msg.append({"role": "user", tag: "\n\n".join(prompt_splits[1:])})
    # breakpoint()
    return msg

class GPT4:
    def __init__(self, engine="gpt-4o", temp=0, max_tokens=500, n=1, stop=['\n\n\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def generate(self, query_prompt):
        '''query_prompt: query with task description and in-contex examples splited with \n\n'''
        complete = False
        ntries = 0
        msg = prompt2msg(query_prompt)
        # breakpoint()
        while not complete and ntries < 15:
            # try:
                raw_responses = client.chat.completions.create(model=self.engine,
                messages=prompt2msg(query_prompt),
                temperature=self.temp,
                n=self.n,
                stop=self.stop,
                max_tokens=self.max_tokens)
                complete = True
            # except:
            #     sleep(30)
            #     logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...")
            #     # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
            #     logging.info("OK continue")
            #     ntries += 1
        if self.n == 1:
            responses = [raw_responses.choices[0].message.content.strip()]
        else:
            responses = [choice["message"]["content"].strip() for choice in raw_responses.choices]
        return responses
    
    def generate_multimodal(self, query_prompt, imgs, max_tokens=500):
        '''separate function on purpose to call multimodal API. It will have the function to have mixed but ordered img & text input'''
        complete = False
        ntries = 0

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        # breakpoint()
        txts = prompt2msg(query_prompt, vision=True)
        payload = {
            "model": self.engine,
            "messages": [],
            "max_tokens": max_tokens}
        msg = {"role": "user", "content": []}
        for line_txt in txts:
            line_txt["type"] = "text"
            msg["content"].append(line_txt)
        for img in imgs:
            base64_img = encode_image(img)
            line_img = {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{base64_img}"
          }}
            msg["content"].append(line_img)
        payload["messages"].append(msg)

        # while not complete and ntries < 15:
        raw_responses = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        complete = True

        if self.n == 1:
            responses = [raw_responses["choices"][0]["message"]["content"].strip()]
        else:
            responses = [choice["message"]["content"].strip() for choice in raw_responses["choices"]]
        return responses

# ai2thor utils
def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    breakpoint()
    pose = copy.deepcopy(event.metadata["actionReturn"])
    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)

if __name__ == "__main__":
    gpt = GPT4()
    imgs = ["test_imgs/test_0.png", "test_imgs/test_1.png"]
    imgs = ["test_imgs/pickup.png"]
    imgs = ["test_imgs/success.png", "test_imgs/failure.png"]
    imgs = ["test_imgs/failure.png"]
    imgs = ["test_imgs/caption.png"]
    imgs = ["test_imgs/1.jpg", "test_imgs/3.jpg"]

    # txt = "The robot is executing pickup() action. There are certain PDDL predicates that are related to the task. Please propose them."
    # txt = "The robot exectued an action called pickup(Apple). The two images are egocentric observation of the robot before and after the execution. Can you tell which one is before and which one is after execution?"
    # txt = 'A robot is executing tasks in the envrionment. Here is what the robot sees from an egocentric view. Please provide a general description of the type of the environment, such as household or facotry, and the robots, such as its mobility and embodiement.'
    txt = "You are a robot,  and all the images are exactly what you see. You are commanded to execute PickUp() action, and the first image shows a successful attempt, while the second one is a failure. How can you guide the robot from the failure image to the successful image using the provided actions: MoveGripperLeft, MoveGripperRight, MoveGripperForward, MoveGripperBackward, MoveGripperUp, MoveGripperDown?"
    # txt = 'If this is what you see exactly, in which picture the robot gripper is on the left of the image? Is it the first one or the second?'
    # txt = "If this is what you see exactly, is the bread located on the left of the table or the right in both pictures?"
    # txt = 'What is on the left of the table?'
    # txt = "If this is what you see exactly, is the bread on the left of the gripper or on the right?"
    # txt = "You are a mobile manipulator robot. Is this image from a first-person view or a third-person view?"
    # txt = "If these images are what you see exactly, from the second image to the first image, which direction did the gripper move to, left or right?"
    
    # txt = "You are a robot,  and all the images are exactly what you see. You are commanded to execute PickUp() action, how can you guide the robot from the second image to the first image using the provided actions: MoveGripperLeft, MoveGripperRight, MoveGripperForward, MoveGripperBackward, MoveGripperUp, MoveGripperDown?"
    txt = "You are a robot,  this image is exactly what you see right now. You are commanded to execute PickUp() action, and you have two actions available: MoveGripperLeft() amd MoveGripperRight(), if this image is exactly what you are seeing from your eyes, what would be your next action?"
    txt = "What are the objects in this picture?"
    txt = """
    There are certain predicates associated with different skills, please find out the ones that have changed their truth value by comparing the visual observation before and after the execution of the skill. You only have to fill the predicates and truth values before and after the execution on the "Effect" line without explanation. Note that not all predicates are necessary to form the effect, you should only select the most essential ones based on the visual observation.
    Skill: PickUp(object, location)
    Predicates: 'AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsReachable(object)', 'IsFreeHand()'
    object = Book, location = Table
    """
    responses = gpt.generate_multimodal(txt, imgs)
    # responses = gpt.generate(txt)
    print(responses)