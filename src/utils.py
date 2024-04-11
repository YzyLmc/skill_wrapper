from openai import OpenAI
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
from copy import deepcopy
import itertools

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

def prompt2msg(query_prompt):
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

    msg = [{"role": "system", "content": task_description}]
    # for example in examples:
    #     if "\n" in example:
    #         example_splits = example.split("\n")
    #         q = '\n'.join(example_splits[0:-1])  # every line except the last in 1 example block
    #         a_splits = example_splits[-1].split(" ")  # last line is the response
    #         q += f"\n{a_splits.pop(0)}"
    #         a = " ".join(a_splits)
    #         msg.append({"role": "user", "content": q})
    #         msg.append({"role": "assistant", "content": a})
    #     else:  # info should be in system prompt, e.g., landmark list
    #         msg[0]["content"] += f"\n{example}"
    # msg.append({"role": "user", "content": query})
    msg.append({"role": "user", "content": "\n\n".join(prompt_splits[1:])})
    # breakpoint()
    return msg

class GPT4:
    def __init__(self, engine="gpt-4-0613", temp=0.0, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def generate(self, query_prompt):
        complete = False
        ntries = 0
        while not complete:
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