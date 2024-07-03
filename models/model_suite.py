from openai import OpenAI
import base64
import numpy as np
import pandas as pd
import time

import google.generativeai as genai
#Using Gemini Guide: https://ai.google.dev/gemini-api/docs/get-started/python?_gl=1*1tyxd27*_up*MQ..&gclid=Cj0KCQjw0MexBhD3ARIsAEI3WHJWMzD8_zedKR_LoV2Zc0e23VzI7kMDhS_cHWDTOfrv-ROgfZa5W7waAlwnEALw_wcB

#import clip
from PIL import Image
import torch
import pdb
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

#import vertexai
# from vertexai.generative_models import GenerationConfig, GenerativeModel, Part

#from IPython.display import display
#from IPython.display import Markdown

#from lavis.models import load_model_and_preprocess


#from LLaVa.llava.model.builder import load_pretrained_model
#from LLaVa.llava.mm_utils import get_model_name_from_path
#from LLaVa.llava.eval.run_llava import eval_model
#Using LAVIS InstructBLIP: https://github.com/salesforce/LAVIS/tree/main

#https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md

#Using LlaVa: https://llava-vl.github.io/ | https://github.com/haotian-liu/LLaVA



class FoundationModel():

    def __init__(self, model_type):

        assert model_type in set(['openai', 'gemini', 'clip']), "Error: model type for FoundationModel should be either openai, gemini or clip"


        if model_type == 'openai':
            self.model = OpenAI(api_key='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')

            self.model_args = {
                'temperature': 0.7,
                'presence_penalty': 0.4,
                'frequency_penalty': 0.3,
                'top_p': 1.0,
                'stop': '',
                'max_tokens':80,
            }

        elif model_type == 'gemini':
            # vertexai.init(project='test', location="us-central1")
            self.model = genai.GenerativeModel('gemini-pro-vision')
            # self.model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")

            genai.configure(api_key = 'AIzaSyDcllVAUVVmw-YiJKXOljWGBQ4C4r8v4_0')

            self.model_args = {
                'temperature': 0.7,
                'presence_penalty': 0.4,
                'frequency_penalty': 0.3,
                'top_p': 1.0,
                'max_tokens':80,
            }

        elif model_type == 'clip':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        
        foundation_models = {'openai': self.run_openai_api, 'gemini': self.run_gemini_api, 'clip': self.run_clip_api, 'clipcap': self.run_clipcap_api}
        self.run_model = foundation_models[model_type]


    
    def run_clipcap_api(self, kwargs):
        raise NotImplementedError('ERROR: model has not been implemented yet')

    def run_clip_api(self, kwargs):

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
                    
        







    def run_openai_api(self, kwargs, max_iters = 100, engine = 'gpt-4o', with_vision=True):

        #gpt-4-turbo'

        def load_image(image_paths):

            encoded_images = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_image)
            return encoded_images

        def create_payload(context_prompt: str, prompt: str, encoded_images, kwargs):

            messages = [
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": [] }
            ]

            for encoded_image in encoded_images:
                messages[1]["content"].append(
                {'type':'image_url', 'image_url':{'url': f"data:image/png;base64,{encoded_image}"}}
                )
            
            messages[1]["content"].append({'type': 'text', 'text': prompt})

            other_params = {}
            for (k,v) in kwargs.items():

                if k == 'image_paths' or k=='prompt' or k=='context_prompt':
                    continue

                other_params[k] = v

            return messages, other_params

        curr_iter = 0
        self.max_iters = 100
        response = None #populate with response if API call does not throw an error

        while curr_iter < self.max_iters:

            try:

                if with_vision:
                    encoded_images = load_image(kwargs['image_paths'])
                else:
                    encoded_images = None
                    raise NotImplementedError

                messages, other_params = create_payload(kwargs['context_prompt'],kwargs['prompt'], encoded_images, kwargs)

                response = self.model.chat.completions.create(model=engine, messages=messages, temperature=other_params['temperature'], presence_penalty=other_params['presence_penalty'], frequency_penalty=other_params['frequency_penalty'], top_p=other_params['top_p'], stop=other_params['stop'], max_tokens=other_params['max_tokens'])
                break

            except (openai.error.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as err:

                curr_iter += 1

                print(f'ERROR: [{curr_iter}/{max_iters}] API returns {err.__class__.__name__}: {err}')

                sleep_time = np.random.randint(low=10, high=30)
                time.sleep(sleep_time)
                continue

        print(response.to_dict()['choices'][0]['message']['content'])
        return response.to_dict()['choices'][0]['message']['content']

    
    def run_gemini_api(self, kwargs, engine = 'gemini-pro-vision', with_vision=True):

        # config = GenerationConfig(max_output_tokens=kwargs['max_tokens'], temperature=kwargs['temperature'], top_p=kwargs['top_p'], frequency_penalty=kwargs['frequency_penalty'], presence_penalty=kwargs['presence_penalty'])

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
        
        if with_vision:
            #NOTE: 'image_prompt' is a PIL Image loaded with img = PIL.Image.open('image.jpg')

            # encoded_images = [base64.b64encode(open(img, "rb").read()).decode("utf-8") for img in kwargs['image_paths']]
            # image_contents = [Part.from_data(
            #     data=base64.b64decode(enc_img), mime_type="image/png"
            # ) for enc_img in encoded_images]


            # response = self.model.generate_content([kwargs['context_prompt'] + '\n\n'+ kwargs['prompt'], image_contents[0]], generation_config=config)

            image_prompt = [Image.open(img) for img in kwargs['image_paths']]

           
            response = self.model.generate_content([kwargs['context_prompt'] + '\n\n'+ kwargs['prompt'], image_prompt[0]], stream=True)
            
        else:
            response = self.model.generate_content(kwargs['context_prompt'] + '\n\n'+ kwargs['prompt'], generation_config=config)

        response.resolve()
        # to_markdown(response.text)
        
        return response.text


class OpensourceModels():

    def __init__ (self, model_type, model_name=None, finetuned=False):


        '''
        we expand the positive and negative labels into a slightly broader set of verbalizers to exploit word frequencies in natural text (e.g., yes and true for the positive class; no and false for the negative class
        '''

        if model_type == 'instruct_BLIP':
            self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

            if finetuned:
                #TODO: load the trained path for finetuned opensource models
                pass
            
            '''
            model names: (blip2_vicuna_instruct, vicuna7b) (blip2_vicuna_instruct, vicuna13b) (blip2_t5, pretrain_flant5xxl)
            '''
	    #self.model, self.image_preprocessor, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
            self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.model.to(self.device)

        elif model_type == 'PaLME':
            pass
        elif model_type == 'LLaVa':
	    #self.model_path = "liuhaotian/llava-v1.5-7b"
            self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            
        elif model_type == 'Llama':
            pass
        #NOTE: skipping OpenFlamingo because it is poor performing compared to LLava and InstructBLIP
        opensource_models = {'instruct_BLIP': self.run_instruct_blip, 'LLaVa': self.run_llava, 'Llama': self.run_llama}
        self.run_model = opensource_models[model_type]

    
    def run_instruct_blip(self, kwargs):

        #image = self.image_preprocessor["eval"](kwargs['image_prompt']).unsqueeze(0).to(self.device)
        #response = self.model.generate({"image": kwargs['image_prompt'], "prompt": kwargs['text_prompt'], 
        #    "length_penalty": kwargs['length_penalty'], 
        #    "repetition_penalty": kwargs['repetition_penalty'],
        #    "num_beams": kwargs['num_beams'],
        #    "max_length": kwargs['max_len'],
        #    "min_length": kwargs['min_len'],
        #    "top_p": kwargs['top_p'],
        #    "use_nucleus_sampling": kwargs['use_nucleus_sampling']})
        image = Image.open(kwargs['image_prompt']).convert('RGB')
        prompt = kwargs['text_prompt']

        model_inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(device)

        output = self.model.generate(**model_inputs, do_sample=kwargs['do_sample'], num_beams=kwargs['num_beams'], max_length = kwargs['max_length'], min_length = kwargs['min_length'], top_p=kwargs['top_p'], repetition_penalty = kwargs['repetition_penalty'], length_penalty = kwargs['length_penalty'], temperature = kwargs['temperature'])

        response = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        return response
        

    def run_llava(self, kwargs, finetuned=False):
	    #pdb.set_trace() 
        prompt = kwargs['context_prompt'] +'\n\n'+ kwargs['prompt']
        #args = type('Args', (), {
        #    "model_path": self.model_path,
        #    "model_base": None,
        #    "model_name": get_model_name_from_path(self.model_path),
        #    "query": prompt,
        #    "conv_mode": None,
        #    "image_file": kwargs['image_paths'][0],
        #    "sep": ",",
        #    "temperature": 0.7,
        #    "top_p": 1.0,
        #    "num_beams": 1,
        #    "max_new_tokens": 80
        #})()

        #response = eval_model(args)
            
        prompt = "USER: <image>\n{}\nASSISTANT:".format(prompt)
        image = Image.open(kwargs['image_paths'][0])
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generate_ids = self.model.generate(**inputs, max_new_tokens=kwargs['max_tokens'])
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        response = response.split('ASSISTANT:')[1].strip()
        return response

    def run_llama(self, kwargs, finetuned=False):
        pass



class MessagePrompt():

    def __init__(self, skill):

        skill_predicates = {'pickup': ["The robot is facing 'x'", "The robot is not holding 'x'", "The robot arm is empty", "The object 'x' is movable", "The robot is nearby object 'x'", "The path between the object 'x' and robot arm is not obstructed", "The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'", "The object 'x' is on a receptacle"],
        
        'putdown': ["The blue sphere (region of influence) around the robot arm is above the receptacle 'r'", "The receptacle 'r' is not closed", "The receptacle 'r' is not occupied with other objects", "The robot is facing receptacle 'r'", "The robot is holding object 'x'", "The object 'x' is not on receptacle 'r'"],

        'turnon': ["The robot is near object 'x'", "The robot is facing object 'x'", "The robot is not holding any object", "The object 'x' is togglable", "The object 'x' is not on", "The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'"],

        'turnoff': ["The robot is near object 'x'", "The robot is facing object 'x'", "The robot is not holding any object", "The object 'x' is togglable", "The object 'x' is on", "The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'"]
        
        }
        
        self.num_predicates  = len(skill_predicates[skill])
    
    def create_pickup_prompt(self, obj, mode, predicate_idx=None):
        predicates = ["The robot is facing 'x'", "The robot is not holding 'x'", "The robot arm is empty", "The object 'x' is movable", "The robot is nearby object 'x'", "The path between the object 'x' and robot arm is not obstructed", "The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'", "The object 'x' is on a receptacle"]
        prompt = {
            'precondition': f"A robot is attempting to execute an action called PickUp on object 'x' denoted as PickUp(x). The preconditions for PickUp(x) are:\n- The robot is facing 'x'\n- The robot is not holding 'x'\n- The robot arm is empty\n- The object 'x' is movable\n- The robot is nearby object 'x'\n- The path between the object 'x' and robot arm is not obstructed\n- The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'\n- The object 'x' is on a receptacle\n\nIn the image, the robot is attempting to PickUp({obj}). Given the precondition list, can the robot execute PickUp({obj}) (answer Yes/No)? If not, state briefly what preconditions prevent PickUp({obj}) from being executed?",
            'predicate': [f"A robot is attempting to execute an action called PickUp on object 'x' denoted as PickUp(x). One of the predicates that are required for PickUp(x) is {predicates[predicate_idx].lower()}\n\nIn the image, the robot is attempting to PickUp({obj}). The blue sphere represents the region of influence around the robot arm gripper. Does the predicate {predicates[predicate_idx].lower()} evaluate to True or False (answer True/False)?", "A robot is attempting to execute an action called PickUp on object 'x' denoted as PickUp(x). The evaluated preconditions for PickUp(x) are:{}",f"\n\nIn the image, the robot is attempting to PickUp({obj}). Given the precondition list, can the robot execute PickUp({obj}) (answer Yes/No)? If not, state briefly what preconditions prevent PickUp({obj}) from being executed?"],
        }

        return prompt[mode], predicates

    def create_putdown_prompt(self, obj, receptacle, mode, predicate_idx=None):
        predicates = ["The blue sphere (region of influence) around the robot arm is above the receptacle 'r'", "The receptacle 'r' is not closed", "The receptacle 'r' is not occupied with other objects", "The robot is facing receptacle 'r'", "The robot is holding object 'x'", "The object 'x' is not on receptacle 'r'"]
        prompt = {
            'precondition': f"A robot is attempting to execute an action called PutDown for object 'x' on receptacle 'r' denoted as PutDown(x,r). The preconditions for PutDown(x,r) are:\n- The blue sphere (region of influence) around the robot arm is above the receptacle 'r'\n- The receptacle 'r' is not closed\n- The receptacle 'r' is not occupied with other objects\n- The robot is facing receptacle 'r'\n- The robot is holding object 'x'\n- The object 'x' is not on receptacle 'r'\n\nIn the image, the robot is attempting to PutDown({obj}, {receptacle}). Given the precondition list, can the robot execute PutDown({obj},{receptacle}) (answer Yes/No)? If not, state briefly what preconditions prevent PutDown({obj}, {receptacle}) from being executed?",
            'predicate': [f"A robot is attempting to execute an action called PutDown for object 'x' on receptacle 'r' denoted as PutDown(x,r). One of the predicates that are required for PutDown(x,r) is {predicates[predicate_idx].lower()}\n\nIn the image, the robot is attempting to PutDown({obj}, {receptacle}). The blue sphere represents the region of influence around the robot arm gripper. Does the predicate {predicates[predicate_idx].lower()} evaluate to True or False (answer True/False)?", "A robot is attempting to execute an action called PutDown for object 'x' on receptacle 'r' denoted as PutDown(x,r). The evaluated preconditions for PutDown(x,r) are:{}",f"\n\nIn the image, the robot is attempting to PutDown({obj}, {receptacle}). Given the precondition list, can the robot execute PutDown({obj},{receptacle}) (answer Yes/No)? If not, state briefly what preconditions prevent PutDown({obj}, {receptacle}) from being executed?"],
        }
        return prompt[mode], predicates

    def create_turnon_prompt(self, obj, mode, predicate_idx=None):
        predicates  = ["The robot is near object 'x'", "The robot is facing object 'x'", "The robot is not holding any object", "The object 'x' is togglable", "The object 'x' is not on", "The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'"]

        prompt = {
            'precondition': f"A robot is attempting to execute an action called ToggleOn for object 'x' denoted as ToggleOn(x). The preconditions for ToggleOn(x) are:\n-The robot is near object 'x'\n- The robot is facing object 'x'\n- The robot is not holding any object\n- The object 'x' is togglable\n- The object 'x' is not on\n- The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'\n\nIn the image, the robot is attempting to ToggleOn({obj}). Given the precondition list, can the robot execute ToggleOn({obj}) (answer Yes/No)? If not, state briefly what preconditions prevent ToggleOn({obj}) from being executed?",

            'predicate': [f"A robot is attempting to execute an action called ToggleOn for object 'x' denoted as ToggleOn(x). One of the predicates that are required for ToggleOn(x) is that {predicates[predicate_idx].lower()}\n\nIn the image, the robot is attempting to ToggleOn({obj}). The blue sphere represents the region of influence around the robot arm gripper. Does the predicate {predicates[predicate_idx].lower()} evaluate to True or False (answer True/False)?", "A robot is attempting to execute an action called ToggleOn for object 'x' denoted as ToggleOn(x). The evaluated preconditions for ToggleOn(x) are:{}",f"\n\nIn the image, the robot is attempting to ToggleOn({obj}). Given the precondition list, can the robot execute ToggleOn({obj}) (answer Yes/No)? If not, state briefly what preconditions prevent ToggleOn({obj}) from being executed?"],
        }

        return prompt[mode], predicates
    
    def create_turnoff_prompt(self, obj, mode, predicate_idx=None):

        predicates  = ["The robot is near object 'x'", "The robot is facing object 'x'", "The robot is not holding any object", "The object 'x' is togglable", "The object 'x' is on", "The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'"]

        prompt = {
            'precondition':f"A robot is attempting to execute an action called ToggleOff for object 'x' denoted as ToggleOff(x). The preconditions for ToggleOff(x) are:\n-The robot is near object 'x'\n- The robot is facing object 'x'\n- The robot is not holding any object\n- The object 'x' is togglable\n- The object 'x' is on\n- The blue sphere (region of influence) around the robot arm gripper overlaps with object 'x'\n\nIn the image, the robot is attempting to ToggleOff({obj}). Given the precondition list, can the robot execute ToggleOff({obj}) (answer Yes/No)? If not, state briefly what preconditions prevent ToggleOff({obj}) from being executed?",
            'predicate': [f"A robot is attempting to execute an action called ToggleOff for object 'x' denoted as ToggleOff(x). One of the predicates that are required for ToggleOff(x) is that {predicates[predicate_idx].lower()}\n\nIn the image, the robot is attempting to ToggleOff({obj}). The blue sphere represents the region of influence around the robot arm gripper. Does the predicate {predicates[predicate_idx].lower()} evaluate to True or False (answer True/False)?", "A robot is attempting to execute an action called ToggleOff for object 'x' denoted as ToggleOff(x). The evaluated preconditions for ToggleOff(x) are:{}",f"\n\nIn the image, the robot is attempting to ToggleOff({obj}). Given the precondition list, can the robot execute ToggleOff({obj}) (answer Yes/No)? If not, state briefly what preconditions prevent ToggleOff({obj}) from being executed?"],
        }
        return prompt[mode], predicates


    def create_prompt_content(self, mode):

        context_dict = {'precondition':'You will be given visual observations of a robot in simulation attempting to perform certain actions. Given visual observations and descriptions about the action precondition or effects, you need to determine if the action can/has been successfully completed',
                        'predicate': 'You will be given visual observations of a robot in simulation attempting to perform certain actions. Given visual observations and descriptions about the predicates that form the action preconditions, you need to determine whether the given predicate evaluates to true or false',
        }

        return context_dict[mode]



'''
[DONE] OpenAI GPT4 
[DONE] Google Gemini
[HALF] Instruct BLIP (with + without fine-tuning)
LLaVa (with + without fine-tuning)
OpenFlamingo
[DONE] CLIP With Pair
Trained Binary Classifier (for each task)
'''

def run_model_preconditions(dataframe, message_generator, model):
    responses = []

    for index, row in dataframe.iterrows():

        
        

        obj = row['argument']
	    #receptacle = row['argument'].split(',')[1].split(')')[0].strip()

        context = message_generator.create_prompt_content(mode='precondition')
        prompt, predicates = message_generator.create_turnoff_prompt(obj)

        model_args = {
            'image_paths': [row['image']],
            'context_prompt': context,
            'prompt': prompt,
        }

        for k, v in model.model_args.items():
            model_args[k] = v

	    #pdb.set_trace()
        response = model.run_model(model_args)
        

        responses.append(response)
    
    return responses

def run_model_predicates(dataframe, message_generator, model, skill):
    responses = []
    joined_predicate_responses = []

    for index, row in tqdm(dataframe.iterrows()):

        predicate_responses = []

        for predicate_idx in range(message_generator.num_predicates):

            if skill == 'putdown':

                obj = row['argument'].split(',')[0].split('(')[1].strip()
                receptacle = row['argument'].split(',')[1].split(')')[0].strip()

                prompt, predicates = message_generator.create_putdown_prompt(obj, receptacle, mode='predicate', predicate_idx = predicate_idx)
            else:

                obj = row['argument']

                prompt_generator_for_skill = {'pickup': message_generator.create_pickup_prompt, 'turnon': message_generator.create_turnon_prompt, 'turnoff': message_generator.create_turnoff_prompt}
                prompt, predicates = prompt_generator_for_skill[skill](obj, mode='predicate', predicate_idx = predicate_idx)


            context = message_generator.create_prompt_content(mode='predicate')
            

            model_args = {
                'image_paths': [row['image']],
                'context_prompt': context,
                'prompt': prompt[0],
            }

            for k, v in model.model_args.items():
                model_args[k] = v

            if 'max_tokens' in model.model_args.keys():
                model_args['max_tokens'] = 5

            response = model.run_model(model_args)
            predicate_responses.append(response)
            time.sleep(0.2)
        

        # pdb.set_trace()
        overall_context = message_generator.create_prompt_content(mode='precondition')
        

        predicates_segment = []

        for response, predicate in zip(predicate_responses, predicates):
            predicates_segment.append('- {}: {}'.format(predicate, response))
        
        predicates_segment = '\n'.join(predicates_segment)

        overall_prompt = prompt[1].format('\n' + predicates_segment) + prompt[2]


        model_args = {
            'image_paths': [row['image']],
            'context_prompt': overall_context,
            'prompt': overall_prompt
        }
       

        for k, v in model.model_args.items():
            model_args[k] = v

        # pdb.set_trace()
        response = model.run_model(model_args)
        responses.append(response)
        joined_predicate_responses.append(';'.join(predicate_responses))
    
    return responses, joined_predicate_responses
        





if __name__ == '__main__':
    pdb.set_trace()
    model = FoundationModel('gemini')
    # model = OpensourceModels(model_type='instruct_BLIP')

    message_generator = MessagePrompt('pickup')

    dataframe = pd.read_csv('./pick_up_data.csv')

    
    responses, predicate_responses = run_model_predicates(dataframe, message_generator, model, 'pickup')

    pdb.set_trace()
    dataframe['responses'] = responses
    dataframe['predicate_responses'] = predicate_responses
    dataframe.to_csv('temp.csv')

    


'''
MODEL SETUP OPENAI GPT
model_args = model.model_args
model_args['image_paths'].append(row['image'])
model_args['prompt'] = message_generator.create_turnoff_prompt(obj)


response = model.run_model(model_args,with_vision=True,engine = 'gpt-4o')
'''
