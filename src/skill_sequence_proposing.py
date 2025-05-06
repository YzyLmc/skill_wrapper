
from openai import OpenAI
import numpy as np
import os
import copy
import re
from typing import Union
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util as st_utils
import torch
import base64

from RCR_bridge import LiftedPDDLAction, RCR_bridge, PDDLState, Parameter, generate_possible_groundings
from data_structure import Skill, PredicateState
from utils import load_from_file

def lifted_pred_list_to_predicate_dictionary(lifted_pred_list):
    """
    Args:
        lifted_pred_list :: list[Predicate]
    Returns:
        preddicate_dictionary :: dict[str, str]
    """
    return {str(pred): pred.semantic for pred in lifted_pred_list}

class SkillSequenceProposing():
    def __init__(self, lifted_pred_list={}, skill2operator={}, tasks=None, init_state=None, prompt_fpath="prompts/skill_sequence_proposal.yaml", task_config_fpath="task_config/dorfl.yaml"):
        '''
        DONE:
        5) Hyperparameters for LLM
        1) Coverage ==> what type of entropy? do we take final entropy or difference in entropy for information gain? (maybe value function type: where repeatedly taking samples of same pair leads to stagnation)
        1.5) Choice of skill pairs for "least explored" in task proposing prompt
        '''
        self.task_config = load_from_file(task_config_fpath)
        self.prompt_dict = load_from_file(prompt_fpath)
        #predicate dictionary: {predicate: definition/description}
        self.predicate_dictionary = lifted_pred_list_to_predicate_dictionary(lifted_pred_list)
        self.operator_dictionary = skill2operator
        self.skill_dictionary = {lifted_skill: {'arguments': {ptype: sem for ptype, sem in lifted_skill.semantics.items()}} for lifted_skill in self.task_config['skills'].values()}
        self.operator_to_skill = {k: re.sub(r'_\d+', '', k) for (k,v) in self.operator_dictionary.items()}
        self.init_state = init_state

        #global object set for the scene
        self.objects_type_dict = self.task_config['objects']
        self.env_description = self.task_config['Env_description']
        self.curr_observation  = self.task_config['Initial_observation']['img_fpath']

        #global frequency count for all pairs of skills 
        self.skill_to_index = {x: i for i,x in enumerate(self.skill_dictionary.keys())}
        if tasks is not None:
            self.attempted_skill_pair_count, self.curr_skill_count = self.get_skill_pair_matrix_from_tasks(tasks)
        else:
            self.attempted_skill_pair_count = np.zeros((len(self.skill_dictionary.keys()), len(self.skill_dictionary.keys())))
            self.curr_skill_count = 0
        self.curr_shannon_entropy = 0.0

        #LLM hyperparameters: GPT4O
        self.task_generation_args = {
            'temperature': 0.6,
            'presence_penalty': 0.3,
            'frequency_penalty': 0.35,
            'top_p': 1.0,
            # 'max_tokens':550,
            'engine': 'gpt-4o',
            # 'engine': 'o3-mini',
            'stop': ''
        }

        #GPT4O model to query for new proposed tasks
        self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        #embedding model for grounding LLM output to groundable/executable skills and objects
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('mps') # for my m1 macbook: mps
        self.embedding_model = SentenceTransformer('stsb-roberta-large').to(self.device)
        self.all_skill_embeddings = self.embedding_model.encode([skill.name for skill in list(self.skill_dictionary.keys())], batch_size=32, convert_to_tensor=True, device=self.device)
        self.all_param_embeddings = self.embedding_model.encode(list(self.objects_type_dict.keys()), batch_size=32, convert_to_tensor=True, device=self.device)

        #parameters for kernel density estimation
        self.h = 1
        #scaling parameters for pareto-optimal task selection
        self.k = 10 #set period after how many skill executions to switch mode
        #all alphas are in the range [1,3]
        self.chainability_alpha = lambda x: 1
        self.entropy_gain_alpha = lambda x: np.cos( ( np.pi / self.k) * x) + 2

    def get_skill_pair_matrix_from_tasks(self, tasks):
        """
        Read the skill executions from previous tasks, and count the number.

        Args:
            tasks :: dict(task_name: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool)))
        Returns:
            skill_pair_count :: numpy array
            skill_count :: int
        """
        # skill pair count matrix
        skill_pair_count = np.zeros((len(self.skill_dictionary.keys()), len(self.skill_dictionary.keys())))
        # number of skills have been executed
        skill_count = 0
        for _, task_meta in tasks.items():
            last_skill = None
            for step, state_meta in task_meta.items():
                if not step == 0:
                    curr_skill = state_meta['skill']
                    if last_skill is not None:
                        skill1_idx = self.skill_to_index[last_skill.lifted()]
                        skill2_idx = self.skill_to_index[curr_skill.lifted()]
                        skill_pair_count[skill1_idx, skill2_idx] += 1
                    skill_count += 1

        return skill_pair_count, skill_count

    def create_foundation_model_prompt(self):
        skill_prompts = []
        for skill in self.skill_dictionary:
            types = skill.types
            skill_prompt = str(skill) + '\n' + '\n'.join([t + ': ' + skill.semantics[t] for t in types])
            skill_prompts.append(skill_prompt)
        objects_with_types = [f"{obj}: {str(types)}" for obj, types in self.task_config['objects'].items()]
        
        least_explored_skills = self.get_least_explored_skills()
        prompt_context = self.prompt_dict[self.task_config['env']]["system_prompt"]
        prompt = self.prompt_dict[self.task_config['env']]["prompt"]
        prompt = prompt.replace("[SKILL_PROMPT]", "\n\n".join(skill_prompts))\
                        .replace("[OBJECT_IN_SCENE]", str("\n".join(objects_with_types)))\
                        .replace("[ENV_DESCRIPTION]", self.env_description)\
                        .replace("[LEAST_EXPLORED_SKILLS]", ','.join(least_explored_skills))
        
        return prompt, prompt_context

    '''
   COVERAGE:  Functions for entropy computation + Functions to determine least explored tasks
    '''
    def compute_entropy_for_one_sequence(self, skill_sequence: list[Skill]):
        """
        new_shannon_entropy :: entropy value after executing the skill sequence
        new_skill_pair_count :: updated skill pair count matrix
        """
        new_skill_pair_count = copy.deepcopy(self.attempted_skill_pair_count)
        p1 = 0; p2 = min(1, len(skill_sequence))
        while p2 < len(skill_sequence):
            skill1_idx = self.skill_to_index[skill_sequence[p1].lifted()]
            skill2_idx = self.skill_to_index[skill_sequence[p2].lifted()]
            new_skill_pair_count[skill1_idx, skill2_idx] += 1
            p1 = p2
            p2 += 1

        normalized_skill_pair_prob =  new_skill_pair_count / np.sum(new_skill_pair_count) if np.sum(new_skill_pair_count) > 0 else new_skill_pair_count
        log_skill_pair_prob = np.where(normalized_skill_pair_prob > 0.0 , np.log(normalized_skill_pair_prob), 0.0)
        new_shannon_entropy = np.sum(-1 * normalized_skill_pair_prob * log_skill_pair_prob)
        return new_shannon_entropy, new_skill_pair_count
    
    def compute_shannon_entropy(self, skill_sequences: list[list[Skill]]):
        normalized_skill_pair_prob =  self.attempted_skill_pair_count / np.sum(self.attempted_skill_pair_count) if np.sum(self.attempted_skill_pair_count) > 0 else  self.attempted_skill_pair_count 
        log_skill_pair_prob = np.where(normalized_skill_pair_prob > 0 , np.log(normalized_skill_pair_prob), 0)
        curr_shannon_entropy = np.sum(-1 * normalized_skill_pair_prob * log_skill_pair_prob)
        
        skill_sequence_entropy_gains = []
        skill_sequence_skill_counts = []
        #measure entropy gain for each task
        for skill_sequence in skill_sequences:
            entropy, counts = self.compute_entropy_for_one_sequence(skill_sequence)
            skill_sequence_entropy_gains.append(entropy - curr_shannon_entropy) #entropy gain is maximum of difference
            skill_sequence_skill_counts.append(counts)
        
        return np.array(skill_sequence_entropy_gains), skill_sequence_skill_counts

    def get_least_explored_skills(self, k=5):
        # Find the minimum value in the matrix
        min_value = np.min(self.attempted_skill_pair_count)
        # Find all indices where the value equals the minimum
        min_indices = np.argwhere(self.attempted_skill_pair_count == min_value)
        # If there are more than k minimum entries, randomly select k of them
        if len(min_indices) > k:
            selected_indices = min_indices[np.random.choice(len(min_indices), size=k, replace=False)]
        else:
            selected_indices = min_indices

        least_explored_pairs = []
        skill_list = list(self.skill_to_index.keys())
        for idx1, idx2 in selected_indices:
            skill1 = skill_list[idx1]
            skill2 = skill_list[idx2]
            least_explored_pairs.append(f'({skill1}, {skill2})')

        return least_explored_pairs
    
    '''
    CHAINABILITY
    '''
    def get_skill_sequence_executability(self, skill_sequence: list[Skill], init_state: Union[PredicateState , None]) -> list[bool]:
        """
        self.operator_dictionary :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str})]}
        """

        def apply_skill(grounded_skill, pddl_state: PDDLState, pid2type, type_dict) -> Union[bool, PDDLState]:
            """
            Check if there exist an operator that makes the skill executable.
            Returns:
                bool :: if the skill is executable
                pddl_state :: next state if executable, the original state otehrwise
            """
            for lifted_operator, pid2type in self.operator_dictionary[grounded_skill.lifted()]:
                possible_groundings = generate_possible_groundings(pid2type, type_dict, fixed_grounding=grounded_skill.params)
                for grounding in possible_groundings:
                    param_name2param_object = {str(param): param.get_grounded_parameter(grounding[int(str(param).split("_p")[-1])]) for param in lifted_operator.parameters if not str(param).startswith("_")} | {'_p1': Parameter(None, "", None)}
                    grounded_operator: LiftedPDDLAction = lifted_operator.get_grounded_action(param_name2param_object, 0)
                    if grounded_operator.check_applicability(pddl_state):
                        return True, grounded_operator.apply(pddl_state)
            return False, pddl_state
        
        if init_state is None: # No predicate has been proposed yet. Chainability are always the same, i.e, always chainable
            return [True] * len(skill_sequence)
            
        executable_list = []
        bridge = RCR_bridge()
        pddl_state = bridge.predicatestate_to_pddlstate(init_state)
        for grounded_skill in skill_sequence:
            executable, pddl_state = apply_skill(grounded_skill, pddl_state)
            executable_list.append(executable)
        return executable_list
    
    def chainability_per_sequence(self, executable_list):
        return - abs(float(sum(executable_list)/ (len(executable_list) if len(executable_list) > 0 else 1)) - 0.5)
    
    def compute_chainability(self, skill_sequences):
        """
        Chainability counted as the ratio executable skills in the sequence
        """
        skill_sequence_chainabilities = []
        for skill_sequence in skill_sequences:
            executable_list = self.get_skill_sequence_executability(skill_sequence, self.init_state)
            skill_sequence_chainability = self.chainability_per_sequence(executable_list)
            skill_sequence_chainabilities.append(skill_sequence_chainability)
        return np.array(skill_sequence_chainabilities)
        
    '''
    OVERALL SCORING: Function to run general scoring at the task level, combining coverage and chainability
    '''
    def generate_scores_and_choose_skill_sequence(self, skill_sequences):
        #run the 2 scoring functions
        task_chainabilities = self.compute_chainability(skill_sequences)
        task_entropy_gains, task_skill_counts = self.compute_shannon_entropy(skill_sequences)

        #collect the maximum and minimum for each metric
        max_entropy_gain = max(task_entropy_gains); min_entropy_gain = min(task_entropy_gains)
        max_chainability = max(task_chainabilities); min_chainability = min(task_chainabilities)

        #combine the outputted lists for task chainability and entropy gain to ensure pareto optimality
        pareto_front_set = {0: (task_entropy_gains[0], task_chainabilities[0])}

        curr_idx = 1
        for (eg_new, chain_new) in zip(task_entropy_gains[1:], task_chainabilities[1:]):
            domination = False
            for k in list(pareto_front_set.keys()):
                (eg2, chain2) = pareto_front_set[k]
                #if the new score combo dominates something from the pareto-front set, remove the original example from pareto-front set 
                if (eg_new >= eg2) and (chain_new >= chain2) and (eg_new > eg2 or chain_new > chain2):
                    domination = True
                    del pareto_front_set[k]
                    pareto_front_set[curr_idx] = (eg_new, chain_new)
                #if pareto set already dominates new example then skip
                elif (eg2 >= eg_new) and (chain2 >= chain_new) and (eg2 > eg_new or chain2 > chain_new):
                    domination = True
                    continue   
            #if some variables dominated then new example is part of pareto set 
            if not domination:
                pareto_front_set[curr_idx] = (eg_new, chain_new)
            curr_idx += 1
        
        #define the weightages for each metric based on the current number of tasks and choose something from pareto-front that maximizes the desired combination
        chain_alpha = self.chainability_alpha(self.curr_skill_count)
        entr_alpha = self.entropy_gain_alpha(self.curr_skill_count)
        max_score = float('-inf'); max_score_idx = None

        for (k, v) in pareto_front_set.items():
            if v is None:
                continue
            (entr, chain) = v
            entropy_score = (entr - min_entropy_gain)/(max_entropy_gain - min_entropy_gain) if (max_entropy_gain - min_entropy_gain) > 0 else 0
            chainability_score = (max_chainability - chain) / (max_chainability - min_chainability)  if (max_chainability - min_chainability) > 0 else 0
            combined_score = entr_alpha * (entropy_score) + chain_alpha * (chainability_score)
            if combined_score > max_score:
                max_score = combined_score
                max_score_idx = k

        #find the task with the maximum combined score + set the current skill count to that task's skill count + set the current task execution matrix counts
        self.curr_skill_count += len(skill_sequences[max_score_idx])
        self.attempted_skill_pair_count = task_skill_counts[max_score_idx] # this should be calculated only when updating the tasks, together with skill_count
        return skill_sequences[max_score_idx]
    
    '''
    FOUNDATION MODEL: Functions to run LLM (GPT4-O) as well as generate dynamic prompting structure using least explored tasks
    '''
    def run_foundation_model(self, prompt_context, prompt, image_paths):

        #NOTE: output the task list that is grounded and decomposed into a dictionary of {task: [[list of skills with arguments], max executable steps]}
        def load_image(image_paths):
            encoded_images = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_image)
            return encoded_images
        
        def create_payload(prompt_context: str, prompt: str, encoded_images):
            messages = [
                {"role": "system", "content": prompt_context},
                {"role": "user", "content": [] }
            ]
            for encoded_image in encoded_images:
                messages[1]["content"].append(
                {'type':'image_url', 'image_url':{'url': f"data:image/png;base64,{encoded_image}"}}
                )
            messages[1]["content"].append({'type': 'text', 'text': prompt})
            return messages

        encoded_images = load_image(image_paths)
        messages = create_payload(prompt_context, prompt, encoded_images)
        # response = self.model.chat.completions.create(model=self.task_generation_args['engine'], messages=messages, temperature=self.task_generation_args['temperature'], presence_penalty=self.task_generation_args['presence_penalty'], frequency_penalty=self.task_generation_args['frequency_penalty'], top_p=self.task_generation_args['top_p'], stop=self.task_generation_args['stop'], max_tokens=self.task_generation_args['max_tokens'])
        response = self.model.chat.completions.create(model=self.task_generation_args['engine'], messages=messages, top_p=self.task_generation_args['top_p'], stop=self.task_generation_args['stop'],)
        response = response.choices[0].message.content
        return response

    def construct_skill_sequences(self, foundation_model_output) -> list[list[Skill]]:
        #use RoBERTa embeddings to get the most similar skills and object names in case of mismatch
        output_text = foundation_model_output.split('\n')
        skill_sequence_dictionary = {}
        curr_task = None
        #iterate through output text and parse
        for line in output_text:
            match = re.match(r"(\w+)\((.*)\)", line.strip())
            if match and curr_task is not None:
                skill_name = match.group(1)
                parameters = match.group(2).split(",")

                # construct skill objects, if skill/parameter names don't match, ground it to the closest one
                if skill_name not in [skill.name for skill in self.skill_dictionary]:
                    #get the closest similarity skill embedding
                    query_skill_embedding = self.embedding_model.encode(skill_name, convert_to_tensor=True, device=self.device)
                    cos_scores = st_utils.pytorch_cos_sim(query_skill_embedding.to(self.device), self.all_skill_embeddings.to(self.device))[0]
                    cos_scores = cos_scores.detach().cpu().numpy()
                    closest_skill_idx = np.argsort(-cos_scores)[0]
                    closest_skill_name: Skill = list(self.skill_dictionary.keys())[closest_skill_idx]
                else:
                    closest_skill_name = skill_name

                # assuming every different skills has different names
                lifted_skill: Skill = [skill for skill in self.skill_dictionary if skill.name== closest_skill_name][0]

                # TODO: implement typing, if typing doesn't match ground to closest matched object
                #       but at the same time try to make sure FM won't propose such skills, otherwise coverage will be problematic
                # get the closest similarity argument embedding
                for i, parameter in enumerate(parameters):
                    need_grounding = False
                    if parameter not in self.objects_type_dict.keys():
                        need_grounding = True
                    elif lifted_skill.types[i] not in self.objects_type_dict[parameter]["types"]:
                        need_grounding = True
                    if need_grounding:
                        query_param_embedding = self.embedding_model.encode(parameter, convert_to_tensor=True, device=self.device)
                        cos_scores = st_utils.pytorch_cos_sim(query_param_embedding.to(self.device), self.all_param_embeddings.to(self.device))[0]
                        cos_scores = cos_scores.detach().cpu().numpy()
                        idx = 0
                        closest_param_idx = np.argsort(-cos_scores)[idx]
                        closest_param = list(self.objects_type_dict.keys())[closest_param_idx]
                        while lifted_skill.types[i] not in self.objects_type_dict[closest_param]["types"] and idx < len(self.objects_type_dict) - 1:
                            idx += 1
                            closest_param_idx = np.argsort(-cos_scores)[idx]
                            closest_param = list(self.objects_type_dict.keys())[closest_param_idx]

                    else:
                        closest_param = parameter
                    parameters[i] = closest_param
                grounded_skill: Skill = lifted_skill.ground_with(parameters)
                skill_sequence_dictionary[curr_task].append(grounded_skill)
            elif len(line) > 0 and 'Skill Sequence' in line:
                skill_sequence_dictionary[line] = []
                curr_task = line

        return list(skill_sequence_dictionary.values())

    def run_skill_sequence_proposing(self, lifted_pred_list=None, skill2operator=None, tasks=None, curr_observation_path=None, init_state: PredicateState=None) -> list[Skill]:
        #Step 0: before running algorithm, update the predicate and skill dictionary available to the FM for prompting and skill generation
        if lifted_pred_list is not None:
            self.predicate_dictionary = lifted_pred_list_to_predicate_dictionary(lifted_pred_list)
        if skill2operator is not None:
            self.operator_dictionary = skill2operator
        if init_state is not None:
            self.init_state = init_state
        if curr_observation_path is not None:
            self.curr_observation_path = curr_observation_path
        if tasks is not None:
            self.attempted_skill_pair_count, self.curr_skill_count = self.get_skill_pair_matrix_from_tasks(tasks)

        #Step 1: create prompt with least explored skill pairs and object set
        prompt, prompt_context = self.create_foundation_model_prompt()
        #Step 2: run foundation model using the generated prompt
        foundation_model_output = self.run_foundation_model(prompt_context, prompt, self.curr_observation)
        #Step 3: parse and ground FM output into a task dictionary
        skill_sequences = self.construct_skill_sequences(foundation_model_output)
        for i, skill_sequence in enumerate(skill_sequences):
            print(f"Output skill sequence {i}")
            for skill in skill_sequence:
                print(skill)
            print('\n')

        #Step 4: generate scores + combine for pareto optimal way for coverage and chainability for all tasks + choose the best most pareto-optimal sequence to run
        chosen_skill_sequence = self.generate_scores_and_choose_skill_sequence(skill_sequences)

        return chosen_skill_sequence
    
if __name__ == '__main__':
    skill_sequence_proposing = SkillSequenceProposing()
    curr_observation_path = []
    chosen_skill_sequence = skill_sequence_proposing.run_skill_sequence_proposing()
    [print(p) for p in chosen_skill_sequence]
    breakpoint()