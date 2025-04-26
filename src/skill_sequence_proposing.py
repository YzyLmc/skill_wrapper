
from openai import OpenAI
import numpy as np
import os
import copy
import re
from sentence_transformers import SentenceTransformer, util as st_utils
import torch
import base64

from RCR_bridge import LiftedPDDLAction
from utils import load_from_file

'''
TODO: Use ChatGPT for working example - check that skills are diverse + concered about diverse skill set
'''

# data structure conversion function
def to_replay_buffer(tasks, grounded_predicate_truth_value_log):
    """
    Args:
        tasks :: dict(task_name: (step: dict("skill": grounded_skill : Skill, 'image':img_path : str, 'success': Bool)))
        grounded_predicate_truth_value_log :: dict :: {task_name:{step:PredicateState}}
    Returns:
        replay_buffer :: dict(image_before: list[str], image_after: list[str], skill: [str], predicate_eval: list[list[bool]]])
    """
    replay_buffer = {
        "image_before":[],
        "image_after": [],
        "skill": [],
        "predicate_eval": []
    }
    assert sorted(list(tasks.keys())) == sorted(list(tasks.keys(grounded_predicate_truth_value_log))), \
            "Predicate turth values of all tasks have to be evalauted before proposing the next skill sequence."
    for task_name, task_meta in tasks.items():
        for step, state in task_meta.items():
            if not step == 0: # skip the first step where no skill is executed
                replay_buffer["image_before"].append(last_state["image"])
                replay_buffer["image_after"].append(state["image"])
                replay_buffer["skill"].append(str(state["skill"]))
                grounded_pred_list = grounded_predicate_truth_value_log[task_name][step].get_pred_list()
                # NOTE: these predicate are grounded, while operators will be lifted
                replay_buffer['predicate_eval'].append([grounded_predicate_truth_value_log[task_name][step].get_pred_value(grounded_pred) for grounded_pred in grounded_pred_list])
            last_state = state

    return replay_buffer

# TODO: see if RCR data structure works, if not do conversion
def skill2operator_to_operator_dictionary(skill2operator) -> dict[str, tuple[LiftedPDDLAction, dict]]:
    """
    Args:
        skill2operator :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str})]}
    Returns:
        operator_dictionary :: {operator_name: (LiftedPDDLAction, pid2type: dict)}
    """
    operator_dictionary = {}
    for lifted_skill, operator_metas in skill2operator.items():
        for i, operator_meta in enumerate(operator_metas): # inner tuple :: tuple[LifetdPDDLAction, dict[int, str]]
            operator_name = f"{str(lifted_skill)}_{i}"
            operator_dictionary[operator_name]= operator_meta

    return operator_dictionary
    

def lifted_pred_list_to_predicate_dictionary(lifted_pred_list):
    """
    Args:
        lifted_pred_list :: list[Predicate]
    Returns:
        preddicate_dictionary :: dict[str, str]
    """
    return {str(pred): pred.semantic for pred in lifted_pred_list}

class SkillSequenceProposing():
    def __init__(self, lifted_pred_list={}, tasks={}, grounded_predicate_truth_value_log={}, skill2operator={}, prompt_fpath="prompts/skill_sequence_proposal.yaml", env_config_fpath="task_config/dorfl.yaml"):

        #coverage of tasks: entropy measure
        #chainability of tasks: building partial state + approximations of other predicates
        #sufficience of tasks: building partial state and count of overlaps from dataset

        '''
        Things we need + access to
        1) Replay buffer of previous observations and skills we have run
        2) Predicates we have built so far with their definitions (predicate dictionary)
        3) Current visual observation to classify list of objects we see in the scene
        4) Set of all objects in the current scene observed
        
        6) Prompting structure and context for LLM
       
        8) System for grounding the output skills by the LLM

        Algorithms we need to solidify
        2) Bayesian approach to estimating aspects of partial state we do not know ==> based on statistical evidence of replay buffer
        3) Pareto-optimality measure: combining coverage, chainability in a way that ensures 


        DONE:
        5) Hyperparameters for LLM
        1) Coverage ==> what type of entropy? do we take final entropy or difference in entropy for information gain? (maybe value function type: where repeatedly taking samples of same pair leads to stagnation)
        1.5) Choice of skill pairs for "least explored" in task proposing prompt
        7) Counts of all pairs of grounded skills

        '''
        self.env_config = load_from_file(env_config_fpath)

        #predicate dictionary: {predicate: definition/description}
        self.predicate_dictionary = lifted_pred_list_to_predicate_dictionary(lifted_pred_list)

        #operator dictionary: {operator name: {arguments: {argument: description}, preconditions: [predicate name], effects_positive: [predicate name], effects_negative: [predicate name]}}
        # TODO: this is not operator dictionary, it's a skill dictionary
        # self.operator_dictionary = {
        #         str(lifted_skill): {'arguments': {ptype: sem for ptype, sem in lifted_skill.semantics.items()}, 'preconditions': {}, 'effects_positive':[], 'effects_negative': []} \
        #             for lifted_skill in self.env_config['skills'].values()
        #     }
        
        self.operator = skill2operator_to_operator_dictionary(skill2operator) if skill2operator else skill2operator

        #skill dictionary: {skill name: {arguments: {argument: description}}}
        self.skill_dictionary = {str(lifted_skill): {'arguments': {ptype: sem for ptype, sem in lifted_skill.semantics.items()}} for lifted_skill in self.env_config['skills'].values()}
        self.operator_to_skill = {k: re.sub(r'_\d+', '', k) for (k,v) in self.operator_dictionary.items()}

        #replay buffer: {image before, image after, skill, predicate eval}
        if not (tasks and grounded_predicate_truth_value_log):
            self.replay_buffer = {"image_before":[], "image_after":[], "skill":[], "predicate_eval":[]}
        else:
            assert (tasks and grounded_predicate_truth_value_log), "You need to provide both to compute the replay buffer"
            self.replay_buffer = to_replay_buffer(tasks, grounded_predicate_truth_value_log)

        #global object set for the scene
        self.objects_in_scene = list(self.env_config['objects'].keys())
        self.objects_in_scene_with_types = [f"{obj}: {str(types)}" for obj, types in self.env_config['objects'].items()]

        #spatial relationship of objects
        self.env_description = self.env_config['Env_description']

        #initial visual observation of the environment
        self.initial_observation = self.env_config['Initial_observation']['img_fpath']
        self.curr_observation = self.initial_observation
       
        #global frequency count for all pairs of skills 
        self.skill_to_index = {x: i for i,x in enumerate(self.skill_dictionary.keys())}
        self.attempted_skill_pair_count = np.zeros((len(self.skill_dictionary.keys()), len(self.skill_dictionary.keys())))

        self.prompt_dict = load_from_file(prompt_fpath)
        self.curr_shannon_entropy = 0.0

        #LLM hyperparameters: GPT4O
        self.task_generation_args = {
            'temperature': 0.6,
            'presence_penalty': 0.3,
            'frequency_penalty': 0.35,
            'top_p': 1.0,
            # 'max_tokens':550,
            'engine': 'o1',
            'stop': ''
        }


        #GPT4O LLM model to query for new proposed tasks
        self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        #embedding model for grounding LLM output to groundable/executable skills and objects
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('mps') # for my m1 macbook: mps
        self.embedding_model = SentenceTransformer('stsb-roberta-large').to(self.device)

        self.all_operator_embeddings = self.embedding_model.encode(list(self.operator_dictionary.keys()), batch_size=32, convert_to_tensor=True, device=self.device)
        self.all_arg_embeddings = self.embedding_model.encode(self.objects_in_scene, batch_size=32, convert_to_tensor=True, device=self.device)

        #other metrics to track: number of skills executed and logging frequency for predicates at certain skill intervals
        self.curr_skill_count = 0

        #parameters for kernel density estimation
        self.h = 1

        #scaling parameters for pareto-optimal task selection
        self.k = 10 #set period after how many skill executions to switch mode
        #all alphas are in the range [1,3]
        self.chainability_alpha = lambda x: 1
        self.sufficience_alpha = lambda x: np.sin((np.pi/self.k) * x - (np.pi/2)) + 2
        self.entropy_gain_alpha = lambda x: np.cos( ( np.pi / self.k) * x) + 2

    def parse_skill_sequence(self, proposed_skill_sequence: str):
        #Function parses the skill sequence proposed by the LLM, removing the skills with 'False' label and returning only skill names
        # Split by lines
        lines = proposed_skill_sequence.strip().split('\n')

        # Filter and collect only skills labeled "True"
        skills = []
        for line in lines:
            if " - True:" in line:
                skill = line.split(" - True:")[0]
                skills.append(skill)

        # Print each skill on a new line
        output = "\n".join(skills)
        return output

    def create_foundation_model_prompt(self):

        skill_prompts = []
        for skill in self.operator_dictionary:
            args = self.operator_dictionary[skill]['arguments']
            skill_prompt = skill + '\n' + '\n'.join([a + ': ' + args[a] for a in args.keys()])
            skill_prompts.append(skill_prompt)
        
        least_explored_skills = self.get_least_explored_skills()
        prompt_context = self.prompt_dict["prompt_context"]
        prompt = self.prompt_dict["prompt"]
        prompt = prompt.replace("[SKILL_PROMPT]", "\n\n".join(skill_prompts))\
                        .replace("[OBJECT_IN_SCENE]", str("\n".join(self.objects_in_scene_with_types)))\
                        .replace("[ENV_DESCRIPTION]", self.env_description)\
                        .replace("[LEAST_EXPLORED_SKILLS]", ','.join(least_explored_skills))
        
        return prompt, prompt_context
        
    '''
    AUXILLIARY: functions to update the predicate dictionary and skill_dictionary after refinement
    '''
    def update_obj_set(self, new_object_set):
        if new_object_set is not None:
            self.objects_in_scene = new_object_set

    def update_curr_obs(self, new_obs_path):
        if new_obs_path is not None:
            self.curr_observation = new_obs_path
            
    '''
   COVERAGE:  Functions for entropy computation + Functions to determine least explored tasks
    '''
    def compute_entropy_for_task(self, skill_sequence, executable_sequence):

        new_skill_pair_count = copy.deepcopy(self.attempted_skill_pair_count)

        p1 = 0; p2 = min(1, len(skill_sequence))

        while p2 < len(skill_sequence):

            if executable_sequence[p2] is True and executable_sequence[p1] is True:
                skill1_idx = self.skill_to_index[skill_sequence[p1]]
                skill2_idx = self.skill_to_index[skill_sequence[p2]]

                new_skill_pair_count[skill1_idx, skill2_idx] += 1

                p1 = p2 
            elif executable_sequence[p1] is False:
                p1 = p2
                
            p2 += 1

        normalized_skill_pair_prob =  new_skill_pair_count / np.sum(new_skill_pair_count) if np.sum(new_skill_pair_count) > 0 else new_skill_pair_count
        log_skill_pair_prob = np.where(normalized_skill_pair_prob > 0.0 , np.log(normalized_skill_pair_prob), 0.0)

        new_shannon_entropy = np.sum(-1 * normalized_skill_pair_prob * log_skill_pair_prob)
        return new_shannon_entropy, new_skill_pair_count
    
    def compute_shannon_entropy(self, task_dictionary):

        normalized_skill_pair_prob =  self.attempted_skill_pair_count / np.sum(self.attempted_skill_pair_count) if np.sum(self.attempted_skill_pair_count) > 0 else  self.attempted_skill_pair_count 
        
        log_skill_pair_prob = np.where(normalized_skill_pair_prob > 0 , np.log(normalized_skill_pair_prob), 0)

        curr_shannon_entropy = np.sum(-1 * normalized_skill_pair_prob * log_skill_pair_prob)
        

        task_entropy_gains = []
        task_skill_counts = []

        #measure entropy gain for each task
        for task in task_dictionary.keys():

            skill_sequence = task_dictionary[task]['lifted']
            skill_sequence = [self.operator_to_skill[action] for action in skill_sequence]
            executable_sequence = task_dictionary[task]['executable_sequence']
            # executable_sequence = [re.sub(r'_\d+', '', action) for action in executable_sequence]
            # pdb.set_trace()
            entropy, counts = self.compute_entropy_for_task(skill_sequence, executable_sequence)

            task_entropy_gains.append(entropy - curr_shannon_entropy) #entropy gain is maximum of difference
            task_skill_counts.append(counts)
        
        return np.array(task_entropy_gains), task_skill_counts

    def get_least_explored_skills(self, k=5):

        flattened_count = self.attempted_skill_pair_count.flatten()

        # Step 2: Get the indices that would sort the array
        sorted_indices = np.argsort(flattened_count)

        # Step 3: Get the indices of the top 'k' smallest elements
        min_k_indices = sorted_indices[:k]

        # Step 4: Convert the flattened indices back to 2D indices
        min_k_2d_indices = np.unravel_index(min_k_indices, self.attempted_skill_pair_count.shape)

        least_explored_pairs = []

        for (idx1, idx2) in zip(min_k_2d_indices[0], min_k_2d_indices[1]):

            skill1 = list(self.skill_to_index.keys())[idx1]
            skill2 = list(self.skill_to_index.keys())[idx2]

            least_explored_pairs.append('( ' + skill1 + ', ' + skill2 +' )')

        return least_explored_pairs


    '''
    CHAINABILITY & SUFFICIENCE: Functions to approximate the predicate-level information
    '''
    def generate_incremental_predicate_for_task(self, skill_sequence, lifted_skill_sequence):

        '''
        Sufficiency: relies on abstract state changes to be abstract with respect to the object
        Chainabilty: relies on the non-abstract state changes to be abstract

        Can abstract and non-abstract state changes occur simultaneously?
            How is predicate failure handled?
        '''

        #maintain dicitonary of updated predicates so far: {predicate: known assignemnt from precondition/effects of skills that are executed}
        abstract_current_predicates = {}

        #skill dictionary: {skill name: {arguments: {argument: description}, preconditions: [predicate name], effects_positive: [predicate name], effects_negative: [predicate name]}}
        #TODO: maybe instead setup initial set of calls to VLM to estimate all the predicates first -- using inital_observation
        #TODO: @Ziyi: maybe you could replace naively adding '1' to abstract_current_predicates[p] with VLM_Eval(p, initial_observation_path) instead to make the algorithm more appropriate
        initial_observation = self.initial_observation # TODO
        for p in self.predicate_dictionary:
            abstract_current_predicates[p] = 1

        current_predicates = {}

        #list of predicates for each step
        predicate_sequence = []
        max_executable = 0
        executable_sequence = []

        abstract_executable = True
        executable = True
        # TODO: change the logic of choosing operators, always check if 
        #iterate through the skill sequence and add/change predicate labels until something is not executable
        for i, skill in enumerate(skill_sequence):

            #generate abstract skill shell
            lifted_skill = lifted_skill_sequence[i]

            #extract arguments for skill in case of chainability
            match = re.match(r"(\w+)\((.*)\)", skill)
            arguments = match.group(2).split(",")

            match = re.match(r"(\w+)\((.*)\)", lifted_skill)
            abstract_arguments = match.group(2).split(",")

            #find correlating arguments with abstract/lifted arguments
            abstract_to_grounded_args = {k.strip():v.strip() for (k,v) in zip(abstract_arguments, arguments)}
                      

            abstract_preconditions = self.operator_dictionary[lifted_skill]['preconditions']
            abstract_effects_pos  = self.operator_dictionary[lifted_skill]['effects_positive']
            abstract_effects_neg = self.operator_dictionary[lifted_skill]['effects_negative']

            
            #turn predicates true assuming skill can be executed
            for pre, value in abstract_preconditions.items():
                
                #pre = pre.split('(')[0] + '()'

                if pre in abstract_current_predicates and abstract_current_predicates[pre] != int(value):
                    abstract_executable = False
                    break

                
                abstract_current_predicates[pre] = int(value)
            
            predicate_sequence.append(np.array(list(abstract_current_predicates.values())))
            

            if not abstract_executable:
                abstract_executable = True
            else:
                
                #turn predicates positive or negative based on effects

                for eff in abstract_effects_neg:
                    abstract_current_predicates[eff] = 0
                
                for eff in abstract_effects_pos:
                    abstract_current_predicates[eff] = 1

            predicate_sequence.append(np.array(list(abstract_current_predicates.values())))
            
            for pre, value in abstract_preconditions.items():

                match = re.match(r"(\w+)\((.*)\)", pre)
                pred_name = match.group(1)
                pred_args = '('+pre.split('(')[1]


                for a in abstract_to_grounded_args.keys():
                    pred_args = pred_args.replace(a, abstract_to_grounded_args[a])

                pre = pred_name + pred_args

                if (pre in current_predicates and current_predicates[pre] != int(value)):
                    executable = False
                    break
              
                current_predicates[pre] = int(value)
                 
            executable_sequence.append(executable)
            if not executable:
                executable = True
            else:
                max_executable += 1

                #turn predicates positive or negative based on effects
                for eff in abstract_effects_neg:

                    match = re.match(r"(\w+)\((.*)\)", eff)
                    pred_name = match.group(1)
                    #pred_args = match.group(2).split(",")
                    pred_args = '('+eff.split('(')[1]

                    for a in abstract_to_grounded_args.keys():
                        pred_args = pred_args.replace(a, abstract_to_grounded_args[a])
                   
                    eff = pred_name + pred_args
                    current_predicates[eff] = 0
                
                for eff in abstract_effects_pos:

                    match = re.match(r"(\w+)\((.*)\)", eff)
                    pred_name = match.group(1)
                    #pred_args = match.group(2).split(",")
                    pred_args = '('+eff.split('(')[1]

                    for a in abstract_to_grounded_args.keys():
                        pred_args = pred_args.replace(a, abstract_to_grounded_args[a])
                
                    eff = pred_name + pred_args
                    current_predicates[eff] = 1

            predicate_sequence.append(np.array(list(abstract_current_predicates.values())))
            
        return predicate_sequence, max_executable, executable_sequence

    def compute_task_chainability(self, executable_sequence, max_executable):

        return abs(float(max_executable / (len(executable_sequence) if len(executable_sequence) > 0 else 1)) - 0.5)

    def compute_task_sufficience_probability(self, predicate_sequence):
        '''
        Option 1: count of number that is the same / total 
        Option 2: use KDE to estimate the probability that current state will match [CHOSEN FOR NOW]

        Why use KDE?
            Do not know the underlying distribution of predicate assignments or symbolic states in the environment
            Symbolic state should be fairly low dimensional (hopefully) as predicates are built in
            Dataset should not be too big (hopefully??)
            Can do density estimation of particular s_hat within s_buffer space
        '''

        #total negative log probability of P(s_buffer = s_hat) for  all s_hat in the predicate sequence
        total_logprob = 0


        def compute_density_estimation(pred, h):
            
            
            buffer_predicates = np.array(self.replay_buffer['predicate_eval']).astype(int)
            hamming_distance = np.sum((buffer_predicates - pred)!=0, axis=1 if len(pred.shape)>1 else 0)
            hamming_distance = hamming_distance.reshape(-1,1)

            kde = np.exp(-1*hamming_distance/h)
            kde = np.sum(kde) / len(kde)

            return kde
            
        #estimate P(s_buffer = s_hat) using symbolic state space as "feature" representation
        for pred in predicate_sequence:
            kernel_density = compute_density_estimation(pred,self.h)
            total_logprob += -1*np.log(kernel_density)

        return total_logprob
    
    def compute_chainability(self, skill_sequence_dictionary):

        # 
        task_chainabilities = []

        for skill_sequence in skill_sequence_dictionary:
            task_chainability = self.chainability_per_task(skill_sequence)
            task_chainabilities.append(task_chainability)

        return np.array(task_chainabilities)
        
    def compute_chainability_and_sufficience(self, task_dictionary):
        
        #track to select task with maximum chainability and sufficience
        task_chainabilities = []
        task_sufficience_logprobs = []

        #compute chainability and sufficience score for each task
        for task in task_dictionary.keys():

            skill_sequence = task_dictionary[task]['grounded']
            lifted_skill_sequence = task_dictionary[task]['lifted']

            predicate_sequence, max_executable, executable_sequence = self.generate_incremental_predicate_for_task(skill_sequence, lifted_skill_sequence)

            #update the task dictionary with maximum number of steps that can be executed
            task_dictionary[task]['max_executable'] = max_executable
            task_dictionary[task]['executable_sequence'] = executable_sequence

            #compute and aggregate the chainability and sufficience prob per task
            chainability = self.compute_task_chainability(executable_sequence, max_executable)
            sufficience_logprob = self.compute_task_sufficience_probability(predicate_sequence)

            task_chainabilities.append(chainability)
            task_sufficience_logprobs.append(sufficience_logprob)
        
        return np.array(task_chainabilities), np.array(task_sufficience_logprobs)

    '''
    OVERALL SCORING: Function to run general scoring at the task level, combining coverage and chainability
    '''
    def generate_scores_and_choose_task(self, task_dictionary):
        #run the 3 scoring functions
        task_chainabilities, task_sufficience_logprobs = self.compute_chainability_and_sufficience(task_dictionary)
        task_entropy_gains, task_skill_counts = self.compute_shannon_entropy(task_dictionary)

        #collect the maximum and minimum for each metric
        max_entropy_gain = max(task_entropy_gains); min_entropy_gain = min(task_entropy_gains)
        max_chainability = max(task_chainabilities); min_chainability = min(task_chainabilities)
        max_sufficiency = max(task_sufficience_logprobs); min_sufficiency = min(task_sufficience_logprobs)

        #combine the outputted lists for task chainability, sufficience and entropy gain to ensure pareto optimality
        pareto_front_set = {0: (task_entropy_gains[0], task_sufficience_logprobs[0], task_chainabilities[0])}

        curr_idx = 1

        for (eg_new, suf_new, chain_new) in zip(task_entropy_gains[1:], task_sufficience_logprobs[1:], task_chainabilities[1:]):

            domination = False

            for k in list(pareto_front_set.keys()):

                (eg2, suf2, chain2) = pareto_front_set[k]
                
                #if the new score combo dominates something from the pareto-front set, remove the original example from pareto-front set 
                if (eg_new >= eg2) and (suf_new <= suf2) and (chain_new <= chain2) and (eg_new > eg2 or suf_new < suf2 or chain_new < chain2):

                    domination = True
                    del pareto_front_set[k]
                    pareto_front_set[curr_idx] = (eg_new, suf_new, chain_new)
                
                #if pareto set already dominates new example then skip
                elif (eg2 >= eg_new) and (suf2 <= suf_new) and (chain2 <= chain_new) and (eg2 > eg_new or suf2 < suf_new or chain2 < chain_new):
                    domination = True
                    continue 
                
            
            #if some variables dominated then new example is part of pareto set 
            if not domination:
                pareto_front_set[curr_idx] = (eg_new, suf_new, chain_new)


            curr_idx += 1
        

        #define the weightages for each metric based on the current number of tasks and choose something from pareto-front that maximizes the desired combination
        chain_alpha = self.chainability_alpha(self.curr_skill_count)
        suff_alpha = self.sufficience_alpha(self.curr_skill_count)
        entr_alpha = self.entropy_gain_alpha(self.curr_skill_count)

        max_score = float('-inf'); max_score_idx = None

        for (k, v) in pareto_front_set.items():

            if v is None:
                continue
            
            (entr, suff, chain) = v

            entropy_score = (entr - min_entropy_gain)/(max_entropy_gain - min_entropy_gain) if (max_entropy_gain - min_entropy_gain) > 0 else 0
            # sufficience_score = (suff - min_sufficiency) / (max_sufficiency - min_sufficiency) if (max_sufficiency - min_sufficiency) > 0 else 0
            sufficience_score = (max_sufficiency - suff) / (max_sufficiency - min_sufficiency) if (max_sufficiency - min_sufficiency) > 0 else 0
            chainability_score = (max_chainability - chain) / (max_chainability - min_chainability)  if (max_chainability - min_chainability) > 0 else 0

            combined_score = entr_alpha * (entropy_score) + suff_alpha * (sufficience_score) + chain_alpha * (chainability_score)

            if combined_score > max_score:
                max_score = combined_score
                max_score_idx = k

        #find the task with the maximum combined score + set the current skill count to that task's skill count + set the current task execution matrix counts
        self.curr_skill_count += len(task_dictionary[list(task_dictionary.keys())[max_score_idx]]['grounded'])
        self.attempted_skill_pair_count = task_skill_counts[max_score_idx]
        # pdb.set_trace()
        return list(task_dictionary.values())[max_score_idx]['grounded']

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

    def construct_task_dictionary(self, foundation_model_output):

        #use RoBERTa embeddings to get the most similar skills and object names in case of mismatch
        output_text = foundation_model_output.split('\n')

        task_dictionary = {}

        curr_task = None

        #iterate through output text and parse
        for line in output_text:

            match = re.match(r"(\w+)\((.*)\)", line.strip())

            if match and curr_task is not None:
                skill_name = match.group(1)  # The function name
                arguments = match.group(2).split(",")  # The arguments, split by commas


                #get the closest similarity skill embedding
                query_skill_embedding = self.embedding_model.encode(skill_name, convert_to_tensor=True, device=self.device)
                cos_scores = st_utils.pytorch_cos_sim(query_skill_embedding.to(self.device), self.all_operator_embeddings.to(self.device))[0]
                cos_scores = cos_scores.detach().cpu().numpy()
                closest_operator_idx = np.argsort(-cos_scores)[0]
                closest_grounded_skill = list(self.operator_dictionary.keys())[closest_operator_idx].split('(')[0]
                closest_grounded_skill_abstract = list(self.operator_dictionary.keys())[closest_operator_idx]

                max_args = len(re.match(r"(\w+)\((.*)\)", closest_grounded_skill_abstract).group(2).split(","))

                #get the closest similarity argument embedding
                for i, arg in enumerate(arguments):
                    query_arg_embedding = self.embedding_model.encode(arg, convert_to_tensor=True, device=self.device)
                    cos_scores = st_utils.pytorch_cos_sim(query_arg_embedding.to(self.device), self.all_arg_embeddings.to(self.device))[0]
                    cos_scores = cos_scores.detach().cpu().numpy()
                    closest_grounded_arg = np.argsort(-cos_scores)[0]
                    closest_grounded_arg = self.objects_in_scene[closest_grounded_arg]

                    arguments[i] = closest_grounded_arg
                
                task_dictionary[curr_task]['lifted'].append(closest_grounded_skill_abstract)
                task_dictionary[curr_task]['grounded'].append(closest_grounded_skill + '(' + ','.join(arguments[:max_args])+')')

            elif len(line) > 0 and 'Skill Sequence' in line:
                task_dictionary[line] = {'grounded':[], 'lifted':[]}
                curr_task = line
        
        return task_dictionary

    def run_skill_sequence_proposing(self, tasks=None, lifted_pred_list=None, grounded_predicate_truth_value_log=None, skill2operator=None, new_object_list=None, curr_observation_path=None):

        #Step 0: before running algorithm, update the predicate and skill dictionary available to the FM for prompting and skill generation
        if tasks is not None:
            assert grounded_predicate_truth_value_log is not None, "predicate turth values must be provided for updating replay buffer"
            self.replay_buffer = to_replay_buffer(tasks, grounded_predicate_truth_value_log)
        if lifted_pred_list is not None:
            self.predicate_dictionary = lifted_pred_list_to_predicate_dictionary(lifted_pred_list)
        if skill2operator is not None:
            # TODO: also update skill dictionary?
            self.operator_dictionary = skill2operator_to_operator_dictionary(skill2operator)
        self.update_obj_set(new_object_list)
        self.update_curr_obs(curr_observation_path)

        #Step 1: create prompt with least explored skill pairs and object set
        prompt, prompt_context = self.create_foundation_model_prompt()
        
        #Step 2: run foundation model using the generated prompt
        foundation_model_output = self.run_foundation_model(prompt_context, prompt, self.curr_observation)

        #Step 3: parse and ground FM output into a task dictionary
        task_dictionary = self.construct_task_dictionary(foundation_model_output)
 
        #Step 4: generate scores + combine for pareto optimal way for coverage, chainability and sufficience for all tasks + choose the best most pareto-optimal sequence to run
        chosen_skill_sequence = self.generate_scores_and_choose_task(task_dictionary)

        return chosen_skill_sequence
    
if __name__ == '__main__':

    # grounded_skill_dictionary = {'PickUp_1(obj, loc)': {'preconditions': {'IsAt(loc)':False}, 'effects_positive': [], 'effects_negative': [], 'arguments': {'obj': 'the object to be picked up', 'loc': 'the receptacle that the object is picked up from'}}, 
    # 'PickUp_2(obj, loc)': {'preconditions': {}, 'effects_positive': ['IsAt(loc)'], 'effects_negative': [], 'arguments': {'obj': 'the object to be picked up', 'loc': 'the receptacle that the object is picked up from'}}, 
    # 'PickUp_3(obj, loc)': {'preconditions': {'HasEmptyGripper()':True, 'IsAt(loc)':True}, 'effects_positive': [], 'effects_negative': ['HasEmptyGripper()'], 'arguments': {'obj': 'the object to be picked up', 'loc': 'the receptacle that the object is picked up from'}}, 
    # 'DropAt_1(obj, loc)': {'preconditions': {'IsAt(loc)':True}, 'effects_positive': ['HasEmptyGripper()'], 'effects_negative': [], 'arguments': {'obj': 'the object to be dropped', 'loc': 'the receptacle onto which object is dropped'}}, 
    # 'GoTo_1(init, goal)': {'preconditions': {'IsAt(init)':True}, 'effects_positive': ['HasEmptyGripper()', 'IsAt(goal)'], 'effects_negative': ['IsAt(init)'], 'arguments': {'init': 'the location for the robot to start from', 'goal': 'the location for the robot to go to'}}, 
    # 'GoTo_2(init, goal)': {'preconditions': {'IsAt(init)':True, 'IsAt(goal)':False}, 'effects_positive': ['IsAt(goal)'], 'effects_negative': ['IsAt(init)'], 'arguments': {'init': 'the location for the robot to start from', 'goal': 'the location for the robot to go to'}}}

    # grounded_predicate_dictionary = {'HasEmptyGripper()': "The robot's gripper is empty and ready to grasp an object.", 'IsAt(loc)': 'The robot is at the location `loc`.', 'IsAt(init)': 'The robot is at the location `init`.', 'IsAt(goal)': 'The robot is at the location `goal`.'}

    replay_buffer = {
        'image_before': ['tasks/exps/GoTo/Before_GoTo_Sofa_CoffeeTable_True_1.jpg', 'tasks/exps/PickUp/Before_PickUp_Vase_CoffeeTable_True_1.jpg', 'tasks/exps/DropAt/Before_DropAt_Vase_Sofa_False_1.jpg', 'tasks/exps/GoTo/Before_GoTo_CoffeeTable_DiningTable_True_1.jpg', 'tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_False_1.jpg', 'tasks/exps/GoTo/Before_GoTo_DiningTable_CoffeeTable_True_1.jpg', 'tasks/exps/DropAt/Before_DropAt_Bowl_CoffeeTable_False_1.jpg', 'tasks/exps/PickUp/Before_PickUp_Vase_Sofa_False_1.jpg', 'tasks/exps/DropAt/Before_DropAt_Vase_CoffeeTable_True_1.jpg'],
        'image_after': ['tasks/exps/GoTo/After_GoTo_Sofa_CoffeeTable_True_1.jpg', 'tasks/exps/PickUp/After_PickUp_Vase_CoffeeTable_True_1.jpg', 'tasks/exps/DropAt/After_DropAt_Vase_Sofa_False_1.jpg', 'tasks/exps/GoTo/After_GoTo_CoffeeTable_DiningTable_True_1.jpg', 'tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_False_1.jpg', 'tasks/exps/GoTo/After_GoTo_DiningTable_CoffeeTable_True_1.jpg', 'tasks/exps/DropAt/After_DropAt_Bowl_CoffeeTable_False_1.jpg', 'tasks/exps/PickUp/After_PickUp_Vase_Sofa_False_1.jpg', 'tasks/exps/DropAt/After_DropAt_Vase_CoffeeTable_True_1.jpg'],
        'skill': ['GoTo(Sofa,CoffeeTable)', 'PickUp(Vase,CoffeeTable)', 'DropAt(Vase,Sofa)', 'GoTo(CoffeeTable,DiningTable)', 'PickUp(Bowl,DiningTable)', 'GoTo(DiningTable,CoffeeTable)', 'DropAt(Bowl,CoffeeTable)', 'PickUp(Vase,Sofa)', 'DropAt(Vase,CoffeeTable)'], \
        'predicate_eval': [[True, True, False, False, True], [True, False, False, True, True], [True, False, True, True, False], [False, False, False, False, True], [False, False, False, False, True], [False, False, False, False, False], [False, False, False, False, True], [True, False, False, True, False], [True, True, True, True, True],[False, False, False, False, False]], 
    }
    skill_sequence_proposing = SkillSequenceProposing()
    curr_observation_path = []
    chosen_skill_sequence = skill_sequence_proposing.run_skill_sequence_proposing()
    print(chosen_skill_sequence)
    breakpoint()