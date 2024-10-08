
from openai import OpenAI
import numpy as np
import copy
import re
from sentence_transformers import SentenceTransformer, util as st_utils
import torch

import pdb

'''
TODO: Use ChatGPT for working example - check that skills are diverse + concered about diverse skill set
TODO: want to see that foundation model can generate plans that produce different types of failure cases (predicate failure) without presenting the predicates to the foundation model
- Diverse failure cases

'''

class TaskProposing():


    def __init__(self, grounded_skill_dictionary, grounded_predicate_dictionary, max_skill_count, skill_save_interval, replay_buffer, objects_in_scene, env_description):

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
        3) Pareto-optimality measure: combining coverage, chainability and sufficience in a way that ensures 


        DONE:
        5) Hyperparameters for LLM
        1) Coverage ==> what type of entropy? do we take final entropy or difference in entropy for information gain? (maybe value function type: where repeatedly taking samples of same pair leads to stagnation)
        1.5) Choice of skill pairs for "least explored" in task proposing prompt
        7) Counts of all pairs of grounded skills

        '''

        #predicate dictionary: {predicate: definition/description}
        self.predicate_dictionary = grounded_predicate_dictionary

        #skill dictionary: {skill name: {arguments: {argument: description}, preconditions: [predicate name], effects_positive: [predicate name], effects_negative: [predicate name]}}
        self.skill_dictionary = grounded_skill_dictionary

        #replay buffer: {image before, image after, skill, predicate eval}
        self.replay_buffer = replay_buffer #TODO: coordinate with Ziyi

        #global object set for the scene
        self.objects_in_scene = objects_in_scene

        #spatial relationship of objects
        self.env_description = env_description
       
        #global frequency count for all pairs of skills 
        self.skill_to_index = {x: i for i,x in enumerate(grounded_skill_dictionary.keys())}
        self.attempted_skill_pair_count = np.zeros((len(grounded_skill_dictionary.keys()), len(grounded_skill_dictionary.keys())))



        self.curr_shannon_entropy = 0.0

        #LLM hyperparameters: GPT4O
        self.task_generation_args = {
            'temperature': 0.6,
            'presence_penalty': 0.3,
            'frequency_penalty': 0.4,
            'top_p': 1.0,
            'max_tokens':550,
            'engine': 'gpt-4o',
            'stop': ''
        }


        #GPT4O LLM model to query for new proposed tasks
        self.model = OpenAI(api_key='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')

        #embedding model for grounding LLM output to groundable/executable skills and objects
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('mps') # for my m1 macbook
        self.embedding_model = SentenceTransformer('stsb-roberta-large').to(self.device)

        self.all_skill_embeddings = self.embedding_model.encode(list(self.skill_dictionary.keys()), batch_size=32, convert_to_tensor=True, device=self.device)
        self.all_arg_embeddings = self.embedding_model.encode(self.objects_in_scene, batch_size=32, convert_to_tensor=True, device=self.device)



        #other metrics to track: number of skills executed and logging frequency for predicates at certain skill intervals
        self.curr_skill_count = 0
        self.save_skill_freq = skill_save_interval
        self.max_skill_count = max_skill_count

        #parameters for kernel density estimation
        self.h = 1

        #scaling parameters for pareto-optimal task selection
        self.k = 10 #set period after how many skill executions to switch mode
        #all alphas are in the range [1,3]
        # self.chainability_alpha = lambda x : np.cos((np.pi/self.k) * x) + 2
        self.chainability_alpha = lambda x: 1
        self.sufficience_alpha = lambda x: np.sin((np.pi/self.k) * x - (np.pi/2)) + 2
        self.entropy_gain_alpha = lambda x: np.cos( ( np.pi / self.k) * x) + 2

    def create_foundation_model_prompt(self):

        skill_prompts = []

        for skill in self.skill_dictionary:

            args = self.skill_dictionary[skill]['arguments']
            preconds = self.skill_dictionary[skill]['preconditions']
            effects_pos = [x+'=True' for x in self.skill_dictionary[skill]['effects_positive']]
            effects_neg = [x+'=False' for x in self.skill_dictionary[skill]['effects_negative']]

            skill_prompt = skill + '\n' + '\n'.join([a + ': ' + args[a] for a in args.keys()]) + '\npreconditions: [' + ', '.join(preconds) + ']\neffects: {' + ', '.join(effects_pos) + ', '.join(effects_neg) + '}'
            skill_prompts.append(skill_prompt)
        
        # pdb.set_trace()
        least_explored_skills = self.get_least_explored_skills()

        prompt_context = "A robot with a single gripper is attempting to learn the preconditions and effects for a finite set of skills by performing tasks in an environment" 
        
        prompt = "Propose a set of tasks for a robot to execute along with a sequence of skills to achieve these tasks. The robot is attempting to learn the preconditions and effects for a finite set of skills. The robot can navigate the environment freely but only has one gripper. The robot has access to the following skills with their associated arguments, precondition estimate and effect estimate:\n\n{}\n\nThe list of objects the robot has previously encountered in the environment are: {}\n{}\n\nThe pairs of consecutive skills (skill1, skill2) that have been least explored are: [{}].\n\n You should keep in mind the type of arguments that each skill can take. Using the list of objects and the skill preconditions / effects learned, generate 5 tasks and their sequence of skills such that: (1) the tasks purposefully violate skill preconditions often (2) the ordering of skills in each task is unique (3) at least 1 unexplored skill pair is used in each task (4) all tasks have at least 8 skills in sequence.\n\nOutput only the task name and the sequence of skills to execute. Output 1 skill every new line, following the format below:\nTask 1: Pick up the apple:\nwalk_to(CounterTop)\npick_up(Apple)\n\nTask 1:".format('\n\n'.join(skill_prompts), self.objects_in_scene, self.env_description, ','.join(least_explored_skills))
    
        return prompt, prompt_context
        

    '''
    AUXILLIARY: functions to update the predicate dictionary and skill_dictionary after refinement
    '''
    def update_predicate_dictionary(self, new_predicate_dictionary):
        if new_predicate_dictionary is not None:
            self.predicate_dictionary = new_predicate_dictionary
    
    def update_skill_dictionary(self, new_skill_dictionary):
        if new_skill_dictionary is not None:
            self.skill_dictionary = new_skill_dictionary

    def update_obj_set(self, new_object_set):
        if new_object_set is not None:
            self.objects_in_scene = new_object_set

    def add_obj_to_set(self, new_obj):
        if new_obj is not None:
            self.objects_in_scene.append(new_obj)
    
    def update_replay_buffer(self, new_replay_buffer):
        if new_replay_buffer is not None:
            self.replay_buffer = new_replay_buffer
    def update_skill_pair_count(self, new_skill_pair_count):
        self.attempted_skill_pair_count = new_skill_pair_count
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
            skill_sequence = [re.sub(r'_\d+', '', action) for action in skill_sequence]
            executable_sequence = task_dictionary[task]['executable_sequence']
            # executable_sequence = [re.sub(r'_\d+', '', action) for action in executable_sequence]
            breakpoint()
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
    def generate_incremental_predicate_for_task(self, skill_sequence, lifted_skill_sequence, initial_observation_path):

        '''
        Sufficiency: relies on abstract state changes to be abstract with respect to the object
        Chainabilty: relies on the non-abstract state changes to be abstract

        Can abstract and non-abstract state changes occur simultaneously?
            How is predicate failure handled?
        '''

        
        #maintain dicitonary of updated predicates so far: {predicate: known assignemnt from precondition/effects of skills that are executed}
        abstract_current_predicates = {}

        #skill dictionary: {skill name: {arguments: {argument: description}, preconditions: [predicate name], effects_positive: [predicate name], effects_negative: [predicate name]}}
        for p in self.predicate_dictionary:
            p = p.split('(')[0] + '()'
            abstract_current_predicates[p] = 1

        current_predicates = {}

        #list of predicates for each step
        predicate_sequence = []
        max_executable = 0
        executable_sequence = []

        abstract_executable = True
        executable = True

        #TODO: maybe instead setup initial set of calls to VLM to estimate all the predicates first -- using inital_observation
       
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
                      

            abstract_preconditions = self.skill_dictionary[lifted_skill]['preconditions']
            abstract_effects_pos  = self.skill_dictionary[lifted_skill]['effects_positive']
            abstract_effects_neg = self.skill_dictionary[lifted_skill]['effects_negative']

            
            #turn predicates true assuming skill can be executed
            for pre in abstract_preconditions:
                
                pre = pre.split('(')[0] + '()'

                if pre in abstract_current_predicates and abstract_current_predicates[pre] == 0:
                    abstract_executable = False
                    break
                
                abstract_current_predicates[pre] = 1
            
            predicate_sequence.append(np.array(list(abstract_current_predicates.values())))
            

            if not abstract_executable:
                abstract_executable = True
            else:
                
                #turn predicates positive or negative based on effects

                for eff in abstract_effects_neg:

                    eff = eff.split('(')[0] + '()'
                    abstract_current_predicates[eff] = 0
                
                for eff in abstract_effects_pos:

                    eff = eff.split('(')[0] + '()'

                    abstract_current_predicates[eff] = 1

            predicate_sequence.append(np.array(list(abstract_current_predicates.values())))
            
            #-----------------------------------------------------------#

            for pre in abstract_preconditions:

                match = re.match(r"(\w+)\((.*)\)", pre)
                pred_name = match.group(1)
                pred_args = '('+pre.split('(')[1]


                for a in abstract_to_grounded_args.keys():
                    pred_args = pred_args.replace(a, abstract_to_grounded_args[a])
                # if '(x)' in pre:
                #     pre = pre.replace('x', arguments[0])
                
                # elif '(r)' in pre:
                #     pre = pre.replace('(r)', f'({arguments[0]})')
                pre = pred_name + pred_args

                if (pre in current_predicates and current_predicates[pre] == 0):
                    executable = False
                    break

               
                current_predicates[pre] = 1
            
            
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
                   

                    # if '(x)' in eff:
                    #     eff = eff.replace('x', arguments[0])
                    # elif '(r)' in eff:
                    #     eff = eff.replace('(r)', f'({arguments[0]})')
                    eff = pred_name + pred_args
                    

                    

                    current_predicates[eff] = 0
                
                for eff in abstract_effects_pos:

                    match = re.match(r"(\w+)\((.*)\)", eff)
                    pred_name = match.group(1)
                    #pred_args = match.group(2).split(",")
                    pred_args = '('+eff.split('(')[1]


                    for a in abstract_to_grounded_args.keys():
                        pred_args = pred_args.replace(a, abstract_to_grounded_args[a])
                    

                    # if'(x)' in eff:
                    #     eff = eff.replace('x', arguments[0])
                    # elif '(r)' in eff:
                    #     eff = eff.replace('(r)', f'({arguments[1]})')
                    eff = pred_name + pred_args

                    
                    current_predicates[eff] = 1
            
            
            
            predicate_sequence.append(np.array(list(abstract_current_predicates.values())))
            

        # pdb.set_trace()
        return predicate_sequence, max_executable, executable_sequence

    def compute_task_chainability(self, executable_sequence, max_executable):

        return abs(float(max_executable / (len(executable_sequence) if len(executable_sequence) > 0 else 1)) - 0.5)

    def compute_task_sufficience_probability(self, predicate_sequence, max_executable):
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
            return 0.1
            buffer_predicates = np.array(self.replay_buffer['predicate_eval'])
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


    def compute_chainability_and_sufficience(self, task_dictionary, initial_observation_path):
        
        #track to select task with maximum chainability and sufficience
        task_chainabilities = []
        task_sufficience_logprobs = []

        #compute chainability and sufficience score for each task
        for task in task_dictionary.keys():

            skill_sequence = task_dictionary[task]['grounded']
            lifted_skill_sequence = task_dictionary[task]['lifted']

            predicate_sequence, max_executable, executable_sequence = self.generate_incremental_predicate_for_task(skill_sequence, lifted_skill_sequence, initial_observation_path)

            #update the task dictionary with maximum number of steps that can be executed
            task_dictionary[task]['max_executable'] = max_executable
            task_dictionary[task]['executable_sequence'] = executable_sequence


            #compute and aggregate the chainability and sufficience prob per task
            chainability = self.compute_task_chainability(executable_sequence, max_executable)
            sufficience_logprob = self.compute_task_sufficience_probability(predicate_sequence, max_executable)


            task_chainabilities.append(chainability)
            task_sufficience_logprobs.append(sufficience_logprob)
        
        
        return np.array(task_chainabilities), np.array(task_sufficience_logprobs)


    '''
    OVERALL SCORING: Function to run general scoring at the task level, combining coverage, sufficience and chainability
    '''
    def generate_scores_and_choose_task(self, task_dictionary, curr_observation_path):
        
        # pdb.set_trace()
        #run the 3 scoring functions
        task_chainabilities, task_sufficience_logprobs = self.compute_chainability_and_sufficience(task_dictionary, curr_observation_path)
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
                elif (eg_new <= eg2) and (suf_new >= suf2) and (chain_new >= chain2):
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
        return list(task_dictionary.keys())[max_score_idx], list(task_dictionary.values())[max_score_idx]['grounded']




    '''
    FOUNDATION MODEL: Functions to run LLM (GPT4-O) as well as generate dynamic prompting structure using least explored tasks
    '''
    def run_foundation_model(self, context_prompt, prompt, image_paths):

        #NOTE: output the task list that is grounded and decomposed into a dictionary of {task: [[list of skills with arguments], max executable steps]}
        def load_image(image_paths):

            encoded_images = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_image)
            return encoded_images
        
        def create_payload(context_prompt: str, prompt: str, encoded_images):

            messages = [
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": [] }
            ]

            for encoded_image in encoded_images:
                messages[1]["content"].append(
                {'type':'image_url', 'image_url':{'url': f"data:image/png;base64,{encoded_image}"}}
                )
            
            messages[1]["content"].append({'type': 'text', 'text': prompt})

            
            return messages

        encoded_images = load_image(image_paths)

        messages = create_payload(context_prompt,prompt, encoded_images)

        
        response = self.model.chat.completions.create(model=self.task_generation_args['engine'], messages=messages, temperature=self.task_generation_args['temperature'], presence_penalty=self.task_generation_args['presence_penalty'], frequency_penalty=self.task_generation_args['frequency_penalty'], top_p=self.task_generation_args['top_p'], stop=self.task_generation_args['stop'], max_tokens=self.task_generation_args['max_tokens'])

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
                cos_scores = st_utils.pytorch_cos_sim(query_skill_embedding.to(self.device), self.all_skill_embeddings.to(self.device))[0]
                cos_scores = cos_scores.detach().cpu().numpy()
                closest_skill_idx = np.argsort(-cos_scores)[0]
                closest_grounded_skill = list(self.skill_dictionary.keys())[closest_skill_idx].split('(')[0]
                closest_grounded_skill_abstract = list(self.skill_dictionary.keys())[closest_skill_idx]

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

            elif len(line) > 0 and 'Task' in line:
                task_dictionary[line] = {'grounded':[], 'lifted':[]}
                curr_task = line
        
        # pdb.set_trace()
        return task_dictionary
                


           



    def run_task_proposing(self, new_predicate_dictionary, new_skill_dictionary, new_object_list, new_replay_buffer, curr_observation_path):

        # if self.curr_skill_count > self.max_skill_count:
        #     print('SKILL EXECUTION HAS REACHED LIMIT: curr_skill_count > max_skill_count')
        #     return None, None

        #Step 0: before running algorithm, update the predicate and skill dictionary available to the FM for prompting and skill generation
        self.update_predicate_dictionary(new_predicate_dictionary)
        self.update_skill_dictionary(new_skill_dictionary)
        self.update_obj_set(new_object_list)

        #Step 1: create prompt with least explored skill pairs and object set
        prompt, prompt_context = self.create_foundation_model_prompt()
        breakpoint()
        #Step 2: run foundation model using the generated prompt
        foundation_model_output = self.run_foundation_model(prompt, prompt_context, curr_observation_path)


        #Step 3: parse and ground FM output into a task dictionary
        task_dictionary = self.construct_task_dictionary(foundation_model_output)

        #Step 4: generate scores + combine for pareto optimal way for coverage, chainability and sufficience for all tasks + choose the best most pareto-optimal task to run
        chosen_task, chosen_skill_sequence = self.generate_scores_and_choose_task(task_dictionary, curr_observation_path)

       


        return chosen_task, chosen_skill_sequence




if __name__ == '__main__':
    # pdb.set_trace()

    # grounded_skill_dictionary = {
    #     'pick_up(x)':{'arguments': {'x': "the object to be picked up"}, 'preconditions': ['is_gripper_empty()','is_nearby(x)'], 'effects_positive':['is_holding(x)'], 'effects_negative': ['is_gripper_empty()']},
    #     'put_down(x,r)': {'arguments': {'x': "the object to be dropped", 'r': "the receptacle onto which object 'x' is dropped"}, 'preconditions': ['is_nearby(r)', 'is_holding(x)'], 'effects_positive':['is_gripper_empty()'], 'effects_negative': ['is_holding(x)']},
    #     'walk_to(x)': {'arguments': {'x': "the location for the robot to walk to"}, 'preconditions': [], 'effects_positive':['is_nearby(x)'], 'effects_negative':[]}
    # }

    grounded_skill_dictionary = {
        'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
        'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
        'GoTo(init, goal)': {'arguments': {'init': "the location or object for the robot to start from", 'goal': "the location or object for the robot to go to"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
    }

    grounded_skill_dictionary = {
        'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
        'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
        'GoTo(init, goal)': {'arguments': {'init': "the location or object for the robot to start from", 'goal': "the location or object for the robot to go to"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
    }

    # grounded_predicate_dictionary = {}
    grounded_predicate_dictionary = {'is_at_location([OBJ], [LOC])': 'The object `obj` is currently located at the location `loc`.',
                                     'is_holding([OBJ])': 'The robot is currently holding the object `obj` before attempting to drop it at the location `loc`.',
                                     'is_at([OBJ], [LOC])': 'The object `obj` is located at the location `loc` after the execution of the skill.'}
    # grounded_predicate_dictionary = {
    #     'is_gripper_empty()': "the robot's single gripper is empty with no objects held",
    #     'is_nearby(x)': "the robot can interact with object or receptacle 'x' using only it's single gripper without needing to move the body closer to 'x'",
    #     'is_holding(x)': "the robot is holding object 'x' specifically in it's gripper"
    # }


    # new_skill_pair_count = np.zeros((3,3))
    # new_skill_pair_count[0,1] += 1
    # new_skill_pair_count[1,2] += 1
    # new_skill_pair_count[2,0]+= 1
    # new_skill_pair_count[0,2] += 1
    # new_skill_pair_count[2,1] += 1

    # objects_in_scene = ['Apple', 'Bread', 'Fridge', 'Egg', 'Cabinet1', 'Cabinet2', 'Cabinet3', 'CounterTop', 'Newspaper', 'PaperTowerRoll', 'Toaster', 'Faucet', 'LightSwitch', 'Mug', 'Kettle', 'Statue', 'Bowl', 'Bin', 'Lettuce', 'Tomato', 'Potato', 'Microwave']

    objects_in_scene = ['Book', 'Vase', 'TissueBox', 'Bowl', 'DiningTable', 'Sofa']
    env_description = 'Book, Vase, and Bowl are on the DiningTable, and RemoteConrtol is on the sofa. Robot is at the DiningTable initially.'

    # replay_buffer = {'image_before':[], 'image_after':[], 'skill':['pick_up(Apple)','put_down(Apple,CounterTop)','walk_to(Fridge)','pick_up(Potato)','walk_to(Toaster)','put_down(Potato,Toaster)'], 'predicate_eval':[[0,1,0],[0,0,0],[1,0,0],[1,1,0],[0,0,0],[0,0,1]]}
    # replay_buffer = {'image_before':[], 'image_after':[], 'skill':['GoTo(Sofa, Book)','PickUp(Book,DiningTable)','GoTo(Book, Sofa)','DropAt(Book, Sofa)'], 'predicate_eval':[[], [], [], [],[],[]]}
    # replay_buffer = {'image_before':[], 'image_after':[], 'skill':[], 'predicate_eval':[[], [], [], [],[],[]]}
    replay_buffer = {'image_before':[], 'image_after':[], 'skill':['GoTo(Sofa,Sofa)', 'PickUp(TissueBox,Sofa)', 'GoTo(Sofa,DiningTable)', 'DropAt(TissueBox,DiningTable)', 'PickUp(Book,DiningTable)', 'DropAt(Book,DiningTable)', 'PickUp(TissueBox,DiningTable)', 'DropAt(TissueBox,Sofa)'], 'predicate_eval':[[1,0,1],[],[],[], []]}
    curr_observation_path = []


    task_proposing = TaskProposing(grounded_skill_dictionary = grounded_skill_dictionary, grounded_predicate_dictionary = grounded_predicate_dictionary, max_skill_count=20, skill_save_interval=2, replay_buffer = replay_buffer, objects_in_scene = objects_in_scene, env_description=env_description)
    # task_proposing.update_skill_pair_count(new_skill_pair_count)

    chosen_task, chosen_skill_sequence = task_proposing.run_task_proposing(grounded_predicate_dictionary, grounded_skill_dictionary, None, replay_buffer, curr_observation_path)
    print(chosen_task, chosen_skill_sequence)
    pdb.set_trace()