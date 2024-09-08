'refinement and proposal loop'
import argparse

from utils import GPT4
from task_proposing import TaskProposing
from symbolize import refine_pred, merge_predicates, cross_assignment
from ai2thor_task_exec import convert_task_to_code

def update_tasks(skill2tasks, task_fpath="tasks/exps"):
    '''
    Update skill2tasks after executing one task
    '''
    pass

def single_run(model, task_proposing, skill2tasks, replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary, skill2operators, curr_observation_path, max_t):

    # new_predicate_dictionary, new_skill_dictionary, new_object_list, new_replay_buffer
    chosen_task, chosen_skill_sequence = task_proposing.run_task_proposing(grounded_predicate_dictionary, grounded_skill_dictionary, None, replay_buffer, curr_observation_path)
    generated_code = convert_task_to_code(chosen_skill_sequence)
    exec(generated_code)

    # imgs will be stored at tasks/exps
    # TODO convert into data structure

    skill2triedpred = None
    for skill in skill2operators:
        skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict, skill2triedpred=skill2triedpred, max_t=max_t)


    # TODO: update prediacte dictionary, grounded_predicate_dictionary after refinement and update replay buffer

    # TODO: log the new predicate dict and skill2operators

    return skill2operators,  grounded_predicate_dictionary, replay_buffer, grounded_skill_dictionary

def main():
    # init parameters
    if args.exp == "ai2thor":
        model = GPT4(engine='gpt-4o-2024-08-06')
        # semantic of the skills
        grounded_skill_dictionary = {
            'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
            'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
            'GoTo(init, goal)': {'arguments': {'from': "the location or object for the robot to start from", 'to': "the location or object for the robot to go to"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
        }
        # env description
        grounded_predicate_dictionary = {}
        objects_in_scene = ['Book', 'Vase', 'TissueBox', 'Bowl', 'DiningTable', 'Sofa']
        env_description = 'Book, Vase, and Bowl are on the DiningTable, and Tissue is onthe sofa. Robot is at the DiningTable initially.'
        replay_buffer = {'image_before':[], 'image_after':[], 'skill':[], 'predicate_eval':[]}
        curr_observation_path = []
        # init skill to operators
        skill2operators = {}
        skill2tasks = {}
        for skill in grounded_skill_dictionary:
            skill2operators[skill] = {'precond':{}, 'eff':{}}
            skill2tasks[skill] = {}
        # init task proposing system
        task_proposing = TaskProposing(grounded_skill_dictionary = grounded_skill_dictionary, grounded_predicate_dictionary = grounded_predicate_dictionary, max_skill_count=20, skill_save_interval=2, replay_buffer = replay_buffer, objects_in_scene = objects_in_scene, env_description=env_description)
        
        
        for i in range(args.num_iter):
            pass
        
        merge_predicates
        cross_assignment

    # TODO: integrate with habitat env    
    elif args.exp == 'habitat':
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["ai2thor", "habitat"], default="ai2thor")
    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", "chatgpt-4o-latest"], default="gpt-4o-2024-08-06")
    parser.add_argument("--num_iter", type=int, help="num of iter run the full refinement and proposal loop.")
    parser.add_argument("--step_by_step", action="store_true")
    parser.add_argument("--max_retry_time", type=int, help="maximum time to use ")
    parser.add_argument("--no_log", action='store_true')
    args = parser.parse_args()

    main()