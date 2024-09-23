'refinement and proposal loop'
'remember to change the for loop in update_tasks() if you have changed the input skill'
import argparse
import os
from copy import deepcopy

from utils import GPT4, load_from_file, save_to_file
from task_proposing import TaskProposing
from symbolize import refine_pred, merge_predicates, cross_assignment

# bunch of conversion functions to seam refinement and task proposal
def skill2operators2grounded_skill_dict(skill2operators, grounded_skill_dictionary):
    for skill in skill2operators:
        skill_prefix = skill.split('(')[0]
        for s in grounded_skill_dictionary:
            if skill_prefix in s:
                grounded_skill_dictionary[s]['preconditions'] = [precond.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])') for precond in skill2operators[skill]['precond'].keys()]
                grounded_skill_dictionary[s]['effects_positive'] = [eff.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])') for eff, value in skill2operators[skill]['eff'].items() if value == 1]
                grounded_skill_dictionary[s]['effects_negative'] = [eff.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])') for eff, value in skill2operators[skill]['eff'].items() if value == -1]
    return grounded_skill_dictionary

def pred_dict2grounded_predicates_dict(pred_dict):
    grounded_predicates_dictionary = {}
    for pred in pred_dict:
        origin_pred = deepcopy(pred)
        while "[OBJ]" in pred or "[LOC]" in pred or "[LOC_1]" in pred or "[LOC_2]" in pred:
            pred = pred .replace("[OBJ]", "obj")
            pred  = pred .replace("[LOC]", "loc")
            pred  = pred .replace("[LOC_1]", "init")
            pred  = pred .replace("[LOC_2]", "goal")
        grounded_predicates_dictionary[pred] = pred_dict[origin_pred]['semantic']
    return grounded_predicates_dictionary

def update_tasks(skill_list, task_fpath="tasks/exps"):
    '''
    Update skill2tasks after executing one task
    '''
    skill2tasks_new = {}
    for skill in ['DropAt([OBJ], [LOC])', 'GoTo([LOC_1], [LOC_2])', 'PickUp([OBJ], [LOC])']: # I don't know why I have to do it in this way
        skill2tasks_new[skill] = {}
        skill_prefix = skill.split('(')[0]
        dir = f"{task_fpath}/{skill_prefix}"
        if not os.path.exists(dir):
            os.makedirs(dir)
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for f in files:
            f_name = f.split('.')[0]
            args = f_name.split('_')
            task_id = "_".join(args[1:])
            if  not f"Before_{task_id}.jpg" in files or not f"After_{task_id}.jpg" in files:
                print('filename unmatch. fix it')
                breakpoint()
            skill2tasks_new[skill][task_id] = {
                's0':[f"{task_fpath}/{skill_prefix}/Before_{task_id}.jpg"],
                's1':[f"{task_fpath}/{skill_prefix}/After_{task_id}.jpg"],
                'success': True if args[-2] == 'True' else False
            }
            assert f"Before_{task_id}.jpg" in files
            assert f"After_{task_id}.jpg" in files
            if "DropAt" in skill or "PickUp" in skill:
                skill2tasks_new[skill][task_id]['obj'] = args[2]
                skill2tasks_new[skill][task_id]['loc'] = args[3]
                skill2tasks_new[skill][task_id]['loc_1'] = ''
                skill2tasks_new[skill][task_id]['loc_2'] = ''
            elif "GoTo" in skill:
                skill2tasks_new[skill][task_id]['loc_1'] = args[2]
                skill2tasks_new[skill][task_id]['loc_2'] = args[3]
                skill2tasks_new[skill][task_id]['obj'] = ''
                skill2tasks_new[skill][task_id]['loc'] = ''
    return skill2tasks_new

def single_run(model, simulator_env, task_proposing, pred_dict, skill2operators, skill2tasks, replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary, curr_observation_path, args, log_data=None):
    '''
    run one iteration of refinement and proposal
    pred_dict, skill2operators and skill2tasks are from refinement. 
    replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary are from task proposal.
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''
    # new_predicate_dictionary, new_skill_dictionary, new_object_list, new_replay_buffer
    chosen_skill_sequence = None
    t = 0
    task_success = False
    while t < 10 and not task_success:
        chosen_task, chosen_skill_sequence = task_proposing.run_task_proposing(grounded_predicate_dictionary, grounded_skill_dictionary, None, replay_buffer, curr_observation_path)
        t += 1
        print('Task:', chosen_skill_sequence)
        try:
            task_success = True
        except:
            pass
    
    simulator_env.execute_skill_sequence(chosen_skill_sequence)

    # breakpoint()
    replay_buffer['skill'] = chosen_task
    if args.step_by_step:
            print('Task done. You should check the images labels')
            breakpoint()

    # imgs will be stored at tasks/exps
    # skill2tasks = update_tasks(list(skill2tasks.keys()))

    for skill in skill2operators:
        skill2triedpred = {} # reset tried_predicate buffer after each skill
        skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict, skill2triedpred=skill2triedpred, max_t=args.max_retry_time)

    # we need to save skill2tasks, skill2operators, and pred_dict at each step (task)
    # logfile:: {num: {'skill2tasks':dict, 'skill2operators':dict, 'pred_dict':dict} }
    if not log_data:
        log_data = {0:{'model': args.model, 'env': args.env}}
    time_step = max([int(key) for key in list(log_data.keys())]) + 1
    log_data[time_step] = {'skill2tasks':skill2tasks, 'skill2operators':skill2operators, 'pred_dict':pred_dict, 'grounded_skill_dictionary': skill2operators2grounded_skill_dict(skill2operators, grounded_skill_dictionary), 'generated_task': chosen_skill_sequence}

    grounded_skill_dictionary = skill2operators2grounded_skill_dict(skill2operators, grounded_skill_dictionary)
    grounded_predicate_dictionary = pred_dict2grounded_predicates_dict(pred_dict)

    return log_data, skill2operators, skill2tasks, pred_dict, grounded_predicate_dictionary, replay_buffer, grounded_skill_dictionary


def main():
    # init parameters
    if args.env == "habitat":
        model = GPT4(engine=args.model)
        if args.continue_learning:
            assert args.load_fpath
            # load from log file
            log_data = load_from_file(args.load_fpath)
            last_run_num = max([key for key in log_data.keys() if key.isdigit()])
            skill2tasks, skill2operators, pred_dict, grounded_skill_dictionary = log_data[last_run_num]["skill2tasks"], log_data[last_run_num]["skill2operators"], log_data[last_run_num]["pred_dict"], log_data[last_run_num]["grounded_skill_dictionary"]

        else:   
            # start from scratch
            # semantic of the skills
            grounded_skill_dictionary = {
                'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
                'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
                'GoTo(init, goal)': {'arguments': {'init': "the location or object for the robot to start from", 'goal': "the location or object for the robot to reach"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
            }
            
            # init skill to operators
            skill2operators = {}
            skill2tasks = {}
            pred_dict = {}
            for skill in grounded_skill_dictionary:
                skill = skill.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])')
                skill2operators[skill] = {'precond':{}, 'eff':{}}
                skill2tasks[skill] = {}

            skill2tasks = update_tasks(skill2tasks)
            log_data = None
        # breakpoint()

        # env description
        grounded_predicate_dictionary = {}
        curr_observation_path = []
        # init task proposing system
        replay_buffer = {'image_before':[], 'image_after':[], 'skill':[], 'predicate_eval':[]}
        objects_in_scene = ['BeanCan', 'WoodenCube', 'MeatCan', 'RedBowl','Couch', 'Bed', 'Cabinet', 'Nightstand', 'WorkDesk', 'Sink', 'Bedroom', 'Painting', 'LivingRoom', 'Television', 'DiningTable', 'TowelRack', 'Fridge']
        env_description = 'BeanCan is on the Couch, WoodenCube is on the Bed, MeatCan is on WorkDesk and RedBowl is on the Cabinet. Robot is at the Bedroom initially.'
        task_proposing = TaskProposing(grounded_skill_dictionary = grounded_skill_dictionary, grounded_predicate_dictionary = grounded_predicate_dictionary, max_skill_count=8*args.num_iter, skill_save_interval=2, replay_buffer = replay_buffer, objects_in_scene = objects_in_scene, env_description=env_description)
        
        #create the habitat environment
        habitat_env = HabitatSimulator()

        counter = 1
        if not args.no_log:
            while True:
                log_save_path = f"{args.save_dir}/{args.env}_{args.num_iter}_log_{counter}{'continue' if args.continue_learning else ''}.json"
                if not os.path.exists(log_save_path):
                    break
                counter += 1
        if args.continue_learning:
            start_num = max([int(key) for key in log_data.keys() if key.isdigit()])
        else:
            start_num = 0
        for i in range(start_num, args.num_iter):
            log_data, skill2operators, skill2tasks, pred_dict, grounded_predicate_dictionary, replay_buffer, grounded_skill_dictionary = single_run(model, habitat_env, task_proposing, pred_dict, skill2operators, skill2tasks, replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary, curr_observation_path, args, log_data=log_data)
            if pred_dict:
                try:
                     if not len(set([len(pred_dict[pred]['task']) for pred in pred_dict])) == 1: # all predicates should be evaluated on all tasks
                        breakpoint()
                except:
                    breakpoint()
            print(f"iteration #{i+1} is done")
            print(f"current operators:{skill2operators}")
            save_to_file(log_data, log_save_path)
            print(f"result has been saved to {log_save_path}")

            if args.step_by_step:
                print('about to cross assign and merge')
                breakpoint()

            # try
            try:
                merged_skill2operators, equal_preds = merge_predicates(model, skill2operators, pred_dict)
                assigned_skill2operators = cross_assignment(merged_skill2operators, skill2tasks, pred_dict, equal_preds=equal_preds)

                log_data[i+1]['assigned_skill2operators'] = assigned_skill2operators
                log_data[i+1]['merged_skill2operators'] = merged_skill2operators
                save_to_file(log_data, log_save_path)
                print('Final operators this round:\n', merged_skill2operators)
                print(f"result has been saved to {log_save_path}")
            except:
                print('merge failed. Will continue next iteration. merge can be done later manually')
                pass
            if args.step_by_step:
                breakpoint()

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["ai2thor", "habitat"], default="ai2thor")
    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", "chatgpt-4o-latest"], default="gpt-4o-2024-08-06")
    parser.add_argument("--num_iter", type=int, default=5, help="num of iter run the full refinement and proposal loop.")
    parser.add_argument("--step_by_step", action="store_true")
    parser.add_argument("--max_retry_time", type=int, default=5, help="maximum time to generate predicate to distinguish two states.")
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--continue_learning", action='store_true')
    parser.add_argument("--load_fpath", type=str, help="provide the log file to restore from a previous checkpoint. must specify if continue learning is true")
    parser.add_argument("--save_dir", type=str, default='tasks/log', help="directory to save log files")
    args = parser.parse_args()

    main()