'refinement and proposal loop'
import argparse

from task_proposing import TaskProposing
from symbolize import refine_pred, merge_predicates, cross_assignment
from ai2thor_task_exec import convert_task_to_code

def single_run():
    task_proposing = TaskProposing(grounded_skill_dictionary = grounded_skill_dictionary, grounded_predicate_dictionary = grounded_predicate_dictionary, max_skill_count=20, skill_save_interval=2, replay_buffer = replay_buffer, objects_in_scene = objects_in_scene, env_description=env_description)

    chosen_task, chosen_skill_sequence = task_proposing.run_task_proposing(None, None, None, None, curr_observation_path)
    generated_code = convert_task_to_code(chosen_skill_sequence)
    exec(generated_code)
    # imgs will be stored at tasks/exps
    # TODO convert into data structure

    refine_pred(model, skill, skill2operators, skill2tasks, pred_dict, skill2triedpred=None, max_t=3)
    pass

def main():
    # init parameters
    if args.exp == "ai2thor":
        
        grounded_skill_dictionary = {
            'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
            'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
            'GoTo(init, goal)': {'arguments': {'init': "the location or object for the robot to start from", 'goal': "the location or object for the robot to go to"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
        }
        grounded_predicate_dictionary = {}
        objects_in_scene = ['Book', 'Vase', 'RemoteControl', 'Bowl', 'DiningTable', 'Sofa']
        env_description = 'Book, Vase, and Bowl are on the DiningTable, and RemoteConrtol is onthe sofa. Robot is at the DiningTable initially.'
        replay_buffer = {'image_before':[], 'image_after':[], 'skill':[], 'predicate_eval':[]}
        curr_observation_path = []
        
    elif args.exp == 'habitat':
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["ai2thor", "habitat"])
    args = parser.parse_args()

    main()