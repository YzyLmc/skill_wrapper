'demo usage of learned operators: convert multimodal input into pddl definition'
from utils import GPT4, load_from_file
from collections import defaultdict
import inspect
import random

def operators_to_pddl_domain():
    pass
def nl_to_pddl():
    pass
def img_to_pddl():
    pass

def dict_to_pddl(action_dict):
    # Extract name
    action_name = action_dict['name'][0].split('_')[0]

    # Define parameters based on detected types in preconditions/effects
    parameters = "?i - item ?l - location"

    # Initialize preconditions and effects lists
    preconditions = []
    effects = []

    # Process preconditions
    for precond, value in action_dict['precond'].items():
        pddl_precond = precond.replace('[OBJ]', '?i').replace('[LOC]', '?l').replace('[LOC_1]', '?l1').replace('[LOC_2]', '?l2')
        if value:
            preconditions.append(f"({pddl_precond.lower()})")
        else:
            preconditions.append(f"(not ({pddl_precond.lower()}))")

    # Process effects
    for effect, value in action_dict['effect'].items():
        pddl_effect = effect.replace('[OBJ]', '?i').replace('[LOC]', '?l').replace('[LOC_1]', '?l').replace('[LOC_2]', '?l')
        if value == 1:
            effects.append(f"({pddl_effect.lower()})")
        elif value == -1:
            effects.append(f"(not ({pddl_effect.lower()}))")

    # Construct the PDDL action statement
    pddl_statement = f"(:action {action_name}\n"
    pddl_statement += f"    :parameters ({parameters})\n"
    pddl_statement += "    :precondition (and " + " ".join(preconditions) + ")\n"
    pddl_statement += "    :effect (and " + " ".join(effects) + ")\n"
    pddl_statement += ")"
    
    return action_name, pddl_statement

if __name__ == "__main__":
    # Example input dictionary
    example_dict = {
        'name': ['PickUp_Bowl_DiningTable_True_2', 'PickUp_Bowl_DiningTable_True_3'],
        'effect': {
            'HasEmptyHands()': -1, 
            'IsTooHeavy([OBJ])': -1, 
            'isFixed([OBJ])': -1, 
            'isHolding([OBJ])': 1, 
            'isMovable([OBJ])': 1
        },
        'precond': {
            'AtLocation([LOC])': True, 
            'AtLocation([LOC_1])': False, 
            'AtLocation([LOC_2])': False, 
            'HasEmptyHands()': True, 
            'IsNear([OBJ])': True, 
            'IsTooHeavy([OBJ])': True, 
            'isFixed([OBJ])': True, 
            'isHolding([OBJ])': False, 
            'isMovable([OBJ])': False
        }
    }

    # Convert to PDDL
    action_name, pddl_output = dict_to_pddl(example_dict)
    print(pddl_output)
    breakpoint()