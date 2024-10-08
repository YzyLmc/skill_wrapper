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
def extract_parameters(action_dict):
    # Identify parameters from the dictionary
    parameters = set()
    for condition_dict in [action_dict['precond'], action_dict['effect']]:
        for condition in condition_dict:
            if '[OBJ]' in condition:
                parameters.add('?o - object')
            if '[LOC]' in condition:
                parameters.add('?l - location')
            if '[LOC_1]' in condition:
                parameters.add('?l1 - location')
            if '[LOC_2]' in condition:
                parameters.add('?l2 - location')
    return ' '.join(parameters)

def dict_to_pddl(action_dict, action_counter):
    # Extract base action name
    base_action_name = action_dict['name'][0].split('_')[0]

    # Update the counter for this action type
    action_counter[base_action_name] += 1

    # Formulate the indexed action name
    indexed_action_name = f"{base_action_name}_{action_counter[base_action_name]}"

    # Extract parameters dynamically
    parameters = extract_parameters(action_dict)

    # Initialize preconditions and effects lists
    preconditions = []
    effects = []

    # Process preconditions
    for precond, value in action_dict['precond'].items():
        if 'GoTo' in base_action_name:
            if not '[OBJ]' in precond and not '[LOC]' in precond:
                pddl_precond = precond.replace('[OBJ]', '?o').replace('[LOC]', '?l').replace('[LOC_1]', '?l1').replace('[LOC_2]', '?l2').replace('(', ' ').replace(')', '').replace(',', ' ')
                if value:
                    preconditions.append(f"({pddl_precond.lower()})")
                else:
                    preconditions.append(f"(not ({pddl_precond.lower()}))")
        else:
            if not '[LOC_1]' in precond and not '[LOC_2]' in precond:
                pddl_precond = precond.replace('[OBJ]', '?o').replace('[LOC]', '?l').replace('[LOC_1]', '?l1').replace('[LOC_2]', '?l2').replace('(', ' ').replace(')', '').replace(',', ' ')
                if value:
                    preconditions.append(f"({pddl_precond.lower()})")
                else:
                    preconditions.append(f"(not ({pddl_precond.lower()}))")

    # Process effects
    for effect, value in action_dict['effect'].items():
        pddl_effect = effect.replace('[OBJ]', '?o').replace('[LOC]', '?l').replace('[LOC_1]', '?l1').replace('[LOC_2]', '?l2').replace('(', ' ').replace(')', '').replace(',', ' ')
        if value == 1:
            effects.append(f"({pddl_effect.lower()})")
        elif value == -1:
            effects.append(f"(not ({pddl_effect.lower()}))")

    # Construct the PDDL action statement
    pddl_statement = f"(:action {indexed_action_name}\n"
    pddl_statement += f"    :parameters ({parameters})\n"
    pddl_statement += "    :precondition (and " + " ".join(preconditions) + ")\n"
    pddl_statement += "    :effect (and " + " ".join(effects) + ")\n"
    pddl_statement += ")"
    
    return action_counter, pddl_statement

def generate_pddl_domain(domain_name, action_dicts):
    # Define domain header
    domain_header = f"(define (domain {domain_name})\n"
    domain_header += "    (:requirements :strips :typing)\n"
    
    # Extract object types from action dictionaries
    types_set = extract_types_from_dicts(action_dicts)
    types_section = "    (:types " + " ".join(types_set) + ")\n"
    
    # Collect all predicates from the action dictionaries
    predicates_set = extract_predicates_from_dicts(action_dicts)
    
    # Create the predicates section
    predicates_section = "    (:predicates\n"
    for predicate in sorted(predicates_set):
        predicates_section += f"        {predicate}\n"
    predicates_section += "    )\n"

    action_counter = defaultdict(int)
    actions = []
    # Convert action dictionaries to PDDL strings
    for i, skill in action_dicts.items():
        action_counter, operator= dict_to_pddl(skill, action_counter)
        actions.append(operator)
    breakpoint()
    # Combine all parts to form the complete PDDL domain definition
    pddl_domain = domain_header + types_section + predicates_section
    
    # Add all actions to the domain
    for action in actions:
        pddl_domain += action + "\n\n"
    
    # Close the domain definition
    pddl_domain += ")"
    
    return pddl_domain

def extract_predicates_from_dicts(action_dicts):
    predicates = set()
    for i, action_dict in action_dicts.items():
        for condition_dict in [action_dict['precond'], action_dict['effect']]:
            for predicate, _ in condition_dict.items():
                # Extract base predicate name and add parameters based on placeholders
                base_predicate = predicate.split('(')[0].lower()
                if '[OBJ]' in predicate and '[LOC]'in predicate:
                    predicates.add(f"({base_predicate} ?o - object ?l - location)")
                elif '[OBJ]' in predicate:
                    predicates.add(f"({base_predicate} ?o - object)")
                elif '[LOC]' in predicate or '[LOC_1]' in predicate or '[LOC_2]' in predicate:
                    predicates.add(f"({base_predicate} ?l - location)")
                else:
                    predicates.add(f"({base_predicate})")
    return predicates

def extract_types_from_dicts(action_dicts):
    types_set = set()
    for i, action_dict in action_dicts.items():
        # Check the placeholders in the preconditions and effects to determine types
        for condition_dict in [action_dict['precond'], action_dict['effect']]:
            for predicate in condition_dict.keys():
                if '[OBJ]' in predicate:
                    types_set.add("object")
                if '[LOC]' in predicate or '[LOC_1]' in predicate or '[LOC_2]' in predicate:
                    types_set.add("location")
    return types_set


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