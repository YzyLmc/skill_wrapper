import os
import sys
import argparse

from data_structure import yaml
from subprocess import check_output, CalledProcessError

planner_path = 'C:/Users/david/downward/fast-downward.py'

algorithms = ['astar', 'eager', 'lazy', ]
heuristics = ['lmcut', 'ff', ]

def find_plan(
    domain_file: str,
    problem_file: str,
    algorithm: str = 'astar',
    heuristic: str = 'lmcut',
    trial: int = 0,
) -> str:

    # NOTE: define plan execution function that can be called for different parameters:

    command = [
        'python3', planner_path, domain_file, problem_file,
        '--search', f'{algorithm}({heuristic}())',
        '--plan-file', f'trial_{trial}.plan',
    ]

    planner_output = None
    try:
        planner_output = check_output(command)
    except CalledProcessError as e:
        print(f"error code: {e.returncode}\n\t-- Actual message: {e.output}")

    return planner_output


def run_trials(
    num_trials: int = 10,
):
    pass


def parse_predicate(pred: str):
    # -- change all parentheses into commas for easy parsing and remove whitespaces; then remove any empty strings:
    pred = list(filter(None, str(pred).replace('(', ',').replace(')', ',').replace(' ', '').split(',')))
    # -- extract the predicate name and all proceeding arguments :
    name, args_no_variables = pred[0], pred[1:]

    # -- we need to format predicates with question marks for variables:
    args_with_variables = []
    for arg in args_no_variables:
        # -- we will format as "?<obj_type> - <obj_type>":
        args_with_variables.append(f'?{arg} - {arg}')

    # -- return a PDDL-structured predicate:
    return f"({name} {' '.join(args_with_variables)})"


def create_domain_file(
    method: str,
    yaml_data: list,
):
    all_predicates = [parse_predicate(P) for P in yaml_data['predicates']]
    # print(all_predicates)

    all_operators = [O.pop() for _, O in yaml_data['operators'].items()]
    # print(all_operators)

    all_objects = yaml_data['objects']['objects']

    object_types = set()
    for obj_name in all_objects:
        for obj_type in all_objects[obj_name]['types']:
            object_types.add(f'{obj_type} - object')

        for obj_type in all_objects[obj_name]['types']:
            object_types.add(f'{obj_name} - {obj_type}')

    # print(object_types)

    with open(f'{method}_domain.pddl', 'w') as nf:
        prototype_content = None

        # -- read all content from the prototype file:
        with open('domain_prototype.pddl', 'r') as df:
            prototype_content = df.read()

        # -- find and replace placeholders in the prototype file:
        new_content = prototype_content.replace('<actions>', "\n\n".join(all_operators))
        new_content = new_content.replace('<types>', "\n\t\t".join(list(object_types)))
        new_content = new_content.replace('<predicates>', "\n\t\t".join(all_predicates))

        # -- write content to new PDDL file:
        nf.write(new_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_operators",
        type=str,
        default=None,
        help="This specifies the path to the YAML file containing all operators.",
    )

    parser.add_argument(
        "--yaml_predicates",
        type=str,
        default=None,
        help="This specifies the path to the YAML file containing all predicates.",
    )

    parser.add_argument(
        "--yaml_objects",
        type=str,
        default=None,
        help="This specifies the path to the YAML file containing all objects.",
    )

    args = parser.parse_args()

    data_predicates, data_operators, data_objects = [], [], []

    if args.yaml_operators:
        with open(args.yaml_operators, "r") as f:
            data_operators = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('-- Missing YAML operators file!')
        sys.exit()

    if args.yaml_predicates:
        with open(args.yaml_predicates, "r") as f:
            data_predicates = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('-- Missing YAML predicates file!')
        sys.exit()

    if args.yaml_objects:
        with open(args.yaml_objects, "r") as f:
            data_objects = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('-- Missing YAML objects file!')
        sys.exit()

    if data_predicates and data_operators and data_objects:
        create_domain_file(
            method='skillwrapper',
            yaml_data={
                'predicates': data_predicates,
                'operators': data_operators,
                'objects': data_objects,
            }
        )

