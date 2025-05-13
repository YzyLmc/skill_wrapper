import os
import sys
from random import randint, choice
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
        # '--plan-file', f'"trial_{trial}.plan"',
    ]

    plan = []

    try:
        _ = check_output(command)
    except CalledProcessError as e:
        print(f"error code: {e.returncode}\n\t-- Actual message: {str(e.output)}")
    else:
        with open('sas_plan', 'r') as f:
            for _line in f.readlines():
                if ';' not in _line: plan.append(_line.strip())

    return plan


def run_trials(
    domain_fpath: str,
    num_trials: int = 10,
):

    count = 0
    for T in range(num_trials):
        problem_fpath = create_problem_file(
            robot=args.robot,
            trial=T,
        )

        solution = find_plan(
            problem_file=problem_fpath,
            domain_file=domain_fpath,
        )

        count += int(len(solution) > 0)

        if solution:
            print("plan has been found!")
            for x in range(len(solution)):
                print(f"{x+1} : {solution[x]}")



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
) -> str:
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

    domain_fpath = os.path.join(os.getcwd(), f'{method}_domain.pddl')

    with open(domain_fpath, 'w') as nf:
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

    return domain_fpath


def create_problem_file(
    robot: str = "dorfl",
    trial: int = 0,
) -> str:

    if robot == "dorfl":
        state = [
            f"(is_graspable j {choice(['left_gripper', 'right_gripper'])})",
            f"(is_graspable k {choice(['left_gripper', 'right_gripper'])})",
            ("(hand_empty left_gripper)"if bool(randint(0, 1)) else f"(is_holding left_gripper {choice(['k', 'j'])})") ,
            ("(hand_empty right_gripper)"if bool(randint(0, 1)) else f"(is_holding right_gripper {choice(['k', 'j'])})") ,
            ("(contains j pb)" if bool(randint(0, 1)) else ""),
            ("(is_opened j)" if bool(randint(0, 1)) else ""),
            "(on_location b t)",
            "(on_location k t)",
            "(on_location j t)",
        ]

    state = list(filter(None, state))

    problem_fpath = f"{robot}_problem_trial-{trial}.pddl"

    with open(problem_fpath, 'w') as nf:
        prototype_content = None

        # -- read all content from the prototype file:
        with open(f'{robot}_problem_template.pddl', 'r') as df:
            prototype_content = df.read()

        # -- find and replace placeholders in the prototype file:
        new_content = prototype_content.replace('<init_state>', "\n\t".join(state))

        # -- write content to new PDDL file:
        nf.write(new_content)

    return problem_fpath


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

    parser.add_argument(
        "--robot",
        type=str,
        default="dorfl",
        help="This specifies the robot being used: ['dorfl', 'spot', 'panda'].",
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
        domain_fpath = create_domain_file(
            method='skillwrapper',
            yaml_data={
                'predicates': data_predicates,
                'operators': data_operators,
                'objects': data_objects,
            }
        )

        run_trials(
            domain_fpath,
            num_trials=10,
        )


