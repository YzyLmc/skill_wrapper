import os
import sys
from random import randint, choice
import argparse

from data_structure import yaml
from subprocess import check_output, CalledProcessError

from evaluate_predicates import eval_all_predicates
from utils import GPT4, load_from_file

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

        print(problem_fpath)
        print(domain_fpath)

        solution = find_plan(
            problem_file=problem_fpath,
            domain_file=domain_fpath,
        )

        print(solution)

        count += int(len(solution) > 0)

        if solution:
            print("plan has been found!")
            for x in range(len(solution)):
                print(f"{x+1} : {solution[x]}")



def parse_predicate(pred: str, is_domain: bool = True):
    # -- change all parentheses into commas for easy parsing and remove whitespaces; then remove any empty strings:
    pred = list(filter(None, str(pred).replace('(', ',').replace(')', ',').replace(' ', '').split(',')))
    # -- extract the predicate name and all proceeding arguments :
    name, args_no_variables = pred[0], pred[1:]

    # -- we need to format predicates with question marks for variables:
    args_with_variables = []
    for arg in args_no_variables:
        # -- we will format as "?<obj_type> - <obj_type>":
        if is_domain:
            args_with_variables.append(f'?{arg} - {arg}')
        else:
            args_with_variables.append(f'{arg}')

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
        with open('prompts/domain_prototype.pddl', 'r') as df:
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

    init_state, goal_state = [], []

    # -- generating the initial state for problem file:
    if args.init_img:
        args.img_fpath = args.init_img

        task_config = load_from_file(args.task_config_fpath)
        args.env = task_config["env"]
        type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}
        lifted_pred_list = load_from_file(args.predicate_fpath)
        model = GPT4(engine=args.model)

        predicate_state = eval_all_predicates(model, lifted_pred_list, type_dict, args=args)

        for pred in predicate_state.iter_predicates():
            if predicate_state.get_pred_value(pred):
                init_state.append(parse_predicate(pred, is_domain=False))

    else:
        if robot == "dorfl":
            init_state = [
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

    # -- generating the goal state for problem file:
    if args.goal_img:
        args.img_fpath = args.goal_img

        task_config = load_from_file(args.task_config_fpath)
        args.env = task_config["env"]
        type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}
        lifted_pred_list = load_from_file(args.predicate_fpath)
        model = GPT4(engine=args.model)

        predicate_state = eval_all_predicates(model, lifted_pred_list, type_dict, args=args)

        for pred in predicate_state.iter_predicates():
            if predicate_state.get_pred_value(pred):
                    goal_state.append(parse_predicate(pred, is_domain=False))

    else:
        if robot == "dorfl":
            goal_state = [
                "(is_spread PeanutButter Bread)",
            ]

    init_state = list(filter(None, init_state))
    goal_state = list(filter(None, goal_state))

    problem_fpath = os.path.join(os.getcwd(), f"{robot}_problem_trial-{trial}.pddl")

    with open(problem_fpath, 'w') as nf:
        prototype_content = None

        # -- read all content from the prototype file:
        with open(f'prompts/{robot}_problem_template.pddl', 'r') as df:
            prototype_content = df.read()

        # -- find and replace placeholders in the prototype file:
        new_content = prototype_content.replace('<init_state>', "\n\t".join(init_state))
        new_content = new_content.replace('<goal_state>', "\n\t".join(goal_state))

        # -- write content to new PDDL file:
        nf.write(new_content)

    return problem_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot", type=str, default="dorfl", help="This specifies the robot being used: ['dorfl', 'spot', 'panda'].", )

    parser.add_argument("--operator_fpath", type=str, default=None, help="This specifies the path to the YAML file containing all operators.", )
    parser.add_argument("--predicate_fpath", type=str, default=None, help="This specifies the path to the YAML file containing all predicates.", )
    parser.add_argument("--task_config_fpath", type=str, default=None, help="This specifies the path to the YAML file containing all objects.", )

    parser.add_argument("--init_img", type=str, default=None, help="This specifies the path to an image of the robot's INITIAL STATE observation.", )
    parser.add_argument("--goal_img", type=str, default=None, help="This specifies the path to an image of the robot's FINAL STATE observation.", )

    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20'], default='gpt-4o-2024-11-20')
    parser.add_argument("--save_dir", type=str, default=".", help="directory to save log files")

    args = parser.parse_args()

    data_predicates, data_operators, data_objects = [], [], []

    if args.operator_fpath:
        with open(args.operator_fpath, "r") as f:
            data_operators = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('-- Missing YAML file containing operator definitions!')
        sys.exit()

    if args.predicate_fpath:
        with open(args.predicate_fpath, "r") as f:
            data_predicates = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('-- Missing YAML file containing predicate definitions!')
        sys.exit()

    if args.task_config_fpath:
        with open(args.task_config_fpath, "r") as f:
            data_objects = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('-- Missing YAML file containing object definitions!')
        sys.exit()

    observation_imgs = {
        'init': args.init_img,
        'goal': args.goal_img,
    }

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


