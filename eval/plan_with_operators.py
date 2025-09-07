"""
Load preadicate states of init state and goal state of each problem, learned operators & predicates, and plan with K star or FD.

"""
import os
import sys
sys.path.append(f".")
import argparse
import json

from subprocess import check_output, CalledProcessError

# Check here for more info on Kstar Planner: https://github.com/IBM/kstar
try:
    from kstar_planner import planners
    from pathlib import Path
except ImportError:
    print("WARNING: 'kstar-planner' has not been installed!\n -- Link to repository: https://github.com/IBM/kstar")
    kstar_available = False
else:
    kstar_available = True

from src.utils import save_to_file, load_from_file, get_save_fpath
from src.evaluate_predicates import evaluate_predicates
from src.data_structure import yaml

planner_path = '/home/ziyi/git/downward/fast-downward.py'

algorithms = ['astar', 'eager', 'lazy', ]
heuristics = ['lmcut', 'ff', ]

original_skills = None

def find_topk_plans(
    domain_file: str,
    problem_file: str,
    num_k: int = 10,
    topq: bool = False,
):
    if not kstar_available: return None

    domain_fpath = Path(domain_file)
    problem_fpath = Path(problem_file)

    # NOTE: refer to the following section of README: 
    # https://github.com/IBM/kstar?tab=readme-ov-file#quickstart-using-kstar-planner-as-a-python-package

    if topq:
        heuristic = "ipdb(transform=undo_to_origin())"

        output = planners.plan_topq(
            domain_file=domain_file, 
            problem_file=problem_file, 
            quality_bound=1.0,
            number_of_plans_bound=num_k, # NOTE: you can define some value of k
            timeout=30, # NOTE: max amount of time in seconds to allow the planner to find the k solutions
            search_heuristic=heuristic,
        )

    else:
        output = planners.plan_topk(
            domain_file=domain_fpath, 
            problem_file=problem_fpath, 
            number_of_plans_bound=num_k, # NOTE: you can define some value of k
            timeout=30, # NOTE: max amount of time in seconds to allow the planner to find the k solutions
        )

    # NOTE: output will be in the form of a dictionary; just extract the "plans" value!
    return output['plans']

def find_plan(
    domain_file: str,
    problem_file: str,
    algorithm: str = 'astar',
    heuristic: str = 'lmcut',
    verbose: bool = False,
) -> str:

    # NOTE: define plan execution function that can be called for different parameters:
    command = [
        'python3', planner_path, 
        domain_file, problem_file,
        '--search', f'{algorithm}({heuristic}())',
    ]

    plan = []

    try:
        _ = check_output(command)
    except CalledProcessError as e:
        if verbose:
            print(f"error code: {e.returncode}\n\t-- Actual message: {str(e.output)}")
    else:
        with open('sas_plan', 'r') as f:
            for _line in f.readlines():
                if ';' not in _line: plan.append(_line.strip())

    return plan

def run_trials(
    domain_fpath: str,
    yaml_data: dict,
    num_trials: int = 10,
    method: str = 'fd',
):

    # -- count the number of successful plans were found:
    # TODO: how do we account for trials that are indeed unsolvable?
    count = 0

    data_per_trial = {
        "setting": args.setting,
        "plan_method": method,
    }

    for T in range(num_trials):
        data_per_trial[T] = {}

        problem_fpath, state = create_problem_file(setting=args.setting, trial=T, )

        data_per_trial[T]['state'] = state

        print(f"\n{'*' * 10} TRIAL {T+1} {'*' * 10}")

        if method == 'fd':
            solution = find_plan(
                problem_file=problem_fpath,
                domain_file=domain_fpath,
            )

            # -- check to see if there has been a solution (if plan is not empty):
            count += int(len(solution) > 0)

            if solution:
                print(" -- plan has been found!")
                for x in range(len(solution)):
                    print(f"\t{x+1} : {solution[x]}")
            else:
                print(" -- no solution found!")

            data_per_trial[T]['all_plans'] = [solution]

            data_per_trial[T]['all_parsed_plans'] = postprocess_plans(
                plans=[solution], 
                yaml_data=yaml_data, 
                setting=args.setting,
            )

        elif method == 'kstar':
            solutions = find_topk_plans(
                problem_file=problem_fpath,
                domain_file=domain_fpath,
            )

            count += int(len(solutions) > 0)
            
            if solutions:
                print(f" -- {len(solutions)} plans have been found!")
                for x in range(len(solutions)):
                    print(f"\tplan {x}:")
                    for y in range(len(solutions[x]['actions'])):
                        print(f"\t\t{y+1} : {solutions[x]['actions'][y]}")
            else:
                print(" -- no solution found!")

            data_per_trial[T]['all_plans'] = solutions

            data_per_trial[T]['all_parsed_plans'] = postprocess_plans(
                plans=solutions, 
                yaml_data=yaml_data, 
                setting=args.setting,
            )
    timestamp = dt.today().strftime('%Y-%m-%d_%H-%M-%SS')

    json.dump(data_per_trial, open(f"{timestamp}_all_trials.json", "w"), indent=4)

def postprocess_plans(plans: list[str], yaml_data: dict, setting: str):
    all_processed_plans = []

    skills = yaml_data['skills']
    objects = yaml_data['objects']

    for plan in plans:
        processed_plan = []
        for action in plan:
            action_parts = action.replace("(", "").replace(")", "").split(" ")

            # -- we will replace any auxiliary expert-defined skills with those of the basic ones in the YAML:
            for S in skills:
                if action_parts[0].startswith(S.lower()):
                    action_parts[0] = S

            for x in range(1, len(action_parts)):
                for O in objects:
                    if action_parts[x].startswith(O.lower()):
                        action_parts[x] = O
                        
            processed_plan.append(
                f"{action_parts[0]}({', '.join([x for x in action_parts[1:]])})"
            )

        all_processed_plans.append(processed_plan)

    return all_processed_plans

def parse_predicate(pred: str):
    # -- change all parentheses into commas for easy parsing and remove whitespaces; then remove any empty strings:
    pred = list(filter(None, str(pred).replace('(', ',').replace(')', ',').replace(' ', '').split(',')))
    # -- extract the predicate name and all proceeding arguments :
    name, args_no_variables = pred[0], pred[1:]

    # -- we need to format predicates with question marks for variables:
    args_with_variables = []
    for a in range(len(args_no_variables)):
        # -- we will format as "?<obj_type> - <obj_type>":
        args_with_variables.append(f'?arg{a} - {args_no_variables[a]}')

    # -- return a PDDL-structured predicate:
    return f"({name} {' '.join(args_with_variables)})"

def create_domain_file(
    method: str,
    yaml_data: list,
    setting: str,
) -> str:

    object_types = set()
    for obj in yaml_data['objects']:
        for obj_type in yaml_data['objects'][obj]['types']:
            object_types.add(f'{obj} - {obj_type}')
            object_types.add(f'{obj_type} - object')

    domain_fpath = os.path.join(os.getcwd(), f'{setting}_domain_{method}.pddl')

    with open(domain_fpath, 'w') as nf:
        prototype_content = None

        # -- read all content from the prototype file:
        with open('planning/domain_prototype.pddl', 'r') as df:
            prototype_content = df.read()

        # -- find and replace placeholders in the prototype file:
        new_content = prototype_content.replace('<actions>', "\n\n".join(yaml_data["operators"]))
        new_content = new_content.replace('<types>', "\n\t\t".join(list(object_types)))
        new_content = new_content.replace('<predicates>', "\n\t\t".join(yaml_data["predicates"]))
        new_content = new_content.replace('<domain>', f"{setting}_{method}")

        # -- write content to new PDDL file:
        nf.write(new_content)

    return domain_fpath

def create_problem_file(
    trial: int = 0,
) -> str:

    init_state, goal_state = [], []

    # -- generating the initial state for problem file:

    predicate_state = load_from_file()

    for pred in predicate_state.iter_predicates():
        if predicate_state.get_pred_value(pred):
            init_state.append(parse_predicate(pred, is_domain=False))

    # -- generating the goal state for problem file:
    predicate_state = load_from_file()

    for pred in predicate_state.iter_predicates():
        if predicate_state.get_pred_value(pred):
                goal_state.append(parse_predicate(pred, is_domain=False))

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
        "--setting",
        type=str,
        default="dorfl",
        help="This specifies the task setting being considered: ['dorfl', 'spot', 'panda', 'burger'] (default: 'dorfl').",
    )

    parser.add_argument(
        "--planner",
        type=str,
        default="fd",
        help="This specifies the planner to use: ['fd', 'kstar'] (default: 'fd').",
    )

    parser.add_argument(
        "--ntrials",
        type=int,
        default=2,
        help="This specifies the number of trials to run this process (default: 1).",
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
        yaml_data = {
            "operators": [O.pop() for _, O in data_operators.items()],   
            "predicates": [parse_predicate(P) for P in data_predicates],
            "objects": data_objects['objects'],
            "skills": data_objects['skills'],
        }

        domain_fpath = create_domain_file(
            method='skillwrapper',
            yaml_data=yaml_data,
            setting=args.setting,
        )

        data_skills = data_objects

        run_trials(
            domain_fpath,
            yaml_data,
            num_trials=args.ntrials,
            method=args.planner,
        )


