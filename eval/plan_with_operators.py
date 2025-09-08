"""
Load preadicate states of init state and goal state of each problem, learned operators & predicates, and plan with K star or FD.

Example usage:
    python eval/plan_with_operators.py --baseline expert_operators --dataset test
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

from src.utils import save_to_file, load_from_file
from src.data_structure import PredicateState, Skill

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

    print(" ".join(command))

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
    init_state: PredicateState,
    goal_state: PredicateState,
    problem_dir: str,
    num_trials: int = 10,
    method: str = 'fd',
):

    # -- count the number of successful plans were found:
    # TODO: how do we account for trials that are indeed unsolvable?
    count = 0

    data_per_trial = {
        "env": args.env,
        "plan_method": method,
    }

    for T in range(num_trials):
        data_per_trial[T] = {}

        problem_fpath = create_problem_file(init_state, goal_state, problem_dir, domain=f"{args.env}_skillwrapper", trial=T, )

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
            )
    # lowest level directory of problem_dir
    problem_num = problem_dir.split('/')[-1]
    save_dir = f"results/{args.baseline}/{args.env}/plans/{args.dataset}/{problem_num}"
    save_fpath = os.path.join(save_dir, "plan.yaml")
    os.makedirs(save_dir, exist_ok=True)
    save_to_file(data_per_trial, save_fpath)
    print(f"Saved plans to {save_fpath}")
    # json.dump(data_per_trial, open(save_fpath, "w"), indent=4)

def postprocess_plans(plans: list[str], yaml_data: dict):
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
                    skill = S

            for x in range(1, len(action_parts)):
                for O in objects:
                    if action_parts[x].startswith(O.lower()):
                        action_parts[x] = O

            grounded_skill = Skill(name=action_parts[0], params=action_parts[1:])
            processed_plan.append(
                f"{action_parts[0]}({', '.join([x for x in action_parts[1:]])})"
            )

        all_processed_plans.append(processed_plan)

    return all_processed_plans

def parse_predicate(pred: str, is_domain: bool = True):
    # # -- change all parentheses into commas for easy parsing and remove whitespaces; then remove any empty strings:
    # pred = list(filter(None, str(pred).replace('(', ',').replace(')', ',').replace(' ', '').split(',')))
    # # -- extract the predicate name and all proceeding arguments :
    # name, args_no_variables = pred[0], pred[1:]
    if is_domain:
        name, args_no_variables = pred.name, pred.types
    else:
        name, args_no_variables = pred.name, pred.params

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
    env: str,
) -> str:

    object_types = set()
    for obj in yaml_data['objects']:
        for obj_type in yaml_data['objects'][obj]['types']:
            object_types.add(f'{obj} - {obj_type}')
            object_types.add(f'{obj_type} - object')

    problem_dir = f"results/{args.baseline}/{args.env}/runs/{args.run_idx}/"

    domain_fpath = os.path.join(problem_dir, f'{env}_domain_{method}.pddl')

    with open(domain_fpath, 'w') as nf:
        prototype_content = None

        # -- read all content from the prototype file:
        with open('planning/domain_prototype.pddl', 'r') as df:
            prototype_content = df.read()

        # -- find and replace placeholders in the prototype file:
        new_content = prototype_content.replace('<actions>', "\n\n".join(yaml_data["operators"]))
        new_content = new_content.replace('<types>', "\n\t\t".join(list(object_types)))
        new_content = new_content.replace('<predicates>', "\n\t\t".join(yaml_data["predicates"]))
        new_content = new_content.replace('<domain>', f"{env}_{method}")

        # -- write content to new PDDL file:
        nf.write(new_content)

    return domain_fpath

def create_problem_file(
    init_state: PredicateState,
    goal_state: PredicateState,
    problem_dir,
    domain: str,
    trial: int = 0,
) -> str:

    init_state_strs, goal_state_strs = [], []

    # -- generating the initial state for problem file:
    for pred in init_state.iter_predicates():
        if init_state.get_pred_value(pred):
            init_state_strs.append(parse_predicate(pred, is_domain=False))

    # -- generating the goal state for problem file:

    for pred in goal_state.iter_predicates():
        if goal_state.get_pred_value(pred):
                goal_state_strs.append(parse_predicate(pred, is_domain=False))

    init_state_strs = list(filter(None, init_state_strs))
    goal_state_strs = list(filter(None, goal_state_strs))

    problem_fpath = os.path.join(problem_dir, f"{args.env}_problem_trial-{trial}.pddl")

    with open(problem_fpath, 'w') as nf:
        prototype_content = None

        # -- read all content from the prototype file:
        with open(f'planning/{args.env}_problem_template.pddl', 'r') as df:
            prototype_content = df.read()

        # -- find and replace placeholders in the prototype file:
        new_content = prototype_content.replace('<init_state>', "\n\t".join(init_state_strs))
        new_content = new_content.replace('<goal_state>', "\n\t".join(goal_state_strs))
        new_content = new_content.replace('<domain>', domain)

        # -- write content to new PDDL file:
        nf.write(new_content)

    return problem_fpath

def main():

    # load predicates
    pred_fpath = f"results/{args.baseline}/{args.env}/runs/{args.run_idx}/predicates/predicates.yaml"
    data_predicates = load_from_file(pred_fpath)
    print(f"Loaded predicates from {pred_fpath}")

    # load operators
    op_fpath = f"results/{args.baseline}/{args.env}/runs/{args.run_idx}/operators/operators.yaml"
    data_operators = load_from_file(op_fpath)
    print(f"Loaded operators from {op_fpath}")

    # load task config
    task_config_fpath = f"task_config/{args.env}.yaml"
    data_objects = load_from_file(task_config_fpath)

    # create domain file
    yaml_data = {
        "operators": [O.pop() for _, O in data_operators.items()],   
        "predicates": [parse_predicate(P) for P in data_predicates],
        "objects": data_objects['objects'],
        "skills": data_objects['skills'],
    }

    domain_fpath = create_domain_file(
                method='skillwrapper',
                yaml_data=yaml_data,
                env=args.env,
            )

    # loop through all problems under a dataset
    problem_dir = f"results/{args.baseline}/{args.env}/pred_state/{args.dataset}/"
    for root, dirs, files in os.walk(problem_dir):
        for d in dirs:
            print(f"Processing problem {d} in {root}...")

            init_state = load_from_file(os.path.join(root, d, f"init_state_{args.input_modality}.yaml"))
            goal_state = load_from_file(os.path.join(root, d, f"goal_state_{args.input_modality}.yaml"))

            run_trials(
                domain_fpath,
                yaml_data,
                init_state,
                goal_state,
                problem_dir=os.path.join(root, d),
                num_trials=args.ntrials,
                method=args.planner,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, choices=["dorfl", "franka", "spot", "burger"], default="burger", help="the name of the environment")
    parser.add_argument("--baseline", type=str, choices=["FMinvent", "oracle_predicates", "expert_operators", "random_explore", "skillwrapper"], help="the name of the baseline")
    parser.add_argument("--dataset", type=str, choices=["test", "unseen", "easy", "hard"], help="the name of the dataset")
    parser.add_argument("--run_idx", type=int, default=0, help="index of the run that produce the best operators.")
    parser.add_argument("--input_modality", type=str, choices=["image", "text"], default="image", help="the input modality of the state")

    parser.add_argument(
        "--planner",
        type=str,
        default="fd",
        help="This specifies the planner to use: ['fd', 'kstar'] (default: 'fd').",
    )

    parser.add_argument(
        "--ntrials",
        type=int,
        default=1,
        help="This specifies the number of trials to run this process (default: 1).",
    )

    args = parser.parse_args()

    main()