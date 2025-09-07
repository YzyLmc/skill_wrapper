"Read initial state, ask for human input to reach goal state, and record initial state and goal state in img, yaml, and text."

import os
import sys
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append(f".") # if you run this script from the root directory
sys.path.append("robotouille")
import robotouille
from robotouille.skills import render_img
from robotouille.agents import NAME_TO_AGENT
from robotouille.robotouille.robotouille_env import create_robotouille_env
from src.data_structure import Predicate, PredicateState
from src.utils import save_to_file, load_from_file

def env_state_to_pred_state(env, save_fpath):
    """
    Save a predicate state in env.current_state using predicates in data structure.
    
    Parameters
    ----------
    env : robotouille.Environment
        The environment object containing the current state.
    save_fpath : str
        The file path to save the predicate state yaml file.
    """
    state = env.current_state
    pred_state = PredicateState([])
    # NOTE: We might need bread onion tomato chicken and patato later.
    bad_preds = ["istable", "isfryer", "issink", "isbread", "isonion", "istomato", "ischicken", "ispotato", "isfryable", "isfryableifcut", "isfried", "iscooking", "ispot", "isbowl", "iswater", "isboiling", "loc", "container_empty", "vacant", "has_container", "in", "addedto",  "container_at"]
    type_dict = {"item": "pickupable", "station": "location", "player": "robot"}
    for literal, is_true in state.predicates.items():
            if literal.name not in bad_preds:
                name = literal.name
                params = literal.params
                types = [type_dict[t] for t in literal.types]
                grounded_pred = Predicate(name=name, params=params, types=types)
                pred_state.pred_dict[grounded_pred] = is_true
    save_to_file(pred_state, save_fpath)

def env_state_to_text(env, save_fpath):
    wait = [a for a in env.current_state.get_valid_actions_and_str()[0] if a[0].name == "wait"][0]
    obs, reward, done, info = env.step([wait])
    lines = obs.split("\n\n")
    cleared_lines = []
    for line in lines:
        # Remove lines that contains "table"
        l = line.split("\n")
        cleared_line = [i for i in l if "table" not in i]
        if cleared_line:
            cleared_lines.append("\n".join(cleared_line).strip())
    obs = "\n\n".join(cleared_lines[:-2]) # remove available actions and the goal state description

    save_to_file(obs, save_fpath)

@hydra.main(version_base=None, config_path="../robotouille/conf", config_name="data_collection_config")
def main(cfg: DictConfig):
    """
    Use a GUI to collect initial and goal states from the user. User will be prompted with the goal state description.
    Only the two states is recorded, not the trajectory.
    The process terminates as soon as the user reach the goal state.
    """
    kwargs = OmegaConf.to_container(cfg.game, resolve=True)
    environment_name = kwargs.pop('environment_name')
    agent_name = kwargs.pop('agent_name')

    # Initialize environment
    seed = kwargs.get('seed', None)
    noisy_randomization = False
    env = create_robotouille_env(environment_name, seed, noisy_randomization)
    renderer = env.renderer
    render_mode = kwargs.get('render_mode', 'human')
    # Initialize agent
    llm_kwargs = kwargs.get('llm_kwargs', {})
    agent = NAME_TO_AGENT[agent_name](llm_kwargs)
    agent_done_cond = lambda a: a.is_done() if a is not None else False

    obs, info = env.reset()
    print("Goal State:")
    print(env.current_state.goal)

    dataset = os.path.split(environment_name)[0]
    save_fpath = os.path.join("eval", "data", "burger", dataset)
    print(save_fpath)
    if save_fpath is not None:
        os.makedirs(save_fpath, exist_ok=True)
    init_img_fpath = os.path.join(save_fpath, "init_state.jpg")
    render_img(env, env.current_state, init_img_fpath)
    init_pred_state_fpath = os.path.join(save_fpath, "init_state.yaml")
    env_state_to_pred_state(env, init_pred_state_fpath)
    init_pred_state_text = os.path.join(save_fpath, "init_state.txt")
    env_state_to_text(env, init_pred_state_text)

    done = False
    queued_actions = []
    while not done and not agent_done_cond(agent):
        img = env.render(render_mode)
        if len(queued_actions) == 0:
            # Retrieve action(s) from agent output
            proposed_actions = agent.propose_actions(obs, env)
            if len(proposed_actions) == 0:
                # Reprompt agent for action(s)
                continue
            action, param_arg_dict = proposed_actions[0]
            queued_actions = proposed_actions[1:]
        else:
            action, param_arg_dict = queued_actions.pop(0)
        
        # Assign action to players
        actions = []
        current_state = env.current_state
        for player in current_state.get_players():
            if player == current_state.current_player:
                actions.append((action, param_arg_dict))
            else:
                actions.append((None, None))
        
        # Step environment
        obs, _, done, _ = env.step(actions)

    goal_img_fpath = os.path.join(save_fpath, "goal_state.jpg")
    render_img(env, env.current_state, goal_img_fpath)
    goal_pred_state_fpath = os.path.join(save_fpath, "goal_state.yaml")
    env_state_to_pred_state(env, goal_pred_state_fpath)
    goal_pred_state_text = os.path.join(save_fpath, "goal_state.txt")
    env_state_to_text(env, goal_pred_state_text)

    print("Data saved to: ", save_fpath)

if __name__ == "__main__":
    """
    Arguments & Default values:
    environment_name: The json file full path respect to eval/data/burger

    Example terminal command:
        python eval/burger_data_collection.py ++game.environment_name=test/problems/2/problem 
    """
    main()