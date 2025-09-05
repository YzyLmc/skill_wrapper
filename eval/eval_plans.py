"For all problems, run the result skill sequence, return Done if the goal state is reached, and log them."
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append(f".") # if you run this script from the root directory
import robotouille
from robotouille.skills import SkillManager
from robotouille.robotouille.robotouille_env import create_robotouille_env
from src.utils import save_to_file, load_from_file
from robotouille.agents import NAME_TO_AGENT

@hydra.main(version_base=None, config_path="../robotouille/conf", config_name="test_config")
def main(cfg: DictConfig):
    kwcfg = OmegaConf.to_container(cfg.game, resolve=True)
    _ = kwcfg.pop('environment_name') # remove 
    problem_dir = f"eval/data/{cfg.env}/{cfg.dataset}/problems/"

    # list all directories under problem_dir
    results = {}
    for root, dirs, files in os.walk(problem_dir):
        for d in dirs:
            root_components = root.split(os.sep)[-3:]
            root_path = os.sep.join(root_components)

            environment_name = os.path.join(root_path, d, "problem")
            env = create_robotouille_env(environment_name, cfg.game.seed)
            plan_dir = os.path.join("results", cfg.baseline, cfg.env, "plans", cfg.dataset, d)
            # List all files under plan_dir
            file_list = os.listdir(plan_dir)
            results[d] = {"suc": {}}
            for f in file_list:
                plan_fpath = os.path.join(plan_dir, f)
                plan = load_from_file(plan_fpath)["plan"]
                suc = get_successfulness(env, plan)
                results[d]["suc"][f] = suc

            if any(results[d]["suc"].values()):
                results[d]["solved"] = True
                # planning budget is the index of first successful plan
                results[d]["planning_budget"] = min(i for i, suc in enumerate(results[d]["suc"].values()) if suc)
            else:
                results[d]["solved"] = False
                results[d]["planning_budget"] = len(results[d]["suc"])

    assert results, "No results found!"

    log_file = os.path.join("results", cfg.baseline, cfg.env, f"{cfg.env}_{root_components[-3]}_results.json")
    save_to_file(results, log_file)
    print("Results saved to", log_file)


def get_successfulness(env, plan) -> bool:
    # run the plan and return true if the goal state is reached at the end of the plan
    skill_manager = SkillManager(env)
    for i, skill in enumerate(plan):
        suc = skill_manager.execute_skill(skill)
        # Ugly way of getting "done"
        wait = [a for a in skill_manager.env.current_state.get_valid_actions_and_str()[0] if a[0].name == "wait"][0]
        _, _, done, _ = env.step([wait])
    return done

if __name__ == "__main__":
    """
    Arguments & Default values:
    baseline: Name of the baseline {"FMinvent", "oracle", "random_explore", "vila", "skillwrapper"}
    dataset: ["test", "seen", "unseen"]
    env: Name of the environment. For now only burger is supported.

    Example command:
        python eval/eval_plans.py +dataset=seen +baseline=vila ++env=burger
    """
    main()