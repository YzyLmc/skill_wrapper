# SkillWrapper: Autonomously Learning Interpretable Skill Abstractions with Foundation Models

## Updated Installation

If you don't have `uv`, run the first command to install it. Run the second command to create a `uv` virtual environment with the project's dependencies.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

TODO: Evaluation requires fast downward dependency.

## Updated Run Instructions

**Phase 1**: Skill Sequence Proposal - Run the command:
```sh
uv run src/main.py --task_config_fpath task_config/franka.yaml --propose_skill_sequence_only
```

TODO: The proposed skill sequences are not automatically saved in files; make sure to manually save them.

**Phase 2**: Execute the Skills - TODO (needs instructions).

Skill execution will produce images capturing the states between skills, and a YAML file storing the metadata needed to run predicate invention.

**Phase 3**: Predicate Proposal - Run the command (TODO: What path should be used for the task config? Does it need to be given the YAML file there?)
```sh
uv run src/main.py --task_config_fpath task_config/franka.yaml ----invent_pred_only
```

## Files
### Source Code Overview
All source code are under `src/`. Specifically for constructing operators, we have:
- `main.py`: main function for running both predicate invention and skill sequence proposal.
- `invent_predicate.py`: functions for inventing predicate and construct operators from skill sequence images.
- `data_structure.py`: navtive data structures for storing predicates and abstract states, and logging purpose.
- `RCR_bridge.py`: data structures from Relational Critical Region for constructing and lifting operators from abstract transitions per cluster.
- `skill_sequenec_proposal.py`: functions for proposing new skill sequences based on previous skill sequences, abstract states of the skill sequences, and learned operators.

For evaluating learned PDDL operators, we have:
- `evaluate_predicates.py`: functions for evaluating truth values of all predicates in their possible grounded format given one image.
- `evaluate_pddl.py`: functions for planning with operators and predicates given initial image and goal image.
### Skill Sequence Data
Currently all skill sequences are stored under `test_tasks`, with the yaml file under `test_tasks/{ROBOT_NAME}` and images under `test_tasks/{ROBOT_NAME}_images`.
### Prompts
All prompts are stored in yaml files under `prompts/` indexed by robot name. PDDL templates for dorfl are temporarily stored under the same folder too.
### Metadata
Metadata of each environment are stored in yaml files under `task_config`. Each config file contains the name of the environment, the skills and their semantics, objects and their types, and one sentence of verbal description.
### Planning
All planning scripts are under `planning`. We have templates of planning problems, oracle operators and predicates for dorfl domain ready. (See experiments section for more details.)
### Baselines
All baselines should be under `baselines` (currently only ViLa is implemented).

## Running the Code
Before running the code, export your openai key by `export OPENAI_API_KEY={YOUR_KEY}`.

`main.py` meant to automate skill sequence proposal and predicate invention and construct operators at the end of each iteration, but the natural of proposing and executing skill sequences blocks this automation from happening. Currently, the main function contains two flags `--propose_skill_sequence_only` and `--invent_pred_only` to separate one iteration into two parts.

To propose skill sequences, run:
```
python src/main.py --task_config_fpath task_config/franka.yaml --propose_skill_sequence_only
```
the proposed skill sequences are not automatically saved in files. The feature is desired but now manually saving it from the logging info is good enough.

As one complete iteration, the skill sequence will be executed on real robots, either franka, dorfl, or spot. The execution will result in images of each sequence and one yaml file to store all necessary metadata for running predicate invention.

To invent predicates from the skill sequences, run:
```
python src/main.py --task_config_fpath task_config/franka.yaml ----invent_pred_only
```
The invented predicate set, the truth values of all possible grounded predicates, the constructed operators, and the logging info will by default be saved to `tasks/log`. This will result in five files:
- `{ROBOT_NAME}_log_raw_results_{IDX}.log`: raw logging info from the command line for inspecting purpose later.
- `grounded_predicate_truth_value_log.yaml`: truth values of all possible grounded predicates at each step across all skill sequences.
- `lifted_pred_list.yaml`: lifted predicates used to construct operators that are invented and passed the scoring function during the invention process.
- `skill2operator.pkl`: dictionary of skills to their corresponding operators, where the oeprators are saved in the native format using the data structure in RCR.
- `skill2operator.yaml`: Everything is the same as the previous file, except the operators are saved in string format.
All files except the log files are necessary for running more iterations using the main function; The lifted predicate list and `skill2operator.yaml` are necessary for evaluation scripts using fast downward for planning.

From the second iteration onward, it's required to load results from previous iterations. To do that, append the following flags to both commands:
```
--load_fpath {PATH_TO_PREVIOUS_RESULT_FILES}
```
Where the path will be `tasks/log` again if the files are saved to the default path. NOTE: You should also put the up-to-date yaml file for skill sequences under the same directory.

An example of result files can be found under `example_dorfl_result/`.

## Experiments (WIP)
This section meant to proide verbal descriptions to the empty result table on overleaf.
Many of the experiment section are either ongoing or not started. The finished part will be mentioned, otherwise they are not done yet.
### Overview
We planned to have two sets of experiments (end-to-end, planning-only) on three robots. For each set, we want to have 12 trials, 6 for solved and planning budge, and 6 for impossible. We will compare our system against three other baselines.
### End-to-end vs Planning only
Since the baselines will have different predicates and thus different abstract states for same low-level state. **End-to-end** means the abstract states will be obtained by querying foundation models and could be inaccurate, and it aims to evaluate the performance of the system as a whole; **Planning-only** means the abstract states are provided by oracles and thus accurate, and it aims to evaluate the quality of learning operators for planning purpose.
### Baselines
- Oracle operators: operators and its predicates provided by human experts. (Note: oracle operators for dorfl is collected)
- ViLa: VLM-planning based method. Given the current observation, task specification, and action history, prompt the foundation model for the next action. The script is available and tested under `baselines/`, but haven't been run on real planning tasks.
- FM Invent: An ablation of our method: instead of inventing predicates using our method, directly prompt the foundation models for the predicate set and construct PDDL operators using the predicates.
- Our method.
### Metrics
- Solved: Present 6 solvable planning problems, report the percentage of trials where the method found a valid plan that is executable on the robot.
- Planning Budget: For the 6 solvable planning problems, report the number of plans return from the planner have been tried executing until find a valid plan. The budget should have a cap, so that the budget can be used up and the number won't explode by one edge case. (This metric is borrowed from VisualPredicator)
- Impossible: Present 6 impossible planning problems, report the percentage of trials where the method successfully identify the planning problem has no solutions.
