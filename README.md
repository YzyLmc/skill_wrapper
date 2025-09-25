# SkillWrapper: Generative Predicate Invention For Skill Abstraction

## Installation
This codebase relies on `openai`, `pytorch`, and `sentence_transformer`. Try installing them by:
```
conda create -n sw_env python=3.10 pytorch torchvision torchaudio -c pytorch && conda activate sw_env && pip install openai sentence-transformers
```
Evaluation requires fast downward dependency. Burger experiments require robotouille dependecy. You can install robotouille by
```
git submodule init
```
And then
```
git submodule update --recursive
```

## Files
### Source Code Overview
For constructing operators, we have:
- `main.py`: main function for running both predicate invention and skill sequence proposal.
- `invent_predicate.py`: functions for inventing predicate and construct operators from skill sequence images.
- `data_structure.py`: navtive data structures for storing predicates and abstract states, and logging purpose.
- `RCR_bridge.py`: data structures from Relational Critical Region for constructing and lifting operators from abstract transitions per cluster.
- `skill_sequenec_proposal.py`: functions for proposing new skill sequences based on previous skill sequences, abstract states of the skill sequences, and learned operators.

For evaluating learned PDDL operators, we have:
- `evaluate_predicates.py`: functions for evaluating truth values of all predicates in their possible grounded format given one image.
- `evaluate_pddl.py`: functions for planning with operators and predicates given initial image and goal image.
### Skill Sequence Data
Currently all skill sequences are stored under `transitions` of each iterations under each run `{METHOD}/{ENV_NAME}/runs/{RUN_IDX}/{ITER_IDX}/`, with the yaml file under `test_tasks/{ROBOT_NAME}` and images under `transitions/{DATETIME_STAMP}`.
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

`main.py` meant to automate skill sequence proposal and predicate invention and construct operators at the end of each iteration, but the natural of proposing and executing skill sequences blocks this automation from happening. Currently, the main function contains two flags `++skill_seq_only` and `++invent_pred_only` to separate one iteration into two parts.

To propose skill sequences, run:
```
python main.py ++skill_seq_only=True ++env=burger
```
the proposed skill sequences are not automatically saved in files. The feature is desired but now manually saving it from the logging info is good enough.

As one complete iteration, the skill sequence will be executed on real robots, either franka, dorfl, or spot. The execution will result in images of each sequence and one yaml file to store all necessary metadata for running predicate invention.

To invent predicates from the skill sequences, run:
```
python main.py ++invent_pred_only=True ++env=burger
```
The invented predicate set, the truth values of all possible grounded predicates, the constructed operators, and the logging info will by default be saved to `tasks/log`. This will result in five files:
- `{ROBOT_NAME}_log_raw_results_{IDX}.log`: raw logging info from the command line for inspecting purpose later.
- `grounded_predicate_truth_value_log.yaml`: truth values of all possible grounded predicates at each step across all skill sequences.
- `lifted_pred_list.yaml`: lifted predicates used to construct operators that are invented and passed the scoring function during the invention process.
- `skill2operator.pkl`: dictionary of skills to their corresponding operators, where the oeprators are saved in the native format using the data structure in RCR.
- `operators.yaml`: Everything is the same as the previous file, except the operators are saved in string format.
All files except the log files are necessary for running more iterations using the main function; The lifted predicate list and `operators.yaml` are necessary for evaluation scripts using fast downward for planning.

From the second iteration onward, it's required to load results from previous iterations. To do that, append the following flags to both commands:
```
--iter_idx --run_idx
```

### Baselines
- Oracle operators: operators and its predicates provided by human experts. (Note: oracle operators for dorfl is collected)
- ViLa: VLM-planning based method. Given the current observation, task specification, and action history, prompt the foundation model for the next action. The script is available and tested under `baselines/`, but haven't been run on real planning tasks.
- FM Invent: An ablation of our method: instead of inventing predicates using our method, directly prompt the foundation models for the predicate set and construct PDDL operators using the predicates.
- Our method.

For running baselines, check the readme under `baselines/`.

### Metrics
- Solved: Present 6 solvable planning problems, report the percentage of trials where the method found a valid plan that is executable on the robot.
- Planning Budget: For the 6 solvable planning problems, report the number of plans return from the planner have been tried executing until find a valid plan. The budget should have a cap, so that the budget can be used up and the number won't explode by one edge case. (This metric is borrowed from VisualPredicator)
- Impossible: Present 6 impossible planning problems, report the percentage of trials where the method successfully identify the planning problem has no solutions.

### Evaluation
For baselines that learn operators, run their corresponding files (either `main.py` or `baselines/{baseline_name}.py`). After learning the operators, run evaluation on a set of problem by first get the abstrat state with `eval_pred.py`, then run fastdownward to get the plans using `plan_with_operators.py`, finally, run `eval_plans.py` if you are running burger environment. Example command of running each script is in the header of each of these files.

