## Baselines

All baselines of the paper.

- vila: 
    - Train: Directly output plans. No operators or predicates.
    - Eval: Given skills and objects, iteratively takes in observation & goal and action history, generate the next action. 
    - Relevant files: `vila.py`, `eval_plan.py`

- expert operators: 
    - Train: PDDL predicates and operators written by experts. 
    - Eval: Takes in observation & goal, grounds them into abstract state, and use a planner to compute the plan.
    - Relevant files: `expert_pddl.py`, `eval_preds.py`, `plan_with_operators.py`, `eval_plan.py`

- oracle predicates: 
    - Train:  Same transitions as skillwrapper but are directly in abstract state, predicates are provided by burger. Operators are learned using skillwrapper.
    - Eval: Takes in observation & goal, grounds them into abstract state, and use a planner to compute the plan.
    - Relevant files: `oracle_predicates.py`, `eval_preds.py`, `plan_with_operators.py`, `eval_plan.py`.

- random explore:
    - Train: Transitions collected randomly, predicates invented by skillwrapper, operators learned using skillwrapper.
    - Eval: Takes in observation & goal, grounds them into abstract state, and use a planner to compute the plan.
    - Relevant files: `random_explore_data_gen.py`, `eval_preds.py`, `random_explore_exec.py`, `plan_with_operators.py`, `eval_plan.py`

- FM Invent:
    - Train: Transition using skillwrapper, predicates invented by foundation model in one-shot, oeprators learned using skillwrapper.
    - Eval: Takes in observation & goal, grounds them into abstract state, and use a planner to compute the plan.
    - Relevant files: `fm_invent.py`, `eval_preds.py`, `plan_with_operators.py`, `eval_plan.py`
