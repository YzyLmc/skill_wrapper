## Results & Baselines
Learning:
- For each baseline except ViLa, we run the algorithms 5 times. At each time, the propose skill seqeunces, transitions in abstract states, and learned operators are stored in `{method}/{env}/{runs}/{run_number}/`.

Evaluation:
- For each baseline except ViLa, we pick the best operators out of the 5 using a test set consisting of 10 planning problems under `eval/data/{domain}/test/problems/{problem_num}`. Then, we obtain the abstract state using the learned predicates of each method and stored them under `{method}/{env}/pred_state/{dataset_name}/{problem_num}`. After that, we use off-the-shelf PDDL planner to find the plan using the abstrat states and the learned operators. The plans is saved at `{method}/{env}/plans/{dataset_name}/{problem_num}/plan.yaml`. Finally, we evaluate the returned plans by executing them in the actual domain, and save the results as `results/{method}/{domain}/{domain}_{dataset_name}_results.json`