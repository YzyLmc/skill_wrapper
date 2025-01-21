## Solve Example Problem with Fast Downward

To run the examples, first make sure you have [docker](https://docs.docker.com/get-docker/) installed and docker daemon running.

After that, you can pull and run docker image for Downward and solve PDDL problems in one line (after export the absolute path to `downward_examples/` to `$D_PATH`):

```
sudo docker run --rm -v $D_PATH:/downward_examples aibasel/downward --alias lama-first /downward_examples/domain_simple.pddl /downward_examples/problem_simple.pddl
```

To run a harder example, which is the task we proposed for the workshop submission, run the following command:

```
sudo docker run --rm -v $D_PATH:/downward_examples aibasel/downward --alias lama-first /downward_examples/domain_kitchen.pddl /downward_examples/problem_kitchen.pddl
```

Meanwhile, there is an [online editor](https://editor.planning.domains/#) for PDDL that can solve PDDL problems if you cannot make Downward working and need some detail explanantion of the error.