import subprocess

# Path to the Fast Downward directory
fd_path = '/path/to/FastDownward.py'

# Path to domain and problem files
domain_file = 'path/to/domain.pddl'
problem_file = 'path/to/problem.pddl'

# Command to run Fast Downward with some search algorithm, here using A*
command = [fd_path, domain_file, problem_file, '--search', 'astar(blind)']

# Execute the command
result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

# Print the output from Fast Downward
print(result.stdout)
