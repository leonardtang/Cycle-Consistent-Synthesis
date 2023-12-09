import itertools

selection = ["cycle-match", "logprob", "random", "judge-program-docstring", "judge-docstring-docstring"]
temps = ['0.5']
reps = ['10']
few_shot = ["0", "3", "5"]
# com_synths = ["codellama", "llama", "kdf", "mistral"]
sim_match = ["sentence-transformer"]

all_params = [selection, few_shot, temps, reps] 

# Generate all combinations
for combination in itertools.product(*all_params):
    print(','.join(combination))
