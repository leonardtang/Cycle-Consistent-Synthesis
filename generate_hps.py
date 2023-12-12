import itertools

selection = ["logprob", "random", "judge-program-docstring"]
doc_strs_selection = ["fb-logprob", "judge-docstring-docstring", "cycle-match"]
temps = ['0.1', '0.2', '0.3', '0.5']
reps = ['10']
few_shot = ["0", "3", "5"]
com_synths = ["codellama", "llama", "kdf", "mistral"]
sim_match = ["sentence-transformer"]

all_nodocstr_params = [selection, ["0"], temps, reps] 
hp_combos = list(itertools.product(*all_nodocstr_params))
all_docstr_params = [doc_strs_selection, few_shot, temps, reps] 
hp_combos += list(itertools.product(*all_docstr_params))


# Generate all combinations
for combination in hp_combos:
    print(','.join(combination))
