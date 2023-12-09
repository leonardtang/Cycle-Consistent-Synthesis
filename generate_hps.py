import itertools

selection = ["logprob"]
few_shot = [0, 3, 5]
rankers = ["codellama", "llama", "kdf", "mistral"]

# Create a list of all arrays
all_params = [param1_values, param2_values] # Add all your arrays here

# Generate all combinations
for combination in itertools.product(*all_params):
    print(','.join(combination))
