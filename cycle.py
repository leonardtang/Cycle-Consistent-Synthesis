import re
import torch
from datasets import load_dataset
from pprint import pprint
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from utils import setup_model_tokenizer

# Invariants
BATCH_SIZE = 32
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the HumanEval dataset
# Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem
# features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
dataset = load_dataset("openai_humaneval")
print("Dataset:", dataset)
print("Sample problem:", dataset["test"][0])  # Print the first problem as a sample


def extract_docstring(text):
    triple_quoted_strings = re.findall(
        r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", text, re.DOTALL
    )
    extracted_strings = [
        f'"""{match[0]}"""' if match[0] else f'"""{match[1]}"""' for match in triple_quoted_strings
    ]
    return extracted_strings


def preprocess_prompt(example):
    example["docstring"] = extract_docstring(example["prompt"])
    return example

dataset = dataset.map(preprocess_prompt)

# Generator setup
gen_checkpoint = "bigcode/starcoder"
gen_device = "cuda"
gen_tokenizer, gen_model = setup_model_tokenizer(
    gen_checkpoint, bit_4=True, device=gen_device, bnb_config=bnb_config
)

# Ranker setup
rank_checkpoint = "kdf/python-docstring-generation"
rank_device = "cuda"
rank_tokenizer, rank_model = setup_model_tokenizer(rank_checkpoint).to(rank_device)

# # Main loop
# data_collator = DataCollatorForLanguageModeling(tokenizer=gen_tokenizer, mlm=False)
# train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

# HACK: Dataset is so small that let's just loop through it one-by-one for now LOL
for i, data in enumerate(dataset):
    prompt = data['prompt']
    gen_inputs = gen_tokenizer.encode(prompt, return_tensors="pt").to(gen_device)
    gen_outputs = gen_model.generate(**gen_inputs)
    print('Gen Outputs', gen_outputs)
    
    rank_inputs = rank_tokenizer.encode(gen_outputs, return_tensors="pt").to(rank_device)

# doc_max_length = 128

# generated_ids = rank_model.generate(
#     **inputs,
#     max_length=inputs.input_ids.shape[1] + doc_max_length,
#     do_sample=False,
#     return_dict_in_generate=True,
#     num_return_sequences=1,
#     output_scores=True,
#     pad_token_id=50256,
#     eos_token_id=50256  # <|endoftext|>
# )
