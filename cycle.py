import re
import torch
from datasets import load_dataset
from pprint import pprint
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from utils import construct_stopping_criteria, setup_model_tokenizer

# Invariants
BATCH_SIZE = 32
NEW_TOKENS = 128
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


def extract_docstring(text):
    triple_quoted_strings = re.findall(
        r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", text, re.DOTALL
    )
    extracted_strings = [
        f'"""{match[0]}"""' if match[0] else f'"""{match[1]}"""'
        for match in triple_quoted_strings
    ]
    return extracted_strings[0]


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
rank_tokenizer, rank_model = setup_model_tokenizer(
    rank_checkpoint, bit_4=True, device=gen_device, bnb_config=bnb_config
)

dataloader = DataLoader(dataset["test"], batch_size=1, shuffle=False)
# HACK: Dataset is so small that let's just loop through it one-by-one for now LOL
# Fuck the Stanford people fr how dare they scoop my idea
for i, data in enumerate(dataloader):
    # print("data", data)
    og_prompt = data["prompt"][0]
    # StarCoder magic
    magic_prefix = (
        "<filename>solutions/solution_1.py\n"
        "# Here is the correct implementation of the code exercise\n"
    )
    prompt = magic_prefix + og_prompt
    print("##### Prompt #####")
    print(prompt)
    docstring = data["docstring"][0]
    # print("##### Docstring #####", docstring)
    # Forward-generate program.
    # Definitely want this to be sampled
    gen_inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_device)
    # print("gen_inputs shape", gen_inputs['input_ids'].shape)
    # TODO: think about temperature and top-p
    gen_outputs_dict = gen_model.generate(
        **gen_inputs,
        pad_token_id=gen_tokenizer.eos_token_id,
        max_new_tokens=NEW_TOKENS,
        return_dict_in_generate=True,
        stopping_criteria=construct_stopping_criteria(
            "code", [], gen_tokenizer, gen_device
        ),
    )

    # print('gen_inputs["input_ids"].shape[1]', gen_inputs["input_ids"].shape[1])
    # print('gen_outputs_dict.sequences', gen_outputs_dict.sequences)
    gen_outputs = gen_outputs_dict.sequences[
        :, gen_inputs["input_ids"].shape[1] :
    ].squeeze()

    # print("Gen Outputs Shape", gen_outputs.shape)

    # Attempt to recover doctring
    # TODO: think about better prompting or (e.g. some form of estimate P(y|x))
    # Not sure if we also want this to be sampled
    rank_inputs = gen_tokenizer.decode(gen_outputs, skip_special_tokens=True)
    # print("OG rank inputs", rank_inputs)
    # Based on HF page
    full_code = og_prompt + rank_inputs

    # TOOD: run full_code on the test set

    print('##### FULL CODE #####')
    print(full_code)

    rank_inputs += f"\n\n#docstring"
    print("##### Formatted rank inputs #####")
    print(rank_inputs)
    rank_inputs = rank_tokenizer(rank_inputs, return_tensors="pt").to(rank_device)
    # TODO: better stopping criteria; also figureo out what to do with empty string
    rank_outputs = rank_model.generate(
        **rank_inputs,
        pad_token_id=rank_tokenizer.eos_token_id,
        max_new_tokens=NEW_TOKENS,
    )[0]
    rank_outputs = rank_tokenizer.decode(rank_outputs, skip_special_tokens=True)
    print("##### Rank outputs #####", rank_outputs)

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
