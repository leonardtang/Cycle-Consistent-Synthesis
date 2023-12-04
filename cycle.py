import numpy as np
import random
import re
import torch
from datasets import load_dataset
from numpy import dot
from numpy.linalg import norm
from pprint import pprint
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from execution import check_correctness
from utils import (
    construct_stopping_criteria,
    filter_code,
    format_indent,
    setup_model_tokenizer,
    STOP_SEQS,
    Trimmer,
)

# Invariants
BATCH_SIZE = 32
NEW_TOKENS = 128
REPEAT = 10
TEMP = 0.8
TIMEOUT = 30
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load HumanEval
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


def selection(scheme, code_choices, docstring, recovered):
    if scheme == "random":
        return random.choice(code_choices)
    elif scheme == "cycle-match":
        # Choose the program that yields the most faithful docstring recovery
        og_embed = sim_model.encode(docstring)
        recov_embeds = sim_model.encode(recovered)
        print('norm(og_embed)', norm(og_embed))
        sims = [dot(og_embed, b) / (norm(og_embed) * norm(b)) for b in recov_embeds]
        best_answer = np.argmax(sims)
        print('sims', sims)
        print('best_answer', best_answer)
        return code_choices[best_answer]
    else:
        raise Exception(f"Scheme ({scheme}) not implemented")


dataset = dataset.map(preprocess_prompt)

# Generator setup
# Benchmark: CodeGen2.0-7B-multi is 18.83; reproduce to 18.90
# gen_checkpoint = "bigcode/starcoder"
gen_checkpoint = "Salesforce/codegen2-7B"
gen_device = "cuda"
gen_tokenizer, gen_model = setup_model_tokenizer(
    gen_checkpoint, bit_4=True, device=gen_device, bnb_config=bnb_config
)

# Ranker setup
# Salesforce/codegen-350M-mono fine-tuned on codeparrot/github-code-clean (Python only)
rank_checkpoint = "kdf/python-docstring-generation"
rank_device = "cuda"
rank_tokenizer, rank_model = setup_model_tokenizer(
    rank_checkpoint, bit_4=True, device=gen_device, bnb_config=bnb_config
)

# Matcher setup
sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

dataloader = DataLoader(dataset["test"], batch_size=1, shuffle=False)
total_correct = 0
for i, data in enumerate(dataloader):
    og_prompt = data["prompt"][0]
    prompt = og_prompt
    # StarCoder magic
    if "starcoder" in gen_device:
        magic_starcoder_prefix = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n"
        prompt += magic_starcoder_prefix

    print("##### Prompt #####")
    print(prompt)
    docstring = data["docstring"][0]
    # Forward-generate program.
    # Definitely want this to be sampled
    prompt_copies = [prompt for _ in range(REPEAT)]

    # Batch HERE
    gen_inputs = gen_tokenizer(prompt_copies, return_tensors="pt").to(gen_device)
    gen_outputs_dict = gen_model.generate(
        **gen_inputs,
        pad_token_id=gen_tokenizer.eos_token_id,
        max_new_tokens=NEW_TOKENS,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=TEMP,
        top_p=0.95,
        top_k=0,
        # stopping_criteria=construct_stopping_criteria(
        #     "code", STOP_SEQS, gen_tokenizer, gen_device
        # ),
    )

    gen_outputs = gen_outputs_dict.sequences[:, gen_inputs["input_ids"].shape[1] :]
    gen_outputs = gen_outputs.squeeze(dim=0)
    # gen_outputs = trimmer.trim_generation(gen_outputs)

    # Attempt to recover doctring
    # TODO: think about better prompting or (e.g. some form of estimate P(y|x))
    # Not sure if we also want this to be sampled
    generated_code_list = gen_tokenizer.batch_decode(
        gen_outputs,
        skip_special_tokens=True,
        truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
    )

    generated_code_list = [format_indent(gc) for gc in generated_code_list]
    generated_code_list = [filter_code(gc) for gc in generated_code_list]
    # correct_list = [check_correctness(data, gc, TIMEOUT)["passed"] for gc in generated_code_list]

    # Prompting setup for docstring synthesizer
    rank_inputs_list = [gc + f"\n\n#docstring" for gc in generated_code_list]
    # print("##### Formatted rank inputs #####")
    # print(rank_inputs)
    rank_inputs = rank_tokenizer(
        rank_inputs_list, return_tensors="pt", padding="max_length", truncation=True
    ).to(rank_device)
    # TODO: better stopping criteria; also figure out what to do with empty string
    rank_outputs = rank_model.generate(
        **rank_inputs,
        pad_token_id=rank_tokenizer.eos_token_id,
        max_new_tokens=NEW_TOKENS,
    )[0]
    rank_outputs = rank_tokenizer.decode(rank_outputs, skip_special_tokens=True)
    print("##### Rank outputs #####", rank_outputs)

    # Rank + choose final solution
    final_program = selection(
        "cycle-match", generated_code_list, docstring, rank_outputs
    )
    if check_correctness(data, final_program, TIMEOUT)["passed"]:
        total_correct += 1


total_correct /= len(dataloader)
print(f"Total Pass: {total_correct}")
