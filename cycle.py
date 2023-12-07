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
    format_prompt,
    setup_model_tokenizer,
    STOP_SEQS,
    Trimmer,
)

# Invariants
BATCH_SIZE = 32
NEW_TOKENS = 128
# TODO: probably figure out way to batch generations
REPEAT = 10
GEN_TEMP = 0.8
TIMEOUT = 30
SELECT_CRITERIA = "cycle-match"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

rankers = {
    "codellama": "codellama/CodeLlama-7b-hf",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "kdf": "kdf/python-docstring-generation",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}


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
        sims = [dot(og_embed, b) / (norm(og_embed) * norm(b)) for b in recov_embeds]
        best_answer = np.argmax(sims, axis=0)
        rank = np.argsort(sims)
        ranked_answers = np.array(recovered)[rank]
        print("##### RANKED ANSWERS #####")
        pprint(ranked_answers)
        return code_choices[best_answer]
    # TODO: implement more ranking procedures to benchmark against
    # TODO: language model benchmarking procedure ("Is this {code} an instance of {docstring}")
    elif scheme == "judge":
        # TODO: test code models on the isInstance functionality
        pass
    else:
        raise Exception(f"Scheme ({scheme}) not implemented")


def setup_docstring_prompt(str, ranker, tokenizer):
    if "kdf" in ranker:
        return str + f"\n\n#docstring"
    elif "codellama" in ranker:
        return (
            str
            + "\n\nWrite an appropriate English docstring for the above program. Do not generate any code."
        )
    elif "meta-llama" in ranker:
        raw_prompt = (
            str + "\n\nWrite an appropriate English docstring for the above program."
        )
        formatted_prompt = format_prompt(raw_prompt, ranker)
        return tokenizer.decode(
            tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        )
    elif "mistralai" in ranker:
        raw_prompt = (
            "You are a helpful documentation assistant that looks at a piece of code and provides an English description of what the code does."
            + "You should be concise and precise with your descriptions."
            + "Write an appropriate English docstring for the following program:\n\n"
            + str
        )
        formatted_prompt = format_prompt(raw_prompt, ranker)
        return tokenizer.decode(
            tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        )


if __name__ == "__main__":
    # Load HumanEval
    # Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem
    # features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
    dataset = load_dataset("openai_humaneval")
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
    # rank_checkpoint = "kdf/python-docstring-generation"
    # rank_checkpoint = "codellama/CodeLlama-7b-hf"
    # rank_checkpoint = "codellama/CodeLlama-7b-hf"
    rank_checkpoint = rankers["mistral"]
    # TODO: try other models, like CodeLlama
    rank_device = "cuda"
    rank_tokenizer, rank_model = setup_model_tokenizer(
        rank_checkpoint, bit_4=True, device=rank_device, bnb_config=bnb_config
    )

    # Matcher setup
    sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    dataloader = DataLoader(dataset["test"], batch_size=1, shuffle=False)
    total_correct = 0
    for i, data in enumerate(dataloader):
        og_prompt = data["prompt"][0]
        prompt = og_prompt
        # TODO: move this into separate function
        # StarCoder magic
        if "starcoder" in gen_device:
            magic_starcoder_prefix = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n"
            prompt += magic_starcoder_prefix

        # TODO: check that docstring is actually reasonable
        docstring = data["docstring"][0]
        print("##### INTENDED DOCSTRING #####")
        print(docstring)

        # Forward-generate program.
        # Definitely want this to be sampled
        prompt_copies = [prompt for _ in range(REPEAT)]

        gen_inputs = gen_tokenizer(prompt_copies, return_tensors="pt").to(gen_device)
        gen_outputs_dict = gen_model.generate(
            **gen_inputs,
            pad_token_id=gen_tokenizer.eos_token_id,
            max_new_tokens=NEW_TOKENS,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=GEN_TEMP,
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
            # TODO: check this final line?
            truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
        )

        generated_code_list = [format_indent(gc) for gc in generated_code_list]
        generated_code_list = [filter_code(gc) for gc in generated_code_list]

        # Backward-generate docstring.
        # Prompting setup for docstring synthesizer
        rank_inputs_list = [
            setup_docstring_prompt(gc, rank_checkpoint, rank_tokenizer)
            for gc in generated_code_list
        ]

        rank_inputs = rank_tokenizer(
            rank_inputs_list, return_tensors="pt", padding=True, truncation=True
        ).to(rank_device)
        # TODO: better stopping criteria; also figure out what to do with empty string
        rank_outputs_dict = rank_model.generate(
            **rank_inputs,
            pad_token_id=rank_tokenizer.eos_token_id,
            max_new_tokens=NEW_TOKENS,
            return_dict_in_generate=True,
            do_sample=False,
        )
        rank_outputs = rank_outputs_dict.sequences[
            :, rank_inputs["input_ids"].shape[1] :
        ]
        rank_outputs = rank_outputs.squeeze(dim=0)
        rank_outputs = rank_tokenizer.batch_decode(
            rank_outputs, skip_special_tokens=True
        )

        # TODO: might want this in batch_decode?
        # truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
        print("##### Rank Outputs Example #####")
        print(rank_outputs[0])

        # Rank + choose final solution
        final_program = selection(
            SELECT_CRITERIA, generated_code_list, docstring, rank_outputs
        )
        if check_correctness(data, final_program, TIMEOUT)["passed"]:
            print("Correct!")
            total_correct += 1

    total_correct /= len(dataloader)
    print(f"Total Pass: {total_correct}")
