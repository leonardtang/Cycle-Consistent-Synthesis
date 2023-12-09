import argparse
import re
import torch
from datasets import load_dataset
from pprint import pprint
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from docsynth import setup_docstring_prompt
from execution import check_correctness
from selection import selection
from utils import (
    construct_stopping_criteria,
    filter_code,
    format_indent,
    format_indent_docstring,
    format_prompt,
    setup_model_tokenizer,
    STOP_SEQS,
    Trimmer,
)

NEW_TOKENS = 128
# TODO: probably figure out way to batch generations
REPEAT = 10
BATCH_SIZE = 10
GEN_TEMP = 0.8
TIMEOUT = 30
SELECT_CRITERIA = "logprob"
# Program --> Documentation examples to set up documentation generation
FEW_SHOT = 0
SIM_MATCH = "sentence-transformer"
DEBUG = False

rankers = {
    "codellama": "codellama/CodeLlama-7b-hf",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    # Salesforce/codegen-350M-mono fine-tuned on codeparrot/github-code-clean (Python only)
    "kdf": "kdf/python-docstring-generation",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# TODO: might want to avoid the example usages, such as ">>>"
def extract_docstring(text):
    triple_quoted_strings = re.findall(
        r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", text, re.DOTALL
    )
    extracted_strings = [
        # f'"""{match[0]}"""' if match[0] else f'"""{match[1]}"""'
        f"{match[0]}" if match[0] else f"{match[1]}"
        for match in triple_quoted_strings
    ]
    return extracted_strings[0]


def preprocess_prompt(example):
    example["docstring"] = format_indent_docstring(extract_docstring(example["prompt"]))
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gen-temp", type=float, default=GEN_TEMP)
    parser.add_argument("--select-crit", type=str, default=SELECT_CRITERIA)
    parser.add_argument("--few-shot", type=int, default=FEW_SHOT)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--sim-match", type=str, default=SIM_MATCH)
    args = parser.parse_args()

    print("ARGS", args)

    print("########## HYPERPARAMETERS ##########")
    print("NEW_TOKENS:", args.new_tokens)
    print("REPEAT:", args.repeat)
    print("BATCH_SIZE:", args.batch_size)
    print("GEN_TEMP:", args.gen_temp)
    print("TIMEOUT:", args.timeout)
    print("SELECT_CRITERIA:", args.select_crit)
    print("FEW_SHOT", args.few_shot)

    # Load HumanEval
    # Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem
    # features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
    dataset = load_dataset("openai_humaneval")
    dataset = dataset.map(preprocess_prompt)

    if args.few_shot:
        # Calculate split percentage roughly based on few-shot number
        split = args.few_shot / 164
        dataset = dataset["test"].train_test_split(test_size=split)
        dataset, few_shot_dataset = dataset["train"], dataset["test"]
        few_shot_dataloader = DataLoader(few_shot_dataset, batch_size=1, shuffle=False)
    else:
        dataset = dataset["test"]
        few_shot_dataloader = None

    # Generator setup
    # Benchmark: CodeGen2.0-7B-multi is 18.83; reproduce to 18.90
    gen_device = "cuda"
    gen_checkpoint = "Salesforce/codegen2-7B"
    gen_tokenizer, gen_model = setup_model_tokenizer(
        gen_checkpoint, bit_4=True, device=gen_device, bnb_config=bnb_config
    )

    # Ranker setup
    docsynth_device = "cuda"
    docsynth_checkpoint = rankers["mistral"]
    docsynth_tokenizer, docsynth_model = setup_model_tokenizer(
        docsynth_checkpoint, bit_4=True, device=docsynth_device, bnb_config=bnb_config
    )

    print("DOCSYNTH:", docsynth_checkpoint)
    print("GENERATOR:", gen_checkpoint)

    # Matcher setup
    sim_model = None
    if args.sim_match == "sentence-transformer":
        sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_correct = 0
    for i, data in enumerate(dataloader):
        
        print(f"\n\n\n!!!!!!!!!!!!! Evaluating Question {i} !!!!!!!!!!!!!")
        og_prompt = data["prompt"][0]
        prompt = og_prompt
        
        # StarCoder magic prefix
        if "starcoder" in gen_device:
            magic_starcoder_prefix = "<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n"
            prompt += magic_starcoder_prefix

        docstring = data["docstring"][0]
        print("##### Intended Docstring #####")
        print(docstring)

        # (==>) Forward-generate program.
        prompt_copies = [prompt for _ in range(args.repeat)]
        global_generated_code_list, global_docsynth_outputs = [], []
        
        for i in range(0, args.repeat, args.batch_size):
            cand_size = min(args.batch_size, args.repeat - i)
            gen_inputs = gen_tokenizer(prompt_copies, return_tensors="pt").to(
                gen_device
            )
            gen_outputs_dict = gen_model.generate(
                **gen_inputs,
                pad_token_id=gen_tokenizer.eos_token_id,
                max_new_tokens=args.new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=args.gen_temp,
                top_p=0.95,
                top_k=0,
                stopping_criteria=construct_stopping_criteria(
                    "code", STOP_SEQS, gen_tokenizer, gen_device
                ),
            )

            gen_outputs_raw = gen_outputs_dict.sequences[
                :, gen_inputs["input_ids"].shape[1] :
            ]
            gen_scores = gen_outputs_dict.scores
            gen_outputs = gen_outputs_raw.squeeze(dim=0)
            
            generated_code_list = gen_tokenizer.batch_decode(
                gen_outputs,
                skip_special_tokens=True,
                truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
            )

            generated_code_list = [format_indent(gc) for gc in generated_code_list]
            generated_code_list = [filter_code(gc) for gc in generated_code_list]
            global_generated_code_list.extend(generated_code_list)

            # (<==) Backward-generate docstring.
            # Prompting setup for docstring synthesizer
            if args.select_crit in ["cycle-match", "judge-docstring-docstring"]:
                # Attempt to recover doctring
                # TODO: think about better prompting or (e.g. some form of estimate P(y|x))
                # Not sure if we also want this to be sampled
                docsynth_inputs_list = [
                    setup_docstring_prompt(
                        gc, docsynth_checkpoint, docsynth_tokenizer, few_shot_dataloader
                    )
                    for gc in generated_code_list
                ]

                docsynth_inputs = docsynth_tokenizer(
                    docsynth_inputs_list, return_tensors="pt", padding=True
                ).to(docsynth_device)
                # TODO: better stopping criteria; also figure out what to do with empty string
                docsynth_outputs_dict = docsynth_model.generate(
                    **docsynth_inputs,
                    pad_token_id=docsynth_tokenizer.eos_token_id,
                    max_new_tokens=args.new_tokens,
                    return_dict_in_generate=True,
                    do_sample=False,
                )
                docsynth_outputs = docsynth_outputs_dict.sequences[
                    :, docsynth_inputs["input_ids"].shape[1] :
                ]
                docsynth_outputs = docsynth_outputs.squeeze(dim=0)
                docsynth_outputs = docsynth_tokenizer.batch_decode(
                    docsynth_outputs, skip_special_tokens=True
                )

                global_docsynth_outputs.extend(docsynth_outputs)

        if global_docsynth_outputs:
            print("##### Global Docstring Example #####")
            print(global_docsynth_outputs[0])

        # Rank + choose final solution
        if args.select_crit != "random":
            final_program, ranked_docstrings, ranked_programs, scores = selection(
                args.select_crit,
                global_generated_code_list,
                docstring,
                global_docsynth_outputs,
                gen_model,
                gen_outputs_raw,
                gen_scores,
                sim_model,
                docsynth_model,
                docsynth_tokenizer,
            )
        else:
            # Rank + choose final solution
            final_program = selection(
                args.select_crit,
                global_generated_code_list,
                docstring,
                global_docsynth_outputs,
                gen_model,
                gen_outputs_raw,
                gen_scores,
                sim_model,
                docsynth_model,
                docsynth_tokenizer,
            )

        if DEBUG:
            if any(
                [
                    check_correctness(data, p, args.timeout)["passed"]
                    for p in ranked_programs
                ]
            ):
                for i, p in enumerate(ranked_programs):
                    print(f"*** Program has Score {scores[i]}")
                    # For judge-docstring-docstring, cycle-match
                    if ranked_docstrings:
                        print("*** Docstring: ***")
                        print(ranked_docstrings[i])
                    print("*** Program ***")
                    print(p)
                    is_correct = check_correctness(data, p, args.timeout)
                    pprint(is_correct)

        if check_correctness(data, final_program, args.timeout)["passed"]:
            print("Correct!")
            total_correct += 1

    total_correct /= len(dataloader)
    print(f"Total Pass: {total_correct}")