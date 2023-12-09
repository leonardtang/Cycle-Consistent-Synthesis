import numpy as np
import re
import torch
from datasets import load_dataset
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

# Invariants
NEW_TOKENS = 128
# TODO: probably figure out way to batch generations
REPEAT = 10
BATCH_SIZE = 10
GEN_TEMP = 0.8
TIMEOUT = 30
# SELECT_CRITERIA = "judge-docstring-docstring"
SELECT_CRITERIA = "logprob"
# SELECT_CRITERIA = "cycle-match"
# To be used in conjunction with the docstring generator
# FEW_SHOT = 3
FEW_SHOT = 0

rankers = {
    "codellama": "codellama/CodeLlama-7b-hf",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    # Salesforce/codegen-350M-mono fine-tuned on codeparrot/github-code-clean (Python only)
    "kdf": "kdf/python-docstring-generation",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}
gen_checkpoint = "Salesforce/codegen2-7B"
rank_checkpoint = rankers["mistral"]

print("########## HYPERPARAMETERS ##########")
print("NEW_TOKENS:", NEW_TOKENS)
print("REPEAT:", REPEAT)
print("BATCH_SIZE:", BATCH_SIZE)
print("GEN_TEMP:", GEN_TEMP)
print("TIMEOUT:", TIMEOUT)
print("SELECT_CRITERIA:", SELECT_CRITERIA)
print("FEW_SHOT", FEW_SHOT)
print("RANKER:", rank_checkpoint)
print("GENERATOR:", gen_checkpoint)

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


# TODO: maybe the goal is to elicit more of a "summary" or "goal" type of response, which more closely match the docstring.
def setup_docstring_prompt(str, ranker, tokenizer, fs_loader=None):

    # print("FS EXAMPLES", fs_examples)
    # System Prompts
    prompt = ""
    if "kdf" in ranker:
        pass
    elif "codellama" in ranker:
        pass
    elif "meta-llama" in ranker:
        pass
    elif "mistralai" in ranker:
        prompt += (
            "You are a documentation assistant that summarizes programs. "
            + "Be concise and precise with your descriptions. Summarize at a high level the program's intent. Do not explain the low-level details, just tell me what the program is meant to do. "
        )

    # Few-Shot
    fs_examples = []
    if fs_loader:
        for data in fs_loader:
            fs_examples.append((data["docstring"][0], data["canonical_solution"][0]))
    for ds, prog in fs_examples:
        if "kdf" in ranker:
            prompt += prog + f"\n\n#docstring" + f"\n{ds}\n\n"
        elif "codellama" in ranker:
            prompt += (
                prog
                + "\n\nWrite an appropriate English docstring for the above program. Do not generate any code. "
                + f"{prog}\n"
                + "Documentation:\n"
            )
        elif "meta-llama" in ranker:
            prompt += (
                prog
                + "\n\nWrite an appropriate English docstring for the above program. "
                + f"\n{ds}\n\n"
            )
        elif "mistralai" in ranker:
            prompt += (
                "Write documentation for the following program:\n\n"
                + f"{prog}\n"
                + "Documentation:\n"
                + f"{ds}\n\n"
            )

    # Final prompt
    if "kdf" in ranker:
        return prompt + str + f"\n\n#docstring"
    elif "codellama" in ranker:
        return (
            prompt
            + str
            + "\n\nWrite an appropriate English docstring for the above program. Do not generate any code."
        )
    elif "meta-llama" in ranker:
        raw_prompt = (
            prompt + str + "\n\nWrite an appropriate English docstring for the above program."
        )
        formatted_prompt = format_prompt(raw_prompt, ranker)
        return tokenizer.decode(
            tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        )
    elif "mistralai" in ranker:
        raw_prompt = (
            prompt
            + "Write documentation for the following program:\n\n"
            + f"{str}\n"
            + "Documentation:\n"
        )
        formatted_prompt = format_prompt(raw_prompt, ranker)
        # print("FINAL PROMPT?")
        # test = prompt + tokenizer.decode(
        #     tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        # )
        # print(tokenizer.decode(
        #     tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        # ))
        return tokenizer.decode(
            tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        )


if __name__ == "__main__":
    # Load HumanEval
    # Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem
    # features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
    dataset = load_dataset("openai_humaneval")
    dataset = dataset.map(preprocess_prompt)
    # Calculate split percentage roughly based on few-shot
    if FEW_SHOT:
        split = FEW_SHOT / 164
        dataset = dataset["test"].train_test_split(test_size=split)
        dataset, few_shot_dataset = dataset["train"], dataset["test"]
        few_shot_dataloader = DataLoader(few_shot_dataset, batch_size=1, shuffle=False)
    else:
        dataset = dataset["test"]
        few_shot_dataloader = None

    # Generator setup
    # Benchmark: CodeGen2.0-7B-multi is 18.83; reproduce to 18.90
    # gen_checkpoint = "bigcode/starcoder"
    gen_device = "cuda"
    gen_tokenizer, gen_model = setup_model_tokenizer(
        gen_checkpoint, bit_4=True, device=gen_device, bnb_config=bnb_config
    )

    # Ranker setup
    rank_device = "cuda"
    rank_tokenizer, rank_model = setup_model_tokenizer(
        rank_checkpoint, bit_4=True, device=rank_device, bnb_config=bnb_config
    )

    # Matcher setup
    sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_correct = 0
    for i, data in enumerate(dataloader):
        print(f"\n\n\n!!!!!!!!!!!!! Evaluating Question {i} !!!!!!!!!!!!!")
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

        # (==>) Forward-generate program.
        # Definitely want this to be sampled
        prompt_copies = [prompt for _ in range(REPEAT)]
        for i in range(0, REPEAT, BATCH_SIZE):
            cand_size = min(BATCH_SIZE, REPEAT - i)
            global_generated_code_list, global_rank_outputs = [], []

            gen_inputs = gen_tokenizer(prompt_copies, return_tensors="pt").to(
                gen_device
            )
            gen_outputs_dict = gen_model.generate(
                **gen_inputs,
                pad_token_id=gen_tokenizer.eos_token_id,
                max_new_tokens=NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=GEN_TEMP,
                top_p=0.95,
                top_k=0,
                # stopping_criteria=construct_stopping_criteria(
                #     "code", STOP_SEQS, gen_tokenizer, gen_device
                # ),
            )

            print('gen_outputs_raw pre', gen_outputs_dict.sequences.shape)
            gen_outputs_raw = gen_outputs_dict.sequences[
                :, gen_inputs["input_ids"].shape[1] :
            ]
            print('gen_outputs_raw post', gen_outputs_raw.shape)
            # gen_outputs_dict.scores[0].shape is 10 x 51200
            print('gen_outputs_dict.scores', len(gen_outputs_dict.scores))
            gen_scores = gen_outputs_dict.scores
            # print('stacked dims', gen_scores.shape)
            # gen_scores = gen_scores[
            #     gen_inputs["input_ids"].shape[1]:, :, :
            # ]
            # print('gen_scores post truncate', gen_scores.shape)
            # gen_scores = torch.unbind(gen_scores)
            gen_outputs = gen_outputs_raw.squeeze(dim=0)
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
            global_generated_code_list.extend(generated_code_list)

            # (<==) Backward-generate docstring.
            # Prompting setup for docstring synthesizer
            rank_inputs_list = [
                setup_docstring_prompt(
                    gc, rank_checkpoint, rank_tokenizer, few_shot_dataloader
                )
                for gc in generated_code_list
            ]

            rank_inputs = rank_tokenizer(
                rank_inputs_list, return_tensors="pt", padding=True
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

            global_rank_outputs.extend(rank_outputs)

        # TODO: might want this in batch_decode?
        # truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
        print("##### Docstring Example #####")
        print(rank_outputs[0])

        # Rank + choose final solution
        final_program, ranked_docstrings, ranked_programs, scores = selection(
            SELECT_CRITERIA,
            global_generated_code_list,
            docstring,
            global_rank_outputs,
            gen_model,
            gen_outputs_raw,
            gen_scores,
            sim_model,
            rank_model,
            rank_tokenizer,
        )
        # # Rank + choose final solution
        # final_program = selection(
        #     SELECT_CRITERIA, global_generated_code_list, docstring, global_rank_outputs, rank_model, rank_tokenizer
        # )

        # DEBUG TIME:
        if any(
            [check_correctness(data, p, TIMEOUT)["passed"] for p in ranked_programs]
        ):
            for i, p in enumerate(ranked_programs):
                print(f"PROGRAM has SCORE {scores[i]}")
                print("DOCSTRING: ", ranked_docstrings[i])
                print(p)
                is_correct = check_correctness(data, p, TIMEOUT)
                pprint(is_correct)

        if check_correctness(data, final_program, TIMEOUT)["passed"]:
            print("Correct!")
            total_correct += 1

    total_correct /= len(dataloader)
    print(f"Total Pass: {total_correct}")