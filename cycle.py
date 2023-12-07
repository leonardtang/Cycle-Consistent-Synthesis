import numpy as np
import random
import re
import torch
import torch.nn.functional as F
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
    autoreg_generate,
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
REPEAT = 100
BATCH = 32
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
    example["docstring"] = extract_docstring(example["prompt"])
    return example


def selection(scheme, code_choices, docstring, recovered, judge_model, judge_tokenizer):
    if scheme == "random":
        return random.choice(code_choices)
    elif scheme == "cycle-match":
        # Choose the program that yields the most faithful docstring recovery
        og_embed = sim_model.encode(docstring)
        recov_embeds = sim_model.encode(recovered)
        sims = [dot(og_embed, b) / (norm(og_embed) * norm(b)) for b in recov_embeds]
        # print("SiMS", sims)
        best_answer = np.argmax(sims, axis=0)
        # print("BEST INDX", np.amax(sims, axis=0))
        rank = np.argsort(sims)
        ranked_answers = np.array(code_choices)[rank]
        # print("##### RANKED ANSWERS #####")
        # pprint(ranked_answers)
        return (
            code_choices[best_answer],
            np.array(recovered)[rank],
            ranked_answers,
            np.array(sims)[rank],
        )
    # TODO: implement more ranking procedures to benchmark against
    # TODO: language model benchmarking procedure ("Is this {code} an instance of {docstring}")
    elif scheme == "judge":
        # TODO: test code models on the isInstance functionality
        judge_scores = []
        judge_prompts = [
            (
                f"Consider the following program: {code}\n\n"
                + f"Does this program achieve the following goal: {docstring}."
                + "Answer strictly with 'Yes' or 'No'. Do not say anything else."
                + "If the program shows any sign of not achieving the intended goal, or has any errors, you should bias your response towards 'No'."
                for code in code_choices
            )
        ]
        
        formatted_judge_prompts = [format_prompt(p, "mistralai/Mistral-7B-Instruct-v0.1") for p in judge_prompts]
        tokens = [
            judge_tokenizer.apply_chat_template(fjp, return_tensors="pt").to(
                judge_model.device
            )[0]
            for fjp in formatted_judge_prompts
        ]
        
        text = judge_tokenizer.batch_decode(tokens)
        _, judge_model_output_logits = autoreg_generate(
            text,
            judge_model,
            judge_tokenizer,
            max_new_tokens=10,
            sample=False,
            device=judge_model.device,
            return_raw_ids=True,
        )

        word_to_id = {
            "Yes": 5592,
            "yes": 5081,
            "No": 1770,
            "no": 708,
        }

        # For now, just check first token generated. Each logits is going to be V-long
        first_dist = F.softmax(judge_model_output_logits[0].detach().cpu(), dim=1)
        positive = [
            logits[word_to_id["Yes"]] + logits[word_to_id["yes"]] for logits in first_dist
        ]
        del judge_model_output_logits
        return positive

    else:
        raise Exception(f"Scheme ({scheme}) not implemented")


# TODO: maybe the goal is to elicit more of a "summary" or "goal" type of response, which more closely match the docstring.
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
            + "You should be concise and precise with your descriptions. Summarize at a high level the program's intent. Do not explain the low-level details, just tell me what the program is meant to do."
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
        print(f"\n\n\nEvaluating Question {i} !!!!!!!!!!!!!")
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
                do_sample=True,
                temperature=GEN_TEMP,
                top_p=0.95,
                top_k=0,
                # stopping_criteria=construct_stopping_criteria(
                #     "code", STOP_SEQS, gen_tokenizer, gen_device
                # ),
            )

            gen_outputs = gen_outputs_dict.sequences[
                :, gen_inputs["input_ids"].shape[1] :
            ]
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
            global_generated_code_list.extend(generated_code_list)

            # Backward-generate docstring.
            # Prompting setup for docstring synthesizer
            rank_inputs_list = [
                setup_docstring_prompt(gc, rank_checkpoint, rank_tokenizer)
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
        # print("##### Rank Outputs Example #####")
        # print(rank_outputs[0])

        # Rank + choose final solution
        final_program, ranked_docstrings, ranked_programs, scores = selection(
            SELECT_CRITERIA, global_generated_code_list, docstring, global_rank_outputs, None, None
        )

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

        # if check_correctness(data, final_program, TIMEOUT)["passed"]:
        #     print("Correct!")
        #     total_correct += 1

    total_correct /= len(dataloader)
    print(f"Total Pass: {total_correct}")
