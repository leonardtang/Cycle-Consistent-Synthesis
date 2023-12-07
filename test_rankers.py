"""
Light-weight script to test the validity of various ranker models.
"""

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from cycle import setup_docstring_prompt, setup_model_tokenizer, rankers

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    rank_checkpoint = rankers["llama"]
    rank_device = "cuda"
    rank_tokenizer, rank_model = setup_model_tokenizer(
        rank_checkpoint, bit_4=True, device=rank_device, bnb_config=bnb_config
    )

    dataset = load_dataset("openai_humaneval")
    testset = dataset["test"]
    example = testset[0]["prompt"]
    print('Example Problem')
    print(example)

    rank_inputs_list = [setup_docstring_prompt(example, rank_checkpoint)]
    print("Ranker Prompt")
    print(rank_inputs_list[0])

    rank_inputs = rank_tokenizer(
        rank_inputs_list, return_tensors="pt", padding=True, truncation=True
    ).to(rank_device)
    # TODO: better stopping criteria; also figure out what to do with empty string
    rank_outputs_dict = rank_model.generate(
        **rank_inputs,
        pad_token_id=rank_tokenizer.eos_token_id,
        max_new_tokens=512,
        return_dict_in_generate=True,
        do_sample=False,
        # do_sample=True,
    )
    rank_outputs = rank_outputs_dict.sequences[:, rank_inputs["input_ids"].shape[1] :]
    rank_outputs = rank_outputs.squeeze(dim=0)
    rank_outputs = rank_tokenizer.decode(rank_outputs, skip_special_tokens=True)
    # TODO: might want this in batch_decode?
    # truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
    print("##### Rank Outputs Example #####")
    print(rank_outputs)