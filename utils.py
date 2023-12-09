import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List

# The if looks questionabe
# STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
STOP_SEQS = ["\nclass", "\ndef", "\nif", "\nprint"]


# TODO: figure out how to stop only after second time witness of stop_token_ids
class CodingStop(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, device) -> None:
        StoppingCriteria.__init__(self)
        stop_token_ids = [
            tokenizer(x, return_tensors="pt")["input_ids"].squeeze() for x in stop_words
        ]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


# Also need to check tab sizing??
def construct_stopping_criteria(type, stop_words, tokenizer, device):
    if type == "code":
        return StoppingCriteriaList([CodingStop(stop_words, tokenizer, device)])
    elif type == "doc":
        pass


def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


class Trimmer:
    def __init__(self, stop_words, tokenizer, device) -> None:
        stop_token_ids = [
            tokenizer(x, return_tensors="pt")["input_ids"].squeeze() for x in stop_words
        ]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        self.stop_token_ids = stop_token_ids
        self.max_stop_len = max([len(stop_ids) for stop_ids in stop_token_ids])

    def trim_generation(
        self,
        generated_ids: List,
    ):
        """
        Removes the stopping sequence suffixes
        """
        for stop_ids in self.stop_token_ids:
            if torch.eq(generated_ids[-len(stop_ids) :], stop_ids).all():
                return generated_ids[: -len(stop_ids)]
        return generated_ids


def format_indent(completion: str): 
    return completion.replace("#     ", "    # ").replace('#    ', "    # ")

def format_indent_docstring(docstring: str): 
    return docstring.replace("    ", "")

def setup_tokenizer(path):
    # revision="main" if CodeGen2
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True
    )
    if "starcoder" in path:
        pass
    elif "kdf/python-docstring" in path:
        pass

    # For open ended generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def setup_model_tokenizer(
    path,
    device=None,
    bit_4=False,
    bit_8=False,
    max_memory=None,
    bnb_config=None,
):
    tokenizer = setup_tokenizer(path)
    if torch.cuda.device_count() > 1:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            device_map="auto",
            load_in_4bit=bit_4,
            load_in_8bit=bit_8,
            max_memory=max_memory,
            quantization_config=bnb_config,
        ).eval()
    else:
        if not bit_4 and not bit_8:
            model = (
                AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
                .to(device)
                .eval()
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                load_in_4bit=bit_4,
                load_in_8bit=bit_8,
                quantization_config=bnb_config,
            ).eval()
    return tokenizer, model


def format_prompt(prompt, model_name):
    if "TinyLlama" in model_name:
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    elif "Mistral-7B-Instruct" in model_name or "on-policy-mistral" in model_name:
        formatted_prompt = [
            {"role": "user", "content": prompt},
        ]
    elif "Llama-2-7b-chat-hf" in model_name:
        judge_system_prompt = """
        You are a helpful documentation assistant that looks at a piece of code and provides an English description of what the code does.
        You should be concise and precise with your descriptions.
        """
        formatted_prompt = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        formatted_prompt = prompt

    return formatted_prompt


def autoreg_generate(
    text,
    model,
    tokenizer,
    sample=False,
    stopping_criteria=None,
    num_beams=3,
    max_new_tokens=512,
    return_prompt=False,
    device="cuda",
    return_raw_ids=False,
    input_tokens=None,
    temp=None,
    top_p=None,
):
    # Don't think any of the models we're looking at have max_lengths to truncate too
    if not input_tokens:
        input_tokens = tokenizer(
            text, return_tensors="pt", padding=True
        ).to(device)

    logits, outputs = None, None
    with torch.no_grad():
        output_dict = model.generate(
            **input_tokens,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=sample,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temp,
            top_p=top_p
        )
        logits = output_dict.scores
        if not return_prompt:
            out_seq = output_dict.sequences[:, input_tokens["input_ids"].shape[1] :].detach().cpu()
        else:
            out_seq = output_dict.sequences.detach().cpu()

        if return_raw_ids:
            raw_ids = [os for os in out_seq]
            return raw_ids, logits

        outputs = tokenizer.batch_decode(
            out_seq,
            skip_special_tokens=True,
        )

    return outputs, logits