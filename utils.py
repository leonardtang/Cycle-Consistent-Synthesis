import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


# TODO: figure out how to stop only after second time witness of stop_token_ids
class CodingStop(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, device) -> None:
        StoppingCriteria.__init__(self)
        stop_token_ids = [
            tokenizer(x, return_tensors="pt")["input_ids"].squeeze() for x in stop_words
        ]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False

# Potential stopping criteria based on StarCoder observations
# if __name__ == "__main__":

# Also need to check tab sizing??
def construct_stopping_criteria(type, stop_words, tokenizer, device):
    if type == 'code':
        return StoppingCriteriaList([CodingStop(stop_words, tokenizer, device)])
    elif type == 'doc':
        pass


# Tests example:
# METADATA = {
#     'author': 'jt',
#     'dataset': 'test'
# }


# def check(candidate):
#     assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
#     assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
#     assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
#     assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
#     assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
#     assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
#     assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
def evaluate_solution(program, test):
    # TODO: you know what to do
    pass


def setup_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if "starcoder" in path:
        pass

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
