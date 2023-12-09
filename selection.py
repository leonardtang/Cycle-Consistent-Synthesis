"""
Accurate code samples can be selected via heuristic ranking instead of fully evaluating each sample, 
the latter of which may not be possible or practical in deployment.
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm
from utils import format_prompt, autoreg_generate


def selection(
    scheme, code_choices, docstring, recovered, gen_model, gen_outputs, gen_scores, sim_model, judge_model, judge_tokenizer
):
    if scheme == "random":
        return random.choice(code_choices)
    # Choose the program that yields the most faithful docstring recovery
    elif scheme == "cycle-match":
        # TODO: check to see if we can speed up this encoding process..?
        og_embed = sim_model.encode(docstring)
        recov_embeds = sim_model.encode(recovered)
        sims = [dot(og_embed, b) / (norm(og_embed) * norm(b)) for b in recov_embeds]
        best_answer = np.argmax(sims, axis=0)
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
    # LLM isInstance() characteristic: "Is this {program} an instance of {docstring}?"
    elif "judge" in scheme:
        if scheme == "judge-program-docstring":
            judge_prompts = [
                f"Consider the following program: {code}\n\n"
                + f"Does this program achieve the following goal: {docstring}. "
                + "Answer strictly with 'Yes' or 'No'. Do not say anything else. "
                + "If the program shows any sign of not achieving the intended goal, or has any errors, you should bias your response towards 'No'."
                for code in code_choices
            ]
        elif scheme == "judge-docstring-docstring":
            judge_prompts = [
                f"Consider the following summarized inten: {rec}\n\n"
                + f"Does this summarized intent match the following goal intent: {docstring}. "
                + "Answer strictly with 'Yes' or 'No'. Do not say anything else. "
                + "If the summarized intent shows any sign of not matching the intended goal, you should bias your response towards 'No'."
                for rec in recovered
            ]
        else:
            raise Exception(f"Scheme ({scheme}) is not implemented.")

        formatted_judge_prompts = [
            format_prompt(p, "mistralai/Mistral-7B-Instruct-v0.1")
            for p in judge_prompts
        ]
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
            logits[word_to_id["Yes"]] + logits[word_to_id["yes"]]
            for logits in first_dist
        ]
        print("POSITIVE")
        print(positive)
        del judge_model_output_logits
        best_answer = np.argmax(positive, axis=0)

        # return code_choices[best_answer]

        rank = np.argsort(positive)
        return (
            code_choices[best_answer],
            np.array(recovered)[rank],
            np.array(code_choices)[rank],
            np.array(positive)[rank],
        )
    # Choose program with the maximum likelihood
    elif scheme == "logprob":
        # gen outputs is 10 x 128
        # gen scores is 128 x 10
        # print('gen outputs', gen_outputs.shape)
        # print('gen scores', gen_scores.shape)
        transition_scores = gen_model.compute_transition_scores(
            gen_outputs, gen_scores, normalize_logits=True
        ).detach().cpu()

        # for score in transition_scores[0]:
        #     print
        # This is 10 x 128
        # print("TRANSITION SCORES")
        # print(transition_scores.shape)

        # Same as sum of log probs, which is same as product of probs
        mean_log_probs = torch.sum(transition_scores, axis=1)
        best_answer = np.argmax(mean_log_probs, axis=0)
        rank = np.argsort(mean_log_probs)
        return (
            code_choices[best_answer],
            np.array(recovered)[rank],
            np.array(code_choices)[rank],
            np.array(mean_log_probs)[rank],
        )

    else:
        raise Exception(f"Scheme ({scheme}) not implemented")
