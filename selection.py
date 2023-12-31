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
from docsynth import setup_docstring_prompt
from utils import format_prompt, autoreg_generate


def selection(
    args,
    scheme,
    code_choices,
    docstring,
    recovered,
    gen_model,
    gen_outputs,
    gen_scores,
    docsynth_checkpoint,
    docsynth_device,
    docsynth_model,
    docsynth_tokenizer,
    sim_model,
    judge_model,
    judge_tokenizer,
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
                f"Consider the following summarized intent: {rec}\n\n"
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
        if scheme == "judge-docstring-docstring":
            return (
                code_choices[best_answer],
                np.array(recovered)[rank],
                np.array(code_choices)[rank],
                np.array(positive)[rank],
            )
        else:
            return (
                code_choices[best_answer],
                None,
                np.array(code_choices)[rank],
                np.array(positive)[rank],
            )
    # Choose program with the maximum likelihood
    elif scheme == "logprob":
        transition_scores = (
            gen_model.compute_transition_scores(
                gen_outputs, gen_scores, normalize_logits=True
            )
            .detach()
            .cpu()
        )
        # Same as sum of log probs, which is same as product of probs
        mean_log_probs = torch.mean(transition_scores, axis=1)
        best_answer = np.argmax(mean_log_probs, axis=0)
        rank = np.argsort(mean_log_probs)
        return (
            code_choices[best_answer],
            None,
            np.array(code_choices)[rank],
            np.array(mean_log_probs)[rank],
        )
    elif scheme == "fb-logprob":
        forward_transition_scores = (
            gen_model.compute_transition_scores(
                gen_outputs, gen_scores, normalize_logits=True
            )
            .detach()
            .cpu()
        )
        # (==>)  P(C|D)
        mean_foward_logprobs = torch.mean(forward_transition_scores, axis=1)

        # (<==) P(D|C)
        # Specifically: C is generated, and D is the original fixed docstring

        docsynth_inputs_list = [
            setup_docstring_prompt(gc, docsynth_checkpoint, docsynth_tokenizer)
            for gc in code_choices
        ]
        # Calculate conditional docstring probability
        docsynth_inputs = docsynth_tokenizer(
            docsynth_inputs_list, return_tensors="pt", padding=True
        ).to(docsynth_device)
        docsynth_outputs_dict = docsynth_model.generate(
            **docsynth_inputs,
            pad_token_id=docsynth_tokenizer.eos_token_id,
            max_new_tokens=args.new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        docsynth_outputs = docsynth_outputs_dict.sequences[
            :, docsynth_inputs["input_ids"].shape[1] :
        ]

        backward_transition_scores = (
            docsynth_model.compute_transition_scores(
                docsynth_outputs, docsynth_outputs_dict.scores, normalize_logits=True
            )
            .detach()
            .cpu()
        )
        mean_backward_logprobs = torch.mean(backward_transition_scores, axis=1)

        mean_logprobs = mean_foward_logprobs + mean_backward_logprobs
        rank = np.argsort(mean_logprobs)
        best_answer = np.argmax(mean_logprobs, axis=0)

        return (
            code_choices[best_answer],
            None,
            np.array(code_choices)[rank],
            np.array(mean_logprobs)[rank],
        )

    else:
        raise Exception(f"Scheme ({scheme}) not implemented")
