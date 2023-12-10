from utils import format_prompt

# TODO: maybe the goal is to elicit more of a "summary" or "goal" type of response, which more closely match the docstring.
def setup_docstring_prompt(str, ranker, tokenizer, fs_loader=None):
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
            prompt
            + str
            + "\n\nWrite an appropriate English docstring for the above program."
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
        return tokenizer.decode(
            tokenizer.apply_chat_template(formatted_prompt, return_tensors="pt")[0]
        )
