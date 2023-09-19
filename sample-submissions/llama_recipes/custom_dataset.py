# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools

from llama_recipes.datasets.utils import Concatenator


B_INST, E_INST = "", ""

def tokenize_dialog(dialog, tokenizer):
    prompt_tokens = [tokenizer(f"{B_INST} {(prompt['content']).strip()} {E_INST}") for prompt in dialog[::2]]
    answer_tokens = [tokenizer(f"{answer['content'].strip()} ") for answer in dialog[1::2]]
    answer_tokens = [{k:v[1:] for k,v in items.items()} for items in answer_tokens]
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    dialog_tokens = [dict(c, labels=len(c["input_ids"])*[-100,]if i % 2 == 0 else c["input_ids"]) for i,c in enumerate(dialog_tokens)]

    if len(dialog) % 2:
        dialog_tokens += [prompt_tokens[-1]]
        dialog_tokens[-1] = dict(dialog_tokens[-1], labels=[-100]*len(dialog_tokens[-1]["input_ids"]))

    combined_tokens = {}
    for k in dialog_tokens[0].keys():
        combined_tokens[k] = list(itertools.chain(*(t[k] for t in dialog_tokens)))
    return combined_tokens


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("OpenAssistant/oasst1", split=split)

    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset.features),)

    nodes = {}

    messages = {}
    root_ids = []

    for data in dataset:
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]

    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]

    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads

    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)

    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog}

    dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))
    dataset = dataset.map(Concatenator(add_labels=False), batched=True)

    return dataset
