"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

DATA_FILE_URL = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
DATA_FILE_NAME = "alpaca_data_cleaned_archive.json"
DESTINATION_PATH = Path("data/alpaca")
CHECKPOINT_DIR = Path("checkpoints/stabilityai/stablelm-base-alpha-3b")
TEST_SPLIT_FRACTION = 0.03865    # to get exactly 2000 test samples
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
SEED = 42


def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    test_split_fraction: float = TEST_SPLIT_FRACTION,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    data_file_name: str = DATA_FILE_NAME,
    data_file_url: str = DATA_FILE_URL,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    print("Loading data file...")
    download_if_missing(data_file_path, data_file_url)
    with open(data_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        data, [1. - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
