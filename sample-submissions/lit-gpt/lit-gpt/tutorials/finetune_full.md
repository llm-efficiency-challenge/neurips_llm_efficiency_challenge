# Finetuning the whole model

If you are interested in parameter-efficient finetuning, check out [finetune_adapter.md](finetune_adapter.md). In contrast to parameter-efficient finetuning, this "full" approach finetunes all model parameters, which is substantially more expensive. It may only be recommended as a baseline for comparison studies.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_stablelm.md).
3. Download the data and generate the Alpaca instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

or [prepare your own dataset](#tune-on-your-dataset).

## Running the finetuning

```bash
python finetune/full.py --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

Finetuning the falcon-7b model requires at least 8 GPUs with ~40 GB memory each.

You can speed up training by setting the `devices` variable in the script to utilize more GPUs if available.
Depending on the available GPU memory, you can also tune the `micro_batch_size` parameter to utilize the GPU efficiently.

This script will save checkpoints periodically to the `out_dir` directory. If you are finetuning different models or on your own dataset, you can specify an output directory with your preferred name:

```bash
python finetune/full.py --out_dir out/full/my-model-finetuned
```

If your GPU does not support `bfloat16`, you can pass the `--precision 32-true` argument.
For instance, to fine-tune on MPS (the GPU on modern Macs), you can run

```bash
python finetune/full.py --out_dir out/full/my-model-finetuned --precision 32-true
```

Note that `mps` as the accelerator will be picked up automatically by Fabric when running on a modern Mac.

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
python generate/full.py \
    --prompt "Recommend a movie to watch on the weekend." \
    --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

Output:
```
A good movie to watch on the weekend would be The Lion King, since it's a classic family film that everyone can enjoy...
```
If your GPU supports `bfloat16`, the script will automatically use it.

## Tune on your dataset

With only a few modifications, you can prepare and train on your own instruction dataset.

1. Create a json file in which each row holds one instruction-response pair. 
   A row has an entry for 'instruction', 'input', and 'output', where 'input' is optional an can be 
   the empty string if the instruction doesn't require a context. Below is an example json file:

    ```
    [
        {
            "instruction": "Arrange the given numbers in ascending order.",
            "input": "2, 4, 0, 8, 3",
            "output": "0, 2, 3, 4, 8"
        },
        ...
    ]
    ```

2. Make a copy of `scripts/prepare_alpaca.py` and name it what you want:

    ```bash
    cp scripts/prepare_alpaca.py scripts/prepare_mydata.py
    ```

3. Modify `scripts/prepare_mydata.py` to read the json data file.
4. Run the script to generate the preprocessed, tokenized train-val split:

    ```bash
    python scripts/prepare_mydata.py --destination_path data/mydata/
    ```

5. Run `finetune/full.py` by passing in the location of your data (and optionally other parameters):
   
    ```bash
    python finetune/full.py \
        --data_dir data/mydata/ \
        --checkpoint_dir checkpoints/tiiuae/falcon-7b \
        --out_dir data/mydata-finetuned
    ```


## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the finetune script (see https://github.com/Lightning-AI/lit-llama/issues/101).
