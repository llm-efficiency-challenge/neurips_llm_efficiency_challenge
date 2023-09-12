#!/bin/bash

huggingface-cli login --token $HUGGINGFACE_TOKEN

python3 -m llama_recipes.finetuning  --model_name meta-llama/Llama-2-7b --use_peft --peft_method lora --quantization --batch_size_training 2 --dataset custom_dataset --custom_dataset.file /workspace/custom_dataset.py --output_dir /volume/output_dir