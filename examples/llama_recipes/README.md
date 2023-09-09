# Llama-recipes example
This example demonstrates how to fine-tune and serve a Llama v2 model with llama-recipes for submission in the LLM efficiency challenge using the [toy-submission](../../toy-submission/) as a template.
Llama-recipes provides an easy way to fine-tune a Llama v2 with custom datasets using efficient techniques like LoRA or Llama-adapters.

# Getting started
In order to use llama-recipes we need to install the following pip package:
```
pip install llama-recipes
```

In order to obtain access to the model weights you need to fill out this [form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) to accept the license terms and acceptable use policy.

After access has been granted, you need to acknowledge this in your HuggingFace account for the model you want to fine-tune. In thsi example we will continue with the 7B parameter version available under this identifier: meta-llama/Llama-2-7b-hf

# Fine-tune the model
Fine-tuning the model on one of the preconfigured datasets can than be done with a single line command line:
```
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-hf --output_dir peft_model_output
```




# Create submission


docker build -f ./Dockerfile.train -t train .

export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"
docker run --gpus all --rm -ti -v ./:/workspace -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN  docker.io/library/train