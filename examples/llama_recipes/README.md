# Llama-recipes example
This example demonstrates how to fine-tune and serve a Llama 2 model with llama-recipes for submission in the LLM efficiency challenge using the [toy-submission](../../toy-submission/) as a template.
Llama-recipes provides an easy way to fine-tune a Llama 2 model with custom datasets using efficient techniques like LoRA or Llama-adapters.

# Getting started
In order to use llama-recipes we need to install the following pip package:
```
pip install llama-recipes
```

To obtain access to the model weights you need to fill out this [form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) to accept the license terms and acceptable use policy.

After access has been granted, you need to acknowledge this in your HuggingFace account for the model you want to fine-tune. In this example we will continue with the 7B parameter version available under this identifier: meta-llama/Llama-2-7b-hf
In this example the authentication will be done through a token created in the settings of your HuggingFace account.
When running the Docker, the token will be expected in an environment variable that can be created with:

```bash
export  HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```
Make sure that the token allows read access to meta-llama/Llama-2-7b-hf.

# Fine-tune the model
Fine-tuning the model on one of the preconfigured datasets can than be done with a single line command line:
```bash
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-hf --output_dir peft_model_output
```

To use load a custom dataset we need to implement a function (get_custom_datset) following [this](https://github.com/facebookresearch/llama-recipes/blob/main/examples/custom_dataset.py) example.
We can than train on this dataset using this command line:
```bash
python3 -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b --dataset custom_dataset --custom_dataset.file /workspace/custom_dataset.py --output_dir /volume/output_dir
```

# Create submission
The creation of a submission will be split in two Dockerfiles. The training Docker will perform a fine tuning with the LoRA method and load the base Llama weight before training. The resulting LoRA weights are then saved on a volume that mounts a folder from the host file system. The inference Docker will copy the LoRA weights into the Docker to create a self-sufficient Docker which only depends on the access token to download the Llama 2 base weights from the HuggingFace hub.

To build and run the taining Docker we need to execute:
```bash
docker build -f ./Dockerfile.train -t llama_recipes_train .

export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"
docker run --gpus all --rm -ti -v ./:/volume -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN  llama_recipes_train
```

The inferenc Docker is created and started with:
```bash
docker build -f ./Dockerfile.inference -t llama_recipes_inference .

docker run --gpus all -p 8080:80 --volume ./:/volume -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN llama_recipes_inference
```

To test the inference docker we can run this query:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "[INST]Was is the capital of france?[/INST] "}' http://localhost:8080/tokenize
```