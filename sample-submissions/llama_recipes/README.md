# Llama-recipes example
This example demonstrates how to fine-tune and serve a Llama 2 model with llama-recipes for submission in the LLM efficiency challenge using the [lit-gpt](../lit-gpt/) example as a template.
Llama-recipes provides an easy way to fine-tune a Llama 2 model with custom datasets using efficient techniques like LoRA or Llama-adapters.

# Getting started
In order to use llama-recipes we need to install the following pip package:

```
pip install llama-recipes
```

To obtain access to the model weights you need to fill out this [form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) to accept the license terms and acceptable use policy.

After access has been granted, you need to acknowledge this in your HuggingFace account for the model you want to fine-tune. In this example we will continue with the 7B parameter version available under this identifier: meta-llama/Llama-2-7b-hf

**NOTE** In this example the training result will be uploaded and downloaded through huggingface_hub. The authentication will be done through a token created in the settings of your HuggingFace account.
Make sure to give write access to the token and set the env variables in the Dockerfiles to your token and repo:

```bash
ENV HUGGINGFACE_TOKEN="YOUR_TOKEN"
ENV HUGGINGFACE_REPO="YOUR_USERNAME/YOUR_REPO"
```

# Fine-tune the model
With llama-recipes its possible to fine-tune Llama on custom data with a single command. To fine-tune on a custom dataset we need to implement a function (get_custom_dataset) that provides the custom dataset following this example [custom_dataset.py](https://github.com/facebookresearch/llama-recipes/blob/main/examples/custom_dataset.py).
We can then train on this dataset using this command line:

```bash
python3 -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b --dataset custom_dataset --custom_dataset.file /workspace/custom_dataset.py --output_dir /volume/output_dir
```

**Note** The custom dataset in this example is dialog based. This is only due to the nature of the example but not a necessity of the custom dataset functionality. To see other examples of get_custom_dataset functions (btw the name of the function get_custom_dataset can be changed in the command line by using this syntax: /workspace/custom_dataset.py:get_foo_dataset) have a look at the [built-in dataset in llama-recipes](https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/datasets/__init__.py).

# Create submission
The creation of a submission will be split in two Dockerfiles. The training Docker will perform a fine tuning with the LoRA method and load the base Llama weight before training. The resulting LoRA weights are then uploaded to huggingface_hub. The inference Docker will download base and LoRA weights from huggingface_hub (make sure to set token and repo in the Dockerfiles).

To build and run the taining Docker we need to execute:

```bash
docker build -f ./Dockerfile.train -t llama_recipes_train .

docker run --gpus "device=0" --rm -ti llama_recipes_train
```

The inference Docker is created and started with:

```bash
docker build -f ./Dockerfile.inference -t llama_recipes_inference .

docker run --gpus "device=0" -p 8080:80 --rm -ti llama_recipes_inference
```

To test the inference docker we can run this query:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital of france? "}' http://localhost:8080/tokenize
OR
curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital of france? "}' http://localhost:8080/process
```