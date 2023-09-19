import os

from huggingface_hub import login, HfApi 
from llama_recipes.finetuning import main as finetuning

def main():
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    
    kwargs = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "use_peft": True,
        "peft_method": "lora",
        "quantization": True,
        "batch_size_training": 2,
        "dataset": "custom_dataset",
        "custom_dataset.file": "./custom_dataset.py",
        "output_dir": "./output_dir ",
    }
    
    finetuning(**kwargs)

    api = HfApi() 

    api.upload_folder( 
        folder_path='./output_dir/', 
        repo_id=os.environ["HUGGINGFACE_REPO"], 
        repo_type='model', 
    )

if __name__ == "__main__":
    main()