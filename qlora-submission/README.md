# QLoRA Submission Example

## Instructions
1. Pull all git submodules
2. Install the QLoRA requirements that can be found at `qlora-submission/qlora/requirements.txt`.
3. Finetune a Guanaco model on the OpenAssistant dataset using QLoRA by running `qlora-submission/qlora/scripts/finetune_guanaco_7b.sh` from `qlora-submission/qlora/`. The checkpoints of the model will be saved at `qlora-submission/qlora/output/guanaco-7b`.
4. Install the FastAPI requirements (see instructions in the [docker file](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/blob/8ad5d49e2c4a40e099225fe07f6a0216cded6d63/toy-submission/Dockerfile))
and start a server. For requirements: `pip install --no-cache-dir --upgrade -r fast_api_requirements.txt`. Then start the server with: `uvicorn main:app --host 0.0.0.0 --port 80`

5. Install the HELM requirements and run the HELM tasks. (see [comment here](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/pull/3#issuecomment-1612140732))



## Current State
Need to test end-to-end but have mostly finished writing the code.