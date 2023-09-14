# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /lit-gpt/ /submission/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt
RUN apt-get update && apt-get install -y git
RUN pip install -r requirements.txt huggingface_hub sentencepiece

# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b

# Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
