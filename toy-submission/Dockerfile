# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /lit-llama/ /submission/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-llama
RUN apt-get update && apt-get install -y git
RUN pip install -r requirements.txt

# get open-llama weights
RUN python scripts/download.py --repo_id openlm-research/open_llama_7b --local_dir checkpoints/open-llama/7B
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/open-llama/7B --model_size 7B

# Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
