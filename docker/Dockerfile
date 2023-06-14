# Starting from a Python 3.8 image
FROM python:3.8

# Updating apt-get and installing git
RUN apt-get update -y && apt-get install -y git

# Install HELM
# Install from source necessary because of some bugs with local data downloads
# Also necessary to install from source to add a new model
# We can download a user fork here
RUN git clone https://github.com/stanford-crfm/helm /helm
WORKDIR /helm
RUN pip install .

# Installing torch, torchvision, and torchaudio
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the task and model
# Add model to Huggingface like so https://github.com/stanford-crfm/helm/blob/main/docs/huggingface_models.md
RUN echo 'entries: [{description: "mmlu:subject=philosophy,model=huggingface/gpt2", priority: 1}]' > run_specs.conf

# Run helm
# Results will be in benchmark_output/v1
RUN helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10 && \
    helm-summarize --suite v1

# Start the web server
CMD ["helm-server"]
