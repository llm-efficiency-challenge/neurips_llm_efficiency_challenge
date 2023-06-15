# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /lit-llama/ /submission/

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y git
RUN pip install torch numpy scipy
RUN pip install -r requirements.txt

# Run app.py when the container launches
CMD ["python3", "generate.py", "--quantize","llm.int8"]
