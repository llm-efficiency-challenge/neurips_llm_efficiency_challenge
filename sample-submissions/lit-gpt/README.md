# Toy Submission
This toy-submission contains a dockerfile that exposes a HTTP server. Requests will be made against this server during the evaluation phase of the competition

### Getting Started
Make sure you have recursively cloned the top this repository in order to get lit-gpt. 

❗ Make sure the repo is cloned with git submodule support either:

```sh
git clone --recurse-submodules ...
```

or if you cloned the repo but are missing the `lit-gpt` folder

```sh
git submodule update --init --recursive
```

### Structure
* lit-gpt/ 
    * unmodified submodule that contains a hackable `torch.nn.Module` GPT definition as well as optional fine-tuning
      and inference code.
* main.py
    * The process/ and tokenize/ endpoints are defined here
* helper.py
    * Applies logic on top of lit-gpt's generate in order to produce responses in accordance with the spec.
* api.py
    * Defines the pydantic classes for the FASTapi server
* Dockerfile
    * Definition of the image that will set-up the server used for submissions
  
### Make your GPUs visible to Docker 
Follow this guide to install [nvidia-ctk](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```sh
nvidia-ctk runtime configure
systemctl restart docker
```

### Build and run 
```sh
docker build -t toy_submission .
docker run --gpus all -p 8080:80 toy_submission
```
### Send requests
```sh
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "The capital of france is "}' http://localhost:8080/process
```
