# Toy Submission
This toy-submission contains a dockerfile that exposes a HTTP server. Requests will be made against this server during the evaluation phase of the competition

### Getting Started
Make sure you have recursively clone the top this repository in order to get lit-llama. 


### Structure
* lit-llama/ 
    * unmodified submodule
* main.py
    * The process/ and tokenize/ endpoints are defined here
* helper.py
    * Applies logic on top of lit-llama's generate in order to produce responses in accordance with the spec.
* api.py
    * Defines the pydantic classes for the FASTapi server
### Make your GPUs visible to Docker 
```
nvidia-ctk runtime configure
systemctl restart docker
```

### Build and run 
```
docker build -t toy_submission .
docker run --gpus all -p 8080:80 toy_submission
```
### Send requests
`curl -X POST -H "Content-Type: application/json" -d '{"prompt": "The capital of france is "}' http://localhost:8080/process`