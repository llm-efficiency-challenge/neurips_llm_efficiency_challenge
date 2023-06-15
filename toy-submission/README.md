### Make your GPUs visible to Docker 
```
nvidia-ctk runtime configure
systemctl restart docker
```

### Build and run 
```
docker build -t toy_submission neurips_llm_efficiency_challenge/toy-submission/
docker run --gpus all toy_submission
```
