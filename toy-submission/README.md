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