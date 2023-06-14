# Eval setup
* Build `docker build -t 111:latest .`
* Run `docker run -p 8000:8000 111:latest`

# TODO
* [ ] Eval creates a folder with all the relevant answers a model gave in `benchmark_model/model_name`. If we have more than one folder then `helm-server` can create a leaderboard automatically for us
* [ ] Host this `Dockerfile` somewhere, Github page is probably enough
