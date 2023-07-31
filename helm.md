# How to test HELM locally

## Install the NeurIPS client

Install HELM from our branch: `pip install git+https://github.com/drisspg/helm.git@neruips_client`


## Setup an HTTP server

Follow instructions in [toy-submission](/toy-submission/) to setup a simple HTTP client that can use to local tests

## Configure HELM

You can configure which datasets to run HELM on by editing a `run_specs.conf`, to run your model on a large set of datasets, take a look at https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs_lite.conf for some inspiration

Here's how you can create a simple one for the purposes of making sure that your Dockerfile works

```bash
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
helm-summarize --suite v1
```

## Analyze your results

You can launch a web server to visually inspect the results of your run, `helm-summarize` can also print the results textually for you in your terminal but we've found the web server to be useful.

```
helm-server
```

This will launch a server on your local host, if you're working on a remote machine you might need to setup port forwarding. If everything worked correctly you should see a page that looks like [this](https://user-images.githubusercontent.com/3282513/249620854-080f4d77-c5fd-4ea4-afa4-cf6a9dceb8c9.png)
