# Neurips 1 LLM 1 GPU Challenge

This repository contains a toy submission for the [NeurIPS 1 LLM 1 GPU Competition](https://llm-efficiency-challenge.github.io/). It provides a simple implementation that serves as a starting point for participants to understand the problem and build their own solutions.

At a high level the key thing you will contribute is a `Dockerfile` which will be a reproducible artifact that we can use to test your submission. The `Dockerfile` should contain all the code and dependencies needed to run your submission. We will use this `Dockerfile` to build a docker image and then run it against a set of tasks. The tasks will be a subset of the [HELM](https://crfm.stanford.edu/helm/latest/) tasks. We will run your submission against the tasks and then evaluate the results.

Your `Dockerfile` will expose a simple HTTP server which needs to implement 2 endpoints `/process` and `/tokenize`. We will build that `Dockerfile` and expect it to launch an HTTP server. Once that server is launched, we will make requests to it via HELM and record your results.

## Contents

- [Submission](#submission)
- [HELM](#helm)

## Submission

The submission in this repository is a basic implementation of the setting up a HTTP server in accordance to the open_api spec. It includes a sample solution built off of [Lit-GPT](https://github.com/Lightning-AI/lit-gpt) and open-llama weights that participants can reference or modify as they see fit.

### Usage

You can use the provided code as a reference or starting point for your own implementation. The `main.py` file contains the simple FastAPI server, and you can modify it to suit your needs.

You can find the toy submission [here](/toy-submission)

### OpenAPI Specification

The `openapi.json` file in this repository contains the OpenAPI specification for the Competition API. Competitors can use this specification to understand the API endpoints, request and response structures, and overall requirements for interacting with the competition platform.

The OpenAPI specification provides a standardized way to describe the API, making it easier for competitors to develop their own solutions and integrate them seamlessly with the competition infrastructure.


## HELM

Every submission will be tested against [HELM](https://crfm.stanford.edu/helm/latest/) which is a standard suite to evaluate LLMs on a broad set of datasets. This competition will leverage HELM for its evaluation infrastructure. The organizers will leverage standard STEM tasks from HELM although we will keep the exact set a secret and in addition we'll be including some heldout tasks that are presently not in HELM.

As you're working on your submission `Dockerfile` you'll want to test it out locally to make sure your contribution works as epxected before you submit it

To learn more about how to test your submission with HELM, please follow instructions [here](helm.md)

