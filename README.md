# Neurips 1 LLM 1 GPU Challenge

This repository provides a starting point for those who are interested in the [NeurIPS 1 LLM 1 GPU Competition](https://llm-efficiency-challenge.github.io/). It provides detailed clarifications on what a submission looks like exactly, and how it will be evaluated and submitted.

At a high level, the key thing you will contribute is a `Dockerfile`, which will be a reproducible artifact that we can use to test your submission. The `Dockerfile` should contain all the code and dependencies needed to run your submission. We will use this `Dockerfile` to build a docker image and then run it against a set of tasks which will be a subset of the [HELM](https://crfm.stanford.edu/helm/latest/) tasks.

Your `Dockerfile` will expose a simple HTTP server, which needs to implement 2 endpoints `/process` and `/tokenize`. We will build that `Dockerfile` and expect it to launch an HTTP server. Once that server is launched, we will make requests to it via HELM and record your results.

At a high level the flow you should follow to ensure a strong submission:
1. Pick approved LLMs and datasets from [here](https://llm-efficiency-challenge.github.io/challenge)
2. Start with one of [sample-submissions](sample-submissions) and make sure it runs
3. Evaluate it locally on your own 40Gb A100 or 4090, if you don't have funding for either please see the [GPU funding](#gpu-funding) section for some more options
4. Once you have something working you can make a submission on our [Discord Leaderboard](https://discord.com/channels/1124130156336922665/1124134272631054447/1151718598818156645) to see how you fare up against other competitors
5. On the competition deadline make sure you have the final eval Dockerfile you'd like us to run in your github repo, refer to the [timeline](https://llm-efficiency-challenge.github.io/dates)
6. If your entry makes the shortlist, we will work with you to reproduce all of your artifacts with another finetuning Dockerfile

## Contents

- [Approved LLM & Dataset](#approved-llm-and-dataset)
- [Submission](#submission)
- [Evaluate Your Model Locally Using HELM](#evaluate-your-model-locally-using-helm)
- [Finetune](#finetune)
- [Create your own submission template](#create-your-own-submission-template)
- [Discord Leaderboard](#discord-leaderboard)
- [Final Leaderboard Submission](#final-eval-submission)
- [Evaluating the Final Submission](#evaluating-the-final-submission)
- [GPU funding](#gpu-funding)

## Approved LLM and dataset

The LLM space has complex licenses which can make it difficult to figure out what's permittable to use in a competition to streamline this process we've shortlisted a few models and datasets we know are safe to use [here](https://llm-efficiency-challenge.github.io/challenge)

That said the LLM space is fast moving so if you'd like to use a dataset or model that isn't on our list make sure to ask us about it on [https://discord.gg/XJwQ5ddMK7](https://discord.gg/XJwQ5ddMK7)

## Submission

The submission in this repository is a basic implementation of the setting up an HTTP server in accordance to the `open_api` spec. It includes a sample solution built off of [Lit-GPT](https://github.com/Lightning-AI/lit-gpt) and open-llama weights that participants can reference or modify as they see fit.

You can use the provided code as a reference or starting point for your own implementation. The `main.py` file contains the simple FastAPI server, and you can modify it to suit your needs.

You can find the Lit-GPT submission [here](sample-submissions/lit-gpt/) and the llama-recipes submission [here](sample-submissions/llama_recipes/) with instructions on how to run each locally.

Make sure that your final submission has only a single `Dockerfile` and that your weights are not directly included in the repo, they need to be downloaded during docker build or at runtime.

## Evaluate Your Model Locally Using HELM

Every submission will be tested against [HELM](https://crfm.stanford.edu/helm/latest/) which is a standard suite to evaluate LLMs on a broad set of datasets. This competition will leverage HELM for its evaluation infrastructure. The organizers will leverage standard STEM tasks from HELM although we will keep the exact set a secret and in addition we'll be including some heldout tasks that are presently not in HELM.

As you're working on your submission `Dockerfile` you'll want to test it out locally to make sure your contribution works as expected before you submit it.

HELM makes it easy to add new evaluation datasets by just adding another line in a config file so make sure to experiment with the different datasets they have available and feel free to contribute your own.

To learn more about how to test your submission with HELM, please follow the instructions [here](helm.md).

## Finetune

It's likely that an untuned base model won't give you satisfactory results, in that case you might find it helpful to do some additional finetuning. There are many frameworks to do this but we've created 2 sample submissions to do so
1. [lit-gpt](/sample-submissions/lit-gpt/)
2. [llama-recipes](/sample-submissions/llama_recipes/)


### Create Your Own Submission Template

Note that we've offered 2 sample submissions, our evaluation infrastructure is generic and only assumes an HTTP client so you can use a finetuning framework in Python like the ones we've suggested but also any non based Python framework you like using.

The `openapi.json` file in this repository contains the OpenAPI specification for the Competition API. Competitors can use this specification to understand the API endpoints, request and response structures, and overall requirements for interacting with the competition platform.

The OpenAPI specification provides a standardized way to describe the API, making it easier for competitors to develop their own solutions and integrate them seamlessly with the competition infrastructure.


## Discord Leaderboard

The [Lightning AI](https://lightning.ai/) has built a Discord based for us. You can find it on discord by its name `evalbot#4372`.

You can interact with it by DM'ing it with a zipped file of your sample submission and message it to either `eval A100` or `eval 4090`. More details on the bot are [here](https://discord.com/channels/1124130156336922665/1124134272631054447/1151718598818156645)

Once you make a submission the bot will inform you whether your submission failed or succeeded and after a few hours will publicly post your results. If you're at the top of the queue you can expect the eval to take 1-2h but depending on the size of the queue this could be longer. So please be mindful to not hurt other competitors trying to use the limited amount of hardware and ensure that your submissions work locally first.

Your submission will remain private to other competitors.

The end to end flow is described [here](leaderboard.md)

## Final Leaderboard Submission

When you registered for the competition you would have needed to create a github repo. When the submission deadline is reached make sure your Github repo has a `Dockerfile`, in case the location is ambiguous please sure to let us know in your `README.md`. The organizers will take your `Dockerfile` and run it as is and compute a baseline eval score. The purpose of this step is to primarily filter out broken submissions or submissions that can't outperform the unfinetuned sample submissions.

The deadline is on Oct 25 2023 with important dates listed [here](https://llm-efficiency-challenge.github.io/dates)

## Evaluating the Final Submission

Once the organizers have identified a shortlist of strong submissions, we will message you directly for another `Dockerfile` that would reproduce all of your artifacts. The best submission among this shortlist will win the competition and be invited to present their work at NeurIPS at our workshop.

## GPU funding

[AWS](https://aws.amazon.com/) has graciously agreed to provide $500 in AWS credits to 25 participating teams in the LLM efficiency competition. You will be able to pick and choose from available hardware to experiment before you make your final submission. To be eligible, please make sure to sign up at https://llm-efficiency-challenge.github.io/submission and write a short proposal in your `README.md` and add [@jisaacso](https://github.com/jisaacso) to your repos who will review your proposals.

We'll be prioritizing the first teams with serious proposals. Good luck!

There are some other free ways of getting GPUs that people have posted on discord [here](https://discord.com/channels/1124130156336922665/1149283885524463637/1149283885524463637) and you can shop around for both 4090 and A100 on cloud on [https://cloud-gpus.com/](https://cloud-gpus.com/)
