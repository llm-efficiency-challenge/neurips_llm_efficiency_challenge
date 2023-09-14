# Leaderboard usage 


## How to use the leaderboard
The [Lightning AI](https://lightning.ai/) team has built us a leaderboard on Discord. This is the single best way you can make sure your submissions actually work before the submission, try to beat the unfinetuned toy submission as a starting point.

You might have noticed a new friendly bot has joined the server called @evalbot  to use it
1. DM the bot with `eval 4090` or `eval A100` and attach a zipped file of your submission to the message (You can also just openly message the bot but DM'ing will protect your secret sauce)
2. If successful the bot will give you a job ID and a running status, the eval will take roughly 1-2h so be patient if you're top of queue
3. Once the bot completes your run it will update either the ⁠leaderboard_4090  or ⁠leaderboard_a100 channel, we will not be monitoring these 2 text channels they will be purely for the bot to post the new updated leaderboard

## How to create a zip submission

We will showcase an example using our actual repo https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge
1. `git clone --recurse-submodules https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge` to ensure `lit-gpt` folder is actually in the repo
2. `rm -rf sample-submissions/llama_recipes`, the leaderboard will recursively traverse your repo and find the first `Dockerfile` and assume that's the submission
3. `zip -r neurips_llm_efficiency_challenge.zip neurips_llm_efficiency_challenge/`

And once you have that submission DM the `evalbot` with either `eval 4090` or `eval A100` with the zip file attached to your submission. Discord does impose size limits on messages so make sure your artifacts aren't stored directly in the repo but that you `wget` from somewhere else.


**Note**: 
1. The way the bot works is it will recursively scan your repo for the first Dockerfile and use only that to eval against
Providing free GPUs is expensive so if you're up to funny business like opening multiple discord accounts and/or spamming our bot we will disqualify you from the competition
2. You will be allowed a maximum of 3 submissions a day
3. Depending on volume of submissions eval might take a long time while you wait in the queue, the 2 techniques we have of resolving this are either adding more GPUs in our pool or reducing the number of eval instances, we will communicate whenever we make either of 2 decisions on Discord directly
