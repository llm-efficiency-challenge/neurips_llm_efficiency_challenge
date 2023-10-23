from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
import torch

torch.set_float32_matmul_precision("high")

from lit_gpt import GPT, Tokenizer, Config
from lit_gpt.utils import lazy_load, quantization

# Toy submission imports
from helper import toysubmission_generate
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

quantize = "bnb.nf4-dq"  # 4-bit NormalFloat with Double-Quantization (see QLoRA paper)
checkpoint_dir = Path("checkpoints/openlm-research/open_llama_3b")
precision = "bf16-true"  # weights and data in bfloat16 precision

fabric = L.Fabric(devices=1, accelerator="cuda", precision=precision)

with open(checkpoint_dir / "lit_config.json") as fp:
    config = Config(**json.load(fp))

checkpoint_path = checkpoint_dir / "lit_model.pth"
logger.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
with fabric.init_module(empty_init=True), quantization(quantize):
    model = GPT(config)

with lazy_load(checkpoint_path) as checkpoint:
    model.load_state_dict(checkpoint, strict=quantize is None)

model.eval()
model = fabric.setup(model)

tokenizer = Tokenizer(checkpoint_dir)


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        L.seed_everything(input_data.seed)
    logger.info("Using device: {}".format(fabric.device))
    encoded = tokenizer.encode(
        input_data.prompt, bos=True, eos=False, device=fabric.device
    )
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)


    t0 = time.perf_counter()
    tokens, logprobs, top_logprobs = toysubmission_generate(
        model,
        encoded,
        max_returned_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
    )

    t = time.perf_counter() - t0

    if input_data.echo_prompt is False:
        output = tokenizer.decode(tokens[prompt_length:])
        tokens = tokens[prompt_length:]
        logprobs = logprobs[prompt_length:]
        top_logprobs = top_logprobs[prompt_length:]
    else:
        output = tokenizer.decode(tokens)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.processor.decode([idx])
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprobs_sum = sum(logprobs)
    # Process the input data here
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    logger.info("Using device: {}".format(fabric.device))
    t0 = time.perf_counter()
    encoded = tokenizer.encode(
        input_data.text, bos=True, eos=False, device=fabric.device
    )
    t = time.perf_counter() - t0
    tokens = encoded.tolist()
    return TokenizeResponse(tokens=tokens, request_time=t)


@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    logger.info("Using device: {}".format(fabric.device))
    t0 = time.perf_counter()
    # decoded = tokenizer.decode(torch.Tensor(input_data.tokens))
    decoded = tokenizer.processor.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)