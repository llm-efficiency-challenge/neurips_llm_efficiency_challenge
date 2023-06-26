from typing import Union, List, Dict
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from dataclasses import dataclass, field


# Lit-llama imports
from generate import neurips_generate
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

torch.set_float32_matmul_precision("high")

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import lazy_load, llama_model_lookup, quantization

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

quantize = "llm.int8"

checkpoint_path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")

precision = "bf16-true"
fabric = L.Fabric(devices=1, precision=precision)

logger.info("Loading model ...")
t0 = time.time()
with lazy_load(checkpoint_path) as checkpoint:
    name = llama_model_lookup(checkpoint)

    with fabric.init_module(empty_init=True), quantization(mode=quantize):
        model = LLaMA.from_name(name)

    model.load_state_dict(checkpoint)
logger.info(f"Time to load model: {time.time() - t0:.02f} seconds.")

model.eval()
model = fabric.setup(model)

tokenizer = Tokenizer(tokenizer_path)


class ProcessRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    max_new_tokens: int = 50
    top_k: int = 200
    temperature: float = 0.8
    seed: Optional[int] = None

@dataclass
class Token:
    text: str
    logprob: float
    top_logprob: Dict[str, float]

class ProcessResponse(BaseModel):
    text: str
    tokens: List[Token]
    logprob: float
    request_time: float


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        L.seed_everything(input_data.seed)
    logger.info("Using device: {}".format(fabric.device))
    encoded = tokenizer.encode(input_data.prompt, bos=True, eos=False, device=fabric.device)
    prompt_length = encoded.size(0)
    # for i in range(input_data.num_samples):
    t0 = time.perf_counter()
    tokens, logprobs, top_logprobs = neurips_generate(
        model,
        encoded,
        input_data.max_new_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
    )

    t = time.perf_counter() - t0

    model.reset_cache()
    output = tokenizer.decode(tokens)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec")

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    tokens = []
    for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
        print(t, lp, tlp)
        idx , val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        tokens.append(Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp))
    tokens = [Token(text=tokenizer.decode(t), logprob=lp, top_logprob=dict([tlp])) for t, lp, tlp in zip(tokens, logprobs, top_logprobs)]
    logprobs_sum = sum(logprobs)
    # Process the input data here
    return ProcessResponse(text=output, tokens=tokens, logprob=logprobs_sum, request_time=t)


class TokenizeRequest(BaseModel):
        text: str
        truncation: bool = True
        max_length: int = 2048


class TokenizeResponse(BaseModel):
    tokens: List[int]
    request_time: float


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    logger.info("Using device: {}".format(fabric.device))
    t0 = time.perf_counter()
    encoded = tokenizer.encode(input_data.text, bos=True, eos=False, device=fabric.device)
    t = time.perf_counter() - t0
    tokens = encoded.tolist()
    return TokenizeResponse(tokens=tokens, request_time=t)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
