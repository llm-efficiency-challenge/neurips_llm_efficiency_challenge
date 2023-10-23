from typing import List, Optional, Tuple

import torch


@torch.no_grad()
def toysubmission_generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[int], List[float], List[Tuple[int, float]]]:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.

    Returns:
        Tuple containing a list of token indexes, id of the top log probability, and the actual log probability of the
        selected token.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    # prefill empty with the prompt token indexes
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    top_logprob = []
    logprob = []

    # Generate log_prob and top_log_prob for the prompt
    logits = model(idx[:T].view(1, -1), input_pos)
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    prompt_log_probs = torch.log(probs)
    prompt_max_probs, prompt_argmax_probs = torch.max(probs, dim=-1)
    # Grab the logprob for all the tokens in the prompt
    logprob.extend(
        prompt_log_probs.gather(-1, idx[:T, None].to(torch.int64)).squeeze(-1).tolist()
    )
    top_logprob.extend(
        [
            (argmax.item(), max_prob.item())
            for argmax, max_prob in zip(prompt_argmax_probs, prompt_max_probs)
        ]
    )

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # append the logprob of selected token
        logprob.append(torch.log(probs[idx_next]).item())

        # append th idx and logprob of top token
        top_logprob.append((torch.argmax(probs).item(), torch.log(probs).max().item()))

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos], logprob, top_logprob  # include the EOS token

    return idx, logprob, top_logprob
