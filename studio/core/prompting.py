"""SDXL prompt encoding with A1111-style weights and unlimited length.

Supports ``(word)``, ``(word:1.3)``, ``[word]``, escaped ``\\(`` and ``BREAK``.
Prompts longer than 75 tokens are split into chunks that are encoded
separately and concatenated, as in A1111/Forge.
"""
from __future__ import annotations

import re

import torch

_RE_ATTENTION = re.compile(
    r"""
\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:
""",
    re.X,
)
_RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)
CHUNK = 75


def parse_prompt_attention(text: str) -> list[list]:
    """Return ``[[text, weight], ...]``; a weight of -1 marks BREAK."""
    res: list[list] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []

    def multiply_range(start: int, mult: float) -> None:
        for p in range(start, len(res)):
            if res[p][1] != -1:
                res[p][1] *= mult

    for m in _RE_ATTENTION.finditer(text):
        token, weight = m.group(0), m.group(1)
        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), 1.1)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), 1 / 1.1)
        else:
            for i, part in enumerate(_RE_BREAK.split(token)):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, 1.1)
    for pos in square_brackets:
        multiply_range(pos, 1 / 1.1)

    res = [r for r in res if r[1] == -1 or r[0]]
    if not res:
        return [["", 1.0]]
    merged = [res[0]]
    for r in res[1:]:
        if r[1] == merged[-1][1] and r[1] != -1:
            merged[-1][0] += r[0]
        else:
            merged.append(r)
    return merged


def _chunk_tokens(tokenizer, segments) -> list[tuple[list[int], list[float]]]:
    chunks: list[tuple[list[int], list[float]]] = [([], [])]
    for text, weight in segments:
        if weight == -1:
            if chunks[-1][0]:
                chunks.append(([], []))
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
        for t in ids:
            if len(chunks[-1][0]) == CHUNK:
                chunks.append(([], []))
            chunks[-1][0].append(t)
            chunks[-1][1].append(weight)
    return chunks


def _pad_chunk(tokenizer, ids: list[int], weights: list[float]) -> tuple[list[int], list[float]]:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    n = CHUNK - len(ids)
    return (
        [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id] + [pad] * n,
        [1.0] + weights + [1.0] + [1.0] * n,
    )


def _encode_chunks(text_encoder, tokenizer, chunks, n_chunks, layer, device, want_pooled):
    chunks = chunks + [([], [])] * (n_chunks - len(chunks))
    embeds, pooled = [], None
    for ids, weights in chunks:
        ids, weights = _pad_chunk(tokenizer, ids, weights)
        out = text_encoder(torch.tensor([ids], device=device), output_hidden_states=True)
        if want_pooled and pooled is None:
            pooled = out[0]
        h = out.hidden_states[layer]
        if any(w != 1.0 for w in weights):
            w = torch.tensor(weights, dtype=h.dtype, device=h.device).view(1, -1, 1)
            original_mean = h.float().mean()
            h = h * w
            new_mean = h.float().mean()
            if new_mean.abs() > 1e-6:
                h = h * (original_mean / new_mean).to(h.dtype)
        embeds.append(h)
    return torch.cat(embeds, dim=1), pooled


@torch.no_grad()
def encode_prompts(pipe, prompt: str, negative: str, clip_skip: int = 2, device=None):
    """Return ``(prompt_embeds, negative_embeds, pooled, negative_pooled)`` for an SDXL pipeline.

    ``clip_skip`` follows the A1111 convention: 1 = last CLIP layer, 2 = penultimate
    (the SDXL default).
    """
    device = device or pipe._execution_device
    layer = -max(1, int(clip_skip))
    pos_segments = parse_prompt_attention(prompt or "")
    neg_segments = parse_prompt_attention(negative or "")

    pairs = [(pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)]
    tokenized = [(_chunk_tokens(tok, pos_segments), _chunk_tokens(tok, neg_segments)) for tok, _ in pairs]
    n_chunks = max(len(c) for pair in tokenized for c in pair)

    pos_parts, neg_parts = [], []
    pooled = neg_pooled = None
    for i, ((tok, te), (pos_chunks, neg_chunks)) in enumerate(zip(pairs, tokenized)):
        last = i == len(pairs) - 1
        p, pp = _encode_chunks(te, tok, pos_chunks, n_chunks, layer, device, last)
        n, nn = _encode_chunks(te, tok, neg_chunks, n_chunks, layer, device, last)
        pos_parts.append(p)
        neg_parts.append(n)
        if last:
            pooled, neg_pooled = pp, nn

    dtype = pipe.text_encoder_2.dtype
    return (
        torch.cat(pos_parts, dim=-1).to(device, dtype),
        torch.cat(neg_parts, dim=-1).to(device, dtype),
        pooled.to(device, dtype),
        neg_pooled.to(device, dtype),
    )
