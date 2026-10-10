# SPDX-License-Identifier: Apache-2.0
"""Dense functional Huginn oracle, independent of paged KV and model modules."""

import math

import torch
from torch.nn import functional as F


@torch.inference_mode()
def dense_huginn_reference(model, tokens, state):
    config, weights = model.config, model.state_dict()
    length = len(tokens)
    phase = torch.outer(
        torch.arange(length, device=tokens.device).float(),
        config.rope_base
        ** (-torch.arange(0, config.head_dim, 2, device=tokens.device).float() / config.head_dim),
    )[:, None]
    future = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)

    def norm(value, name):
        value32 = value.float()
        value32 = value32 / torch.sqrt(value32.square().mean(-1, keepdim=True) + config.norm_eps)
        return value32.to(value.dtype) * weights[name + ".weight"]

    def linear(value, name):
        return F.linear(value, weights[name + ".weight"])

    def rotate(value):
        pairs = value.float().reshape(length, config.n_heads, -1, 2)
        real = pairs[..., 0] * phase.cos() - pairs[..., 1] * phase.sin()
        imag = pairs[..., 1] * phase.cos() + pairs[..., 0] * phase.sin()
        return torch.stack((real, imag), -1).flatten(-2).to(value.dtype)

    def block(hidden, name):
        q, k, v = linear(norm(hidden, name + ".norm_1"), name + ".attn.Wqkv").chunk(3, -1)
        shape = (length, config.n_heads, config.head_dim)
        bias = weights[name + ".attn.qk_bias"]
        q, k = rotate(q.reshape(shape) + bias[0]), rotate(k.reshape(shape) + bias[1])
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(config.head_dim)
        scores.masked_fill_(future, -torch.inf)
        probabilities = scores.float().softmax(-1).to(v.dtype)
        attended = torch.einsum("hts,shd->thd", probabilities, v.reshape(shape))
        hidden = norm(
            linear(attended.reshape(length, -1), name + ".attn.proj") + hidden,
            name + ".norm_2",
        )
        gate, up = linear(norm(hidden, name + ".norm_3"), name + ".mlp.fc").chunk(2, -1)
        return norm(linear(F.silu(gate) * up, name + ".mlp.proj") + hidden, name + ".norm_4")

    injection = weights["transformer.wte.weight"][tokens] * math.sqrt(config.n_embd)
    for index in range(config.n_layers_in_prelude):
        injection = block(injection, f"transformer.prelude.{index}")
    depths = []
    for _ in range(config.mean_recurrence):
        state = linear(torch.cat((state, injection), -1), "transformer.adapter")
        for index in range(config.n_layers_in_recurrent_block):
            state = block(state, f"transformer.core_block.{index}")
        depths.append(state.clone())
    state = norm(state, "transformer.ln_f")
    for index in range(config.n_layers_in_coda):
        state = block(state, f"transformer.coda.{index}")
    return depths, linear(norm(state, "transformer.ln_f"), "lm_head")
