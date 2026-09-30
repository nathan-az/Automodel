# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for nemo_automodel/components/attention/ffpa_attention.py."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.attention import ffpa_attention as ffpa_mod
from nemo_automodel.components.attention.ffpa_attention import (
    ffpa_attention_forward,
    register_ffpa_attention,
    setup_ffpa_backend,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    ffpa_mod._FALLBACK_WARNED.clear()
    ffpa_mod._SDPA_FN = None
    ffpa_mod._EAGER_FN = None
    ffpa_mod._FLEX_FN = None
    ffpa_mod._FFPA_HIGH_LEVEL = None
    ffpa_mod._FFPA_LOW_LEVEL_READY = None
    yield
    ffpa_mod._FALLBACK_WARNED.clear()


def _module(head_dim, training=False):
    m = nn.Module()
    m.head_dim = head_dim
    m.training = training
    return m


def _qkv(B, Hq, Hkv, S, D, dtype=torch.bfloat16):
    return (
        torch.randn(B, Hq, S, D, dtype=dtype),
        torch.randn(B, Hkv, S, D, dtype=dtype),
        torch.randn(B, Hkv, S, D, dtype=dtype),
    )


def _patch_sdpa(rv=None):
    rv = rv if rv is not None else (torch.zeros(1, 16, 8, 512), None)
    return mock.patch("transformers.integrations.sdpa_attention.sdpa_attention_forward", return_value=rv)


def _patch_eager(rv=None):
    rv = rv if rv is not None else (torch.zeros(1, 16, 8, 512), None)
    return mock.patch("transformers.models.gemma4.modeling_gemma4.eager_attention_forward", return_value=rv)


def _warns(caplog, substr):
    return [r for r in caplog.records if r.name == ffpa_mod.__name__ and substr in r.getMessage()]


def test_module_exports_only_public_entry_points():
    assert ffpa_mod.__all__ == [
        "ffpa_attention_forward",
        "ffpa_mask",
        "register_ffpa_attention",
        "setup_ffpa_backend",
    ]


def test_registration_idempotent_and_visible():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ffpa_mod._REGISTERED = False
    ALL_ATTENTION_FUNCTIONS._global_mapping.pop("ffpa", None)

    assert register_ffpa_attention() is True
    assert ALL_ATTENTION_FUNCTIONS._global_mapping["ffpa"] is ffpa_attention_forward
    assert register_ffpa_attention() is False


# The HF 'ffpa' backend calls the op directly, bypassing the SDPA swap ring CP installs
# and packed-sequence masking; both must be rejected (for custom models like Gemma4 this
# is the only check before a silently-wrong run).
@pytest.mark.parametrize(
    "cp_size, has_packed_sequence, match",
    [(8, False, "context parallelism"), (1, True, "packed sequences")],
)
def test_setup_ffpa_backend_rejects_unsupported(cp_size, has_packed_sequence, match):
    with pytest.raises(ValueError, match=match):
        setup_ffpa_backend(cp_size, has_packed_sequence)


def test_setup_ffpa_backend_registers_when_eligible():
    # Patch the registrar so the eligible path doesn't mutate the global HF registry.
    with mock.patch.object(ffpa_mod, "register_ffpa_attention") as reg:
        setup_ffpa_backend(cp_size=1, has_packed_sequence=False)
    reg.assert_called_once_with()


def test_non_512_head_dim_routes_to_sdpa_silently(caplog):
    q, k, v = _qkv(1, 4, 4, 8, 64)
    sentinel = (torch.zeros(1, 8, 4, 64), None)
    with _patch_sdpa(sentinel) as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        out, weights = ffpa_attention_forward(_module(64), q, k, v, None, scaling=0.125)
    mock_sdpa.assert_called_once()
    assert out is sentinel[0]
    assert weights is sentinel[1]
    assert not _warns(caplog, "fallback")


def test_fp32_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 8, 512, dtype=torch.float32)
    with _patch_sdpa() as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        ffpa_attention_forward(_module(512), q, k, v, None, scaling=0.0442)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "dtype")) == 1


# softcap layers must stay on eager: SDPA silently drops the softcap kwarg.
def test_softcap_routes_to_eager_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 8, 512)
    with (
        _patch_sdpa() as mock_sdpa,
        _patch_eager() as mock_eager,
        caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__),
    ):
        ffpa_attention_forward(_module(512), q, k, v, None, softcap=50.0, scaling=0.0442)
    mock_eager.assert_called_once()
    mock_sdpa.assert_not_called()
    assert len(_warns(caplog, "softcap")) == 1


def test_4d_mask_routes_to_sdpa_silently():
    q, k, v = _qkv(1, 8, 4, 16, 512)
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16)
    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        _patch_sdpa() as mock_sdpa,
    ):
        ffpa_attention_forward(_module(512), q, k, v, attention_mask=mask, scaling=0.0442)
    mock_sdpa.assert_called_once()


def test_training_dropout_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 16, 512)
    with _patch_sdpa() as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        ffpa_attention_forward(_module(512, training=True), q, k, v, None, dropout=0.1, scaling=0.0442)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "dropout")) == 1


def test_none_scaling_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 16, 512)
    with _patch_sdpa() as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        ffpa_attention_forward(_module(512), q, k, v, None, scaling=None)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "scaling")) == 1


def test_missing_ffpa_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 8, 512)
    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=False),
        _patch_sdpa() as mock_sdpa,
        caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__),
    ):
        ffpa_attention_forward(_module(512), q, k, v, None, scaling=0.0442)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "ffpa_unavailable")) == 1


def test_degrades_to_eager_when_sdpa_unimportable():
    q, k, v = _qkv(1, 4, 4, 8, 64)
    with (
        mock.patch.object(ffpa_mod, "_get_sdpa", return_value=None),
        _patch_eager() as mock_eager,
    ):
        ffpa_attention_forward(_module(64), q, k, v, None, scaling=0.125)
    mock_eager.assert_called_once()


def test_warning_is_deduped_across_calls(caplog):
    q, k, v = _qkv(1, 8, 4, 16, 512, dtype=torch.float32)
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16)
    with _patch_sdpa(), caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        for _ in range(5):
            ffpa_attention_forward(_module(512), q, k, v, attention_mask=mask, scaling=0.0442)
    assert len(_warns(caplog, "dtype")) == 1


def test_dense_path_calls_high_level_func_with_fa_layout():
    B, Hq, Hkv, S, D = 1, 8, 4, 16, 512
    q, k, v = _qkv(B, Hq, Hkv, S, D)
    backend = mock.MagicMock()
    ffpa_fn = mock.Mock(return_value=torch.zeros(B, Hq, S, D, dtype=torch.bfloat16))

    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        mock.patch.object(ffpa_mod, "_get_ffpa_high_level", return_value=(ffpa_fn, backend)),
    ):
        out, weights = ffpa_attention_forward(_module(512), q, k, v, attention_mask=None, dropout=0.0, scaling=0.0442)

    ffpa_fn.assert_called_once()
    args, kwargs = ffpa_fn.call_args
    assert args[0] is q and args[1] is k and args[2] is v
    assert kwargs == {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": True,
        "scale": 0.0442,
        "enable_gqa": True,
        "backend": backend,
    }
    assert out.shape == (B, S, Hq, D)
    assert weights is None


def test_2d_bool_mask_returns_padded_output():
    B, Hq, Hkv, S, D = 1, 2, 1, 2, 512
    q, k, v = _qkv(B, Hq, Hkv, S, D)
    attention_mask = torch.tensor([[True, False]])
    out_pack = q.transpose(1, 2)[attention_mask]
    expected = torch.zeros_like(q.transpose(1, 2))
    expected[attention_mask] = out_pack

    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        mock.patch.object(ffpa_mod, "_ffpa_varlen_fwd", return_value=(out_pack, torch.empty(Hq, 1))),
    ):
        out, weights = ffpa_attention_forward(_module(D), q, k, v, attention_mask, scaling=0.0442)

    assert torch.equal(out, expected)
    assert weights is None


def test_ffpa_mask_returns_only_nontrivial_padding_mask():
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    assert ffpa_mod.ffpa_mask(1, 4, 4, attention_mask=attention_mask) is None
    attention_mask[0, 1] = False
    assert torch.equal(ffpa_mod.ffpa_mask(1, 4, 4, attention_mask=attention_mask), attention_mask)


@pytest.mark.parametrize("causal", [False, True])
def test_ffpa_mask_routes_only_sliding_to_flex(causal):
    from transformers.masking_utils import causal_mask_function

    mask_function = causal_mask_function if causal else (lambda b, h, qi, ki: ki <= qi)
    sentinel = object()
    with mock.patch("transformers.masking_utils.flex_attention_mask", return_value=sentinel) as mock_flex_mask:
        out = ffpa_mod.ffpa_mask(
            batch_size=1, q_length=8, kv_length=8, mask_function=mask_function, attention_mask=None, device="cpu"
        )
    assert (mock_flex_mask.call_count == 1) is (not causal)
    assert out is (None if causal else sentinel)


def test_ffpa_mask_can_route_sliding_to_sdpa():
    from transformers.masking_utils import sliding_window_causal_mask_function

    config = SimpleNamespace(ffpa_sliding_attn_backend="sdpa")
    out = ffpa_mod.ffpa_mask(
        batch_size=1,
        q_length=4,
        kv_length=4,
        mask_function=sliding_window_causal_mask_function(2),
        device="cpu",
        config=config,
        local_size=2,
    )

    expected = torch.tensor(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, True, True, False],
                    [False, False, True, True],
                ]
            ]
        ]
    )
    assert torch.equal(out, expected)


def test_ffpa_mask_rejects_unknown_sliding_backend():
    with pytest.raises(ValueError, match="ffpa_sliding_attn_backend must be 'flex', 'sdpa' or 'fa', got 'invalid'"):
        ffpa_mod.ffpa_mask(
            batch_size=1,
            q_length=4,
            kv_length=4,
            mask_function=lambda b, h, qi, ki: ki <= qi,
            config=SimpleNamespace(ffpa_sliding_attn_backend="invalid"),
        )


def test_block_mask_routes_to_flex_not_sdpa():
    q, k, v = _qkv(1, 8, 4, 16, 256)
    flex_out = (torch.zeros(1, 16, 8, 256), None)
    fake_flex = mock.Mock(return_value=flex_out)
    with (
        mock.patch.object(ffpa_mod, "_is_block_mask", return_value=True),
        mock.patch.object(ffpa_mod, "_get_flex", return_value=fake_flex),
        _patch_sdpa() as mock_sdpa,
    ):
        out, weights = ffpa_attention_forward(_module(256), q, k, v, attention_mask=object(), scaling=0.0625)
    fake_flex.assert_called_once()
    mock_sdpa.assert_not_called()
    assert out is flex_out[0]
    assert weights is None


def test_block_mask_requires_flex_no_sdpa_fallback():
    q, k, v = _qkv(1, 8, 4, 16, 256)
    with (
        mock.patch.object(ffpa_mod, "_is_block_mask", return_value=True),
        mock.patch.object(ffpa_mod, "_get_flex", return_value=None),
        _patch_sdpa() as mock_sdpa,
        pytest.raises(RuntimeError),
    ):
        ffpa_attention_forward(_module(256), q, k, v, attention_mask=object(), scaling=0.0625)
    mock_sdpa.assert_not_called()


# --- ffpa_sliding_attn_backend="fa": FlashAttention-2 for sliding-window layers ---

_FA_CONFIG = SimpleNamespace(ffpa_sliding_attn_backend="fa")


def _plain_sliding_mask_kwargs(**overrides):
    """Keyword arguments Transformers passes ffpa_mask for an unpacked, overlay-free sliding-window mask."""
    from transformers.masking_utils import sliding_window_causal_mask_function

    kwargs = dict(
        batch_size=2,
        q_length=4,
        kv_length=4,
        mask_function=sliding_window_causal_mask_function(2),
        attention_mask=None,
        dtype=torch.bfloat16,
        config=_FA_CONFIG,
        local_size=2,
        allow_is_causal_skip=True,
        device="cpu",
    )
    kwargs.update(overrides)
    return kwargs


def _sliding_module(head_dim, sliding_window, backend="fa"):
    m = _module(head_dim)
    m.sliding_window = sliding_window
    m.config = SimpleNamespace(ffpa_sliding_attn_backend=backend)
    return m


def test_ffpa_mask_fa_skips_block_mask_for_plain_sliding_window(monkeypatch):
    monkeypatch.setattr(ffpa_mod, "HAS_FLASH_ATTN", True)
    padding = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    with mock.patch("transformers.masking_utils.flex_attention_mask") as mock_flex_mask:
        unpadded = ffpa_mod.ffpa_mask(**_plain_sliding_mask_kwargs())
        padded = ffpa_mod.ffpa_mask(**_plain_sliding_mask_kwargs(attention_mask=padding))
    mock_flex_mask.assert_not_called()
    assert unpadded is None
    assert torch.equal(padded, padding.bool())


@pytest.mark.parametrize(
    "overrides",
    [
        dict(allow_is_causal_skip=False),  # Transformers composed an overlay (vision, packing, custom fn)
        dict(local_size=None),  # not a sliding-window mask
        dict(q_length=1, q_offset=3),  # cache / decoding
        dict(dtype=torch.float32),
    ],
    ids=["composed", "not_sliding", "decoding", "fp32"],
)
def test_ffpa_mask_fa_keeps_flex_block_mask_when_fa_cannot_own_the_mask(monkeypatch, overrides):
    monkeypatch.setattr(ffpa_mod, "HAS_FLASH_ATTN", True)
    sentinel = object()
    with mock.patch("transformers.masking_utils.flex_attention_mask", return_value=sentinel) as mock_flex_mask:
        out = ffpa_mod.ffpa_mask(**_plain_sliding_mask_kwargs(**overrides))
    mock_flex_mask.assert_called_once()
    assert out is sentinel


def test_ffpa_mask_fa_keeps_flex_block_mask_without_flash_attn(monkeypatch, caplog):
    monkeypatch.setattr(ffpa_mod, "HAS_FLASH_ATTN", False)
    sentinel = object()
    with (
        caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__),
        mock.patch("transformers.masking_utils.flex_attention_mask", return_value=sentinel),
    ):
        out = ffpa_mod.ffpa_mask(**_plain_sliding_mask_kwargs())
    assert out is sentinel
    assert _warns(caplog, "flash_attn is not importable")


def test_fa_route_calls_flash_attn_with_hf_sliding_window_semantics():
    q, k, v = _qkv(1, 8, 4, 16, 256)
    fa_out = torch.zeros(1, 16, 8, 256)
    with (
        mock.patch.object(ffpa_mod, "_flash_attn_func", return_value=fa_out) as mock_fa,
        _patch_sdpa() as mock_sdpa,
    ):
        out, weights = ffpa_attention_forward(_sliding_module(256, 4), q, k, v, attention_mask=None, scaling=1.0)
    mock_sdpa.assert_not_called()
    assert out is fa_out and weights is None
    fa_q, fa_k, _ = mock_fa.call_args.args
    assert fa_q.shape == (1, 16, 8, 256) and fa_k.shape == (1, 16, 4, 256)  # [batch, sequence, heads, head_dim]
    assert mock_fa.call_args.kwargs == dict(
        dropout_p=0.0, softmax_scale=1.0, causal=True, window_size=(3, 0), softcap=0.0
    )


@pytest.mark.parametrize(
    "module",
    [_sliding_module(256, None), _sliding_module(256, 4, backend="flex")],
    ids=["full_layer", "flex_backend"],
)
def test_fa_route_is_only_for_sliding_layers_with_fa_backend(module):
    q, k, v = _qkv(1, 8, 4, 16, 256)
    with mock.patch.object(ffpa_mod, "_flash_attn_func") as mock_fa, _patch_sdpa() as mock_sdpa:
        ffpa_attention_forward(module, q, k, v, attention_mask=None, scaling=1.0)
    mock_fa.assert_not_called()
    mock_sdpa.assert_called_once()


def test_fa_route_rejects_head_dim_above_flash_attn_limit():
    q, k, v = _qkv(1, 8, 4, 16, 512)
    with pytest.raises(RuntimeError, match="head_dim <= 256"):
        ffpa_attention_forward(_sliding_module(512, 4), q, k, v, attention_mask=None, scaling=1.0)


@pytest.mark.skipif(not (torch.cuda.is_available() and ffpa_mod.HAS_FLASH_ATTN), reason="requires CUDA and flash_attn")
@pytest.mark.parametrize("padding_side", ["none", "right", "left"])
def test_fa_sliding_matches_fp32_reference(padding_side):
    batch, q_heads, kv_heads, seq, head_dim, window = 2, 4, 2, 96, 256, 16
    padding = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    if padding_side == "right":
        padding[1, -20:] = 0
    elif padding_side == "left":
        padding[1, :20] = 0
    mask = ffpa_mod.ffpa_mask(
        **_plain_sliding_mask_kwargs(
            batch_size=batch,
            q_length=seq,
            kv_length=seq,
            local_size=window,
            attention_mask=padding if padding_side != "none" else None,
            device="cuda",
        )
    )
    assert (mask is None) is (padding_side == "none")
    g = torch.Generator(device="cuda").manual_seed(0)
    q32 = torch.randn(batch, q_heads, seq, head_dim, device="cuda", generator=g)
    k32 = torch.randn(batch, kv_heads, seq, head_dim, device="cuda", generator=g)
    v32 = torch.randn(batch, kv_heads, seq, head_dim, device="cuda", generator=g)
    grad_out = torch.randn(batch, seq, q_heads, head_dim, device="cuda", generator=g)
    valid = padding.bool()

    def run(q, k, v, attn):
        """Run ``attn(q, k, v)`` and backprop ``grad_out`` over valid query rows.

        Args:
            q: Query tensor of shape [batch, query_heads, sequence, head_dim].
            k: Key tensor of shape [batch, key_value_heads, sequence, head_dim].
            v: Value tensor of shape [batch, key_value_heads, sequence, head_dim].
            attn: Callable returning an output tensor of shape [batch, sequence, query_heads, head_dim].

        Returns:
            List of fp32 tensors: the output on valid rows of shape [valid_tokens, query_heads, head_dim], then
            the query, key, and value gradients in their input shapes.
        """
        q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
        out = attn(q, k, v)
        (out.float() * grad_out)[valid].sum().backward()
        return [out.float()[valid], q.grad.float(), k.grad.float(), v.grad.float()]

    def reference(q, k, v):
        """fp32 math attention with HF sliding-window semantics (keep kv_idx in (q_idx - window, q_idx]).

        Args:
            q: Query tensor of shape [batch, query_heads, sequence, head_dim].
            k: Key tensor of shape [batch, key_value_heads, sequence, head_dim].
            v: Value tensor of shape [batch, key_value_heads, sequence, head_dim].

        Returns:
            Output tensor of shape [batch, sequence, query_heads, head_dim].
        """
        idx = torch.arange(seq, device="cuda")
        keep = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - window)
        keep = keep[None, None] & valid[:, None, None, :]
        rep = q_heads // kv_heads
        scores = q @ k.repeat_interleave(rep, 1).transpose(-1, -2)
        probs = scores.masked_fill(~keep, float("-inf")).softmax(-1).nan_to_num()
        return (probs @ v.repeat_interleave(rep, 1)).transpose(1, 2)

    def bf16_sdpa(q, k, v):
        """bf16 SDPA with the same explicit mask; its error vs fp32 is the bf16 noise floor.

        Args:
            q: Query tensor of shape [batch, query_heads, sequence, head_dim].
            k: Key tensor of shape [batch, key_value_heads, sequence, head_dim].
            v: Value tensor of shape [batch, key_value_heads, sequence, head_dim].

        Returns:
            Output tensor of shape [batch, sequence, query_heads, head_dim].
        """
        idx = torch.arange(seq, device="cuda")
        keep = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - window)
        keep = keep[None, None] & valid[:, None, None, :]
        keep = keep | ~keep.any(-1, keepdim=True)  # fully padded query rows: any finite output (ignored)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=1.0, enable_gqa=True)
        return out.transpose(1, 2)

    module = _sliding_module(head_dim, window)
    expected = run(q32, k32, v32, reference)
    bf16 = [t.bfloat16() for t in (q32, k32, v32)]
    floor = run(*bf16, bf16_sdpa)
    actual = run(*bf16, lambda q, k, v: ffpa_attention_forward(module, q, k, v, attention_mask=mask, scaling=1.0)[0])
    for name, a, f, e in zip(["out", "dq", "dk", "dv"], actual, floor, expected):
        rel, rel_floor = ((a - e).norm() / e.norm()).item(), ((f - e).norm() / e.norm()).item()
        assert rel <= 1.5 * rel_floor + 1e-3, f"{name}: rel-L2 {rel:.4f} vs bf16 SDPA floor {rel_floor:.4f}"
