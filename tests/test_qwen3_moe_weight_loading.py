# SPDX-License-Identifier: MIT
# Tests for Qwen3 MoE / Qwen3 Coder MXFP4 weight-loading fix.
#
# The Qwen3-Coder-480B-A35B-Instruct-MXFP4 checkpoint excludes all attention
# projections (q_proj, k_proj, v_proj, o_proj) and the MoE router gate from
# MXFP4 quantisation -- those weights are stored as bf16.  Before the fix,
# QKVParallelLinear did not accept a ``prefix`` parameter, so it never
# consulted the exclude list and always created fp4x2 parameters, causing a
# shape / dtype mismatch in ``weight_loader_process``.
#
# These tests verify:
#   1. ``should_ignore_layer`` / ``get_quant_config_for_layer`` correctly
#      identify excluded attention layers given the Qwen3-Coder exclude list.
#   2. ``QKVParallelLinear`` now accepts a ``prefix`` kwarg and, when given an
#      excluded prefix, falls back to unquantised (bf16) weights.
#   3. ``Qwen3MoeAttention`` propagates the prefix to its sub-layers.
#   4. End-to-end weight_loader_process shape compatibility.

import importlib
import inspect
import sys
import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# The conftest.py in this directory stubs out ``atom`` and ``atom.config``
# with lightweight mocks to let other tests run without GPU / HuggingFace.
# Our tests need the *real* atom.config (QuantizationConfig) and the full
# linear / model modules backed by aiter.  Undo the stubs before importing.
# ---------------------------------------------------------------------------

_ATOM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ATOM_ROOT not in sys.path:
    sys.path.insert(0, _ATOM_ROOT)

# Remove stubs that conftest may have installed
for _key in list(sys.modules):
    if _key == "atom" or _key.startswith("atom."):
        del sys.modules[_key]

# Re-import the real atom package
import atom  # noqa: E402
import atom.config  # noqa: E402

from atom.models.utils import should_ignore_layer, get_quant_config_for_layer
from atom.config import QuantizationConfig


# -- helpers ----------------------------------------------------------------

def _build_qwen3_coder_exclude_list(num_layers: int = 3) -> List[str]:
    """Return an exclude list in the same style as Qwen3-Coder-480B MXFP4."""
    excludes: list[str] = []
    for i in range(num_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            excludes.append(f"model.layers.{i}.self_attn.{proj}")
        excludes.append(f"model.layers.{i}.mlp.gate")
    excludes.append("lm_head")
    return excludes


def _make_fake_tp_group(world_size=1, rank=0):
    """Create a mock object compatible with aiter's GroupCoordinator."""
    tp = MagicMock()
    tp.world_size = world_size
    tp.rank_in_group = rank
    return tp


@pytest.fixture
def mxfp4_quant_config():
    """A QuantizationConfig that mimics the Qwen3 Coder MXFP4 setup."""
    try:
        from aiter import QuantType
    except ImportError:
        pytest.skip("aiter not installed")
    return QuantizationConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=torch.float4_e2m1fn_x2,
        is_dynamic=True,
        quant_method="quark",
        exclude_layers=_build_qwen3_coder_exclude_list(num_layers=3),
    )


# Patch targets: these are where the TP functions are *looked up* at runtime.
_PATCH_TP_LINEAR = "atom.model_ops.linear.get_tp_group"
_PATCH_TP_WS_MOE = "atom.models.qwen3_moe.get_tensor_model_parallel_world_size"
_PATCH_TP_SOURCE = "aiter.dist.parallel_state._TP"


# == 1. should_ignore_layer / get_quant_config_for_layer ====================


class TestShouldIgnoreLayer:
    """Verify that the exclude-list logic matches the Qwen3-Coder pattern."""

    def test_none_config_always_ignored(self):
        assert should_ignore_layer(None, "anything") is True

    def test_empty_exclude_list_not_ignored(self):
        cfg = QuantizationConfig(exclude_layers=[])
        assert should_ignore_layer(cfg, "model.layers.0.self_attn") is False

    # -- Attention projections (should be excluded) -------------------------

    def test_self_attn_prefix_excluded(self, mxfp4_quant_config):
        """The QKV merged prefix 'model.layers.0.self_attn' is a substring
        of 'model.layers.0.self_attn.q_proj' -> should be excluded."""
        assert (
            should_ignore_layer(mxfp4_quant_config, "model.layers.0.self_attn")
            is True
        )

    def test_q_proj_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.0.self_attn.q_proj"
            )
            is True
        )

    def test_k_proj_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.1.self_attn.k_proj"
            )
            is True
        )

    def test_v_proj_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.2.self_attn.v_proj"
            )
            is True
        )

    def test_o_proj_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.0.self_attn.o_proj"
            )
            is True
        )

    def test_mlp_gate_excluded(self, mxfp4_quant_config):
        """The MoE router 'mlp.gate' is in the exclude list."""
        assert (
            should_ignore_layer(mxfp4_quant_config, "model.layers.0.mlp.gate")
            is True
        )

    def test_lm_head_excluded(self, mxfp4_quant_config):
        assert should_ignore_layer(mxfp4_quant_config, "lm_head") is True

    # -- Layers that should NOT be excluded ---------------------------------

    def test_mlp_experts_not_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.0.mlp.experts"
            )
            is False
        )

    def test_gate_up_proj_not_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.0.mlp.gate_up_proj"
            )
            is False
        )

    def test_down_proj_not_excluded(self, mxfp4_quant_config):
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.0.mlp.down_proj"
            )
            is False
        )

    def test_layer_beyond_exclude_range_not_excluded(self, mxfp4_quant_config):
        """Layer 5 is not in our 3-layer exclude list."""
        assert (
            should_ignore_layer(
                mxfp4_quant_config, "model.layers.5.self_attn.q_proj"
            )
            is False
        )


class TestGetQuantConfigForLayer:
    """get_quant_config_for_layer returns None for excluded layers, the
    original config otherwise."""

    def test_excluded_layer_returns_none(self, mxfp4_quant_config):
        result = get_quant_config_for_layer(
            mxfp4_quant_config, "model.layers.0.self_attn"
        )
        assert result is None

    def test_non_excluded_layer_returns_config(self, mxfp4_quant_config):
        result = get_quant_config_for_layer(
            mxfp4_quant_config, "model.layers.0.mlp.experts"
        )
        assert result is mxfp4_quant_config

    def test_none_config_returns_none(self):
        assert get_quant_config_for_layer(None, "anything") is None


# == 2. QKVParallelLinear prefix support ====================================


class TestQKVParallelLinearPrefix:
    """Verify QKVParallelLinear accepts and uses the prefix parameter."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_aiter(self):
        try:
            from aiter import QuantType
        except ImportError:
            pytest.skip("aiter not available")

    def test_accepts_prefix_kwarg(self):
        """QKVParallelLinear.__init__ should accept 'prefix' without error."""
        from atom.model_ops.linear import QKVParallelLinear

        sig = inspect.signature(QKVParallelLinear.__init__)
        assert "prefix" in sig.parameters, (
            "QKVParallelLinear.__init__ must accept a 'prefix' keyword argument"
        )

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_excluded_prefix_creates_bf16_weight(self, _mock_tp, mxfp4_quant_config):
        """When prefix matches an excluded layer, the weight should be bf16
        (QuantType.No), not fp4x2."""
        from aiter import QuantType
        from atom.model_ops.linear import QKVParallelLinear

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=64,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            quant_config=mxfp4_quant_config,
            prefix="model.layers.0.self_attn",
        )
        # Excluded -> should fall back to no quantisation
        assert layer.quant_type == QuantType.No
        assert layer.weight.dtype == torch.bfloat16

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_non_excluded_prefix_creates_quantised_weight(
        self, _mock_tp, mxfp4_quant_config
    ):
        """When prefix does NOT match an excluded layer, the weight should be
        quantised (fp4x2)."""
        from aiter import QuantType
        from atom.model_ops.linear import QKVParallelLinear

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=64,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            quant_config=mxfp4_quant_config,
            prefix="model.layers.5.self_attn",  # not in 3-layer exclude list
        )
        assert layer.quant_type == QuantType.per_1x32
        assert layer.weight.dtype == torch.float4_e2m1fn_x2

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_empty_prefix_keeps_quant(self, _mock_tp, mxfp4_quant_config):
        """An empty prefix (backward-compat default) should keep quantisation."""
        from aiter import QuantType
        from atom.model_ops.linear import QKVParallelLinear

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=64,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            quant_config=mxfp4_quant_config,
            # prefix defaults to ""
        )
        assert layer.quant_type == QuantType.per_1x32

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_no_quant_config_still_works(self, _mock_tp):
        """No quant_config at all should not break, even with a prefix."""
        from aiter import QuantType
        from atom.model_ops.linear import QKVParallelLinear

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=64,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
        assert layer.quant_type == QuantType.No


# == 3. RowParallelLinear prefix support (pre-existing, regression test) ====


class TestRowParallelLinearPrefix:
    """RowParallelLinear already had prefix support -- regression test."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_aiter(self):
        try:
            from aiter import QuantType
        except ImportError:
            pytest.skip("aiter not available")

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_excluded_prefix_creates_bf16_weight(self, _mock_tp, mxfp4_quant_config):
        from aiter import QuantType
        from atom.model_ops.linear import RowParallelLinear

        layer = RowParallelLinear(
            input_size=128,
            output_size=128,
            bias=False,
            quant_config=mxfp4_quant_config,
            prefix="model.layers.0.self_attn.o_proj",
        )
        assert layer.quant_type == QuantType.No
        assert layer.weight.dtype == torch.bfloat16


# == 4. Qwen3MoeAttention wiring ==========================================


class TestQwen3MoeAttentionPrefixWiring:
    """Verify Qwen3MoeAttention accepts and passes prefix to sub-layers.

    Instantiating Qwen3MoeAttention fully requires deep framework state
    (Attention -> get_current_atom_config, get_rope, etc.).  Instead we
    mock QKVParallelLinear / RowParallelLinear and verify they receive the
    correct ``prefix`` argument.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_aiter(self):
        try:
            from aiter import QuantType
        except ImportError:
            pytest.skip("aiter not available")

    def test_attention_accepts_prefix(self):
        """Qwen3MoeAttention.__init__ should accept a 'prefix' kwarg."""
        from atom.models.qwen3_moe import Qwen3MoeAttention

        sig = inspect.signature(Qwen3MoeAttention.__init__)
        assert "prefix" in sig.parameters, (
            "Qwen3MoeAttention.__init__ must accept a 'prefix' parameter"
        )

    @patch("atom.models.qwen3_moe.Attention")
    @patch("atom.models.qwen3_moe.get_rope")
    @patch("atom.models.qwen3_moe.RMSNorm", return_value=MagicMock())
    @patch(_PATCH_TP_SOURCE, _make_fake_tp_group())
    @patch(_PATCH_TP_WS_MOE, return_value=1)
    @patch("atom.models.qwen3_moe.RowParallelLinear")
    @patch("atom.models.qwen3_moe.QKVParallelLinear")
    def test_prefix_passed_to_qkv_and_o_proj(
        self, mock_qkv_cls, mock_row_cls, _mock_ws,
        _mock_norm, _mock_rope, _mock_attn, mxfp4_quant_config
    ):
        """QKVParallelLinear gets the attention prefix,
        RowParallelLinear gets prefix + '.o_proj'."""
        from atom.models.qwen3_moe import Qwen3MoeAttention

        PREFIX = "model.layers.0.self_attn"
        Qwen3MoeAttention(
            hidden_size=128,
            num_heads=2,
            num_kv_heads=2,
            head_dim=64,
            quant_config=mxfp4_quant_config,
            prefix=PREFIX,
        )

        # QKVParallelLinear should have been called with prefix=PREFIX
        _, qkv_kwargs = mock_qkv_cls.call_args
        assert qkv_kwargs.get("prefix") == PREFIX, (
            f"QKVParallelLinear got prefix={qkv_kwargs.get('prefix')!r}, "
            f"expected {PREFIX!r}"
        )

        # RowParallelLinear should have been called with prefix=PREFIX.o_proj
        _, row_kwargs = mock_row_cls.call_args
        expected_o_prefix = f"{PREFIX}.o_proj"
        assert row_kwargs.get("prefix") == expected_o_prefix, (
            f"RowParallelLinear got prefix={row_kwargs.get('prefix')!r}, "
            f"expected {expected_o_prefix!r}"
        )

    @patch("atom.models.qwen3_moe.Attention")
    @patch("atom.models.qwen3_moe.get_rope")
    @patch("atom.models.qwen3_moe.RMSNorm", return_value=MagicMock())
    @patch(_PATCH_TP_SOURCE, _make_fake_tp_group())
    @patch(_PATCH_TP_WS_MOE, return_value=1)
    @patch("atom.models.qwen3_moe.RowParallelLinear")
    @patch("atom.models.qwen3_moe.QKVParallelLinear")
    def test_different_layer_prefix_propagated(
        self, mock_qkv_cls, mock_row_cls, _mock_ws,
        _mock_norm, _mock_rope, _mock_attn, mxfp4_quant_config
    ):
        """Verify a different layer index is propagated correctly."""
        from atom.models.qwen3_moe import Qwen3MoeAttention

        PREFIX = "model.layers.42.self_attn"
        Qwen3MoeAttention(
            hidden_size=128,
            num_heads=2,
            num_kv_heads=2,
            head_dim=64,
            quant_config=mxfp4_quant_config,
            prefix=PREFIX,
        )

        _, qkv_kwargs = mock_qkv_cls.call_args
        assert qkv_kwargs.get("prefix") == PREFIX

        _, row_kwargs = mock_row_cls.call_args
        assert row_kwargs.get("prefix") == f"{PREFIX}.o_proj"

    @patch("atom.models.qwen3_moe.Attention")
    @patch("atom.models.qwen3_moe.get_rope")
    @patch("atom.models.qwen3_moe.RMSNorm", return_value=MagicMock())
    @patch(_PATCH_TP_SOURCE, _make_fake_tp_group())
    @patch(_PATCH_TP_WS_MOE, return_value=1)
    @patch("atom.models.qwen3_moe.RowParallelLinear")
    @patch("atom.models.qwen3_moe.QKVParallelLinear")
    def test_empty_prefix_still_propagated(
        self, mock_qkv_cls, mock_row_cls, _mock_ws,
        _mock_norm, _mock_rope, _mock_attn, mxfp4_quant_config
    ):
        """An empty prefix should still be passed through (backward compat)."""
        from atom.models.qwen3_moe import Qwen3MoeAttention

        Qwen3MoeAttention(
            hidden_size=128,
            num_heads=2,
            num_kv_heads=2,
            head_dim=64,
            quant_config=mxfp4_quant_config,
            prefix="",
        )

        _, qkv_kwargs = mock_qkv_cls.call_args
        assert qkv_kwargs.get("prefix") == ""

        _, row_kwargs = mock_row_cls.call_args
        assert row_kwargs.get("prefix") == ".o_proj"


# == 5. End-to-end weight_loader_process shape compatibility ================


class TestWeightLoaderProcessCompat:
    """Simulate the actual weight-loading copy that was failing before the fix."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_aiter(self):
        try:
            from aiter import QuantType
        except ImportError:
            pytest.skip("aiter not available")

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_bf16_weight_loads_into_excluded_qkv(self, _mock_tp, mxfp4_quant_config):
        """A bf16 checkpoint weight should successfully load into an excluded
        (bf16) QKVParallelLinear -- this was the crash scenario."""
        from atom.model_ops.linear import QKVParallelLinear

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=64,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            quant_config=mxfp4_quant_config,
            prefix="model.layers.0.self_attn",
        )
        # Simulate checkpoint weight: bf16, same shape as the param
        fake_weight = torch.randn_like(layer.weight.data, dtype=torch.bfloat16)

        # This should NOT raise -- it was the crash path before the fix
        layer.weight.weight_loader_process(layer.weight.data, fake_weight)
        # Verify the data was copied
        assert torch.equal(layer.weight.data, fake_weight)

    @patch(_PATCH_TP_LINEAR, return_value=_make_fake_tp_group())
    def test_bf16_weight_loads_into_excluded_row_parallel(
        self, _mock_tp, mxfp4_quant_config
    ):
        """bf16 checkpoint weight loads into excluded RowParallelLinear."""
        from atom.model_ops.linear import RowParallelLinear

        layer = RowParallelLinear(
            input_size=128,
            output_size=128,
            bias=False,
            quant_config=mxfp4_quant_config,
            prefix="model.layers.0.self_attn.o_proj",
        )
        fake_weight = torch.randn_like(layer.weight.data, dtype=torch.bfloat16)
        layer.weight.weight_loader_process(layer.weight.data, fake_weight)
        assert torch.equal(layer.weight.data, fake_weight)
