# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Regression test for the MXFP4 MoE uninitialized bias bug.

Root cause:
  FusedMoE defaulted has_bias=True, but Qwen3MoE experts have no bias
  in the checkpoint. Mxfp4MoEMethod.create_weights allocated bias
  parameters with torch.empty() that never got loaded, causing the
  kernel to add garbage bias to every expert output.

Fix:
  - FusedMoE default changed to has_bias=False
  - Qwen3MoeSparseMoeBlock and Qwen3NextSparseMoeBlock explicitly
    pass has_bias=False
"""

import sys
import unittest

# Clear cached atom modules (conftest.py stubs)
for mod_name in list(sys.modules):
    if mod_name.startswith("atom"):
        del sys.modules[mod_name]


class TestFusedMoEDefaultHasBias(unittest.TestCase):
    """FusedMoE must default to has_bias=False."""

    def test_default_is_false(self):
        import inspect
        from atom.model_ops.moe import FusedMoE

        sig = inspect.signature(FusedMoE.__init__)
        default = sig.parameters["has_bias"].default
        self.assertFalse(
            default,
            "FusedMoE default has_bias must be False to prevent "
            "uninitialized bias when checkpoint has no expert bias",
        )


class TestQwen3MoeExplicitHasBias(unittest.TestCase):
    """Qwen3 MoE models must explicitly pass has_bias=False."""

    def _check_source_has_bias_false(self, module_path: str, class_name: str):
        import importlib
        import inspect

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        source = inspect.getsource(cls.__init__)
        self.assertIn(
            "has_bias=False",
            source,
            f"{class_name} must pass has_bias=False to FusedMoE",
        )

    def test_qwen3_moe_sparse_block(self):
        self._check_source_has_bias_false(
            "atom.models.qwen3_moe", "Qwen3MoeSparseMoeBlock"
        )

    def test_qwen3_next_sparse_block(self):
        self._check_source_has_bias_false(
            "atom.models.qwen3_next", "Qwen3NextSparseMoeBlock"
        )


class TestGptOssKeepsBias(unittest.TestCase):
    """gpt_oss explicitly uses has_bias=True and must not be affected."""

    def test_gpt_oss_has_bias_true(self):
        import inspect
        from atom.models.gpt_oss import MLPBlock as SparseMoeBlock

        source = inspect.getsource(SparseMoeBlock.__init__)
        self.assertIn(
            "has_bias=True",
            source,
            "gpt_oss SparseMoeBlock must keep has_bias=True",
        )


class TestMxfp4NoBiasCreated(unittest.TestCase):
    """When has_bias=False, Mxfp4MoEMethod must not create bias parameters."""

    def test_no_bias_when_has_bias_false(self):
        import torch
        from unittest.mock import MagicMock

        from atom.model_ops.moe import Mxfp4MoEMethod
        from atom.config import QuantizationConfig
        from aiter import QuantType

        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        moe_config = MagicMock()
        method = Mxfp4MoEMethod(qc, moe_config)

        # Create a mock layer with has_bias=False
        layer = MagicMock()
        layer.has_bias = False
        layer.hidden_size = 6144
        layer.intermediate_size_per_partition = 2560
        layer.activation = "silu"

        # Track what register_parameter is called with
        registered = {}

        def mock_register(name, param):
            registered[name] = param

        layer.register_parameter = mock_register

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=6144,
            intermediate_size_per_partition=2560,
            params_dtype=torch.float4_e2m1fn_x2,
            weight_loader=lambda *a: None,
        )

        # Bias should be None when has_bias=False
        self.assertIsNone(
            registered.get("w13_bias"),
            "w13_bias must be None when has_bias=False",
        )
        self.assertIsNone(
            registered.get("w2_bias"),
            "w2_bias must be None when has_bias=False",
        )

    def test_bias_created_when_has_bias_true(self):
        import torch
        from unittest.mock import MagicMock

        from atom.model_ops.moe import Mxfp4MoEMethod
        from atom.config import QuantizationConfig
        from aiter import QuantType

        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        moe_config = MagicMock()
        method = Mxfp4MoEMethod(qc, moe_config)

        # Create a mock layer with has_bias=True
        layer = MagicMock()
        layer.has_bias = True
        layer.hidden_size = 6144
        layer.intermediate_size_per_partition = 2560
        layer.activation = "silu"

        registered = {}

        def mock_register(name, param):
            registered[name] = param

        layer.register_parameter = mock_register

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=6144,
            intermediate_size_per_partition=2560,
            params_dtype=torch.float4_e2m1fn_x2,
            weight_loader=lambda *a: None,
        )

        # Bias should be a Parameter when has_bias=True
        self.assertIsNotNone(registered.get("w13_bias"))
        self.assertIsInstance(registered["w13_bias"], torch.nn.Parameter)
        self.assertIsNotNone(registered.get("w2_bias"))
        self.assertIsInstance(registered["w2_bias"], torch.nn.Parameter)


if __name__ == "__main__":
    unittest.main()
