# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from modelopt.torch.export.quant_utils import QUANTIZATION_NVFP4

from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
    _canonicalize_nvfp4_weight_scale,
)
from nemo_rl.modelopt.models.policy.workers.qat_weight_exporter import (
    QATWeightExporter,
    _compute_nvfp4_weight_scale,
    _QuantMeta,
)
from nemo_rl.modelopt.utils import (
    build_vllm_modelopt_nvfp4_config,
    matches_quant_ignore_pattern,
)


def test_w4a16_real_quant_config_keeps_weight_only_default():
    cfg = build_vllm_modelopt_nvfp4_config()

    group = cfg["config_groups"]["group_0"]
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "NVFP4"
    assert cfg["group_size"] == 16
    assert group["input_activations"] is None
    assert group["weights"] == {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }
    assert cfg["ignore"] == [
        "lm_head",
        "*output_layer*",
        "*mlp.gate",
        "*router*",
        "*block_sparse_moe.gate*",
        "*self_attention*",
        "*self_attn*",
    ]


def test_real_quant_config_allows_explicit_ignore_override():
    cfg = build_vllm_modelopt_nvfp4_config(ignore=["lm_head"])

    assert cfg["ignore"] == ["lm_head"]


def test_exporter_matches_default_ignore_patterns_with_or_without_model_prefix():
    ignore_patterns = build_vllm_modelopt_nvfp4_config()["ignore"]

    assert matches_quant_ignore_pattern(
        "model.layers.0.self_attn.o_proj.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "layers.0.self_attn.o_proj.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern("model.layers.0.router.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("lm_head.weight", ignore_patterns)
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.weight_scale", ignore_patterns
    )
    assert not matches_quant_ignore_pattern(
        "model.layers.0.mlp.experts.0.w1.weight", ignore_patterns
    )


def test_w4a16_exporter_uses_nvfp4_global_weight_scale():
    exporter = object.__new__(QATWeightExporter)
    meta = _QuantMeta(
        qformat=QUANTIZATION_NVFP4,
        block_size=4,
        weight_amax=torch.tensor([2688.0]),
    )

    tensors = dict(
        exporter._quantize_nvfp4(
            "model.layers.0.mlp.up_proj.weight",
            torch.tensor([[-1.0, 0.25, 0.5, 2.0]], dtype=torch.float32),
            meta,
        )
    )

    torch.testing.assert_close(
        tensors["model.layers.0.mlp.up_proj.weight_scale_2"],
        torch.tensor([1.0]),
    )


def test_nvfp4_exporter_emits_non_negative_fp8_block_scales():
    weight = torch.tensor([[-1.0, 0.25, 0.5, 2.0]], dtype=torch.float32)
    weight_scale = _compute_nvfp4_weight_scale(
        weight,
        block_size=4,
        weight_scale_2=torch.tensor(1.0 / 448.0),
    )

    assert weight_scale.dtype == torch.float8_e4m3fn
    assert (weight_scale.to(torch.float32) >= 0).all()


def test_vllm_reload_canonicalizes_nvfp4_scales_before_kernel_conversion():
    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [-0.5, 4.0]]),
        requires_grad=False,
    )

    _canonicalize_nvfp4_weight_scale(layer)

    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
    )
