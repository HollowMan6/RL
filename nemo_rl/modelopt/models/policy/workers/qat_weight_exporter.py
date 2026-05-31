# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Export ModelOpt QAT weights in the real NVFP4 format expected by vLLM."""

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from megatron.core.utils import unwrap_model
from modelopt.torch.export.quant_utils import (
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    get_quantization_format,
    get_weight_block_size,
    to_quantized_weight,
)
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

from nemo_rl.modelopt.utils import matches_quant_ignore_pattern

_NVFP4_AMAX_DENOMINATOR = 6.0 * 448.0


@dataclass
class _QuantMeta:
    """Quantization metadata for one Megatron parameter."""

    qformat: str
    block_size: int
    weight_amax: torch.Tensor | None


class QATWeightExporter:
    """Wrap Megatron-Bridge HF export and emit packed NVFP4 tensors."""

    def __init__(
        self,
        actor_module: list,
        bridge: Any,
        ignore_patterns: list[str] | None = None,
    ):
        self._actor_module = actor_module
        self._registry = bridge._model_bridge.mapping_registry()
        self._ignore_patterns = ignore_patterns or []

        from megatron.core import parallel_state as mpu

        pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._pp_group = (
            mpu.get_pipeline_model_parallel_group() if pp_size > 1 else None
        )

        self._config = self._unwrap_first(actor_module[0]).config
        self._metadata: dict[str, _QuantMeta] = {}
        self._collect_metadata(actor_module)

        if self._pp_group is not None:
            self._sync_metadata(self._pp_group)

    def process_weights_iterator(
        self,
        per_tensor_param: Iterator[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield packed NVFP4 weights and scales for quantized linear weights."""
        for hf_name, weight in per_tensor_param:
            if "_quantizer." in hf_name:
                continue

            ignored = matches_quant_ignore_pattern(hf_name, self._ignore_patterns)
            meta = None
            if not ignored and hf_name.endswith(".weight") and "norm" not in hf_name:
                for resolved in _iter_hf_to_megatron_matches(self._registry, hf_name):
                    meta = self._metadata.get(resolved.megatron_param)
                    if meta is not None:
                        break
            if meta is None:
                requires_metadata = (
                    not ignored
                    and hf_name.endswith(".weight")
                    and not any(
                        token in hf_name
                        for token in ("embed_tokens", "lm_head", "norm")
                    )
                    and any(
                        token in hf_name
                        for token in (
                            ".mlp.gate_proj.",
                            ".mlp.up_proj.",
                            ".mlp.down_proj.",
                        )
                    )
                )
                if requires_metadata:
                    raise RuntimeError(
                        f"Missing ModelOpt quant metadata for exported weight {hf_name}"
                    )
                yield hf_name, weight.detach()
            else:
                if meta.qformat != QUANTIZATION_NVFP4:
                    raise RuntimeError(
                        f"Unsupported qformat for real NVFP4 rollout: {meta.qformat}"
                    )
                yield from self._quantize_nvfp4(hf_name, weight, meta)

    @staticmethod
    def _unwrap_first(module):
        unwrapped = unwrap_model(module)
        if isinstance(unwrapped, (list, tuple)):
            return unwrapped[0]
        return unwrapped

    def _collect_metadata(self, actor_module: list) -> None:
        from megatron.bridge.models.conversion.model_bridge import (
            _megatron_local_name_to_global,
        )

        for vpp_idx, module in enumerate(actor_module):
            model = self._unwrap_first(module)
            for name, submodule in model.named_modules():
                qformat = get_quantization_format(submodule)
                if qformat == QUANTIZATION_NONE:
                    continue
                block_size = get_weight_block_size(submodule)
                if block_size == 0:
                    continue

                weight_quantizer = submodule.weight_quantizer
                weight_amax = (
                    weight_quantizer._amax.clone().cpu()
                    if weight_quantizer._amax is not None
                    else None
                )

                meta = _QuantMeta(
                    qformat=qformat,
                    block_size=block_size,
                    weight_amax=weight_amax,
                )

                for param_name, _ in submodule.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name
                    global_name = _megatron_local_name_to_global(
                        self._actor_module,
                        self._config,
                        full_name,
                        vpp_idx,
                    )
                    self._metadata[global_name] = meta

    def _sync_metadata(self, group) -> None:
        world_size = torch.distributed.get_world_size(group=group)
        local_info = {
            name: {
                "qformat": meta.qformat,
                "block_size": meta.block_size,
                "weight_amax": meta.weight_amax,
            }
            for name, meta in self._metadata.items()
        }

        gathered: list[dict | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_info, group=group)

        for rank_info in gathered:
            for name, info in rank_info.items():
                if name in self._metadata:
                    continue
                self._metadata[name] = _QuantMeta(
                    qformat=info["qformat"],
                    block_size=info["block_size"],
                    weight_amax=info["weight_amax"],
                )

    def _quantize_nvfp4(
        self,
        name: str,
        weight: torch.Tensor,
        meta: _QuantMeta,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Quantize one BF16/FP16 weight tensor into vLLM ModelOpt NVFP4 tensors."""
        if meta.weight_amax is None:
            raise RuntimeError(
                f"Missing ModelOpt weight amax for quantized parameter {name}"
            )

        weight_amax = meta.weight_amax.to(weight.device).float().abs()
        weight_scale_2 = weight_amax / _NVFP4_AMAX_DENOMINATOR
        weight_scale = _compute_nvfp4_weight_scale(
            weight,
            meta.block_size,
            weight_scale_2,
        )
        quantized = to_quantized_weight(
            weight,
            weight_scale,
            meta.qformat,
            weight_scale_2,
            meta.block_size,
        )

        yield name, quantized.detach()
        yield name.replace(".weight", ".weight_scale"), weight_scale.detach()
        yield name.replace(".weight", ".weight_scale_2"), weight_scale_2.detach()


def _iter_hf_to_megatron_matches(registry, hf_name: str):
    """Yield resolved bridge mappings whose HF pattern matches ``hf_name``."""
    for pattern_info, mapping in registry._reverse_patterns:
        if isinstance(mapping.hf_param, str):
            pattern = pattern_info
            if pattern is None:
                if mapping.hf_param == hf_name:
                    yield mapping
            else:
                match = pattern.match(hf_name)
                if match:
                    yield mapping.resolve(match.groups())
        else:
            patterns_dict = pattern_info
            for key, pattern in patterns_dict.items():
                if pattern is None:
                    if mapping.hf_param[key] == hf_name:
                        yield mapping.resolve(())
                else:
                    match = pattern.match(hf_name)
                    if match:
                        yield mapping.resolve(match.groups())


def _compute_nvfp4_weight_scale(
    weight: torch.Tensor,
    block_size: int,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        block_size,
        weights_scaling_factor_2=weight_scale_2.to(weight.device),
        keep_high_precision=True,
    )[0]
    weight_scale = weight_scale.to(torch.float32).abs()
    weight_scale[weight_scale == 0] = 1.0
    return weight_scale.to(torch.float8_e4m3fn)
