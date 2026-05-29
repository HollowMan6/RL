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

import contextlib
import itertools
import json
import os
import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from nemo_rl.utils.packed_tensor import (
    get_num_buffers,
    get_target_packed_tensor_size,
)

G_PAYLOAD_ALIGNMENT_BYTES = 8
G_PACKED_INDICES_NAME = "__packed_indices__"
G_PACKED_VALUES_NAME = "__packed_values__"
G_INDEX_START_KEY = "index_start"
G_INDEX_END_KEY = "index_end"
G_DEFAULT_SPARSE_ENCODE_COALESCE_BYTES = 256 * 1024**2
G_DEFAULT_BASELINE_PREWARM_CHUNK_BYTES = 256 * 1024**2
G_DEFAULT_BASELINE_STAGE_COALESCE_BYTES = 256 * 1024**2
G_BASELINE_STAGE_FREE_MEMORY_FRACTION = 0.125

DeltaCompressionTransport = Literal["dense", "sparse_indices"]
WeightTransferKind = Literal["full", "delta", "done"]

G_DENSE_TRANSPORT: DeltaCompressionTransport = "dense"
G_SPARSE_INDICES_TRANSPORT: DeltaCompressionTransport = "sparse_indices"
G_FULL_UPDATE_KIND: WeightTransferKind = "full"
G_DELTA_UPDATE_KIND: WeightTransferKind = "delta"
G_TRANSFER_DONE_KIND: WeightTransferKind = "done"

G_FLOAT_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

G_TENSOR_DTYPE_MAP = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.float64,
}

for _float8_dtype_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    _float8_dtype = getattr(torch, _float8_dtype_name, None)
    if _float8_dtype is not None:
        G_TENSOR_DTYPE_MAP[_float8_dtype_name] = _float8_dtype
del _float8_dtype, _float8_dtype_name

NamedTensor = tuple[str, torch.Tensor]
TensorBatch = list[NamedTensor]
WeightLoadFunc = Callable[[TensorBatch], None]
TensorPayload = tuple[TensorBatch, DeltaCompressionTransport, list[dict[str, Any]]]
PayloadEvents = tuple[torch.cuda.Event, ...]
QueuedPayload = tuple[WeightTransferKind, TensorPayload, PayloadEvents]
SparseBucketPayload = tuple[TensorPayload, PayloadEvents]
HeaderRefs = tuple[torch.Tensor, torch.Tensor | None]
TensorMetadata = Mapping[str, tuple[Iterable[int], torch.dtype]]

G_REFIT_SPARSE_ENCODE_COALESCE_BYTES_ENV = "NRL_REFIT_SPARSE_ENCODE_COALESCE_BYTES"
G_REFIT_PREWARM_DELTA_BASELINE_ENV = "NRL_REFIT_PREWARM_DELTA_BASELINE"
G_REFIT_BASELINE_PREWARM_CHUNK_BYTES_ENV = "NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES"
G_REFIT_BASELINE_STAGE_COALESCE_BYTES_ENV = "NRL_REFIT_BASELINE_STAGE_COALESCE_BYTES"

G_HEADER_KIND_TO_CODE = {
    G_TRANSFER_DONE_KIND: 0,
    G_FULL_UPDATE_KIND: 1,
    G_DELTA_UPDATE_KIND: 2,
}
G_HEADER_CODE_TO_KIND = {code: kind for kind, code in G_HEADER_KIND_TO_CODE.items()}
G_HEADER_TRANSPORT_TO_CODE = {
    G_DENSE_TRANSPORT: 0,
    G_SPARSE_INDICES_TRANSPORT: 1,
}
G_HEADER_CODE_TO_TRANSPORT = {
    code: transport for transport, code in G_HEADER_TRANSPORT_TO_CODE.items()
}


@dataclass(frozen=True)
class _SparseTensorInfo:
    name: str
    tensor: torch.Tensor
    flat: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.flat.numel())

    @property
    def byte_size(self) -> int:
        return int(self.flat.numel() * self.flat.element_size())


@dataclass(frozen=True)
class _BaselineEntry:
    arena: torch.Tensor
    offset: int
    numel: int


class _SparseEncodingDenseFallback(Exception):
    """Signal sparse encoding should fall back to a dense transfer."""


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Expected integer value for {name}.") from None


def _metadata_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _baseline_prewarm_enabled() -> bool:
    return _env_flag(
        G_REFIT_PREWARM_DELTA_BASELINE_ENV,
        default=True,
    )


def _baseline_prewarm_chunk_bytes() -> int:
    default = (
        get_target_packed_tensor_size()
        if torch.cuda.is_available()
        else G_DEFAULT_BASELINE_PREWARM_CHUNK_BYTES
    )
    return _env_int(
        G_REFIT_BASELINE_PREWARM_CHUNK_BYTES_ENV,
        default=default,
    )


def _baseline_stage_coalesce_bytes() -> int:
    return _env_int(
        G_REFIT_BASELINE_STAGE_COALESCE_BYTES_ENV,
        default=G_DEFAULT_BASELINE_STAGE_COALESCE_BYTES,
    )


def _memory_limited_stage_bytes(
    device: torch.device | int | str | None,
    requested_bytes: int,
) -> int:
    if requested_bytes <= 0 or not torch.cuda.is_available():
        return requested_bytes

    normalized_device = _normalize_device(device)
    if normalized_device is not None and normalized_device.type != "cuda":
        return requested_bytes

    free_bytes, _ = torch.cuda.mem_get_info(normalized_device)
    free_memory_cap = int(free_bytes * G_BASELINE_STAGE_FREE_MEMORY_FRACTION)
    return min(requested_bytes, free_memory_cap)


class DeltaCompressionTracker:
    """Tracks source-rank full baselines and prepares additive deltas."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        dtype_name = config["dtype"]
        if dtype_name not in G_FLOAT_DTYPE_MAP:
            raise ValueError(
                f"Unsupported delta compression dtype {dtype_name!r}; "
                f"expected one of {sorted(G_FLOAT_DTYPE_MAP)}."
            )
        self.delta_dtype = G_FLOAT_DTYPE_MAP[dtype_name]
        self.full_sync_interval = int(config["full_sync_interval"])
        self.sparse_bucket_size_bytes = int(config["sparse_bucket_size_bytes"])
        if self.full_sync_interval < 1:
            raise ValueError("delta_compression.full_sync_interval must be >= 1")
        if self.sparse_bucket_size_bytes < 1:
            raise ValueError("delta_compression.sparse_bucket_size_bytes must be >= 1")

        self.baseline: dict[str, torch.Tensor] = {}
        self._baseline_entries: dict[str, _BaselineEntry] = {}
        self.committed_syncs = 0
        self._d2h_stream: torch.cuda.Stream | None = None
        self._baseline_ready_events: dict[str, torch.cuda.Event] = {}

    def should_prewarm_baseline(self) -> bool:
        return (
            self.full_sync_interval > 1
            and self.committed_syncs == 0
            and _baseline_prewarm_enabled()
        )

    def prewarm_baseline_from_metadata(
        self,
        metadata: TensorMetadata,
    ) -> None:
        """Allocate baseline storage before the first full refit snapshot."""
        if not self.should_prewarm_baseline():
            return

        chunk_cap = _baseline_prewarm_chunk_bytes()
        if chunk_cap < 1:
            raise ValueError("baseline prewarm chunk bytes must be >= 1")

        pending: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        pending_bytes = 0

        def flush_pending() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            self._allocate_baseline_views(pending)
            pending = []
            pending_bytes = 0

        for name, (shape, dtype) in metadata.items():
            shape_tuple = tuple(int(dim) for dim in shape)
            if self._has_matching_baseline(name, shape_tuple, dtype):
                continue
            if name in self._baseline_ready_events:
                self.flush_baseline([name])

            tensor_bytes = _metadata_numel(shape_tuple) * _dtype_itemsize(dtype)
            if pending and (
                pending[0][2] != dtype or pending_bytes + tensor_bytes > chunk_cap
            ):
                flush_pending()
            pending.append((name, shape_tuple, dtype))
            pending_bytes += tensor_bytes
            if pending_bytes >= chunk_cap:
                flush_pending()

        flush_pending()

    def prepare_chunk(
        self,
        tensors: TensorBatch,
    ) -> tuple[bool, TensorBatch]:
        if (
            self.committed_syncs == 0
            or self.committed_syncs % self.full_sync_interval == 0
        ):
            if self.full_sync_interval > 1:
                self._snapshot_baseline(tensors)
            return False, tensors

        # Keep this order: gate on prior D2H baseline writes, read the old
        # baseline, then snapshot the new weights for the next successful sync.
        self.flush_baseline(name for name, _ in tensors)
        has_non_float = any(not tensor.dtype.is_floating_point for _, tensor in tensors)
        if has_non_float:
            self._snapshot_baseline(tensors)
            return False, tensors

        deltas = self._make_delta_tensors(tensors)

        self._snapshot_baseline(tensors)
        return True, deltas

    def on_sync_succeeded(self) -> None:
        # The baseline snapshot reads from caller-owned model tensors. Finish
        # those reads before the caller can offload or update the weights.
        self.flush_baseline()
        self.committed_syncs += 1

    def on_sync_failed(self) -> None:
        self.flush_baseline()
        self.committed_syncs = 0

    def flush_baseline(self, names: Iterable[str] | None = None) -> None:
        if not self._baseline_ready_events:
            return
        d2h_stream = self._d2h_stream
        assert d2h_stream is not None
        if names is None:
            d2h_stream.synchronize()
            self._baseline_ready_events.clear()
            return

        names_to_wait = self._baseline_ready_events.keys() & set(names)
        current_stream = torch.cuda.current_stream()
        seen_events = set()
        for name in names_to_wait:
            event = self._baseline_ready_events.pop(name)
            if id(event) in seen_events:
                continue
            current_stream.wait_event(event)
            seen_events.add(id(event))

    def _snapshot_baseline(
        self,
        tensors: TensorBatch,
    ) -> None:
        if tensors[0][1].is_cuda and torch.cuda.is_available():
            self._snapshot_cuda_baseline(tensors)
            return
        for name, tensor in tensors:
            baseline = self.baseline.get(name)
            if (
                baseline is None
                or baseline.shape != tensor.shape
                or baseline.dtype != tensor.dtype
            ):
                baseline = torch.empty_like(tensor, device=torch.device("cpu"))
                self.baseline[name] = baseline
                self._baseline_entries[name] = _BaselineEntry(
                    arena=baseline.view(-1),
                    offset=0,
                    numel=baseline.numel(),
                )
            baseline.copy_(tensor.detach())

    def _snapshot_cuda_baseline(
        self,
        tensors: TensorBatch,
    ) -> None:
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream()
        self._ensure_cuda_baseline_buffers(tensors)
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self._d2h_stream):
            self._d2h_stream.wait_event(event)
            self._snapshot_cuda_baseline_to_host(tensors)
            ready_event = self._d2h_stream.record_event()
        for name, _ in tensors:
            self._baseline_ready_events[name] = ready_event

    def _make_delta_tensors(self, tensors: TensorBatch) -> TensorBatch:
        if tensors[0][1].is_cuda and torch.cuda.is_available():
            return self._make_cuda_delta_tensors(tensors)
        return self._make_per_tensor_deltas(tensors)

    def _make_per_tensor_deltas(self, tensors: TensorBatch) -> TensorBatch:
        deltas = []
        for name, tensor in tensors:
            if name not in self.baseline:
                raise KeyError(
                    f"Delta baseline is missing tensor {name!r}; run a full sync "
                    "before delta sync resumes."
                )
            delta = self.baseline[name].to(
                device=tensor.device,
                dtype=self.delta_dtype,
                non_blocking=tensor.is_cuda,
                copy=True,
            )
            torch.sub(tensor, delta, out=delta)
            deltas.append((name, delta))
        return deltas

    def _make_cuda_delta_tensors(self, tensors: TensorBatch) -> TensorBatch:
        if any(name not in self._baseline_entries for name, _ in tensors):
            return self._make_per_tensor_deltas(tensors)

        delta_by_name: dict[str, torch.Tensor] = {}
        max_stage_bytes = _memory_limited_stage_bytes(
            tensors[0][1].device,
            _baseline_stage_coalesce_bytes(),
        )

        def make_span_deltas(
            span: list[tuple[str, torch.Tensor, _BaselineEntry]],
            start: int,
            length: int,
        ) -> None:
            _, first_tensor, first_entry = span[0]
            source = first_entry.arena.narrow(0, start, length)
            try:
                delta_arena = source.to(
                    device=first_tensor.device,
                    dtype=self.delta_dtype,
                    non_blocking=True,
                    copy=True,
                )
            except torch.OutOfMemoryError:
                if len(span) == 1:
                    raise
                midpoint = len(span) // 2
                left = span[:midpoint]
                right = span[midpoint:]
                left_start, left_length = self._baseline_span_bounds(left)
                right_start, right_length = self._baseline_span_bounds(right)
                make_span_deltas(left, left_start, left_length)
                make_span_deltas(right, right_start, right_length)
                return

            for name, tensor, entry in span:
                delta = delta_arena[
                    entry.offset - start : entry.offset - start + entry.numel
                ].view(tensor.shape)
                torch.sub(tensor, delta, out=delta)
                delta_by_name[name] = delta

        for span, start, length in self._iter_baseline_spans(
            tensors,
            itemsize=_dtype_itemsize(self.delta_dtype),
            max_bytes=max_stage_bytes,
        ):
            make_span_deltas(span, start, length)

        return [(name, delta_by_name[name]) for name, _ in tensors]

    def _snapshot_cuda_baseline_to_host(
        self,
        tensors: TensorBatch,
    ) -> None:
        if any(name not in self._baseline_entries for name, _ in tensors):
            for name, tensor in tensors:
                baseline = self.baseline[name]
                baseline.copy_(tensor.detach(), non_blocking=True)
            return

        staging_buffers: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        max_stage_bytes = _memory_limited_stage_bytes(
            tensors[0][1].device,
            _baseline_stage_coalesce_bytes(),
        )

        def copy_tensor_to_host(name: str, tensor: torch.Tensor) -> None:
            baseline = self.baseline[name]
            baseline.copy_(tensor.detach(), non_blocking=True)

        def get_staging_buffer(
            tensor: torch.Tensor,
            length: int,
        ) -> torch.Tensor | None:
            key = (tensor.device, tensor.dtype)
            buffer = staging_buffers.get(key)
            if buffer is not None and buffer.numel() >= length:
                return buffer.narrow(0, 0, length)

            try:
                buffer = torch.empty(length, dtype=tensor.dtype, device=tensor.device)
            except torch.OutOfMemoryError:
                return None

            staging_buffers[key] = buffer
            return buffer

        def copy_span_to_host(
            span: list[tuple[str, torch.Tensor, _BaselineEntry]],
            start: int,
            length: int,
        ) -> None:
            if len(span) == 1:
                name, tensor, _ = span[0]
                copy_tensor_to_host(name, tensor)
                return

            staging = get_staging_buffer(span[0][1], length)
            if staging is None:
                for name, tensor, _ in span:
                    copy_tensor_to_host(name, tensor)
                return

            staging_offset = 0
            for _, tensor, entry in span:
                staging.narrow(0, staging_offset, entry.numel).view(tensor.shape).copy_(
                    tensor.detach(), non_blocking=True
                )
                staging_offset += entry.numel

            first_entry = span[0][2]
            staging.record_stream(torch.cuda.current_stream())
            first_entry.arena.narrow(0, start, length).copy_(
                staging,
                non_blocking=True,
            )

        for span, start, length in self._iter_baseline_spans(
            tensors,
            itemsize=None,
            max_bytes=max_stage_bytes,
        ):
            copy_span_to_host(span, start, length)

    @staticmethod
    def _baseline_span_bounds(
        span: list[tuple[str, torch.Tensor, _BaselineEntry]],
    ) -> tuple[int, int]:
        start = span[0][2].offset
        end = span[-1][2].offset + span[-1][2].numel
        return start, end - start

    def _iter_baseline_spans(
        self,
        tensors: TensorBatch,
        *,
        itemsize: int | None,
        max_bytes: int,
    ) -> Iterator[tuple[list[tuple[str, torch.Tensor, _BaselineEntry]], int, int]]:
        grouped: dict[
            tuple[int, torch.device],
            list[tuple[str, torch.Tensor, _BaselineEntry]],
        ] = {}
        for name, tensor in tensors:
            entry = self._baseline_entries[name]
            key = (id(entry.arena), tensor.device)
            grouped.setdefault(key, []).append((name, tensor, entry))

        for items in grouped.values():
            items.sort(key=lambda item: item[2].offset)
            span: list[tuple[str, torch.Tensor, _BaselineEntry]] = []
            span_start = 0
            span_end = 0
            span_bytes = 0
            for item in items:
                entry = item[2]
                entry_itemsize = (
                    itemsize if itemsize is not None else entry.arena.element_size()
                )
                item_bytes = entry.numel * entry_itemsize
                can_merge = (
                    max_bytes > 0
                    and span
                    and entry.offset == span_end
                    and span_bytes + item_bytes <= max_bytes
                )
                if not span or can_merge:
                    if not span:
                        span_start = entry.offset
                        span_end = entry.offset + entry.numel
                        span_bytes = item_bytes
                    else:
                        span_end = entry.offset + entry.numel
                        span_bytes += item_bytes
                    span.append(item)
                    continue

                yield span, span_start, span_end - span_start
                span = [item]
                span_start = entry.offset
                span_end = entry.offset + entry.numel
                span_bytes = item_bytes

            if span:
                yield span, span_start, span_end - span_start

    def _ensure_cuda_baseline_buffers(
        self,
        tensors: TensorBatch,
    ) -> None:
        missing_by_dtype: dict[torch.dtype, TensorBatch] = {}
        for name, tensor in tensors:
            if self._has_matching_baseline(name, tuple(tensor.shape), tensor.dtype):
                continue
            if name in self._baseline_ready_events:
                self.flush_baseline([name])
            missing_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

        for dtype, missing_tensors in missing_by_dtype.items():
            self._allocate_baseline_views(
                [
                    (name, tuple(tensor.shape), dtype)
                    for name, tensor in missing_tensors
                ],
            )

    def _has_matching_baseline(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> bool:
        baseline = self.baseline.get(name)
        return (
            baseline is not None
            and tuple(baseline.shape) == shape
            and baseline.dtype == dtype
        )

    def _allocate_baseline_views(
        self,
        items: list[tuple[str, tuple[int, ...], torch.dtype]],
    ) -> None:
        if not items:
            return
        dtype = items[0][2]
        if any(item_dtype != dtype for _, _, item_dtype in items):
            raise ValueError("baseline allocation items must have the same dtype")
        total_numel = sum(_metadata_numel(shape) for _, shape, _ in items)
        baseline_arena = torch.empty(
            total_numel,
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=torch.cuda.is_available(),
        )
        offset = 0
        for name, shape, _ in items:
            numel = _metadata_numel(shape)
            self.baseline[name] = baseline_arena[offset : offset + numel].view(shape)
            self._baseline_entries[name] = _BaselineEntry(
                arena=baseline_arena,
                offset=offset,
                numel=numel,
            )
            offset += numel


def create_vllm_delta_transfer_tracker(
    generation_config: Mapping[str, Any] | None,
) -> DeltaCompressionTracker | None:
    """Create a vLLM delta transfer tracker when config enables it."""
    if generation_config is None:
        return None
    delta_config = generation_config.get("delta_compression")
    if delta_config is None or not delta_config["enabled"]:
        return None
    if generation_config["backend"] != "vllm":
        raise ValueError("Delta compression is currently supported only for vLLM.")
    if generation_config["colocated"]["enabled"]:
        raise ValueError(
            "Delta compression is supported only for non-colocated vLLM refit."
        )
    if generation_config.get("quant_cfg") is not None:
        raise NotImplementedError(
            "Delta compression for vLLM ModelOpt quantized weights is not implemented."
        )
    if generation_config["vllm_cfg"].get("precision") == "fp8":
        raise NotImplementedError(
            "Delta compression for vLLM FP8 model weights is not implemented."
        )
    return DeltaCompressionTracker(delta_config)


def packed_weight_transfer_producer(
    iterator: Iterable[NamedTensor],
    *,
    group: Any,
    src: int,
    delta_tracker: DeltaCompressionTracker | None = None,
) -> None:
    """Broadcast full or delta weight chunks with explicit chunk metadata."""
    encode_streams = _cuda_streams()
    broadcast_streams = _cuda_streams()
    buffer_idx = 0
    transfer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header_refs: list[HeaderRefs | None] = [None for _ in broadcast_streams]
    payload_refs: list[torch.Tensor | None] = [None for _ in broadcast_streams]
    # The source reads one chunk ahead before sending the current payload. Peer
    # ranks must advance the same amount because Megatron export can issue
    # collectives while yielding source-rank tensors.
    source_lookahead_chunks = 2

    if group.rank != src:
        pending_item = None
        iterator_exhausted = False
        tensor_iterator = iter(iterator)

        def prefetch_non_source_chunk() -> None:
            nonlocal pending_item, iterator_exhausted
            if iterator_exhausted:
                return
            pending_item, iterator_exhausted = _advance_chunk(
                tensor_iterator,
                get_target_packed_tensor_size(),
                pending_item=pending_item,
            )

        for _ in range(source_lookahead_chunks):
            prefetch_non_source_chunk()

        while True:
            with _use_stream(broadcast_streams, buffer_idx):
                header, header_ref = _broadcast_header(
                    {},
                    group=group,
                    src=src,
                    device=transfer_device,
                )
                header_refs[buffer_idx] = header_ref
                _record_header_stream(header_ref)
                if header["kind"] == G_TRANSFER_DONE_KIND:
                    break
                payload = _recv_payload(
                    int(header["payload_numel"]),
                    group=group,
                    src=src,
                    device=transfer_device,
                )
                payload_refs[buffer_idx] = payload
                _record_tensor_stream(payload)
            prefetch_non_source_chunk()
            buffer_idx = (buffer_idx + 1) % len(broadcast_streams)
        _sync_streams(broadcast_streams)
        _synchronize_current_transfer_stream(transfer_device)
        return

    target_chunk_size = get_target_packed_tensor_size()
    tensor_iterator = iter(iterator)
    pending_item = None
    queued: list[QueuedPayload] = []
    sparse_bucket: list[SparseBucketPayload] = []
    sparse_bucket_bytes = 0
    sparse_bucket_cap = (
        delta_tracker.sparse_bucket_size_bytes if delta_tracker is not None else 0
    )

    def queue_empty_delta_payload() -> None:
        queued.append(
            (
                G_DELTA_UPDATE_KIND,
                ([], G_SPARSE_INDICES_TRANSPORT, []),
                (),
            )
        )

    def flush_sparse_bucket() -> None:
        nonlocal sparse_bucket, sparse_bucket_bytes
        if sparse_bucket:
            payloads = [payload for payload, _ in sparse_bucket]
            ready_events = tuple(
                event for _, events in sparse_bucket for event in events
            )
            _wait_for_payload_events(ready_events)
            queued.append(
                (
                    G_DELTA_UPDATE_KIND,
                    payloads[0]
                    if len(payloads) == 1
                    else _merge_sparse_payloads(payloads),
                    _record_payload_readiness_events(),
                )
            )
            sparse_bucket = []
            sparse_bucket_bytes = 0

    def queue_payload(kind: WeightTransferKind, payload: TensorPayload) -> None:
        nonlocal sparse_bucket_bytes
        tensors, _, metadata = payload
        if kind != G_DELTA_UPDATE_KIND:
            flush_sparse_bucket()
            queued.append((kind, payload, _record_payload_readiness_events()))
            return

        payload_bytes = _wire_bytes(tensors)
        if sparse_bucket and sparse_bucket_bytes + payload_bytes > sparse_bucket_cap:
            flush_sparse_bucket()
        if metadata:
            sparse_bucket.append((payload, _record_payload_readiness_events()))
            sparse_bucket_bytes += payload_bytes
        if sparse_bucket_bytes >= sparse_bucket_cap:
            flush_sparse_bucket()

    def queue_chunk(chunk: TensorBatch) -> None:
        if delta_tracker is None:
            queue_payload(G_FULL_UPDATE_KIND, (chunk, G_DENSE_TRANSPORT, []))
            return

        queued_before = len(queued)
        is_delta, tensors = delta_tracker.prepare_chunk(chunk)
        if not is_delta:
            queue_payload(G_FULL_UPDATE_KIND, (tensors, G_DENSE_TRANSPORT, []))
            return

        try:
            payload = _encode_sparse_indices(tensors)
        except _SparseEncodingDenseFallback:
            queue_payload(G_FULL_UPDATE_KIND, (chunk, G_DENSE_TRANSPORT, []))
            return
        if payload[2] and _wire_bytes(payload[0]) >= _wire_bytes(chunk):
            queue_payload(G_FULL_UPDATE_KIND, (chunk, G_DENSE_TRANSPORT, []))
        else:
            queue_payload(G_DELTA_UPDATE_KIND, payload)
        if len(queued) == queued_before:
            # Keep one control message per source chunk so non-source ranks can
            # advance Megatron export collectives in lockstep with source lookahead.
            queue_empty_delta_payload()

    def pop_wire_chunk() -> tuple[dict[str, Any], torch.Tensor]:
        kind, payload, ready_events = queued.pop(0)
        _wait_for_payload_events(ready_events)
        tensors, transport, metadata = payload
        if not tensors:
            header = {
                "kind": kind,
                "transport": transport,
                "payload_entries": [],
                "payload_numel": 0,
                "sparse_metadata": metadata,
            }
            return header, torch.empty(0, dtype=torch.uint8, device=transfer_device)
        packed, entries = pack_named_tensors(tensors)
        header = {
            "kind": kind,
            "transport": transport,
            "payload_entries": entries,
            "payload_numel": int(packed.numel()),
            "sparse_metadata": metadata,
        }
        return header, packed

    def send(header: Mapping[str, Any], payload: torch.Tensor, event) -> None:
        nonlocal buffer_idx
        with _use_stream(broadcast_streams, buffer_idx):
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            _, header_ref = _broadcast_header(
                header,
                group=group,
                src=src,
                device=payload.device,
            )
            header_refs[buffer_idx] = header_ref
            _record_header_stream(header_ref)
            payload_refs[buffer_idx] = payload
            if payload.numel() > 0:
                group.broadcast(payload, src=src)
                _record_tensor_stream(payload)
        buffer_idx = (buffer_idx + 1) % len(broadcast_streams)

    read_idx = 0
    encode_idx = 0
    pack_idx = 0

    def read_next_chunk() -> TensorBatch:
        nonlocal pending_item, read_idx
        with _use_stream(encode_streams, read_idx):
            chunk, pending_item = _next_chunk(
                tensor_iterator,
                target_chunk_size,
                pending_item=pending_item,
            )
        read_idx = (read_idx + 1) % len(encode_streams)
        return chunk

    def prepare_chunk_payload(
        chunk: TensorBatch,
    ) -> tuple[
        dict[str, Any],
        torch.Tensor,
        torch.cuda.Event | None,
    ]:
        nonlocal encode_idx, pack_idx
        with _use_stream(encode_streams, encode_idx):
            queue_chunk(chunk)
        encode_idx = (encode_idx + 1) % len(encode_streams)
        with _use_stream(encode_streams, pack_idx):
            header, payload = pop_wire_chunk()
            event = _record_stream_event(encode_streams[pack_idx])
        pack_idx = (pack_idx + 1) % len(encode_streams)
        return header, payload, event

    try:
        chunk = read_next_chunk()
        while chunk:
            header, payload, event = prepare_chunk_payload(chunk)
            next_chunk = read_next_chunk()
            send(header, payload, event)
            chunk = next_chunk

        flush_sparse_bucket()
        while queued:
            with _use_stream(encode_streams, pack_idx):
                header, payload = pop_wire_chunk()
                event = _record_stream_event(encode_streams[pack_idx])
            pack_idx = (pack_idx + 1) % len(encode_streams)
            send(header, payload, event)

        _sync_streams(encode_streams)
        _sync_streams(broadcast_streams)
        _, header_refs[0] = _broadcast_header(
            {"kind": G_TRANSFER_DONE_KIND},
            group=group,
            src=src,
            device=transfer_device,
        )
        _synchronize_current_transfer_stream(transfer_device)
    except Exception:
        if delta_tracker is not None:
            delta_tracker.on_sync_failed()
        raise

    if delta_tracker is not None:
        delta_tracker.on_sync_succeeded()


def packed_weight_transfer_consumer(
    *,
    group: Any,
    src: int,
    load_full_weights_func: WeightLoadFunc,
    load_delta_weights_func: WeightLoadFunc,
    device: torch.device | int | str,
    delta_load_batch_size_bytes: int | None = None,
) -> None:
    """Receive full or delta chunks from ``packed_weight_transfer_producer``."""
    if delta_load_batch_size_bytes is not None and delta_load_batch_size_bytes < 1:
        raise ValueError("delta_load_batch_size_bytes must be >= 1 when set.")

    streams = _cuda_streams(device)
    buffer_idx = 0
    header_refs: list[HeaderRefs | None] = [None for _ in streams]
    payload_refs: list[torch.Tensor | None] = [None for _ in streams]
    load_queue: _AsyncWeightLoadQueue | None = None
    decode_queue: _AsyncSparseDecodeQueue | None = None
    if delta_load_batch_size_bytes is not None:
        load_queue = _AsyncWeightLoadQueue(
            device=device,
            max_pending_batches=len(streams),
        )
        decode_queue = _AsyncSparseDecodeQueue(
            device=device,
            byte_cap=delta_load_batch_size_bytes,
            load_queue=load_queue,
            load_delta_weights_func=load_delta_weights_func,
            max_pending_payloads=len(streams),
        )

    transfer_done = False
    try:
        while True:
            with _use_stream(streams, buffer_idx):
                header, header_ref = _broadcast_header(
                    {},
                    group=group,
                    src=src,
                    device=device,
                )
                header_refs[buffer_idx] = header_ref
                _record_header_stream(header_ref)
                if header["kind"] == G_TRANSFER_DONE_KIND:
                    transfer_done = True
                    break

                payload_numel = int(header["payload_numel"])
                payload_tensors: TensorBatch = []
                if payload_numel > 0:
                    payload = _recv_payload(
                        payload_numel,
                        group=group,
                        src=src,
                        device=device,
                    )
                    payload_refs[buffer_idx] = payload
                    _record_tensor_stream(payload)
                    payload_tensors = unpack_named_tensors(
                        payload,
                        entries=header["payload_entries"],
                    )
                else:
                    payload_refs[buffer_idx] = None

                if header["kind"] == G_FULL_UPDATE_KIND:
                    if load_queue is None:
                        load_full_weights_func(payload_tensors)
                    else:
                        decode_queue.flush_pending()
                        event = _record_stream_event(streams[buffer_idx])
                        load_queue.enqueue(
                            load_full_weights_func,
                            payload_tensors,
                            ready_events=[] if event is None else [event],
                        )
                elif payload_numel > 0:
                    event = _record_stream_event(streams[buffer_idx])
                    decode_queue.enqueue(
                        payload_tensors,
                        header["sparse_metadata"],
                        ready_events=[] if event is None else [event],
                    )

            if decode_queue is not None:
                decode_queue.raise_if_failed()
            if load_queue is not None:
                load_queue.raise_if_failed()
            buffer_idx = (buffer_idx + 1) % len(streams)
    finally:
        if decode_queue is not None:
            decode_queue.close()
        if transfer_done:
            _sync_streams(streams)
        if load_queue is not None:
            load_queue.close()


class _AsyncWeightLoadQueue:
    def __init__(
        self,
        *,
        device: torch.device | int | str,
        max_pending_batches: int,
    ) -> None:
        self._device = device
        self._requests: queue.Queue[
            tuple[WeightLoadFunc, TensorBatch, list[torch.cuda.Event]] | None
        ] = queue.Queue(maxsize=max_pending_batches)
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-weight-load",
            daemon=True,
        )
        self._thread.start()

    def enqueue(
        self,
        load_func: WeightLoadFunc,
        batch: TensorBatch,
        *,
        ready_events: list[torch.cuda.Event],
    ) -> None:
        if batch:
            self.raise_if_failed()
            self._requests.put((load_func, batch, ready_events))
            self.raise_if_failed()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        with _cuda_device(_normalize_device(self._device)):
            while True:
                request = self._requests.get()
                try:
                    if request is None:
                        return
                    if self._error is None:
                        load_func, batch, events = request
                        for event in events:
                            torch.cuda.current_stream().wait_event(event)
                        load_func(batch)
                        _synchronize_current_transfer_stream(self._device)
                except Exception as error:
                    self._error = error
                finally:
                    self._requests.task_done()


class _AsyncSparseDecodeQueue:
    def __init__(
        self,
        *,
        device: torch.device | int | str,
        byte_cap: int,
        load_queue: _AsyncWeightLoadQueue,
        load_delta_weights_func: WeightLoadFunc,
        max_pending_payloads: int,
    ) -> None:
        self._device = device
        self._byte_cap = byte_cap
        self._load_queue = load_queue
        self._load_delta_weights_func = load_delta_weights_func
        self._requests: queue.Queue[
            tuple[TensorBatch, list[dict[str, Any]], list[torch.cuda.Event]] | None
        ] = queue.Queue(maxsize=max_pending_payloads)
        self._delta_batch: TensorBatch = []
        self._delta_batch_bytes = 0
        self._delta_events: list[torch.cuda.Event] = []
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-sparse-decode",
            daemon=True,
        )
        self._thread.start()

    def enqueue(
        self,
        payload_tensors: TensorBatch,
        metadata: list[dict[str, Any]],
        *,
        ready_events: list[torch.cuda.Event],
    ) -> None:
        if metadata:
            self.raise_if_failed()
            self._requests.put((payload_tensors, metadata, ready_events))
            self.raise_if_failed()

    def flush_pending(self) -> None:
        self.raise_if_failed()
        self._requests.join()
        self.raise_if_failed()
        self._flush_delta_batch()
        self.raise_if_failed()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _flush_delta_batch(self) -> None:
        if not self._delta_batch:
            return
        self._load_queue.enqueue(
            self._load_delta_weights_func,
            self._delta_batch,
            ready_events=self._delta_events,
        )
        self._delta_batch = []
        self._delta_batch_bytes = 0
        self._delta_events = []

    def _add_delta_batch(
        self,
        batch: TensorBatch,
        stream: torch.cuda.Stream | None,
    ) -> None:
        if not batch:
            return
        batch_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in batch)
        if self._delta_batch and self._delta_batch_bytes + batch_bytes > self._byte_cap:
            self._flush_delta_batch()
        self._delta_batch.extend(batch)
        self._delta_batch_bytes += batch_bytes
        event = _record_stream_event(stream)
        if event is not None:
            self._delta_events.append(event)
        if self._delta_batch_bytes >= self._byte_cap:
            self._flush_delta_batch()

    def _run(self) -> None:
        with _cuda_device(_normalize_device(self._device)):
            streams = _cuda_streams(self._device)
            stream_idx = 0
            while True:
                request = self._requests.get()
                try:
                    if request is None:
                        if self._error is None:
                            self._flush_delta_batch()
                        return
                    if self._error is None:
                        payload_tensors, metadata, events = request
                        stream = streams[stream_idx]
                        with _use_stream(streams, stream_idx):
                            for event in events:
                                torch.cuda.current_stream().wait_event(event)
                            for _, tensor in payload_tensors:
                                _record_tensor_stream(tensor)
                            for batch in _decode_sparse(
                                payload_tensors,
                                metadata,
                                self._device,
                                self._byte_cap,
                            ):
                                self._add_delta_batch(batch, stream)
                        stream_idx = (stream_idx + 1) % len(streams)
                except Exception as error:
                    self._error = error
                finally:
                    self._requests.task_done()


def pack_named_tensors(tensors: TensorBatch) -> tuple[torch.Tensor, list[dict]]:
    """Pack tensors with mixed dtypes into one uint8 tensor."""
    chunks = []
    entries = []
    for name, tensor in tensors:
        tensor = tensor.contiguous()
        byte_view = tensor.view(torch.uint8).view(-1)
        byte_size = int(byte_view.numel())
        pad = (-byte_size) % G_PAYLOAD_ALIGNMENT_BYTES
        chunks.append(byte_view)
        if pad:
            chunks.append(torch.zeros(pad, dtype=torch.uint8, device=tensor.device))
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": _dtype_to_name(tensor.dtype),
                "byte_size": byte_size,
                "wire_byte_size": byte_size + pad,
            }
        )
    return torch.cat(chunks, dim=0), entries


def unpack_named_tensors(
    payload: torch.Tensor, entries: list[dict[str, Any]]
) -> TensorBatch:
    """Unpack a uint8 transfer payload according to header metadata."""
    byte_views = payload.split_with_sizes(
        [int(entry["wire_byte_size"]) for entry in entries]
    )
    return [
        (
            entry["name"],
            byte_view[: int(entry["byte_size"])]
            .view(_dtype_from_name(entry["dtype"]))
            .view(tuple(entry["shape"])),
        )
        for entry, byte_view in zip(entries, byte_views, strict=True)
    ]


@contextlib.contextmanager
def additive_weight_load_context(target_tensors: Iterable[torch.Tensor]):
    """Make weight loaders add into model tensors instead of overwriting them."""
    original_copy = torch.Tensor.copy_
    original_fill = torch.Tensor.fill_
    original_setitem = torch.Tensor.__setitem__
    target_storage_ptrs = {
        tensor.untyped_storage().data_ptr() for tensor in target_tensors
    }

    def should_add(tensor: torch.Tensor) -> bool:
        return (
            tensor.dtype.is_floating_point
            and tensor.untyped_storage().data_ptr() in target_storage_ptrs
        )

    def additive_copy(self, src, non_blocking=False):
        if should_add(self):
            self.add_(src.to(self.device, self.dtype, non_blocking=non_blocking))
            return self
        return original_copy(self, src, non_blocking=non_blocking)

    def additive_fill(self, value, *args, **kwargs):
        if should_add(self):
            self.add_(value)
            return self
        return original_fill(self, value, *args, **kwargs)

    def additive_setitem(self, index, value):
        destination = self[index]
        if should_add(destination):
            if isinstance(value, torch.Tensor):
                value = value.to(device=destination.device, dtype=destination.dtype)
            destination.add_(value)
            return
        return original_setitem(self, index, value)

    torch.Tensor.copy_ = cast(Any, additive_copy)
    torch.Tensor.fill_ = cast(Any, additive_fill)
    torch.Tensor.__setitem__ = cast(Any, additive_setitem)
    try:
        yield
    finally:
        torch.Tensor.copy_ = cast(Any, original_copy)
        torch.Tensor.fill_ = cast(Any, original_fill)
        torch.Tensor.__setitem__ = cast(Any, original_setitem)


def _encode_sparse_indices(tensors: TensorBatch) -> TensorPayload:
    packed_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    metadata = []
    packed_offset = 0
    value_offset = 0
    dense_wire_bytes = _wire_bytes(tensors)

    for group in _iter_sparse_index_groups(tensors):
        packed_offset, value_offset = _append_sparse_index_group_payload(
            group=group,
            packed_parts=packed_parts,
            value_parts=value_parts,
            metadata=metadata,
            packed_offset=packed_offset,
            value_offset=value_offset,
        )
        if (
            _sparse_payload_wire_bytes(
                packed_offset,
                value_offset,
                packed_dtype=torch.int32,
                value_dtype=tensors[0][1].dtype,
            )
            > dense_wire_bytes
        ):
            raise _SparseEncodingDenseFallback

    device = tensors[0][1].device
    payload_tensors = [
        (
            G_PACKED_INDICES_NAME,
            (
                torch.cat(packed_parts, dim=0)
                if packed_parts
                else torch.empty(0, dtype=torch.int32, device=device)
            ),
        ),
        (
            G_PACKED_VALUES_NAME,
            (
                torch.cat(value_parts, dim=0)
                if value_parts
                else torch.empty(0, dtype=tensors[0][1].dtype, device=device)
            ),
        ),
    ]
    return payload_tensors, G_SPARSE_INDICES_TRANSPORT, metadata


def _append_sparse_index_group_payload(
    *,
    group: list[_SparseTensorInfo],
    packed_parts: list[torch.Tensor],
    value_parts: list[torch.Tensor],
    metadata: list[dict[str, Any]],
    packed_offset: int,
    value_offset: int,
) -> tuple[int, int]:
    if len(group) == 1:
        info = group[0]
        try:
            locations, values = _sparse_indices_for_tensor(info.flat)
        except torch.OutOfMemoryError:
            raise _SparseEncodingDenseFallback from None
        if values.numel() == 0:
            return packed_offset, value_offset
        if _sparse_payload_exceeds_dense(
            int(locations.numel()),
            int(values.numel()),
            packed_dtype=torch.int32,
            value_dtype=info.tensor.dtype,
            dense_wire_bytes=_wire_bytes([(info.name, info.tensor)]),
        ):
            raise _SparseEncodingDenseFallback
        packed = locations.to(torch.int32)
        return _append_sparse_tensor_payload(
            info=info,
            packed=packed,
            values=values,
            packed_parts=packed_parts,
            value_parts=value_parts,
            metadata=metadata,
            packed_offset=packed_offset,
            value_offset=value_offset,
        )

    try:
        locations, values, counts = _sparse_indices_for_group(group)
    except torch.OutOfMemoryError:
        for info in group:
            packed_offset, value_offset = _append_sparse_index_group_payload(
                group=[info],
                packed_parts=packed_parts,
                value_parts=value_parts,
                metadata=metadata,
                packed_offset=packed_offset,
                value_offset=value_offset,
            )
        return packed_offset, value_offset

    if values.numel() == 0:
        return packed_offset, value_offset

    value_start = 0
    packed_locations = locations.to(torch.int32)
    for info, count in zip(group, counts, strict=True):
        if count == 0:
            continue
        value_end = value_start + count
        packed_offset, value_offset = _append_sparse_tensor_payload(
            info=info,
            packed=packed_locations[value_start:value_end],
            values=values[value_start:value_end],
            packed_parts=packed_parts,
            value_parts=value_parts,
            metadata=metadata,
            packed_offset=packed_offset,
            value_offset=value_offset,
        )
        value_start = value_end
    return packed_offset, value_offset


def _sparse_encode_coalesce_bytes() -> int:
    return _env_int(
        G_REFIT_SPARSE_ENCODE_COALESCE_BYTES_ENV,
        default=G_DEFAULT_SPARSE_ENCODE_COALESCE_BYTES,
    )


def _iter_sparse_index_groups(
    tensors: TensorBatch,
) -> Iterator[list[_SparseTensorInfo]]:
    coalesce_bytes = _sparse_encode_coalesce_bytes()
    current: list[_SparseTensorInfo] = []
    current_bytes = 0

    for name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        if flat.numel() > torch.iinfo(torch.int32).max:
            raise _SparseEncodingDenseFallback

        info = _SparseTensorInfo(name=name, tensor=tensor, flat=flat)
        can_coalesce = (
            coalesce_bytes > 0
            and current
            and current[0].flat.dtype == flat.dtype
            and current[0].flat.device == flat.device
            and current_bytes + info.byte_size <= coalesce_bytes
        )
        if current and not can_coalesce:
            yield current
            current = []
            current_bytes = 0

        current.append(info)
        current_bytes += info.byte_size

    if current:
        yield current


def _sparse_indices_for_tensor(
    flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    locations = torch.nonzero(flat, as_tuple=True)[0]
    values = flat[locations]
    return locations, values


def _sparse_indices_for_group(
    group: list[_SparseTensorInfo],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    flat = _alias_sparse_index_group(group)
    if flat is None:
        flat = torch.cat([info.flat for info in group], dim=0)
    global_locations, values = _sparse_indices_for_tensor(flat)
    if global_locations.numel() == 0:
        return global_locations, values, [0 for _ in group]
    if _sparse_payload_exceeds_dense(
        int(global_locations.numel()),
        int(values.numel()),
        packed_dtype=torch.int32,
        value_dtype=flat.dtype,
        dense_wire_bytes=_wire_bytes([(info.name, info.tensor) for info in group]),
    ):
        raise _SparseEncodingDenseFallback

    offsets = list(itertools.accumulate(info.numel for info in group))
    boundaries = torch.tensor(
        offsets[:-1],
        dtype=global_locations.dtype,
        device=global_locations.device,
    )
    segment_ids = torch.bucketize(global_locations, boundaries, right=True)
    starts = torch.tensor(
        [0, *offsets[:-1]],
        dtype=global_locations.dtype,
        device=global_locations.device,
    )
    locations = global_locations - starts[segment_ids]
    counts = torch.bincount(segment_ids, minlength=len(group)).cpu().tolist()
    return locations, values, [int(count) for count in counts]


def _alias_sparse_index_group(group: list[_SparseTensorInfo]) -> torch.Tensor | None:
    if len(group) < 2:
        return None

    first = group[0].flat
    if not first.is_contiguous():
        return None

    storage_ptr = first.untyped_storage().data_ptr()
    expected_offset = first.storage_offset()
    total_numel = 0
    for info in group:
        flat = info.flat
        if (
            not flat.is_contiguous()
            or flat.dtype != first.dtype
            or flat.device != first.device
            or flat.untyped_storage().data_ptr() != storage_ptr
            or flat.storage_offset() != expected_offset
        ):
            return None
        expected_offset += flat.numel()
        total_numel += flat.numel()

    return torch.as_strided(
        first,
        (total_numel,),
        (1,),
        storage_offset=first.storage_offset(),
    )


def _append_sparse_tensor_payload(
    *,
    info: _SparseTensorInfo,
    packed: torch.Tensor,
    values: torch.Tensor,
    packed_parts: list[torch.Tensor],
    value_parts: list[torch.Tensor],
    metadata: list[dict[str, Any]],
    packed_offset: int,
    value_offset: int,
) -> tuple[int, int]:
    nnz = int(values.numel())
    metadata.append(
        {
            "name": info.name,
            "dtype": _dtype_to_name(info.tensor.dtype),
            "shape": list(info.tensor.shape),
            "numel": info.numel,
            G_INDEX_START_KEY: packed_offset,
            G_INDEX_END_KEY: packed_offset + int(packed.numel()),
            "value_start": value_offset,
            "value_end": value_offset + nnz,
        }
    )
    packed_parts.append(packed)
    value_parts.append(values)
    return packed_offset + int(packed.numel()), value_offset + nnz


def _decode_sparse(
    payload_tensors: TensorBatch,
    metadata: list[dict[str, Any]],
    device: torch.device | int | str,
    byte_cap: int,
) -> Iterator[TensorBatch]:
    payload = dict(payload_tensors)
    packed_values = payload[G_PACKED_VALUES_NAME].to(device=device)
    packed_locations = payload[G_PACKED_INDICES_NAME].to(
        device=device,
        dtype=torch.long,
    )
    batch: TensorBatch = []
    batch_bytes = 0
    for item in metadata:
        numel = int(item["numel"])
        dtype = _dtype_from_name(item["dtype"])
        values = packed_values[int(item["value_start"]) : int(item["value_end"])].to(
            dtype=dtype
        )
        tensor = torch.zeros(numel, dtype=dtype, device=device)
        tensor.index_copy_(
            0,
            packed_locations[int(item[G_INDEX_START_KEY]) : int(item[G_INDEX_END_KEY])],
            values,
        )
        tensor = tensor.view(tuple(item["shape"]))
        tensor_bytes = tensor.numel() * tensor.element_size()
        if batch and batch_bytes + tensor_bytes > byte_cap:
            yield batch
            batch = []
            batch_bytes = 0
        batch.append((item["name"], tensor))
        batch_bytes += tensor_bytes
    if batch:
        yield batch


def _merge_sparse_payloads(payloads: list[TensorPayload]) -> TensorPayload:
    packed_parts = []
    value_parts = []
    metadata = []
    packed_offset = 0
    value_offset = 0
    for tensors, _, sparse_metadata in payloads:
        payload = dict(tensors)
        packed = payload[G_PACKED_INDICES_NAME]
        values = payload[G_PACKED_VALUES_NAME]
        packed_parts.append(packed)
        value_parts.append(values)
        for item in sparse_metadata:
            item = dict(item)
            item[G_INDEX_START_KEY] += packed_offset
            item[G_INDEX_END_KEY] += packed_offset
            item["value_start"] += value_offset
            item["value_end"] += value_offset
            metadata.append(item)
        packed_offset += int(packed.numel())
        value_offset += int(values.numel())
    return (
        [
            (G_PACKED_INDICES_NAME, torch.cat(packed_parts, dim=0)),
            (G_PACKED_VALUES_NAME, torch.cat(value_parts, dim=0)),
        ],
        G_SPARSE_INDICES_TRANSPORT,
        metadata,
    )


def _wire_bytes(tensors: TensorBatch) -> int:
    return sum(
        (byte_size := int(tensor.numel() * tensor.element_size()))
        + (-byte_size) % G_PAYLOAD_ALIGNMENT_BYTES
        for _, tensor in tensors
    )


def _sparse_payload_wire_bytes(
    packed_numel: int,
    value_numel: int,
    *,
    packed_dtype: torch.dtype,
    value_dtype: torch.dtype,
) -> int:
    packed_bytes = packed_numel * _dtype_itemsize(packed_dtype)
    value_bytes = value_numel * _dtype_itemsize(value_dtype)
    return (
        packed_bytes
        + (-packed_bytes) % G_PAYLOAD_ALIGNMENT_BYTES
        + value_bytes
        + (-value_bytes) % G_PAYLOAD_ALIGNMENT_BYTES
    )


def _sparse_payload_exceeds_dense(
    packed_numel: int,
    value_numel: int,
    *,
    packed_dtype: torch.dtype,
    value_dtype: torch.dtype,
    dense_wire_bytes: int,
) -> bool:
    return (
        _sparse_payload_wire_bytes(
            packed_numel,
            value_numel,
            packed_dtype=packed_dtype,
            value_dtype=value_dtype,
        )
        > dense_wire_bytes
    )


def _next_chunk(
    iterator: Iterator[NamedTensor],
    byte_cap: int,
    *,
    pending_item: NamedTensor | None = None,
) -> tuple[TensorBatch, NamedTensor | None]:
    chunk: TensorBatch = []
    chunk_bytes = 0
    items: Iterable[NamedTensor] = iterator
    if pending_item is not None:
        items = itertools.chain((pending_item,), iterator)
    for item in items:
        tensor_bytes = item[1].numel() * item[1].element_size()
        if chunk and chunk_bytes + tensor_bytes > byte_cap:
            return chunk, item
        chunk.append(item)
        chunk_bytes += tensor_bytes
    return chunk, None


def _advance_chunk(
    iterator: Iterator[NamedTensor],
    byte_cap: int,
    *,
    pending_item: NamedTensor | None = None,
) -> tuple[NamedTensor | None, bool]:
    chunk_bytes = 0
    consumed_item = False
    items: Iterable[NamedTensor] = iterator
    if pending_item is not None:
        items = itertools.chain((pending_item,), iterator)
    for item in items:
        tensor_bytes = item[1].numel() * item[1].element_size()
        if consumed_item and chunk_bytes + tensor_bytes > byte_cap:
            return item, False
        consumed_item = True
        chunk_bytes += tensor_bytes
    return None, True


def _broadcast_header(
    header: Mapping[str, Any],
    *,
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> tuple[dict[str, Any], HeaderRefs]:
    encoded = _encode_header_metadata(header)
    if group.rank == src:
        control_tensor = torch.tensor(
            _header_control_values(header, len(encoded)),
            dtype=torch.int64,
            device=device,
        )
    else:
        control_tensor = torch.empty(4, dtype=torch.int64, device=device)
    group.broadcast(control_tensor, src=src)
    kind, transport, payload_numel, metadata_len = _decode_header_control(
        control_tensor
    )

    metadata_tensor = None
    metadata: dict[str, Any] = {}
    if metadata_len > 0:
        if group.rank == src:
            metadata_tensor = torch.tensor(
                list(encoded), dtype=torch.uint8, device=device
            )
        else:
            metadata_tensor = torch.empty(
                metadata_len, dtype=torch.uint8, device=device
            )
        group.broadcast(metadata_tensor, src=src)
        if group.rank != src:
            metadata = json.loads(
                metadata_tensor.cpu().numpy().tobytes().decode("utf-8")
            )

    if group.rank == src:
        received_header = dict(header)
    else:
        received_header = {
            "kind": kind,
            "transport": transport,
            "payload_entries": [],
            "payload_numel": payload_numel,
            "sparse_metadata": [],
        }
        received_header.update(metadata)
    return received_header, (control_tensor, metadata_tensor)


def _header_control_values(header: Mapping[str, Any], metadata_len: int) -> list[int]:
    kind = header["kind"]
    if kind not in G_HEADER_KIND_TO_CODE:
        raise ValueError(f"Unsupported weight transfer header kind: {kind}")
    transport = header.get("transport", G_DENSE_TRANSPORT)
    if transport not in G_HEADER_TRANSPORT_TO_CODE:
        raise ValueError(f"Unsupported weight transfer header transport: {transport}")
    return [
        G_HEADER_KIND_TO_CODE[kind],
        G_HEADER_TRANSPORT_TO_CODE[transport],
        int(header.get("payload_numel", 0)),
        metadata_len,
    ]


def _decode_header_control(
    control_tensor: torch.Tensor,
) -> tuple[WeightTransferKind, DeltaCompressionTransport, int, int]:
    kind_code, transport_code, payload_numel, metadata_len = [
        int(value) for value in control_tensor.cpu().tolist()
    ]
    try:
        kind = G_HEADER_CODE_TO_KIND[kind_code]
        transport = G_HEADER_CODE_TO_TRANSPORT[transport_code]
    except KeyError:
        raise ValueError(
            f"Unsupported weight transfer header control values: "
            f"kind={kind_code}, transport={transport_code}"
        ) from None
    return (
        cast(WeightTransferKind, kind),
        cast(DeltaCompressionTransport, transport),
        payload_numel,
        metadata_len,
    )


def _encode_header_metadata(header: Mapping[str, Any]) -> bytes:
    metadata = {
        key: header[key]
        for key in ("payload_entries", "sparse_metadata")
        if header.get(key)
    }
    return json.dumps(metadata).encode("utf-8") if metadata else b""


def _recv_payload(
    payload_numel: int,
    *,
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> torch.Tensor:
    payload = torch.empty(payload_numel, dtype=torch.uint8, device=device)
    if payload.numel() > 0:
        group.broadcast(payload, src=src)
    return payload


def _dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return G_TENSOR_DTYPE_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unsupported tensor dtype in weight transfer: {name}"
        ) from None


def _cuda_streams(
    device: torch.device | int | str | None = None,
) -> list[torch.cuda.Stream | None]:
    if not torch.cuda.is_available():
        return [None]
    normalized_device = _normalize_device(device)
    if normalized_device is not None and normalized_device.type != "cuda":
        return [None]
    with _cuda_device(normalized_device):
        return [torch.cuda.Stream() for _ in range(get_num_buffers())]


@contextlib.contextmanager
def _use_stream(
    streams: list[torch.cuda.Stream | None],
    index: int,
):
    stream = streams[index]
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def _record_header_stream(refs: HeaderRefs) -> None:
    control_tensor, metadata_tensor = refs
    _record_tensor_stream(control_tensor)
    if metadata_tensor is not None:
        _record_tensor_stream(metadata_tensor)


def _record_tensor_stream(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        return
    tensor.record_stream(torch.cuda.current_stream())


def _sync_streams(streams: list[torch.cuda.Stream | None]) -> None:
    for stream in streams:
        if stream is not None:
            stream.synchronize()


def _synchronize_current_transfer_stream(device: torch.device | int | str) -> None:
    if not torch.cuda.is_available():
        return
    normalized_device = _normalize_device(device)
    if normalized_device is None or normalized_device.type == "cuda":
        torch.cuda.current_stream(normalized_device).synchronize()


def _record_stream_event(stream: torch.cuda.Stream | None) -> torch.cuda.Event | None:
    if stream is None:
        return None
    return stream.record_event()


def _record_payload_readiness_events() -> PayloadEvents:
    if not torch.cuda.is_available():
        return ()
    return (torch.cuda.current_stream().record_event(),)


def _wait_for_payload_events(events: Iterable[torch.cuda.Event]) -> None:
    seen_events = set()
    current_stream = None
    for event in events:
        if id(event) in seen_events:
            continue
        if current_stream is None:
            current_stream = torch.cuda.current_stream()
        current_stream.wait_event(event)
        seen_events.add(id(event))


@contextlib.contextmanager
def _cuda_device(device: torch.device | None):
    if device is None or device.type != "cuda":
        yield
        return
    with torch.cuda.device(device):
        yield


def _normalize_device(device: torch.device | int | str | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)
