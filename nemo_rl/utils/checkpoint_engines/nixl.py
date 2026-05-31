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

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import logging
import socket
import time
import uuid
from collections import defaultdict, deque
from typing import Any, AsyncGenerator, Generator, Iterable, TypedDict

import ray
import torch
import zmq
import zmq.asyncio

from nemo_rl.utils.checkpoint_engines.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunk_batches,
    split_weight_chunks,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NIXLCheckpointEngine",
    "NixlAgentMetadata",
    "preinit_nixl_agent",
]


class NixlAgentMetadata(TypedDict):
    """Serializable NIXL agent metadata exchanged through Ray."""

    agent_name: str
    agent_metadata: bytes
    zmq_ip: str
    zmq_port: int


class NixlBucketMetadata(TypedDict):
    """Metadata for one NIXL transfer bucket."""

    bucket_meta: dict[str, TensorMeta]
    bucket_bytes: int
    is_last: bool


def _is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _get_free_port(address: str) -> int:
    family = socket.AF_INET6 if _is_valid_ipv6_address(address) else socket.AF_INET
    with socket.socket(family=family, type=socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, 0))
        return int(sock.getsockname()[1])


def _tcp_address(ip_address: str, port: int) -> str:
    if _is_valid_ipv6_address(ip_address):
        return f"tcp://[{ip_address}]:{port}"
    return f"tcp://{ip_address}:{port}"


def _require_module(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name!r} is required for NIXL checkpoint-engine refit. "
            f"{install_hint}"
        ) from exc


def _optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _normalize_device(device: str | torch.device) -> torch.device:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch_device


def _sync_devices(devices: Iterable[torch.device]) -> None:
    synced_cuda_indices: set[int] = set()
    for device in devices:
        if device.type != "cuda":
            continue
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if index in synced_cuda_indices:
            continue
        torch.cuda.synchronize(torch.device("cuda", index))
        synced_cuda_indices.add(index)


def _create_nixl_agent(
    *,
    agent_name: str,
    backend_name: str,
    backend_init_params: dict[str, Any] | None = None,
) -> Any:
    nixl_api = _require_module(
        "nixl._api",
        "Install NIXL in the runtime environment or disable "
        "policy.generation.checkpoint_engine.enabled.",
    )
    if backend_name == "UCX" and backend_init_params is None:
        return nixl_api.nixl_agent(agent_name)

    nixl_config = nixl_api.nixl_agent_config(backends=[])
    agent = nixl_api.nixl_agent(agent_name, nixl_config)
    init_params = {
        key: str(value) for key, value in (backend_init_params or {}).items()
    }
    agent.create_backend(backend_name, init_params)
    return agent


def preinit_nixl_agent(
    *,
    backend_name: str = "UCX",
    backend_init_params: dict[str, Any] | None = None,
) -> Any:
    """Create a lightweight NIXL agent to initialize backend plugins early."""
    agent = _create_nixl_agent(
        agent_name=f"preinit-{uuid.uuid4()}",
        backend_name=backend_name,
        backend_init_params=backend_init_params,
    )
    agent.get_agent_metadata()
    return agent


def _bucket_metadata(
    bucket_meta: dict[str, TensorMeta],
    *,
    bucket_bytes: int,
    is_last: bool,
) -> NixlBucketMetadata:
    return {
        "bucket_meta": bucket_meta,
        "bucket_bytes": bucket_bytes,
        "is_last": is_last,
    }


def _bucket_chunks(
    metadata: NixlBucketMetadata,
    buffer: torch.Tensor,
) -> list[tuple[TensorMeta, torch.Tensor]]:
    chunks = []
    for tensor_meta in metadata["bucket_meta"].values():
        if tensor_meta.offset is None:
            raise RuntimeError(f"Missing NIXL offset for {tensor_meta.name}.")
        tensor = buffer[
            tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size
        ]
        chunks.append((tensor_meta, tensor))
    return chunks


class NixlAgent:
    """NIXL agent wrapper using ZMQ for bucket metadata notifications."""

    def __init__(
        self,
        backend_name: str = "UCX",
        backend_init_params: dict[str, str] | None = None,
    ) -> None:
        self.agent_name = str(uuid.uuid4())
        self.agent = _create_nixl_agent(
            agent_name=self.agent_name,
            backend_name=backend_name,
            backend_init_params=backend_init_params,
        )
        self.notifications: dict[str, deque[bytes]] = defaultdict(deque)
        self.messages: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self.zmq_clients: dict[str, zmq.Socket] = {}
        self.zmq_client_context = zmq.Context()
        self._closed = False
        self._start_zmq_server()

    def _start_zmq_server(self) -> None:
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port = _get_free_port(self.ip)

        self.zmq_context = zmq.asyncio.Context()
        self.socket = self.zmq_context.socket(zmq.PULL)
        if _is_valid_ipv6_address(self.ip):
            self.socket.setsockopt(zmq.IPV6, 1)
        self.socket.bind(_tcp_address(self.ip, self.listen_port))

    def get_agent_metadata(self) -> NixlAgentMetadata:
        return {
            "agent_name": self.agent_name,
            "agent_metadata": self.agent.get_agent_metadata(),
            "zmq_ip": self.ip,
            "zmq_port": self.listen_port,
        }

    def add_remote_agent(self, metadata: NixlAgentMetadata) -> str:
        remote_agent_name = self.agent.add_remote_agent(
            metadata["agent_metadata"]
        ).decode("utf-8")
        if remote_agent_name != metadata["agent_name"]:
            raise RuntimeError(
                f"NIXL remote agent mismatch: expected {metadata['agent_name']}, "
                f"got {remote_agent_name}"
            )

        client_socket = self.zmq_client_context.socket(zmq.PUSH)
        if _is_valid_ipv6_address(metadata["zmq_ip"]):
            client_socket.setsockopt(zmq.IPV6, 1)
        client_socket.connect(_tcp_address(metadata["zmq_ip"], metadata["zmq_port"]))
        self.zmq_clients[remote_agent_name] = client_socket
        return remote_agent_name

    def remove_remote_agent(self, agent_name: str) -> None:
        self.agent.remove_remote_agent(agent_name)
        client_socket = self.zmq_clients.pop(agent_name)
        client_socket.close(linger=0)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        for client_socket in self.zmq_clients.values():
            client_socket.close(linger=0)
        self.zmq_clients.clear()
        self.socket.close(linger=0)
        self.zmq_client_context.destroy(linger=0)
        self.zmq_context.destroy(linger=0)

    def send_message(self, agent_name: str, message: dict[str, Any]) -> None:
        self.zmq_clients[agent_name].send_pyobj(
            (self.agent_name, message), zmq.DONTWAIT
        )

    async def read_message(self, agent_name: str) -> dict[str, Any]:
        while len(self.messages[agent_name]) == 0:
            recv_agent_name, message = await self.socket.recv_pyobj()
            self.messages[recv_agent_name].append(message)
            await asyncio.sleep(0)
        return self.messages[agent_name].popleft()

    async def get_notification(self, remote_name: str) -> bytes:
        while len(self.notifications[remote_name]) == 0:
            notifications = self.agent.get_new_notifs()
            for agent_name, agent_notifications in notifications.items():
                self.notifications[agent_name].extend(agent_notifications)
            await asyncio.sleep(0)
        return self.notifications[remote_name].popleft()

    def register_memory(self, buffer: torch.Tensor) -> Any:
        return self.agent.register_memory(buffer)

    def get_xfer_descs(self, buffer: torch.Tensor) -> Any:
        return self.agent.get_xfer_descs(buffer)

    def initialize_xfer(
        self,
        operation: str,
        local_descs: Any,
        remote_descs: Any,
        remote_agent: str,
        notify_key: bytes,
    ) -> Any:
        return self.agent.initialize_xfer(
            operation,
            local_descs,
            remote_descs,
            remote_agent,
            notify_key,
        )

    def transfer(self, xfer_handle: Any) -> str:
        return self.agent.transfer(xfer_handle)

    def check_xfer_state(self, xfer_handle: Any) -> str:
        return self.agent.check_xfer_state(xfer_handle)

    def release_xfer_handle(self, xfer_handle: Any) -> None:
        self.agent.release_xfer_handle(xfer_handle)


class ReadableOperation:
    """Remote-readable bucket exposed through NIXL."""

    def __init__(
        self,
        agent: NixlAgent,
        remote_agent: str,
        local_descs: Any,
        metadata: NixlBucketMetadata,
    ) -> None:
        self.agent = agent
        self.remote_agent = remote_agent
        self.notify_key = uuid.uuid4().bytes
        message = {
            "notify_key": self.notify_key,
            "remote_descs": local_descs,
            **metadata,
        }
        self.agent.send_message(self.remote_agent, message)

    async def wait_for_complete(self) -> None:
        notification = await self.agent.get_notification(self.remote_agent)
        if self.notify_key != notification:
            raise RuntimeError(
                f"NIXL notification mismatch: expected {self.notify_key}, "
                f"got {notification}"
            )


class ReadOperation:
    """NIXL read operation from a remote readable bucket."""

    def __init__(
        self,
        agent: NixlAgent,
        remote_agent: str,
        local_descs: Any,
        bucket_size: int,
    ) -> None:
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.remote_descs: Any | None = None
        self.xfer_handle: Any | None = None
        self.notify_key: bytes | None = None
        self.bucket_size = bucket_size
        self.start_time: float | None = None

    async def read_metadata(self) -> NixlBucketMetadata:
        metadata = await self.agent.read_message(self.remote_agent)
        self.remote_descs = metadata.pop("remote_descs")
        self.notify_key = metadata.pop("notify_key")
        return metadata

    def begin_read(self) -> None:
        if self.remote_descs is None or self.notify_key is None:
            raise RuntimeError("NIXL read metadata must be received before begin_read.")
        self.xfer_handle = self.agent.initialize_xfer(
            "READ",
            self.local_descs,
            self.remote_descs,
            self.remote_agent,
            self.notify_key,
        )
        state = self.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise RuntimeError(f"NIXL read from {self.remote_agent} entered ERR state.")
        self.start_time = time.time()

    async def wait_for_complete(self) -> None:
        if self.xfer_handle is None or self.start_time is None:
            raise RuntimeError("NIXL read must be started before waiting.")
        while True:
            state = self.agent.check_xfer_state(self.xfer_handle)
            if state == "ERR":
                raise RuntimeError(
                    f"NIXL read from {self.remote_agent} entered ERR state."
                )
            if state == "DONE":
                break
            await asyncio.sleep(0)
        self.agent.release_xfer_handle(self.xfer_handle)
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            bandwidth = self.bucket_size / elapsed / (1024 * 1024 * 1024)
            logger.debug(
                "NIXL read from %s completed at %.2f GB/s",
                self.remote_agent,
                bandwidth,
            )


@CheckpointEngineRegistry.register("nixl")
class NIXLCheckpointEngine(CheckpointEngine):
    """NIXL checkpoint engine for non-colocated policy-to-generation refit."""

    def __init__(
        self,
        bucket_size: int,
        device: str | torch.device = "cuda",
        backend_name: str = "UCX",
        backend_init_params: dict[str, str] | None = None,
        cleanup_after_load: bool = True,
    ) -> None:
        self.bucket_size = bucket_size
        self.device = _normalize_device(device)
        self.cleanup_after_load = cleanup_after_load
        self.agent = NixlAgent(
            backend_name=backend_name,
            backend_init_params=backend_init_params,
        )
        self.rank: int | None = None
        self.world_size: int | None = None
        self.prev_agent: str | None = None
        self.next_agent: str | None = None
        self.send_buf: torch.Tensor | None = None
        self.recv_buf: torch.Tensor | None = None
        self.send_reg_descs: Any | None = None
        self.recv_reg_descs: Any | None = None
        self.send_descs: Any | None = None
        self.recv_descs: Any | None = None
        self._cupy_buffers: list[Any] = []

    def _allocate_transfer_buffer(self) -> torch.Tensor:
        if self.device.type != "cuda":
            return torch.zeros(
                self.bucket_size,
                dtype=torch.uint8,
                device=self.device,
                pin_memory=self.device.type == "cpu",
            )

        torch.cuda.set_device(self.device)
        cupy = _optional_module("cupy")
        if cupy is None:
            logger.warning(
                "CuPy is not installed; using torch CUDA buffers for NIXL memory "
                "registration. If registration fails with expandable CUDA segments, "
                "install CuPy or disable expandable segments."
            )
            return torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)

        with cupy.cuda.Device(self.device.index):
            cupy_buffer = cupy.zeros(self.bucket_size, dtype=cupy.uint8)
        self._cupy_buffers.append(cupy_buffer)
        return torch.as_tensor(cupy_buffer, dtype=torch.uint8, device=self.device)

    def prepare(self) -> NixlAgentMetadata:
        if (
            self.send_buf is not None
            and self.recv_buf is not None
            and self.send_reg_descs is not None
            and self.recv_reg_descs is not None
            and self.send_descs is not None
            and self.recv_descs is not None
        ):
            return self.agent.get_agent_metadata()

        self.send_buf = self._allocate_transfer_buffer()
        self.recv_buf = self._allocate_transfer_buffer()
        self.send_reg_descs = self.agent.register_memory(self.send_buf)
        self.recv_reg_descs = self.agent.register_memory(self.recv_buf)
        self.send_descs = self.agent.get_xfer_descs(self.send_buf)
        self.recv_descs = self.agent.get_xfer_descs(self.recv_buf)
        return self.agent.get_agent_metadata()

    def _prepared_transfer_buffers(self) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
        if (
            self.send_buf is None
            or self.recv_buf is None
            or self.send_descs is None
            or self.recv_descs is None
        ):
            raise RuntimeError("NIXL transfer buffers are not prepared.")
        return self.send_buf, self.recv_buf, self.send_descs, self.recv_descs

    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[NixlAgentMetadata],
    ) -> None:
        if train_world_size >= rollout_world_size:
            world_size = 2
            rank = 0 if worker_rank < rollout_world_size else -1
            next_agent_metadata = (
                metadata[train_world_size + worker_rank] if rank == 0 else None
            )
        else:
            world_size = rollout_world_size + 1
            rank = 0 if worker_rank == 0 else -1
            next_agent_metadata = metadata[train_world_size] if rank == 0 else None
        self.init_process_group(
            rank=rank,
            world_size=world_size,
            prev_agent_metadata=None,
            next_agent_metadata=next_agent_metadata,
        )

    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[NixlAgentMetadata],
    ) -> None:
        if rollout_rank < 0 or rollout_rank >= rollout_world_size:
            raise ValueError(
                f"rollout_rank must be in [0, {rollout_world_size}), got {rollout_rank}"
            )

        if train_world_size >= rollout_world_size:
            world_size = 2
            rank = 1
            prev_agent_metadata = metadata[rollout_rank]
            next_agent_metadata = None
        else:
            world_size = rollout_world_size + 1
            rank = rollout_rank + 1
            prev_agent_metadata = (
                metadata[0]
                if rollout_rank == 0
                else metadata[train_world_size + rollout_rank - 1]
            )
            next_agent_metadata = (
                metadata[train_world_size + rollout_rank + 1]
                if rollout_rank < rollout_world_size - 1
                else None
            )
        self.init_process_group(
            rank=rank,
            world_size=world_size,
            prev_agent_metadata=prev_agent_metadata,
            next_agent_metadata=next_agent_metadata,
        )

    def init_process_group(
        self,
        *,
        rank: int,
        world_size: int,
        prev_agent_metadata: NixlAgentMetadata | None,
        next_agent_metadata: NixlAgentMetadata | None,
    ) -> None:
        if rank < 0:
            if prev_agent_metadata is not None or next_agent_metadata is not None:
                raise ValueError(f"Idle NIXL rank {rank} should not have peers.")
        elif rank == 0:
            if prev_agent_metadata is not None or next_agent_metadata is None:
                raise ValueError("NIXL source rank must only have a next peer.")
        elif rank < world_size - 1:
            if prev_agent_metadata is None or next_agent_metadata is None:
                raise ValueError("NIXL middle ranks must have previous and next peers.")
        elif prev_agent_metadata is None or next_agent_metadata is not None:
            raise ValueError("NIXL final rank must only have a previous peer.")

        self.rank = rank
        self.world_size = world_size
        self.prev_agent = None
        self.next_agent = None

        if prev_agent_metadata is not None:
            self.prev_agent = self.agent.add_remote_agent(prev_agent_metadata)
        if next_agent_metadata is not None:
            self.next_agent = self.agent.add_remote_agent(next_agent_metadata)

    def finalize(self) -> None:
        if self.prev_agent is not None:
            self.agent.remove_remote_agent(self.prev_agent)
        if self.next_agent is not None:
            self.agent.remove_remote_agent(self.next_agent)

        # Keep NIXL memory registrations alive for the lifetime of the long-lived
        # worker engine. In real Ray/vLLM actors, deregistering the UCX-backed
        # descriptors during every refit can segfault in ucp_mem_unmap; reusing
        # the registered buffers also avoids re-registering multi-GB buckets on
        # every policy-to-generation update.
        self.rank = None
        self.world_size = None
        self.prev_agent = None
        self.next_agent = None

    def close(self) -> None:
        """Close the long-lived NIXL agent when the engine is discarded."""
        self.finalize()
        self.agent.close()

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> None:
        if self.rank is None:
            raise RuntimeError(
                "NIXL checkpoint engine process group is not initialized."
            )
        if self.prev_agent is not None:
            raise RuntimeError("Only NIXL source ranks may send weights.")
        if self.rank < 0:
            for _ in weights:
                pass
            return
        if self.next_agent is None:
            raise RuntimeError("NIXL source rank has no next peer.")

        send_buf, recv_buf, send_descs, recv_descs = self._prepared_transfer_buffers()
        readable_op: ReadableOperation | None = None
        bucket_meta: dict[str, TensorMeta] = {}
        bucket_sync_devices: set[torch.device] = {send_buf.device}
        offset = 0
        start_time = time.time()

        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            if offset + tensor_meta.chunk_size > self.bucket_size:
                _sync_devices(bucket_sync_devices)
                if readable_op is not None:
                    await readable_op.wait_for_complete()
                readable_op = ReadableOperation(
                    self.agent,
                    self.next_agent,
                    send_descs,
                    _bucket_metadata(
                        bucket_meta,
                        bucket_bytes=offset,
                        is_last=False,
                    ),
                )
                send_buf, recv_buf = recv_buf, send_buf
                send_descs, recv_descs = recv_descs, send_descs
                bucket_meta = {}
                bucket_sync_devices = {send_buf.device}
                offset = 0

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            bucket_sync_devices.add(chunk.device)
            # CPU NIXL buffers can receive async GPU-to-CPU copies. Sync the
            # source CUDA devices before exposing each filled bucket to NIXL.
            send_buf[offset : offset + tensor_meta.chunk_size].copy_(
                chunk,
                non_blocking=True,
            )
            offset += tensor_meta.chunk_size

        _sync_devices(bucket_sync_devices)
        if readable_op is not None:
            await readable_op.wait_for_complete()

        readable_op = ReadableOperation(
            self.agent,
            self.next_agent,
            send_descs,
            _bucket_metadata(
                bucket_meta,
                bucket_bytes=offset,
                is_last=True,
            ),
        )
        await readable_op.wait_for_complete()
        logger.info("NIXL send_weights completed in %.2fs", time.time() - start_time)

    async def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        async for batch in merge_weight_chunk_batches(
            self._receive_weight_chunk_batches(), self.bucket_size
        ):
            yield batch

    async def _receive_weight_chunk_batches(
        self,
    ) -> AsyncGenerator[list[tuple[TensorMeta, torch.Tensor]], None]:
        if self.prev_agent is None:
            raise RuntimeError("NIXL receiver rank has no previous peer.")

        send_buf, recv_buf, send_descs, recv_descs = self._prepared_transfer_buffers()
        total_bytes = 0
        total_chunks = 0
        start_time = time.time()

        read_op = ReadOperation(
            self.agent, self.prev_agent, recv_descs, self.bucket_size
        )
        metadata = await read_op.read_metadata()
        read_op.begin_read()
        await read_op.wait_for_complete()
        total_bytes += metadata["bucket_bytes"]
        total_chunks += len(metadata["bucket_meta"])
        send_buf, recv_buf = recv_buf, send_buf
        send_descs, recv_descs = recv_descs, send_descs

        while not metadata["is_last"]:
            readable_op = None
            if self.next_agent is not None:
                readable_op = ReadableOperation(
                    self.agent,
                    self.next_agent,
                    send_descs,
                    metadata,
                )

            read_op = ReadOperation(
                self.agent, self.prev_agent, recv_descs, self.bucket_size
            )
            next_metadata = await read_op.read_metadata()
            read_op.begin_read()

            yield _bucket_chunks(metadata, send_buf)

            if readable_op is not None:
                await readable_op.wait_for_complete()
            await read_op.wait_for_complete()
            total_bytes += next_metadata["bucket_bytes"]
            total_chunks += len(next_metadata["bucket_meta"])
            _sync_devices((self.device,))
            metadata = next_metadata
            send_buf, recv_buf = recv_buf, send_buf
            send_descs, recv_descs = recv_descs, send_descs

        readable_op = None
        if self.next_agent is not None:
            readable_op = ReadableOperation(
                self.agent, self.next_agent, send_descs, metadata
            )

        yield _bucket_chunks(metadata, send_buf)

        if readable_op is not None:
            await readable_op.wait_for_complete()

        elapsed = time.time() - start_time
        if elapsed > 0:
            bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024)
            logger.info(
                "NIXL receive_weights completed: chunks=%d time=%.2fs bandwidth=%.2fGB/s",
                total_chunks,
                elapsed,
                bandwidth,
            )
