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

import asyncio

import torch

from nemo_rl.utils.checkpoint_engine import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunk_batches,
    split_weight_chunks,
)
from nemo_rl.utils.checkpoint_engines.nixl import (
    NixlAgentMetadata,
    NIXLCheckpointEngine,
)


class _DummyNixlAgent:
    def __init__(self):
        self.closed = False
        self.registered = []
        self.xfer_descs = []

    def add_remote_agent(self, metadata):
        raise AssertionError(f"idle rank should not add remote agents: {metadata}")

    def remove_remote_agent(self, agent_name):
        raise AssertionError(f"idle rank should not remove remote agents: {agent_name}")

    def register_memory(self, buf):
        desc = f"reg_{len(self.registered)}"
        self.registered.append(buf)
        return desc

    def get_xfer_descs(self, buf):
        desc = f"xfer_{len(self.xfer_descs)}"
        self.xfer_descs.append(buf)
        return desc

    def get_agent_metadata(self):
        return {
            "agent_name": "dummy",
            "agent_metadata": b"",
            "zmq_ip": "127.0.0.1",
            "zmq_port": 0,
        }

    def close(self):
        self.closed = True


class _RecordingNixlAgent(_DummyNixlAgent):
    def __init__(self):
        super().__init__()
        self.added = []
        self.removed = []

    def add_remote_agent(self, metadata):
        agent_name = metadata["agent_name"]
        self.added.append(agent_name)
        return agent_name

    def remove_remote_agent(self, agent_name):
        self.removed.append(agent_name)


class _PluginCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size, plugin_arg):
        self.bucket_size = bucket_size
        self.plugin_arg = plugin_arg

    def prepare(self):
        return None

    def init_policy_process_group(
        self, *, worker_rank, train_world_size, rollout_world_size, metadata
    ):
        pass

    def init_rollout_process_group(
        self, *, rollout_rank, train_world_size, rollout_world_size, metadata
    ):
        pass

    def finalize(self):
        pass

    async def send_weights(self, weights):
        pass

    async def receive_weight_batches(self):
        if False:
            yield []


def _metadata(agent_name: str) -> NixlAgentMetadata:
    return {
        "agent_name": agent_name,
        "agent_metadata": b"",
        "zmq_ip": "127.0.0.1",
        "zmq_port": 0,
    }


async def _collect_weight_batches(chunks, bucket_size):
    results = []
    async for batch in merge_weight_chunk_batches(chunks, bucket_size):
        results.extend((name, weight.clone()) for name, weight in batch)
    return results


def test_split_and_merge_weight_chunks_roundtrip():
    weights = [
        ("small", torch.arange(4, dtype=torch.float32)),
        ("large", torch.arange(18, dtype=torch.float32).reshape(3, 6)),
    ]
    bucket_size = 16

    async def run_roundtrip():
        async def chunk_batches():
            async for tensor_meta, chunk in split_weight_chunks(
                iter(weights), bucket_size
            ):
                yield [(tensor_meta, chunk)]

        return await _collect_weight_batches(chunk_batches(), bucket_size)

    merged = asyncio.run(run_roundtrip())

    assert [name for name, _weight in merged] == ["small", "large"]
    assert torch.equal(merged[0][1], weights[0][1])
    assert torch.equal(merged[1][1], weights[1][1])


def test_merge_weight_chunk_batches_preserves_complete_weight_batches():
    small = torch.arange(4, dtype=torch.float32)
    large = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    bucket_size = 16

    async def chunk_batches():
        small_buffer = small.view(-1).view(torch.uint8)
        yield [
            (
                TensorMeta(
                    name="small",
                    shape=small.shape,
                    dtype=small.dtype,
                    chunk_offset=0,
                    chunk_size=small.nbytes,
                    offset=0,
                ),
                small_buffer,
            )
        ]

        async for tensor_meta, chunk in split_weight_chunks(
            iter([("large", large)]), bucket_size
        ):
            yield [(tensor_meta, chunk)]

    async def run_roundtrip():
        results = []
        async for batch in merge_weight_chunk_batches(chunk_batches(), bucket_size):
            results.append([(name, weight.clone()) for name, weight in batch])
        return results

    batches = asyncio.run(run_roundtrip())

    assert [[name for name, _weight in batch] for batch in batches] == [
        ["small"],
        ["large"],
    ]
    assert torch.equal(batches[0][0][1], small)
    assert torch.equal(batches[1][0][1], large)


def test_checkpoint_engine_registry_loads_class_path_plugin():
    engine = CheckpointEngineRegistry.new(
        f"{__name__}:_PluginCheckpointEngine",
        bucket_size=123,
        plugin_arg="ok",
    )

    assert isinstance(engine, _PluginCheckpointEngine)
    assert engine.bucket_size == 123
    assert engine.plugin_arg == "ok"


def test_nixl_prepare_reuses_registered_buffers():
    agent = _DummyNixlAgent()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.send_buf = None
    engine.recv_buf = None
    engine.send_reg_descs = None
    engine.recv_reg_descs = None
    engine.send_descs = None
    engine.recv_descs = None
    engine._allocate_transfer_buffer = lambda: object()

    assert engine.prepare()["agent_name"] == "dummy"
    first_state = (
        engine.send_buf,
        engine.recv_buf,
        engine.send_reg_descs,
        engine.recv_reg_descs,
        engine.send_descs,
        engine.recv_descs,
    )

    assert engine.prepare()["agent_name"] == "dummy"

    assert (
        engine.send_buf,
        engine.recv_buf,
        engine.send_reg_descs,
        engine.recv_reg_descs,
        engine.send_descs,
        engine.recv_descs,
    ) == first_state
    assert len(agent.registered) == 2
    assert len(agent.xfer_descs) == 2


def test_nixl_idle_rank_finalizes_without_closing_buffers():
    agent = _DummyNixlAgent()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.rank = None
    engine.world_size = None
    engine.prev_agent = None
    engine.next_agent = None
    engine.send_buf = object()
    engine.recv_buf = object()
    engine.send_reg_descs = "send_desc"
    engine.recv_reg_descs = "recv_desc"
    engine.send_descs = object()
    engine.recv_descs = object()
    engine._cupy_buffers = []

    engine.init_process_group(
        rank=-1,
        world_size=9,
        prev_agent_metadata=None,
        next_agent_metadata=None,
    )
    engine.finalize()

    assert engine.send_buf is not None
    assert engine.recv_buf is not None
    assert engine.send_reg_descs == "send_desc"
    assert engine.recv_reg_descs == "recv_desc"
    assert not agent.closed
    engine.close()
    assert agent.closed


def _make_topology_engine(agent):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.rank = None
    engine.world_size = None
    engine.prev_agent = None
    engine.next_agent = None
    return engine


def test_nixl_uses_paired_topology_when_policy_can_cover_rollout():
    metadata = [_metadata(f"policy{i}") for i in range(4)] + [
        _metadata(f"rollout{i}") for i in range(4)
    ]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_policy_process_group(
        worker_rank=2,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert policy_engine.rank == 0
    assert policy_engine.world_size == 2
    assert policy_engine.prev_agent is None
    assert policy_engine.next_agent == "rollout2"
    assert policy_agent.added == ["rollout2"]

    rollout_agent = _RecordingNixlAgent()
    rollout_engine = _make_topology_engine(rollout_agent)
    rollout_engine.init_rollout_process_group(
        rollout_rank=2,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert rollout_engine.rank == 1
    assert rollout_engine.world_size == 2
    assert rollout_engine.prev_agent == "policy2"
    assert rollout_engine.next_agent is None
    assert rollout_agent.added == ["policy2"]


def test_nixl_policy_ranks_without_rollout_pair_are_idle():
    metadata = [_metadata(f"policy{i}") for i in range(6)] + [
        _metadata(f"rollout{i}") for i in range(4)
    ]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_policy_process_group(
        worker_rank=5,
        train_world_size=6,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert policy_engine.rank == -1
    assert policy_engine.world_size == 2
    assert policy_engine.prev_agent is None
    assert policy_engine.next_agent is None
    assert policy_agent.added == []


def test_nixl_falls_back_to_chain_when_rollout_exceeds_policy_workers():
    metadata = [_metadata("policy0")] + [_metadata(f"rollout{i}") for i in range(3)]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_policy_process_group(
        worker_rank=0,
        train_world_size=1,
        rollout_world_size=3,
        metadata=metadata,
    )

    assert policy_engine.rank == 0
    assert policy_engine.world_size == 4
    assert policy_engine.next_agent == "rollout0"

    middle_agent = _RecordingNixlAgent()
    middle_engine = _make_topology_engine(middle_agent)
    middle_engine.init_rollout_process_group(
        rollout_rank=1,
        train_world_size=1,
        rollout_world_size=3,
        metadata=metadata,
    )

    assert middle_engine.rank == 2
    assert middle_engine.world_size == 4
    assert middle_engine.prev_agent == "rollout0"
    assert middle_engine.next_agent == "rollout2"
