# Checkpoint Engine Design

Checkpoint engines provide a backend-neutral way to transfer policy weights to
non-colocated generation workers during refit. They are used by GRPO when
`policy.generation.checkpoint_engine.enabled=true`.

The user-facing guide is [Checkpoint-Engine Refit](../guides/checkpoint-engine-refit.md).
This document describes the implementation contract and how to add new transfer
backends.

## Goals

Checkpoint engines are designed to:

- decouple refit orchestration from the transport implementation
- let each backend manage its own metadata, buffers, and process topology
- stream weight batches instead of materializing a full model copy in the driver
- support plugin backends without changing GRPO or vLLM code

Checkpoint engines do not replace normal checkpoint save/load. They are a
runtime refit transport used between policy workers and generation workers.

## Control Flow

The GRPO refit flow is:

1. Read `policy.generation.checkpoint_engine`.
2. Instantiate the configured backend on every policy worker and generation
   worker.
3. Call `prepare()` on every backend instance and collect Ray-serializable
   metadata.
4. Initialize the backend topology with the combined metadata list.
5. Ask policy workers to send model weights.
6. Ask generation workers to receive weight batches and load them into the
   generation backend.
7. Call `finalize()` on all backend instances in a `finally` block.

The policy metadata is placed first in the combined metadata list, followed by
generation metadata. Backends receive `train_world_size` and
`rollout_world_size` so they can interpret that list.

## Configuration Contract

Checkpoint-engine config is stored under `policy.generation`:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        nixl:
          device: cpu
          backend_name: UCX
```

`backend` can be either:

- a registered backend name, such as `nixl`
- a class path, such as `my_pkg.refit:MyCheckpointEngine`

`engine_kwargs` must be keyed by the exact `backend` value. For a class-path
plugin:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: "my_pkg.refit:MyCheckpointEngine"
      update_weights_bucket_megabytes: 1024
      engine_kwargs:
        "my_pkg.refit:MyCheckpointEngine":
          transport: my_transport
```

The factory passes `bucket_size` in bytes plus the selected backend kwargs to
the backend constructor. It also provides a backend-neutral default `device`
unless the config already specifies one.

## Backend Interface

Backends subclass
{py:class}`CheckpointEngine <nemo_rl.utils.checkpoint_engines.base.CheckpointEngine>`.

```python
from typing import Any, AsyncGenerator, Generator

import torch

from nemo_rl.utils.checkpoint_engines import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)


@CheckpointEngineRegistry.register("my_backend")
class MyCheckpointEngine(CheckpointEngine):
    cleanup_after_load = True

    def __init__(self, bucket_size: int, device: str | torch.device = "cuda"):
        self.bucket_size = bucket_size
        self.device = torch.device(device)

    def prepare(self) -> Any:
        """Allocate or register buffers and return Ray-serializable metadata."""
        ...

    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a policy worker to the backend topology."""
        ...

    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a generation worker to the backend topology."""
        ...

    def finalize(self) -> None:
        """Release per-refit topology state."""
        ...

    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
    ) -> None:
        """Send `(name, tensor)` weights from the policy side."""
        ...

    async def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        """Yield `(name, tensor)` batches on the generation side."""
        ...
```

The `weights` generator is consumed once. Do not assume it can be replayed.

`receive_weight_batches()` should yield tensors with the original parameter
names and values. The generation backend loads each yielded batch immediately,
so yielding at transfer-bucket boundaries allows transfer and loading to overlap.

`cleanup_after_load` is read by the vLLM generation worker after the receive
loop. Set it to `False` when the backend can keep stable buffers and avoiding
extra cache cleanup is safe for steady-state training.

## Registry and Plugins

Built-in backends are lazy-imported by name through
{py:class}`CheckpointEngineRegistry <nemo_rl.utils.checkpoint_engines.base.CheckpointEngineRegistry>`.
External backends have two options:

1. Register a short name with `@CheckpointEngineRegistry.register("name")`.
2. Use a class path directly in config.

Class-path plugins do not need an import side effect. The registry imports the
module, looks up the class, validates that it subclasses `CheckpointEngine`, and
caches the result.

Supported class-path formats are:

```text
my_pkg.refit:MyCheckpointEngine
my_pkg.refit.MyCheckpointEngine
```

## Worker Integration

Policy workers use `BasePolicyWorker` helpers to instantiate the engine, prepare
metadata, join the backend topology, and send weights.

vLLM generation workers forward checkpoint-engine calls into vLLM internal
workers. The internal worker extension receives weight batches and calls the
normal vLLM load path for each batch. It also prints refit timing:

```text
[vLLM refit] Loaded ... via checkpoint engine; bytes=... total=... receive=... load=...
```

Async vLLM uses the same backend interface through async worker wrappers.

## NIXL Backend

The built-in NIXL backend is registered as `nixl`. It uses:

- NIXL agents for memory registration and transfer
- ZMQ messages for bucket metadata and transfer notifications
- two reusable transfer buffers per worker for pipelined bucket movement
- `split_weight_chunks()` and `merge_weight_chunk_batches()` for tensors larger
  than one bucket

The NIXL backend chooses one of two topologies:

- If `train_world_size >= rollout_world_size`, each rollout rank is paired with
  a policy rank. Extra policy ranks drain their local weight generators and do
  not send.
- If `rollout_world_size > train_world_size`, policy rank 0 sends into a chain
  of rollout ranks that forward buckets.

`finalize()` removes remote peer connections after each refit, but keeps memory
registrations and transfer buffers alive for the lifetime of the worker. This
avoids repeated multi-GB memory registration and avoids UCX teardown issues in
long-lived Ray/vLLM actors.

### Fault-Tolerance Boundary

The NIXL backend is restart-safe, not actor-healing. It is designed so a failed
transfer becomes a failed refit attempt that the driver can observe:

- `ReadOperation.begin_read()` raises if NIXL immediately returns `ERR`.
- `ReadOperation.wait_for_complete()` polls `check_xfer_state()` and raises if
  the transfer enters `ERR`.
- vLLM catches checkpoint-engine update failures and returns `False` to the
  GRPO refit orchestration.
- GRPO raises a refit error when any generation worker reports failure.
- GRPO calls `finalize()` in a `finally` block to remove per-refit peer
  connections.

The backend does not currently rebuild the NIXL topology, recreate Ray actors,
or reload vLLM inside the same training step after a peer disappears. That
responsibility belongs to the scheduler or a fault-tolerant launcher that
restarts the training process from a durable NeMo RL checkpoint.

For production runs, configure UCX so peer failures are reported to NIXL:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      engine_kwargs:
        nixl:
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: peer
```

And use bounded UCX retry/keepalive settings:

```sh
export UCX_RC_TIMEOUT=30s
export UCX_RC_RETRY_COUNT=7
export UCX_KEEPALIVE_INTERVAL=1s
export UCX_KEEPALIVE_NUM_EPS=10
```

`ucx_error_handling_mode: none` should be reserved for performance experiments
on stable clusters. With peer error handling disabled, a dead endpoint may not
surface as a NIXL `ERR` state promptly enough for job-level restart logic.

## vLLM NIXL Preinit

vLLM starts internal worker processes during engine setup. For NIXL/UCX, the
backend needs to be initialized inside those internal workers before the normal
vLLM worker setup path finishes.

NeMo RL patches the vLLM internal worker constructor and injects a config-driven
preinit call when:

- `policy.generation.checkpoint_engine.enabled=true`
- `policy.generation.checkpoint_engine.backend=nixl`

The preinit call uses the configured NIXL `backend_name` and
`backend_init_params`; it does not require NeMo RL feature environment
variables. A healthy vLLM run prints:

```text
NIXL vLLM worker preinit completed: backend=UCX
```

Backends other than NIXL should initialize themselves through the normal
`CheckpointEngine` constructor unless they also need code to run in nested vLLM
worker processes before engine setup.

## Bucket Helpers

`split_weight_chunks()` converts the policy weight stream into byte chunks no
larger than the configured bucket size. It records `TensorMeta` for each chunk:

- original tensor name
- shape
- dtype
- chunk offset
- chunk size
- byte offset inside the transfer bucket

`merge_weight_chunk_batches()` reconstructs tensors that were split across
multiple chunks while preserving bucket boundaries for normal tensors. Backend
implementations can use these helpers when their transport operates on flat
byte buffers.

## Adding a New Backend

1. Implement a `CheckpointEngine` subclass.
2. Decide whether to register a short name or use a class path in config.
3. Make `prepare()` allocate/register buffers and return metadata that Ray can
   serialize.
4. Use `init_policy_process_group()` and `init_rollout_process_group()` to
   connect peers from the combined metadata list.
5. Implement `send_weights()` as a streaming send of `(name, tensor)` pairs.
6. Implement `receive_weight_batches()` as a streaming receive that yields
   loadable `(name, tensor)` batches.
7. Make `finalize()` release per-refit peer state without destroying reusable
   buffers unless the backend cannot safely reuse them.
8. Define the backend's failure behavior. Transfer errors should become explicit
   exceptions or `False` update results rather than silent partial updates.
9. Add unit tests for registry loading, metadata setup, topology, failure
   propagation, and a small tensor roundtrip.
10. Run a non-colocated GRPO job and verify the `[vLLM refit]` timing line.

Good starting tests are:

```sh
uv run pytest tests/unit/utils/test_checkpoint_engine.py
uv run pytest tests/unit/algorithms/test_grpo.py -k checkpoint_engine
```

## Compatibility Notes

- Checkpoint-engine refit currently targets non-colocated policy-to-vLLM refit.
- SGLang non-colocated checkpoint-engine refit is not implemented.
- The backend must be installed in every Ray worker environment that imports it.
- The backend must preserve parameter names exactly, because generation workers
  use those names to load weights into the target model.
- The backend should avoid driver-side model materialization. The driver should
  orchestrate futures and metadata only.
