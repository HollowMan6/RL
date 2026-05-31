# Checkpoint-Engine Refit

Checkpoint-engine refit updates non-colocated generation workers from policy
weights through a pluggable transfer backend. The first built-in backend is
NIXL, which can use UCX/RDMA for large policy-to-vLLM refits.

Use this path when generation runs on dedicated resources:

- `policy.generation.colocated.enabled=false`
- `policy.generation.backend=vllm`
- `policy.generation.checkpoint_engine.enabled=true`

For colocated generation, NeMo RL continues to use the colocated IPC refit path.
For non-colocated generation without checkpoint-engine refit, NeMo RL uses the
collective update path.

## Enable NIXL Refit

Add a `checkpoint_engine` block under `policy.generation`:

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
      resources:
        num_nodes: 1
        gpus_per_node: 8
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        nixl:
          device: cpu
          cleanup_after_load: false
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: none
```

`backend` selects the checkpoint-engine transfer backend. Built-in backends use
short names such as `nixl`. External backends can use a class path; see
[Checkpoint Engine Design](../design-docs/checkpoint-engines.md).

`update_weights_bucket_megabytes` controls the transfer-buffer size. Larger
buckets reduce per-bucket overhead, but reserve more memory on every
participating worker. `2048` MiB is a good starting point for large models.

`engine_kwargs.<backend>` is passed to the backend constructor. For NIXL:

| Key | Meaning |
|---|---|
| `device` | Transfer-buffer device. Use `cpu` for host-pinned buffers when the NIXL UCX build does not support CUDA memory. Use `cuda` only when UCX CUDA memory support is available. |
| `cleanup_after_load` | Whether vLLM should run garbage collection and `torch.cuda.empty_cache()` after loading each refit. Disabling this avoids extra steady-state overhead when memory is stable. |
| `backend_name` | NIXL backend plugin name, usually `UCX`. |
| `backend_init_params` | Optional NIXL backend initialization parameters. Values are converted to strings before NIXL receives them. |

## Fault Tolerance

NIXL/UCX can help a training job fail fast on transport errors, which lets the
outer job launcher restart from the latest durable checkpoint. The current
checkpoint-engine refit path does not transparently replace a dead Ray actor or
rebuild the vLLM generation group inside the same training step. Treat
NIXL/UCX fault tolerance as transport-level error detection plus clean failure
propagation.

For runs where restartability matters, enable UCX peer error handling in the
NIXL backend config when your NIXL/UCX build supports it:

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

`ucx_error_handling_mode: none` is useful for performance experiments on stable
clusters, but it gives UCX less ability to report failed peers to NIXL. Use
`peer` for production or fault-injection testing so NIXL transfers can enter an
error state instead of waiting on a lost endpoint indefinitely.

Pair UCX peer error handling with bounded transport retry and keepalive values:

```sh
export UCX_RC_TIMEOUT=30s
export UCX_RC_RETRY_COUNT=7
export UCX_KEEPALIVE_INTERVAL=1s
export UCX_KEEPALIVE_NUM_EPS=10
```

These settings do not make an individual transfer magically recover. They bound
how long UCX waits before declaring a peer unhealthy. When UCX/NIXL reports an
error, NeMo RL marks the refit as failed, tears down per-refit NIXL peer state
through `finalize()`, and the training process should exit or be restarted by
the scheduler or fault-tolerant launcher.

Use normal NeMo RL checkpointing for restartable training:

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: /path/to/restartable/checkpoints
```

During fault-injection tests, look for one of these outcomes:

- NIXL/UCX reports an error and the job fails promptly.
- The scheduler restarts the job from the latest NeMo RL checkpoint.
- After restart, vLLM logs `NIXL vLLM worker preinit completed: backend=UCX`
  and subsequent refits print `[vLLM refit]` timing again.

## Command-Line Override Example

The same settings can be passed as Hydra overrides:

```sh
uv run --extra mcore --extra vllm examples/run_grpo.py \
  --config examples/configs/grpo_math_8B.yaml \
  cluster.num_nodes=2 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=8 \
  policy.generation.checkpoint_engine.enabled=true \
  policy.generation.checkpoint_engine.backend=nixl \
  policy.generation.checkpoint_engine.update_weights_bucket_megabytes=2048 \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.device=cpu \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.cleanup_after_load=false \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.backend_name=UCX \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.backend_init_params.ucx_error_handling_mode=none
```

Adjust `cluster.num_nodes` and
`policy.generation.colocated.resources.{num_nodes,gpus_per_node}` so the cluster
has enough policy and generation resources. For example, on two 8-GPU nodes, the
snippet above dedicates one node to vLLM generation and leaves one node for
policy workers.

## Runtime Requirements

NIXL must be importable in every Python environment that participates in refit:

- the driver/base environment
- policy worker environments
- vLLM worker environments, including async vLLM worker environments when used

Install the appropriate NIXL packages in those environments or bake them into
the container image. For CUDA 12 environments this is typically:

```sh
uv pip install nixl-cu12 nixl
```

UCX transport selection is controlled by UCX runtime environment variables.
Keep the checkpoint-engine feature selection in YAML/config, and use UCX
environment variables only for transport-level settings:

```sh
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,self,sm
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_MAX_RNDV_RAILS=1
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

When vLLM uses nested Ray workers, make sure transport variables are copied into
those workers using vLLM's normal environment-copy settings:

```sh
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=MELLANOX_
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=LD_LIBRARY_PATH,NIXL_LOG_LEVEL,NVIDIA_VISIBLE_DEVICES,UCX_NET_DEVICES,UCX_TLS,UCX_IB_ROCE_REACHABILITY_MODE,UCX_MAX_RNDV_RAILS,UCX_WARN_UNUSED_ENV_VARS
```

Do not configure the checkpoint-engine backend through ad hoc NeMo RL
environment variables. The backend, bucket size, device, and backend parameters
belong in `policy.generation.checkpoint_engine`.

## Performance Best Practices

For large non-colocated vLLM refits, start with CUDA transfer buffers when the
NIXL/UCX runtime supports CUDA memory registration. Keep checkpoint-engine
feature settings in YAML/config and use environment variables only for UCX/NIXL
transport runtime selection.

Recommended checkpoint-engine settings for large 30B-class MoE refits:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_megabytes: 1536
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: none
```

Recommended UCX runtime settings for a stable performance run:

```sh
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_4:1
export UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm
export UCX_MAX_RNDV_RAILS=4
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

Use the NIC list that is valid on your nodes; do not copy interface names
blindly across clusters. If CUDA buffers are not available, use
`device: cpu` and remove CUDA transports from `UCX_TLS`, for example
`UCX_TLS=rc,self,sm`.

On Ray launchers that start Ray inside the container, apply the container setup
and transport environment before `ray start` on every head and worker node.
Otherwise the driver can see the intended settings while nested Ray/vLLM worker
processes still inherit the wrong UCX library path or transport variables.

Use measured timings to tune from there:

- Prefer RDMA transports; logs should show `rc_mlx5`, not TCP-only transport.
- Sweep bucket size around `1024` to `2048` MiB instead of assuming larger is
  always faster. On a tested 30B MoE refit, `1536` MiB was faster than `2048`,
  `1024`, and `512`.
- Try multiple UCX rails when the fabric supports it. On the same 30B MoE
  setup, four rails were faster than two rails, and both were much faster than
  one rail.
- For fault-tolerant production runs, prefer
  `backend_init_params.ucx_error_handling_mode=peer`; reserve `none` for stable
  benchmarking where lowest transport overhead is the goal.

## Verify the Run

The driver log should show that the checkpoint engine is selected:

```text
Using checkpoint-engine refit backend: nixl
```

For NIXL + vLLM, the internal vLLM worker processes should initialize NIXL
before vLLM worker setup:

```text
NIXL vLLM worker preinit completed: backend=UCX
```

With `UCX_LOG_LEVEL=info`, UCX should report an RDMA transport such as:

```text
rma(rc_mlx5/mlx5_0:1)
```

If UCX reports only TCP transports, the run is not using RDMA and refit will be
much slower.

During each update, vLLM prints checkpoint-engine load timing:

```text
[vLLM refit] Loaded 18867 tensors in 29 batches via checkpoint engine; bytes=56.87GiB total=11.90s receive=10.71s load=1.18s sync=0.00s postprocess=0.00s cleanup=0.00s
```

The step timing also includes the end-to-end update:

```text
prepare_for_generation/transfer_and_update_weights: 11.94s
```

The `[vLLM refit]` line measures receive plus vLLM load time inside the vLLM
worker. The `transfer_and_update_weights` timer includes the full orchestration
window from the GRPO driver.

## Try a Correctness Smoke Test

`tools/refit_verifier.py` compares vLLM and Megatron logprobs after a refit:

```sh
uv run --extra mcore --extra vllm python tools/refit_verifier.py \
  --model_name /path/to/model \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1
```

This tool is useful for validating refit correctness and model compatibility.
It currently exercises the colocated refit path. To test the NIXL
checkpoint-engine path, run a non-colocated GRPO job with
`policy.generation.checkpoint_engine.enabled=true` and inspect the log markers
above.

## Troubleshooting

### The run errors with "checkpoint-engine refit is only supported for non-colocated generation"

Set:

```yaml
policy:
  generation:
    colocated:
      enabled: false
```

Checkpoint-engine refit is for non-colocated generation only.

### NIXL cannot be imported

Install NIXL in the environment that failed. In Ray runs, the driver, policy
workers, vLLM workers, and async vLLM workers may use different virtual
environments.

### UCX logs say CUDA support was not found

This is expected when `engine_kwargs.nixl.device=cpu`, because the transfer
buffers live in host memory. If you set `device=cuda`, the NIXL/UCX build must
support CUDA memory registration.

### NIXL works but is slow

Check the UCX transport line. RDMA should show `rc_mlx5` or another expected
RDMA transport. If it shows TCP only, verify `UCX_NET_DEVICES`, `UCX_TLS`,
container device visibility, and network interface availability on every node.

Also check bucket sizing. Very small buckets increase metadata and synchronization
overhead. Start with `update_weights_bucket_megabytes=2048` for large models and
adjust only after measuring.

### The vLLM preinit marker is missing

Confirm that:

- `policy.generation.checkpoint_engine.enabled=true`
- `policy.generation.checkpoint_engine.backend=nixl`
- the run is using vLLM generation
- the vLLM worker code in the active runtime environment matches the current
  NeMo RL checkout

The NIXL preinit is config-driven. It should not require any
`NRL_VLLM_NIXL_*` environment variables.

### A node or NIC failure causes the job to hang

Use `backend_init_params.ucx_error_handling_mode=peer` and set bounded UCX retry
and keepalive values. Without UCX peer error handling, some transport failures
can look like an indefinitely pending NIXL transfer rather than a clean
`ERR` state.
