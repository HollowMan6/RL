# Delta-Compressed Collective Refit

Delta-compressed collective refit reduces the amount of weight data sent from
policy workers to vLLM generation workers in non-colocated deployments. Instead
of sending a full dense weight update every time, NeMo RL sends a full baseline
periodically and sends additive deltas for eligible floating-point weight chunks
between those full refreshes.

Use this feature when policy and generation run on separate GPU resources and
the same vLLM generation workers are refitted repeatedly during training.

## Support Matrix

Delta-compressed refit currently supports:

- non-colocated vLLM generation
- collective NCCL refit
- floating-point model weights

It does not currently support:

- colocated IPC/ZMQ refit
- SGLang generation
- ModelOpt quantized vLLM weights
- vLLM FP8 model weights

## Configuration

Enable the feature under `policy.generation.delta_compression`:

```yaml
policy:
  generation:
    backend: "vllm"
    colocated:
      enabled: false
      resources:
        num_nodes: 1
        gpus_per_node: 8
    delta_compression:
      enabled: true
      dtype: "bfloat16"
      full_sync_interval: 20
      sparse_bucket_size_bytes: 1073741824
      delta_load_batch_size_bytes: 536870912
```

| Field | Description |
|---|---|
| `enabled` | Enables delta-compressed collective refit. |
| `dtype` | Dtype used for delta tensors. Typical values are `"bfloat16"` or `"float32"`. |
| `full_sync_interval` | Force a dense full sync every N successful syncs. The first sync is always full. With `20`, sync 1 is full, syncs 2 through 20 are delta-eligible, and sync 21 is full. |
| `sparse_bucket_size_bytes` | Maximum sparse payload bytes to bucket before broadcasting. Try `536870912` for 512 MiB or `1073741824` for 1 GiB. |
| `delta_load_batch_size_bytes` | Maximum decoded delta tensor bytes to batch before calling the vLLM weight loader. Try `536870912` for 512 MiB. |

NeMo RL falls back to a dense full update for chunks that cannot be represented
efficiently as sparse deltas, including non-floating tensors or sparse payloads
that would be larger than the dense payload.

## Try It With `tools/refit_verifier.py`

`tools/refit_verifier.py` creates policy and vLLM workers, performs one or more
refits, generates with vLLM, and compares generated-token logprobs against the
policy backend. Delta compression requires `--non_colocated`.

For a two-node Qwen3-30B-A3B verifier run with one 8-GPU policy node and one
8-GPU generation node:

```bash
uv run --extra mcore python3 tools/refit_verifier.py \
  --model_name /path/to/Qwen3-30B-A3B-Base \
  --non_colocated \
  --policy_num_nodes 1 \
  --generation_num_nodes 1 \
  --policy_gpus_per_node 8 \
  --generation_gpus_per_node 8 \
  --tp_size 1 \
  --ep_size 8 \
  --pp_size 1 \
  --vllm_tp_size 8 \
  --vllm_ep_size 8 \
  --vllm_pp_size 1 \
  --max_new_tokens 1 \
  --max_sequence_length 128 \
  --num_refits 3 \
  --enable_delta_compression \
  --delta_sparse_bucket_size_bytes 536870912 \
  --delta_load_batch_size_bytes 536870912 \
  --vllm_gpu_memory_utilization 0.8
```

Expected success markers include:

```text
Collective refit initialized
Refit pass 1/3
Refit pass 2/3
Refit pass 3/3
Model refitting completed
Script completed successfully!
```

The verifier also prints the mean and maximum absolute logprob differences for
the generated tokens. Use those values to confirm the refitted vLLM model still
matches the policy backend within expected backend and precision tolerance.

## Compare Against Full Weight Transfer

Run the same verifier once without `--enable_delta_compression` and once with
it enabled. Keep the model path, topology, sequence lengths, and number of
refits the same.

Full transfer baseline:

```bash
uv run --extra mcore python3 tools/refit_verifier.py \
  --model_name /path/to/Qwen3-30B-A3B-Base \
  --non_colocated \
  --policy_num_nodes 1 \
  --generation_num_nodes 1 \
  --policy_gpus_per_node 8 \
  --generation_gpus_per_node 8 \
  --tp_size 1 \
  --ep_size 8 \
  --pp_size 1 \
  --vllm_tp_size 8 \
  --vllm_ep_size 8 \
  --vllm_pp_size 1 \
  --max_new_tokens 1 \
  --max_sequence_length 128 \
  --num_refits 3 \
  --vllm_gpu_memory_utilization 0.8
```

Delta transfer:

```bash
uv run --extra mcore python3 tools/refit_verifier.py \
  --model_name /path/to/Qwen3-30B-A3B-Base \
  --non_colocated \
  --policy_num_nodes 1 \
  --generation_num_nodes 1 \
  --policy_gpus_per_node 8 \
  --generation_gpus_per_node 8 \
  --tp_size 1 \
  --ep_size 8 \
  --pp_size 1 \
  --vllm_tp_size 8 \
  --vllm_ep_size 8 \
  --vllm_pp_size 1 \
  --max_new_tokens 1 \
  --max_sequence_length 128 \
  --num_refits 3 \
  --enable_delta_compression \
  --delta_sparse_bucket_size_bytes 536870912 \
  --delta_load_batch_size_bytes 536870912 \
  --vllm_gpu_memory_utilization 0.8
```

The first delta-compressed refit is a full baseline sync by design. Compare
later refits when you want to measure delta-transfer behavior. Repeating refit
without any weight changes mostly exercises the control path; for representative
payload sizes, benchmark after real optimizer steps or otherwise make the policy
weights change between refits.

## Measure Refit Time in Training

For end-to-end GRPO runs, compare the timing entry named
`prepare_for_generation/transfer_and_update_weights` in the training log. That
timer covers the refit transfer and vLLM update phase. Keep the recipe fixed and
run two jobs:

1. Full transfer, with no `policy.generation.delta_compression` block.
2. Delta transfer, with `policy.generation.delta_compression.enabled: true`.

Compare runs after the first refit, since the first delta-compressed refit is a
full baseline sync. In async or exposed-generation workflows, also check any
`weight_sync` timing entry because those paths can report the synchronizer phase
under that label.

## Slurm Launch Pattern

When launching the verifier through `ray.sub`, mount the model path into the
container and pass the verifier command as `COMMAND`:

```bash
COMMAND="uv run --extra mcore python3 tools/refit_verifier.py \
  --model_name /path/to/Qwen3-30B-A3B-Base \
  --non_colocated \
  --policy_num_nodes 1 \
  --generation_num_nodes 1 \
  --policy_gpus_per_node 8 \
  --generation_gpus_per_node 8 \
  --tp_size 1 \
  --ep_size 8 \
  --pp_size 1 \
  --vllm_tp_size 8 \
  --vllm_ep_size 8 \
  --vllm_pp_size 1 \
  --num_refits 3 \
  --enable_delta_compression \
  --delta_sparse_bucket_size_bytes 536870912 \
  --delta_load_batch_size_bytes 536870912" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD,/path/to/models:/path/to/models" \
sbatch \
  --nodes=2 \
  --gres=gpu:8 \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_PARTITION \
  --time=2:00:00 \
  ray.sub
```

Use the normal NeMo RL container or environment for these runs. Dependency
import failures for CUDA bindings, NCCL bindings, vLLM, Megatron, Ray, or
logging libraries indicate an environment issue, not a delta-compression issue.

## Tuning Guidance

Start with:

- `full_sync_interval: 20`
- `sparse_bucket_size_bytes: 536870912` or `1073741824`
- `delta_load_batch_size_bytes: 536870912`

Lower `sparse_bucket_size_bytes` can let the receiver start sparse decode and
weight loading earlier while later sparse broadcasts are still running. Too
small a value can increase header, packing, and loader overhead. Use end-to-end
refit timing to choose the best value for the model and interconnect.

The feature keeps an additional CPU baseline copy on the policy side for the
floating-point tensors it delta-encodes. During a sync, it also uses transient
GPU buffers for staged baseline reads, sparse payloads, decoded deltas, and
vLLM load batches. The transient memory is bounded by the normal refit chunking
and the bucket/load-batch settings above.

## Profiling

For overlap and timeline analysis, use the NeMo RL Nsight Systems workflow in
[Nsight Systems Profiling](../nsys-profiling.md). In a healthy delta run, sparse
encoding and payload broadcasts should overlap on the policy side, and sparse
decode/load can overlap with later payload receives when the sparse bucket and
load-batch sizes are not too large.

## Troubleshooting

- `--enable_delta_compression requires --non_colocated`: delta compression only
  works with collective refit.
- The first refit is not faster: this is expected because it establishes the
  full baseline.
- Delta refits are not faster: confirm weights actually changed between refits,
  try 512 MiB or 1 GiB sparse buckets, and compare against a full-transfer run
  with the same topology.
- Out-of-memory during baseline staging: reduce model parallel shard size if
  possible, reduce bucket/load sizes, or lower other vLLM memory pressure such as
  `vllm_gpu_memory_utilization`.
- Logprob differences are high: first confirm the same model path, tokenizer,
  precision, TP/EP/PP settings, prompt, and sequence lengths are used for both
  policy and vLLM.
