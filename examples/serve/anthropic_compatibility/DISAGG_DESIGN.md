# Disaggregated serving through the existing gateway — design proposal

Status: proposal, no code written. Reviewed against the tree at `serli_anthropic_messages`
(3c866f89a) and the reference launchers under `examples/disaggregated/slurm/`.

## Verdict

`gateway.py` needs **no changes at all**. `serve.sh` needs a **second launch path**
inside `cmd_launch`, plus a config-schema extension — roughly 150–250 lines. Everything
that took real effort to get right (fleet relay, election, drain/reclaim, audit, capture,
the Anthropic surface, `start/stop/status`) carries over untouched.

## Why the gateway is free

The disagg proxy is not a different kind of thing from the gateway's point of view. It is
one `host:port` speaking the same protocol:

| gateway needs | disagg proxy provides | evidence |
|---|---|---|
| `/v1/messages` | yes | `openai_disagg_server.py:208` |
| `/v1/messages/count_tokens` | yes | `:209` |
| `/health` | yes | `:210` |
| Anthropic adapter, audit, capture, LCP | yes | `:46-61` imports `convert_anthropic_request`, `create_anthropic_audit_record`, `capture_anthropic_message_request`, `collect_anthropic_lcp_observation` |

So the `TRTLLM_ANTHROPIC_LCP_TRACKING` env, the `anthropic_audit.jsonl` records and the
`anthropic_message_capture/` tree all keep working in disagg mode. The gateway continues to
probe `/health` (`gateway.py:726-742`) and to route to whatever URL the fleet record names.

`/v1/models` is **not** on the proxy (compare the aggregated server's route list) — only a
convenience for our own checks, not something the gateway depends on.

## The one architectural decision: service discovery, not static URLs

The two reference launchers differ in how workers find the proxy, and the choice is
**forced** by how readiness is computed:

```python
# openai_disagg_service.py
async def is_ready(self) -> bool:
    if self._disagg_cluster_manager:                       # service-discovery mode
        return await self._disagg_cluster_manager.is_ready_with_router(
            self._ctx_router.num_prepared_servers,
            self._gen_router.num_prepared_servers)
    return True                                            # static urls mode
```

```python
# disagg_auto_scaling.py
async def is_ready(self) -> bool:
    return (self.current_ctx_worker_num >= self._minimal_ctx_worker_num
            and self.current_gen_worker_num >= self._minimal_gen_worker_num)
```

- **Static `urls:` mode** (`simple_example/launch.slurm`): `is_ready()` is hardcoded `True`,
  so `/health` returns 200 the instant the proxy binds its socket — before a single worker
  exists. The gateway would elect that backend, declare it healthy, and route live traffic
  into a proxy with no workers. It would also force `serve.sh` to resolve node names and
  template `disagg_config.yaml` before launching anything.
- **Service-discovery mode** (`service_discovery_example/launch.slurm`): `/health` gates on
  ctx and gen worker counts *and* on the routers having prepared servers. This is exactly
  the semantics the gateway already assumes.

**Decision: service-discovery mode.** It makes the existing health probe correct rather than
misleading, and it removes hostname templating from `serve.sh` entirely.

Corollary: set `disagg_cluster.minimal_instances` explicitly. It defaults to
`context_servers: 1, generation_servers: 1` (`disagg_utils.py:70-71`), so a deployment with
4 generation workers would go green with 1 of 4 registered.

## Changes to serve.sh

### 1. `resolve()` — config schema (`serve.sh:51-200`)

Today the layout is validated as one number:

```python
world_size = tp * pp * cp
if ntasks != world_size:  die(...)
```

Disagg needs ctx and gen sized independently, plus one CPU task for the proxy. Proposed
deployment YAML, additive and backward compatible — absence of `server.disagg` means today's
aggregated path:

```yaml
server:
  port: 8333                     # the PROXY port; this is what the fleet advertises
  disagg:
    ctx:
      config: server_configs/deepseek_v4_pro_ctx_tep8.yaml
      instances: 1               # each instance is TP*PP*CP ranks from its own config
    gen:
      config: server_configs/deepseek_v4_pro_gen_tep8.yaml
      instances: 1
    ctx_port: 8001
    gen_port: 8002
    cluster_name: aga_pro        # namespace inside the discovery server
    transceiver_backend: UCX     # or DEFAULT / NIXL; must match on both sides
```

New derived values: `CFG_DISAGG=1`, `CFG_CTX_CONFIG`, `CFG_GEN_CONFIG`,
`CFG_CTX_RANKS`, `CFG_GEN_RANKS`, `CFG_CTX_INSTANCES`, `CFG_GEN_INSTANCES`,
`CFG_CTX_PORT`, `CFG_GEN_PORT`, `CFG_CLUSTER_NAME`.

New validation replacing the single equality:

```
ctx_ranks*ctx_instances + gen_ranks*gen_instances  <=  nodes * gpus_per_node
```

`KNOWN_MODELS` (`serve.sh:68`) also needs `deepseek_v4_pro` if Pro is the first target.

### 2. `cmd_launch()` — one srun becomes `1 + n_ctx + n_gen` (`serve.sh:611-700`)

Today: a single backgrounded `srun --ntasks $CFG_WORLD_SIZE` running
`trtllm-llmapi-launch trtllm-serve <model> --host $(hostname) --port $CFG_PORT
--config <cfg> --tool_parser <parser>`.

Disagg: **one srun per server instance**, not three in total. Each instance is its own MPI
world (its own `trtllm-llmapi-launch`), so it cannot share an srun with another instance.
`submit.py` makes this explicit — it allocates per instance and hands each its own port:

```python
assign_servers(allocations, "CTX", num_ctx_servers, ctx_world_size, gpus_per_node)
assign_servers(allocations, "GEN", num_gen_servers, gen_world_size, gpus_per_node)
for i in range(num_servers):
    server_allocation = {"port": port, "nodes": {}}
    assign_server(server_allocation, world_size, gpus_per_node)
    port += 1                      # every instance gets its own port
```

Each instance also gets its own hostfile and gpu_map, and `disaggr_torch.slurm:167` starts
them by reading a generated command file line by line. So 2 ctx + 4 gen is **7 sruns**, and
the ports are a base that increments — not the fixed `ctx_port` / `gen_port` pair sketched
in the schema above, which is only correct for 1 + 1.

The launcher generates the per-instance YAMLs into the attempt dir, starts every srun in the
same allocation with `--overlap`, and waits on all of them.

```
proxy   -w <first node>  -N1 --ntasks 1
        trtllm-serve disaggregated -c <attempt>/disagg_config.yaml

ctx     -N ceil(ctx_ranks/gpus_per_node) --ntasks ctx_ranks
        trtllm-llmapi-launch trtllm-serve <model> --port $CFG_CTX_PORT
          --config <attempt>/ctx_config.yaml
          --disagg_cluster_uri http://<first node>:$CFG_PORT
          --server-role context

gen     ... --server-role generation --tool_parser $CFG_TOOL_PARSER
```

Three things worth calling out:

- **`--tool_parser` goes on the generation workers only.** `trtllm-serve disaggregated` has
  no such flag (its full option list is `--config`, `-m/--metadata_server_config_file`,
  `-t/--server_start_timeout`, `-r/--request_timeout`, `-l/--log_level`,
  `-s/--schedule_style`, `--metrics-log-interval`) and `disagg_utils.py` has no such field.
  The reference launcher reads it per role (`submit.py:599` for gen, `:605` for ctx), so the
  mechanism allows either — but the context server does not need it and **must not have it**:

  ```python
  ctx_request = request.model_copy(update={
      "disaggregated_params": DisaggregatedParams(request_type="context_only", ...),
      "stream": False, "stream_options": None})
  ...
  ctx_response_disagg_params = ctx_response.choices[0].disaggregated_params
  ```

  The proxy takes only `disaggregated_params` off the ctx response; its text never reaches
  the client. Corroborating: `num_postprocess_workers` appears only in the gen section of the
  reference config. Putting a tool parser on ctx would feed it a deliberately truncated
  output — precisely the shape that used to raise `Incomplete DSML invoke` on every request.
- **The proxy must be pinned to a known node** (`-w $(head -1 <(scontrol show hostnames
  $SLURM_JOB_NODELIST))`), because its address is both the fleet URL and the discovery URI.
  That matches what `FLEET_URL` already does (`serve.sh:490` uses `nodes[0]`), so no change
  there.
- **Rank→GPU mapping.** `serve.sh` forces `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, which
  collides as soon as two workers share a node. This does **not** need to be a v2 problem:
  the reference already solves it with `compact_packing`, emitting a per-worker
  `gpu_map_<role>_<id>.txt` of `<rank> <host> <local_gpu_id>` that `start_worker.sh` looks up
  by `SLURM_PROCID` under `srun --distribution=arbitrary` (its comment gives the case of two
  TP=6 ctx workers packed 4+2 / 2+4 across three 4-GPU nodes). Copy that rather than
  reinventing it; whole-node ownership stays available as the simpler default.

### 3. Attempt snapshot (`serve.sh:416-437`)

`start_attempt` copies exactly one file:

```bash
cp "${CFG_SERVER_CONFIG}" "${attempt_dir}/server_config.yaml"
```

Must become ctx + gen + the generated disagg config, so an attempt stays replayable. The
forensic work we have been doing depends on this snapshot being complete.

### 4. Teardown (`serve.sh:568-582`)

```bash
pkill -f '[t]rtllm-serve.*${CFG_PROC_PATTERN}'     # CFG_PROC_PATTERN = basename(model.path)
```

Two defects in disagg: the pattern now matches ctx *and* gen workers (fine, both should
die), but the **proxy's command line contains no model path at all** — it is
`trtllm-serve disaggregated -c .../disagg_config.yaml`. It would survive every signal and
keep holding the port. Needs a third pattern keyed on the attempt dir or the config path.

### 5. Untouched

`write_fleet` (`:340`), `job_is_gone` (`:372`), `clear_fleet`, `cmd_control`,
`gateway_submit`, the whole of `gateway.py`. The fleet record already carries only
`job_id / url / run_dir / state / end_time / heartbeat` — nothing agg-specific.

## Risks

1. ~~**The DSML tool-parser crash follows us.**~~ **Fixed** (2026-08-18, uncommitted). The
   three raises on the streaming path (`deepseekv32_parser.py` :294 / :305 / :314) now
   degrade unparsable DSML to plain text instead of aborting the response, and
   `postproc_worker.py` keeps a `_failed` set so a torn-down stream's later chunks cannot
   re-enter the record creator and assert. `finish()` still raises on a genuinely truncated
   control token — that path is tested and semantically different. 300 tests pass, including
   8 new regression cases that reproduce the production failure against the old parser.
2. **Cache transceiver is a new failure domain.** `cache_transceiver_config.backend` must
   match on ctx and gen, and UCX vs NIXL vs DEFAULT have different node/fabric requirements.
   `slurm/cache_transceiver_test/` exists precisely because this is finicky — run it on aga
   before trusting a real deployment.
3. **Relay across a disagg fleet is untested.** The gateway's drain/reclaim logic counts
   in-flight requests against one backend URL. That still holds (the proxy is the only URL),
   but a mid-stream KV transfer between a draining ctx and a live gen has no equivalent in
   the aggregated path. `lead_time` may need to grow.
4. **`num_scheduled_requests == 1`.** The aggregated Flash deployment never batched beyond 1
   across 106k iterations. If that is a scheduler-level issue rather than a traffic-level
   one, disagg will not fix it and may mask it.

## Suggested staging

1. ~~Fix the DSML parser crash.~~ Done — see Risks 1.
2. Run `slurm/cache_transceiver_test/` on aga to pick a transceiver backend.
3. Implement `resolve()` schema + validation; `serve.sh submit --dry-run` only.
4. Implement the three-srun `cmd_launch`; bring up Pro TEP8 disagg by hand first.
5. Wire the attempt snapshot and teardown; then let the gateway relay it.

## First target

DeepSeek-V4-Pro is the natural first deployment: 850.4 GiB of weights (61 layers, 384
experts top-6, 128 heads, `index_topk 1024`, `num_nextn_predict_layers: 1`), already staged
at `coreai_dlalgo_ci/artifacts/model/nvidia_deepseek-v4-pro-nvfp4/hf/hf-1449d1e_orig`.
Both 4 and 8 divide its head and expert counts. At that size the ctx/gen split is worth
having rather than academic.
