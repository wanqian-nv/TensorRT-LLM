<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Anthropic Compatibility Serving

`serve.sh` is the single entry point. One deployment YAML fully describes a
cluster + model pair; everything else is derived.

```bash
./serve.sh submit --yaml deployments/computelab_glm5.2.yaml --label bringup
./serve.sh status  <run_dir>
./serve.sh restart <run_dir>
./serve.sh quit    <run_dir>
```

To hand the server to other people, put a [gateway](#gateway) in front of it:
one address that survives the serving job being rescheduled, and that decides on
request whether any GPUs are held at all.

```bash
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --submit  # once, lives a week
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --start   # take GPUs, serve
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --stop    # give them back
```

Inside an existing `salloc` allocation, skip `sbatch` and run the controller
directly:

```bash
nohup ./serve.sh run --yaml deployments/computelab_glm5.2.yaml \
    > controller.log 2>&1 &
```

The controller owns the server: when it exits, it tears the server down. Under
`sbatch` that is what you want, since the controller *is* the job. From an
interactive shell, detach it with `nohup` — otherwise closing the terminal
sends SIGHUP and kills a server that may still be loading weights.

## Layout

```text
serve.sh                            launcher: submit / run / launch / gateway / start|restart|stop|quit|status
                                    (gateway --start|--stop|--status drives a running gateway)
gateway.py                          the gateway process serve.sh starts (standard library only)
deployments/
  computelab_glm5.2.yaml            one file per cluster + model pair
  computelab_deepseek_v4.yaml
  coreai_deepseek_v4_flash.yaml
  gateway_users.txt                 who may use the gateway, one username per line
                                    (enforcement currently commented out -- see below)
  server_configs/                   trtllm-serve --config files (one per model)
analysis/                           audit-log analysis and plotting
```

## Supported models

`model.name` must be one of `glm5.2`, `deepseek_v4`, or `deepseek_v4_flash` —
the whitelist lives in `serve.sh`, so a typo cannot silently produce a job,
container and trace directory that look almost right. Everything else about the
model comes from the YAML:

```yaml
cluster_name: computelab

model:
  name: deepseek_v4
  path: /home/scratch.trt_llm_data/llm-models/DeepSeek-V4-Pro
  tool_parser: deepseek_v4

server:
  config: server_configs/deepseek_v4_agg_tep8.yaml
  extra_args: []      # extra trtllm-serve flags, if a model ever needs them
```

Adding a checkpoint means one new deployment YAML, one file under
`server_configs/`, and one entry in the `serve.sh` whitelist.

## What the YAML holds, and what it does not

Everything the run needs is in the deployment YAML: `cluster_name`, `repo_dir`,
`model.{name, path, tool_parser}`, `slurm.{account, partition, reservation, qos,
segment, time, nodes, gpus_per_node, extra_args}`, `container.{image, mounts}`,
`server.{config, port, install_repo, numactl, capture, extra_args, env}`,
`trace.root`, and the optional `gateway.{port, partition, time, account, qos,
users, lead_time, script, extra_args}`.

`gateway` is optional and a YAML without it still resolves; only
`serve.sh gateway` needs it. It belongs here rather than on the command line
because the gateway and the serving jobs have to agree on a rendezvous
directory, and both derive it from this file's `trace.root`. Passing it
separately would let them point at different directories with no error at all --
the gateway would simply report zero backends forever.

The layout is explicit: `slurm.nodes × slurm.gpus_per_node` gives `ntasks`, and
the server config's `tensor_parallel_size × pipeline_parallel_size ×
context_parallel_size` must agree with it. A mismatch is a hard error, so
changing TP without changing the node count fails at submit time rather than
half-way through a job.

These are derived and must not be written by hand:

| Derived | From |
|---|---|
| `ntasks`, `ntasks-per-node`, `gres` | `slurm.nodes` and `slurm.gpus_per_node` |
| Slurm job name, container name, `--output` | `<cluster_name>_<model.name>` + job ID |
| pkill pattern used to clear a previous server | `basename(model.path)` — that is what appears on the `trtllm-serve` command line, unlike `model.name` |
| Trace directory | `${USER}_$(date +%m%d%H)_${SLURM_JOB_ID}_<cluster_name>_<model.name>[_<label>]` |
| `TRTLLM_ANTHROPIC_AUDIT_LOG`, `TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR` | pointed at the attempt directory; on unless `server.capture: false` |
| `TRTLLM_ANTHROPIC_LCP_TRACKING` | always on |
| Fleet registration directory | `<trace.root>/_fleet/<cluster_name>_<model.name>` |

One code path covers any node count, so a one-node and a two-node layout differ
only by `slurm.nodes`.

`server.env` overrides the built-in defaults: `TLLM_LOG_LEVEL=INFO`,
`TRTLLM_{SERVER,WORKER}_DISABLE_GC=1`, `TRTLLM_ENABLE_PDL=1`,
`ENROOT_ALLOW_DEV=yes`, `NCCL_GRAPH_MIXING_SUPPORT=0`, `MIMALLOC_PURGE_DELAY=0`.

## Run directory

```text
<trace.root>/serli_073023_3378564_computelab_glm5.2_bringup/
├── deployment.yaml  run_metadata.txt  server_url
├── control/                 job_id, nodes, attempt, state, current_attempt_dir
└── attempt-001/
    ├── launch_cmd.sh        the srun command lines this attempt actually ran
    ├── server_config.yaml   snapshot of the config this attempt served with
    ├── launcher.log  install.log  server.log
    ├── anthropic_audit.jsonl
    └── anthropic_message_capture/
```

The server config is snapshotted per attempt rather than per run: it can be
edited between a `stop` and the next `start`, and the attempt copy is the one
that ran.

`run_metadata.txt` records the branch, commit, model path, container, topology,
and node list, so a run is reproducible from the directory alone.

> The capture directory holds raw `/v1/messages` bodies. Treat it as sensitive.
> Set `server.capture: false` before sharing the URL, or every user's prompts
> land in your run directory.

## Gateway

A serving job runs for hours and lands wherever Slurm puts it, so its URL
changes every time it is rescheduled. That is fine for one person and unworkable
for ten. The gateway holds one address and forwards to whichever backend is
current, so `ANTHROPIC_BASE_URL` is written once and never edited again.

```text
users ── http://<gateway>:8333 ──▶ gateway ──▶ trtllm-serve on umbriel-b200-0XX
                                      │             (4h job, rotates)
                                      └── reads <trace.root>/_fleet/<name>/
```

It needs no GPU, so it runs as a CPU-only Slurm job on a partition that allows
seven days -- long enough to outlive dozens of serving jobs. The computelab
deployments pin their gateways to named nodes. The coreai deployment deliberately
does not pin one because CPU capacity is scarce; Slurm chooses the node, and the
URL remains valid for that gateway job's seven-day lifetime:

| Deployment | Address |
|---|---|
| `computelab_glm5.2` | `http://lego-c2-qs-26:8333` |
| `computelab_deepseek_v4` | `http://lego-c2-qs-36:8333` |
| `coreai_deepseek_v4_flash` | read `<trace.root>/_fleet/coreai_deepseek_v4_flash/gateway_url` after submission |

The two computelab gateways sit on different nodes deliberately. Both would fit
on one -- 4 cores and ~13 GB each against 144 and 490 -- but then a single node
failure takes both models' front doors down together.

### Starting it

Once per deployment, then only when the gateway's seven days run out:

```bash
vi deployments/gateway_users.txt                                        # one username per line (not enforced right now)
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --submit
curl -s http://lego-c2-qs-26:8333/_gateway/health
# {"status": "stopped", "deployment": "computelab_glm5.2", "active": null, "desired": "stopped", ...}
```

For the unpinned coreai deployment, wait for Slurm to start the job and read the
chosen hostname instead of assuming one:

```bash
./serve.sh gateway --yaml deployments/coreai_deepseek_v4_flash.yaml --submit
cat /lustre/fsw/portfolios/coreai/users/serli/claude-traces/_fleet/coreai_deepseek_v4_flash/gateway_url
```

`stopped` is the correct answer here, and not a fault: the gateway is up and it
is deliberately holding nothing. It costs four cores, so it is meant to sit
there for a week; the GPUs are the expensive half and they are only taken when
somebody asks.

### Starting and stopping the serving job

```bash
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --start
# {"action": "submitted", "desired": "running", "job_id": "3452201", ...}
```

That submits the serving job, and from then on the gateway keeps one up by
itself: it relays a successor in 45 minutes before the wall clock, moves routing
when the successor is healthy, reclaims the predecessor after three minutes, and
resubmits outright if the fleet is ever lost. Weights take a while, and
`/health` flips when they are loaded:

```bash
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --status
# {"status": "ok", "active": "3452201", "desired": "running", "pending": null, ...}
```

Giving the GPUs back is the same command with `--stop`:

```bash
./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml --stop
# {"action": "stopped", "released": ["3452201"], "still_allocated": [], "failed": [], ...}
```

Every serving job is released through `serve.sh quit`, which stops the server
and lets the controller exit, which under `sbatch` ends the job and returns the
nodes. A job that never got far enough to have a run directory is `scancel`ed
instead, and so is one whose `quit` fails outright. Routing stops in the same
instant, before anything is torn down, so a client sees a clean `503` rather
than a truncated reply. The gateway keeps running on the same address.

`quit` only writes a control file, so it returning cleanly does not by itself
mean the GPUs came back. `--stop` waits for Slurm to agree before it says
`released`, and reports anything that stopped serving but is still holding
nodes under `still_allocated` instead of claiming otherwise:

- `released` -- the server is down and Slurm has the nodes back.
- `still_allocated` -- the server is down, the allocation is not this script's
  to end. Normal for a controller started by hand inside an `salloc`, where the
  allocation is your shell. `scancel` it yourself if you want the nodes; the
  gateway will not, because guessing there ends your session too.
- `failed` -- neither `quit` nor `scancel` worked. The job id is in the reply
  and in the log; it is still routable, because it is still serving.

Before it sends either POST, `--start` and `--stop` ask the address in
`gateway_url` which deployment it serves and refuse if the answer is not this
one. That file outlives the gateway that wrote it and a pinned hostname can come
back hosting something else; releasing another model's GPUs is not a mistake
worth being able to make. `--status` skips the check, since it changes nothing
and is what you reach for when you suspect the file is stale.

`--start` is idempotent: it adopts a serving job that is already there instead
of submitting a second one, including one submitted by hand with `serve.sh
submit`, and including one that is still loading weights. So retrying it when
you are not sure the first call landed is safe.

The endpoints behind those three flags are plain HTTP, if you would rather not
go through `serve.sh`:

```bash
curl -XPOST http://lego-c2-qs-26:8333/_gateway/start_server
curl -XPOST http://lego-c2-qs-26:8333/_gateway/stop_server
curl -XPOST 'http://lego-c2-qs-26:8333/_gateway/start_server?label=bringup'
```

They are unauthenticated, like `/v1/*`. `POST` is required and nothing else is
accepted, so a mistyped `GET` or a link preview cannot release an allocation --
which is the only guard there is. Anyone who can reach the port can start and
stop the GPUs; that is the same trust boundary the model itself already sits
behind on this network.

### Using it

What to hand to somebody who wants to use the model:

```bash
export ANTHROPIC_BASE_URL=http://lego-c2-qs-26:8333
export ANTHROPIC_API_KEY=$USER        # your own username, not a token
export ANTHROPIC_MODEL=<name from /v1/models>

claude
```

For a pinned deployment, this is worth putting in `~/.bashrc`. For an unpinned
one, update it from `gateway_url` whenever the gateway job is resubmitted. The
model name comes from the server:

```bash
curl -s -H "x-api-key: $USER" http://lego-c2-qs-26:8333/v1/models
```

One request, to check the whole path end to end:

```bash
curl -s http://lego-c2-qs-26:8333/v1/messages \
  -H "x-api-key: $USER" -H "content-type: application/json" \
  -d '{"model":"<name>","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}'
```

The Claude client may run on another cluster. It does not need access to the
backend nodes or the fleet directory; it only needs DNS and TCP connectivity to
the gateway host and port:

```text
Claude client on another cluster ──▶ <gateway-host>:8333 ──▶ active backend
```

Check that path from the machine where Claude will run:

```bash
getent hosts lego-c2-qs-26
curl --connect-timeout 5 http://lego-c2-qs-26:8333/_gateway/health
curl --connect-timeout 5 \
  -H "x-api-key: $USER" \
  http://lego-c2-qs-26:8333/v1/models
```

If those commands work, use the same environment variables shown above. If DNS
does not resolve the short hostname, use its FQDN or the gateway CNAME instead.
Network policy may allow login nodes to reach the gateway while isolating compute
nodes, so run the checks from the same kind of node that will run Claude.

When direct routing is unavailable, tunnel through a login node that can reach
the gateway:

```bash
ssh -N -L 8333:lego-c2-qs-26:8333 <reachable-login-node>
```

Keep that tunnel running and point Claude at the local end:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8333
export ANTHROPIC_API_KEY=$USER
claude
```

The username *is* the key. That makes the allowlist an attribution trail rather
than authentication -- a name is guessable, so the access log records who asked,
not who proved it. Reasonable among colleagues on the internal network; do not
read the log as an audit record.

What the three failures mean:

| Response | Meaning | What to do |
|---|---|---|
| `401` | Not on the allowlist | **The allowlist check is currently commented out in `Gateway.handle`, so `/v1/*` no longer returns this.** With it on: ask for a line in `gateway_users.txt`; it takes effect without a restart |
| `503` | Backends are rotating, or none has started | Wait -- clients retry on their own. The message says which of the two it is; `no serving job is running` means nobody has called `--start` |
| `overloaded_error` partway through a reply | The backend went away mid-stream | Resend; the partial reply cannot be recovered |

### Registration and election

Each serving job writes `<trace.root>/_fleet/<cluster>_<model>/<job_id>.json` and
refreshes a heartbeat every 10s. One file per job means one writer per file, so
nothing locks; the write is a rename, so a reader never sees half a record. The
directory is namespaced by cluster and model because the deployments share a
trace root, and a flat directory would let one model's gateway route to another
model's backend.

The gateway elects **the healthy backend with the latest end time**. Handover
needs no orchestration: a successor outlives the job it replaces, so it wins the
election by itself the moment it answers `/health`. Since `trtllm-serve` only
opens its port after the weights are loaded, answering at all already means
ready.

### Relay

45 minutes before the wall clock the gateway submits the successor, then hands
routing over as soon as it is healthy. In-flight streams stay on the old backend
-- they are bound to sockets that nobody touches -- and finish normally.

Reclaiming the predecessor waits until the successor has been healthy for three
minutes. Routing is reversible and moves immediately; releasing an allocation is
not, so it does not move on a single probe. A successor that passes one check
and then dies would otherwise take its predecessor with it and leave nothing
serving until the next job finishes loading.

For the same reason, a backend that merely fails a probe is never reclaimed. It
stays in the table, keeps being probed, and is elected again when it recovers --
which is what `serve.sh restart` looks like from here.

### Failure handling

| Situation | What the gateway does |
|---|---|
| Connection refused, or `/health` non-200 | Out of rotation at once; the process is gone |
| `/health` times out | Tolerated for 20 consecutive probes |
| No healthy backend | `503` with `Retry-After`, so clients back off instead of erroring |
| Stopped on request | `503` naming `start_server`, and `/_gateway/health` reports `stopped` rather than `no_backend` -- nothing to page on |
| Serving a job nobody started | `/_gateway/health` reports `unmanaged` and the log warns every sweep; traffic is unaffected until that job's wall clock |
| Fleet lost while running | Resubmits on its own, backing off 30s, 60s, 120s ... up to 10 minutes if `sbatch` keeps refusing |
| Backend disappears mid-stream | Injects an Anthropic `error` event, so the client sees a protocol error rather than a reset |
| Job vanishes without deregistering | Heartbeat ages out after 30s |

A timeout is tolerated so long because a server too busy to answer a health
check is usually still generating tokens fine, and taking it out of rotation
would turn "slow" into "503" with nowhere better to send the traffic.

### Inspecting it

```bash
curl -s http://<gateway>:8333/_gateway/health                       # unauthenticated, for monitoring
curl -s -H "x-api-key: $USER" http://<gateway>:8333/_gateway/fleet  # every backend, with its state
```

`/_gateway/health` answers one of four, plus `deployment`, `desired` and the id
of any job still loading:

| `status` | Meaning |
|---|---|
| `ok` | Serving, and this gateway owns the job: it will relay a successor in |
| `unmanaged` | Serving a job it merely found. Nothing will be submitted before that job's wall clock -- one `--start` adopts it |
| `no_backend` | A serving job was asked for and is not up yet |
| `stopped` | None was asked for. Not a fault, nothing to page on |

`/_gateway/fleet` reports `healthy`, `healthy_for_s`, `probe_timeouts`,
`inflight`, `ends_in_s`, `superseded`, `draining` and `stopping` per backend,
which is enough to see exactly where a handover is stuck, and
`submit_retry_in_s` when `sbatch` has been refusing the job.

### Limits

- **The pinned node is a dynamic one.** Every node in this partition reports
  `DYNAMIC_NORM`, meaning it can be deregistered from the cluster. If that
  happens the gateway job dies and `sbatch` rejects the next one outright --
  loudly, at least, rather than queueing forever. Recovery is to point
  `gateway.extra_args` at another `lego-c2-qs-*` node, which changes the URL.
  A CNAME at https://itss.nvidia.com/dns/hostrecord (self-service for
  3rd-level `*.nvidia.com` names) turns that into a two-minute edit nobody else
  has to hear about. Worth having before it is needed.
- **A restarted gateway comes up idle.** It boots holding nothing, by design, so
  a gateway whose seven days ran out -- or that was cancelled with its serving
  job, or requeued after a node failure -- needs `--start` after it is
  resubmitted. It still routes to a serving job it finds already running, so
  nobody loses access; what it will not do is relay that job in before its wall
  clock, or reclaim it. That state is `status: "unmanaged"` in
  `/_gateway/health` and a warning in the log on every supervisor sweep, so it
  is alertable rather than something to remember. One `--start` adopts the job
  and clears it. The alternative -- persisting `desired` across restarts --
  would also resurrect a deployment somebody had deliberately stopped, so the
  idle boot is the safer default and the health status is the mitigation.
- **A third deployment needs its own node or port.** The two checked-in gateways
  both use 8333 and stay out of each other's way by being pinned to different
  nodes; a new one has to keep that true.
