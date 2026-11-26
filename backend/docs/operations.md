# Operations

## Process topology

Run the Render API, frontend, dedicated PostgreSQL runtime database, AWS S3 bucket, and at least
one portable Linux worker as distinct services. Render does not run the worker in the current
topology. The API image has core dependencies only. The worker image has ML dependencies and uses
an external rootless Docker daemon to launch a fresh candidate container for each evaluation.

The current deployment uses a separate logical `agent_k_runtime` database on the existing
AGENT-K managed PostgreSQL instance; the frontend continues to use `agent_k_db`. Set the API's
`AGENT_K_DATABASE_URL` manually with the runtime database path. Render's `fromDatabase` reference
selects the instance's original database and cannot derive this alternate path. This arrangement
separates runtime tables without adding a second paid instance; it shares instance capacity.

Render's `/health` check establishes API process availability. Without S3 configuration, `/ready`
and mission creation return 503 while authenticated competition discovery remains available.
The health response includes `X-Revision`, and the frontend's `/api/version` returns its revision;
CI waits for both to match the pushed commit before running live browser smoke tests.
Native Render builds recreate `backend/.venv` before syncing the lockfile because restored
build caches can retain virtual environments whose Python executable is no longer present.

The API alone holds PostgreSQL and AWS credentials. Workers authenticate to the API with a
rotatable worker token and transfer artifacts through short-lived, checksum-bound S3 URLs.
Candidate containers receive neither API/provider credentials nor hidden validation or acceptance
labels.

## Build and pin images

Run image commands from the repository root through the same rootless Docker context that will run
candidates.

```bash
docker build --file backend/Dockerfile.candidate --tag agent-k-candidate:local backend
docker build --file backend/Dockerfile.worker --tag agent-k-worker:local backend
```

For a local gate, use the immutable image ID returned by Docker:

```bash
export AGENT_K_SANDBOX_IMAGE="$(docker image inspect --format '{{.Id}}' agent-k-candidate:local)"
```

`AGENT_K_SANDBOX_IMAGE` accepts a local `sha256:<image-id>` or a registry
`registry.example/agent-k-candidate@sha256:<manifest-digest>`. Production workers must use the
registry digest produced after the candidate image is pushed, not a mutable tag. The worker checks
that the pinned image is already present before claiming jobs; it never pulls an image after a job
has started.

## Rootless Docker worker host

Use a dedicated Linux account. Install Docker Engine rootless prerequisites and the rootless setup
tool using the host's package manager, then initialize and start the user daemon:

```bash
dockerd-rootless-setuptool.sh install
systemctl --user enable --now docker
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock"
docker info --format '{{json .SecurityOptions}}'
docker info --format 'cgroup={{.CgroupVersion}} driver={{.CgroupDriver}}'
```

The security options must include `rootless`. Resource enforcement also requires cgroup v2 with
the systemd cgroup driver and delegated controllers. AGENT-K fails worker startup if these checks,
the pinned candidate image, or the Docker CLI is unavailable. See Docker's official
[rootless mode](https://docs.docker.com/engine/security/rootless/) and
[rootless tips](https://docs.docker.com/engine/security/rootless/tips/) for host prerequisites.

The host work directory must appear at the same absolute path inside the worker container. This is
required because the nested candidate bind mounts are resolved by the host daemon:

```bash
export AGENT_K_WORK_HOST_ROOT="$HOME/.local/share/agent-k-worker"
install -d -m 700 \
  "$AGENT_K_WORK_HOST_ROOT/work" \
  "$AGENT_K_WORK_HOST_ROOT/assets/kumo" \
  "$AGENT_K_WORK_HOST_ROOT/assets/transformer" \
  "$AGENT_K_WORK_HOST_ROOT/assets/vision"
export AGENT_K_DOCKER_SOCKET="$XDG_RUNTIME_DIR/docker.sock"
```

Store worker settings and provider/Kaggle credentials in a file readable only by the worker
account. Do not put PostgreSQL, AWS, gateway, or frontend secrets in it.

```dotenv
AGENT_K_API_URL=https://agent-k-api.onrender.com
AGENT_K_WORKER_ID=linux-worker-01
AGENT_K_WORKER_TOKEN=replace-with-the-rotated-worker-token
AGENT_K_WORKER_CAPABILITIES=cpu,cuda
AGENT_K_WORK_ROOT=/absolute/host/path/.local/share/agent-k-worker/work
AGENT_K_SANDBOX_IMAGE=registry.example/agent-k-candidate@sha256:replace-with-digest
AGENT_K_MODEL=openai:replace-with-model
AGENT_K_GPU_DEVICES=["0"]
AGENT_K_KUMO_CACHE=/absolute/host/path/.local/share/agent-k-worker/assets/kumo
AGENT_K_TRANSFORMER_CACHE=/absolute/host/path/.local/share/agent-k-worker/assets/transformer
AGENT_K_VISION_WEIGHTS=/absolute/host/path/.local/share/agent-k-worker/assets/vision/resnet18.pt
KAGGLE_USERNAME=replace-with-username
KAGGLE_KEY=replace-with-key
OPENAI_API_KEY=replace-with-provider-key
```

Worker capabilities are advertised during claim, but newly enqueued mission jobs do not yet carry
capability requirements. Do not mix CPU-only and CUDA-required workers on the same queue: a
CPU-only worker can otherwise claim a Kumo or GPU vision mission. Until capability-aware enqueue
is implemented, use workers that satisfy every model requested by that deployment.

Start the portable worker container with the rootless socket and identical host work path:

```bash
chmod 600 /secure/path/agent-k-worker.env
docker run --rm --name agent-k-worker-01 \
  --user 0:0 \
  --env-file /secure/path/agent-k-worker.env \
  --env DOCKER_HOST=unix:///run/agent-k/docker.sock \
  --env XDG_RUNTIME_DIR=/run/agent-k \
  --mount "type=bind,src=$AGENT_K_DOCKER_SOCKET,dst=/run/agent-k/docker.sock" \
  --mount "type=bind,src=$AGENT_K_WORK_HOST_ROOT,dst=$AGENT_K_WORK_HOST_ROOT" \
  agent-k-worker:local
```

Under a validated rootless daemon, container UID 0 maps to the unprivileged daemon account on the
host. This mapping lets the worker access that account's mode-`0700` workspace and rootless socket
without making either path group- or world-readable. Do not use host numeric UID/GID as the
container user: rootless user namespaces map that numeric container UID into the subordinate ID
range instead.

The Docker image command uses `python -m agent_k.worker run`. `run` is optional; an installed
worker can also be started with `python -m agent_k.worker`.

## CUDA and offline assets

GPU devices are never exposed wholesale. `AGENT_K_GPU_DEVICES` is a JSON array containing explicit
NVIDIA indices, GPU UUIDs, or MIG device IDs. Candidate containers receive only those devices via
Docker's `--gpus device=...` option.

Configure the NVIDIA Container Toolkit for the rootless daemon and verify it before enabling a GPU
worker:

```bash
nvidia-ctk runtime configure \
  --runtime=docker \
  --config="$HOME/.config/docker/daemon.json"
systemctl --user restart docker
docker info --format '{{json .Runtimes}}'
docker run --rm --gpus '"device=0"' "$PINNED_CUDA_VALIDATION_IMAGE" nvidia-smi
```

The runtime listing must contain `nvidia`; the validation image must itself be pinned by digest.
Kumo production candidates default to `cuda_required`. CPU Kumo is only an explicit small-runtime
or test policy, not an automatic production fallback.

Offline assets are copied into the candidate's read-only input tree and are never downloaded by a
candidate:

- `AGENT_K_KUMO_CACHE` is a directory containing `manifest.json` and `hf-cache/`; the manifest must
  identify the supported `nvidia/Kumo-Tabular` revision and model size.
- `AGENT_K_TRANSFORMER_CACHE` is a complete local Hugging Face model/tokenizer directory.
- `AGENT_K_VISION_WEIGHTS` is a local ResNet18 state-dictionary file.

Worker startup rejects configured asset paths that do not exist.

## AWS S3

Configure the API, not workers, with:

```dotenv
AGENT_K_S3_BUCKET=replace-with-private-bucket
AGENT_K_S3_REGION=us-west-2
AWS_ACCESS_KEY_ID=replace-with-api-role-key
AWS_SECRET_ACCESS_KEY=replace-with-api-role-secret
AWS_SESSION_TOKEN=replace-when-using-temporary-credentials
```

Prefer short-lived credentials supplied to the API service. Restrict the role to the configured
bucket/prefix and the object operations used by AGENT-K: checksum-bound `PutObject`, `GetObject`
(including `HeadObject` authorization), and migration-source reads. Keep public access blocked and
enforce TLS. Workers must not receive these variables.

Uploads do not become resumable merely because a worker completed an HTTP PUT. The API performs an
S3 `HEAD` with checksum mode enabled, compares the reserved size and SHA-256 checksum, and only then
marks the artifact available. A completed job is rejected if any referenced artifact is not in
that verified state.

## Mission controls and leases

Pause and cancel are requests, not immediate process kills. The worker observes the directive and
creates a durable checkpoint first. Pause emits a nonterminal state delta, rewrites the queued job
to `resume` from the latest snapshot, and remains unclaimable until an owner resumes it. Cancel is
terminal. Retryable worker failures remain nonterminal; the store inserts `mission-complete`
atomically only when a failure is actually terminal.

Set `AGENT_K_HEARTBEAT_SECONDS` to less than half `AGENT_K_LEASE_SECONDS`; invalid settings fail
worker startup. Attempt and lease IDs fence stale workers. An expired attempt resumes from its
latest verified snapshot when one exists.

Each candidate has managed, worker-owner, and server-issued lease labels. A worker holds an
exclusive lock on its private work root, reconciles only containers carrying its stable owner
label at startup, and removes only that lease's containers before deleting an attempt directory.
Never run a broad Docker prune as part of worker recovery, and never share one work root between
worker processes.

## Key rotation

Frontend identity and worker credentials use separate key families.

1. Put the old value in the corresponding `*_PREVIOUS_*` setting.
2. Deploy the new current value to all producers and verifiers.
3. Wait longer than the maximum frontend token lifetime and worker deployment overlap.
4. Remove the previous value.

Never reuse the Auth.js secret, gateway secret, worker token, provider keys, or cloud credentials.

## Observability

`agent_k.observability.configure_observability` configures Logfire explicitly at API, worker, and
mission-subprocess startup with content and binary-content capture disabled. Logs use identifiers
and structured status. They must not contain prompts, dataset rows, snapshot payloads, bearer
tokens, signed URLs, provider keys, or cloud credentials. Mission-child stdout and stderr are
drained into bounded, mode-`0600` files inside the private attempt directory; candidate stdout is
never used as fitness evidence.

## Live gates

Run backend commands from `backend/` with `.venv/bin/python`.

Real PostgreSQL queue, fencing, pause/resume, and terminal-event integration:

```bash
AGENT_K_TEST_DATABASE_URL='postgresql://agent_k:password@127.0.0.1:5432/agent_k_test' \
  .venv/bin/python -m pytest tests/test_runtime_store_integration.py -v
```

Rootless daemon, immutable candidate image, and optional NVIDIA runtime readiness:

```bash
AGENT_K_TEST_CANDIDATE_IMAGE="$AGENT_K_SANDBOX_IMAGE" \
AGENT_K_TEST_GPU_DEVICES='0' \
  .venv/bin/python -m pytest \
  tests/test_runtime_live_gates.py::test_configured_candidate_image_is_ready -v
```

Read-only legacy source inspection, with no upload or import:

```bash
AGENT_K_TEST_LEGACY_ROOT=/absolute/path/to/.agent_k \
  .venv/bin/python -m pytest \
  tests/test_runtime_live_gates.py::test_configured_legacy_source_is_structurally_valid -v
```

`GET /health` proves that the FastAPI lifespan started. `GET /ready` additionally requires artifact
storage configuration. Neither proves that a real Kaggle mission can complete.

## Remaining release acceptance

The repository and local gates are not deployment or external-service acceptance. Before release,
record separate evidence for:

- Render API and frontend deployment plus dedicated PostgreSQL migration/readiness;
- an AWS S3 signed PUT, server-side checksum finalization, signed GET, and resume on the configured
  bucket and IAM role;
- a native Linux rootless Docker candidate run with network disabled and enforced CPU, memory,
  pids, disk, and wall-time limits;
- NVIDIA Container Toolkit operation for each configured GPU/MIG device and a real Kumo CUDA run;
- representative image-regression predictive quality beyond the bounded natural-MOS execution
  gate recorded in [Acceptance evidence](acceptance.md);
- real model-provider and Kaggle authentication, competition discovery, download, and a full
  no-submission mission;
- an explicitly authorized Kaggle submission gate, separate from the no-submission mission;
- owner-authorized legacy upload/import/resume; inspection alone is not import acceptance.

No Render service, AWS resource, Linux worker, GPU runtime, live provider/Kaggle flow, or real
legacy import was provisioned or accepted by the local implementation run.
