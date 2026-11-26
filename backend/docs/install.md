# Installation and configuration

AGENT-K requires Python 3.13. Use the repository virtual environment for every Python command.
Install only the dependencies required by each process role.

## Backend roles

Install the API and migration-operator dependencies:

```bash
cd backend
uv sync --frozen
.venv/bin/python -m uvicorn agent_k.api:app --host 0.0.0.0 --port 9000
```

Install the portable worker and candidate ML dependencies. Add `--extra kumo` only on workers
that will run Kumo candidates:

```bash
cd backend
uv sync --frozen --extra worker --extra kumo
.venv/bin/python -m agent_k.worker
```

`python -m agent_k.worker run` is equivalent. The worker Docker image uses the explicit `run`
form; the installed quick-start command intentionally accepts either form.

The worker must use a rootless Docker daemon and a candidate image pinned by immutable registry
digest or local image ID. Build, pin, mount, and start it as described in
[Operations](operations.md#build-and-pin-images). A candidate container is not a replacement for
the worker process: the worker handles leases and trusted evaluation, while candidate code runs
in the isolated container.

Install the frontend separately:

```bash
cd frontend
pnpm install --frozen-lockfile
```

## API configuration

The API reads settings only during FastAPI lifespan startup. Secrets are never read while
importing `agent_k.api`.

| Variable | Purpose | Required |
| --- | --- | --- |
| `AGENT_K_DATABASE_URL` | Dedicated PostgreSQL runtime queue/state database | API |
| `AGENT_K_GATEWAY_SECRET` | Current HMAC key for frontend identities, at least 32 characters | API and frontend |
| `AGENT_K_GATEWAY_PREVIOUS_SECRET` | Previous HMAC key during rotation | Optional API |
| `AGENT_K_GATEWAY_ISSUER` | Identity issuer; default `agent-k-frontend` | Optional API and frontend |
| `AGENT_K_GATEWAY_AUDIENCE` | Identity audience; default `agent-k-api` | Optional API and frontend |
| `AGENT_K_WORKER_TOKEN` | Current internal worker bearer token, at least 32 characters | API and worker |
| `AGENT_K_WORKER_PREVIOUS_TOKEN` | Previous worker token during rotation | Optional API |
| `AGENT_K_S3_BUCKET` | Private AWS S3 artifact bucket | Artifact and migration routes |
| `AGENT_K_S3_REGION` | AWS region; default `us-east-1` | API |
| `AWS_ACCESS_KEY_ID` | API storage credential | API unless its runtime supplies a role |
| `AWS_SECRET_ACCESS_KEY` | API storage credential | API unless its runtime supplies a role |
| `AWS_SESSION_TOKEN` | Session token for temporary credentials | When issued |

Use separate values for `AGENT_K_GATEWAY_SECRET`, `AGENT_K_WORKER_TOKEN`, Auth.js `AUTH_SECRET`,
and cloud/provider credentials. The frontend requires its own application database and Auth.js
configuration. `PYTHON_BACKEND_URL` points server-side frontend routes to the backend; the gateway
secret is also server-only and must never use a `NEXT_PUBLIC_` prefix.

The API owns database and S3 access. Do not copy either credential set into a worker. See
[AWS S3](operations.md#aws-s3) for the required upload-finalization behavior and IAM boundary.

## Worker configuration

Create a mode-`0600` environment file for each worker. These variables configure the queue client
and rootless sandbox:

| Variable | Purpose | Required |
| --- | --- | --- |
| `AGENT_K_API_URL` | HTTPS base URL for the API | Worker |
| `AGENT_K_WORKER_ID` | Stable identifier sent with authenticated worker requests | Worker |
| `AGENT_K_WORKER_TOKEN` | Rotatable worker bearer token | Worker |
| `AGENT_K_WORKER_CAPABILITIES` | Comma-separated capabilities, such as `cpu,cuda` | Optional; default `cpu` |
| `AGENT_K_WORK_ROOT` | Dedicated absolute work directory, never home or `/` | Worker |
| `AGENT_K_SANDBOX_IMAGE` | Candidate registry digest or local `sha256:` image ID | Worker |
| `AGENT_K_LEASE_SECONDS` | Queue lease duration, from 30 to 3600 seconds | Optional; default `60` |
| `AGENT_K_HEARTBEAT_SECONDS` | Heartbeat interval, less than half the lease | Optional; default `15` |
| `AGENT_K_POLL_SECONDS` | Empty-queue polling interval | Optional; default `5` |
| `AGENT_K_GPU_DEVICES` | JSON array of NVIDIA index, UUID, or MIG IDs | GPU worker |
| `DOCKER_HOST` | Rootless Docker socket URL | Worker |
| `DOCKER_CONTEXT` | Rootless context when the host requires one | Optional worker |
| `XDG_RUNTIME_DIR` | Rootless daemon runtime directory | Worker |

Mission subprocesses also use:

| Variable | Purpose |
| --- | --- |
| `AGENT_K_MODEL` | Pydantic-AI model specification |
| `AGENT_K_KUMO_CACHE` | Read-only Kumo manifest and Hugging Face cache |
| `AGENT_K_TRANSFORMER_CACHE` | Complete read-only transformer model/tokenizer directory |
| `AGENT_K_VISION_WEIGHTS` | Read-only ResNet18 state-dictionary file |
| `AGENT_K_EVALUATION_PARALLELISM` | Concurrent isolated evaluations, from 1 to 16 |
| `AGENT_K_EVALUATION_TIMEOUT_SECONDS` | Candidate evaluation wall-time policy |
| `AGENT_K_PHASE_TIMEOUT_SECONDS` | Mission phase timeout |
| `AGENT_K_EVOLUTION_ITERATIONS` | OpenEvolve iteration limit |
| `AGENT_K_ALLOW_SUBMISSIONS` | Release gate; defaults to `false` |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle API credentials |
| Provider-specific API variables | Credentials for the model selected by `AGENT_K_MODEL` |

Only configure cache variables for assets that exist on the worker. Worker startup rejects a
configured missing path. Kumo uses CUDA by default; configure explicit devices and the NVIDIA
Container Toolkit before advertising `cuda`. Candidate containers never receive Kaggle, provider,
worker, database, S3, or gateway credentials.

## Build the images

From the repository root:

```bash
docker build --file backend/Dockerfile.candidate --tag agent-k-candidate:local backend
docker build --file backend/Dockerfile.worker --tag agent-k-worker:local backend
export AGENT_K_SANDBOX_IMAGE="$(docker image inspect \
  --format '{{.Id}}' agent-k-candidate:local)"
```

The local image ID is suitable for a local gate. Push the exact release image and use its registry
digest on production workers. Never configure a mutable tag for candidate execution.

## Local services

`docker-compose.dev.yml` starts PostgreSQL and the API. It deliberately does not provide fake S3;
set the AWS S3, gateway, worker-auth, and Kaggle variables before starting it:

```bash
docker compose --file docker-compose.dev.yml up --build runtime-db api
```

The optional `worker` profile requires a running rootless host daemon, a pinned candidate image,
the daemon socket bind, and absolute host work/asset paths. Configure those first, then run:

```bash
docker compose --file docker-compose.dev.yml --profile worker up --build worker
```

The `migration` profile runs an explicitly identified migration as an operator process. It is not
a background API task and must not be started without an owner-authorized migration ID. See
[Legacy state migration](migration.md).

After PostgreSQL, AWS S3, frontend authentication, and a worker are configured, `./run.sh` starts
the API on port 9000 and frontend on port 3000. Starting local processes or passing `/health` and
`/ready` is not live Kaggle, storage, GPU, migration, or deployment acceptance. Run the applicable
[live gates](operations.md#live-gates) and record each remaining release gate separately.
