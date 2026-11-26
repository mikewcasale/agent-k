# Architecture

## Control plane

`agent_k.api` exposes owner-authenticated browser routes and token-authenticated worker routes. Its
lifespan constructs the PostgreSQL store, S3 signer, HTTP client, and optional Kaggle discovery
gateway. Importing the module performs no network, database, filesystem, or model-provider access.

`agent_k.stores` owns durable state and fencing. A claim creates a lease identifier. Heartbeats,
event appends, snapshots, artifact operations, completion, and failure all require that live lease.
Expired attempts cannot mutate a replacement attempt.

## Mission execution

`agent_k.worker` claims one job and launches a fresh mission subprocess. `agent_k.mission` runs the
Pydantic Graph defined in `agent_k.lycurgus`:

1. Discovery authenticates to Kaggle and resolves a competition.
2. Research downloads the actual competition files, discovers their structure, and creates trusted
   validation and acceptance splits.
3. Prototype evaluates a generic baseline, preferring LightGBM for tabular problems. Kumo may run
   as a challenger when enabled.
4. Evolution executes native OpenEvolve search, publishes per-iteration checkpoints, and never
   accesses hidden acceptance labels from candidate code.
5. Submission independently evaluates the selected program, prepares an immutable artifact, and
   submits only when service and mission authorization both allow it.

The worker reports the `mission-complete` event before committing a terminal job. The store rejects
a completion whose terminal event, sequence, mission identifier, or artifacts do not match.

## Kumo Tabular integration

`agent_k.tabular` integrates NVIDIA Kumo Tabular through
[`structured-data-models` 0.3.0](https://github.com/NVIDIA/structured-data-models/releases/tag/v0.3.0),
using the `nvidia/Kumo-Tabular` v1.0.1 checkpoints. This is a tabular prediction model, separate from
the language models that perform research and generate candidate programs. See the
[model documentation](https://docs.nvidia.com/sdgm/structured-data-models/models/kumo-tabular)
and [published checkpoint license](https://huggingface.co/nvidia/Kumo-Tabular/blob/main/LICENSE).

The implementation accepts single-target classification or regression with at most 500 features.
It verifies the offline checkpoint manifest and every file hash before inference. Production
missions require explicitly assigned CUDA devices; CPU execution is available for the small-model
validation gate. Missing assets or an unsuccessful challenger preserve the measured LightGBM
baseline.

Weights remain fixed. Evolution may vary context selection, context size, estimator count,
preprocessing code, and the blend with LightGBM. Validation selects a candidate; a separate hidden
acceptance split compares it with the reference baseline and same-engine seed before submission.
Local small-model CPU inference and blending have been exercised with real breast-cancer data;
that evidence does not establish CUDA throughput or competition-specific gains.

## Resume and controls

Pause and cancellation are cooperative. The mission checks control at durable boundaries, saves a
snapshot, uploads its runtime assets, and acknowledges the control. A paused job retains its event
sequence and resumes from the latest snapshot. Cancellation is terminal. A stale worker cannot
continue after the lease is fenced.

Snapshot JSON contains internal snake_case mission state. It is a private worker-replay contract,
not a browser DTO. Browser state is reconstructed from stable camelCase events.

## Trust boundaries

- The frontend signs only the authenticated user identifier; it cannot claim arbitrary owners.
- The API owns database and storage credentials.
- Workers receive provider and Kaggle credentials, but candidate sandboxes receive only an explicit
  allowlist and staged inputs.
- Acceptance labels, gateway secrets, worker tokens, and cloud credentials never enter candidate
  environments or event payloads.
- Chat text is parsed at the API boundary but is not copied into durable mission state; only the
  validated criteria and explicit competition identifiers are enqueued.
