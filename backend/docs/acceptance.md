# Live acceptance gates

This page records the scope of live checks performed on 2026-11-25. A passing component check
does not establish a complete mission, deployment, or production acceptance.

## Current gate status

| Gate | Observed result | Remaining boundary |
| --- | --- | --- |
| Kaggle | Authentication, search, details, leaderboard, and authorized data download passed | No submission and no complete worker mission |
| Model provider | Production discovery and research passed through an explicitly free OpenRouter model; a typed local Ollama request also passed | Complete sandboxed evolution still requires the worker |
| Natural-image regression | Real human-rated images acquired; execution and integrity passed; BatchNorm instability fixed | Fresh default RMSE 0.6506 did not beat the 0.6383 mean baseline |
| AWS S3 | AWS credentials authenticate; explicit SigV4 signing is verified locally | No designated AGENT-K bucket; signed upload/download and resume are not executed |
| Linux/CUDA | No authorized NVIDIA Linux worker is configured | Container execution, CUDA Kumo, and runtime enforcement require that host |
| Deployed API | Existing public health endpoint returns 200 | It returns 404 for `/ready` and has no new worker-claim route |

## Local regression verification

The final suite ran against a fresh UTF-8 PostgreSQL 17 database over a private Unix socket,
with the acquired natural-image fixture enabled: **61 passed, 10 skipped**. The API lifespan,
atomic lease claims, pause/resume checkpoints, stale-lease rejection, and terminal-failure
transactions passed against the actual database. The task-owned database server was stopped
afterward. An initial disposable cluster used SQL_ASCII and returned byte strings; that test
setup was corrected to UTF-8 before the successful run.

The ten skips explicitly cover four Linux/container/CUDA checks, one legacy-source gate, the
unconfigured S3 gate, two unavailable GLUE corpora, an unavailable transformer asset, and local
Kumo assets. They are not counted as passes. Ruff formatting/lint and strict mypy passed for
application code and the acquisition tool; strict MkDocs also passed. Third-party deprecation
warnings remain in OpenTelemetry, Pydantic Evals, and scikit-learn.

## Kaggle and provider checks

The production Kaggle adapter authenticated successfully, parsed seven active competitions,
retrieved exact competition details and five observed leaderboard entries, and downloaded three
already-authorized files totaling 93,081 bytes. Dataset discovery identified 891 training rows,
418 prediction rows, ten features, and one classification target. The live test used an existing
public competition as a test input; no competition-specific production branch was added.

The live responses exposed two compatibility issues that are now corrected: tag objects contain
mixed value types, and the reported metric name `Categorization Accuracy` denotes accuracy.
The download adapter now enforces its byte limit across all downloaded files. Dataset discovery
also uses generic text evidence so a long, unique name field alone does not misclassify a tabular
problem as text classification.
No competition was joined, no terms were accepted, and no submission was made.

OpenAI returned HTTP 429 and the paid OpenRouter request returned HTTP 402. Anthropic model
inventory authenticated, but even a separate minimal text request returned a billing-related
HTTP 400. These outcomes do not count as successful inference, and no account was funded.

An inventory-confirmed zero-priced OpenRouter model,
`baidu/ernie-5.5-flash-preview:free`, did execute the production discovery agent. It made two
model requests and one real discovery-tool call, consumed 1,520 input and 502 output tokens,
and selected from seven actually observed competitions. This establishes that provider/tool
path for the observed run, not a guarantee of future free capacity.

The same Baidu model then completed the production research agent in a bounded, instrumented run:
two model requests, one leaderboard-tool call, one arXiv search, 6,308 input tokens, and 1,295 output
tokens. It returned a validated `ResearchReport` containing five findings, five recommendations,
and five URLs actually observed from arXiv. External data remained disallowed. The run completed
within a six-request, six-tool-call, 5,000-output-token, 40,000-total-token, 115-second ceiling.
An earlier attempt with an inadequate local usage cap did not complete; the successful original-
model run supersedes it. No failed or capped attempt is counted as successful research.

A task-owned localhost Ollama 0.7.1 server also loaded the existing Llama 3.2 3B Instruct Q4_K
weights without downloading a model. The production `get_model("devstral:llama3.2")` adapter and
a typed Pydantic Agent completed one request using 185 input and 24 output tokens. This exercised
the configurable local OpenAI-compatible provider, not Devstral weights or CUDA. Cloud keys were
unset in that process. The model was unloaded and the task-owned server was stopped.

## S3 and durable artifact resume

Read-only AWS authentication succeeded. Neither local configuration nor the AGENT-K Render
service/environment group specifies the new artifact bucket. The configured AWS region also has
no existing EC2 instance tagged for AGENT-K. No bucket, worker, role, or other infrastructure was
created while checking these facts.

The installed Boto3 client reproduced a presigning issue: even with its default client metadata
showing `s3v4`, presigned URLs in the default region used legacy signature fields. The production
store now explicitly supplies `Config(signature_version="s3v4")`. The resulting URL uses
`AWS4-HMAC-SHA256` and signs `content-type`, `host`, and `x-amz-checksum-sha256`. AWS documents the
[legacy presigning default](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/generate_presigned_url.html)
and the [SigV4 requirement for SHA-256 checksums](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html).

`tests/test_storage.py` provides the real service gate, gated by an explicitly designated
`AGENT_K_TEST_S3_BUCKET` and a disposable UTF-8 `AGENT_K_TEST_DATABASE_URL`.
`AGENT_K_TEST_S3_REGION` defaults to `us-east-1` and must match the designated bucket. The gate:

- reserves an artifact using the production PostgreSQL/S3 stores;
- requires S3 to reject an incorrect SHA-256 body before uploading the actual repository license;
- finalizes the correct upload using S3's server-side size/checksum response;
- saves a snapshot, pauses, and reclaims the job under a new lease;
- rejects the stale lease and verifies the signed download under the new lease;
- deletes only its own unique object/version beneath `agent-k-acceptance/`.

After those two real resources have been designated, run:

```bash
backend/.venv/bin/python -m pytest backend/tests/test_storage.py -v --tb=short
```

This test has not executed against S3 in this acceptance run. AWS authentication and local URL
generation alone are insufficient to mark it passed. The checked local AWS identity is also
separate from validation of a deployed API role and its exact bucket permissions.

## Linux, CUDA, and complete missions

The current machine is macOS on Apple Silicon. Its local Docker Desktop and Podman Linux VM
runtimes are stopped and do not provide an NVIDIA GPU. No relevant remote Docker context, SSH
host alias, or AGENT-K EC2 worker was found. A user-specified existing NVIDIA Linux host is
therefore required to execute the remaining tests; no unrelated host was probed.

The live tests in `test_runtime_live_gates.py` now cover actual isolated LightGBM prediction and
trusted host scoring, inspection of configured CPU/memory/pid/network/read-only controls, removal
of a running candidate by its lease labels, and CUDA-required Kumo inference. Their fixture uses
real breast-cancer measurements, with hidden labels outside candidate mounts. The CPU and CUDA
execution tests remain unrun here.

On the authorized worker, build the current candidate image and configure
`AGENT_K_TEST_CANDIDATE_IMAGE` with its immutable digest, `AGENT_K_TEST_GPU_DEVICES` with explicit
device IDs, and `AGENT_K_KUMO_ASSETS` with provisioned fixed-weight assets. Then run:

```bash
backend/.venv/bin/python -m pytest backend/tests/test_runtime_live_gates.py -v --tb=short
```

The existing Render API still exposes the old runtime: `/health` returned 200, `/ready` returned
404, and its OpenAPI document omitted `/api/worker/v1/jobs/claim`. A full mission on the new runtime
must therefore be demonstrated with the new API, a real artifact bucket, a qualifying worker, and
an available provider. No such end-to-end success, production import, deployment, or Kaggle
submission is claimed by the component checks above.

## Natural-image MOS regression

Status: **execution and integrity passed locally on 2026-11-25** using CPU and Python 3.13.
Predictive-quality acceptance remains unresolved; see the fresh post-fix evaluation below.

The fixture comes from the publisher's
[KonIQ-10k IQA Database](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2435&version=1.0)
(DOI `10.18419/DARUS-2435`, version 1.0). KonIQ-10k contains authentic, in-the-wild
YFCC100M photographs rather than generated images or artificially distorted project data. The
[publisher's database page](https://database.mmsp-kn.de/koniq-10k-database.html) documents 10,073
quality-scored images and the human mean-opinion-score (MOS) collection. DaRUS licenses the
dataset under CC BY 4.0 while noting that each image retains its source Creative Commons license.

The bounded fixture uses 120 deterministic MOS-quantile rows from the complete publisher score
table: 96 training images and 24 held-out images. The publisher MOS uses a five-point ACR scale
from 1 (bad) through 5 (excellent); this selection spans 1.09615384615385 through 4.31 and occupies
9.7 MiB. It range-extracts only those images from the 766,965,996-byte publisher archive; it does
not download the full archive. The generated `provenance.json` records SHA-256 digests for every
CSV, source metadata file, and image. Pinned upstream evidence is:

- archive `koniq10k_512x384.zip`: MD5
  `dc213332574e8431a86c7ff34e1fa924`, as published by DaRUS;
- publisher MOS table: SHA-256
  `a5173fb66831b2049a581d2396ef6e50d341f2af509e1f9b252d4bd49fa03b5f`;
- DaRUS version metadata: SHA-256
  `0c9bc4c767de8df0e7cad18b27a10d3c1c9a6c0413da46d578783dc35f051f88`;
- deterministic 120-row fixture provenance: SHA-256
  `a72a7575f842f2e6e8e135b6f095619885862e9118e4887466a00949d2dad079`.

Acquire a clean fixture from the official public distribution:

```bash
backend/.venv/bin/python backend/tools/koniqfixture.py \
  --output /private/tmp/agent-k-koniq-mos-fixture \
  --rows 120
```

Run the real ResNet18 regression gate:

```bash
AGENT_K_IMAGE_REGRESSION_FIXTURE=/private/tmp/agent-k-koniq-mos-fixture \
PYTHONPATH=backend \
backend/.venv/bin/python -m pytest \
  backend/tests/test_modalities.py::test_resnet18_executes_real_mos_regression_when_fixture_is_available \
  -q --tb=short
```

Before BatchNorm calibration was added, the verified smoke run trained the real ResNet18
implementation for one epoch at 64-pixel input size, generated 24 predictions, and passed host-side
ID, column, finiteness, digest, and RMSE validation. Its RMSE was 5.70138307151222. That score is
historical execution evidence for the intentionally small CPU smoke gate, not a model-quality
benchmark or the expected result from the corrected path. The candidate-visible directory
contained only training data, prediction rows, and images; the publisher score table and held-out
MOS labels remained outside that directory.

### Untuned default-configuration diagnostic

Before BatchNorm calibration was added, one separate diagnostic used the same immutable 96/24
split with an empty hyperparameter map, therefore exercising the scratch ResNet18 production
defaults: five epochs, 224-pixel images, batch size 32, and automatic device selection, which
resolved to CPU. Training and trusted scoring took 39.69 seconds. Its held-out RMSE was
30.78630170875045, while predicting the training-target mean of 3.1730244426560206 for every
held-out row produced RMSE 0.6451341009285437. These values are on the publisher's 1–5 MOS scale,
with the fixture spanning 1.09615384615385–4.31. This was a single untuned diagnostic; no
parameter, implementation, or model selection decision was adjusted using the held-out result.

### BatchNorm cause and correction

A diagnosis used only the 96 labeled training rows, split into 72 fit rows and 24 internal
diagnostic rows; it did not read the official held-out labels. Mean batch MSE fell from 7.5240 in
epoch one to 0.4455 in epoch five, and fit-set predictions made in training mode had RMSE 0.7426.
The same weights in evaluation mode produced fit RMSE 9.3531 and internal-diagnostic RMSE 7.4625,
including predictions as low as -37.95. The first BatchNorm layer had accumulated only 15 running-
statistic updates. Recomputing running statistics from the 72 fit images with PyTorch's public
`torch.optim.swa_utils.update_bn` changed evaluation-mode fit RMSE to 0.5244 and internal-
diagnostic RMSE to 0.5983, with predictions between 1.64 and 4.16. The internal training-mean
baseline RMSE was 0.5779.

The production vision path now runs that standard BatchNorm calibration once over training images
after the final optimizer step and before inference. It does not read targets during calibration,
scale targets, clip predictions, change hyperparameters, or select a model from held-out results.
Candidate diagnostics record the actual public BatchNorm buffer count after recalibration. The
real-data regression check trains for two epochs and requires that count to equal one calibration
pass rather than the accumulated training-epoch count.

### Fresh disjoint post-fix evaluation

A second bounded fixture shifted every deterministic MOS-quantile source row by one position. Its
120 official images have zero overlap with the first fixture, and its provenance SHA-256 is
`788a3483e1aafbee0c432293b98ddaac7a2398c04cb1dea4dd2f684c9f0dbdc4`. Acquire it with:

```bash
backend/.venv/bin/python backend/tools/koniqfixture.py \
  --output /private/tmp/agent-k-koniq-mos-fresh-offset1 \
  --rows 120 \
  --selection-offset 1
```

Exactly one post-fix evaluation used the unmodified defaults on this fresh 96/24 split. CPU
training, one BatchNorm calibration pass, and trusted scoring took 44.45 seconds. The fixture spans
MOS 1.14678899082569–4.24528301886792; calibrated predictions remained between 2.4721 and 3.4197.
The candidate RMSE was 0.6506332399769087, compared with 0.6382822989911419 from predicting the
training mean of 3.1726743262300956. Calibration fixes the evaluation-statistics explosion, but the
scratch model did not beat the training-mean baseline. Predictive-quality acceptance therefore
remains unresolved. No parameter or code decision was adjusted using this fresh held-out result.
