# Quick start

This workflow assumes PostgreSQL, S3-compatible storage, Auth.js, a pinned sandbox image, Kaggle
credentials, and a model provider are configured. It does not substitute offline or generated data
for a live competition.

1. Install backend and frontend dependencies as described in [Installation](install.md).
2. Start the API and frontend with `./run.sh`.
3. Start a separately configured worker:

   ```bash
   cd backend
   .venv/bin/python -m agent_k.worker
   ```

4. Sign in through the frontend. Search for or paste a Kaggle competition, choose evolution models
   and constraints, and start the mission.
5. Follow phase, iteration, checkpoint, and terminal events in the dashboard.

The frontend server—not browser JavaScript—adds the signed user identity to every backend request.
Calling protected backend routes with an unsigned `curl` request should return `401`.

## Safety defaults

- `AGENT_K_ALLOW_SUBMISSIONS` defaults to false.
- Mission criteria `allowSubmission` also defaults to false.
- Kumo is enabled as a tabular challenger but does not replace independent acceptance.
- Model/provider selection happens when a worker runs the mission, not during imports.
- Missing Kaggle credentials make discovery routes unavailable rather than returning examples.

For pause, resume, cancellation, replay, and failure behavior, see [Operations](operations.md).
