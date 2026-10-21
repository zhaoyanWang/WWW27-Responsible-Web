# NodeDiffRec: anonymous reference code

Core implementations for graph generation, structural denoising, activity/quality-aware branch construction, score fusion, and ranking evaluation are in `nodediffrec/`.

`data/` contains the sampled Goodreads, ALB, and ML-100K interactions. `train.npz` contains training users; `context.npz`, `validation.npz`, and `test.npz` share evaluation-user rows and item columns. Matrix dimensions are listed in `data/manifest.json`.

Parameter domains are summarized in `PARAMETER_RANGES.md`.
