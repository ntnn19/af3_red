# Source Code Changes in AF3-ReD

AF3-ReD introduces the Gaussian biasing potential into the denoising process of the diffusion module in AlphaFold3.
To support this feature, the following source files in the original AlphaFold 3.0.1 (latest code on Mar 17, 2025; commit id: 8151373) are modified:

- `run_alphafold.py`
- `model/model.py`
- `model/network/diffusion_head.py`

Additionally, the new module below is added:

- `model/bias.py` — implements the Gaussian biasing potential and its gradient

Comments tagged with `# AF3_ReD` indicate code changes.
