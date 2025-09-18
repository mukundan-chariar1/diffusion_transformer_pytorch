# DATASET

## Images:

- Images are random sized. CHose 256x256x3 size since it is what the reference DiT paper chooses. 
- Current plan is to implement the same transforms in the DiT paper here. 


## DiT One-Epoch Training – To-Do

### ✅ Clarify & scope
- [ ] Confirm “one epoch” means exactly one full pass over the train split
- [ ] Confirm deliverable is a single Python script (CLI runnable) and not a notebook
- [ ] Confirm whether validation + sample images are expected after the epoch

### ⚙️ Environment
- [x] Pin Python/PyTorch versions
- [x] Add `requirements.txt` (torch/torchvision/tqdm)
- [ ] Set deterministic seed toggle (and note GPU/CPU assumptions)

### 📦 Data
- [ ] Point to dataset location (`--data_dir`)
- [ ] Implement/verify `ImageDataset` in `dataloader.py`
- [ ] Normalize to [-1, 1]; ensure image size divisible by patch size
- [ ] Add DataLoader with `num_workers`, `pin_memory`, `drop_last`

### 🧠 Model (from Task 2)
- [ ] Instantiate DiT with the same config as Task 2 (patch size, heads, dims)
- [ ] Ensure forward signature is `(x_t, t)`
- [ ] Verify positional/time embeddings and output shape match ε-prediction

### 🌫️ Diffusion
- [ ] Verify beta schedule (linear/cosine) and `T`
- [ ] Implement uniform time sampling `t ~ U{0..T-1}`
- [ ] Ensure `forward(x, t)` returns `(x_t, ε)` targets

### 🔁 Training loop (one epoch)
- [ ] Build epoch-based loop with `--epochs 1` default
- [ ] Loss: MSE(ε̂, ε)
- [ ] Optimizer: Adam/AdamW with sane defaults
- [ ] (Optional) AMP mixed precision flag `--amp`
- [ ] (Optional) Grad clipping `--grad_clip`

### 📉 Logging & artifacts
- [ ] Print per-iteration loss; show running avg
- [ ] Save final loss summary to stdout/CSV
- [ ] (If required) Sample a small image grid at end of epoch

### 🧪 Validation (if required)
- [ ] Add val DataLoader
- [ ] Compute/print val loss once after the epoch

### 💾 Checkpointing (optional for demo)
- [ ] Save `state_dict` (+ optimizer/scaler) at end of epoch to `--ckpt`

### 🔌 CLI & entrypoint
- [ ] Add argparse flags: data_dir, batch_size, lr, epochs, amp, seed, out_dir
- [ ] Provide `if __name__ == "__main__":` with sensible defaults

### 🔍 Tests & smoke checks
- [ ] Add a tiny smoke test (8×8 RGB toy data, 10 steps) to catch import/shape errors
- [ ] Verify a single epoch completes on CPU-only (reduced batch/size)

### 🧹 Quality
- [ ] Run formatter/linter (Black/ruff or flake8)
- [ ] Type hints for public functions
- [ ] Remove stray debug code

### 📝 Docs & submission
- [ ] README: how to run, expected runtime, outputs, sample command
- [ ] Include example logs and (if required) a sample image grid
- [ ] Tag/report that only one epoch was used as per task
