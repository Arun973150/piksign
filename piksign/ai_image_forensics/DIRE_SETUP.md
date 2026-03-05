# DIRE detector setup

The AI detection pipeline can use the **DIRE** (Diffusion Reconstruction Error) classifier for better detection of diffusion-generated images. Our implementation is aligned with the [official DIRE repo](https://github.com/ZhendongWang6/DIRE) (ICCV 2023).

## Getting the pretrained model

Pre-trained weights are **not** included in this repo. Download them from:

- **[BaiduDrive](https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg)** (password: `dire`)
- **[RecDrive](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070)** (password: `dire`)

Place the checkpoint file (e.g. `model_epoch_latest.pth`) in **one** of these locations:

1. **DIRE-repo (recommended)**  
   If you cloned the official repo at the project root:
   ```
   <project_root>/DIRE-repo/data/exp/ckpt/lsun_adm/model_epoch_latest.pth
   ```
   Or any `*.pth` under `DIRE-repo/` or `DIRE-repo/data/exp/ckpt/`.

2. **This package**
   ```
   piksign/piksign/ai_image_forensics/model/model_epoch_latest.pth
   ```
   Create the `model` folder if it does not exist.

3. **ai-image folder** (either structure)
   ```
   piksign/piksign/ai-image/ai-image/model/model_epoch_latest.pth
   piksign/piksign/piksign/ai-image/ai-image/model/model_epoch_latest.pth
   ```

The pipeline will search these paths automatically. Without a checkpoint, DIRE is skipped and only ELA, PRNU, and geometric checks run.

## Full DIRE accuracy: use DIRE maps

The official DIRE checkpoints are trained on **DIRE maps** (reconstruction error from a diffusion model), not raw RGB. For full accuracy:

1. **Get the diffusion model** (for computing DIRE maps): same [BaiduDrive](https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg) / [RecDrive](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070) (password: `dire`). Download e.g. **256x256_diffusion_uncond.pt** and place it in:
   - `DIRE-repo/guided-diffusion/models/256x256_diffusion_uncond.pt`, or
   - any path you pass as `--diffusion-model`.

2. **Run with DIRE maps** so the pipeline (a) computes the DIRE map per image via `DIRE-repo/guided-diffusion/compute_dire_single.py`, then (b) runs the ResNet classifier on the map:
   ```bash
   python run_ai_detection_only.py <images_dir> --use-dire-maps -v -o results.json
   # Or with explicit diffusion model path:
   python run_ai_detection_only.py <images_dir> --use-dire-maps --diffusion-model path/to/256x256_diffusion_uncond.pt -v -o results.json
   ```

3. **Requirements for DIRE map computation**: the [DIRE repo](https://github.com/ZhendongWang6/DIRE) must be cloned (e.g. at `DIRE-repo/` next to the piksign package), and its `guided-diffusion` dependencies installed (`pip install -r requirements.txt` in the guided-diffusion folder; the repo uses `blobfile`, `mpi4py` for the original script; the single-image script `compute_dire_single.py` does **not** require MPI).

## DIRE gives 1.0 (or confidence 1) for every image

If DIRE score is **1.0 for every image**, the DIRE map or classifier input is often wrong (e.g. maps are nearly black, or diffusion model/checkpoint mismatch). The pipeline now **ignores DIRE for the verdict** when the score is ≥ 0.98, so the verdict falls back to ELA/PRNU/geometric instead of labelling everything "Ai Generated". The raw DIRE value is still stored in the JSON.

To fix the root cause:

- Check that the **diffusion model** matches the one used to train the DIRE classifier (256×256 unconditional).
- Ensure **compute_dire_single.py** runs without errors and produces non‑black DIRE map images when you inspect them.
- Use the **same Python/torch environment** as the DIRE repo (e.g. `guided-diffusion` deps installed).
