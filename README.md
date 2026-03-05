# PikSign - AI Image Security Platform (v3.0)

PikSign detects AI-generated/manipulated images and protects authentic images from deepfake exploitation.

## Architecture

```
[Browser] <---> [Streamlit Cloud (CPU)]  <--HTTP/ngrok-->  [Google Colab (GPU)]
                 app.py                                     colab_server.py
                 Detection (full)                           LEAT (7 encoders)
                 Protection (watermarks, C2PA)              ContentAnalyzer
                 Forensics (ELA, PRNU, etc.)                SemanticDrift
```

**Detection** runs entirely on CPU (Streamlit Cloud). **Protection** offloads GPU-heavy LEAT perturbation to a Colab server; watermarking/C2PA stays on CPU.

---

## Quick Start (Local)

```bash
git clone https://github.com/Arun973150/piksign.git
cd piksign
pip install -r requirements.txt
```

Create `.env`:
```env
REALITY_DEFENDER_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

---

## Cloud Deployment

### 1. Streamlit Cloud (Frontend)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo: `Arun973150/piksign`
3. Set **Main file path**: `app.py`
4. Add secret in Settings > Secrets:
   ```toml
   [general]
   REALITY_DEFENDER_API_KEY = "your_key_here"
   ```
5. Deploy

### 2. Google Colab (GPU Backend)

1. Open `piksign_colab.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells in order:
   - Cell 1: Clones repo + installs deps
   - Cell 2: Downloads encoder weights (~4.6 GB, takes 5-10 min)
   - Cell 3: Enter your [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Cell 4: Starts server + prints your ngrok URL
   - Cell 5: Keep-alive monitor
4. Copy the ngrok URL (e.g. `https://xxxx.ngrok-free.app`)

### 3. Connect

Paste the ngrok URL into the **GPU Server (Colab)** field in the PikSign sidebar. You'll see "GPU: CONNECTED" when linked.

---

## Detection Pipeline (v3.0)

| Step | What | Method | Runs On |
|------|------|--------|---------|
| 0 | PikSign check | PNG metadata / EXIF marker | CPU |
| 1 | AI Manipulation | ELA, PRNU, Geometric, DIRE (optional) | CPU |
| 2 | Deepfake | Reality Defender API + local face analysis | CPU + Cloud API |
| 3 | Forensics | Frequency, Embedding, Color, Patch-level (GLCM, LBP, Wavelet, Edge, Benford) | CPU |
| 4 | Verdict | Weighted blend: 65% Reality Defender + 35% AI Manipulation | CPU |

Run detection standalone:
```bash
python run_detection.py path/to/image.jpg
```

## Protection Pipeline (9-step)

| Step | What | Runs On |
|------|------|---------|
| 0a | Already protected check | CPU |
| 0b | AI content gate (blocks AI images) | CPU |
| 1 | Load image | CPU |
| 2 | Perceptual hashes (pHash, dHash, etc.) | CPU |
| 3 | Content analysis (ResNet50 features) | **GPU (Colab)** or CPU fallback |
| 4 | Adaptive epsilon + frequency weights | CPU |
| 5 | LEAT adversarial perturbation (7 encoders, 50 PGD iterations) | **GPU (Colab)** or confusion-only fallback |
| 6 | Watermarks (Spectral + Multi-band + Stable Signature) | CPU |
| 7 | Quality metrics (PSNR, SSIM, Embedding drift) | **GPU (Colab)** or CPU |
| 8 | Hash verification | CPU |
| 9 | C2PA metadata binding | CPU |

Run protection standalone:
```bash
python run_protection.py path/to/image.jpg
```

## LEAT Encoders (Protection)

The Latent Ensemble Attack disrupts 7 deepfake encoder latent spaces simultaneously:

| Encoder | Target | Weight File | Size |
|---------|--------|------------|------|
| ArcFace IResNet-50 | SimSwap face identity | `arcface_r50.pth` | 167 MB |
| e4e GradualStyle | StyleCLIP W+ latent | `e4e_ffhq_encode.pt` | 1.2 GB |
| DiffAE Semantic | Diffusion autoencoder | `diffae_ffhq256.pt` | 2.6 GB |
| ICface Neutral | Face reenactment | `icface_neutral.pth` | 240 MB |
| VGG-19 Perceptual | Deepfake training loss | PyTorch built-in | - |
| SD VAE | Stable Diffusion latent | `sd_vae_ft_mse.bin` | 320 MB |
| SDXL VAE | SDXL latent | (same arch) | - |

Download weights:
```bash
python -m piksign.protection.download_weights
```

---

## Project Structure

```
piksign/
  app.py                        # Streamlit frontend (3 tabs)
  colab_server.py                # FastAPI GPU server for Colab
  piksign_colab.ipynb            # Colab setup notebook
  run_detection.py               # Standalone detection runner
  run_protection.py              # Standalone protection runner
  requirements.txt               # Streamlit Cloud dependencies
  requirements-colab.txt         # Colab GPU dependencies
  .streamlit/config.toml         # Streamlit theme config
  piksign/
    __init__.py
    cli.py                       # CLI entry point
    gpu_client.py                # HTTP client for Colab GPU server
    detection/
      __init__.py                # PikSignDetector (main orchestrator)
      deepfake_detector.py       # Reality Defender + local face analysis
      reality_defender.py        # Reality Defender API wrapper
      forensics.py               # Frequency, Embedding, Color analysis
      manipulation_forensics.py  # Patch-level: GLCM, LBP, Wavelet, Edge, Benford
      exif_validator.py          # EXIF metadata validation
      watermark_detector.py      # PikSign watermark detection
      real_image_validator.py    # Authentic image heuristics
    protection/
      __init__.py
      shield.py                  # PikSignShield (main orchestrator)
      leat_attack.py             # LEAT adversarial perturbation
      deepfake_encoders.py       # 7 encoder architectures
      content_analyzer.py        # ResNet50 content analysis
      semantic_drift.py          # Embedding drift controller
      adaptive_engine.py         # Adaptive epsilon computation
      confusion_transform.py     # CPU-based confusion transforms
      spectral_watermark.py      # Spectral domain watermark
      multiband_watermark.py     # Multi-band frequency watermark
      stable_signature.py        # Stable Signature watermark
      perceptual_hash.py         # pHash, dHash, aHash, wHash
      c2pa.py                    # C2PA metadata binding
      config.py                  # Protection configuration
      download_weights.py        # Weight downloader
    ai_image_forensics/
      __init__.py
      pipeline.py                # ELA + PRNU + Geometric + DIRE pipeline
      ela.py                     # Error Level Analysis
      prnu.py                    # Photo Response Non-Uniformity
      geometric.py               # Geometric consistency
      dire_detector.py           # DIRE classification
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REALITY_DEFENDER_API_KEY` | Yes | API key from [Reality Defender](https://realitydefender.com) |

---

## CLI Usage

```bash
# Full detection
python -m piksign.cli detect image.jpg

# Protect an image
python -m piksign.cli protect image.jpg

# Verify C2PA provenance
python -m piksign.cli verify protected.png
```

---

## GPU Server API (Colab)

When running `colab_server.py`, these endpoints are available:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | GPU info, loaded encoders |
| POST | `/api/v1/leat/generate` | LEAT perturbation (base64 image in, perturbation + metrics out) |
| POST | `/api/v1/content/analyze` | Content analysis (faces, vulnerability, risk) |
| POST | `/api/v1/drift/compute` | Embedding drift between original and protected |

---

## Fallback Behavior

When the Colab GPU server is not connected, PikSign degrades gracefully:

| Feature | With GPU | Without GPU |
|---------|----------|-------------|
| LEAT perturbation | 7-encoder attack | Confusion transforms only |
| Content analysis | ResNet50 features | OpenCV face detection + defaults |
| Embedding drift | ResNet50 cosine distance | Reports 0.0 |
| Detection | Full pipeline | Full pipeline (all CPU) |
