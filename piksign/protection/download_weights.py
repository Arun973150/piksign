# -*- coding: utf-8 -*-
"""
Download pre-trained encoder weights for LEAT deepfake disruption.

All 4 encoders from the LEAT paper + VGG (auto-downloaded by PyTorch):
  1. ArcFace IResNet-50  -- SimSwap face identity encoder
  2. e4e GradualStyleEncoder -- StyleCLIP/e4e W+ latent encoder
  3. DiffAE Semantic Encoder -- Diffusion Autoencoder encoder
  4. ICface Neutral Generator -- ICface reenactment encoder
  5. VGG-19 (no download needed -- uses PyTorch ImageNet weights)

Usage:
    python -m piksign.protection.download_weights
    python -m piksign.protection.download_weights --encoder arcface
    python -m piksign.protection.download_weights --encoder e4e
    python -m piksign.protection.download_weights --encoder diffae
    python -m piksign.protection.download_weights --encoder icface
"""

import os
import sys
import argparse
import urllib.request
import tarfile

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


def download_file(url, dest_path, desc=""):
    """Download a file with progress."""
    print(f"  Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [{pct:3d}%] {mb:.1f} / {total_mb:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print("\n  Done!")


# -- 1. ArcFace (SimSwap) ----------------------------------------------------

def download_arcface():
    """
    Download ArcFace IResNet-50 weights from SimSwap releases.

    This is the exact face identity encoder used by SimSwap for face
    swapping. LEAT attacks this to prevent identity extraction.

    Source: https://github.com/neuralchen/SimSwap/releases
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, 'arcface_r50.pth')

    if os.path.exists(dest):
        print(f"  ArcFace weights already exist at {dest}")
        return

    tar_path = os.path.join(WEIGHTS_DIR, 'arcface_checkpoint.tar')
    url = "https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar"

    try:
        download_file(url, tar_path, "ArcFace IResNet-50 (~167 MB)")

        print("  Extracting...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(WEIGHTS_DIR)

        # Look for .pth files in extracted directories
        for root, dirs, files in os.walk(WEIGHTS_DIR):
            for f in files:
                if f.endswith('.pth') and 'arcface' in root.lower():
                    src = os.path.join(root, f)
                    os.rename(src, dest)
                    print(f"  Moved {src} -> {dest}")
                    break

        if os.path.exists(tar_path):
            os.remove(tar_path)

        print(f"  ArcFace weights saved to {dest}")
    except Exception as e:
        print(f"  ERROR: Failed to download ArcFace weights: {e}")
        print()
        print("  Manual download instructions:")
        print(f"    1. Download from: {url}")
        print(f"    2. Extract and place the .pth file at: {dest}")
        print()
        print("  Alternative (insightface model zoo):")
        print("    1. Download ms1mv3_arcface_r50_fp16/backbone.pth from:")
        print("       https://github.com/deepinsight/insightface/tree/master/model_zoo")
        print(f"    2. Rename to: {dest}")


# -- 2. e4e (StyleCLIP) ------------------------------------------------------

def download_e4e():
    """
    Download e4e (encoder4editing) weights for StyleGAN latent encoding.

    This is the exact encoder used by StyleCLIP and e4e for mapping face
    images to StyleGAN's W+ latent space. LEAT attacks this to prevent
    correct latent encoding for any text-driven manipulation.

    Source: https://github.com/omertov/encoder4editing
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, 'e4e_ffhq_encode.pt')

    if os.path.exists(dest):
        print(f"  e4e weights already exist at {dest}")
        return

    url = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt"

    try:
        download_file(url, dest, "e4e StyleGAN encoder (~1.2 GB)")
        print(f"  e4e weights saved to {dest}")
    except Exception as e:
        print(f"  ERROR: Failed to download e4e weights: {e}")
        print()
        print("  Manual download instructions:")
        print(f"    Option A (Hugging Face, recommended):")
        print(f"      wget {url}")
        print()
        print("    Option B (Google Drive):")
        print("      pip install gdown")
        print("      gdown 1cUv_reLE6k3604or78EranS7XzuVMWeO")
        print()
        print(f"    Place the file at: {dest}")


# -- 3. DiffAE (Diffusion Autoencoders) --------------------------------------

def download_diffae():
    """
    Download DiffAE semantic encoder weights.

    This is the semantic encoder from Diffusion Autoencoders that maps
    face images to a 512-d semantic vector. LEAT attacks this to prevent
    diffusion-based face attribute manipulation.

    Source: https://github.com/phizaz/diffae
    The full DiffAE checkpoint (~2 GB) contains both encoder and decoder.
    LEAT only loads the encoder portion.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, 'diffae_ffhq256.pt')

    if os.path.exists(dest):
        print(f"  DiffAE weights already exist at {dest}")
        return

    # DiffAE checkpoints are hosted on the authors' storage
    # Try Hugging Face mirror first
    url = "https://huggingface.co/multimodalart/diffae/resolve/main/diffae_ffhq256_autoenc.ckpt"

    try:
        download_file(url, dest, "DiffAE semantic encoder (~2 GB)")
        print(f"  DiffAE weights saved to {dest}")
    except Exception as e:
        print(f"  ERROR: Failed to download DiffAE weights: {e}")
        print()
        print("  Manual download instructions:")
        print("    1. Visit: https://github.com/phizaz/diffae")
        print("    2. Download the FFHQ 256x256 autoencoder checkpoint")
        print("       (Look for 'ffhq256_autoenc' in their model links)")
        print(f"    3. Place the file at: {dest}")
        print()
        print("    Alternative (using gdown for Google Drive):")
        print("      pip install gdown")
        print("      Check the diffae repo README for the Google Drive ID")
        print(f"      gdown <DRIVE_ID> -O {dest}")


# -- 4. ICface (Face Reenactment) --------------------------------------------

def download_icface():
    """
    Download ICface neutral face generator weights.

    This is the encoder-decoder from ICface that produces neutral
    (expressionless) face images. LEAT attacks the neutral face output
    to prevent face reenactment regardless of driving Action Units.

    Source: https://github.com/Blade6570/icface
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, 'icface_neutral.pth')

    if os.path.exists(dest):
        print(f"  ICface weights already exist at {dest}")
        return

    # ICface weights are hosted on the authors' GitHub
    url = "https://github.com/Blade6570/icface/releases/download/v1.0/icface_generator.pth"

    try:
        download_file(url, dest, "ICface neutral generator (~200 MB)")
        print(f"  ICface weights saved to {dest}")
    except Exception as e:
        print(f"  ERROR: Failed to download ICface weights: {e}")
        print()
        print("  Manual download instructions:")
        print("    1. Visit: https://github.com/Blade6570/icface")
        print("    2. Download the pre-trained generator checkpoint")
        print("       (Check their README or releases page)")
        print(f"    3. Place the file at: {dest}")
        print()
        print("    Note: ICface weights may require contacting the authors.")
        print("    LEAT still works without them (random init provides")
        print("    architectural diversity for the gradient ensemble).")


# -- 5. SD VAE (Stable Diffusion VAE) ------------------------------------------

def download_sdvae():
    """
    Download Stable Diffusion VAE encoder weights (sd-vae-ft-mse).

    This is the standard KL-regularized VAE used by Stable Diffusion and
    architecturally similar to the proprietary VAEs in models like Google's
    Nano Banana / Imagen. LEAT attacks the VAE encoder bottleneck to disrupt
    latent space encoding for ALL diffusion-based image editors.

    Source: https://huggingface.co/stabilityai/sd-vae-ft-mse
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, 'sd_vae_ft_mse.bin')

    if os.path.exists(dest):
        print(f"  SD VAE weights already exist at {dest}")
        return

    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"

    try:
        download_file(url, dest, "SD VAE encoder (~335 MB)")
        print(f"  SD VAE weights saved to {dest}")
    except Exception as e:
        print(f"  ERROR: Failed to download SD VAE weights: {e}")
        print()
        print("  Manual download instructions:")
        print(f"    Option A (direct download):")
        print(f"      wget {url} -O {dest}")
        print()
        print("    Option B (using huggingface_hub):")
        print("      pip install huggingface_hub")
        print("      python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download('stabilityai/sd-vae-ft-mse', 'diffusion_pytorch_model.bin', "
              f"local_dir='{WEIGHTS_DIR}')\"")
        print()
        print(f"    Place the file at: {dest}")


# -- 6. SDXL VAE (Stable Diffusion XL VAE) -------------------------------------

def download_sdxl_vae():
    """
    Download SDXL VAE encoder weights.

    Same architecture as SD 1.5 VAE but trained on higher-quality data
    with different objectives. Having two VAE variants in the ensemble
    improves black-box transfer to unknown proprietary VAEs.

    Source: https://huggingface.co/stabilityai/sdxl-vae
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, 'sdxl_vae.bin')

    if os.path.exists(dest):
        print(f"  SDXL VAE weights already exist at {dest}")
        return

    url = "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.bin"

    try:
        download_file(url, dest, "SDXL VAE encoder (~335 MB)")
        print(f"  SDXL VAE weights saved to {dest}")
    except Exception as e:
        print(f"  ERROR: Failed to download SDXL VAE weights: {e}")
        print()
        print("  Manual download instructions:")
        print(f"    wget {url} -O {dest}")
        print()
        print("  Alternative: SDXL VAE will fall back to SD 1.5 VAE weights")
        print("  if not available (still provides architectural diversity).")
        print(f"    Place the file at: {dest}")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download LEAT encoder weights for deepfake disruption"
    )
    parser.add_argument(
        '--encoder',
        choices=['arcface', 'e4e', 'diffae', 'icface', 'sdvae', 'sdxl_vae', 'all'],
        default='all',
        help='Which encoder weights to download (default: all)'
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  LEAT Encoder Weight Downloader")
    print("  Downloads the actual deepfake model encoders from the paper")
    print("=" * 65)
    print(f"  Weights directory: {WEIGHTS_DIR}")
    print()

    if args.encoder in ('arcface', 'all'):
        print("[1/6] ArcFace IResNet-50 (SimSwap -- face swapping)")
        download_arcface()
        print()

    if args.encoder in ('e4e', 'all'):
        print("[2/6] e4e GradualStyleEncoder (StyleCLIP -- face manipulation)")
        download_e4e()
        print()

    if args.encoder in ('diffae', 'all'):
        print("[3/6] DiffAE Semantic Encoder (DiffAE -- diffusion manipulation)")
        download_diffae()
        print()

    if args.encoder in ('icface', 'all'):
        print("[4/6] ICface Neutral Generator (ICface -- face reenactment)")
        download_icface()
        print()

    if args.encoder in ('sdvae', 'all'):
        print("[5/6] SD VAE Encoder (Stable Diffusion -- diffusion editors)")
        download_sdvae()
        print()

    if args.encoder in ('sdxl_vae', 'all'):
        print("[6/6] SDXL VAE Encoder (SDXL -- newer diffusion editors)")
        download_sdxl_vae()
        print()

    print("=" * 65)
    print("  VGG-19 uses PyTorch ImageNet weights (auto-downloaded)")
    print()
    print("  Weight status summary:")
    for name, filename in [
        ("ArcFace", "arcface_r50.pth"),
        ("e4e/StyleGAN", "e4e_ffhq_encode.pt"),
        ("DiffAE", "diffae_ffhq256.pt"),
        ("ICface", "icface_neutral.pth"),
        ("SD VAE", "sd_vae_ft_mse.bin"),
        ("SDXL VAE", "sdxl_vae.bin"),
    ]:
        path = os.path.join(WEIGHTS_DIR, filename)
        status = "READY" if os.path.exists(path) else "MISSING"
        icon = "+" if status == "READY" else "-"
        print(f"    [{icon}] {name}: {status}")
    print()
    print("  The shield will auto-detect available weights on next run.")
    print("  Encoders without weights use random init (still effective")
    print("  due to architectural diversity in the gradient ensemble).")
    print("=" * 65)


if __name__ == '__main__':
    main()
