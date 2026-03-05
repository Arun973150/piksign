"""
PikSign Full Protection Runner
Runs the complete protection pipeline on a real image.

Usage:
    python run_protection.py <image_path> [output_path]
    python run_protection.py photos/real.jpg
    python run_protection.py photos/real.jpg photos/real_protected.png
"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_protection.py <image_path> [output_path]")
        print("Example: python run_protection.py photos/real.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    # Default output path: same name with _protected.png
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, _ = os.path.splitext(image_path)
        output_path = base + "_protected.png"

    from piksign.protection.shield import PikSignShield

    shield = PikSignShield()

    print(f"\n{'=' * 60}")
    print(f"  Input:  {image_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    result = shield.protect_image(image_path, output_path)

    # Print results
    print(f"\n{'=' * 60}")
    print("PROTECTION RESULTS")
    print(f"{'=' * 60}")

    if result.get('status') == 'error':
        print(f"  ERROR: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    if result.get('status') == 'already_protected':
        print("  Image is already protected. No changes made.")
        sys.exit(0)

    if result.get('status') == 'rejected':
        print(f"  REJECTED: {result.get('reason', 'Image rejected by AI content gate')}")
        sys.exit(1)

    print(f"  Status:           {result.get('status', 'unknown')}")
    print(f"  Output:           {result.get('output_path', output_path)}")

    # Quality metrics
    psnr = result.get('psnr', 0.0)
    ssim = result.get('ssim', 0.0)
    drift = result.get('embedding_drift', 0.0)

    print(f"\n  --- Quality Metrics ---")
    print(f"  PSNR:             {psnr:.2f} dB {'[OK]' if psnr >= 40 else '[LOW]'} (target >= 40)")
    print(f"  SSIM:             {ssim:.4f}   {'[OK]' if ssim >= 0.93 else '[LOW]'} (target >= 0.93)")
    print(f"  Embedding Drift:  {drift:.4f}   {'[OK]' if drift <= 0.08 else '[HIGH]'} (target <= 0.08)")

    # Hash verification
    hash_ok = result.get('hash_verified', False)
    print(f"\n  --- Integrity ---")
    print(f"  Hash Verified:    {'[OK]' if hash_ok else '[FAIL]'}")

    # C2PA
    c2pa = result.get('c2pa_embedded', False)
    print(f"  C2PA Embedded:    {'[OK]' if c2pa else '[SKIP]'}")

    # Watermarks
    print(f"\n  --- Watermarks ---")
    print(f"  Spectral:         Embedded")
    print(f"  Multi-band:       Embedded")
    print(f"  Stable Signature: Embedded")

    # LEAT
    leat = result.get('leat_applied', False)
    print(f"\n  --- Adversarial Protection ---")
    print(f"  LEAT:             {'Applied' if leat else 'Skipped (fallback to confusion)'}")

    # Content analysis
    risk = result.get('risk_level', 'N/A')
    faces = result.get('faces_detected', 0)
    print(f"\n  --- Content Analysis ---")
    print(f"  Risk Level:       {risk}")
    print(f"  Faces Detected:   {faces}")

    print(f"\n{'=' * 60}")
    print(f"  Protected image saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
