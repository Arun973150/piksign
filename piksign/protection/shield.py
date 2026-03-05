# -*- coding: utf-8 -*-
"""
PikSign Shield - Main Protection Orchestrator
Coordinates all protection components to create AI-resistant images.
"""

import os
import numpy as np
import torch

from piksign.protection.config import Config
from piksign.protection.utils import (
    load_image, save_image, compute_psnr, compute_ssim,
    array_to_tensor, tensor_to_array, print_banner
)
from piksign.protection.content_analyzer import ContentAnalyzer
from piksign.protection.perceptual_hash import PerceptualHashSystem
from piksign.protection.semantic_drift import SemanticDriftController
from piksign.protection.spectral_watermark import SpectralWatermark
from piksign.protection.multiband_watermark import MultiBandFrequencyWatermark
from piksign.protection.stable_signature import StableSignatureWatermark
from piksign.protection.adaptive_engine import AdaptiveTransformEngine
from piksign.protection.confusion_transform import ConfusionTransform

# Try to import LEAT
try:
    from piksign.protection.leat_attack import LEATAttack
    HAS_LEAT = True
except ImportError:
    HAS_LEAT = False

# Try to import C2PA
try:
    from piksign.protection.c2pa import C2PAMetadataBinding
    HAS_C2PA = True
except ImportError:
    HAS_C2PA = False


class PikSignShield:
    """
    PikSign Shield - Image Protection System
    
    Makes images human-viewable but AI-unreadable through:
    - Adaptive confusion transforms
    - Multi-layer watermarking
    - C2PA metadata binding
    - Perceptual hash fingerprinting
    """
    
    def __init__(self, config: Config = None, gpu_client=None):
        self.config = config or Config()
        self.gpu_client = gpu_client

        # When gpu_client is provided, skip loading GPU-heavy models locally
        use_remote_gpu = gpu_client is not None

        if use_remote_gpu:
            self.device = torch.device("cpu")
        else:
            self.device = self.config.DEVICE

        print_banner("INITIALIZING PIKSIGN SHIELD")

        if use_remote_gpu:
            print("\n[*] GPU operations will be delegated to remote Colab server")

        print("\n[1/9] Content analyzer...")
        if use_remote_gpu:
            self.content_analyzer = None
            print("      [OK] Using remote GPU")
        else:
            self.content_analyzer = ContentAnalyzer(self.device)

        print("[2/9] Adaptive engine...")
        self.adaptive_engine = AdaptiveTransformEngine(self.config)

        print("[3/9] Spectral watermark...")
        self.spectral_watermark = SpectralWatermark(self.config.WATERMARK_STRENGTH)

        print("[4/9] Stable Signature...")
        self.stable_signature = StableSignatureWatermark(
            self.config.WATERMARK_MESSAGE,
            self.config.WATERMARK_METHOD
        )

        print("[5/9] Drift controller...")
        if use_remote_gpu:
            self.drift_controller = None
            print("      [OK] Using remote GPU")
        else:
            self.drift_controller = SemanticDriftController(
                self.device,
                self.config.SEMANTIC_DRIFT_TARGET
            )

        print("[6/9] Perceptual Hash System...")
        self.phash_system = PerceptualHashSystem(self.config.PHASH_SIZE)

        print("[7/9] Multi-Band Frequency Watermark...")
        self.multiband_watermark = MultiBandFrequencyWatermark(
            self.config.MULTIBAND_STRENGTH,
            self.config.FREQUENCY_BANDS
        )

        print("[8/9] LEAT Deepfake Disruption...")
        self.leat = None
        if use_remote_gpu:
            print("      [OK] Using remote GPU")
        elif self.config.LEAT_ENABLED and HAS_LEAT:
            try:
                self.leat = LEATAttack(
                    device=self.device,
                    iterations=self.config.LEAT_ITERATIONS,
                    step_size=self.config.LEAT_STEP_SIZE,
                    epsilon=self.config.LEAT_EPSILON,
                    encoder_names=self.config.LEAT_ENCODERS,
                )
                print("      [OK] LEAT enabled with Normalized Gradient Ensemble")
            except Exception as e:
                print(f"      [!] LEAT init failed ({e}), falling back to confusion only")
                self.leat = None
        elif not HAS_LEAT:
            print("      [!] LEAT not available (missing dependencies)")
        else:
            print("      - LEAT disabled in config")

        print("[9/9] C2PA Metadata Binding...")
        if HAS_C2PA:
            self.c2pa = C2PAMetadataBinding(
                creator_name="PikSign Shield",
                creator_url="https://piksign.protection",
                enable_no_ai=True,
                enable_no_mining=True
            )
            print("      [OK] C2PA with No AI rights enabled")
        else:
            self.c2pa = None
            print("      [!] C2PA not available")

        print("\n[OK] All systems ready!")
        print("=" * 80)
    
    def _is_piksign_protected(self, image_path: str) -> bool:
        """Check if an image already carries PikSign protection markers."""
        try:
            from PIL import Image as _Image
            img = _Image.open(image_path)
            # PNG text chunk
            if hasattr(img, 'text') and img.text.get('PikSign_Protected') == 'true':
                return True
            # JPEG EXIF ImageDescription
            import piexif
            raw_exif = img.info.get('exif')
            if raw_exif:
                exif_dict = piexif.load(raw_exif)
                desc = exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, b'')
                if b'PikSign_Protected:true' in desc:
                    return True
        except Exception:
            pass
        return False

    def _quick_ai_check(self, image_path: str) -> dict:
        """
        Quick AI-generation / manipulation check before protecting.
        Uses ai_image_forensics pipeline (ELA, PRNU, Geometric, DIRE).
        Returns dict with 'is_ai' bool and 'verdict' string.
        """
        try:
            from piksign.ai_image_forensics import run_forensics_pipeline
            from piksign.ai_image_forensics import VERDICT_AI_GENERATED, VERDICT_AI_MANIPULATED
            r = run_forensics_pipeline(
                image_path,
                run_ela=True,
                run_prnu=True,
                run_geometric=True,
                run_dire=True,
                save_artifacts=False,
            )
            is_ai = r.verdict in (VERDICT_AI_GENERATED, VERDICT_AI_MANIPULATED)
            # Also block if combined anomaly score is above threshold
            if not is_ai and r.combined_anomaly_score is not None and r.combined_anomaly_score > 0.55:
                is_ai = True
            return {'is_ai': is_ai, 'verdict': r.verdict, 'score': r.combined_anomaly_score}
        except ImportError:
            # ai_image_forensics not available, allow protection to proceed
            return {'is_ai': False, 'verdict': 'unknown', 'score': 0.0}
        except Exception as e:
            print(f"      [!] AI pre-check failed ({e}), proceeding with protection")
            return {'is_ai': False, 'verdict': 'error', 'score': 0.0}

    def protect_image(self, image_path: str,
                      output_path: str = None) -> tuple:
        """
        Protect an image from AI interpretation.

        Args:
            image_path: Path to input image
            output_path: Path for output (default: protected_<name>.png)

        Returns:
            tuple: (output_path, metrics_dict)
        """
        print_banner("PROTECTION PROCESS")

        # -- Step 0a: Already protected? ------------------------------------
        print("\n[0a/9] Checking for existing PikSign protection...")
        if self._is_piksign_protected(image_path):
            print("      [OK] Image is ALREADY PikSign-protected. No action needed.")
            return None, {'status': 'already_protected', 'message': 'Image already carries PikSign protection markers.'}

        # -- Step 0b: AI content gate ----------------------------------------
        print("\n[0b/9] Running AI content pre-check (ELA + PRNU + Geo + DIRE)...")
        ai_check = self._quick_ai_check(image_path)
        if ai_check['is_ai']:
            verdict = ai_check['verdict']
            score = ai_check.get('score', 0.0)
            print(f"      [NO] AI content detected! Verdict: {verdict}  Score: {score:.2f}")
            print("      Protection BLOCKED -- we do not protect AI-generated or AI-manipulated images.")
            return None, {
                'status': 'ai_content_detected',
                'verdict': verdict,
                'score': score,
                'message': 'Protection refused: image appears to be AI-generated or AI-manipulated.'
            }
        print(f"      [OK] Image appears authentic (score: {ai_check.get('score', 0.0):.2f}). Proceeding.")

        # Step 1: Load image
        print("\n[1/9] Loading...")
        img_array, img_tensor, img_pil = load_image(image_path)
        img_tensor = img_tensor.to(self.device)
        print(f"      [OK] Size: {img_pil.size[0]}x{img_pil.size[1]}")
        
        # Step 2: Compute perceptual hashes
        print("\n[2/9] Computing perceptual hashes...")
        original_hashes = self.phash_system.generate_all_hashes(img_array)
        fingerprint = self.phash_system.generate_composite_fingerprint(original_hashes)
        print(f"      [OK] pHash: {original_hashes['phash']['string'][:16]}...")
        print(f"      [OK] Composite: {fingerprint['composite_hash'][:16]}...")
        
        # Step 3: Analyze content
        print("\n[3/9] Analyzing content...")
        content_analysis = None
        if self.gpu_client:
            content_analysis = self.gpu_client.analyze_content(img_pil)
            if content_analysis:
                print("      [OK] (remote GPU)")
        if content_analysis is None and self.content_analyzer is not None:
            content_analysis = self.content_analyzer.analyze(img_tensor, img_array)
        if content_analysis is None:
            # Lightweight CPU fallback: face detection + defaults
            import cv2
            gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            try:
                fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = fc.detectMultiScale(gray, 1.1, 4)
                face_count = len(faces)
            except Exception:
                face_count = 0
            content_analysis = {
                'has_faces': face_count > 0, 'face_count': face_count,
                'clip_vulnerability': 0.5, 'texture_complexity': 0.5,
                'risk_level': 'HIGH' if face_count > 0 else 'MEDIUM',
            }
            print("      [OK] (CPU fallback)")
        print(f"      [OK] Risk: {content_analysis['risk_level']}")
        print(f"      [OK] Faces: {content_analysis['face_count']}")
        print(f"      [OK] Vulnerability: {content_analysis['clip_vulnerability']:.3f}")
        
        # Step 4: Compute adaptive parameters
        print("\n[4/9] Computing parameters...")
        epsilon = self.adaptive_engine.compute_adaptive_epsilon(content_analysis)
        freq_weights = self.adaptive_engine.compute_frequency_weights(content_analysis)
        print(f"      [OK] Epsilon: {epsilon*255:.3f}/255")
        
        # Step 5: Apply LEAT adversarial perturbation + confusion transforms
        print("\n[5/9] Applying deepfake disruption...")
        leat_metrics = None
        leat_perturbation = None

        # Try remote GPU first
        if self.gpu_client and self.gpu_client.is_available():
            print("      Requesting LEAT from remote GPU server...")
            leat_result = self.gpu_client.generate_leat(
                img_pil,
                iterations=self.config.LEAT_ITERATIONS,
                epsilon=self.config.LEAT_EPSILON,
                step_size=self.config.LEAT_STEP_SIZE,
            )
            if leat_result is not None:
                leat_perturbation = torch.from_numpy(
                    leat_result["perturbation"]
                ).to(self.device)
                leat_metrics = leat_result["leat_metrics"]
                avg_latent_dist = np.mean([
                    m.get('latent_cosine_distance', 0.0)
                    for m in leat_metrics.values()
                    if isinstance(m, dict)
                ])
                print(f"      [OK] Remote LEAT done (avg disruption: {avg_latent_dist:.4f})")
            else:
                print("      [!] Remote LEAT failed, falling back...")

        # Try local LEAT if remote didn't work
        if leat_perturbation is None and self.leat is not None:
            print("      Generating LEAT adversarial perturbation locally...")
            leat_perturbation = self.leat.generate_perturbation(img_tensor)
            leat_metrics = self.leat.compute_latent_disruption(
                img_tensor, leat_perturbation
            )
            avg_latent_dist = np.mean([
                m['latent_cosine_distance'] for m in leat_metrics.values()
            ])
            print(f"      [OK] LEAT perturbation generated (avg latent disruption: {avg_latent_dist:.4f})")

        if leat_perturbation is not None:
            # Scale LEAT perturbation to respect adaptive epsilon
            leat_scale = min(epsilon / self.config.LEAT_EPSILON, 1.0)
            leat_perturbation = leat_perturbation * leat_scale

            # Also compute confusion transforms for supplementary protection
            confusion = ConfusionTransform(epsilon, freq_weights)
            patch_noise, freq_distorted = confusion.apply(img_tensor, img_array)

            # Blend LEAT perturbation with confusion noise
            leat_blend = self.config.LEAT_BLEND_RATIO
            combined_perturbation = (
                leat_perturbation * leat_blend
                + patch_noise.to(self.device) * (1 - leat_blend)
            )

            protected_tensor = img_tensor.clone() + combined_perturbation
            protected_tensor = torch.clamp(protected_tensor, 0, 1)
            protected_array = tensor_to_array(protected_tensor)

            # Light frequency distortion blend
            freq_blend = self.config.TRANSFORM_BLEND_RATIO * 0.5
            protected_array = (
                freq_distorted * freq_blend
                + protected_array * (1 - freq_blend)
            )
            protected_array = np.clip(protected_array, 0, 1)
            print("      [OK] LEAT + confusion transforms applied")
        else:
            # Fallback: confusion transforms only (original behavior)
            print("      Using confusion transforms (LEAT not available)...")
            confusion = ConfusionTransform(epsilon, freq_weights)
            patch_noise, freq_distorted = confusion.apply(img_tensor, img_array)

            protected_tensor = img_tensor.clone() + patch_noise.to(self.device)
            protected_tensor = torch.clamp(protected_tensor, 0, 1)
            protected_array = tensor_to_array(protected_tensor)

            blend_ratio = self.config.TRANSFORM_BLEND_RATIO
            protected_array = (
                freq_distorted * blend_ratio
                + protected_array * (1 - blend_ratio)
            )
            protected_array = np.clip(protected_array, 0, 1)
            print("      [OK] Confusion transforms applied")
        
        # Step 6: Embed watermarks
        print("\n[6/9] Embedding watermarks...")
        
        protected_array = self.spectral_watermark.embed(protected_array)
        print("      [OK] Spectral embedded")
        
        protected_array = self.multiband_watermark.embed(protected_array)
        print("      [OK] Multi-band frequency embedded")
        
        protected_array = self.stable_signature.embed(protected_array)
        print("      [OK] Stable Signature embedded")
        
        # Step 7: Compute metrics
        print("\n[7/9] Computing metrics...")
        protected_tensor = array_to_tensor(protected_array, self.device)
        
        psnr_value = compute_psnr(img_array, protected_array)
        ssim_value = compute_ssim(img_array, protected_array)

        # Compute embedding drift (remote GPU or local)
        embedding_drift = 0.0
        if self.gpu_client and self.gpu_client.is_available():
            from PIL import Image as _PILImage
            prot_pil = _PILImage.fromarray((protected_array * 255).clip(0, 255).astype(np.uint8))
            drift_val = self.gpu_client.compute_drift(img_pil, prot_pil)
            if drift_val is not None:
                embedding_drift = drift_val
                print("      [OK] Drift computed (remote GPU)")
        elif self.drift_controller is not None:
            embedding_drift = self.drift_controller.compute_embedding_drift(
                img_tensor,
                protected_tensor.detach()
            )
        
        psnr_ok = psnr_value >= self.config.TARGET_PSNR
        ssim_ok = ssim_value >= self.config.TARGET_SSIM
        
        print(f"      {'[OK]' if psnr_ok else '[!]'} PSNR: {psnr_value:.2f} dB (target: {self.config.TARGET_PSNR})")
        print(f"      {'[OK]' if ssim_ok else '[!]'} SSIM: {ssim_value:.4f} (target: {self.config.TARGET_SSIM})")
        print(f"      [OK] Drift: {embedding_drift:.4f}")
        
        # Step 8: Verify hashes
        print("\n[8/9] Verifying hashes...")
        protected_hashes = self.phash_system.generate_all_hashes(protected_array)
        hash_verification = self.phash_system.verify_hashes(
            original_hashes,
            protected_hashes,
            threshold=self.config.PHASH_THRESHOLD
        )
        
        for hash_type, result in hash_verification.items():
            status = "[OK]" if result['match'] else "[!]"
            print(f"      {status} {hash_type}: distance={result['distance']}")
        
        # Step 9: Save and add C2PA metadata
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"protected_{base_name}.png"
        
        save_image(protected_array, output_path)
        
        # C2PA Metadata
        c2pa_manifest = None
        if self.c2pa:
            print("\n[9/9] Embedding C2PA Metadata...")
            protection_metrics = {
                'psnr': float(psnr_value),
                'ssim': float(ssim_value),
                'drift': float(embedding_drift)
            }
            c2pa_manifest = self.c2pa.create_manifest(
                original_path=image_path,
                protected_path=output_path,
                protection_metrics=protection_metrics,
                fingerprint=fingerprint
            )
            self.c2pa.embed_in_image(output_path, c2pa_manifest)
            print("      [OK] C2PA manifest embedded")
            print("      [OK] 'No AI Training' rights applied")
        
        print("\n" + "=" * 80)
        print("[OK] PROTECTION COMPLETE")
        print("=" * 80)
        print(f"[SAVE] Saved: {output_path}")
        
        return output_path, {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'embedding_drift': embedding_drift,
            'content_analysis': content_analysis,
            'original_hashes': original_hashes,
            'protected_hashes': protected_hashes,
            'hash_verification': hash_verification,
            'fingerprint': fingerprint,
            'c2pa_manifest': c2pa_manifest,
            'leat_metrics': leat_metrics,
        }
    
    def verify_protection(self, protected_path: str,
                          original_hashes: dict = None,
                          fingerprint: dict = None) -> dict:
        """
        Verify protection on an image.
        
        Args:
            protected_path: Path to protected image
            original_hashes: Original hashes for comparison (optional)
            fingerprint: Original fingerprint (optional)
            
        Returns:
            dict: Verification results
        """
        print_banner("VERIFICATION")
        
        img_array, _, _ = load_image(protected_path)
        
        results = {
            'stable_signature': {},
            'spectral': {},
            'multiband': {},
            'hashes': {},
            'c2pa': {}
        }
        
        # Stable Signature
        print("\n1. Stable Signature...")
        extracted, confidence = self.stable_signature.extract(img_array)
        results['stable_signature'] = {
            'detected': confidence > 0.5,
            'confidence': confidence,
            'message': extracted
        }
        if confidence > 0.5:
            print(f"   [OK] Detected with {confidence*100:.1f}% confidence")
        else:
            print(f"   [!] Weak detection: {confidence*100:.1f}%")
        
        # Spectral
        print("\n2. Spectral Watermark...")
        correlation = self.spectral_watermark.detect(img_array)
        results['spectral'] = {
            'detected': correlation > 0.15,
            'correlation': correlation
        }
        print(f"   {'[OK]' if correlation > 0.15 else '[!]'} Correlation: {correlation:.3f}")
        
        # Multi-band
        print("\n3. Multi-Band Frequency...")
        multiband_corr = self.multiband_watermark.detect(img_array)
        band_strengths = self.multiband_watermark.get_band_strengths(img_array)
        results['multiband'] = {
            'detected': multiband_corr > 0.1,
            'correlation': multiband_corr,
            'band_strengths': band_strengths
        }
        print(f"   {'[OK]' if multiband_corr > 0.1 else '[!]'} Correlation: {multiband_corr:.3f}")
        
        # Perceptual Hashes
        print("\n4. Perceptual Hashes...")
        current_hashes = self.phash_system.generate_all_hashes(img_array)
        
        if original_hashes:
            hash_verification = self.phash_system.verify_hashes(
                original_hashes,
                current_hashes,
                threshold=self.config.PHASH_THRESHOLD
            )
            results['hashes'] = hash_verification
            for hash_type, result in hash_verification.items():
                status = "[OK]" if result['match'] else "[!]"
                print(f"   {status} {hash_type}: distance={result['distance']}")
        else:
            print(f"   [OK] pHash: {current_hashes['phash']['string'][:16]}...")
        
        # C2PA
        if self.c2pa:
            print("\n5. C2PA Metadata...")
            c2pa_result = self.c2pa.verify_protection(protected_path)
            results['c2pa'] = c2pa_result
            if c2pa_result['verified']:
                print("   [OK] C2PA manifest verified")
            else:
                print("   [!] C2PA verification incomplete")
        
        print("\n" + "=" * 80)
        print("[OK] VERIFICATION COMPLETE")
        print("=" * 80)
        
        return results
