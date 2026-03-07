# -*- coding: utf-8 -*-
"""
PikSign Detection Module (Phase 2 - Visual Intelligence)
Detects AI-generated content, deepfakes, and manipulated media.

Components:
- AI detection: ai-image forensics pipeline (ELA, PRNU, geometric, DIRE) - replaces former CLIP/DINOv2
- DeepfakeDetector: Reality Defender for deepfake detection
- ForensicsAnalyzer: Frequency/embedding forensic analysis
"""

from typing import Dict, Any, Optional
import numpy as np

# Import detection components (AIContentDetector disabled; AI from ai_image_forensics pipeline)
from piksign.detection.deepfake_detector import DeepfakeDetector
from piksign.detection.forensics import ForensicsAnalyzer

# Import new v2 modules
try:
    from piksign.detection.real_image_validator import RealImageValidator
    HAS_REAL_VALIDATOR = True
except ImportError:
    HAS_REAL_VALIDATOR = False

try:
    from piksign.detection.exif_validator import EXIFValidator
    HAS_EXIF_VALIDATOR = True
except ImportError:
    HAS_EXIF_VALIDATOR = False

try:
    from piksign.detection.watermark_detector import WatermarkDetector
    HAS_WATERMARK_DETECTOR = True
except ImportError:
    HAS_WATERMARK_DETECTOR = False

try:
    from piksign.detection.face_deepfake_detector import FaceDeepfakeDetector
    HAS_FACE_DETECTOR = True
except ImportError:
    HAS_FACE_DETECTOR = False

__all__ = [
    'PikSignDetector',
    'DeepfakeDetector',
    'ForensicsAnalyzer'
]


class PikSignDetector:
    """
    PikSign Visual Intelligence - Detection System (v3.0)

    Gated detection flow:
        Step 0 -> PikSign protection check (early exit if found)
        Step 1 -> AI Manipulation track  (ELA + PRNU + Geometric)
        Step 2 -> Deepfake track          (Reality Defender primary, local fallback)
        Step 3 -> Forensics               (frequency, embedding, color, noise)
        Step 4 -> Final verdict
    """

    # AI probability threshold -- 0.55 balances ~85% TPR / ~10% FPR
    AI_THRESHOLD = 0.55

    def __init__(self, device=None, api_key: str = None, **kwargs):
        import torch
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 60)
        print("PIKSIGN VISUAL INTELLIGENCE - DETECTION SYSTEM v3.0")
        print("=" * 60)

        # Step 1 backend: ai_image_forensics
        print("\n[1/3] AI Manipulation Backend (ELA / PRNU / Geo / DIRE)...")
        self._ai_forensics_ready = True  # lazy init at detect time

        # Step 2 backend: Reality Defender
        print("\n[2/3] Deepfake Detector (Reality Defender + local)...")
        self.deepfake_detector = DeepfakeDetector(device=self.device, api_key=api_key)

        # Step 3 backend: forensics
        print("\n[3/3] Forensics Analyzer...")
        self.forensics = ForensicsAnalyzer(self.device)

        # Optional extras
        self.face_detector = None
        if HAS_FACE_DETECTOR:
            try:
                self.face_detector = FaceDeepfakeDetector(self.device)
            except Exception:
                pass

        self.exif_validator = EXIFValidator() if HAS_EXIF_VALIDATOR else None
        self.watermark_detector = WatermarkDetector() if HAS_WATERMARK_DETECTOR else None

        # C2PA verifier
        self.c2pa = None
        try:
            from piksign.protection.c2pa import C2PAMetadataBinding
            self.c2pa = C2PAMetadataBinding()
        except ImportError:
            pass

        print("\n" + "=" * 60)
        print("Detection system ready.")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _check_piksign_protection(self, image_path: str) -> dict:
        """
        Step 0: Check whether the image carries PikSign protection markers.
        Checks PNG text chunks and (if piexif available) JPEG EXIF.
        Returns dict with 'is_protected' bool and supporting details.
        """
        result = {
            'is_protected': False,
            'method': None,
            'png_marker': False,
            'watermark_signal': False,
            'c2pa_verified': False,
        }

        # 1. PNG text chunk - fastest, most reliable
        try:
            from PIL import Image as _PIL
            img = _PIL.open(image_path)
            if hasattr(img, 'text') and img.text.get('PikSign_Protected') == 'true':
                result['png_marker'] = True
                result['is_protected'] = True
                result['method'] = 'png_text_chunk'
                return result  # definitive -- no need to check further
        except Exception:
            pass

        # 2. JPEG EXIF ImageDescription
        try:
            import piexif
            from PIL import Image as _PIL
            img = _PIL.open(image_path)
            raw_exif = img.info.get('exif')
            if raw_exif:
                exif_dict = piexif.load(raw_exif)
                desc = exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, b'')
                if b'PikSign_Protected:true' in desc:
                    result['is_protected'] = True
                    result['method'] = 'jpeg_exif'
                    return result
        except Exception:
            pass

        # 3. ProtectionForensics watermark signal (spectral / multiband / stable sig)
        try:
            from piksign.detection.forensics import ProtectionForensics
            import numpy as np
            from PIL import Image as _PIL
            img_arr = np.array(_PIL.open(image_path).convert('RGB')).astype(np.float32) / 255.0
            pf = ProtectionForensics()
            pf_result = pf.analyze(img_arr)
            if pf_result.get('protection_detected', False):
                result['watermark_signal'] = True
                result['is_protected'] = True
                result['method'] = 'watermark_signal'
                result['watermark_scores'] = {
                    'spectral': pf_result.get('spectral_score', 0.0),
                    'multiband': pf_result.get('multiband_score', 0.0),
                    'stable_signature': pf_result.get('stable_signature_score', 0.0),
                }
        except Exception:
            pass

        # 4. C2PA manifest (slowest, most authoritative)
        if self.c2pa and not result['is_protected']:
            try:
                c2pa_result = self.c2pa.verify_protection(image_path)
                if c2pa_result.get('verified'):
                    result['c2pa_verified'] = True
                    result['is_protected'] = True
                    result['method'] = 'c2pa_manifest'
            except Exception:
                pass

        return result

    def _run_ai_manipulation_track(self, image_path: str) -> dict:
        """
        Step 1: AI manipulation forensics (ELA + PRNU + Geometric + DIRE).
        Returns structured dict with individual scores and verdict.
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
            ela_score = r.ela.get('anomaly_score', 0.0) if r.ela else 0.0
            prnu_score = r.prnu.get('anomaly_score', 0.0) if r.prnu else 0.0
            geo_score = r.geometric.get('geometric_anomaly_score', 0.0) if r.geometric else 0.0
            dire_score = r.dire if r.dire is not None else 0.0
            combined = r.combined_anomaly_score or max(ela_score, prnu_score, geo_score)
            verdict = r.verdict
            # Derive a single probability
            if verdict in (VERDICT_AI_GENERATED, VERDICT_AI_MANIPULATED):
                ai_prob = max(combined, dire_score)
            else:
                ai_prob = combined * (1.0 - r.verdict_confidence)
            import numpy as np
            ai_prob = float(np.clip(ai_prob, 0.0, 1.0))
            return {
                'status': 'success',
                'ai_probability': ai_prob,
                'verdict': verdict,
                'verdict_confidence': float(r.verdict_confidence),
                'scores': {
                    'ela': ela_score,
                    'prnu': prnu_score,
                    'geometric': geo_score,
                    'dire': dire_score,
                    'combined': combined,
                }
            }
        except ImportError:
            return {'status': 'unavailable', 'ai_probability': 0.0, 'verdict': 'unknown', 'scores': {}}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'ai_probability': 0.0, 'verdict': 'unknown', 'scores': {}}

    def _run_deepfake_track(self, image_path: str) -> dict:
        """
        Step 2: Deepfake detection via Reality Defender (primary).
        Local face analysis runs as supplementary context.
        """
        # Primary: Reality Defender
        rd_result = self.deepfake_detector.detect(image_path)

        # Supplementary: local face deepfake detector (if available)
        face_result = {}
        if self.face_detector is not None and self.face_detector.enabled:
            try:
                face_result = self.face_detector.detect(image_path)
            except Exception:
                pass

        return {
            'status': rd_result.get('status', 'unknown'),
            'deepfake_probability': rd_result.get('deepfake_probability', 0.0),
            'rd_status': rd_result.get('individual_results', {}).get('reality_defender', {}).get('rd_status', 'N/A'),
            'rd_models': rd_result.get('individual_results', {}).get('reality_defender', {}).get('models', []),
            'manipulation_type': rd_result.get('manipulation_type', 'unknown'),
            'face_detection': face_result,
            'faces_detected': face_result.get('faces_detected', 0),
            'face_score': face_result.get('final_face_score', 0.0),
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def detect_ai_content(self, image_path: str) -> dict:
        """Run AI manipulation forensics track (ELA + PRNU + Geometric + DIRE)."""
        return self._run_ai_manipulation_track(image_path)

    def detect_deepfake(self, image_path: str, use_api: bool = True) -> dict:
        """Run deepfake detection track (Reality Defender + local)."""
        return self._run_deepfake_track(image_path)

    def analyze_forensics(self, image_path: str, ai_confidence: float = 0.0) -> dict:
        """Run supplementary forensic analysis (frequency, embedding, color)."""
        return self.forensics.analyze(image_path, ai_confidence)

    def verify_c2pa(self, image_path: str) -> dict:
        """Verify C2PA metadata and provenance."""
        if self.c2pa:
            return self.c2pa.verify_protection(image_path)
        return {'status': 'unavailable', 'verified': False}

    def full_analysis(self, image_path: str,
                      include_deepfake: bool = True,
                      include_forensics: bool = True) -> dict:
        """
        Comprehensive detection using the gated v3.0 flow.

        Flow:
            0. PikSign protection check  -> early exit if protected
            1. AI Manipulation track     (ELA / PRNU / Geo / DIRE)
            2. Deepfake track            (Reality Defender + local)
            3. Forensics                 (supplementary, always runs)
            4. Final verdict
        """
        def _bar(p):
            b = int(float(p) * 10)
            return "#" * b + "." * (10 - b)

        print("\n" + "=" * 60)
        print("FULL ANALYSIS  (PikSign v3.0)")
        print("=" * 60)

        results = {
            'image_path': image_path,
            'piksign_check': {},
            'ai_manipulation': {},
            'deepfake_detection': {},
            'forensics': {},
            'exif': {},
            'watermark': {},
            'c2pa_verification': {},
            'final_verdict': {},
        }

        # -- Step 0: PikSign protection check ---------------------------------
        print("\n[Step 0] PikSign Protection Check...")
        prot = self._check_piksign_protection(image_path)
        results['piksign_check'] = prot

        if prot['is_protected']:
            method = prot.get('method', 'unknown')
            print(f"   [OK] PIKSIGN PROTECTION DETECTED  (via {method})")
            print("   -> No further analysis needed for protected images.")
            results['final_verdict'] = {
                'final_verdict': 'PIKSIGN_PROTECTED',
                'final_confidence': 1.0,
                'P_ai': 0.0,
                'P_manipulated': 0.0,
                'note': f'Protection confirmed via {method}.'
            }
            self._print_verdict(results['final_verdict'], _bar)
            return results

        print("   [FAIL] No PikSign protection found. Running full analysis...")

        # -- Step 1: AI Manipulation Track ------------------------------------
        print("\n[Step 1] AI Manipulation Analysis (ELA / PRNU / Geometric / DIRE)...")
        ai_manip = self._run_ai_manipulation_track(image_path)
        results['ai_manipulation'] = ai_manip

        ela = ai_manip.get('scores', {}).get('ela', 0.0)
        prnu = ai_manip.get('scores', {}).get('prnu', 0.0)
        geo = ai_manip.get('scores', {}).get('geometric', 0.0)
        dire = ai_manip.get('scores', {}).get('dire', 0.0)
        manip_ai_prob = ai_manip.get('ai_probability', 0.0)

        print(f"   ELA  (error level):   {ela*100:5.1f}%  {_bar(ela)}")
        print(f"   PRNU (sensor noise):  {prnu*100:5.1f}%  {_bar(prnu)}")
        print(f"   Geo  (geometry):      {geo*100:5.1f}%  {_bar(geo)}")
        print(f"   DIRE (diffusion err): {dire*100:5.1f}%  {_bar(dire)}")
        print(f"   Combined AI prob:     {manip_ai_prob*100:5.1f}%  {_bar(manip_ai_prob)}")
        print(f"   Verdict:              {ai_manip.get('verdict', 'unknown')}")

        # -- Step 2: Deepfake Track --------------------------------------------
        print("\n[Step 2] Deepfake Detection (Reality Defender + local)...")
        if include_deepfake:
            deepfake = self._run_deepfake_track(image_path)
        else:
            deepfake = {'status': 'skipped', 'deepfake_probability': 0.0,
                        'rd_status': 'N/A', 'rd_models': [],
                        'faces_detected': 0, 'face_score': 0.0}
        results['deepfake_detection'] = deepfake

        rd_prob = deepfake.get('deepfake_probability', 0.0)
        rd_status = deepfake.get('rd_status', 'N/A')
        rd_models = deepfake.get('rd_models', [])
        faces = int(deepfake.get('faces_detected', 0))
        face_score = deepfake.get('face_score', 0.0)

        print(f"   Reality Defender:     {rd_prob*100:5.1f}%  {_bar(rd_prob)}  ({rd_status})")
        for m in rd_models:
            ms = m.get('score', 0.0)
            print(f"     |- {m['name']:20s} {ms*100:5.1f}%  {_bar(ms)}")
        if faces > 0:
            print(f"   Face analysis ({faces} face{'s' if faces > 1 else ''}): {face_score*100:5.1f}%  {_bar(face_score)}")

        # -- Step 3: Forensics ------------------------------------------------
        print("\n[Step 3] Supplementary Forensics (frequency / embedding / color / manipulation)...")
        if include_forensics:
            forensics_res = self.forensics.analyze(image_path, ai_confidence=rd_prob)
        else:
            forensics_res = {}
        results['forensics'] = forensics_res

        freq_score = forensics_res.get('frequency_analysis', {}).get('combined_manipulation_score', 0.0)
        embed_score = forensics_res.get('embedding_analysis', {}).get('manipulation_probability', 0.0)
        color_score = max(
            forensics_res.get('color_analysis', {}).get('correlation', {}).get('anomaly_score', 0.0),
            forensics_res.get('color_analysis', {}).get('noise', {}).get('noise_inconsistency', 0.0)
        )

        # Patch-level manipulation forensics
        manip_analysis = forensics_res.get('manipulation_analysis', {})
        manip_indiv = manip_analysis.get('individual_scores', {})
        manip_combined = manip_analysis.get('manipulation_score', 0.0)

        overall_manip = forensics_res.get('overall_manipulation_score',
                                          (freq_score * 0.25 + embed_score * 0.20 +
                                           color_score * 0.15 + manip_combined * 0.40))

        print(f"   Frequency:            {freq_score*100:5.1f}%  {_bar(freq_score)}")
        print(f"   Embedding:            {embed_score*100:5.1f}%  {_bar(embed_score)}")
        print(f"   Color/Noise:          {color_score*100:5.1f}%  {_bar(color_score)}")
        print(f"   --- Patch-level Manipulation Forensics ---")
        print(f"   GLCM (texture):       {manip_indiv.get('glcm', 0.0)*100:5.1f}%  {_bar(manip_indiv.get('glcm', 0.0))}")
        print(f"   LBP  (micro-texture): {manip_indiv.get('lbp', 0.0)*100:5.1f}%  {_bar(manip_indiv.get('lbp', 0.0))}")
        print(f"   Wavelet (HF kurtosis):{manip_indiv.get('wavelet', 0.0)*100:5.1f}%  {_bar(manip_indiv.get('wavelet', 0.0))}")
        print(f"   Edge density:         {manip_indiv.get('edge_density', 0.0)*100:5.1f}%  {_bar(manip_indiv.get('edge_density', 0.0))}")
        print(f"   Benford (DCT):        {manip_indiv.get('benford', 0.0)*100:5.1f}%  {_bar(manip_indiv.get('benford', 0.0))}")
        print(f"   Manip combined:       {manip_combined*100:5.1f}%  {_bar(manip_combined)}")

        # Optional extras: EXIF + watermark
        print("\n[Step 4] EXIF & Watermark Check...")
        if self.exif_validator:
            exif_val = self.exif_validator.validate(image_path)
            results['exif'] = exif_val
            ai_sig = exif_val.get('ai_signature_found', False)
            print(f"   EXIF AI signature:    {'FOUND' if ai_sig else 'not found'}")
        else:
            print("   EXIF validator:       not available")
        if self.watermark_detector:
            print("   Watermark detection:  running...")
            wm = self.watermark_detector.detect(image_path)
            results['watermark'] = wm
            wm_det = wm.get('watermark_detected', False)
            wm_type = wm.get('watermark_type', 'none')
            wm_conf = wm.get('watermark_confidence', 0.0)
            print(f"   Watermark detected:   {'YES (' + wm_type + ')' if wm_det else 'no'}  {wm_conf*100:.1f}%  {_bar(wm_conf)}")
        else:
            print("   Watermark detector:   not available")

        # C2PA (purely informational at this stage since Step 0 already checked)
        print("\n[Step 5] C2PA Provenance...")
        results['c2pa_verification'] = self.verify_c2pa(image_path)
        c2pa_ok = results['c2pa_verification'].get('verified', False)
        print(f"   C2PA:                 {'VERIFIED AUTHENTIC' if c2pa_ok else 'not found / not verified'}")

        # -- Step 6: Final Verdict --------------------------------------------
        verdict = self._compute_verdict(
            manip_ai_prob=manip_ai_prob,
            rd_prob=rd_prob,
            overall_manip=overall_manip,
            faces_detected=faces,
            face_score=face_score,
            watermark_res=results.get('watermark', {}),
            exif_res=results.get('exif', {}),
            c2pa_verified=results['c2pa_verification'].get('verified', False),
        )
        results['final_verdict'] = verdict
        self._print_verdict(verdict, _bar)

        return results

    def _compute_verdict(self, *,
                         manip_ai_prob: float,
                         rd_prob: float,
                         overall_manip: float,
                         faces_detected: int,
                         face_score: float,
                         watermark_res: dict,
                         exif_res: dict,
                         c2pa_verified: bool) -> dict:
        """
        Compute final verdict from Step 1-3 signals.

        Primary signals:
          * manip_ai_prob  -- ELA/PRNU/Geo/DIRE combined   (50% weight)
          * rd_prob        -- Reality Defender              (50% weight)
        Supplementary:
          * overall_manip  -- forensics manipulation score
          * face_score     -- face deepfake score
          * watermark / EXIF AI signatures                (always override upward)
          * C2PA verified                                 (always override to authentic)
        """
        import numpy as np

        # Blend primary signals: Reality Defender 65%, AI Manipulation 35%
        if rd_prob > 0.0 and manip_ai_prob > 0.0:
            P_ai = 0.65 * rd_prob + 0.35 * manip_ai_prob
        elif rd_prob > 0.0:
            P_ai = rd_prob
        else:
            P_ai = manip_ai_prob
        P_ai = float(np.clip(P_ai, 0.0, 1.0))

        # Watermark / EXIF boosts
        wm_detected = watermark_res.get('watermark_detected', False)
        wm_confidence = watermark_res.get('watermark_confidence', 0.0)
        ai_sig = exif_res.get('ai_signature_found', False)
        if wm_detected:
            P_ai = max(P_ai, wm_confidence)
        if ai_sig:
            P_ai = max(P_ai, 0.65)

        P_manip = float(np.clip(overall_manip, 0.0, 1.0))
        T = self.AI_THRESHOLD  # 0.55

        # Decision tree
        final_verdict = 'UNKNOWN'
        final_confidence = 0.0

        if c2pa_verified:
            final_verdict = 'VERIFIED AUTHENTIC (C2PA)'
            final_confidence = 1.0
            P_ai = 0.0

        elif P_ai >= T and P_manip >= T:
            final_verdict = 'AI-GENERATED (HIGH CONFIDENCE)'
            final_confidence = min((P_ai + P_manip) / 2.0 * 1.05, 0.99)

        elif P_ai >= T:
            final_verdict = 'AI-GENERATED'
            final_confidence = min(P_ai, 0.99)

        elif P_ai >= 0.45:
            if P_manip >= T:
                final_verdict = 'REAL (POSSIBLY MANIPULATED)'
                final_confidence = P_manip
            else:
                final_verdict = 'POSSIBLY AI-GENERATED (UNCERTAIN)'
                final_confidence = P_ai

        elif faces_detected > 0 and face_score >= 0.70:
            final_verdict = 'REAL IMAGE WITH DEEPFAKE FACES'
            final_confidence = face_score

        elif P_manip >= T:
            final_verdict = 'REAL (POSSIBLY MANIPULATED)'
            final_confidence = P_manip

        else:
            final_verdict = 'REAL (LIKELY AUTHENTIC)'
            max_threat = max(P_ai, P_manip, face_score if faces_detected else 0.0)
            final_confidence = 1.0 - max_threat

        return {
            'final_verdict': final_verdict,
            'final_confidence': float(final_confidence),
            'P_ai': float(P_ai),
            'P_manipulated': float(P_manip),
            'manip_ai_prob': float(manip_ai_prob),
            'rd_prob': float(rd_prob),
            'faces_detected': int(faces_detected),
            'face_score': float(face_score),
            'watermark_detected': wm_detected,
            'ai_signature': ai_sig,
            'c2pa_verified': c2pa_verified,
            'threshold_used': T,
        }

    def _print_verdict(self, v: dict, bar_fn) -> None:
        print("\n" + "=" * 60)
        print("[>] FINAL VERDICT")
        print("=" * 60)
        print(f"Classification:     {v['final_verdict']}")
        print(f"Confidence:         {v['final_confidence']*100:.1f}%")
        ai = v.get('P_ai', 0.0)
        manip = v.get('P_manipulated', 0.0)
        print(f"AI score:           {ai*100:.1f}%  {bar_fn(ai)}")
        print(f"Manipulation score: {manip*100:.1f}%  {bar_fn(manip)}")
        print("=" * 60)

    # Legacy compat
    def detect_deepfake_video(self, video_path: str) -> dict:
        return self.deepfake_detector.detect_video(video_path)

