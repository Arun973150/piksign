"""
PikSign Full Detection Runner
Runs the complete detection pipeline with DIRE disabled.

Usage:
    python run_detection.py <image_path>
    python run_detection.py photos/fakeimage.jpeg
    python run_detection.py photos/real.jpg
"""

import sys
import os


def patch_dire_off(detector):
    """Monkey-patch the AI manipulation track to skip DIRE."""
    original_method = detector._run_ai_manipulation_track

    def _run_ai_no_dire(image_path):
        try:
            from piksign.ai_image_forensics import run_forensics_pipeline
            from piksign.ai_image_forensics import VERDICT_AI_GENERATED, VERDICT_AI_MANIPULATED
            import numpy as np

            r = run_forensics_pipeline(
                image_path,
                run_ela=True,
                run_prnu=True,
                run_geometric=True,
                run_dire=False,       # <-- DIRE OFF
                save_artifacts=False,
            )
            ela_score = r.ela.get('anomaly_score', 0.0) if r.ela else 0.0
            prnu_score = r.prnu.get('anomaly_score', 0.0) if r.prnu else 0.0
            geo_score = r.geometric.get('geometric_anomaly_score', 0.0) if r.geometric else 0.0
            dire_score = 0.0  # skipped
            combined = r.combined_anomaly_score or max(ela_score, prnu_score, geo_score)
            verdict = r.verdict

            if verdict in (VERDICT_AI_GENERATED, VERDICT_AI_MANIPULATED):
                ai_prob = combined
            else:
                ai_prob = combined * (1.0 - r.verdict_confidence)

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
                    'dire': 0.0,
                    'combined': combined,
                }
            }
        except ImportError:
            return {'status': 'unavailable', 'ai_probability': 0.0, 'verdict': 'unknown', 'scores': {}}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'ai_probability': 0.0, 'verdict': 'unknown', 'scores': {}}

    detector._run_ai_manipulation_track = _run_ai_no_dire


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_detection.py <image_path>")
        print("Example: python run_detection.py photos/fakeimage.jpeg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    from piksign.detection import PikSignDetector

    detector = PikSignDetector()

    # Patch out DIRE
    patch_dire_off(detector)
    print("\n[*] DIRE is DISABLED for this run.\n")

    # Run full analysis (all steps: protection check, AI manipulation, deepfake, forensics, verdict)
    results = detector.full_analysis(image_path)

    # C2PA standalone check (detailed)
    print("\n" + "=" * 60)
    print("[Step C2PA] Content Provenance & Authenticity Check")
    print("=" * 60)
    c2pa = results.get('c2pa_verification', {})
    c2pa_verified = c2pa.get('verified', False)
    if c2pa_verified:
        print("  Status:         VERIFIED")
        manifest = c2pa.get('manifest', {})
        if manifest:
            print(f"  Creator:        {manifest.get('creator', 'N/A')}")
            print(f"  Timestamp:      {manifest.get('timestamp', 'N/A')}")
            rights = manifest.get('rights_assertions', [])
            if rights:
                print("  Rights:")
                for r in rights:
                    label = r.get('label', r) if isinstance(r, dict) else r
                    print(f"    - {label}")
            sig = manifest.get('signature', {})
            if sig:
                print(f"  Signature:      {sig.get('algorithm', 'N/A')}")
                print(f"  Hash Match:     {sig.get('hash_verified', 'N/A')}")
    else:
        print("  Status:         NOT FOUND / NOT VERIFIED")
        reason = c2pa.get('error', c2pa.get('status', 'No C2PA manifest in image'))
        print(f"  Reason:         {reason}")

    # Manipulation forensics breakdown
    manip_analysis = results.get('forensics', {}).get('manipulation_analysis', {})
    manip_indiv = manip_analysis.get('individual_scores', {})
    if manip_indiv:
        print("\n" + "=" * 60)
        print("[Patch-Level Manipulation Forensics]")
        print("=" * 60)
        print(f"  GLCM (texture):        {manip_indiv.get('glcm', 0.0)*100:5.1f}%")
        print(f"  LBP  (micro-texture):  {manip_indiv.get('lbp', 0.0)*100:5.1f}%")
        print(f"  Wavelet (HF kurtosis): {manip_indiv.get('wavelet', 0.0)*100:5.1f}%")
        print(f"  Edge density:          {manip_indiv.get('edge_density', 0.0)*100:5.1f}%")
        print(f"  Benford (DCT):         {manip_indiv.get('benford', 0.0)*100:5.1f}%")
        print(f"  Combined:              {manip_analysis.get('manipulation_score', 0.0)*100:5.1f}%")

    # Print summary
    v = results['final_verdict']
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Image:          {image_path}")
    print(f"  Verdict:        {v['final_verdict']}")
    print(f"  Confidence:     {v['final_confidence']*100:.1f}%")
    print(f"  AI Score:       {v['P_ai']*100:.1f}%")
    print(f"  Manipulation:   {v['P_manipulated']*100:.1f}%")
    print(f"  Faces Found:    {v.get('faces_detected', 0)}")
    print(f"  C2PA Verified:  {c2pa_verified}")
    print("=" * 60)


if __name__ == '__main__':
    main()
