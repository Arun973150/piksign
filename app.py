import streamlit as st
import os
import sys
import io
import contextlib
import tempfile
import json
import time
import numpy as np
from PIL import Image
from typing import Dict, Any

# Bridge Streamlit secrets to os.environ for backend modules
try:
    for key, val in st.secrets.items():
        if isinstance(val, str):
            os.environ[key] = val
except Exception:
    pass


try:
    from piksign.protection.c2pa import C2PAMetadataBinding
    BACKEND_READY = True
except ImportError as e:
    st.error(f"Backend modules not found. Ensure you are running from the project root. Error: {e}")
    BACKEND_READY = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PikSign Intelligence Hub",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
    }
    .gradient-text {
        background: linear-gradient(90deg, #A855F7, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }
    .verdict-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .verdict-secure { background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; color: #4ade80; }
    .verdict-threat { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; }
    .verdict-warning { background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #fbbf24; }
    .verdict-prot { background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #60a5fa; }
    .score-bar {
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
    }
    .score-label {
        width: 180px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .score-value {
        width: 60px;
        font-weight: 600;
        text-align: right;
        margin-right: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #A855F7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_detector():
    from piksign.detection import PikSignDetector
    return PikSignDetector()

@st.cache_resource
def get_gpu_client(url: str):
    from piksign.gpu_client import ColabGPUClient
    return ColabGPUClient(url)

@st.cache_resource
def get_shield(colab_url: str):
    from piksign.protection.shield import PikSignShield
    client = get_gpu_client(colab_url) if colab_url else None
    return PikSignShield(gpu_client=client)


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### PikSign v3.0")
    if BACKEND_READY:
        st.success("Core Engine: ACTIVE")
    else:
        st.error("Core Engine: OFFLINE")

    st.divider()
    st.markdown("### Detection Config")
    ai_threshold = st.slider("AI Sensitivity Threshold", 0.0, 1.0, 0.55, 0.05)
    st.info(f"Recommended: 0.55 (optimized for TPR/FPR)")
    dire_enabled = st.toggle("Enable DIRE", value=False, help="Diffusion Reconstruction Error analysis. Slow but catches diffusion model outputs.")

    st.divider()
    st.markdown("### Protection Config")
    leat_enabled = st.toggle("Enable LEAT", value=True, help="Latent Ensemble Attack: adversarial perturbations that disrupt deepfake encoder latent spaces")
    leat_iterations = st.slider("LEAT PGD Iterations", 5, 50, 20, 5, help="More iterations = stronger disruption but slower")
    leat_epsilon = st.slider("LEAT Epsilon (x255)", 1.0, 16.0, 12.75, 0.25, help="Max perturbation magnitude (in /255 units)")

    st.divider()
    st.markdown("### GPU Server (Colab)")
    colab_url = st.text_input(
        "Colab Server URL",
        value=st.session_state.get("colab_url", ""),
        placeholder="https://xxxx.ngrok-free.app",
        help="Paste the ngrok URL from your Colab notebook"
    )
    if colab_url:
        st.session_state["colab_url"] = colab_url

    gpu_client = None
    if colab_url:
        try:
            gpu_client = get_gpu_client(colab_url)
            health = gpu_client.get_health()  # single HTTP call; also warms the is_available cache
            if health is not None:
                gpu_name = health.get("gpu", "GPU")
                gpu_client._healthy = True
                gpu_client._last_check = time.time()
                st.success(f"GPU: CONNECTED ({gpu_name})")
            else:
                st.warning("GPU: UNREACHABLE")
                gpu_client._healthy = False
                gpu_client._last_check = time.time()
                gpu_client = None
        except Exception as e:
            st.warning(f"GPU client error: {e}")
            gpu_client = None
    else:
        st.info("GPU: Not configured (LEAT via Colab disabled)")

    st.divider()
    st.caption("AI detection: ELA, PRNU, Geometric, DIRE (optional), Reality Defender, Patch-level forensics (GLCM, LBP, Wavelet, Edge, Benford). Protection: LEAT + 3 watermarks + C2PA.")


def patch_dire_off(detector):
    """Monkey-patch the AI manipulation track to skip DIRE. Saves original for restoration."""
    if not hasattr(detector, '_orig_run_ai_manipulation_track'):
        detector._orig_run_ai_manipulation_track = detector._run_ai_manipulation_track

    def _run_ai_no_dire(image_path):
        try:
            from piksign.ai_image_forensics import run_forensics_pipeline
            from piksign.ai_image_forensics import VERDICT_AI_GENERATED, VERDICT_AI_MANIPULATED

            r = run_forensics_pipeline(
                image_path,
                run_ela=True,
                run_prnu=True,
                run_geometric=True,
                run_dire=False,
                save_artifacts=False,
            )
            ela_score = r.ela.get('anomaly_score', 0.0) if r.ela else 0.0
            prnu_score = r.prnu.get('anomaly_score', 0.0) if r.prnu else 0.0
            geo_score = r.geometric.get('geometric_anomaly_score', 0.0) if r.geometric else 0.0
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


def render_score_bar(label, value, color="#A855F7"):
    """Render a labeled progress bar for a score."""
    st.write(f"**{label}**")
    st.progress(min(max(value, 0.0), 1.0))
    st.caption(f"{value*100:.1f}%")


# --- APP TABS ---
st.markdown('<h1 class="gradient-text">PikSign Intelligence</h1>', unsafe_allow_html=True)
st.caption("Multi-layered Visual Media Security & Forensic Analysis")

tab1, tab2, tab3 = st.tabs(["Detection Hub", "Protection Shield", "C2PA Inspector"])

# =====================================================================
# TAB 1: DETECTION
# =====================================================================
with tab1:
    st.header("Intelligent Forensic Detection")
    st.markdown("Analyze an image for AI manipulation, deepfakes, subtle edits, and PikSign protection status.")

    uploaded_file = st.file_uploader("Drop image for forensic deep-dive", type=['png', 'jpg', 'jpeg'], key="detect_upload")

    if uploaded_file and BACKEND_READY:
        st.image(uploaded_file, width=300, caption="Target Image")

        if st.button("Run Detection", type="primary", key="run_detect_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner("Running Gated v3.0 Detection Flow..."):
                detector = get_detector()
                detector.AI_THRESHOLD = ai_threshold

                # Apply or restore DIRE patch based on sidebar toggle
                if not dire_enabled:
                    patch_dire_off(detector)
                elif hasattr(detector, '_orig_run_ai_manipulation_track'):
                    detector._run_ai_manipulation_track = detector._orig_run_ai_manipulation_track

                _log_buf = io.StringIO()
                with contextlib.redirect_stdout(_log_buf):
                    results = detector.full_analysis(tmp_path)
                _detection_logs = _log_buf.getvalue()

                verdict = results['final_verdict']
                v_text = verdict['final_verdict']
                v_conf = verdict.get('final_confidence', 0.0)

                # Verdict card
                v_class = "verdict-secure"
                if "AI-GENERATED" in v_text or "POSSIBLY AI" in v_text:
                    v_class = "verdict-threat"
                elif "MANIPULATED" in v_text:
                    v_class = "verdict-warning"
                elif "PROTECTED" in v_text:
                    v_class = "verdict-prot"

                st.markdown(f'<div class="verdict-box {v_class}">{v_text}</div>', unsafe_allow_html=True)
                st.metric("Confidence", f"{v_conf*100:.1f}%")

            st.divider()

            # --- Primary Signals ---
            st.markdown("### Primary Signals")
            c1, c2 = st.columns(2)
            with c1:
                ai_p = verdict.get('P_ai', 0.0)
                st.write("**AI Score (blended)**")
                st.progress(min(ai_p, 1.0))
                st.caption(f"{ai_p*100:.1f}%  |  Threshold: {ai_threshold*100:.1f}%")
            with c2:
                mn_p = verdict.get('P_manipulated', 0.0)
                st.write("**Manipulation Score**")
                st.progress(min(mn_p, 1.0))
                st.caption(f"{mn_p*100:.1f}%")

            st.divider()

            # --- Step 1: AI Manipulation Track ---
            st.markdown("### Step 1: AI Manipulation (ELA / PRNU / Geo" + (" / DIRE" if dire_enabled else "") + ")")
            ai_manip = results.get('ai_manipulation', {})
            scores = ai_manip.get('scores', {})

            s1a, s1b, s1c, s1d = st.columns(4)
            s1a.metric("ELA", f"{scores.get('ela', 0.0)*100:.1f}%")
            s1b.metric("PRNU", f"{scores.get('prnu', 0.0)*100:.1f}%")
            s1c.metric("Geometric", f"{scores.get('geometric', 0.0)*100:.1f}%")
            if dire_enabled:
                s1d.metric("DIRE", f"{scores.get('dire', 0.0)*100:.1f}%")
            else:
                s1d.metric("DIRE", "OFF")

            manip_ai_prob = ai_manip.get('ai_probability', 0.0)
            st.caption(f"Combined AI prob: {manip_ai_prob*100:.1f}%  |  Verdict: {ai_manip.get('verdict', 'N/A')}")

            st.divider()

            # --- Step 2: Deepfake Track ---
            st.markdown("### Step 2: Deepfake (Reality Defender)")
            deepfake = results.get('deepfake_detection', {})
            rd_prob = deepfake.get('deepfake_probability', 0.0)
            rd_status = deepfake.get('rd_status', 'N/A')
            rd_models = deepfake.get('rd_models', [])

            st.write(f"**Reality Defender:** {rd_prob*100:.1f}% ({rd_status})")
            st.progress(min(rd_prob, 1.0))

            if rd_models:
                with st.expander("Per-model breakdown"):
                    for m in rd_models:
                        ms = m.get('score', 0.0)
                        st.write(f"**{m['name']}**: {ms*100:.1f}%")
                        st.progress(min(ms, 1.0))

            faces = int(deepfake.get('faces_detected', 0))
            face_score = deepfake.get('face_score', 0.0)
            if faces > 0:
                st.caption(f"Face analysis: {faces} face(s) detected, score: {face_score*100:.1f}%")

            st.divider()

            # --- Step 3: Forensics ---
            st.markdown("### Step 3: Forensic Analysis")
            forensics = results.get('forensics', {})

            freq_score = forensics.get('frequency_analysis', {}).get('combined_manipulation_score', 0.0)
            embed_score = forensics.get('embedding_analysis', {}).get('manipulation_probability', 0.0)
            color_corr = forensics.get('color_analysis', {}).get('correlation', {}).get('anomaly_score', 0.0)
            color_noise = forensics.get('color_analysis', {}).get('noise', {}).get('noise_inconsistency', 0.0)
            color_score = max(color_corr, color_noise)

            f1, f2, f3 = st.columns(3)
            f1.metric("Frequency", f"{freq_score*100:.1f}%")
            f2.metric("Embedding", f"{embed_score*100:.1f}%")
            f3.metric("Color/Noise", f"{color_score*100:.1f}%")

            # --- Patch-level Manipulation Forensics ---
            st.markdown("#### Patch-Level Manipulation Forensics")
            manip_analysis = forensics.get('manipulation_analysis', {})
            manip_indiv = manip_analysis.get('individual_scores', {})
            manip_combined = manip_analysis.get('manipulation_score', 0.0)

            if manip_indiv:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("GLCM", f"{manip_indiv.get('glcm', 0.0)*100:.1f}%")
                m2.metric("LBP", f"{manip_indiv.get('lbp', 0.0)*100:.1f}%")
                m3.metric("Wavelet", f"{manip_indiv.get('wavelet', 0.0)*100:.1f}%")
                m4.metric("Edge", f"{manip_indiv.get('edge_density', 0.0)*100:.1f}%")
                m5.metric("Benford", f"{manip_indiv.get('benford', 0.0)*100:.1f}%")

                st.write(f"**Manipulation Combined:** {manip_combined*100:.1f}%")
                st.progress(min(manip_combined, 1.0))
            else:
                st.caption("Manipulation forensics data not available.")

            st.divider()

            # --- C2PA Check ---
            st.markdown("### C2PA Provenance")
            c2pa_res = results.get('c2pa_verification', {})
            c2pa_verified = c2pa_res.get('verified', False)
            if c2pa_verified:
                st.success("C2PA: VERIFIED AUTHENTIC")
            else:
                st.caption("C2PA: Not found / Not verified")

            # --- Detection Logs ---
            with st.expander("View Detection Logs"):
                st.code(_detection_logs, language="text")

            # --- Full JSON ---
            with st.expander("View Full JSON Report"):
                st.json(results)

            os.unlink(tmp_path)

# =====================================================================
# TAB 2: PROTECTION
# =====================================================================
with tab2:
    st.header("Secure Protection Shield")
    st.markdown("Apply PikSign transforms and C2PA provenance to your authentic images.")

    prot_file = st.file_uploader("Upload image to protect", type=['png', 'jpg', 'jpeg'], key="prot_upload")

    if gpu_client is None:
        st.warning("GPU Server not connected. LEAT protection disabled -- only CPU-based confusion transforms will be applied. Connect a Colab GPU server in the sidebar for full protection.")

    if prot_file and BACKEND_READY:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(prot_file.name)[1]) as tmp:
            tmp.write(prot_file.getvalue())
            in_path = tmp.name

        col1, col2 = st.columns(2)

        with col1:
            st.image(prot_file, use_container_width=True, caption="Original")

        with col2:
            if st.button("Run Protection", type="primary"):
                with st.spinner("Running 9-step protection pipeline..."):
                    out_path = tempfile.mktemp(suffix=".png")

                    shield = get_shield(colab_url or "")
                    _prot_log_buf = io.StringIO()
                    with contextlib.redirect_stdout(_prot_log_buf):
                        out_path_result, metrics = shield.protect_image(in_path, out_path)
                    _protection_logs = _prot_log_buf.getvalue()

                    status = metrics.get('status', '')

                    if out_path_result is None and status == 'already_protected':
                        st.info("Image is already protected. No changes made.")
                    elif out_path_result is None and status == 'ai_content_detected':
                        st.error(f"Rejected: {metrics.get('message', 'AI content detected')}")
                    elif out_path_result is None:
                        st.error(f"Protection failed: {metrics.get('message', metrics.get('error', 'Unknown error'))}")
                    else:
                        st.success("PROTECTION SUCCESSFUL")
                        st.divider()

                        # Quality metrics
                        st.markdown("#### Quality Metrics")
                        q1, q2, q3 = st.columns(3)
                        psnr = metrics.get('psnr', 0.0)
                        ssim = metrics.get('ssim', 0.0)
                        drift = metrics.get('embedding_drift', 0.0)

                        q1.metric("PSNR", f"{psnr:.1f} dB", delta="OK" if psnr >= 40 else "LOW")
                        q2.metric("SSIM", f"{ssim:.4f}", delta="OK" if ssim >= 0.93 else "LOW")
                        q3.metric("Emb. Drift", f"{drift:.4f}", delta="OK" if drift <= 0.08 else "HIGH")

                        # Protection layers
                        st.markdown("#### Protection Layers")
                        leat_data = metrics.get('leat_metrics')
                        hash_ver = metrics.get('hash_verification', {})
                        all_hashes_ok = all(v.get('match', False) for v in hash_ver.values()) if hash_ver else False
                        c2pa_data = metrics.get('c2pa_manifest')

                        p1, p2, p3 = st.columns(3)
                        p1.checkbox("LEAT Applied", value=leat_data is not None, disabled=True)
                        p2.checkbox("Hash Verified", value=all_hashes_ok, disabled=True)
                        p3.checkbox("C2PA Embedded", value=c2pa_data is not None, disabled=True)

                        w1, w2, w3 = st.columns(3)
                        w1.checkbox("Spectral Watermark", value=True, disabled=True)
                        w2.checkbox("Multi-band Watermark", value=True, disabled=True)
                        w3.checkbox("Stable Signature", value=True, disabled=True)

                        # Content analysis
                        content_info = metrics.get('content_analysis', {})
                        st.markdown("#### Content Analysis")
                        ca1, ca2 = st.columns(2)
                        ca1.metric("Risk Level", content_info.get('risk_level', 'N/A'))
                        ca2.metric("Faces Detected", content_info.get('face_count', 0))

                        # LEAT per-encoder metrics
                        if leat_data:
                            st.markdown("#### LEAT Encoder Disruption")
                            lcols = st.columns(min(len(leat_data), 7))
                            for i, (enc_name, enc_met) in enumerate(leat_data.items()):
                                if i < len(lcols):
                                    dist = enc_met.get('latent_cosine_distance', 0.0) if isinstance(enc_met, dict) else 0.0
                                    lcols[i].metric(enc_name, f"{dist:.4f}")

                        st.divider()

                        # Protection logs
                        with st.expander("View Protection Logs"):
                            st.code(_protection_logs, language="text")

                        # Preview + download
                        actual_out = out_path_result if out_path_result else out_path
                        if os.path.exists(actual_out):
                            st.image(actual_out, use_container_width=True, caption="Protected Output")
                            with open(actual_out, "rb") as f:
                                st.download_button(
                                    label="Download Protected Image",
                                    data=f,
                                    file_name=f"protected_{prot_file.name}",
                                    mime="image/png"
                                )

# =====================================================================
# TAB 3: C2PA INSPECTOR
# =====================================================================
with tab3:
    st.header("C2PA Provenance Inspector")
    st.markdown("Verify the authenticity chain and rights assertions of a protected media file.")

    c2pa_file = st.file_uploader("Upload media to inspect", type=['png', 'jpg', 'jpeg'], key="c2pa_upload")

    if c2pa_file and BACKEND_READY:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(c2pa_file.name)[1]) as tmp:
            tmp.write(c2pa_file.getvalue())
            c_path = tmp.name

        with st.spinner("Extracting Manifest..."):
            verifier = C2PAMetadataBinding()
            report = verifier.verify_protection(c_path)

            if report.get('has_manifest'):
                st.success("C2PA Manifest Detected")

                c1, c2, c3 = st.columns(3)
                c1.checkbox("Manifest Presence", value=report['has_manifest'], disabled=True)
                c2.checkbox("No-AI Rights Asserted", value=report['has_rights'], disabled=True)
                c3.checkbox("Signature Validated", value=report['has_signature'], disabled=True)

                if report['verified']:
                    st.info("This asset is cryptographically bound to PikSign protection.")

                st.divider()
                st.markdown("#### Manifest Details")
                manifest_data = verifier.extract_manifest(c_path)
                if manifest_data:
                    # Show key fields
                    m1, m2 = st.columns(2)
                    m1.metric("Creator", manifest_data.get('creator', 'N/A'))
                    m2.metric("Timestamp", manifest_data.get('timestamp', 'N/A'))

                    rights = manifest_data.get('rights_assertions', [])
                    if rights:
                        st.markdown("**Rights Assertions:**")
                        for r in rights:
                            label = r.get('label', r) if isinstance(r, dict) else r
                            st.write(f"- {label}")

                    sig = manifest_data.get('signature', {})
                    if sig:
                        s1, s2 = st.columns(2)
                        s1.metric("Algorithm", sig.get('algorithm', 'N/A'))
                        s2.metric("Hash Match", str(sig.get('hash_verified', 'N/A')))

                with st.expander("View Raw Manifest JSON"):
                    st.json(manifest_data if manifest_data else report)
            else:
                st.warning("No C2PA manifest found in this image.")

        os.unlink(c_path)
