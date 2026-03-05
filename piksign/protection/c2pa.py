# -*- coding: utf-8 -*-
"""
PikSign C2PA Metadata Binding
Coalition for Content Provenance and Authenticity metadata support.
"""

import json
import hashlib
import base64
import uuid
import hmac
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any

# Try to import cryptography for signing
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# Try to import piexif for EXIF embedding
try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False

# Try PIL for XMP
try:
    from PIL import Image, PngImagePlugin
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class C2PAMetadataBinding:
    """
    C2PA Metadata Binding for image protection.
    
    Features:
    - "No AI Training" rights assertion
    - Origin signature with cryptographic proof
    - Protection hash binding
    - Tamper-evident manifest
    """
    
    # C2PA Action Types
    ACTION_CREATED = "c2pa.created"
    ACTION_EDITED = "c2pa.edited"
    ACTION_PROTECTED = "c2pa.protected"
    
    # Rights Assertions
    RIGHT_NO_AI_TRAINING = "c2pa.ai_training"
    RIGHT_NO_AI_GENERATIVE = "c2pa.ai_generative_training"
    RIGHT_NO_DATA_MINING = "c2pa.data_mining"
    
    def __init__(self,
                 creator_name: str = "PikSign Shield",
                 creator_url: str = "https://piksign.protection",
                 enable_no_ai: bool = True,
                 enable_no_mining: bool = True):
        self.creator_name = creator_name
        self.creator_url = creator_url
        self.enable_no_ai = enable_no_ai
        self.enable_no_mining = enable_no_mining
        
        self.private_key = None
        self.public_key = None
        self._initialize_keys()
        
        self.manifests = {}
    
    def _initialize_keys(self):
        """Initialize RSA key pair for signing."""
        if HAS_CRYPTO:
            try:
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
                self.public_key = self.private_key.public_key()
            except:
                self.private_key = None
                self.public_key = None
    
    def _compute_content_hash(self, image_data: bytes) -> str:
        """Compute SHA-256 hash of image content."""
        return hashlib.sha256(image_data).hexdigest()
    
    def _compute_protection_hash(self, original_hash: str, 
                                  protected_hash: str, 
                                  fingerprint: str) -> str:
        """Compute combined protection hash."""
        combined = f"{original_hash}|{protected_hash}|{fingerprint}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _sign_data(self, data: bytes) -> str:
        """Sign data with private key."""
        if HAS_CRYPTO and self.private_key:
            try:
                signature = self.private_key.sign(
                    data,
                    padding.PKCS1v15(),
                    hashes.SHA256()
                )
                return base64.b64encode(signature).decode('utf-8')
            except:
                pass
        
        # Fallback: HMAC-based signature
        secret = b"PikSign2026SecretKey"
        sig = hmac.new(secret, data, hashlib.sha256).hexdigest()
        return sig
    
    def _get_public_key_pem(self) -> Optional[str]:
        """Get public key in PEM format."""
        if HAS_CRYPTO and self.public_key:
            try:
                pem = self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                return pem.decode('utf-8')
            except:
                pass
        return None
    
    def create_rights_assertion(self) -> Dict[str, Any]:
        """Create rights assertion with No AI clause."""
        assertions = []
        
        if self.enable_no_ai:
            assertions.append({
                "label": self.RIGHT_NO_AI_TRAINING,
                "data": {
                    "use": "notAllowed",
                    "constraint_info": "This image is protected and may NOT be used for AI training."
                }
            })
            assertions.append({
                "label": self.RIGHT_NO_AI_GENERATIVE,
                "data": {
                    "use": "notAllowed",
                    "constraint_info": "Use in generative AI systems is prohibited."
                }
            })
        
        if self.enable_no_mining:
            assertions.append({
                "label": self.RIGHT_NO_DATA_MINING,
                "data": {
                    "use": "notAllowed",
                    "constraint_info": "Automated data collection and mining is prohibited."
                }
            })
        
        return {
            "dc:rights": "All rights reserved. No AI training permitted.",
            "assertions": assertions
        }
    
    def create_origin_signature(self, image_hash: str, 
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create cryptographic origin signature."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        sign_data = {
            "image_hash": image_hash,
            "timestamp": timestamp,
            "creator": self.creator_name,
            "metadata_hash": hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
        }
        
        signature = self._sign_data(
            json.dumps(sign_data, sort_keys=True).encode()
        )
        
        return {
            "signature_type": "c2pa.signature",
            "algorithm": "RS256" if HAS_CRYPTO else "HS256",
            "timestamp": timestamp,
            "signed_data": sign_data,
            "signature": signature,
            "public_key": self._get_public_key_pem()
        }
    
    def create_manifest(self, original_path: str, protected_path: str,
                        protection_metrics: Dict[str, Any],
                        fingerprint: Dict[str, str]) -> Dict[str, Any]:
        """Create C2PA-compatible manifest."""
        manifest_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Read image data for hashing
        with open(original_path, 'rb') as f:
            original_data = f.read()
        original_hash = self._compute_content_hash(original_data)
        
        if os.path.exists(protected_path):
            with open(protected_path, 'rb') as f:
                protected_data = f.read()
            protected_hash = self._compute_content_hash(protected_data)
        else:
            protected_hash = "pending"
        
        protection_hash = self._compute_protection_hash(
            original_hash,
            protected_hash,
            fingerprint.get('composite_hash', '')
        )
        
        manifest = {
            "c2pa_manifest": {
                "claim_generator": f"{self.creator_name}/1.0",
                "claim_generator_info": [{
                    "name": self.creator_name,
                    "version": "2.0",
                    "url": self.creator_url
                }],
                "dc:title": os.path.basename(protected_path),
                "dc:format": "image/png",
                "instance_id": f"xmp:iid:{manifest_id}",
                "claim_generator_id": "PikSign.Shield.v2",
                "created": timestamp
            },
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [{
                            "action": self.ACTION_PROTECTED,
                            "when": timestamp,
                            "softwareAgent": self.creator_name,
                            "parameters": {
                                "protection_type": "anti-ai-watermark",
                                "quality_psnr": protection_metrics.get('psnr', 0),
                                "quality_ssim": protection_metrics.get('ssim', 0)
                            }
                        }]
                    }
                },
                self.create_rights_assertion(),
                {
                    "label": "c2pa.hash.data",
                    "data": {
                        "original_hash": original_hash,
                        "protected_hash": protected_hash,
                        "protection_binding_hash": protection_hash,
                        "algorithm": "SHA-256",
                        "perceptual_fingerprint": fingerprint.get('composite_hash', '')
                    }
                },
                {
                    "label": "piksign.protection",
                    "data": {
                        "version": "2.0",
                        "protection_level": "enhanced",
                        "watermarks": ["spectral", "multiband_frequency", "stable_signature"],
                        "metrics": protection_metrics
                    }
                }
            ],
            "signature": self.create_origin_signature(
                protected_hash,
                {"manifest_id": manifest_id, "timestamp": timestamp}
            ),
            "verification": {
                "method": "hash_comparison",
                "protection_hash": protection_hash,
                "verify_url": f"{self.creator_url}/verify/{manifest_id}"
            }
        }
        
        self.manifests[manifest_id] = manifest
        return manifest
    
    def embed_in_image(self, image_path: str, manifest: Dict[str, Any],
                       output_path: Optional[str] = None) -> str:
        """Embed C2PA manifest in image metadata."""
        if output_path is None:
            output_path = image_path
        
        manifest_json = json.dumps(manifest, indent=2)
        
        if HAS_PIL:
            try:
                img = Image.open(image_path)
                is_png = output_path.lower().endswith('.png')
                
                if is_png:
                    # PNG: embed in text chunks
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("C2PA_Manifest", manifest_json)
                    pnginfo.add_text("C2PA_Rights", "No AI Training Permitted")
                    pnginfo.add_text("C2PA_Hash", manifest['verification']['protection_hash'])
                    pnginfo.add_text("PikSign_Protected", "true")
                    pnginfo.add_text("PikSign_Version", "2.0")
                    img.save(output_path, pnginfo=pnginfo)
                else:
                    # JPEG: embed in EXIF UserComment
                    if HAS_PIEXIF:
                        try:
                            exif_dict = {"0th": {}, "Exif": {}, "1st": {}}
                            # Try to load existing EXIF first
                            try:
                                existing = piexif.load(img.info.get('exif', b''))
                                exif_dict.update(existing)
                            except Exception:
                                pass
                            # Embed truncated manifest in UserComment (max ~65535 bytes)
                            comment = manifest_json[:60000].encode('utf-8')
                            user_comment = b'UNICODE\x00' + comment
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                            # Add PikSign marker in ImageDescription
                            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = b'PikSign_Protected:true'
                            exif_bytes = piexif.dump(exif_dict)
                            img.save(output_path, exif=exif_bytes)
                        except Exception as ex:
                            print(f"      [!] EXIF embed failed: {ex}, saving without metadata")
                            img.save(output_path)
                    else:
                        # No piexif: save without EXIF but print warning
                        img.save(output_path)
                        print("      [!] piexif not installed -- C2PA skipped for JPEG. Install: pip install piexif")
                
                return output_path
            except Exception as e:
                print(f"[!] Metadata embedding failed: {e}")
        
        return output_path
    
    def extract_manifest(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract C2PA manifest from image (PNG text chunks or JPEG EXIF)."""
        if not HAS_PIL:
            return None
        try:
            img = Image.open(image_path)
            # 1. PNG text chunks
            if hasattr(img, 'text') and 'C2PA_Manifest' in img.text:
                return json.loads(img.text['C2PA_Manifest'])
            # 2. JPEG EXIF UserComment
            if HAS_PIEXIF and image_path.lower().endswith(('.jpg', '.jpeg')):
                raw_exif = img.info.get('exif')
                if raw_exif:
                    exif_dict = piexif.load(raw_exif)
                    user_comment = exif_dict.get('Exif', {}).get(piexif.ExifIFD.UserComment, b'')
                    if user_comment:
                        # Strip UNICODE header if present
                        comment_bytes = user_comment[8:] if user_comment.startswith(b'UNICODE\x00') else user_comment
                        return json.loads(comment_bytes.decode('utf-8', errors='replace'))
        except Exception:
            pass
        return None
    
    def verify_protection(self, image_path: str,
                          expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """Verify protection integrity."""
        result = {
            "verified": False,
            "has_manifest": False,
            "has_rights": False,
            "has_signature": False,
            "hash_match": None,
            "details": {}
        }
        
        manifest = self.extract_manifest(image_path)
        
        if manifest:
            result["has_manifest"] = True
            result["details"]["manifest_id"] = manifest.get(
                "c2pa_manifest", {}
            ).get("instance_id")
            
            assertions = manifest.get("assertions", [])
            for assertion in assertions:
                if not isinstance(assertion, dict):
                    continue
                label = assertion.get("label", "")
                # Direct label match (e.g. c2pa.actions, c2pa.hash.data)
                if label == self.RIGHT_NO_AI_TRAINING:
                    result["has_rights"] = True
                # Rights assertion: stored as {"dc:rights": ..., "assertions": [...]}
                # -- no top-level "label" key, check for dc:rights key instead
                if "dc:rights" in assertion:
                    result["has_rights"] = True
                # Also scan nested assertions list inside a rights dict
                for sub in assertion.get("assertions", []):
                    if isinstance(sub, dict) and sub.get("label") == self.RIGHT_NO_AI_TRAINING:
                        result["has_rights"] = True
            
            if "signature" in manifest:
                result["has_signature"] = True
                result["details"]["signature_time"] = manifest["signature"].get("timestamp")
            
            if expected_hash:
                actual_hash = manifest.get("verification", {}).get("protection_hash")
                result["hash_match"] = actual_hash == expected_hash
            
            result["verified"] = (
                result["has_manifest"] and
                result["has_rights"] and
                result["has_signature"]
            )
        
        return result
