# -*- coding: utf-8 -*-
"""
PikSign EXIF Validator

Validates EXIF metadata consistency to identify real vs AI images.

PURPOSE:
- Check for valid camera metadata that AI generators typically don't produce
- Detect red flags: missing EXIF, impossible settings, AI tool signatures
- Cross-validate EXIF with image properties

RED FLAGS:
- No EXIF at all -> likely AI or scrubbed
- Generic camera model without lens data
- Impossible settings (ISO 100 in pitch darkness)
- Known AI generator tags (Photoshop Beta = Firefly)
"""

import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Try to import piexif for more detailed EXIF parsing
try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


# Known AI generator software signatures
AI_SOFTWARE_SIGNATURES = [
    'midjourney',
    'dall-e',
    'dalle',
    'stable diffusion',
    'stability ai',
    'firefly',
    'photoshop beta',
    'adobe firefly',
    'leonardo.ai',
    'playground ai',
    'getimg.ai',
    'imagen',
    'dreamstudio',
    'invokeai',
    'automatic1111',
    'comfyui',
    'flux',
    'ideogram',
    'nightcafe',
    'artbreeder'
]

# Known camera manufacturers
KNOWN_CAMERA_MAKERS = [
    'canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
    'leica', 'pentax', 'ricoh', 'sigma', 'hasselblad', 'phase one',
    'apple', 'samsung', 'google', 'huawei', 'xiaomi', 'oneplus',
    'oppo', 'vivo', 'motorola', 'lg', 'htc', 'nokia', 'zte'
]


class EXIFValidator:
    """
    EXIF Metadata Validator for detecting real vs AI images.
    
    Checks:
    1. Presence of EXIF data
    2. Valid camera make/model
    3. Reasonable camera settings
    4. Known AI tool signatures
    5. Timestamp consistency
    6. GPS data presence (strong indicator of real photo)
    """
    
    def __init__(self):
        self.has_piexif = HAS_PIEXIF
    
    def validate(self, image_path: str) -> Dict[str, Any]:
        """
        Validate EXIF metadata of an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Validation results with real_probability and flags
        """
        try:
            results = {
                'status': 'success',
                'has_exif': False,
                'has_camera': False,
                'has_gps': False,
                'ai_signature_found': False,
                'impossible_settings': False,
                'real_indicators': 0,
                'ai_indicators': 0,
                'exif_data': {},
                'flags': []
            }
            
            # Load image and extract EXIF
            pil_image = Image.open(image_path)
            exif_data = self._extract_exif(pil_image, image_path)
            results['exif_data'] = exif_data
            
            # Check 1: EXIF presence
            if not exif_data or len(exif_data) < 3:
                results['flags'].append('NO_EXIF_OR_MINIMAL')
                results['ai_indicators'] += 1
            else:
                results['has_exif'] = True
                results['real_indicators'] += 1
            
            # Check 2: Camera make/model
            camera_score, camera_info = self._check_camera_info(exif_data)
            if camera_score > 0.7:
                results['has_camera'] = True
                results['real_indicators'] += 1
                results['camera_info'] = camera_info
            elif camera_score < 0.3:
                results['flags'].append('NO_VALID_CAMERA')
                results['ai_indicators'] += 1
            
            # Check 3: GPS data
            if self._has_gps_data(exif_data):
                results['has_gps'] = True
                results['real_indicators'] += 2  # GPS is a strong indicator
                
            # Check 4: AI software signatures
            ai_sig, ai_tool = self._check_ai_signatures(exif_data)
            if ai_sig:
                results['ai_signature_found'] = True
                results['ai_tool_detected'] = ai_tool
                results['flags'].append(f'AI_TOOL_DETECTED: {ai_tool}')
                results['ai_indicators'] += 3  # Strong AI indicator
            
            # Check 5: Camera settings validity
            settings_valid, settings_issue = self._check_camera_settings(exif_data)
            if not settings_valid:
                results['impossible_settings'] = True
                results['flags'].append(f'IMPOSSIBLE_SETTINGS: {settings_issue}')
                results['ai_indicators'] += 1
            elif settings_valid and exif_data.get('ExposureTime') or exif_data.get('FNumber'):
                results['real_indicators'] += 1  # Valid camera settings
            
            # Check 6: Timestamp consistency
            timestamp_ok = self._check_timestamp_consistency(image_path, exif_data)
            if not timestamp_ok:
                results['flags'].append('TIMESTAMP_MISMATCH')
            
            # Compute real image probability
            total = results['real_indicators'] + results['ai_indicators']
            if total > 0:
                real_prob = results['real_indicators'] / total
            else:
                real_prob = 0.3  # Default uncertain
            
            results['real_probability'] = _safe_prob(real_prob)
            
            # Recommendation
            if results['ai_signature_found']:
                results['recommendation'] = 'likely_ai_generated'
            elif results['real_indicators'] >= 3:
                results['recommendation'] = 'likely_real_photo'
            elif results['ai_indicators'] >= 2:
                results['recommendation'] = 'possibly_ai_or_edited'
            else:
                results['recommendation'] = 'uncertain'
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'has_exif': False,
                'real_probability': 0.5,
                'recommendation': 'uncertain'
            }
    
    def _extract_exif(self, pil_image: Image.Image, image_path: str) -> Dict[str, Any]:
        """Extract EXIF data from image."""
        exif_data = {}
        
        # Try PIL first
        try:
            raw_exif = pil_image._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)[:100]
                    exif_data[tag] = value
        except:
            pass
        
        # Try piexif for more details
        if HAS_PIEXIF:
            try:
                piexif_data = piexif.load(image_path)
                for ifd in ('0th', 'Exif', 'GPS', '1st'):
                    if ifd in piexif_data and piexif_data[ifd]:
                        for tag, value in piexif_data[ifd].items():
                            tag_name = piexif.TAGS.get(ifd, {}).get(tag, {}).get('name', f'{ifd}_{tag}')
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='ignore').strip('\x00')
                                except:
                                    continue
                            exif_data[tag_name] = value
            except:
                pass
        
        # Also check XMP metadata in image info
        if hasattr(pil_image, 'info'):
            for key, value in pil_image.info.items():
                if key not in exif_data:
                    if isinstance(value, (str, int, float)):
                        exif_data[f'Info_{key}'] = value
        
        return exif_data
    
    def _check_camera_info(self, exif_data: Dict) -> Tuple[float, Dict]:
        """Check for valid camera make/model."""
        camera_info = {}
        
        # Look for camera make
        make = exif_data.get('Make', '') or exif_data.get('ImageMake', '') or ''
        make = str(make).lower().strip()
        
        model = exif_data.get('Model', '') or exif_data.get('ImageModel', '') or ''
        model = str(model).lower().strip()
        
        camera_info['make'] = make
        camera_info['model'] = model
        
        # Check if make is a known camera manufacturer
        is_known_maker = any(maker in make for maker in KNOWN_CAMERA_MAKERS)
        
        score = 0.0
        if is_known_maker:
            score += 0.5
            camera_info['known_maker'] = True
        
        if model and len(model) > 3:
            score += 0.3
            
        # Check for lens info
        lens = exif_data.get('LensModel', '') or exif_data.get('Lens', '')
        if lens:
            score += 0.2
            camera_info['lens'] = str(lens)
        
        return (score, camera_info)
    
    def _has_gps_data(self, exif_data: Dict) -> bool:
        """Check for GPS coordinates."""
        gps_keys = ['GPSLatitude', 'GPSLongitude', 'GPSInfo', 'GPSLatitudeRef']
        return any(key in exif_data for key in gps_keys)
    
    def _check_ai_signatures(self, exif_data: Dict) -> Tuple[bool, Optional[str]]:
        """Check for known AI tool signatures in metadata."""
        # Fields to check
        fields_to_check = ['Software', 'ProcessingSoftware', 'Creator', 
                          'CreatorTool', 'XMPToolkit', 'Info_Software',
                          'ImageDescription', 'UserComment']
        
        for field in fields_to_check:
            value = exif_data.get(field, '')
            if value:
                value_lower = str(value).lower()
                for ai_sig in AI_SOFTWARE_SIGNATURES:
                    if ai_sig in value_lower:
                        return (True, ai_sig)
        
        # Check for "AI Generated" markers
        for key, value in exif_data.items():
            if isinstance(value, str):
                value_lower = value.lower()
                if 'ai generated' in value_lower or 'ai-generated' in value_lower:
                    return (True, 'ai_generated_tag')
        
        return (False, None)
    
    def _check_camera_settings(self, exif_data: Dict) -> Tuple[bool, Optional[str]]:
        """Check if camera settings are physically possible."""
        issues = []
        
        # Get settings
        iso = exif_data.get('ISOSpeedRatings', None)
        exposure = exif_data.get('ExposureTime', None)
        aperture = exif_data.get('FNumber', None)
        focal_length = exif_data.get('FocalLength', None)
        
        # Check ISO (typically 50-409600)
        if iso is not None:
            try:
                iso_val = int(iso) if not isinstance(iso, tuple) else int(iso[0])
                if iso_val < 25 or iso_val > 500000:
                    issues.append(f'Invalid ISO: {iso_val}')
            except:
                pass
        
        # Check aperture (typically f/0.95 to f/64)
        if aperture is not None:
            try:
                if isinstance(aperture, tuple):
                    f_num = aperture[0] / aperture[1] if len(aperture) > 1 else aperture[0]
                else:
                    f_num = float(aperture)
                if f_num < 0.7 or f_num > 128:
                    issues.append(f'Invalid aperture: f/{f_num}')
            except:
                pass
        
        # Check focal length (typically 4mm to 2000mm)
        if focal_length is not None:
            try:
                if isinstance(focal_length, tuple):
                    fl = focal_length[0] / focal_length[1] if len(focal_length) > 1 else focal_length[0]
                else:
                    fl = float(focal_length)
                if fl < 1 or fl > 3000:
                    issues.append(f'Invalid focal length: {fl}mm')
            except:
                pass
        
        if issues:
            return (False, '; '.join(issues))
        return (True, None)
    
    def _check_timestamp_consistency(self, image_path: str, exif_data: Dict) -> bool:
        """Check if EXIF timestamp is consistent with file creation."""
        try:
            # Get file modification time
            file_mtime = os.path.getmtime(image_path)
            file_date = datetime.fromtimestamp(file_mtime)
            
            # Get EXIF datetime
            exif_datetime = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')
            if not exif_datetime:
                return True  # No timestamp to compare
            
            # Parse EXIF datetime
            try:
                exif_date = datetime.strptime(str(exif_datetime), '%Y:%m:%d %H:%M:%S')
            except:
                return True  # Can't parse, skip check
            
            # Allow up to 10 years difference (photos can be old)
            diff_days = abs((file_date - exif_date).days)
            return diff_days < 3650  # ~10 years
            
        except:
            return True


# Convenience function
def validate_exif(image_path: str) -> Dict[str, Any]:
    """
    Validate EXIF metadata of an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Validation results
    """
    validator = EXIFValidator()
    return validator.validate(image_path)
