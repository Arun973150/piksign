# -*- coding: utf-8 -*-
"""
PikSign Protection Module (Phase 1 - Shield)
Makes images and videos human-viewable but AI-unreadable.
"""

from piksign.protection.shield import PikSignShield
from piksign.protection.config import Config
from piksign.protection.leat_attack import LEATAttack
from piksign.protection.deepfake_encoders import (
    ArcFaceEncoder, StyleGANEncoder, DiffAEEncoder,
    ICfaceEncoder, VGGPerceptualEncoder,
)

__all__ = [
    'PikSignShield', 'Config', 'LEATAttack',
    'ArcFaceEncoder', 'StyleGANEncoder', 'DiffAEEncoder',
    'ICfaceEncoder', 'VGGPerceptualEncoder',
]
