# -*- coding: utf-8 -*-
"""
PikSign - AI Security System for Visual Media

Phase 1: Protection (Shield) - Make images AI-unreadable
Phase 2: Detection (Visual Intelligence) - Detect AI-generated/manipulated content
"""

__version__ = "2.0.0"
__author__ = "PikSign Team"

# Protection module
from piksign.protection import PikSignShield

# Detection module
from piksign.detection import PikSignDetector

from piksign.protection.config import Config

__all__ = ['PikSignShield', 'PikSignDetector', 'Config']
