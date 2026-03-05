# -*- coding: utf-8 -*-
"""
PikSign Content Analyzer
Analyzes image content for faces, textures, and AI vulnerability.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


class ContentAnalyzer:
    """Analyzes image content for protection optimization."""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Load ResNet50 for feature extraction
        try:
            self.feature_model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            ).to(device)
        except:
            self.feature_model = models.resnet50(pretrained=True).to(device)
        
        self.feature_model.eval()
        self.feature_model = nn.Sequential(*list(self.feature_model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def detect_faces(self, img_array: np.ndarray) -> tuple:
        """
        Detect faces in the image.
        
        Returns:
            tuple: (has_faces, face_count)
        """
        try:
            gray = cv2.cvtColor(
                (img_array * 255).astype(np.uint8), 
                cv2.COLOR_RGB2GRAY
            )
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0, len(faces)
        except:
            return False, 0
    
    def compute_clip_vulnerability(self, img_tensor: torch.Tensor) -> float:
        """
        Compute vulnerability score based on feature extraction.
        Higher score = more vulnerable to AI interpretation.
        """
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            img_norm = self.transform(img_tensor)
            features = self.feature_model(img_norm)
            features = features.view(features.size(0), -1)
            feature_norm = torch.norm(features, dim=1)
            vulnerability = torch.sigmoid(feature_norm / 100.0)
        return vulnerability.item()
    
    def analyze_texture(self, img_array: np.ndarray) -> float:
        """
        Analyze texture complexity using Laplacian variance.
        
        Returns:
            float: Texture score (0-1)
        """
        try:
            gray = cv2.cvtColor(
                (img_array * 255).astype(np.uint8), 
                cv2.COLOR_RGB2GRAY
            )
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_score = np.var(laplacian) / 1000.0
            return min(texture_score, 1.0)
        except:
            return 0.5
    
    def analyze(self, img_tensor: torch.Tensor, img_array: np.ndarray) -> dict:
        """
        Perform complete content analysis.
        
        Returns:
            dict: Analysis results including faces, vulnerability, texture, risk level
        """
        has_faces, face_count = self.detect_faces(img_array)
        vulnerability = self.compute_clip_vulnerability(img_tensor)
        texture = self.analyze_texture(img_array)
        
        # Determine risk level
        if has_faces or vulnerability > 0.7:
            risk_level = 'HIGH'
        elif vulnerability > 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'has_faces': has_faces,
            'face_count': face_count,
            'clip_vulnerability': vulnerability,
            'texture_complexity': texture,
            'risk_level': risk_level
        }
