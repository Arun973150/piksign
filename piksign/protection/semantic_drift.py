# -*- coding: utf-8 -*-
"""
PikSign Semantic Drift Controller
Measures embedding drift between original and protected images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models


class SemanticDriftController:
    """Controls and measures semantic drift in protected images."""
    
    def __init__(self, device: torch.device, target_drift: float = 0.08):
        self.device = device
        self.target_drift = target_drift
        
        # Load ResNet50 for embedding computation
        try:
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            ).to(device)
        except:
            self.model = models.resnet50(pretrained=True).to(device)
        
        self.model.eval()
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def compute_embedding_drift(self, original_tensor: torch.Tensor, 
                                 protected_tensor: torch.Tensor) -> float:
        """
        Compute embedding drift between original and protected images.
        
        Args:
            original_tensor: Original image tensor
            protected_tensor: Protected image tensor
            
        Returns:
            float: Drift value (0 = identical, 1 = maximally different)
        """
        with torch.no_grad():
            original_tensor = original_tensor.to(self.device)
            protected_tensor = protected_tensor.to(self.device)
            
            orig_norm = self.transform(original_tensor)
            prot_norm = self.transform(protected_tensor)
            
            orig_feat = self.model(orig_norm).view(orig_norm.size(0), -1)
            prot_feat = self.model(prot_norm).view(prot_norm.size(0), -1)
            
            cos_sim = F.cosine_similarity(orig_feat, prot_feat, dim=1)
            drift = 1 - cos_sim.mean().item()
        
        return drift
    
    def is_drift_acceptable(self, drift: float) -> bool:
        """Check if drift is within acceptable range."""
        return drift <= self.target_drift * 2  # Allow some tolerance
