# -*- coding: utf-8 -*-
"""
PikSign Adaptive Transform Engine
Computes adaptive parameters based on content analysis.
"""

from piksign.protection.config import Config


class AdaptiveTransformEngine:
    """Computes adaptive protection parameters based on content analysis."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def compute_adaptive_epsilon(self, content_analysis: dict) -> float:
        """
        Compute adaptive epsilon based on content vulnerability.
        
        Args:
            content_analysis: Results from ContentAnalyzer
            
        Returns:
            float: Epsilon value for perturbation
        """
        base_epsilon = self.config.EPSILON_BASE
        max_epsilon = self.config.EPSILON_MAX
        
        vulnerability = content_analysis['clip_vulnerability']
        
        if vulnerability > 0.7:
            epsilon = max_epsilon
        elif vulnerability > 0.5:
            epsilon = base_epsilon + (max_epsilon - base_epsilon) * 0.6
        else:
            epsilon = base_epsilon

        # Faces are the primary deepfake target — use more budget, not less
        if content_analysis['has_faces']:
            epsilon = min(epsilon * 1.2, max_epsilon)
        
        return epsilon
    
    def compute_frequency_weights(self, content_analysis: dict) -> tuple:
        """
        Compute frequency weights based on texture complexity.
        
        Returns:
            tuple: (low_freq_weight, mid_freq_weight, high_freq_weight)
        """
        texture = content_analysis['texture_complexity']
        
        # Reduced weights for better quality
        low_freq_weight = 0.05 + texture * 0.03
        mid_freq_weight = 0.08
        high_freq_weight = 0.03 + texture * 0.05
        
        return low_freq_weight, mid_freq_weight, high_freq_weight
