# -*- coding: utf-8 -*-
"""
PikSign Visualization
Visualization utilities for comparing original and protected images.
"""

import matplotlib.pyplot as plt
from PIL import Image


def visualize_results(original_path: str, protected_path: str,
                      save_path: str = 'comparison.png',
                      protected_only: bool = False) -> str:
    """
    Visualize results (Protected only or Original vs Protected).
    
    Args:
        original_path: Path to original image
        protected_path: Path to protected image
        save_path: Path to save result
        protected_only: If True, only visualize the protected image
        
    Returns:
        str: Path to saved image
    """
    if protected_only:
        protected = Image.open(protected_path).convert('RGB')
        
        plt.figure(figsize=(8, 8))
        plt.imshow(protected)
        plt.title('PikSign Protected (Ready for Sharing)', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Protected visualization saved to {save_path}")
        return save_path
        
    original = Image.open(original_path).convert('RGB')
    protected = Image.open(protected_path).convert('RGB')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(protected)
    axes[1].set_title('Protected (Quality Optimized)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Comparison saved to {save_path}")
    return save_path


def visualize_metrics(metrics: dict, save_path: str = 'metrics.png') -> str:
    """
    Visualize protection metrics as a chart.
    
    Args:
        metrics: Protection metrics dictionary
        save_path: Path to save metrics visualization
        
    Returns:
        str: Path to saved visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Quality metrics
    quality_labels = ['PSNR (dB)', 'SSIM x 100']
    quality_values = [metrics.get('psnr', 0), metrics.get('ssim', 0) * 100]
    quality_targets = [40.0, 93.0]
    
    x = range(len(quality_labels))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], quality_values, width, 
                label='Achieved', color='#4CAF50')
    axes[0].bar([i + width/2 for i in x], quality_targets, width, 
                label='Target', color='#2196F3', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(quality_labels)
    axes[0].set_title('Quality Metrics', fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, max(max(quality_values), max(quality_targets)) * 1.2)
    
    # Drift metric
    drift = metrics.get('embedding_drift', 0)
    axes[1].bar(['Embedding Drift'], [drift * 100], color='#FF9800')
    axes[1].axhline(y=8, color='red', linestyle='--', label='Target (8%)')
    axes[1].set_ylabel('Drift (%)')
    axes[1].set_title('Semantic Drift', fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(0, 20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path
