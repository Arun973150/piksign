# -*- coding: utf-8 -*-
"""
PikSign LEAT - Latent Ensemble Attack for Deepfake Disruption

Implements the LEAT algorithm from:
"LEAT: Towards Robust Deepfake Disruption in Real-World Scenarios
 via Latent Ensemble Attack" (Shim & Yoon, 2023)

Extended with:
- Weighted encoder ensemble (VAE encoders get higher weight for diffusion model coverage)
- Frequency-domain attack (targets high-frequency patterns that VAE encoders compress)
- Multiple VAE variants (SD 1.5 + SDXL) for better black-box transfer

Attacks the actual latent encoders from deepfake and diffusion models:
- ArcFace IResNet-50: identity encoder (SimSwap -- face swapping)
- pSp/e4e GradualStyleEncoder: W+ encoder (StyleCLIP -- face attribute manipulation)
- DiffAE Semantic Encoder: 512-d encoder (DiffAE -- diffusion-based manipulation)
- ICface Neutral Generator: neutral face (ICface -- face reenactment)
- VGG perceptual encoder: supplementary (deepfake training losses)
- SD VAE encoder: latent bottleneck (Stable Diffusion -- all diffusion editors)
- SDXL VAE encoder: latent bottleneck (SDXL -- newer diffusion editors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from piksign.protection.deepfake_encoders import (
    ArcFaceEncoder,
    StyleGANEncoder,
    DiffAEEncoder,
    ICfaceEncoder,
    VGGPerceptualEncoder,
    SDVAEEncoder,
)

# Default encoder weights for the ensemble.
# VAE encoders get 3x weight to focus the attack budget on the latent
# compression bottleneck that all diffusion-based editors rely on.
DEFAULT_ENCODER_WEIGHTS = {
    'arcface': 1.0,
    'stylegan': 1.0,
    'diffae': 1.0,
    'icface': 1.0,
    'vgg': 0.5,
    'sdvae': 3.0,
    'sdxl_vae': 3.0,
}


class LEATAttack:
    """
    Latent Ensemble Attack (LEAT) with Weighted Normalized Gradient Ensemble.

    Extended version of the LEAT algorithm with:
    - Weighted encoder ensemble: VAE encoders get higher weight (3x) to focus
      the attack budget on the latent compression bottleneck
    - Frequency-domain attack: adds high-frequency perturbation component that
      targets patterns VAE encoders are sensitive to during compression
    - Multiple VAE variants: SD 1.5 + SDXL VAE for better black-box transfer
      to proprietary diffusion models (Google Nano Banana, Imagen, etc.)

    Algorithm 1 from the paper (extended with weights + frequency attack):
        Input: X (image), E1..EK (encoders), w1..wK (weights),
               T (iterations), a (step), eps (bound), freq_weight
        1: Random init eta, X0 = X + eta
        2: for t in T:
        3:   G_normgrad = 0
        4:   for k in K:
        5:     grad = nabla_Xt L(Ek(Xt), Ek(Xt + eta))
        6:     grad = grad / ||grad||_2
        7:     G_normgrad += wk * grad          # WEIGHTED aggregation
        8:   G_freq = frequency_gradient(Xt)     # frequency-domain attack
        9:   G_total = G_normgrad + freq_weight * G_freq
        10:  X't = Xt + a * sign(G_total)
        11:  eta = clip_eps(X't - X)
        12:  Xt+1 = X + eta
        13: return eta
    """

    def __init__(
        self,
        device: torch.device,
        iterations: int = 50,
        step_size: float = 0.01,
        epsilon: float = 0.08,
        encoder_names: list = None,
        encoder_weights: dict = None,
        freq_attack: bool = True,
        freq_weight: float = 0.5,
    ):
        """
        Args:
            device: Torch device (cuda/cpu).
            iterations: Number of PGD iterations (T). Default 50 for stronger attack.
            step_size: Step size per iteration (a).
            epsilon: Maximum L-inf perturbation bound. Default 0.08 (~20/255).
            encoder_names: List of encoder names to use.
                Options: 'arcface', 'stylegan', 'diffae', 'icface', 'vgg',
                         'sdvae', 'sdxl_vae'.
                Default: all 7 encoders.
            encoder_weights: Per-encoder weight dict. Higher weight = more
                attack budget. Default: VAE encoders get 3x weight.
            freq_attack: Enable frequency-domain attack component.
            freq_weight: Weight of frequency gradient in the ensemble.
        """
        self.device = device
        self.iterations = iterations
        self.step_size = step_size
        self.epsilon = epsilon
        self.freq_attack = freq_attack
        self.freq_weight = freq_weight

        if encoder_names is None:
            encoder_names = ['arcface', 'stylegan', 'diffae', 'icface', 'vgg',
                             'sdvae', 'sdxl_vae']

        self.encoder_weights = encoder_weights or DEFAULT_ENCODER_WEIGHTS

        self.encoders = self._build_encoders(encoder_names)
        print(f"      LEAT: {len(self.encoders)} encoders loaded (weighted ensemble)")
        for name in self.encoders:
            w = self.encoder_weights.get(name, 1.0)
            print(f"        - {name} (weight={w:.1f})")
        if self.freq_attack:
            print(f"        + frequency-domain attack (weight={self.freq_weight:.1f})")

    def _build_encoders(self, names: list) -> dict:
        """Instantiate the requested deepfake model encoders."""
        encoders = {}
        for name in names:
            try:
                if name == 'arcface':
                    encoders[name] = ArcFaceEncoder(self.device)
                elif name == 'stylegan':
                    encoders[name] = StyleGANEncoder(self.device)
                elif name == 'diffae':
                    encoders[name] = DiffAEEncoder(self.device)
                elif name == 'icface':
                    encoders[name] = ICfaceEncoder(self.device)
                elif name == 'vgg':
                    encoders[name] = VGGPerceptualEncoder(self.device)
                elif name == 'sdvae':
                    encoders[name] = SDVAEEncoder(self.device)
                elif name == 'sdxl_vae':
                    # SDXL VAE uses the same architecture as SD 1.5 VAE
                    # but with different trained weights. Loading with SD
                    # architecture + SDXL weights provides a second VAE
                    # attack surface for better transfer.
                    encoders[name] = SDVAEEncoder(self.device, weight_path='sdxl')
                else:
                    print(f"        WARNING: Unknown encoder '{name}', skipping")
            except Exception as e:
                print(f"        WARNING: Failed to load {name}: {e}")
        return encoders

    def _compute_frequency_gradient(self, img_tensor: torch.Tensor,
                                     eta: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-domain attack gradient.

        Targets the mid-to-high frequency bands (8-128 cycles) that VAE
        encoders rely on for detail preservation. By injecting structured
        noise in these bands, we disrupt the VAE's ability to accurately
        compress the image into its latent space.

        The attack uses DCT-like frequency targeting via learned Gaussian
        bandpass filters applied in the Fourier domain.
        """
        B, C, H, W = img_tensor.shape
        x_adv = (img_tensor + eta).requires_grad_(True)

        # Compute 2D FFT of the adversarial image
        x_freq = torch.fft.rfft2(x_adv, norm='ortho')

        # Create frequency coordinate grid
        freq_h = torch.fft.fftfreq(H, device=self.device)
        freq_w = torch.fft.rfftfreq(W, device=self.device)
        freq_grid_h, freq_grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        freq_magnitude = torch.sqrt(freq_grid_h**2 + freq_grid_w**2)

        # Bandpass filter targeting mid-high frequencies (0.05-0.4 normalized)
        # These are the frequencies that VAE encoders are most sensitive to:
        # - Too low: VAE preserves these easily (coarse structure)
        # - Too high: VAE discards these anyway (fine noise)
        # - Mid-high: VAE struggles to decide what to keep
        low_cutoff = 0.05
        high_cutoff = 0.4
        bandpass = torch.exp(-((freq_magnitude - (low_cutoff + high_cutoff) / 2) ** 2) /
                             (2 * ((high_cutoff - low_cutoff) / 4) ** 2))

        # Maximize energy in the target frequency band
        target_energy = (x_freq.abs() * bandpass.unsqueeze(0).unsqueeze(0)).sum()
        loss = -target_energy
        loss.backward()

        grad = x_adv.grad.detach().clone()
        grad_norm = torch.norm(grad, p=2)
        if grad_norm > 1e-12:
            grad = grad / grad_norm

        return grad

    def generate_perturbation(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate LEAT adversarial perturbation using Weighted Normalized
        Gradient Ensemble with frequency-domain attack.

        Args:
            img_tensor: Input image tensor of shape (1, C, H, W) in [0, 1].

        Returns:
            Perturbation tensor of same shape, bounded by [-epsilon, epsilon].
        """
        img_tensor = img_tensor.to(self.device).detach()

        # Freeze encoder parameters (we only optimize the perturbation)
        for encoder in self.encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False

        # Pre-compute original latent representations (Eq. 5: E_k(X))
        original_latents = {}
        with torch.no_grad():
            for name, encoder in self.encoders.items():
                original_latents[name] = encoder(img_tensor).detach()

        # Step 1: Random initialization of perturbation (PGD-style)
        eta = torch.empty_like(img_tensor).uniform_(-self.epsilon, self.epsilon)
        eta = eta.to(self.device)

        for t in range(self.iterations):
            # Current adversarial image: X_t = X + eta
            x_adv = (img_tensor + eta).detach().requires_grad_(True)

            # Step 3: Initialize aggregated normalized gradient
            g_normgrad = torch.zeros_like(img_tensor)

            # Step 4-7: Weighted Normalized Gradient Ensemble
            for name, encoder in self.encoders.items():
                z_adv = encoder(x_adv)
                z_orig = original_latents[name]

                # Maximize L(E_k(X), E_k(X+eta)) -> negate for gradient ascent
                loss = -F.mse_loss(z_adv, z_orig)
                loss.backward(retain_graph=True)

                grad = x_adv.grad.detach().clone()

                # Normalize by L2 norm
                grad_norm = torch.norm(grad, p=2)
                if grad_norm > 1e-12:
                    grad = grad / grad_norm

                # WEIGHTED aggregation: VAE encoders get higher weight
                w = self.encoder_weights.get(name, 1.0)
                g_normgrad = g_normgrad + w * grad

                x_adv.grad.zero_()

            # Frequency-domain attack component
            if self.freq_attack:
                g_freq = self._compute_frequency_gradient(img_tensor, eta)
                g_total = g_normgrad + self.freq_weight * g_freq
            else:
                g_total = g_normgrad

            # PGD step: X'_t = X_t + a * sign(G_total)
            x_prime = (img_tensor + eta) + self.step_size * torch.sign(g_total)

            # Clip perturbation to L-inf ball
            eta = torch.clamp(x_prime - img_tensor, -self.epsilon, self.epsilon)

            # Ensure valid pixel range [0, 1]
            eta = torch.clamp(img_tensor + eta, 0.0, 1.0) - img_tensor

        return eta.detach()

    def compute_latent_disruption(self, img_tensor: torch.Tensor,
                                  perturbation: torch.Tensor) -> dict:
        """
        Measure disruption in each encoder's latent space.

        Args:
            img_tensor: Original image tensor (1, C, H, W).
            perturbation: Perturbation tensor of same shape.

        Returns:
            dict with per-encoder disruption metrics (MSE and cosine distance).
        """
        img_tensor = img_tensor.to(self.device)
        perturbation = perturbation.to(self.device)
        x_adv = torch.clamp(img_tensor + perturbation, 0.0, 1.0)

        metrics = {}
        with torch.no_grad():
            for name, encoder in self.encoders.items():
                z_orig = encoder(img_tensor)
                z_adv = encoder(x_adv)

                mse = F.mse_loss(z_orig, z_adv).item()
                cos_sim = F.cosine_similarity(
                    z_orig.view(1, -1), z_adv.view(1, -1), dim=1
                ).mean().item()

                metrics[name] = {
                    'latent_mse': mse,
                    'latent_cosine_distance': 1.0 - cos_sim,
                }

        return metrics
