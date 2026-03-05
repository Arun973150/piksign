# -*- coding: utf-8 -*-
"""
Real Deepfake Model Encoder Architectures for LEAT.

Implements the actual latent encoders used by deepfake models:
- ArcFace IResNet-50: Face identity encoder (SimSwap, FaceShifter)
- pSp GradualStyleEncoder (IRSE-50): StyleGAN encoder (StyleCLIP, e4e)
- DiffAE Semantic Encoder: Diffusion model encoder (DiffAE)
- ICface Neutral Face Generator: Reenactment encoder (ICface)
- VGG Perceptual Encoder: Perceptual features (deepfake training losses)

These are the 4 encoders from the LEAT paper (Table 1) plus VGG for
ensemble diversity. LEAT attacks their latent encoding process (Eq. 5).
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


# =============================================================================
# ArcFace IResNet-50
# Used by: SimSwap, FaceShifter, InfoSwap, and most face swapping models
# Architecture: IResNet from insightface (deepinsight/insightface)
# Input: any resolution (resized to 112x112 internally)
# Output: 512-d L2-normalized identity embedding
# Weights: https://github.com/deepinsight/insightface/tree/master/model_zoo
# =============================================================================

class IBasicBlock(nn.Module):
    """Basic residual block for IResNet with BN-Conv-BN-PReLU-Conv-BN."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out + identity


class IResNet(nn.Module):
    """
    Improved ResNet (IResNet) backbone from InsightFace.

    Used as the face recognition backbone in ArcFace. The architecture
    differs from standard ResNet: pre-activation BN, PReLU activations,
    and BN on the final feature vector.

    IResNet-50: layers=[3, 4, 14, 3], 512-d output
    IResNet-100: layers=[3, 13, 30, 3], 512-d output
    """

    def __init__(self, layers=(3, 4, 14, 3), dropout=0.0, num_features=512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


class ArcFaceEncoder(nn.Module):
    """
    ArcFace face identity encoder.

    The standard face recognition model used by nearly all face swapping
    deepfake systems (SimSwap, FaceShifter, InfoSwap, HifiFace, etc.)
    to extract 512-d identity embeddings from face images.

    The LEAT attack disrupts this encoder's output, causing face swap
    models to fail at extracting the correct identity from the source face.

    Weight download: https://github.com/deepinsight/insightface/tree/master/model_zoo
    Place weights at: piksign/protection/weights/arcface_r50.pth
    """

    def __init__(self, device, weight_path=None):
        super().__init__()
        self.backbone = IResNet(layers=(3, 4, 14, 3))
        self._load_weights(weight_path)
        self.to(device)
        self.eval()

    def _load_weights(self, weight_path):
        paths_to_try = []
        if weight_path:
            paths_to_try.append(weight_path)
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'arcface_r50.pth'))
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'backbone.pth'))

        for path in paths_to_try:
            if os.path.exists(path):
                state_dict = torch.load(path, map_location='cpu', weights_only=True)
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"        ArcFace: loaded weights from {path}")
                return

        print("        ArcFace: no pre-trained weights found, using kaiming init")
        print("        (Place arcface_r50.pth in piksign/protection/weights/)")

    def forward(self, x):
        # Differentiable preprocessing: resize to 112x112 and normalize to [-1, 1]
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        return self.backbone(x)  # (B, 512)


# =============================================================================
# pSp / e4e GradualStyleEncoder (with IRSE-50 backbone)
# Used by: StyleCLIP, e4e, pSp, InterFaceGAN, GANSpace
# Architecture: IRSE-50 backbone + FPN-style progressive style mapping
# Input: any resolution (resized to 256x256 internally)
# Output: 18 x 512 = 9216-d W+ latent code (flattened for LEAT)
# Weights: https://github.com/omertov/encoder4editing
# =============================================================================

class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class BottleneckIRSE(nn.Module):
    """
    IR-SE bottleneck block from face.evoLVe.

    Used in the IRSE-50 backbone that pSp/e4e/StyleCLIP rely on.
    Structure: BN -> Conv3x3 -> PReLU -> Conv3x3 -> BN -> SE + shortcut
    """

    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut = nn.MaxPool2d(1, stride) if stride > 1 else nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        return self.res(x) + self.shortcut(x)


class GradualStyleBlock(nn.Module):
    """
    Map-to-style block from pSp (pixel2style2pixel).

    Progressively downsamples a feature map via strided convolutions
    until it reaches 1x1, then applies a linear layer to produce
    a single 512-d style code for one of StyleGAN's 18 style inputs.
    """

    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c = out_c
        num_pools = int(math.log2(spatial))
        modules = []
        modules.append(nn.Conv2d(in_c, out_c, 3, 2, 1))
        modules.append(nn.LeakyReLU())
        for _ in range(num_pools - 1):
            modules.append(nn.Conv2d(out_c, out_c, 3, 2, 1))
            modules.append(nn.LeakyReLU())
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        return self.linear(x)


def _build_irse50_body():
    """
    Build the IRSE-50 body as a flat list of BottleneckIRSE blocks.

    Block configuration for IRSE-50:
      Block 1: in=64,  depth=64,  3 units (indices 0-2)
      Block 2: in=64,  depth=128, 4 units (indices 3-6)
      Block 3: in=128, depth=256, 14 units (indices 7-20)
      Block 4: in=256, depth=512, 3 units (indices 21-23)

    Feature extraction points for GradualStyleEncoder:
      c1 @ index 6:  128ch, 64x64  (for 256x256 input)
      c2 @ index 20: 256ch, 32x32
      c3 @ index 23: 512ch, 16x16
    """
    blocks_config = [
        # (in_channel, depth, num_units)
        (64, 64, 3),
        (64, 128, 4),
        (128, 256, 14),
        (256, 512, 3),
    ]
    modules = []
    for in_ch, depth, num_units in blocks_config:
        # First unit of each block has stride=2
        modules.append(BottleneckIRSE(in_ch, depth, stride=2))
        for _ in range(num_units - 1):
            modules.append(BottleneckIRSE(depth, depth, stride=1))
    return nn.Sequential(*modules)


class GradualStyleEncoder(nn.Module):
    """
    GradualStyleEncoder from pSp (pixel2style2pixel).

    This is the exact encoder architecture used by:
    - pSp (Pixel2Style2Pixel) for StyleGAN inversion
    - e4e (Encoder4Editing) for StyleGAN editing
    - StyleCLIP for text-driven face manipulation

    Architecture:
    1. IRSE-50 backbone extracts features at 3 scales (FPN-like)
    2. Lateral connections merge features across scales
    3. GradualStyleBlock modules map each scale to style codes
    4. Output: 18 style codes of 512-d each (W+ latent space)

    For LEAT, disrupting these style codes prevents the StyleGAN
    generator from producing coherent manipulated faces, regardless
    of what target attribute (text prompt, direction) is used.
    """

    def __init__(self, n_styles=18):
        super().__init__()
        self.n_styles = n_styles

        # Input layer: Conv -> BN -> PReLU
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        # IRSE-50 backbone (24 bottleneck blocks)
        self.body = _build_irse50_body()

        # Feature extraction indices in the body
        # c1 @ index 6  -> 128ch, 64x64 (end of block 2)
        # c2 @ index 20 -> 256ch, 32x32 (end of block 3)
        # c3 @ index 23 -> 512ch, 16x16 (end of block 4)

        # Style mapping heads
        self.styles = nn.ModuleList()
        self.coarse_ind = 3   # styles 0-2 from c3 (16x16)
        self.middle_ind = 7   # styles 3-6 from p2 (32x32)
        # styles 7-17 from p1 (64x64)

        for i in range(n_styles):
            if i < self.coarse_ind:
                self.styles.append(GradualStyleBlock(512, 512, 16))
            elif i < self.middle_ind:
                self.styles.append(GradualStyleBlock(512, 512, 32))
            else:
                self.styles.append(GradualStyleBlock(512, 512, 64))

        # FPN lateral connections
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1)  # c2 -> 512ch
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1)  # c1 -> 512ch

    def forward(self, x):
        x = self.input_layer(x)

        # Forward through IRSE-50 body, extracting features at 3 scales
        body_modules = list(self.body.children())
        for i, layer in enumerate(body_modules):
            x = layer(x)
            if i == 6:
                c1 = x   # 128ch, 64x64
            elif i == 20:
                c2 = x   # 256ch, 32x32
            elif i == 23:
                c3 = x   # 512ch, 16x16

        # Coarse styles from c3 (16x16)
        latents = []
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        # Medium styles from FPN merge of c3 + c2 (32x32)
        p2 = F.interpolate(c3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        p2 = p2 + self.latlayer1(c2)
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        # Fine styles from FPN merge of p2 + c1 (64x64)
        p1 = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=True)
        p1 = p1 + self.latlayer2(c1)
        for j in range(self.middle_ind, self.n_styles):
            latents.append(self.styles[j](p1))

        # Stack all style codes: (B, 18, 512)
        return torch.stack(latents, dim=1)


class StyleGANEncoder(nn.Module):
    """
    StyleGAN latent encoder for LEAT.

    Wraps the GradualStyleEncoder (from pSp/e4e) with differentiable
    preprocessing. Maps face images to the W+ latent space (18 x 512)
    which is flattened to 9216-d for the LEAT MSE loss.

    Disrupting this latent prevents StyleGAN-based manipulation models
    (StyleCLIP, InterFaceGAN, GANSpace, e4e) from encoding the face
    correctly, causing all subsequent manipulations to fail.

    Weight download (e4e encoder):
      https://github.com/omertov/encoder4editing
    Place weights at: piksign/protection/weights/e4e_ffhq_encode.pt
    """

    def __init__(self, device, weight_path=None, n_styles=18):
        super().__init__()
        self.encoder = GradualStyleEncoder(n_styles=n_styles)
        self._load_weights(weight_path)
        self.to(device)
        self.eval()

    def _load_weights(self, weight_path):
        paths_to_try = []
        if weight_path:
            paths_to_try.append(weight_path)
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'e4e_ffhq_encode.pt'))
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'psp_ffhq_encode.pt'))

        for path in paths_to_try:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                # e4e/pSp checkpoints store the encoder under 'state_dict'
                # with keys like 'encoder.xxx'
                if 'state_dict' in checkpoint:
                    state_dict = {
                        k.replace('encoder.', '', 1): v
                        for k, v in checkpoint['state_dict'].items()
                        if k.startswith('encoder.')
                    }
                else:
                    state_dict = checkpoint
                self.encoder.load_state_dict(state_dict, strict=False)
                print(f"        StyleGAN encoder: loaded weights from {path}")
                return

        print("        StyleGAN encoder: no pre-trained weights found, using random init")
        print("        (Place e4e_ffhq_encode.pt in piksign/protection/weights/)")

    def forward(self, x):
        # Differentiable preprocessing: resize to 256x256, normalize to [-1, 1]
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        codes = self.encoder(x)      # (B, 18, 512)
        return codes.view(x.size(0), -1)  # (B, 9216) -- flattened W+ for MSE loss


# =============================================================================
# VGG-19 Perceptual Encoder
# Used by: Nearly all deepfake training pipelines for perceptual loss,
#   face reenactment (FOMM, PIRenderer), and style transfer
# Kept as a supplementary encoder for ensemble diversity
# =============================================================================

class VGGPerceptualEncoder(nn.Module):
    """
    VGG-19 multi-scale perceptual feature encoder.

    Extracts features at relu1_2, relu2_2, relu3_4, relu4_4 and
    concatenates them via global average pooling. These are the exact
    feature layers used in perceptual loss (LPIPS) and most deepfake
    training pipelines.

    Uses pre-trained ImageNet weights (no separate download needed).
    """

    def __init__(self, device):
        super().__init__()
        try:
            vgg = __import__('torchvision').models.vgg19(
                weights=__import__('torchvision').models.VGG19_Weights.IMAGENET1K_V1
            )
        except Exception:
            vgg = __import__('torchvision').models.vgg19(pretrained=True)

        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:4])    # relu1_2: 64ch
        self.slice2 = nn.Sequential(*features[4:9])   # relu2_2: 128ch
        self.slice3 = nn.Sequential(*features[9:18])  # relu3_4: 256ch
        self.slice4 = nn.Sequential(*features[18:27])  # relu4_4: 512ch

        for module in [self.slice1, self.slice2, self.slice3, self.slice4]:
            module.to(device)
            module.eval()

        self.normalize = __import__('torchvision').transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        x = self.normalize(x)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)

        f1 = F.adaptive_avg_pool2d(h1, 1).view(x.size(0), -1)  # 64-d
        f2 = F.adaptive_avg_pool2d(h2, 1).view(x.size(0), -1)  # 128-d
        f3 = F.adaptive_avg_pool2d(h3, 1).view(x.size(0), -1)  # 256-d
        f4 = F.adaptive_avg_pool2d(h4, 1).view(x.size(0), -1)  # 512-d

        return torch.cat([f1, f2, f3, f4], dim=1)  # (B, 960)


# =============================================================================
# DiffAE Semantic Encoder
# Used by: Diffusion Autoencoders (Preechakul et al. 2022) for face
#   attribute manipulation via diffusion models
# Architecture: BeatGANs UNet encoder -> 512-d semantic vector
# Input: any resolution (resized to 256x256 internally)
# Output: 512-d semantic latent vector
# Weights: https://github.com/phizaz/diffae
# Reference: "Diffusion Autoencoders: Toward a Meaningful and Decodable
#   Representation" (CVPR 2022)
#
# Architecture matches the actual BeatGANs UNet encoder checkpoint with:
#   net_enc_channel_mult: (1, 1, 2, 2, 4, 4, 4)
#   net_ch: 128, net_attn: (16,), net_enc_num_res_blocks: 2
#   net_beatgans_enc_out_channels: 512
# =============================================================================

class DiffAEResBlock(nn.Module):
    """
    Residual block from BeatGANs/DiffAE encoder.

    Structure: GroupNorm -> SiLU -> Conv3x3 + skip_connection
    Key naming matches checkpoint: in_layers.{0,2}, skip_connection
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        )
        if in_ch != out_ch:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, 1)
        elif stride > 1:
            self.skip_connection = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):
        return self.in_layers(x) + self.skip_connection(x)


class DiffAEAttention(nn.Module):
    """
    Self-attention block from BeatGANs/DiffAE encoder.

    Key naming matches checkpoint: norm, qkv, proj_out
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, -1)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(-1, -2), k) * scale
        attn = attn.softmax(dim=-1)
        h = torch.bmm(v, attn.transpose(-1, -2))
        h = self.proj_out(h).reshape(B, C, H, W)
        return x + h


class DiffAESemanticEncoder(nn.Module):
    """
    Semantic encoder from Diffusion Autoencoders (DiffAE).

    Architecture matches the BeatGANs UNet encoder exactly, with key
    naming that loads directly from the official DiffAE checkpoint.

    For FFHQ256 with channel_mult=(1,1,2,2,4,4,4):
      input_blocks[0]:  Conv(3->128)                  256x256
      input_blocks[1-2]: 2x ResBlock(128)            256x256  (level 0)
      input_blocks[3]:  ResBlock(128, stride=2)      ->128x128
      input_blocks[4-5]: 2x ResBlock(128)            128x128  (level 1)
      input_blocks[6]:  ResBlock(128, stride=2)      ->64x64
      input_blocks[7-8]: ResBlock(128->256)+          64x64    (level 2)
      input_blocks[9]:  ResBlock(256, stride=2)      ->32x32
      input_blocks[10-11]: 2x ResBlock(256)          32x32    (level 3)
      input_blocks[12]: ResBlock(256, stride=2)      ->16x16
      input_blocks[13-14]: ResBlock(256->512)+Attn    16x16    (level 4)
      input_blocks[15]: ResBlock(512, stride=2)      ->8x8
      input_blocks[16-17]: 2x ResBlock(512)          8x8      (level 5)
      input_blocks[18]: ResBlock(512, stride=2)      ->4x4
      input_blocks[19-20]: 2x ResBlock(512)          4x4      (level 6)
      middle_block: ResBlock + Attention + ResBlock
      out: GroupNorm -> SiLU -> AdaptivePool -> Conv1x1 -> 512-d
    """

    def __init__(self, model_channels=128, channel_mult=(1, 1, 2, 2, 4, 4, 4),
                 num_res_blocks=2, attn_resolutions=(16,), out_channels=512):
        super().__init__()

        # Build input_blocks as nn.ModuleList of nn.Sequential
        # Each Sequential contains the submodules at indices matching checkpoint keys
        self.input_blocks = nn.ModuleList()

        # Block 0: initial conv
        self.input_blocks.append(nn.Sequential(
            nn.Conv2d(3, model_channels, 3, padding=1)
        ))

        ch = model_channels
        current_res = 256

        for level_idx, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                block_modules = [DiffAEResBlock(ch, out_ch)]
                ch = out_ch
                if current_res in attn_resolutions:
                    block_modules.append(DiffAEAttention(ch))
                self.input_blocks.append(nn.Sequential(*block_modules))

            # Downsample after each level except the last (strided ResBlock)
            if level_idx < len(channel_mult) - 1:
                self.input_blocks.append(nn.Sequential(
                    DiffAEResBlock(ch, ch, stride=2)
                ))
                current_res //= 2

        # Middle block: ResBlock + Attention + ResBlock
        self.middle_block = nn.ModuleList([
            DiffAEResBlock(ch, ch),
            DiffAEAttention(ch),
            DiffAEResBlock(ch, ch),
        ])

        # Output: GroupNorm -> SiLU -> Pool -> Conv1x1
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, out_channels, 1),
        )

    def forward(self, x):
        h = x
        for block in self.input_blocks:
            h = block(h)
        for module in self.middle_block:
            h = module(h)
        h = self.out(h)
        return h.view(h.size(0), -1)  # (B, 512)


class DiffAEEncoder(nn.Module):
    """
    DiffAE semantic encoder wrapper for LEAT.

    Wraps the DiffAESemanticEncoder with differentiable preprocessing.
    The LEAT attack disrupts this 512-d semantic vector, causing the
    DiffAE's conditional DDPM decoder to generate distorted outputs
    regardless of what attribute manipulation is requested.

    This is the encoder from the LEAT paper's Table 1:
      Model: DiffAE
      Intermediate latent: 512-d vector
      Target attribute-independent: yes
      Low-dimensional semantic space: yes

    Weight download: https://github.com/phizaz/diffae
    Place weights at: piksign/protection/weights/diffae_ffhq256.pt
    """

    def __init__(self, device, weight_path=None):
        super().__init__()
        self.encoder = DiffAESemanticEncoder()
        self._load_weights(weight_path)
        self.to(device)
        self.eval()

    def _load_weights(self, weight_path):
        paths_to_try = []
        if weight_path:
            paths_to_try.append(weight_path)
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'diffae_ffhq256.pt'))
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'diffae_ffhq256_autoenc.ckpt'))

        for path in paths_to_try:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if 'state_dict' in checkpoint:
                    # Extract encoder weights, strip 'model.encoder.' prefix
                    state_dict = {}
                    for k, v in checkpoint['state_dict'].items():
                        if k.startswith('model.encoder.'):
                            new_key = k.replace('model.encoder.', '', 1)
                            state_dict[new_key] = v
                    if state_dict:
                        matched, total = 0, len(state_dict)
                        result = self.encoder.load_state_dict(state_dict, strict=False)
                        matched = total - len(result.unexpected_keys)
                        print(f"        DiffAE: loaded {matched}/{total} weight tensors from {path}")
                        return
                else:
                    self.encoder.load_state_dict(checkpoint, strict=False)
                    print(f"        DiffAE: loaded weights from {path}")
                    return

        print("        DiffAE: no pre-trained weights found, using random init")
        print("        (Place diffae_ffhq256.pt in piksign/protection/weights/)")

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        return self.encoder(x)  # (B, 512)


# =============================================================================
# ICface Neutral Face Generator (GN)
# Used by: ICface (Tripathy, Kannala, Rahtu 2020) for face reenactment
# Architecture: Sequential encoder-decoder (no skip connections) with
#   BatchNorm, 23 input channels (3 RGB + 20 AU heatmaps), 6 ResBlocks
# Input: any resolution (resized to 128x128 internally, zero AU channels)
# Output: 3x128x128 neutral face image (flattened to 49152-d for LEAT)
# Weights: latest_net_GN.pth from ICface repo
# Reference: "ICface: Interpretable and Controllable Face Reenactment
#   Using GANs" (WACV 2020)
#
# Note from the LEAT paper (Table 1): ICface's intermediate latent is a
# "Neutral Image" -- NOT embedded in a low-dimensional semantic space.
# But it IS target-attribute-independent (doesn't depend on Action Units),
# so LEAT can still attack it effectively.
#
# The GN (Neutral Generator) takes face + AU heatmaps (23ch) and produces
# a neutral face. For LEAT, we feed the face + 20 zero AU channels.
# =============================================================================

class ICfaceResBlock(nn.Module):
    """
    Residual block from ICface with BatchNorm.

    Key naming matches checkpoint: conv_block.{1,2,6,7}
    Structure: ReflPad -> Conv -> BN -> ReLU -> Dropout -> ReflPad -> Conv -> BN
    """

    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),              # 0
            nn.Conv2d(channels, channels, 3),   # 1
            nn.BatchNorm2d(channels),           # 2
            nn.ReLU(True),                      # 3
            nn.Dropout(0.5),                    # 4
            nn.ReflectionPad2d(1),              # 5
            nn.Conv2d(channels, channels, 3),   # 6
            nn.BatchNorm2d(channels),           # 7
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ICfaceNeutralGenerator(nn.Module):
    """
    Neutral face generator (GN) from ICface.

    Sequential encoder-decoder that produces a neutral (expressionless)
    face from any input face + AU heatmaps. The neutral face is the
    "latent" that LEAT attacks -- it captures identity information while
    being independent of Action Units (target attributes).

    Architecture matches the actual checkpoint (module.model.*):
      model[0]:  ReflectionPad2d(3)
      model[1]:  Conv2d(23->128, 7x7)        Encoder
      model[2]:  BatchNorm2d(128)
      model[3]:  ReLU
      model[4]:  Conv2d(128->256, 3x3, s=2)  Downsample
      model[5]:  BatchNorm2d(256)
      model[6]:  ReLU
      model[7]:  Conv2d(256->512, 3x3, s=2)  Downsample
      model[8]:  BatchNorm2d(512)
      model[9]:  ReLU
      model[10-15]: 6x ResBlock(512)         Bottleneck
      model[16]: ConvTranspose2d(512->256)    Upsample
      model[17]: BatchNorm2d(256)
      model[18]: ReLU
      model[19]: ConvTranspose2d(256->128)    Upsample
      model[20]: BatchNorm2d(128)
      model[21]: ReLU
      model[22]: ReflectionPad2d(3)
      model[23]: Conv2d(128->3, 7x7)         Output
      model[24]: Tanh
    """

    def __init__(self, input_nc=23):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),                                          # 0
            nn.Conv2d(input_nc, 128, 7),                                    # 1
            nn.BatchNorm2d(128),                                            # 2
            nn.ReLU(True),                                                  # 3
            nn.Conv2d(128, 256, 3, stride=2, padding=1),                    # 4
            nn.BatchNorm2d(256),                                            # 5
            nn.ReLU(True),                                                  # 6
            nn.Conv2d(256, 512, 3, stride=2, padding=1),                    # 7
            nn.BatchNorm2d(512),                                            # 8
            nn.ReLU(True),                                                  # 9
            ICfaceResBlock(512),                                            # 10
            ICfaceResBlock(512),                                            # 11
            ICfaceResBlock(512),                                            # 12
            ICfaceResBlock(512),                                            # 13
            ICfaceResBlock(512),                                            # 14
            ICfaceResBlock(512),                                            # 15
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1,
                               output_padding=1),                           # 16
            nn.BatchNorm2d(256),                                            # 17
            nn.ReLU(True),                                                  # 18
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,
                               output_padding=1),                           # 19
            nn.BatchNorm2d(128),                                            # 20
            nn.ReLU(True),                                                  # 21
            nn.ReflectionPad2d(3),                                          # 22
            nn.Conv2d(128, 3, 7),                                           # 23
            nn.Tanh(),                                                      # 24
        )

    def forward(self, x):
        return self.model(x)


class ICfaceEncoder(nn.Module):
    """
    ICface neutral face encoder for LEAT.

    Wraps the ICfaceNeutralGenerator (GN) with differentiable preprocessing.
    Produces a neutral (expressionless) face image, then flattens it
    as the "latent" for the LEAT MSE loss.

    The GN takes 23 channels (3 RGB + 20 AU heatmaps). For LEAT, we feed
    the face image with 20 zero AU channels (producing a neutral face
    independent of any target expression).

    From the LEAT paper (Table 1):
      Model: ICface
      Intermediate latent: Neutral Image
      Target attribute-independent: yes (doesn't depend on Action Units)
      Low-dimensional semantic space: NO (it's a full image)

    Weight: latest_net_GN.pth from ICface
    Place weights at: piksign/protection/weights/icface_neutral.pth
    """

    def __init__(self, device, weight_path=None):
        super().__init__()
        self.generator = ICfaceNeutralGenerator(input_nc=23)
        self._load_weights(weight_path)
        self.to(device)
        self.eval()

    def _load_weights(self, weight_path):
        paths_to_try = []
        if weight_path:
            paths_to_try.append(weight_path)
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'icface_neutral.pth'))
        paths_to_try.append(os.path.join(WEIGHTS_DIR, 'latest_net_GN.pth'))

        for path in paths_to_try:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
                    state_dict = checkpoint['generator']
                else:
                    state_dict = checkpoint

                # Strip 'module.' prefix from DataParallel wrapping
                cleaned = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
                    cleaned[new_key] = v

                result = self.generator.load_state_dict(cleaned, strict=False)
                matched = len(cleaned) - len(result.unexpected_keys)
                print(f"        ICface: loaded {matched}/{len(cleaned)} weight tensors from {path}")
                return

        print("        ICface: no pre-trained weights found, using random init")
        print("        (Place icface_neutral.pth in piksign/protection/weights/)")

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        # Pad with 20 zero AU channels (neutral = no action units)
        zeros = torch.zeros(x.size(0), 20, x.size(2), x.size(3),
                            device=x.device, dtype=x.dtype)
        x_au = torch.cat([x, zeros], dim=1)  # (B, 23, 128, 128)
        neutral = self.generator(x_au)        # (B, 3, 128, 128) neutral face
        return neutral.view(x.size(0), -1)    # (B, 49152) -- flattened image as latent


# =============================================================================
# Stable Diffusion VAE Encoder
# Used by: All latent diffusion models (SD 1.5, SDXL, Flux, and architecturally
#   similar proprietary models like Google's Nano Banana / Imagen)
# Architecture: KL-regularized autoencoder encoder from LDM/diffusers
# Input: any resolution (resized to 256x256 internally)
# Output: 4x32x32 = 4096-d latent (flattened for LEAT)
# Weights: https://huggingface.co/stabilityai/sd-vae-ft-mse
#
# This encoder targets the universal bottleneck in ALL modern diffusion-based
# image editors: the VAE compression step (3x256x256 -> 4x32x32). By disrupting
# this latent representation, LEAT perturbations transfer to any diffusion
# model that uses a similar VAE architecture for latent space encoding --
# including proprietary models whose exact weights are unknown.
#
# Architecture: conv_in -> 4 down stages (ch_mult 1,2,4,4) -> mid -> norm -> conv_out
# Key naming matches diffusers checkpoint exactly (encoder.down.X.block.Y.*)
# =============================================================================

class SDVAEResBlock(nn.Module):
    """
    Residual block from LDM/diffusers VAE encoder.

    Key naming matches checkpoint: norm1, conv1, norm2, conv2, nin_shortcut
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class SDVAEAttention(nn.Module):
    """
    Self-attention block from LDM/diffusers VAE encoder.

    Key naming matches checkpoint: norm, q, k, v, proj_out
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, -1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1)
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = attn.softmax(dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        h = self.proj_out(h)
        return x + h


class SDVAEDownsample(nn.Module):
    """Downsample block: asymmetric padding + stride-2 conv."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x):
        # Asymmetric padding: (0,1,0,1) -- matches diffusers implementation
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class SDVAEDownBlock(nn.Module):
    """
    Down block: 2 ResBlocks + optional downsample.

    Key naming: block.{0,1} for ResBlocks, downsample for stride-2 conv
    """

    def __init__(self, in_ch, out_ch, add_downsample=True):
        super().__init__()
        self.block = nn.ModuleList([
            SDVAEResBlock(in_ch, out_ch),
            SDVAEResBlock(out_ch, out_ch),
        ])
        self.downsample = SDVAEDownsample(out_ch) if add_downsample else None

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SDVAEMidBlock(nn.Module):
    """
    Mid block: ResBlock + Attention + ResBlock.

    Key naming: block_1, attn_1, block_2
    """

    def __init__(self, channels):
        super().__init__()
        self.block_1 = SDVAEResBlock(channels, channels)
        self.attn_1 = SDVAEAttention(channels)
        self.block_2 = SDVAEResBlock(channels, channels)

    def forward(self, x):
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x


class SDVAEEncoderModule(nn.Module):
    """
    VAE encoder from Stable Diffusion (LDM/diffusers).

    Architecture with ch_mult=(1, 2, 4, 4), base_ch=128:
      conv_in:     Conv2d(3->128)                256x256
      down.0:      2x ResBlock(128->128) + DS     ->128x128
      down.1:      2x ResBlock(128->256) + DS     ->64x64
      down.2:      2x ResBlock(256->512) + DS     ->32x32
      down.3:      2x ResBlock(512->512)          32x32 (no DS)
      mid:         ResBlock + Attention + ResBlock
      norm_out:    GroupNorm(32, 512)
      conv_out:    Conv2d(512->8, 3x3)

    The 8-channel output encodes mean and log-variance of the KL posterior.
    For LEAT, we take the first 4 channels (mean) as the latent z.

    Key naming matches diffusers `diffusion_pytorch_model.bin` exactly.
    """

    def __init__(self, ch=128, ch_mult=(1, 2, 4, 4), z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)

        # Down blocks
        self.down = nn.ModuleList()
        in_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            add_ds = (i < len(ch_mult) - 1)  # No downsample on last level
            self.down.append(SDVAEDownBlock(in_ch, out_ch, add_downsample=add_ds))
            in_ch = out_ch

        # Mid block
        self.mid = SDVAEMidBlock(in_ch)

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for down_block in self.down:
            h = down_block(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class SDVAEEncoder(nn.Module):
    """
    Stable Diffusion VAE encoder for LEAT.

    Targets the universal bottleneck in all latent diffusion models:
    the VAE compression step that maps pixel images to latent space.
    By disrupting this latent z, LEAT perturbations transfer to ANY
    diffusion-based image editor that uses a similar VAE architecture --
    including proprietary models like Google's Nano Banana / Imagen.

    The encoder compresses 3x256x256 -> 4x32x32 (4096-d flattened).
    We use the posterior mean (first 4 channels of the 8-channel output)
    plus the quant_conv projection, matching the exact inference pipeline.

    Weight download: https://huggingface.co/stabilityai/sd-vae-ft-mse
    Place weights at: piksign/protection/weights/sd_vae_ft_mse.bin
    """

    def __init__(self, device, weight_path=None):
        super().__init__()
        self.encoder = SDVAEEncoderModule()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self._variant = 'sdxl' if weight_path == 'sdxl' else 'sd15'
        self._load_weights(None if weight_path == 'sdxl' else weight_path)
        self.to(device)
        self.eval()

    @staticmethod
    def _remap_diffusers_keys(state_dict):
        """Remap HuggingFace diffusers key naming to our module naming.

        Diffusers uses:                    Our modules use:
          down_blocks.X.resnets.Y.*   ->    down.X.block.Y.*
          down_blocks.X.downsamplers.0.*->  down.X.downsample.*
          *.conv_shortcut.*            ->    *.nin_shortcut.*
          mid_block.resnets.0.*        ->    mid.block_1.*
          mid_block.resnets.1.*        ->    mid.block_2.*
          mid_block.attentions.0.group_norm.* -> mid.attn_1.norm.*
          mid_block.attentions.0.query.*     -> mid.attn_1.q.*
          mid_block.attentions.0.key.*       -> mid.attn_1.k.*
          mid_block.attentions.0.value.*     -> mid.attn_1.v.*
          mid_block.attentions.0.proj_attn.* -> mid.attn_1.proj_out.*
          conv_norm_out.*              ->    norm_out.*
        """
        import re
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            # Down blocks
            new_k = re.sub(r'^down_blocks\.(\d+)\.resnets\.(\d+)\.',
                           r'down.\1.block.\2.', new_k)
            new_k = re.sub(r'^down_blocks\.(\d+)\.downsamplers\.0\.',
                           r'down.\1.downsample.', new_k)
            # Shortcut naming
            new_k = new_k.replace('.conv_shortcut.', '.nin_shortcut.')
            # Mid block resnets
            new_k = new_k.replace('mid_block.resnets.0.', 'mid.block_1.')
            new_k = new_k.replace('mid_block.resnets.1.', 'mid.block_2.')
            # Mid block attention
            new_k = new_k.replace('mid_block.attentions.0.group_norm.', 'mid.attn_1.norm.')
            new_k = new_k.replace('mid_block.attentions.0.query.', 'mid.attn_1.q.')
            new_k = new_k.replace('mid_block.attentions.0.key.', 'mid.attn_1.k.')
            new_k = new_k.replace('mid_block.attentions.0.value.', 'mid.attn_1.v.')
            new_k = new_k.replace('mid_block.attentions.0.proj_attn.', 'mid.attn_1.proj_out.')
            # Output norm
            new_k = new_k.replace('conv_norm_out.', 'norm_out.')
            # Reshape 2D attention weights (512,512) to Conv2d format (512,512,1,1)
            if 'attn_1.' in new_k and 'weight' in new_k and v.dim() == 2:
                v = v.unsqueeze(-1).unsqueeze(-1)
            remapped[new_k] = v
        return remapped

    def _load_weights(self, weight_path):
        paths_to_try = []
        if weight_path:
            paths_to_try.append(weight_path)
        if self._variant == 'sdxl':
            # SDXL VAE: same architecture, different weights
            paths_to_try.append(os.path.join(WEIGHTS_DIR, 'sdxl_vae.bin'))
            paths_to_try.append(os.path.join(WEIGHTS_DIR, 'sdxl_vae.safetensors'))
            # Fall back to SD 1.5 weights if SDXL not available
            paths_to_try.append(os.path.join(WEIGHTS_DIR, 'sd_vae_ft_mse.bin'))
        else:
            paths_to_try.append(os.path.join(WEIGHTS_DIR, 'sd_vae_ft_mse.bin'))
            paths_to_try.append(os.path.join(WEIGHTS_DIR, 'sd_vae.bin'))
            paths_to_try.append(os.path.join(WEIGHTS_DIR, 'sd_vae_ft_mse.safetensors'))

        for path in paths_to_try:
            if os.path.exists(path):
                if path.endswith('.safetensors'):
                    try:
                        from safetensors.torch import load_file
                        full_state = load_file(path)
                    except ImportError:
                        print("        SD VAE: safetensors not installed, skipping")
                        continue
                else:
                    full_state = torch.load(path, map_location='cpu', weights_only=True)

                # Extract encoder + quant_conv weights from the full VAE checkpoint
                encoder_state = {}
                quant_state = {}
                for k, v in full_state.items():
                    if k.startswith('encoder.'):
                        encoder_state[k.replace('encoder.', '', 1)] = v
                    elif k.startswith('quant_conv.'):
                        quant_state[k.replace('quant_conv.', '', 1)] = v

                # Remap diffusers key naming to our module naming
                encoder_state = self._remap_diffusers_keys(encoder_state)

                variant = self._variant.upper()
                if encoder_state:
                    result = self.encoder.load_state_dict(encoder_state, strict=False)
                    matched = len(encoder_state) - len(result.unexpected_keys)
                    total = len(encoder_state)
                    print(f"        {variant} VAE: loaded {matched}/{total} encoder weights from {path}")
                if quant_state:
                    self.quant_conv.load_state_dict(quant_state, strict=False)
                    print(f"        {variant} VAE: loaded quant_conv weights")
                return

        variant = self._variant.upper()
        print(f"        {variant} VAE: no pre-trained weights found, using random init")
        if self._variant == 'sdxl':
            print("        (Place sdxl_vae.safetensors in piksign/protection/weights/)")
            print("        Download: huggingface.co/stabilityai/sdxl-vae")
        else:
            print("        (Place sd_vae_ft_mse.bin in piksign/protection/weights/)")
            print("        Download: huggingface.co/stabilityai/sd-vae-ft-mse")

    def forward(self, x):
        # Resize to 256x256 and normalize to [-1, 1] (SD preprocessing)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5

        # Encode -> quant_conv -> take posterior mean (first 4 channels)
        h = self.encoder(x)          # (B, 8, 32, 32)
        h = self.quant_conv(h)       # (B, 8, 32, 32)
        z = h[:, :4]                 # (B, 4, 32, 32) -- posterior mean

        return z.reshape(x.size(0), -1)  # (B, 4096) -- flattened latent
