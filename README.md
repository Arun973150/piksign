# PikSign: AI Security System for Visual Media (v3.0)

PikSign is an advanced security suite designed to protect visual media from AI exploitation and detect AI-generated or manipulated content with high precision.

## Core Components

### 🛡️ Protection Shield
- **Adaptive Transforms**: Prevents AI models from accurately interpreting or training on protected images.
- **C2PA Integration**: Embeds cryptographically bound provenance metadata (PNG text chunks and JPEG EXIF).
- **Pre-protection Checks**: Automatically blocks protection for already protected images or content detected as AI-generated/manipulated.

### 🔍 Detection Hub
- **Gated Detection Flow**: Efficient multi-track analysis.
  1. **PikSign Check**: Instant identification of protected assets (Skip analysis).
  2. **AI Manipulation Track**: Forensic analysis (ELA, PRNU, Geometric consistency, DIRE).
  3. **Deepfake Track**: Integration with Reality Defender and local face analysis.
  4. **Supplementary Forensics**: Frequency and noise domain analysis.
- **Verdict Engine**: Computes high-confidence authenticity assessments.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd piksign
   ```

2. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   REALITY_DEFENDER_API_KEY=your_api_key_here
   ```

3. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `piexif` is required for JPEG C2PA support.*

## Usage

### 🚀 Streamlit Dashboard (Recommended)
Launch the premium interactive dashboard:
```bash
streamlit run app.py
```

### 💻 CLI Entry Point
Run operations from the command line:
```bash
# Detect AI content
python -m piksign.cli detect path/to/image.jpg

# Protect an image
python -m piksign.cli protect path/to/image.jpg

# Verify C2PA provenance
python -m piksign.cli verify path/to/protected_image.png
```

## Project Structure
```
piksign/
├── app.py                # Streamlit Dashboard
├── piksign/              # Core Package
│   ├── detection/        # Authenticaton & Verdict logic
│   ├── protection/       # Shield & C2PA logic
│   └── ai_image_forensics/ # Forensic signal pipeline
├── photos/               # Test image repository
├── protected_test/       # Output for protected assets
└── DIRE-repo/            # (Dependency) Forensic model weights
```
