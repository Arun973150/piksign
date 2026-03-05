# -*- coding: utf-8 -*-
"""
PikSign CLI - Command Line Interface
Entry point for both protection and detection operations.
"""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='piksign',
        description='PikSign - AI Security System for Visual Media',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Protect an image
  python -m piksign.cli protect image.jpg
  
  # Protect with custom output path
  python -m piksign.cli protect image.jpg -o protected.png
  
  # Verify protection
  python -m piksign.cli verify protected.png
  
  # Detect AI-generated content (Phase 2)
  python -m piksign.cli detect image.jpg
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Protect command
    protect_parser = subparsers.add_parser(
        'protect', 
        help='Protect an image from AI interpretation'
    )
    protect_parser.add_argument('image', help='Path to image to protect')
    protect_parser.add_argument(
        '-o', '--output', 
        help='Output path (default: protected_<name>.png)'
    )
    protect_parser.add_argument(
        '--no-visualize', 
        action='store_true',
        help='Skip visualization'
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        'verify', 
        help='Verify protection on an image'
    )
    verify_parser.add_argument('image', help='Path to protected image')
    
    # Detect command (Phase 2)
    detect_parser = subparsers.add_parser(
        'detect', 
        help='Detect AI-generated or manipulated content (Phase 2)'
    )
    detect_parser.add_argument('image', help='Path to image to analyze')
    detect_parser.add_argument(
        '--deepfake', 
        action='store_true',
        help='Run deepfake detection'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'protect':
        from piksign.protection import PikSignShield
        from piksign.protection.visualization import visualize_results
        
        shield = PikSignShield()
        output_path, metrics = shield.protect_image(args.image, args.output)
        
        if not args.no_visualize:
            try:
                visualize_results(args.image, output_path)
            except Exception as e:
                print(f"[!] Visualization failed: {e}")
        
        # Print summary
        print("\n[i] SUMMARY")
        print("=" * 60)
        print(f"[OK] PSNR: {metrics['psnr']:.2f} dB")
        print(f"[OK] SSIM: {metrics['ssim']:.4f}")
        print(f"[OK] Drift: {metrics['embedding_drift']:.4f}")
        print(f"[OK] Output: {output_path}")
        
    elif args.command == 'verify':
        from piksign.protection import PikSignShield
        
        shield = PikSignShield()
        results = shield.verify_protection(args.image)
        
        print("\n[i] VERIFICATION SUMMARY")
        print("=" * 60)
        
        if results['stable_signature'].get('detected'):
            print("[OK] Stable Signature: DETECTED")
        else:
            print("[!] Stable Signature: WEAK")
        
        if results['spectral'].get('detected'):
            print("[OK] Spectral Watermark: DETECTED")
        else:
            print("[!] Spectral Watermark: WEAK")
        
        if results['multiband'].get('detected'):
            print("[OK] Multi-band Watermark: DETECTED")
        else:
            print("[!] Multi-band Watermark: WEAK")
        
    elif args.command == 'detect':
        try:
            from piksign.detection import PikSignDetector
            
            detector = PikSignDetector()
            
            if args.deepfake:
                results = detector.detect_deepfake(args.image)
            else:
                results = detector.detect_ai_content(args.image)
            
            print("\n[i] DETECTION RESULTS")
            print("=" * 60)
            print(f"AI Probability: {results.get('ai_probability', 0)*100:.1f}%")
            print(f"Manipulation: {results.get('manipulation_type', 'None')}")
            print(f"Deepfake: {results.get('deepfake_probability', 0)*100:.1f}%")
            
        except ImportError:
            print("[!] Detection module not yet implemented (Phase 2)")
            print("   Coming soon: AI content detection, deepfake analysis")
            sys.exit(1)


if __name__ == '__main__':
    main()
