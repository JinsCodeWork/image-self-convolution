#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Image Self-Convolution Processor

Just place your image as 'image.jpg' in this directory and run this script!
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from self_convolution import SelfConvolution

def load_and_process_image(image_path, max_size=256, preserve_aspect_ratio=True):
    """
    Load and preprocess an image for self-convolution.
    
    Args:
        image_path (str): Path to the image file
        max_size (int): Maximum size for the longer dimension
        preserve_aspect_ratio (bool): Whether to preserve original aspect ratio
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        with Image.open(image_path) as img:
            print(f"Original image size: {img.size} (W x H)")
            print(f"Original image mode: {img.mode}")
            
            # Convert to grayscale for better self-convolution results
            if img.mode != 'L':
                img = img.convert('L')
                print("Converted to grayscale")
            
            # Resize intelligently based on aspect ratio
            original_width, original_height = img.size
            
            if preserve_aspect_ratio:
                # Calculate new size preserving aspect ratio
                if max(original_width, original_height) > max_size:
                    if original_width > original_height:
                        new_width = max_size
                        new_height = int((original_height * max_size) / original_width)
                    else:
                        new_height = max_size
                        new_width = int((original_width * max_size) / original_height)
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"Resized to: {new_width} x {new_height} (preserving aspect ratio)")
                else:
                    print("Image size is acceptable, no resizing needed")
            else:
                # Force square resize (old behavior)
                target_size = (max_size, max_size)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                print(f"Resized to: {target_size} (square)")
            
            # Convert to tensor and normalize to [0, 1]
            img_array = np.array(img) / 255.0
            tensor = torch.from_numpy(img_array).float()
            
            print(f"Final tensor shape: {tensor.shape}")
            return tensor
            
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def visualize_convolution_vs_correlation(comparison_results, save_path="result.png"):
    """
    Create visualization comparing self-convolution vs auto-correlation.
    
    Args:
        comparison_results (dict): Results from compare_convolution_correlation
        save_path (str): Path to save the visualization
    """
    original = comparison_results['original']
    correlation = comparison_results['auto_correlation']
    convolution = comparison_results['self_convolution']
    difference = comparison_results['difference']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Self-Convolution (Autoconvolution) vs Auto-Correlation Comparison', fontsize=16, fontweight='bold')
    
    # Handle size mismatch
    def get_display_tensors(*tensors):
        if not all(t.shape == tensors[0].shape for t in tensors):
            min_h = min(t.shape[0] for t in tensors)
            min_w = min(t.shape[1] for t in tensors)
            return [t[:min_h, :min_w] for t in tensors]
        return list(tensors)
    
    orig_disp, corr_disp, conv_disp, diff_disp = get_display_tensors(original, correlation, convolution, difference)
    
    # Row 1: Original, Auto-correlation, Self-convolution
    axes[0, 0].imshow(orig_disp.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(corr_disp.cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Auto-Correlation\n(What PyTorch conv2d does)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(conv_disp.cpu().numpy(), cmap='gray')
    axes[0, 2].set_title('Self-Convolution (Autoconvolution)\n(With kernel flipping)')
    axes[0, 2].axis('off')
    
    # Row 2: Enhanced views
    axes[1, 0].imshow(corr_disp.cpu().numpy(), cmap='plasma')
    axes[1, 0].set_title('Auto-Correlation (Plasma)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(conv_disp.cpu().numpy(), cmap='plasma')
    axes[1, 1].set_title('Self-Convolution/Autoconvolution (Plasma)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_disp.cpu().numpy(), cmap='hot')
    axes[1, 2].set_title(f'Absolute Difference\nMax: {comparison_results["max_difference"]:.4f}')
    axes[1, 2].axis('off')
    
    # Row 3: Statistics and explanation
    # Statistics for correlation
    axes[2, 0].text(0.1, 0.9, 'Auto-Correlation Stats:', fontsize=12, fontweight='bold')
    axes[2, 0].text(0.1, 0.8, f'Min: {corr_disp.min():.3f}', fontsize=10)
    axes[2, 0].text(0.1, 0.7, f'Max: {corr_disp.max():.3f}', fontsize=10)
    axes[2, 0].text(0.1, 0.6, f'Mean: {corr_disp.mean():.3f}', fontsize=10)
    axes[2, 0].text(0.1, 0.4, 'Formula:', fontsize=11, fontweight='bold')
    axes[2, 0].text(0.1, 0.3, '∑ f(x+u,y+v) × f(u,v)', fontsize=10, family='monospace')
    axes[2, 0].text(0.1, 0.1, '(No kernel flipping)', fontsize=10, style='italic')
    axes[2, 0].set_xlim(0, 1)
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].axis('off')
    
    # Statistics for convolution
    axes[2, 1].text(0.1, 0.9, 'Self-Convolution/Autoconvolution:', fontsize=12, fontweight='bold')
    axes[2, 1].text(0.1, 0.8, f'Min: {conv_disp.min():.3f}', fontsize=10)
    axes[2, 1].text(0.1, 0.7, f'Max: {conv_disp.max():.3f}', fontsize=10)
    axes[2, 1].text(0.1, 0.6, f'Mean: {conv_disp.mean():.3f}', fontsize=10)
    axes[2, 1].text(0.1, 0.4, 'Formula:', fontsize=11, fontweight='bold')
    axes[2, 1].text(0.1, 0.3, '∑ f(x+u,y+v) × f(-u,-v)', fontsize=10, family='monospace')
    axes[2, 1].text(0.1, 0.1, '(With kernel flipping)', fontsize=10, style='italic')
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    
    # Comparison summary
    identical = comparison_results['are_identical']
    axes[2, 2].text(0.1, 0.9, 'Comparison Summary:', fontsize=12, fontweight='bold')
    axes[2, 2].text(0.1, 0.8, f'Identical results: {"Yes" if identical else "No"}', fontsize=11)
    axes[2, 2].text(0.1, 0.7, f'Max difference: {comparison_results["max_difference"]:.6f}', fontsize=11)
    axes[2, 2].text(0.1, 0.6, f'Mean difference: {comparison_results["mean_difference"]:.6f}', fontsize=11)
    
    if identical:
        axes[2, 2].text(0.1, 0.4, 'Your image is likely symmetric\nor near-symmetric!', fontsize=11, 
                       color='green', fontweight='bold')
    else:
        axes[2, 2].text(0.1, 0.4, 'Asymmetric image shows\nclear differences between\nconvolution and correlation', 
                       fontsize=11, color='red', fontweight='bold')
    
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison results saved to: {save_path}")

def main():
    """Main function to process user's image."""
    print("🖼️  Simple Image Self-Convolution Processor  🖼️")
    print("=" * 55)
    
    image_file = "image.jpg"
    
    if not os.path.exists(image_file):
        print(f"❌ Image file '{image_file}' not found!")
        print("\n📝 Instructions:")
        print(f"1. Place your image file as '{image_file}' in this directory")
        print("2. Supported formats: JPG, PNG, BMP, TIFF, etc.")
        print("3. Run this script again")
        print("\n💡 Your image will be automatically:")
        print("   - Converted to grayscale for optimal processing")
        print("   - Resized intelligently (preserving aspect ratio)")
        print("   - Maximum dimension limited to 256 pixels for efficiency")
        return
    
    print(f"✅ Found image: {image_file}")
    print("🔄 Processing image with self-convolution...")
    
    try:
        # Load and preprocess image (preserving aspect ratio)
        image_tensor = load_and_process_image(image_file, max_size=256, preserve_aspect_ratio=True)
        
        if image_tensor is None:
            print("❌ Failed to load image. Please check if the file is valid.")
            return
        
        # Initialize processor
        processor = SelfConvolution()
        
        # Compare self-convolution vs auto-correlation
        print(f"\n🔍 Comparing self-convolution (autoconvolution) vs auto-correlation...")
        comparison = processor.compare_convolution_correlation(image_tensor, normalize=True)
        
        # Print detailed statistics
        print(f"\n📊 Detailed Analysis:")
        print(f"Original image - Min: {image_tensor.min():.3f}, Max: {image_tensor.max():.3f}, Mean: {image_tensor.mean():.3f}")
        print(f"\nAuto-correlation (PyTorch conv2d):")
        print(f"  Min: {comparison['auto_correlation'].min():.3f}, Max: {comparison['auto_correlation'].max():.3f}, Mean: {comparison['auto_correlation'].mean():.3f}")
        print(f"\nSelf-convolution/Autoconvolution (with kernel flip):")
        print(f"  Min: {comparison['self_convolution'].min():.3f}, Max: {comparison['self_convolution'].max():.3f}, Mean: {comparison['self_convolution'].mean():.3f}")
        print(f"\nDifference Analysis:")
        print(f"  Max difference: {comparison['max_difference']:.6f}")
        print(f"  Mean difference: {comparison['mean_difference']:.6f}")
        print(f"  Results identical: {'Yes' if comparison['are_identical'] else 'No'}")
        
        # Create comprehensive visualization
        output_file = "convolution_vs_correlation_comparison.png"
        visualize_convolution_vs_correlation(comparison, save_path=output_file)
        
        print(f"\n🎉 Analysis complete!")
        print(f"\n📁 Output file: {output_file}")
        print("\n🔍 What we discovered:")
        print("   ✅ Auto-correlation: What PyTorch conv2d actually computes")
        print("   ✅ Self-convolution/Autoconvolution: Mathematical convolution (with kernel flipping)")
        
        if comparison['are_identical']:
            print("   🎯 Your image shows identical results - likely symmetric!")
        else:
            print("   🎯 Clear differences observed - your image is asymmetric!")
            
        print("   📚 See visualization for detailed mathematical comparison")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Please make sure your image file is valid and try again.")

if __name__ == "__main__":
    main()