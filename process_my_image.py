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

def visualize_results(original, result, save_path="result.png"):
    """
    Create visualization for the self-convolution results.
    
    Args:
        original (torch.Tensor): Original image
        result (torch.Tensor): Self-convolution result
        save_path (str): Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Your Image Self-Convolution Results', fontsize=16, fontweight='bold')
    
    # Handle size mismatch
    if result.shape != original.shape:
        min_h = min(result.shape[0], original.shape[0])
        min_w = min(result.shape[1], original.shape[1])
        result_display = result[:min_h, :min_w]
        original_display = original[:min_h, :min_w]
        diff = torch.abs(result_display - original_display)
    else:
        result_display = result
        original_display = original
        diff = torch.abs(result - original)
    
    # Original image
    axes[0, 0].imshow(original_display.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Your Original Image')
    axes[0, 0].axis('off')
    
    # Self-convolution result
    axes[0, 1].imshow(result_display.cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Self-Convolution Result')
    axes[0, 1].axis('off')
    
    # Difference map
    axes[0, 2].imshow(diff.cpu().numpy(), cmap='hot')
    axes[0, 2].set_title('Difference Map')
    axes[0, 2].axis('off')
    
    # Result with different colormap
    axes[1, 0].imshow(result_display.cpu().numpy(), cmap='plasma')
    axes[1, 0].set_title('Result (Enhanced View)')
    axes[1, 0].axis('off')
    
    # Edge enhancement visualization
    axes[1, 1].imshow(result_display.cpu().numpy(), cmap='viridis')
    axes[1, 1].set_title('Result (Viridis View)')
    axes[1, 1].axis('off')
    
    # Statistics
    axes[1, 2].text(0.1, 0.8, 'Original Image:', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, f'Min: {original_display.min():.3f}', fontsize=10)
    axes[1, 2].text(0.1, 0.6, f'Max: {original_display.max():.3f}', fontsize=10)
    axes[1, 2].text(0.1, 0.5, f'Mean: {original_display.mean():.3f}', fontsize=10)
    axes[1, 2].text(0.1, 0.3, 'Self-Convolution Result:', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.2, f'Min: {result_display.min():.3f}', fontsize=10)
    axes[1, 2].text(0.1, 0.1, f'Max: {result_display.max():.3f}', fontsize=10)
    axes[1, 2].text(0.1, 0.0, f'Mean: {result_display.mean():.3f}', fontsize=10)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to: {save_path}")

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
        
        # Initialize self-convolution processor
        self_conv = SelfConvolution()
        
        # Perform self-convolution
        result = self_conv.self_convolve(image_tensor, normalize=True)
        
        # Print statistics
        print(f"\n📊 Processing Statistics:")
        print(f"Original - Min: {image_tensor.min():.3f}, Max: {image_tensor.max():.3f}, Mean: {image_tensor.mean():.3f}")
        print(f"Result - Min: {result.min():.3f}, Max: {result.max():.3f}, Mean: {result.mean():.3f}")
        
        # Create visualization
        output_file = "my_image_result.png"
        visualize_results(image_tensor, result, save_path=output_file)
        
        print(f"\n🎉 Success! Your image has been processed.")
        print(f"\n📁 Output file: {output_file}")
        print("\n🔍 What happened:")
        print("   - Your image was convolved with itself")
        print("   - This enhances patterns and smooths noise")
        print("   - The result shows enhanced structural features")
        print("   - Different color maps reveal different aspects")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Please make sure your image file is valid and try again.")

if __name__ == "__main__":
    main()