#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Self-Convolution Core Module

This module provides the core self-convolution functionality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

class SelfConvolution:
    """
    A class to perform image self-convolution operations.
    """
    
    def __init__(self, device=None):
        """Initialize the SelfConvolution class."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
    
    def load_image_file(self, image_path, max_size=512, preserve_aspect_ratio=True, grayscale=True):
        """
        Load an image file and convert it to tensor format.
        
        Args:
            image_path (str): Path to the image file
            max_size (int): Maximum size for the longer dimension
            preserve_aspect_ratio (bool): Whether to preserve original aspect ratio
            grayscale (bool): Whether to convert to grayscale
            
        Returns:
            torch.Tensor: Image tensor
        """
        try:
            with Image.open(image_path) as img:
                print(f"Loaded image: {img.size} ({img.mode})")
                
                # Convert to grayscale if requested
                if grayscale and img.mode != 'L':
                    img = img.convert('L')
                elif not grayscale and img.mode == 'L':
                    img = img.convert('RGB')
                
                # Resize intelligently
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
                        print(f"Resized to: {new_width} x {new_height} (aspect ratio preserved)")
                else:
                    # Force square resize
                    target_size = (max_size, max_size)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    print(f"Resized to: {target_size} (square)")
                
                # Convert to tensor
                img_array = np.array(img) / 255.0
                if grayscale:
                    tensor = torch.from_numpy(img_array).float()
                else:
                    tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
                
                return tensor.to(self.device)
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def self_correlate(self, image, normalize=True):
        """
        Perform auto-correlation of an image (what PyTorch conv2d actually does).
        
        Args:
            image (torch.Tensor): Input image
            normalize (bool): Whether to normalize the kernel
            
        Returns:
            torch.Tensor: Auto-correlated image
        """
        if len(image.shape) == 2:
            # Add batch and channel dimensions
            image_4d = image.unsqueeze(0).unsqueeze(0)
        else:
            image_4d = image
            
        # Use the image itself as the correlation kernel (no flipping)
        kernel = image.clone()
        
        if normalize:
            # Normalize the kernel to prevent amplitude explosion
            kernel = kernel / torch.sum(torch.abs(kernel))
            
        # Add dimensions for convolution
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        # Perform correlation with 'same' padding to maintain size
        padding_h = (kernel.shape[0] - 1) // 2
        padding_w = (kernel.shape[1] - 1) // 2
        result = F.conv2d(image_4d, kernel_4d, padding=(padding_h, padding_w))
        
        # Remove batch and channel dimensions
        return result.squeeze()
    
    def self_convolve(self, image, normalize=True):
        """
        Perform true self-convolution of an image (with kernel flipping).
        
        Args:
            image (torch.Tensor): Input image
            normalize (bool): Whether to normalize the kernel
            
        Returns:
            torch.Tensor: Self-convolved image
        """
        if len(image.shape) == 2:
            # Add batch and channel dimensions
            image_4d = image.unsqueeze(0).unsqueeze(0)
        else:
            image_4d = image
            
        # Use the image itself as the convolution kernel
        kernel = image.clone()
        
        if normalize:
            # Normalize the kernel to prevent amplitude explosion
            kernel = kernel / torch.sum(torch.abs(kernel))
            
        # CRITICAL: True convolution requires flipping the kernel 180°
        kernel = torch.flip(kernel, dims=(-2, -1))  # Flip H and W dimensions
        
        # Add dimensions for convolution
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        # Perform convolution with 'same' padding to maintain size
        padding_h = (kernel.shape[0] - 1) // 2
        padding_w = (kernel.shape[1] - 1) // 2
        result = F.conv2d(image_4d, kernel_4d, padding=(padding_h, padding_w))
        
        # Remove batch and channel dimensions
        return result.squeeze()
    
    def compare_convolution_correlation(self, image, normalize=True):
        """
        Compare self-convolution vs auto-correlation on the same image.
        
        Args:
            image (torch.Tensor): Input image
            normalize (bool): Whether to normalize the kernel
            
        Returns:
            dict: Results containing both operations and statistics
        """
        correlation_result = self.self_correlate(image, normalize)
        convolution_result = self.self_convolve(image, normalize)
        
        # Calculate difference
        difference = torch.abs(convolution_result - correlation_result)
        
        return {
            'original': image,
            'auto_correlation': correlation_result,
            'self_convolution': convolution_result,
            'difference': difference,
            'max_difference': difference.max().item(),
            'mean_difference': difference.mean().item(),
            'are_identical': torch.allclose(correlation_result, convolution_result, atol=1e-6)
        }
    
    def analyze_image(self, image):
        """
        Analyze image statistics and perform self-convolution.
        
        Args:
            image (torch.Tensor): Input image
            
        Returns:
            dict: Analysis results including original image, result, and statistics
        """
        print(f"=== Image Analysis ===")
        print(f"Input image shape: {image.shape}")
        
        # Perform self-convolution
        result = self.self_convolve(image, normalize=True)
        print(f"Output image shape: {result.shape}")
        
        # Calculate statistics
        print(f"\n=== Statistics ===")
        print(f"Original - min: {image.min():.4f}, max: {image.max():.4f}, mean: {image.mean():.4f}")
        print(f"Result - min: {result.min():.4f}, max: {result.max():.4f}, mean: {result.mean():.4f}")
        
        return {
            'original': image,
            'result': result,
            'original_stats': {
                'min': image.min().item(),
                'max': image.max().item(),
                'mean': image.mean().item()
            },
            'result_stats': {
                'min': result.min().item(),
                'max': result.max().item(),
                'mean': result.mean().item()
            }
        }