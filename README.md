# Image Self-Convolution

## Project Overview

This project demonstrates image self-convolution using PyTorch. Self-convolution convolves an image with itself, which enhances patterns and smooths noise while preserving structural features.

## Quick Start

### Process Your Own Image
1. Place your image as `image.jpg` in the project directory
2. Run: `python process_my_image.py`
3. View results in `my_image_result.png`

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+

### Installation Steps
1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Simple Method (Recommended)
```bash
python process_my_image.py
```
This automatically:
- Loads your `image.jpg` file (any size or aspect ratio)
- Converts to grayscale for optimal results
- Intelligently resizes while preserving aspect ratio
- Limits maximum dimension to 256px for efficiency
- Performs self-convolution
- Saves detailed visualization

### Programmatic Usage
```python
from self_convolution import SelfConvolution

# Initialize processor
self_conv = SelfConvolution()

# Load and process image (preserves aspect ratio)
image = self_conv.load_image_file("your_image.jpg", max_size=256, preserve_aspect_ratio=True)
result = self_conv.self_convolve(image, normalize=True)

# Analyze results
analysis = self_conv.analyze_image(image)
```

## Self-Convolution Theory

### Important Note: Auto-correlation vs Convolution
**Technical clarification**: This implementation performs **auto-correlation**, which is what PyTorch's `conv2d` function actually computes. In the deep learning community, this operation is commonly called "convolution" even though mathematically it's cross-correlation. True mathematical convolution would require flipping the kernel, but this is rarely used in practice.

### Mathematical Definition
Our auto-correlation operation:
```
result[i,j] = Σ Σ image[i+m, j+n] × image[m,n]
```
The image serves as both input and correlation kernel.

True convolution would be:
```
result[i,j] = Σ Σ image[i+m, j+n] × image_flipped[m,n]
```

For most practical purposes in computer vision, the auto-correlation operation is what we want and expect.

### Key Effects
- **Pattern Enhancement**: Amplifies recurring structures
- **Noise Smoothing**: Reduces random variations
- **Feature Preservation**: Maintains important edges and textures
- **Structural Analysis**: Reveals underlying image patterns

### Applications
- Feature detection and pattern recognition
- Image preprocessing for computer vision
- Texture analysis and enhancement
- Artistic image effects

## Output Results

The visualization includes:

1. **Original Image**: Your input image
2. **Self-Convolution Result**: The processed image
3. **Difference Map**: Highlights areas of change
4. **Enhanced Views**: Different color mappings to reveal details
5. **Statistics**: Numerical analysis of the transformation

## File Structure

```
SelfConvolution/
├── self_convolution.py      # Core self-convolution class
├── process_my_image.py      # Simple image processor
├── requirements.txt         # Dependencies
├── README.md               # This documentation
├── image.jpg               # Your image (place here)
└── my_image_result.png     # Generated results
```

## Technical Details

- Uses PyTorch's `F.conv2d` for efficient convolution
- Automatic GPU detection and usage
- Kernel normalization prevents numerical overflow
- Smart padding maintains image dimensions
- Supports various image formats (JPG, PNG, BMP, TIFF)

## Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- TIFF
- Any format supported by PIL/Pillow

## Performance Notes

- Images can be any size or aspect ratio (rectangular, square, portrait, landscape)
- Automatic intelligent resizing preserves aspect ratio
- Maximum dimension limited to 256px for efficiency
- Grayscale conversion provides better self-convolution results
- GPU acceleration used when available
- Processing time: typically 1-5 seconds per image

## License

MIT License