# Enhanced StyleGAN-3 Implementation

This document describes the enhanced implementation of StyleGAN-3 included in the AI Avatar Creation Module. This implementation extends the base StyleGAN-3 model with advanced features for higher quality face generation and better control over the generated results.

## Overview

The `EnhancedStyleGAN3` class builds upon the `CustomStyleGAN3` class and adds the following advanced features:

1. **Disentangled Attribute Control**: Precisely control specific facial attributes like age, gender, expression, etc.
2. **Style Mixing at Different Resolutions**: Apply style mixing across specific layer ranges for controlled feature transfer.
3. **Truncation Trick**: Control the trade-off between quality and diversity in generated images.
4. **Advanced Projection with Identity Preservation**: Optimize latent vectors to match target images while preserving identity.
5. **Latent Space Exploration Tools**: Generate variations and analyze the latent space neighborhood.

## Installation

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Initialization

```python
from app.avatar_creation import EnhancedStyleGAN3

# Initialize with default parameters
stylegan = EnhancedStyleGAN3()

# Initialize with custom parameters
stylegan = EnhancedStyleGAN3(
    model_path="path/to/stylegan3_model.pt",
    latent_dim=512,
    image_resolution=1024,
    use_custom_extensions=True,
    attribute_model_path="path/to/attribute_model.pt"
)
```

### Generate Images with Truncation

Truncation controls the trade-off between image quality and diversity. Lower truncation values (closer to 0) produce more average-looking but higher quality faces, while higher values (closer to 1) produce more diverse but potentially lower quality results.

```python
# Generate with default truncation (0.7)
result = stylegan.generate_with_truncation()
image = result["image"]

# Generate with custom truncation
high_quality_result = stylegan.generate_with_truncation(truncation_psi=0.5)
diverse_result = stylegan.generate_with_truncation(truncation_psi=1.0)
```

### Control Specific Attributes

Modify specific facial attributes with fine-grained control:

```python
# Generate base image
result = stylegan.generate_with_truncation()
latent = result["latent"]

# Modify age
younger = stylegan.control_attribute(latent, "age", strength=-1.0)
older = stylegan.control_attribute(latent, "age", strength=1.0)

# Add a smile
smiling = stylegan.control_attribute(latent, "smile", strength=1.0)

# Change pose
turn_left = stylegan.control_attribute(latent, "pose_yaw", strength=-1.0)
turn_right = stylegan.control_attribute(latent, "pose_yaw", strength=1.0)
```

### Advanced Style Mixing

Mix styles from two different latents at specific layer ranges:

```python
latent1 = stylegan.generate_with_truncation()["latent"]
latent2 = stylegan.generate_with_truncation()["latent"]

# Mix coarse styles (pose, shape) from latent2 into latent1
mixed_image = stylegan.style_mixing_advanced(
    latent1, 
    latent2, 
    style_ranges=[(0, 4)]  # Mix only in the first 4 layers
)

# Mix fine styles (colors, details) from latent2 into latent1
mixed_image = stylegan.style_mixing_advanced(
    latent1, 
    latent2, 
    style_ranges=[(8, 14)]  # Mix only in the later layers
)
```

### Project Real Images to Latent Space

Optimize a latent vector to match a target image while preserving identity:

```python
from PIL import Image
import numpy as np

# Load target image
target_image = np.array(Image.open("path/to/image.jpg"))

# Project to latent space
projection_result = stylegan.advanced_projection(
    target_image,
    num_iterations=1000,
    regularization_strength=0.1,
    preserve_identity=True,
    preserve_background=False
)

# Get the optimized image and latent
optimized_image = projection_result["optimized_image"]
optimized_latent = projection_result["optimized_latent"]

# Save for later use
stylegan.save_image(optimized_image, "optimized_image.png")
```

### Latent Space Exploration

Generate variations of a face by exploring its neighborhood in latent space:

```python
# Generate base image
base_result = stylegan.generate_with_truncation()
base_latent = base_result["latent"]

# Generate 5 variations with moderate strength
variations = stylegan.latent_space_exploration(
    base_latent,
    num_variations=5,
    variation_strength=0.5
)

# Save variations
for i, var_img in enumerate(variations):
    Image.fromarray(var_img).save(f"variation_{i}.png")
```

### Generate Expression Sequences

Create a sequence of images with progressively stronger expressions:

```python
# Generate neutral face
neutral_result = stylegan.generate_with_truncation()
neutral_latent = neutral_result["latent"]

# Generate smile sequence (10 frames with increasing smile strength)
smile_sequence = stylegan.generate_expression_sequence(
    neutral_latent,
    expression="smile",
    num_frames=10,
    max_strength=1.5
)

# Generate surprised sequence
surprised_sequence = stylegan.generate_expression_sequence(
    neutral_latent,
    expression="eyes_open",
    num_frames=10,
    max_strength=1.0
)
```

### Create Morphing Sequences

Generate smooth transitions between two faces:

```python
# Generate two random faces
face1_latent = stylegan.generate_with_truncation()["latent"]
face2_latent = stylegan.generate_with_truncation()["latent"]

# Create a morphing sequence (30 frames)
frames = stylegan.create_morph_sequence(
    face1_latent,
    face2_latent,
    num_frames=30,
    output_dir="morph_sequence",
    create_video=True
)
```

## Demo

The module includes a comprehensive demo that showcases all the enhanced StyleGAN-3 features:

```bash
python -m app.avatar_creation.demo --mode stylegan_demo --output_dir ./stylegan_output --quality high
```

This will generate various examples of the enhanced features and save them to the specified output directory.

## Technical Details

### Attribute Directions

The disentangled attribute control is implemented using directions in the latent space. Each attribute (age, gender, smile, etc.) corresponds to a specific direction. Moving along these directions changes the corresponding attribute in the generated image.

For production use, these directions should be properly learned from a labeled dataset. In this implementation, we provide placeholder directions that demonstrate the concept but may not be optimal.

### Identity Preservation

The identity preservation feature uses a simple face recognition model to ensure that the optimized latent produces an image that maintains the same identity as the target image. This is particularly useful for applications like avatar creation where preserving the person's identity is crucial.

### Truncation Trick

The truncation trick is a technique used in StyleGAN to control the trade-off between image quality and diversity. It involves interpolating between the generated latent vector and the average latent vector. The truncation parameter (psi) controls the strength of this interpolation.

## Limitations and Future Work

1. **Pre-trained Models**: This implementation requires pre-trained StyleGAN-3 models to be fully functional. Due to size constraints, these models are not included.

2. **Attribute Directions**: The attribute directions used for disentangled control are placeholders and should be properly learned from labeled data for optimal results.

3. **Performance**: Some operations, particularly advanced projection, can be computationally intensive. Consider using a GPU for better performance.

4. **Facial Recognition**: The identity preservation feature uses a simple face recognition model. For production use, consider using more advanced models like ArcFace or FaceNet.

## License

See the main project license file for details. 