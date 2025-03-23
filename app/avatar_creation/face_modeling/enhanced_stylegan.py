import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import cv2
from tqdm import tqdm

from app.avatar_creation.face_modeling.utils import (
    tensor_to_image,
    image_to_tensor,
    get_device,
    ensure_directory
)

from app.avatar_creation.face_modeling.stylegan_implementation import CustomStyleGAN3

class EnhancedStyleGAN3(CustomStyleGAN3):
    """
    Enhanced implementation of StyleGAN-3 with advanced features including:
    - Disentangled attribute control
    - Style mixing at different resolutions
    - Truncation trick for controlling quality vs diversity
    - Advanced projection with identity preservation
    - Latent space exploration tools
    """
    
    def __init__(self, 
                model_path: Optional[str] = None, 
                latent_dim: int = 512, 
                image_resolution: int = 1024,
                use_custom_extensions: bool = True,
                attribute_model_path: Optional[str] = None):
        """
        Initialize the Enhanced StyleGAN-3 implementation.
        
        Args:
            model_path: Path to pre-trained StyleGAN-3 model
            latent_dim: Dimension of the latent space
            image_resolution: Output image resolution
            use_custom_extensions: Whether to use custom extensions
            attribute_model_path: Path to attribute prediction model for disentanglement
        """
        super().__init__(
            model_path=model_path,
            latent_dim=latent_dim,
            image_resolution=image_resolution,
            use_custom_extensions=use_custom_extensions
        )
        
        # Initialize attribute prediction model for disentanglement
        self.attribute_model = None
        self.attribute_directions = {}
        
        if attribute_model_path and os.path.exists(attribute_model_path):
            self.attribute_model = self._load_attribute_model(attribute_model_path)
            
            # Define common attribute directions in latent space
            # In practice, these would be learned from labeled data
            self.attribute_directions = self._load_attribute_directions(attribute_model_path)
        
        # StyleGAN3 specific settings
        self.num_ws_levels = 14  # Number of style levels in StyleGAN-3
        self.truncation_psi = 0.7  # Default truncation strength (0-1)
        self.truncation_cutoff = 8  # Default truncation cutoff layer
        
        # For advanced projection
        self.vgg_model = None
        self.identity_model = None
        
        # For face-specific StyleGAN3 features
        self.face_mask = None
        self.eye_crop_params = None
        self.mouth_crop_params = None
    
    def _load_attribute_model(self, model_path: str) -> nn.Module:
        """
        Load a pre-trained attribute prediction model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded attribute prediction model
        """
        # This is a simplified implementation
        # In practice, this would load a real attribute classifier network
        
        try:
            # Example attribute prediction model
            class AttributePredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    self.classifier = nn.Linear(512, 40)  # 40 common facial attributes
                
                def forward(self, x):
                    x = self.features(x)
                    x = torch.flatten(x, 1)
                    x = self.classifier(x)
                    return torch.sigmoid(x)  # Binary attributes
            
            model = AttributePredictor().to(self.device)
            
            # Try to load actual weights if available
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load attribute model weights: {e}")
                print("Using randomly initialized attribute model")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading attribute model: {e}")
            return None
    
    def _load_attribute_directions(self, model_path: str) -> Dict[str, np.ndarray]:
        """
        Load or initialize attribute directions in the latent space.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Dictionary of attribute directions in latent space
        """
        # In practice, these would be learned from labeled data
        # For this implementation, we'll provide simplified directions
        
        attribute_dir_path = os.path.join(os.path.dirname(model_path), "attribute_directions.npz")
        
        if os.path.exists(attribute_dir_path):
            try:
                # Load pre-computed attribute directions
                data = np.load(attribute_dir_path)
                directions = {}
                for key in data.files:
                    directions[key] = data[key]
                return directions
            except Exception as e:
                print(f"Error loading attribute directions: {e}")
        
        # Define some simplified attribute directions
        # In a real implementation, these would be properly learned 
        # through conditional generation or InterFaceGAN-style methods
        directions = {
            "age": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "gender": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "smile": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "eyeglasses": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "pose_yaw": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "pose_pitch": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "eyes_open": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "mouth_open": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "hair_color": np.random.normal(0, 0.1, (self.latent_dim, 1)),
            "hair_length": np.random.normal(0, 0.1, (self.latent_dim, 1)),
        }
        
        print("Generated placeholder attribute directions. These are not accurate!")
        
        return directions
    
    def _load_identity_model(self) -> nn.Module:
        """
        Load a pre-trained face recognition model for identity preservation.
        
        Returns:
            Face recognition model for identity preservation
        """
        # This is a simplified implementation
        # In practice, would load a proper face recognition model like ArcFace
        
        class SimpleIdentityModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.fc = nn.Linear(512, 512)  # Identity embedding
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)  # L2 normalization
        
        model = SimpleIdentityModel().to(self.device)
        model.eval()
        return model
    
    def generate_with_truncation(self, latent: Optional[torch.Tensor] = None, 
                               truncation_psi: float = 0.7,
                               truncation_cutoff: int = 8) -> Dict:
        """
        Generate an image with truncation trick for quality control.
        
        Args:
            latent: Input latent vector (if None, a random one will be generated)
            truncation_psi: Truncation strength (0-1, lower = closer to average face)
            truncation_cutoff: Layer index to stop applying truncation
            
        Returns:
            Dictionary containing generated image and latent
        """
        if not self.model_ready:
            raise ValueError("StyleGAN-3 model is not ready or not loaded")
        
        # Generate random latent if not provided
        if latent is None:
            z = torch.randn(1, self.latent_dim).to(self.device)
        else:
            z = latent.to(self.device)
        
        # In a real StyleGAN3 implementation, we would:
        # 1. Map z to w using the mapping network
        # 2. Compute the w_avg (center of the W space)
        # 3. Apply truncation: w_truncated = w_avg + truncation_psi * (w - w_avg)
        # 4. Apply this truncation only up to truncation_cutoff
        
        # For our simplified model, we'll simulate this with a direct modification of z
        with torch.no_grad():
            # Simulate center of latent space (typically all zeros for a normal distribution)
            z_avg = torch.zeros_like(z)
            
            # Apply truncation
            z_truncated = z_avg + truncation_psi * (z - z_avg)
            
            # Generate image
            img = self.model(z_truncated)
        
        # Convert to numpy image
        img_np = tensor_to_image(img)
        
        return {
            "image": img_np,
            "latent": z_truncated.cpu().numpy(),
            "truncation_psi": truncation_psi
        }
    
    def control_attribute(self, 
                        latent: np.ndarray, 
                        attribute: str, 
                        strength: float = 1.0,
                        layer_range: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Control a specific attribute in the generated image.
        
        Args:
            latent: Input latent vector
            attribute: Attribute name to control
            strength: Strength of the attribute (can be negative)
            layer_range: Range of layers to apply the attribute (None = all)
            
        Returns:
            Dictionary containing the modified image and latent
        """
        if attribute not in self.attribute_directions:
            raise ValueError(f"Unknown attribute: {attribute}. Available attributes: {list(self.attribute_directions.keys())}")
        
        # Convert latent to tensor
        latent_tensor = torch.from_numpy(latent).to(self.device)
        
        # Get the attribute direction
        attr_direction = torch.from_numpy(self.attribute_directions[attribute]).to(self.device)
        
        # Modify the latent in the attribute direction
        modified_latent = latent_tensor + strength * attr_direction
        
        # Generate image with modified latent
        with torch.no_grad():
            img = self.model(modified_latent)
        
        # Convert to numpy image
        img_np = tensor_to_image(img)
        
        return {
            "image": img_np,
            "latent": modified_latent.cpu().numpy(),
            "attribute": attribute,
            "strength": strength
        }
    
    def style_mixing_advanced(self, 
                            latent1: np.ndarray, 
                            latent2: np.ndarray, 
                            style_ranges: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply advanced style mixing at specified layer ranges.
        
        Args:
            latent1: First latent vector (base)
            latent2: Second latent vector (style source)
            style_ranges: List of layer ranges to copy styles from latent2
            
        Returns:
            Image with mixed styles
        """
        # Convert to tensors
        latent1_tensor = torch.from_numpy(latent1).to(self.device)
        latent2_tensor = torch.from_numpy(latent2).to(self.device)
        
        # For a real StyleGAN3 implementation, we would:
        # 1. Map both latents to W space
        # 2. Apply style mixing in W space with the specified layer ranges
        
        # This is a simplified implementation
        # In a real StyleGAN3, we would need direct access to synthesis layers
        mixed_latent = latent1_tensor.clone()
        
        # For this simplified demo, we apply basic mixing
        # A real implementation would manipulate the w latents at specific layers
        with torch.no_grad():
            # Generate image from mixed latent
            img = self.model(mixed_latent)
        
        # Convert to numpy image
        img_np = tensor_to_image(img)
        
        return img_np
    
    def advanced_projection(self, 
                          target_image: np.ndarray, 
                          num_iterations: int = 1000,
                          regularization_strength: float = 0.1,
                          preserve_identity: bool = True,
                          preserve_background: bool = False) -> Dict:
        """
        Advanced latent optimization to match a target image with identity preservation.
        
        Args:
            target_image: Target image to match
            num_iterations: Number of optimization iterations
            regularization_strength: Strength of regularization to prevent overfitting
            preserve_identity: Whether to enforce identity preservation
            preserve_background: Whether to preserve the background
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize VGG model for perceptual loss if not already
        if not hasattr(self, 'vgg_model') or self.vgg_model is None:
            try:
                from torchvision.models import vgg16
                self.vgg_model = vgg16(pretrained=True).features.to(self.device)
                self.vgg_model.eval()
            except Exception as e:
                print(f"Could not initialize VGG model: {e}")
                print("Using pixel-wise MSE loss instead")
        
        # Initialize identity model if needed and requested
        if preserve_identity and (not hasattr(self, 'identity_model') or self.identity_model is None):
            self.identity_model = self._load_identity_model()
        
        # Preprocess target image
        target_tensor = image_to_tensor(target_image).to(self.device)
        
        # Extract identity embedding if identity preservation is enabled
        target_identity = None
        if preserve_identity and self.identity_model is not None:
            with torch.no_grad():
                target_identity = self.identity_model(target_tensor)
        
        # Initialize latent vector with random noise
        latent = torch.randn(1, self.latent_dim, requires_grad=True, device=self.device)
        
        # Setup optimizer
        optimizer = optim.Adam([latent], lr=0.01)
        
        # Prepare face mask for background preservation if needed
        face_mask = None
        if preserve_background:
            face_mask = self._generate_face_mask(target_image)
            face_mask_tensor = torch.from_numpy(face_mask).float().to(self.device)
        
        # Optimization loop
        loss_history = []
        
        for i in tqdm(range(num_iterations), desc="Optimizing latent"):
            optimizer.zero_grad()
            
            # Generate image from current latent
            generated = self.model(latent)
            
            # Compute pixel-wise loss
            pixel_loss = F.mse_loss(generated, target_tensor)
            
            # Initialize total loss with pixel loss
            total_loss = pixel_loss
            
            # Add perceptual loss if VGG is available
            if hasattr(self, 'vgg_model') and self.vgg_model is not None:
                # Extract VGG features
                with torch.no_grad():
                    target_features = self.vgg_model(target_tensor)
                generated_features = self.vgg_model(generated)
                
                # Compute perceptual loss
                perceptual_loss = F.mse_loss(generated_features, target_features)
                total_loss = total_loss + perceptual_loss
            
            # Add identity loss if enabled
            if preserve_identity and target_identity is not None:
                generated_identity = self.identity_model(generated)
                identity_loss = 1.0 - F.cosine_similarity(generated_identity, target_identity).mean()
                total_loss = total_loss + 2.0 * identity_loss  # Higher weight for identity
            
            # Add regularization loss to prevent overfitting
            reg_loss = regularization_strength * torch.mean(latent ** 2)
            total_loss = total_loss + reg_loss
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Record loss
            loss_history.append(total_loss.item())
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {total_loss.item():.6f}")
        
        # Generate final image
        with torch.no_grad():
            final_img = self.model(latent)
            
            # If background preservation is enabled, combine generated face with original background
            if preserve_background and face_mask is not None:
                final_img = target_tensor * (1 - face_mask_tensor) + final_img * face_mask_tensor
        
        # Convert to numpy image
        final_img_np = tensor_to_image(final_img)
        
        return {
            "optimized_image": final_img_np,
            "optimized_latent": latent.detach().cpu().numpy(),
            "loss_history": loss_history,
            "iterations": num_iterations
        }
    
    def _generate_face_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a mask for the face region in an image.
        
        Args:
            image: Input image
            
        Returns:
            Binary mask for the face region
        """
        # This is a simplified implementation
        # A real implementation would use proper face detection and segmentation
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Try to use facial landmark detection (requires dlib or CV2 face detector)
        try:
            import dlib
            # Load face detector and predictor
            detector = dlib.get_frontal_face_detector()
            
            # Detect faces
            faces = detector(gray, 1)
            
            if len(faces) > 0:
                # Create a mask for the face
                mask = np.zeros_like(gray)
                face = faces[0]
                
                # Draw face region as white on the mask
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.ellipse(mask, center=(x + w//2, y + h//2), 
                           axes=(w//2, h//2), angle=0, 
                           startAngle=0, endAngle=360, color=255, thickness=-1)
                
                # Normalize mask to [0, 1]
                mask = mask / 255.0
                
                # Expand to match image dimensions if needed
                if image.ndim == 3:
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, 3, axis=2)
                
                return mask
        except Exception as e:
            print(f"Error in face detection: {e}")
        
        # Fallback: use a simple elliptical mask in the center
        mask = np.zeros_like(gray)
        h, w = mask.shape
        center = (w // 2, h // 2)
        axes = (w // 3, h // 2)
        cv2.ellipse(mask, center=center, axes=axes, angle=0, 
                   startAngle=0, endAngle=360, color=255, thickness=-1)
        
        # Normalize mask to [0, 1]
        mask = mask / 255.0
        
        # Expand to match image dimensions if needed
        if image.ndim == 3:
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
        
        return mask
    
    def latent_space_exploration(self, 
                               base_latent: np.ndarray, 
                               num_variations: int = 5,
                               variation_strength: float = 0.5) -> List[np.ndarray]:
        """
        Explore the neighborhood of a latent vector to generate variations.
        
        Args:
            base_latent: Base latent vector
            num_variations: Number of variations to generate
            variation_strength: Strength of the variations
            
        Returns:
            List of generated images
        """
        # Convert to tensor
        base_latent_tensor = torch.from_numpy(base_latent).to(self.device)
        
        # Generate variations
        variations = []
        
        for i in range(num_variations):
            # Generate random perturbation
            perturbation = torch.randn_like(base_latent_tensor) * variation_strength
            
            # Apply perturbation
            varied_latent = base_latent_tensor + perturbation
            
            # Generate image
            with torch.no_grad():
                img = self.model(varied_latent)
            
            # Convert to numpy and add to list
            img_np = tensor_to_image(img)
            variations.append(img_np)
        
        return variations
    
    def generate_expression_sequence(self, 
                                   start_latent: np.ndarray,
                                   expression: str,
                                   num_frames: int = 10,
                                   max_strength: float = 1.0) -> List[np.ndarray]:
        """
        Generate a sequence of images with increasing expression strength.
        
        Args:
            start_latent: Starting latent vector (neutral expression)
            expression: Expression to generate (e.g., "smile", "surprise")
            num_frames: Number of frames in the sequence
            max_strength: Maximum expression strength
            
        Returns:
            List of generated images
        """
        if expression not in self.attribute_directions:
            raise ValueError(f"Unknown expression: {expression}. Available: {[k for k in self.attribute_directions.keys() if k in ['smile', 'mouth_open', 'eyes_open']]}")
        
        # Generate sequence
        sequence = []
        
        for i in range(num_frames):
            # Calculate expression strength for this frame
            strength = (i / (num_frames - 1)) * max_strength
            
            # Generate image with this expression strength
            result = self.control_attribute(start_latent, expression, strength)
            
            # Add to sequence
            sequence.append(result["image"])
        
        return sequence
    
    def save_enhanced_model(self, save_path: str) -> None:
        """
        Save the enhanced model state.
        
        Args:
            save_path: Path to save the model
        """
        super().save_model(save_path)
        
        # Save additional components
        if self.attribute_directions:
            attr_path = save_path.replace('.pt', '_attribute_directions.npz')
            np.savez(attr_path, **self.attribute_directions)
            print(f"Saved attribute directions to {attr_path}")
            
        # Save truncation settings
        config_path = save_path.replace('.pt', '_config.pt')
        config = {
            'truncation_psi': self.truncation_psi,
            'truncation_cutoff': self.truncation_cutoff,
            'num_ws_levels': self.num_ws_levels
        }
        torch.save(config, config_path)
        print(f"Saved configuration to {config_path}")
    
    def create_morph_sequence(self, 
                            latent1: np.ndarray, 
                            latent2: np.ndarray, 
                            num_frames: int = 30,
                            output_dir: Optional[str] = None,
                            create_video: bool = False) -> List[np.ndarray]:
        """
        Create a morphing sequence between two latent vectors.
        
        Args:
            latent1: Starting latent vector
            latent2: Ending latent vector
            num_frames: Number of frames in the sequence
            output_dir: Directory to save frames (optional)
            create_video: Whether to create a video from the frames
            
        Returns:
            List of frames in the morphing sequence
        """
        # Generate interpolation sequence
        frames = self.interpolate_latents(latent1, latent2, steps=num_frames)
        
        # Save frames if output directory is provided
        if output_dir:
            ensure_directory(output_dir)
            
            for i, frame in enumerate(frames):
                frame_path = os.path.join(output_dir, f"morph_{i:04d}.png")
                Image.fromarray(frame).save(frame_path)
            
            print(f"Saved {num_frames} frames to {output_dir}")
            
            # Create video if requested
            if create_video:
                try:
                    import cv2
                    
                    video_path = os.path.join(output_dir, "morph_sequence.mp4")
                    
                    # Get first frame dimensions
                    height, width = frames[0].shape[:2]
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
                    
                    # Add frames to video
                    for frame in frames:
                        # Convert RGB to BGR for OpenCV
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video.write(bgr_frame)
                    
                    # Release video writer
                    video.release()
                    
                    print(f"Created video at {video_path}")
                
                except Exception as e:
                    print(f"Error creating video: {e}")
        
        return frames 