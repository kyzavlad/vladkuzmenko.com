import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from PIL import Image

from app.avatar_creation.face_modeling.utils import (
    load_image,
    tensor_to_image,
    image_to_tensor,
    get_device
)

class IdentityVerification:
    """
    Class for verifying identity consistency between input images and generated 3D models.
    Uses facial recognition and embedding techniques to ensure the 3D model preserves identity.
    """
    
    def __init__(self, model_path: Optional[str] = None, similarity_threshold: float = 0.7):
        """
        Initialize the identity verification system.
        
        Args:
            model_path: Path to pre-trained model (if available)
            similarity_threshold: Threshold for identity matching
        """
        self.device = get_device()
        self.similarity_threshold = similarity_threshold
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load face recognition model if available
        # If a specific model path is not provided, we'll use a simpler comparison approach
        self.face_recognition_model = self._load_face_model(model_path)
    
    def _load_face_model(self, model_path: Optional[str]) -> Optional[torch.nn.Module]:
        """
        Load a pre-trained face recognition model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded model or None if not available
        """
        # This is a placeholder for loading a face recognition model
        # In a real implementation, load a proper face recognition model like FaceNet, ArcFace, etc.
        
        # If model_path is provided, attempt to load model
        if model_path:
            try:
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                return model
            except Exception as e:
                print(f"Failed to load face model: {e}")
                
        return None
    
    def extract_face(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face region from an image.
        
        Args:
            image: Input image
            
        Returns:
            Cropped face image
        """
        # Convert to grayscale for face detection
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect face
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face detected, return the original image
            return image
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add some padding
        padding = int(min(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop the face
        face_image = image[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_image = cv2.resize(face_image, (256, 256))
        
        return face_image
    
    def compute_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Compute a face embedding vector for identity comparison.
        
        Args:
            face_image: Face image
            
        Returns:
            Face embedding vector
        """
        # Preprocess the image for the model
        if self.face_recognition_model:
            # Convert to tensor
            face_tensor = image_to_tensor(face_image).to(self.device)
            
            # Get embedding from model
            with torch.no_grad():
                embedding = self.face_recognition_model(face_tensor)
                
            # Convert to numpy
            embedding = embedding.cpu().numpy()
            
        else:
            # If no model is available, use a simple image-based feature vector
            # This is just a placeholder for demonstration
            # In a real implementation, use a proper face recognition model
            
            # Resize to smaller dimensions for a compact representation
            small_face = cv2.resize(face_image, (32, 32))
            
            # Flatten to a feature vector
            if small_face.ndim == 3:
                embedding = small_face.flatten()
            else:
                embedding = small_face.flatten()
                
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
        return embedding
    
    def verify_identity(self, source_image: np.ndarray, target_image: np.ndarray) -> Dict:
        """
        Verify if two images represent the same identity.
        
        Args:
            source_image: Source image
            target_image: Target image for comparison
            
        Returns:
            Dictionary with verification results
        """
        # Extract faces
        source_face = self.extract_face(source_image)
        target_face = self.extract_face(target_image)
        
        # Compute embeddings
        source_embedding = self.compute_face_embedding(source_face)
        target_embedding = self.compute_face_embedding(target_face)
        
        # Compute similarity
        similarity = self._compute_similarity(source_embedding, target_embedding)
        
        # Determine if same identity
        is_same_identity = similarity >= self.similarity_threshold
        
        return {
            "similarity_score": float(similarity),
            "is_same_identity": bool(is_same_identity),
            "source_face": source_face,
            "target_face": target_face
        }
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        # Reshape if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0, 0]
        
        # Ensure within range 0-1
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def verify_model_consistency(self, 
                                original_image: np.ndarray, 
                                rendered_image: np.ndarray,
                                return_visualization: bool = False) -> Dict:
        """
        Verify identity consistency between original image and rendered 3D model.
        
        Args:
            original_image: Original input image
            rendered_image: Rendered image of the 3D model
            return_visualization: Whether to return visualization
            
        Returns:
            Dictionary with verification results
        """
        # Verify identity
        verification_result = self.verify_identity(original_image, rendered_image)
        
        # Create visualization if requested
        if return_visualization:
            # Create a side-by-side comparison
            source_face = verification_result["source_face"]
            target_face = verification_result["target_face"]
            
            # Resize to same dimensions if needed
            h, w = max(source_face.shape[0], target_face.shape[0]), source_face.shape[1] + target_face.shape[1]
            source_face_resized = cv2.resize(source_face, (source_face.shape[1], h))
            target_face_resized = cv2.resize(target_face, (target_face.shape[1], h))
            
            # Create a canvas
            visualization = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Place images side by side
            visualization[:, :source_face_resized.shape[1]] = source_face_resized
            visualization[:, source_face_resized.shape[1]:] = target_face_resized
            
            # Add text with similarity score
            text = f"Similarity: {verification_result['similarity_score']:.2f}"
            color = (0, 255, 0) if verification_result["is_same_identity"] else (255, 0, 0)
            cv2.putText(visualization, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            verification_result["visualization"] = visualization
        
        return verification_result
    
    def compute_identity_loss(self, 
                            source_embedding: torch.Tensor, 
                            target_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss for model optimization.
        
        Args:
            source_embedding: Source identity embedding
            target_embedding: Target identity embedding
            
        Returns:
            Identity loss tensor
        """
        # Normalize embeddings
        source_norm = F.normalize(source_embedding, p=2, dim=1)
        target_norm = F.normalize(target_embedding, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(source_norm * target_norm, dim=1)
        
        # Identity loss: higher similarity = lower loss
        loss = 1.0 - similarity
        
        return loss
    
    def generate_report(self, verification_result: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate a report on identity verification.
        
        Args:
            verification_result: Verification result dictionary
            save_path: Path to save the report (optional)
            
        Returns:
            Report text
        """
        # Create report text
        report = "Identity Verification Report\n"
        report += "===========================\n\n"
        report += f"Similarity Score: {verification_result['similarity_score']:.4f}\n"
        report += f"Identity Match: {'YES' if verification_result['is_same_identity'] else 'NO'}\n"
        report += f"Threshold: {self.similarity_threshold}\n\n"
        
        if verification_result['is_same_identity']:
            report += "Verification PASSED: The 3D model preserves the identity of the input image.\n"
        else:
            report += "Verification FAILED: The 3D model does not preserve the identity well enough.\n"
            report += "Recommendations:\n"
            report += "- Check if the input image has sufficient quality\n"
            report += "- Verify if the face is clearly visible in the input\n"
            report += "- Adjust feature preservation parameters\n"
            report += "- Use additional reference images\n"
        
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            
            # Save visualization if available
            if "visualization" in verification_result:
                vis_path = save_path.replace('.txt', '_vis.png')
                cv2.imwrite(vis_path, cv2.cvtColor(verification_result["visualization"], cv2.COLOR_RGB2BGR))
        
        return report
