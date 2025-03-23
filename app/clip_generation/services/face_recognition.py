"""
Face Recognition Module

This module provides face recognition capabilities using ArcFace for identity embeddings
and cosine similarity for matching.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from app.clip_generation.services.face_tracking import FaceBox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceIdentity:
    """Represents a face identity with associated embeddings."""
    id: str
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    thumbnail: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_embedding(self) -> np.ndarray:
        """Get the average embedding vector for this identity."""
        if not self.embeddings:
            raise ValueError("No embeddings available for this identity")
        return np.mean(self.embeddings, axis=0)
    
    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add a new embedding to this identity."""
        self.embeddings.append(embedding)


class ArcFaceRecognizer:
    """
    Face recognition using ArcFace embeddings.
    
    This class handles face feature extraction and matching using ArcFace
    embeddings and cosine similarity.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        recognition_threshold: float = 0.6,
        device: str = "cpu",
        db_path: Optional[str] = None
    ):
        """
        Initialize the ArcFace recognizer.
        
        Args:
            model_path: Path to the ArcFace model weights
            recognition_threshold: Threshold for face recognition
            device: Device to run inference on ('cpu' or 'cuda')
            db_path: Path to face database file (if any)
        """
        self.model_path = model_path
        self.recognition_threshold = recognition_threshold
        self.device = device
        self.db_path = db_path
        
        # Default model path if not provided
        if not self.model_path:
            self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                "arcface_resnet50.pth"
            )
        
        # Face database
        self.identities = {}  # Dict[str, FaceIdentity]
        
        # Model instance
        self.model = None
        self.embedding_size = 512  # Default for ArcFace
        
        # Load model
        self.load_model()
        
        # Load database if provided
        if self.db_path and os.path.exists(self.db_path):
            self.load_database()
    
    def load_model(self) -> None:
        """Load the ArcFace model."""
        try:
            # Try importing from different possible implementations
            try:
                # Try insightface
                import insightface
                from insightface.app import FaceAnalysis
                
                logger.info("Loading ArcFace model from insightface")
                app = FaceAnalysis(name="buffalo_l", root=".")
                app.prepare(ctx_id=0 if self.device == "cuda" else -1)
                self.model = app
                
            except ImportError:
                # Fallback to deepface
                try:
                    from deepface import DeepFace
                    from deepface.basemodels import ArcFace
                    
                    logger.info("Loading ArcFace model from deepface")
                    self.model = ArcFace.loadModel()
                    
                except ImportError:
                    # Last resort, use local implementation
                    logger.warning("No ArcFace implementation found. Using placeholder.")
                    self.model = self._get_placeholder_model()
            
            logger.info("ArcFace model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ArcFace model: {str(e)}")
            self.model = self._get_placeholder_model()
    
    def _get_placeholder_model(self):
        """
        Create a placeholder model for when real models are not available.
        This will generate random embeddings that are stable for the same input.
        """
        class PlaceholderArcFace:
            def get_embedding(self, face_img):
                # Generate a deterministic "random" embedding based on image content
                # This is just for demonstration/testing purposes
                if face_img is None or face_img.size == 0:
                    return np.zeros(512)
                
                # Use average pixel values to seed a simple embedding
                np.random.seed(int(np.sum(face_img)) % 10000)
                embedding = np.random.rand(512).astype(np.float32)
                # Normalize
                return embedding / np.linalg.norm(embedding)
        
        logger.warning("Using placeholder ArcFace model with random embeddings")
        return PlaceholderArcFace()
    
    def extract_embedding(self, frame: np.ndarray, face_box: FaceBox) -> np.ndarray:
        """
        Extract face embedding using ArcFace.
        
        Args:
            frame: Input frame
            face_box: Face bounding box
            
        Returns:
            Face embedding vector
        """
        try:
            # Extract face crop
            x1, y1, x2, y2 = map(int, [face_box.x1, face_box.y1, face_box.x2, face_box.y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x1 >= x2 or y1 >= y2:
                logger.warning("Invalid face box dimensions")
                return np.zeros(self.embedding_size)
            
            face_img = frame[y1:y2, x1:x2]
            
            # Different model APIs
            if hasattr(self.model, 'get_embedding'):
                # Custom or placeholder implementation
                embedding = self.model.get_embedding(face_img)
                
            elif hasattr(self.model, 'get_input'):
                # Insightface implementation
                faces = self.model.get(frame, det_size=(224, 224))
                if not faces:
                    return np.zeros(self.embedding_size)
                    
                # Find the face that best matches our detection
                best_face = self._find_matching_face(faces, face_box)
                if best_face is not None:
                    embedding = best_face.embedding
                else:
                    return np.zeros(self.embedding_size)
                
            elif hasattr(self.model, 'predict'):
                # DeepFace implementation
                # Preprocess image
                from deepface.commons import functions
                face_img = functions.preprocess_face(
                    img=face_img,
                    target_size=(112, 112),
                    enforce_detection=False
                )
                embedding = self.model.predict(face_img)[0, :]
                
            else:
                logger.error("Unknown ArcFace model implementation")
                return np.zeros(self.embedding_size)
            
            # Normalize embedding
            if embedding is not None and len(embedding) > 0:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            else:
                embedding = np.zeros(self.embedding_size)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return np.zeros(self.embedding_size)
    
    def _find_matching_face(self, faces: List[Any], face_box: FaceBox) -> Optional[Any]:
        """
        Find the face from insightface that best matches our detection.
        
        Args:
            faces: List of faces from insightface
            face_box: Our face box
            
        Returns:
            Best matching face
        """
        if not faces:
            return None
            
        best_iou = 0
        best_face = None
        
        for face in faces:
            # Calculate IoU
            bbox = face.bbox.astype(int)
            other_box = FaceBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                confidence=face.det_score
            )
            
            iou = self._calculate_iou(face_box, other_box)
            if iou > best_iou:
                best_iou = iou
                best_face = face
        
        # Require minimum IoU of 0.5
        if best_iou < 0.5:
            return None
            
        return best_face
    
    def _calculate_iou(self, box1: FaceBox, box2: FaceBox) -> float:
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection area
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        union_area = box1_area + box2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def find_identity(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the identity matching the given embedding.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Tuple of (identity_id, similarity_score) or (None, 0.0) if no match
        """
        if not self.identities:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for identity_id, identity in self.identities.items():
            try:
                reference_embedding = identity.avg_embedding
                similarity = self._calculate_similarity(embedding, reference_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = identity_id
            except:
                continue
        
        # Apply recognition threshold
        if best_similarity >= self.recognition_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings using cosine similarity.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure embeddings are normalized
        if np.linalg.norm(embedding1) > 0:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
        if np.linalg.norm(embedding2) > 0:
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Scale to 0-1 range
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def register_identity(
        self,
        identity_id: str,
        name: str,
        embedding: np.ndarray,
        face_img: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new identity or update an existing one.
        
        Args:
            identity_id: Unique identity ID
            name: Human-readable name
            embedding: Face embedding vector
            face_img: Thumbnail image of the face
            metadata: Additional metadata
        """
        if identity_id in self.identities:
            # Update existing identity
            identity = self.identities[identity_id]
            identity.name = name
            identity.add_embedding(embedding)
            if face_img is not None:
                identity.thumbnail = face_img
            if metadata:
                identity.metadata.update(metadata)
        else:
            # Create new identity
            identity = FaceIdentity(
                id=identity_id,
                name=name,
                embeddings=[embedding],
                thumbnail=face_img,
                metadata=metadata or {}
            )
            self.identities[identity_id] = identity
        
        logger.info(f"Registered identity: {identity_id} ({name})")
    
    def recognize_face(
        self,
        frame: np.ndarray,
        face_box: FaceBox
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Recognize a face in a frame.
        
        Args:
            frame: Input frame
            face_box: Face bounding box
            
        Returns:
            Tuple of (identity_id, name, confidence) or (None, None, 0.0) if no match
        """
        # Extract embedding
        embedding = self.extract_embedding(frame, face_box)
        
        # Find matching identity
        identity_id, similarity = self.find_identity(embedding)
        
        if identity_id is not None:
            name = self.identities[identity_id].name
            return identity_id, name, similarity
        else:
            return None, None, similarity
    
    def load_database(self) -> None:
        """Load face identities from database file."""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Face database file not found: {self.db_path}")
                return
            
            logger.info(f"Loading face database from {self.db_path}")
            
            import pickle
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
            
            # Deserialize identities
            self.identities = {}
            for identity_id, identity_data in data.items():
                identity = FaceIdentity(
                    id=identity_id,
                    name=identity_data.get('name', identity_id),
                    embeddings=identity_data.get('embeddings', []),
                    thumbnail=identity_data.get('thumbnail'),
                    metadata=identity_data.get('metadata', {})
                )
                self.identities[identity_id] = identity
            
            logger.info(f"Loaded {len(self.identities)} face identities")
            
        except Exception as e:
            logger.error(f"Error loading face database: {str(e)}")
    
    def save_database(self) -> None:
        """Save face identities to database file."""
        try:
            if not self.db_path:
                logger.warning("No database path specified for saving")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
            logger.info(f"Saving face database to {self.db_path}")
            
            # Serialize identities
            data = {}
            for identity_id, identity in self.identities.items():
                data[identity_id] = {
                    'name': identity.name,
                    'embeddings': identity.embeddings,
                    'thumbnail': identity.thumbnail,
                    'metadata': identity.metadata
                }
            
            import pickle
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.identities)} face identities")
            
        except Exception as e:
            logger.error(f"Error saving face database: {str(e)}") 