import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import librosa
from typing import Dict, List, Tuple, Optional, Union, Any

class XVectorNetwork(nn.Module):
    """
    Implementation of X-Vector speaker embedding network.
    Architecture based on the paper "X-vectors: Robust DNN Embeddings for Speaker Recognition"
    by Snyder et al.
    """
    
    def __init__(self, input_dim: int = 40, embedding_dim: int = 512):
        """
        Initialize X-Vector network.
        
        Args:
            input_dim: Dimension of input features (e.g., MFCC features)
            embedding_dim: Dimension of the output embedding
        """
        super().__init__()
        
        # Frame-level layers
        self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.tdnn5 = nn.Conv1d(512, 1500, kernel_size=1, dilation=1)
        self.bn5 = nn.BatchNorm1d(1500)
        
        # Statistics pooling layer
        
        # Segment-level layers
        self.fc1 = nn.Linear(3000, 512)
        self.bn6 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, embedding_dim)
        self.bn7 = nn.BatchNorm1d(embedding_dim)
        
        # Output layers (for training, not used during embedding extraction)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)  # This would be num_speakers for training
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim, time]
            
        Returns:
            Dictionary containing embeddings and other outputs
        """
        # Frame-level layers
        x = F.relu(self.bn1(self.tdnn1(x)))
        x = F.relu(self.bn2(self.tdnn2(x)))
        x = F.relu(self.bn3(self.tdnn3(x)))
        x = F.relu(self.bn4(self.tdnn4(x)))
        x = F.relu(self.bn5(self.tdnn5(x)))
        
        # Statistics pooling: mean and std along time dimension
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        
        # Concatenate mean and standard deviation
        statpool = torch.cat((mean, std), dim=1)
        
        # Segment-level layers
        x = F.relu(self.bn6(self.fc1(statpool)))
        
        # This is the actual x-vector embedding
        embedding = self.bn7(self.fc2(x))
        
        # Additional layers for classification (if needed)
        x = F.relu(self.fc3(embedding))
        output = self.output(x)
        
        return {
            'embedding': embedding,
            'output': output
        }
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract just the embedding vector without the classification head.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim, time]
            
        Returns:
            Embedding tensor
        """
        # Frame-level layers
        x = F.relu(self.bn1(self.tdnn1(x)))
        x = F.relu(self.bn2(self.tdnn2(x)))
        x = F.relu(self.bn3(self.tdnn3(x)))
        x = F.relu(self.bn4(self.tdnn4(x)))
        x = F.relu(self.bn5(self.tdnn5(x)))
        
        # Statistics pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        statpool = torch.cat((mean, std), dim=1)
        
        # Segment-level embedding
        x = F.relu(self.bn6(self.fc1(statpool)))
        embedding = self.bn7(self.fc2(x))
        
        return embedding


class SpeakerEmbedding:
    """
    Class for extracting speaker embeddings (x-vectors) from audio samples.
    These embeddings capture speaker identity information in a fixed-dimensional vector.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                use_gpu: bool = True,
                embedding_dim: int = 512,
                feature_type: str = 'mfcc',
                min_segment_duration: float = 3.0,
                sample_rate: int = 16000):
        """
        Initialize the speaker embedding extractor.
        
        Args:
            model_path: Path to pre-trained model (None for placeholder implementation)
            use_gpu: Whether to use GPU for computation
            embedding_dim: Dimension of the speaker embedding
            feature_type: Feature type for audio preprocessing ('mfcc', 'fbank', 'spectrogram')
            min_segment_duration: Minimum segment duration for reliable embedding
            sample_rate: Sample rate for audio processing
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.embedding_dim = embedding_dim
        self.feature_type = feature_type
        self.min_segment_duration = min_segment_duration
        self.sample_rate = sample_rate
        
        # Feature extraction parameters
        self.n_mfcc = 40 if feature_type == 'mfcc' else 80
        self.n_mels = 80
        self.window_size = 25  # ms
        self.hop_size = 10     # ms
        self.n_fft = int(self.window_size * self.sample_rate / 1000)
        self.hop_length = int(self.hop_size * self.sample_rate / 1000)
        
        # Mean and standard deviation for feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # State tracking
        self.extraction_history = []
        
        # Initialize model
        self._initialize_model()
        
        print(f"Speaker Embedding extractor initialized")
        print(f"  - Model path: {self.model_path or 'Using placeholder model'}")
        print(f"  - Using device: {self.device}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        print(f"  - Feature type: {self.feature_type}")
        print(f"  - Minimum segment duration: {self.min_segment_duration}s")
    
    def _initialize_model(self) -> None:
        """
        Initialize the x-vector model.
        Load pre-trained model if available, otherwise create a placeholder.
        """
        # Initialize feature dimensionality based on feature type
        input_dim = self.n_mfcc if self.feature_type == 'mfcc' else self.n_mels
        
        # Create model instance
        self.model = XVectorNetwork(input_dim=input_dim, embedding_dim=self.embedding_dim)
        
        # Load pre-trained model if available
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Load normalization stats if available
                    if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
                        self.feature_mean = checkpoint['feature_mean']
                        self.feature_std = checkpoint['feature_std']
                else:
                    self.model.load_state_dict(checkpoint)
                
                print(f"Loaded pre-trained x-vector model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing with random weights")
        else:
            # If no model path or file doesn't exist, use random initialization
            print("No pre-trained model provided. Using placeholder model with random weights.")
        
        # Move model to appropriate device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def extract_embedding(self, 
                         audio_path: str, 
                         save_embedding: bool = True,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract speaker embedding from an audio file.
        
        Args:
            audio_path: Path to the audio file
            save_embedding: Whether to save the embedding
            output_dir: Directory to save embeddings (if None, uses directory of audio_path)
            
        Returns:
            Dictionary containing the speaker embedding and metadata
        """
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            features, metadata = self._preprocess_audio(audio_path)
            
            # Split into segments if audio is long
            embeddings = []
            segment_duration = metadata['duration']
            
            if segment_duration < self.min_segment_duration:
                print(f"Warning: Audio segment ({segment_duration:.2f}s) is shorter than the "
                      f"minimum recommended duration ({self.min_segment_duration}s).")
            
            # Convert features to torch tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            
            # Move to device
            features_tensor = features_tensor.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.extract_embedding(features_tensor)
            
            # Convert to numpy array
            embedding_np = embedding.cpu().numpy().squeeze()
            
            # Save embedding if requested
            embedding_path = None
            if save_embedding:
                out_dir = output_dir if output_dir else os.path.dirname(audio_path)
                os.makedirs(out_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(audio_path))[0]
                
                # Save as numpy file
                embedding_path = os.path.join(out_dir, f"{basename}_xvector.npy")
                np.save(embedding_path, embedding_np)
            
            # Create result
            result = {
                'embedding': embedding_np,
                'duration': segment_duration,
                'feature_type': self.feature_type,
                'embedding_dim': self.embedding_dim,
                'audio_path': audio_path,
                'embedding_path': embedding_path,
                'processing_time': time.time() - start_time
            }
            
            # Add to extraction history
            self.extraction_history.append({
                'timestamp': time.time(),
                'audio_path': audio_path,
                'embedding_path': embedding_path,
                'duration': segment_duration
            })
            
            print(f"Speaker embedding extracted in {result['processing_time']:.2f} seconds")
            return result
            
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            return {
                'error': str(e),
                'audio_path': audio_path,
                'processing_time': time.time() - start_time
            }
    
    def _preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess audio file to extract acoustic features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (feature_array, metadata)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Calculate duration
        duration = len(y) / sr
        
        # Calculate energy and apply VAD
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        
        # Normalize audio
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Extract features based on feature type
        if self.feature_type == 'mfcc':
            # Extract MFCCs
            features = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window='hamming'
            )
            
            # Add delta and delta-delta features
            delta = librosa.feature.delta(features)
            delta2 = librosa.feature.delta(features, order=2)
            features = np.vstack([features, delta, delta2])
            
        elif self.feature_type == 'fbank':
            # Extract filter banks
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window='hamming'
            )
            features = librosa.power_to_db(mel_spec)
            
        elif self.feature_type == 'spectrogram':
            # Extract spectrogram
            spec = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
            features = librosa.amplitude_to_db(spec)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Add feature normalization
        if self.feature_mean is not None and self.feature_std is not None:
            # Apply pre-computed normalization
            features = (features - self.feature_mean.reshape(-1, 1)) / (self.feature_std.reshape(-1, 1) + 1e-8)
        else:
            # Normalize features per utterance
            features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        # Create metadata
        metadata = {
            'duration': duration,
            'sample_rate': sr,
            'energy_mean': float(energy_mean),
            'feature_shape': features.shape
        }
        
        return features, metadata
    
    def compare_embeddings(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> Dict[str, float]:
        """
        Compare two speaker embeddings for similarity.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Dictionary of similarity scores
        """
        # Normalize embeddings to unit length
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(embedding1_norm, embedding2_norm)
        
        # Calculate Euclidean distance
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        
        # Calculate PLDA-like scoring (simplified)
        plda_score = -0.5 * euclidean_dist + cosine_sim
        
        # Return similarity metrics
        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'plda_score': float(plda_score)
        }
    
    def load_embedding(self, embedding_path: str) -> np.ndarray:
        """
        Load speaker embedding from file.
        
        Args:
            embedding_path: Path to the embedding file
            
        Returns:
            Speaker embedding as numpy array
        """
        try:
            embedding = np.load(embedding_path)
            return embedding
        except Exception as e:
            print(f"Error loading embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def _segment_audio(self, y: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Segment audio into chunks for processing long files.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            List of audio segments
        """
        # Calculate segment size in samples
        segment_size = int(self.min_segment_duration * sr)
        
        # If audio is shorter than segment size, return as is
        if len(y) <= segment_size:
            return [y]
        
        # Otherwise segment with overlap
        hop_size = segment_size // 2  # 50% overlap
        segments = []
        
        for start in range(0, len(y) - segment_size + 1, hop_size):
            segment = y[start:start + segment_size]
            segments.append(segment)
        
        return segments
    
    def reset(self) -> None:
        """
        Reset the embedding extractor state.
        """
        self.extraction_history = []
        print("Speaker embedding extractor reset to initial state") 