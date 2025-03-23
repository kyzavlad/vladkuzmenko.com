import os
import time
import numpy as np
import torch
import shutil
import json
from typing import Dict, List, Tuple, Optional, Union, Any

from app.avatar_creation.voice_cloning.characteristic_extractor import VoiceCharacteristicExtractor
from app.avatar_creation.voice_cloning.speaker_embedding import SpeakerEmbedding
from app.avatar_creation.voice_cloning.neural_vocoder import NeuralVocoder

class VoiceCloner:
    """
    Main class for voice cloning system.
    
    Coordinates all components:
    1. Voice characteristic extraction
    2. Speaker embedding (x-vector)
    3. Neural vocoder
    4. Voice verification metrics
    5. Prosody transfer
    6. Voice style transfer
    
    Provides an easy-to-use interface for voice cloning.
    """
    
    def __init__(self, 
                models_dir: Optional[str] = None,
                output_dir: str = 'output/voice_cloning',
                use_gpu: bool = True,
                sample_rate: int = 22050,
                min_sample_duration: float = 15.0,
                feature_dimensionality: int = 256,
                embedding_dim: int = 512):
        """
        Initialize the voice cloning system.
        
        Args:
            models_dir: Directory containing pre-trained models
            output_dir: Directory for output files
            use_gpu: Whether to use GPU for computation
            sample_rate: Sample rate for audio processing
            min_sample_duration: Minimum duration of voice samples (in seconds)
            feature_dimensionality: Dimension of voice feature vectors
            embedding_dim: Dimension of speaker embedding vectors
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.sample_rate = sample_rate
        self.min_sample_duration = min_sample_duration
        self.feature_dimensionality = feature_dimensionality
        self.embedding_dim = embedding_dim
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model paths
        self.model_paths = self._get_model_paths()
        
        # Initialize components
        self._initialize_components()
        
        # Cloned voice data
        self.voice_data = {}
        self.cloned_voices = []
        
        print(f"Voice Cloning System initialized")
        print(f"  - Models directory: {self.models_dir or 'Using default paths'}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Using GPU: {self.use_gpu}")
        print(f"  - Sample rate: {self.sample_rate} Hz")
        print(f"  - Min sample duration: {self.min_sample_duration} seconds")
    
    def _get_model_paths(self) -> Dict[str, str]:
        """
        Get paths to pre-trained models.
        
        Returns:
            Dictionary of model paths
        """
        paths = {}
        
        if self.models_dir and os.path.exists(self.models_dir):
            # Look for models in specified directory
            for filename in os.listdir(self.models_dir):
                fullpath = os.path.join(self.models_dir, filename)
                
                if filename.endswith('.pt') or filename.endswith('.pth'):
                    if 'xvector' in filename.lower():
                        paths['speaker_embedding'] = fullpath
                    elif 'wavernn' in filename.lower():
                        paths['vocoder'] = fullpath
                    elif 'synthesizer' in filename.lower() or 'tts' in filename.lower():
                        paths['synthesizer'] = fullpath
        
        return paths
    
    def _initialize_components(self) -> None:
        """
        Initialize all components of the voice cloning system.
        """
        # Voice characteristic extractor
        self.characteristic_extractor = VoiceCharacteristicExtractor(
            sample_rate=self.sample_rate,
            min_sample_duration=self.min_sample_duration,
            use_gpu=self.use_gpu,
            feature_dimensionality=self.feature_dimensionality
        )
        
        # Speaker embedding (x-vector)
        self.speaker_embedding = SpeakerEmbedding(
            model_path=self.model_paths.get('speaker_embedding'),
            use_gpu=self.use_gpu,
            embedding_dim=self.embedding_dim,
            sample_rate=self.sample_rate
        )
        
        # Neural vocoder
        self.vocoder = NeuralVocoder(
            model_path=self.model_paths.get('vocoder'),
            use_gpu=self.use_gpu,
            sample_rate=self.sample_rate
        )
    
    def clone_voice(self, 
                   audio_path: str, 
                   voice_name: str,
                   save_data: bool = True) -> Dict[str, Any]:
        """
        Clone a voice from an audio file.
        
        Args:
            audio_path: Path to the audio file
            voice_name: Name for the cloned voice
            save_data: Whether to save voice data to disk
            
        Returns:
            Dictionary containing voice data
        """
        start_time = time.time()
        
        try:
            # Create output directory for this voice
            voice_dir = os.path.join(self.output_dir, voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            
            print(f"Cloning voice '{voice_name}' from {audio_path}")
            
            # Extract voice characteristics
            print("Extracting voice characteristics...")
            characteristics = self.characteristic_extractor.extract_characteristics(
                audio_path=audio_path,
                save_features=save_data,
                output_dir=voice_dir
            )
            
            # Extract speaker embedding (x-vector)
            print("Extracting speaker embedding...")
            embedding = self.speaker_embedding.extract_embedding(
                audio_path=audio_path,
                save_embedding=save_data,
                output_dir=voice_dir
            )
            
            # Create voice data dictionary
            voice_data = {
                'voice_name': voice_name,
                'audio_path': audio_path,
                'voice_dir': voice_dir,
                'characteristics': characteristics,
                'embedding': embedding,
                'creation_time': time.time(),
                'processing_time': time.time() - start_time
            }
            
            # Copy source audio file to voice directory
            if save_data:
                source_filename = os.path.basename(audio_path)
                source_copy_path = os.path.join(voice_dir, f"source_{source_filename}")
                shutil.copy2(audio_path, source_copy_path)
                voice_data['source_copy_path'] = source_copy_path
                
                # Save voice data as JSON
                json_path = os.path.join(voice_dir, f"{voice_name}_data.json")
                
                # Create a serializable version of the data
                save_data = {
                    'voice_name': voice_name,
                    'audio_path': audio_path,
                    'voice_dir': voice_dir,
                    'creation_time': voice_data['creation_time'],
                    'processing_time': voice_data['processing_time'],
                    'sample_rate': self.sample_rate,
                    'feature_paths': {
                        'characteristics': characteristics.get('feature_path'),
                        'embedding': embedding.get('embedding_path')
                    },
                    'pitch_stats': {
                        'mean': characteristics['pitch']['mean'],
                        'std': characteristics['pitch']['std'],
                        'median': characteristics['pitch']['median'],
                        'min': characteristics['pitch']['min'],
                        'max': characteristics['pitch']['max'],
                        'range': characteristics['pitch']['range']
                    },
                    'voice_quality': {
                        'hnr': characteristics['quality']['hnr'],
                        'jitter': characteristics['quality']['jitter'],
                        'shimmer': characteristics['quality']['shimmer']
                    }
                }
                
                with open(json_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                voice_data['json_path'] = json_path
            
            # Add to list of cloned voices
            self.voice_data[voice_name] = voice_data
            if voice_name not in self.cloned_voices:
                self.cloned_voices.append(voice_name)
            
            print(f"Voice '{voice_name}' cloned successfully in {voice_data['processing_time']:.2f} seconds")
            
            return voice_data
            
        except Exception as e:
            print(f"Error cloning voice: {e}")
            return {'error': str(e)}
    
    def synthesize_speech(self, 
                         text: str, 
                         voice_name: str,
                         output_path: Optional[str] = None,
                         emotion: Optional[str] = None,
                         speaking_rate: float = 1.0,
                         pitch_shift: float = 0.0) -> Dict[str, Any]:
        """
        Synthesize speech using a cloned voice.
        
        Args:
            text: Text to synthesize
            voice_name: Name of the cloned voice to use
            output_path: Path to save the synthesized audio
            emotion: Emotion to apply to the speech
            speaking_rate: Speaking rate multiplier
            pitch_shift: Pitch shift in semitones
            
        Returns:
            Dictionary containing synthesis results
        """
        start_time = time.time()
        
        try:
            # Check if voice exists
            if voice_name not in self.voice_data:
                raise ValueError(f"Voice '{voice_name}' not found. Clone it first.")
            
            voice_data = self.voice_data[voice_name]
            
            print(f"Synthesizing speech for voice '{voice_name}'")
            print(f"Text: {text}")
            
            # Placeholder for a mel-spectrogram
            # In a complete implementation, this would be generated by a text-to-spectogram model
            # using the speaker embedding and voice characteristics
            placeholder_mel = np.random.random((80, 100)) * 2 - 1
            
            # Generate default output path if not specified
            if output_path is None:
                timestamp = int(time.time())
                output_path = os.path.join(voice_data['voice_dir'], f"{voice_name}_{timestamp}.wav")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Synthesizing audio...")
            waveform, metadata = self.vocoder.synthesize(
                mel_spectrogram=placeholder_mel,
                save_path=output_path,
                target_text=text
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create results dictionary
            results = {
                'voice_name': voice_name,
                'text': text,
                'output_path': output_path,
                'processing_time': processing_time,
                'audio_length': metadata['audio_length'],
                'real_time_factor': metadata['real_time_factor'],
                'emotion': emotion,
                'speaking_rate': speaking_rate,
                'pitch_shift': pitch_shift
            }
            
            print(f"Speech synthesized in {processing_time:.2f} seconds")
            print(f"Audio saved to {output_path}")
            
            return results
            
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return {'error': str(e)}
    
    def verify_voice_consistency(self, 
                               original_audio_path: str, 
                               synthesized_audio_path: str) -> Dict[str, float]:
        """
        Verify the consistency between original and synthesized voice.
        
        Args:
            original_audio_path: Path to the original voice audio
            synthesized_audio_path: Path to the synthesized voice audio
            
        Returns:
            Dictionary of consistency metrics
        """
        try:
            print(f"Verifying voice consistency...")
            
            # Extract characteristics from both audio files
            original_chars = self.characteristic_extractor.extract_characteristics(
                audio_path=original_audio_path,
                save_features=False
            )
            
            synthesized_chars = self.characteristic_extractor.extract_characteristics(
                audio_path=synthesized_audio_path,
                save_features=False
            )
            
            # Extract embeddings from both audio files
            original_emb = self.speaker_embedding.extract_embedding(
                audio_path=original_audio_path,
                save_embedding=False
            )
            
            synthesized_emb = self.speaker_embedding.extract_embedding(
                audio_path=synthesized_audio_path,
                save_embedding=False
            )
            
            # Compare feature vectors
            feature_similarity = self.characteristic_extractor.compare_voices(
                original_chars['feature_vector'],
                synthesized_chars['feature_vector']
            )
            
            # Compare embeddings
            embedding_similarity = self.speaker_embedding.compare_embeddings(
                original_emb['embedding'],
                synthesized_emb['embedding']
            )
            
            # Calculate pitch similarity
            pitch_diff = abs(original_chars['pitch']['mean'] - synthesized_chars['pitch']['mean'])
            pitch_similarity = 1.0 - min(1.0, pitch_diff / 100.0)  # Normalize to [0, 1]
            
            # Calculate rhythm similarity
            rhythm_diff = abs(original_chars['temporal']['syllable_rate'] - 
                            synthesized_chars['temporal']['syllable_rate'])
            rhythm_similarity = 1.0 - min(1.0, rhythm_diff / 2.0)  # Normalize to [0, 1]
            
            # Calculate voice quality similarity
            hnr_diff = abs(original_chars['quality']['hnr'] - synthesized_chars['quality']['hnr'])
            hnr_similarity = 1.0 - min(1.0, hnr_diff / 10.0)  # Normalize to [0, 1]
            
            # Combine metrics
            metrics = {
                'feature_cosine_similarity': feature_similarity['cosine_similarity'],
                'feature_euclidean_distance': feature_similarity['euclidean_distance'],
                'embedding_cosine_similarity': embedding_similarity['cosine_similarity'],
                'embedding_plda_score': embedding_similarity['plda_score'],
                'pitch_similarity': pitch_similarity,
                'rhythm_similarity': rhythm_similarity,
                'voice_quality_similarity': hnr_similarity,
                'overall_similarity': (feature_similarity['cosine_similarity'] + 
                                    embedding_similarity['cosine_similarity'] + 
                                    pitch_similarity + rhythm_similarity + 
                                    hnr_similarity) / 5.0
            }
            
            print(f"Voice consistency verification complete")
            print(f"Overall similarity: {metrics['overall_similarity']:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error verifying voice consistency: {e}")
            return {'error': str(e)}
    
    def transfer_prosody(self, 
                        source_audio_path: str, 
                        target_voice_name: str,
                        output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Transfer prosody from source audio to target voice.
        
        Args:
            source_audio_path: Path to the source audio with desired prosody
            target_voice_name: Name of the target cloned voice
            output_path: Path to save the output audio
            
        Returns:
            Dictionary containing transfer results
        """
        start_time = time.time()
        
        try:
            # Check if voice exists
            if target_voice_name not in self.voice_data:
                raise ValueError(f"Voice '{target_voice_name}' not found. Clone it first.")
            
            voice_data = self.voice_data[target_voice_name]
            
            print(f"Transferring prosody from {source_audio_path} to voice '{target_voice_name}'")
            
            # Extract prosody features from source audio
            # This would extract pitch contour, timing, energy, etc.
            
            # Generate default output path if not specified
            if output_path is None:
                timestamp = int(time.time())
                output_path = os.path.join(voice_data['voice_dir'], 
                                         f"{target_voice_name}_prosody_{timestamp}.wav")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Placeholder mel spectrogram (would be generated with source prosody features)
            placeholder_mel = np.random.random((80, 100)) * 2 - 1
            
            # Generate audio with the vocoder
            print(f"Synthesizing audio with transferred prosody...")
            waveform, metadata = self.vocoder.synthesize(
                mel_spectrogram=placeholder_mel,
                save_path=output_path,
                target_text="Prosody transfer example"
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create results dictionary
            results = {
                'source_audio_path': source_audio_path,
                'target_voice_name': target_voice_name,
                'output_path': output_path,
                'processing_time': processing_time,
                'audio_length': metadata['audio_length'],
                'real_time_factor': metadata['real_time_factor']
            }
            
            print(f"Prosody transfer completed in {processing_time:.2f} seconds")
            print(f"Audio saved to {output_path}")
            
            return results
            
        except Exception as e:
            print(f"Error transferring prosody: {e}")
            return {'error': str(e)}
    
    def transfer_style(self, 
                      source_style_path: str, 
                      target_voice_name: str,
                      text: str,
                      style_strength: float = 0.5,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Transfer speaking style from source audio to target voice.
        
        Args:
            source_style_path: Path to the audio with desired speaking style
            target_voice_name: Name of the target cloned voice
            text: Text to synthesize
            style_strength: Strength of style transfer (0.0-1.0)
            output_path: Path to save the output audio
            
        Returns:
            Dictionary containing transfer results
        """
        start_time = time.time()
        
        try:
            # Check if voice exists
            if target_voice_name not in self.voice_data:
                raise ValueError(f"Voice '{target_voice_name}' not found. Clone it first.")
            
            voice_data = self.voice_data[target_voice_name]
            
            print(f"Transferring style from {source_style_path} to voice '{target_voice_name}'")
            print(f"Text: {text}")
            print(f"Style strength: {style_strength}")
            
            # Extract style features from source audio
            # This would extract broader patterns than just prosody
            
            # Generate default output path if not specified
            if output_path is None:
                timestamp = int(time.time())
                output_path = os.path.join(voice_data['voice_dir'], 
                                         f"{target_voice_name}_style_{timestamp}.wav")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Placeholder mel spectrogram (would be generated with source style features)
            placeholder_mel = np.random.random((80, 100)) * 2 - 1
            
            # Generate audio with the vocoder
            print(f"Synthesizing audio with transferred style...")
            waveform, metadata = self.vocoder.synthesize(
                mel_spectrogram=placeholder_mel,
                save_path=output_path,
                target_text=text
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create results dictionary
            results = {
                'source_style_path': source_style_path,
                'target_voice_name': target_voice_name,
                'text': text,
                'style_strength': style_strength,
                'output_path': output_path,
                'processing_time': processing_time,
                'audio_length': metadata['audio_length'],
                'real_time_factor': metadata['real_time_factor']
            }
            
            print(f"Style transfer completed in {processing_time:.2f} seconds")
            print(f"Audio saved to {output_path}")
            
            return results
            
        except Exception as e:
            print(f"Error transferring style: {e}")
            return {'error': str(e)}
    
    def load_voice(self, voice_dir: str) -> Dict[str, Any]:
        """
        Load a previously cloned voice from directory.
        
        Args:
            voice_dir: Directory containing the cloned voice data
            
        Returns:
            Dictionary containing voice data
        """
        try:
            # Check if directory exists
            if not os.path.exists(voice_dir):
                raise ValueError(f"Voice directory not found: {voice_dir}")
            
            # Look for voice data JSON file
            json_files = [f for f in os.listdir(voice_dir) if f.endswith('_data.json')]
            
            if not json_files:
                raise ValueError(f"No voice data found in {voice_dir}")
            
            # Load the first JSON file found
            json_path = os.path.join(voice_dir, json_files[0])
            
            with open(json_path, 'r') as f:
                voice_data = json.load(f)
            
            voice_name = voice_data['voice_name']
            
            print(f"Loading voice '{voice_name}' from {voice_dir}")
            
            # Load feature files
            feature_paths = voice_data.get('feature_paths', {})
            
            characteristics = None
            if 'characteristics' in feature_paths and os.path.exists(feature_paths['characteristics']):
                # Load characteristics
                characteristics = {
                    'feature_vector': self.characteristic_extractor.load_feature_vector(
                        feature_paths['characteristics']
                    ),
                    'feature_path': feature_paths['characteristics'],
                    'pitch': voice_data.get('pitch_stats', {}),
                    'quality': voice_data.get('voice_quality', {})
                }
            
            embedding = None
            if 'embedding' in feature_paths and os.path.exists(feature_paths['embedding']):
                # Load embedding
                embedding = {
                    'embedding': self.speaker_embedding.load_embedding(
                        feature_paths['embedding']
                    ),
                    'embedding_path': feature_paths['embedding']
                }
            
            # Recreate voice data dictionary
            full_voice_data = {
                'voice_name': voice_name,
                'voice_dir': voice_dir,
                'json_path': json_path,
                'characteristics': characteristics,
                'embedding': embedding,
                'creation_time': voice_data.get('creation_time', 0),
                'audio_path': voice_data.get('audio_path')
            }
            
            # Add to list of cloned voices
            self.voice_data[voice_name] = full_voice_data
            if voice_name not in self.cloned_voices:
                self.cloned_voices.append(voice_name)
            
            print(f"Voice '{voice_name}' loaded successfully")
            
            return full_voice_data
            
        except Exception as e:
            print(f"Error loading voice: {e}")
            return {'error': str(e)}
    
    def get_cloned_voices(self) -> List[str]:
        """
        Get list of cloned voices.
        
        Returns:
            List of voice names
        """
        return self.cloned_voices
    
    def get_voice_info(self, voice_name: str) -> Dict[str, Any]:
        """
        Get information about a cloned voice.
        
        Args:
            voice_name: Name of the cloned voice
            
        Returns:
            Dictionary containing voice information
        """
        if voice_name not in self.voice_data:
            return {'error': f"Voice '{voice_name}' not found"}
        
        voice_data = self.voice_data[voice_name]
        
        # Create a simplified version with just the key information
        info = {
            'voice_name': voice_name,
            'creation_time': voice_data.get('creation_time', 0),
            'voice_dir': voice_data.get('voice_dir'),
            'audio_path': voice_data.get('audio_path')
        }
        
        # Add pitch stats if available
        if voice_data.get('characteristics') and 'pitch' in voice_data['characteristics']:
            info['pitch_stats'] = voice_data['characteristics']['pitch']
        
        # Add voice quality if available
        if voice_data.get('characteristics') and 'quality' in voice_data['characteristics']:
            info['voice_quality'] = voice_data['characteristics']['quality']
        
        return info 