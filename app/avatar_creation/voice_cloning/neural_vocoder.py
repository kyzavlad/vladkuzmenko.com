import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import math
from typing import Dict, List, Tuple, Optional, Union, Any

class ResBlock(nn.Module):
    """Residual block for the WaveRNN upsampling network"""
    
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    """MelResNet upsampling network for the WaveRNN vocoder"""
    
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, padding=pad)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)
        
    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    """Upsampling layer for the WaveRNN vocoder"""
    
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.unsqueeze(3).unsqueeze(5)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        x = x.view(batch_size, channels, height * self.y_scale, width * self.x_scale)
        return x


class UpsampleNetwork(nn.Module):
    """Upsampling network for the WaveRNN vocoder"""
    
    def __init__(self, feat_dims, upsample_scales, compute_dims, 
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.prod(upsample_scales)
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            
    def forward(self, mels):
        aux = self.resnet(mels)
        aux = aux.unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        mels = mels.unsqueeze(1)
        for f in self.up_layers:
            mels = f(mels)
        mels = mels.squeeze(1)[:, :, self.indent:-self.indent]
        return mels, aux


class WaveRNN(nn.Module):
    """Neural vocoder based on WaveRNN for real-time speech synthesis"""
    
    def __init__(self, 
                rnn_dims=512, 
                fc_dims=512, 
                bits=9, 
                pad=2,
                upsample_factors=(5, 5, 8), 
                feat_dims=80,
                compute_dims=128, 
                res_out_dims=128, 
                res_blocks=10,
                mode='mold'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        self.n_classes = 3 if mode == 'mold' else 2 ** bits
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims
        self.hop_length = np.prod(upsample_factors)
        
        # Upsampling network
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, 
                                       res_blocks, res_out_dims, pad)
        
        # Main WaveRNN model
        if mode == 'mold':
            self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
            self.R = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
            self.O1 = nn.Linear(rnn_dims, fc_dims)
            self.O2 = nn.Linear(fc_dims, 3)
        else:
            self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
            self.R = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
            self.O1 = nn.Linear(rnn_dims, fc_dims)
            self.O2 = nn.Linear(fc_dims, self.n_classes)
            
        # Display model size
        self.num_params()
        
    def forward(self, x, mels):
        # Upsampling
        mels, aux = self.upsample(mels)
        
        # Adjust time dimensions
        x = x.unsqueeze(-1).expand(-1, -1, self.aux_dims)
        aux = aux.transpose(1, 2).expand(x.size(0), -1, -1)
        mels = mels.transpose(1, 2).expand(x.size(0), -1, -1)
        
        # Combine inputs
        a = torch.cat([x, mels, aux], dim=-1)
        
        # Main RNN
        x = self.I(a)
        res = x
        x, _ = self.R(x)
        x = x + res
        x = F.relu(self.O1(x))
        return self.O2(x)
    
    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in parameters])
        print(f'WaveRNN has {params:,} trainable parameters')


class NeuralVocoder:
    """
    Class implementing a neural vocoder based on WaveRNN for high-quality speech synthesis.
    Converts mel-spectrograms or other acoustic features to waveforms.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                use_gpu: bool = True,
                sample_rate: int = 22050,
                hop_length: int = 275,
                n_mels: int = 80,
                mode: str = 'mold',
                inference_chunk_size: int = 5000,
                use_batched_inference: bool = True):
        """
        Initialize the neural vocoder.
        
        Args:
            model_path: Path to pre-trained WaveRNN model
            use_gpu: Whether to use GPU for computation
            sample_rate: Audio sample rate
            hop_length: Hop length for feature extraction
            n_mels: Number of mel bands
            mode: Mode of operation ('mold' or 'bits')
            inference_chunk_size: Size of chunks for inference (for memory efficiency)
            use_batched_inference: Whether to use batched inference for speed
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mode = mode  # 'mold' is mixture of logistics, 'bits' is for raw audio
        self.inference_chunk_size = inference_chunk_size
        self.use_batched_inference = use_batched_inference
        
        # WaveRNN model
        self.model = None
        
        # Performance monitoring
        self.generation_times = []
        self.real_time_factors = []
        self.last_generation_time = 0
        
        # Initialize model
        self._initialize_model()
        
        print(f"Neural Vocoder initialized")
        print(f"  - Model path: {self.model_path or 'Using placeholder model'}")
        print(f"  - Using device: {self.device}")
        print(f"  - Sample rate: {self.sample_rate}")
        print(f"  - Mode: {self.mode}")
        print(f"  - Batched inference: {self.use_batched_inference}")
    
    def _initialize_model(self) -> None:
        """
        Initialize the WaveRNN model.
        Load pre-trained model if available.
        """
        upsample_factors = (5, 5, 11)  # Default, gives hop_length of 275
        if self.hop_length != np.prod(upsample_factors):
            # Adjust factors to match the hop length
            upsample_factors = self._calculate_upsample_factors(self.hop_length)
        
        # Create WaveRNN model
        self.model = WaveRNN(
            rnn_dims=512, 
            fc_dims=512, 
            bits=9, 
            pad=2,
            upsample_factors=upsample_factors, 
            feat_dims=self.n_mels,
            compute_dims=128, 
            res_out_dims=128, 
            res_blocks=10,
            mode=self.mode
        )
        
        # Load pre-trained model if available
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                print(f"Loaded pre-trained WaveRNN model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing with random weights")
        else:
            # If no model path or file doesn't exist, use random initialization
            print("No pre-trained model provided. Using placeholder model with random weights.")
            print("WARNING: Placeholder model will not produce good audio quality.")
        
        # Move model to appropriate device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def _calculate_upsample_factors(self, hop_length: int) -> Tuple[int, ...]:
        """
        Calculate appropriate upsampling factors for a given hop length.
        
        Args:
            hop_length: Target hop length
            
        Returns:
            Tuple of upsampling factors
        """
        # Try to find 3 factors that multiply to hop_length
        factors = []
        
        # Prime factorization
        n = hop_length
        for i in range(2, int(math.sqrt(n)) + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        
        if n > 1:
            factors.append(n)
        
        # If we have fewer than 3 factors, add some 1s
        while len(factors) < 3:
            factors.insert(0, 1)
        
        # If we have more than 3 factors, combine them
        while len(factors) > 3:
            factors[-2] *= factors[-1]
            factors.pop()
        
        return tuple(factors)
    
    def synthesize(self, 
                  mel_spectrogram: np.ndarray, 
                  save_path: Optional[str] = None,
                  target_text: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesize audio from mel spectrogram.
        
        Args:
            mel_spectrogram: Input mel spectrogram of shape [n_mels, time]
            save_path: Path to save the audio output (optional)
            target_text: Target text for the audio (for metadata)
            
        Returns:
            Tuple of (audio_waveform, metadata)
        """
        start_time = time.time()
        
        try:
            # Ensure proper shape and type
            if mel_spectrogram.shape[0] != self.n_mels:
                if mel_spectrogram.shape[1] == self.n_mels:
                    mel_spectrogram = mel_spectrogram.T  # Transpose if needed
                else:
                    raise ValueError(f"Mel spectrogram shape mismatch. "
                                    f"Expected first dimension to be {self.n_mels}")
            
            # Convert to float32 tensor
            if not isinstance(mel_spectrogram, np.ndarray):
                mel_spectrogram = np.array(mel_spectrogram)
            
            # Normalize if needed
            if np.max(mel_spectrogram) > 1.0 or np.min(mel_spectrogram) < -1.0:
                mel_spectrogram = 2 * ((mel_spectrogram - np.min(mel_spectrogram)) / 
                                      (np.max(mel_spectrogram) - np.min(mel_spectrogram) + 1e-8)) - 1
            
            # Convert to tensor
            mel = torch.FloatTensor(mel_spectrogram).unsqueeze(0)  # Add batch dimension
            mel = mel.to(self.device)
            
            # Generate audio
            output = self._generate_audio(mel)
            
            # Convert to numpy array
            waveform = output.cpu().numpy()
            
            # Normalize audio
            waveform = waveform / np.max(np.abs(waveform))
            
            # Save audio if requested
            if save_path:
                self._save_audio(waveform, save_path)
                
            # Calculate stats
            generation_time = time.time() - start_time
            audio_length = len(waveform) / self.sample_rate
            rtf = generation_time / audio_length  # Real-time factor
            
            # Store stats
            self.generation_times.append(generation_time)
            self.real_time_factors.append(rtf)
            self.last_generation_time = generation_time
            
            # Create metadata
            metadata = {
                'audio_length': audio_length,
                'generation_time': generation_time,
                'real_time_factor': rtf,
                'sample_rate': self.sample_rate,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'text': target_text
            }
            
            print(f"Audio synthesized in {generation_time:.2f}s (RTF: {rtf:.2f})")
            return waveform, metadata
            
        except Exception as e:
            print(f"Error synthesizing audio: {e}")
            return np.zeros(1000), {'error': str(e)}
    
    def _generate_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate audio using the WaveRNN model.
        
        Args:
            mel: Input mel spectrogram tensor [batch, n_mels, time]
            
        Returns:
            Audio waveform tensor
        """
        with torch.no_grad():
            # Get batch and time dimensions
            batch_size, _, steps = mel.size()
            
            # Initialize output buffer
            output = []
            
            if self.mode == 'mold':  # mixture of logistics mode
                # Initial input
                x = torch.zeros(batch_size, 1).to(self.device)
                
                # Process one sample at a time (autoregressive)
                for i in range(steps * self.hop_length):
                    # Forward pass
                    logits = self.model(x, mel)
                    
                    # Sample from mixture of logistics distribution
                    sample = self._sample_from_mol(logits[:, i, :])
                    output.append(sample)
                    
                    # Update input for next step
                    x = sample.unsqueeze(1)
            else:  # raw bits mode
                # Initial input
                x = torch.zeros(batch_size, 1).to(self.device)
                
                # Process in chunks for memory efficiency
                for i in range(0, steps * self.hop_length, self.inference_chunk_size):
                    # Process a chunk
                    chunk_size = min(self.inference_chunk_size, steps * self.hop_length - i)
                    
                    # Forward pass
                    logits = self.model(x, mel)
                    
                    # Sample from categorical distribution
                    sample = self._sample_from_bits(logits[:, i:i+chunk_size, :])
                    output.append(sample)
                    
                    # Update input for next chunk
                    x = sample[:, -1].unsqueeze(1)
            
            # Combine all samples
            output = torch.cat(output, dim=0)
            
            # Convert to proper audio range
            if self.mode == 'mold':
                # Already in [-1, 1]
                pass
            else:
                # Convert from bits to [-1, 1]
                output = 2 * output / (2 ** 9 - 1) - 1
                
            return output
    
    def _sample_from_mol(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from mixture of logistics distribution.
        
        Args:
            logits: Logits from model output
            
        Returns:
            Sampled values
        """
        # Parse logits (mean, log_scale, mixture_weight)
        nr_mix = 10  # Number of mixtures
        logit_probs = logits[:, :nr_mix]
        means = logits[:, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(logits[:, 2 * nr_mix:3 * nr_mix], min=-7.0)
        
        # Sample mixture component
        temp = 1.0  # Temperature for sampling
        probs = F.softmax(logit_probs / temp, dim=1)
        mixture_idx = torch.multinomial(probs, 1).squeeze()
        
        # Select corresponding mean and log_scale
        batch_size = logits.size(0)
        means = means[torch.arange(batch_size), mixture_idx]
        log_scales = log_scales[torch.arange(batch_size), mixture_idx]
        
        # Sample from Gaussian 
        u = torch.rand(batch_size).to(logits.device)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1 - u))
        
        # Clamp to [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)
        
        return x
    
    def _sample_from_bits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from categorical distribution for raw bits.
        
        Args:
            logits: Logits from model output
            
        Returns:
            Sampled values
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from categorical distribution
        samples = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
        
        return samples
    
    def _save_audio(self, waveform: np.ndarray, path: str) -> None:
        """
        Save audio waveform to file.
        
        Args:
            waveform: Audio waveform
            path: Output path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save using librosa
        librosa.output.write_wav(path, waveform, self.sample_rate)
        
        print(f"Audio saved to {path}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the vocoder.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {}
        
        if self.generation_times:
            stats['mean_generation_time'] = np.mean(self.generation_times)
            stats['max_generation_time'] = np.max(self.generation_times)
            stats['min_generation_time'] = np.min(self.generation_times)
            stats['last_generation_time'] = self.last_generation_time
        
        if self.real_time_factors:
            stats['mean_rtf'] = np.mean(self.real_time_factors)
            stats['max_rtf'] = np.max(self.real_time_factors)
            stats['min_rtf'] = np.min(self.real_time_factors)
            
        return stats
    
    def reset_stats(self) -> None:
        """
        Reset performance statistics.
        """
        self.generation_times = []
        self.real_time_factors = []
        self.last_generation_time = 0
        
        print("Performance statistics reset") 