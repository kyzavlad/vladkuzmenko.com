"""
Voice Cloning Module

This module provides functionality for high-quality voice cloning
with minimal audio samples (as little as 15 seconds). It enables
cloning voice characteristics while preserving emotions and natural
intonation in synthesized speech.

Features:
- Voice characteristic extraction from minimal samples
- Speaker embedding using x-vector technology
- Neural vocoder (WaveRNN) for high-quality synthesis
- Voice consistency verification metrics
- Emotion preservation in synthesized speech
- Prosody transfer for natural intonation
- Voice style transfer capabilities
- Real-time voice synthesis optimization
"""

from app.avatar_creation.voice_cloning.voice_cloner import VoiceCloner
from app.avatar_creation.voice_cloning.characteristic_extractor import VoiceCharacteristicExtractor
from app.avatar_creation.voice_cloning.speaker_embedding import SpeakerEmbedding
from app.avatar_creation.voice_cloning.neural_vocoder import NeuralVocoder

# Define the public API
__all__ = [
    'VoiceCloner',
    'VoiceCharacteristicExtractor',
    'SpeakerEmbedding',
    'NeuralVocoder'
]
