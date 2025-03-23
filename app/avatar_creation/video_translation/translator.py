#!/usr/bin/env python3
"""
Neural Translation Engine

This module provides the core translation functionality using neural machine translation
models with context awareness and specialized handling of technical content.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class TranslationOptions:
    """Options for controlling the translation process."""
    preserve_formatting: bool = True
    preserve_named_entities: bool = True
    preserve_technical_terms: bool = True
    adapt_cultural_references: bool = True
    handle_idiomatic_expressions: bool = True
    maintain_original_timing: bool = True
    confidence_threshold: float = 0.85  # Minimum confidence score for automatic translation
    context_window_size: int = 3  # Number of sentences to consider for context


class NeuralTranslator:
    """
    Core neural machine translation engine supporting multiple language pairs.
    """
    
    def __init__(self, 
                model_dir: str = "models/translation",
                device: str = None,
                use_gpu: bool = True):
        """
        Initialize the neural translator.
        
        Args:
            model_dir: Directory containing translation models
            device: Device to use for inference ('cpu', 'cuda', etc.)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Supported language pairs with ISO 639-1 codes
        self.supported_languages = self._load_supported_languages()
        
        # Dictionary to store loaded models
        self.models = {}
        
        # Translation metadata for tracking changes
        self.translation_metadata = {}
        
        print(f"Neural Translator initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Supported languages: {len(self.supported_languages)}")
    
    def _load_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """
        Load the supported languages configuration.
        
        Returns:
            Dictionary of supported languages and their metadata
        """
        # In a real implementation, this would load from a configuration file
        # For now, we'll define a subset of languages inline
        
        languages = {
            'en': {'name': 'English', 'variants': ['en-US', 'en-GB', 'en-AU', 'en-CA']},
            'es': {'name': 'Spanish', 'variants': ['es-ES', 'es-MX', 'es-AR', 'es-CO']},
            'fr': {'name': 'French', 'variants': ['fr-FR', 'fr-CA', 'fr-BE', 'fr-CH']},
            'de': {'name': 'German', 'variants': ['de-DE', 'de-AT', 'de-CH']},
            'it': {'name': 'Italian', 'variants': ['it-IT', 'it-CH']},
            'pt': {'name': 'Portuguese', 'variants': ['pt-PT', 'pt-BR']},
            'ru': {'name': 'Russian', 'variants': ['ru-RU']},
            'zh': {'name': 'Chinese', 'variants': ['zh-CN', 'zh-TW', 'zh-HK']},
            'ja': {'name': 'Japanese', 'variants': ['ja-JP']},
            'ko': {'name': 'Korean', 'variants': ['ko-KR']},
            'ar': {'name': 'Arabic', 'variants': ['ar-SA', 'ar-EG', 'ar-MA']},
            'hi': {'name': 'Hindi', 'variants': ['hi-IN']},
            'bn': {'name': 'Bengali', 'variants': ['bn-IN', 'bn-BD']},
            'nl': {'name': 'Dutch', 'variants': ['nl-NL', 'nl-BE']},
            'tr': {'name': 'Turkish', 'variants': ['tr-TR']},
            'pl': {'name': 'Polish', 'variants': ['pl-PL']},
            'sv': {'name': 'Swedish', 'variants': ['sv-SE']},
            'fi': {'name': 'Finnish', 'variants': ['fi-FI']},
            'da': {'name': 'Danish', 'variants': ['da-DK']},
            'no': {'name': 'Norwegian', 'variants': ['no-NO', 'nb-NO', 'nn-NO']},
            # Additional languages would be added here
        }
        
        return languages
    
    def _load_model(self, source_lang: str, target_lang: str) -> Any:
        """
        Load a translation model for the specified language pair.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Loaded translation model
        """
        model_key = f"{source_lang}-{target_lang}"
        
        if model_key in self.models:
            return self.models[model_key]
        
        # In a real implementation, this would load a pre-trained translation model
        # For now, we'll create a placeholder
        print(f"Loading translation model: {model_key}")
        
        # Simulate model loading
        model = {
            "name": f"{model_key} Transformer",
            "type": "transformer",
            "params": 350000000,  # 350M parameters
            "loaded": True
        }
        
        self.models[model_key] = model
        return model
    
    def translate_text(self, 
                     text: str, 
                     source_lang: str, 
                     target_lang: str,
                     options: Optional[TranslationOptions] = None) -> Tuple[str, Dict]:
        """
        Translate a text from source language to target language.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            options: Translation options
            
        Returns:
            Tuple of (translated_text, metadata)
        """
        if options is None:
            options = TranslationOptions()
        
        # Verify languages are supported
        if source_lang not in self.supported_languages:
            raise ValueError(f"Source language '{source_lang}' is not supported")
        if target_lang not in self.supported_languages:
            raise ValueError(f"Target language '{target_lang}' is not supported")
        
        # Load the appropriate model
        model = self._load_model(source_lang, target_lang)
        
        # In a real implementation, this would use the model to translate
        # For now, we'll return a placeholder translation
        
        # Simulate translation
        translated_text = f"[{target_lang}] {text}"
        
        # Create metadata about the translation
        metadata = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model["name"],
            "confidence_score": 0.95,
            "processing_time_ms": 150,
            "input_tokens": len(text.split()),
            "output_tokens": len(translated_text.split()),
        }
        
        return translated_text, metadata
    
    def translate_batch(self, 
                      texts: List[str], 
                      source_lang: str, 
                      target_lang: str,
                      options: Optional[TranslationOptions] = None) -> List[Tuple[str, Dict]]:
        """
        Translate a batch of texts from source language to target language.
        
        Args:
            texts: List of source texts to translate
            source_lang: Source language code
            target_lang: Target language code
            options: Translation options
            
        Returns:
            List of tuples (translated_text, metadata)
        """
        if options is None:
            options = TranslationOptions()
        
        results = []
        for text in texts:
            translated, metadata = self.translate_text(text, source_lang, target_lang, options)
            results.append((translated, metadata))
        
        return results
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get a list of supported languages with their metadata.
        
        Returns:
            List of language dictionaries
        """
        result = []
        for code, data in self.supported_languages.items():
            entry = {
                "code": code,
                "name": data["name"],
                "variants": data["variants"]
            }
            result.append(entry)
        
        return result
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to detect language
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        # In a real implementation, this would use a language detection model
        # For now, we'll return a placeholder result
        detected_lang = "en"
        confidence = 0.98
        
        return detected_lang, confidence


class ContextAwareTranslator(NeuralTranslator):
    """
    Enhanced translator that considers context for more accurate translations.
    """
    
    def __init__(self, 
                model_dir: str = "models/translation",
                device: str = None,
                use_gpu: bool = True,
                context_window: int = 3):
        """
        Initialize the context-aware translator.
        
        Args:
            model_dir: Directory containing translation models
            device: Device to use for inference
            use_gpu: Whether to use GPU acceleration
            context_window: Number of sentences to consider for context
        """
        super().__init__(model_dir, device, use_gpu)
        
        self.context_window = context_window
        self.semantic_memory = {}  # Store contextual information across translations
        
        print(f"  - Context window: {context_window} sentences")
    
    def translate_with_context(self, 
                             text: str, 
                             source_lang: str, 
                             target_lang: str,
                             context: List[str] = None,
                             document_id: str = None,
                             options: Optional[TranslationOptions] = None) -> Tuple[str, Dict]:
        """
        Translate a text considering surrounding context.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: List of preceding sentences for context
            document_id: Identifier for document to maintain context
            options: Translation options
            
        Returns:
            Tuple of (translated_text, metadata)
        """
        if options is None:
            options = TranslationOptions()
        
        # Create a combined input with context if provided
        if context:
            # Limit to context window size
            context = context[-options.context_window_size:]
            
            # In a real implementation, this would feed context to the model
            pass
        
        # Store context for future translations if document_id is provided
        if document_id:
            if document_id not in self.semantic_memory:
                self.semantic_memory[document_id] = []
            
            # Update the context memory for this document
            self.semantic_memory[document_id].append(text)
            
            # Keep only the most recent entries within the context window
            self.semantic_memory[document_id] = self.semantic_memory[document_id][-self.context_window:]
        
        # Perform the translation with context awareness
        translated_text, metadata = super().translate_text(text, source_lang, target_lang, options)
        
        # Add context information to metadata
        metadata["context_used"] = bool(context)
        metadata["context_size"] = len(context) if context else 0
        
        return translated_text, metadata
    
    def translate_document(self, 
                         sentences: List[str], 
                         source_lang: str, 
                         target_lang: str,
                         options: Optional[TranslationOptions] = None) -> List[Tuple[str, Dict]]:
        """
        Translate a full document while maintaining context between sentences.
        
        Args:
            sentences: List of sentences in the document
            source_lang: Source language code
            target_lang: Target language code
            options: Translation options
            
        Returns:
            List of tuples (translated_text, metadata)
        """
        if options is None:
            options = TranslationOptions()
        
        results = []
        context = []
        
        for sentence in sentences:
            # Translate with the current context
            translated, metadata = self.translate_with_context(
                sentence, source_lang, target_lang, context, options=options
            )
            
            results.append((translated, metadata))
            
            # Update context for next sentence
            context.append(sentence)
            if len(context) > self.context_window:
                context.pop(0)
        
        return results
    
    def clear_context(self, document_id: Optional[str] = None) -> None:
        """
        Clear the stored context.
        
        Args:
            document_id: If provided, clear only this document's context
        """
        if document_id:
            if document_id in self.semantic_memory:
                del self.semantic_memory[document_id]
        else:
            self.semantic_memory = {}


if __name__ == "__main__":
    # Example usage
    translator = ContextAwareTranslator()
    
    # Simple translation
    text = "Hello, how are you today? I hope you're doing well."
    translated, metadata = translator.translate_text(text, "en", "fr")
    
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    print(f"Metadata: {metadata}")
    
    # Context-aware translation
    context_translator = ContextAwareTranslator(context_window=3)
    
    # Example document
    document = [
        "The AI system analyzes the video content.",
        "It extracts key features and linguistic elements.",
        "These features are used to generate the translation.",
        "The system ensures timing is preserved across languages."
    ]
    
    print("\nDocument translation with context:")
    results = context_translator.translate_document(document, "en", "de")
    
    for i, (translated_text, meta) in enumerate(results):
        print(f"\nSentence {i+1}:")
        print(f"Original: {document[i]}")
        print(f"Translated: {translated_text}")
        print(f"Context used: {meta['context_used']}, Size: {meta['context_size']}") 