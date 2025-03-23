"""
Transcript Analysis Module

This module handles transcript analysis for detecting interesting moments,
including sentiment analysis, keyword/phrase importance scoring, and
narrative cohesion analysis.
"""

import os
import re
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment in transcript text to detect interesting moments.
    
    Features:
    - Sentiment peak detection
    - Emotional reaction identification
    - Positive/negative transition detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sentiment_threshold = config.get("sentiment_threshold", 0.7)
        self.sentiment_window_size = config.get("sentiment_window_size", 50)
        
        # Try to load advanced NLP libraries
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            self._has_nltk = True
        except ImportError:
            logger.warning("NLTK not available. Using simplified sentiment analysis.")
            self._has_nltk = False
        
        # Try to load transformers for better sentiment analysis
        try:
            import torch
            from transformers import pipeline
            self._has_transformers = True
            
            # Only load the model if we have transformers
            # This is a resource-intensive step, so we defer it until needed
            self.sentiment_pipeline = None
        except ImportError:
            logger.warning("Transformers library not available. Using NLTK or simplified analysis.")
            self._has_transformers = False
        
        logger.info("Initialized SentimentAnalyzer")
    
    def _ensure_transformers_model(self):
        """Ensure the transformers model is loaded (lazy loading)."""
        if self._has_transformers and self.sentiment_pipeline is None:
            try:
                from transformers import pipeline
                logger.info("Loading sentiment analysis model...")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logger.info("Sentiment analysis model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load transformers model: {str(e)}")
                self._has_transformers = False
    
    def analyze_sentiment_nltk(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using NLTK's VADER sentiment analyzer.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self._has_nltk:
            # Fallback to very simple analysis
            return self._simple_sentiment_analysis(text)
        
        scores = self.sia.polarity_scores(text)
        
        # Determine overall sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Normalize scores to 0-1 range
        pos_norm = scores['pos']
        neg_norm = scores['neg']
        
        # Calculate intensity (0-1 scale)
        intensity = abs(compound)
        
        return {
            "sentiment": sentiment,
            "compound": compound,
            "positive": pos_norm,
            "negative": neg_norm,
            "neutral": scores['neu'],
            "intensity": intensity
        }
    
    def analyze_sentiment_transformers(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using a pre-trained transformer model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self._has_transformers:
            return self.analyze_sentiment_nltk(text)
        
        # Ensure model is loaded
        self._ensure_transformers_model()
        
        # If loading failed, fall back to NLTK
        if not self._has_transformers:
            return self.analyze_sentiment_nltk(text)
        
        try:
            # Truncate text if needed (transformers have token limits)
            max_length = 512
            if len(text.split()) > max_length:
                text = " ".join(text.split()[:max_length])
            
            # Run sentiment analysis
            result = self.sentiment_pipeline(text)[0]
            
            # Parse result
            label = result['label'].lower()
            score = result['score']
            
            # Transform to match NLTK output format
            if label == 'positive':
                compound = score
                pos_norm = score
                neg_norm = 1 - score
            else:  # negative
                compound = -score
                pos_norm = 1 - score
                neg_norm = score
            
            # Calculate intensity (0-1 scale)
            intensity = score
            
            return {
                "sentiment": label,
                "compound": compound,
                "positive": pos_norm,
                "negative": neg_norm,
                "neutral": 0,  # Transformers typically don't give a neutral score
                "intensity": intensity
            }
            
        except Exception as e:
            logger.error(f"Error in transformers sentiment analysis: {str(e)}")
            return self.analyze_sentiment_nltk(text)
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Very simple sentiment analysis fallback using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text = text.lower()
        
        # Simple positive and negative keyword lists
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "terrific", "outstanding", "superb", "brilliant", "awesome",
            "love", "happy", "joy", "exciting", "best", "beautiful", "perfect"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "poor", "disappointing",
            "worst", "hate", "sad", "angry", "annoying", "frustrating",
            "useless", "waste", "failure", "ugly", "stupid", "wrong"
        }
        
        # Count occurrences
        pos_count = sum(1 for word in text.split() if word.strip(".,!?;:") in positive_words)
        neg_count = sum(1 for word in text.split() if word.strip(".,!?;:") in negative_words)
        total_words = len(text.split())
        
        # Calculate normalized scores
        pos_norm = pos_count / max(total_words, 1)
        neg_norm = neg_count / max(total_words, 1)
        
        # Compound score (simple difference)
        compound = pos_norm - neg_norm
        
        # Determine sentiment
        if compound > 0.05:
            sentiment = "positive"
        elif compound < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Intensity based on absolute compound value
        intensity = abs(compound)
        
        return {
            "sentiment": sentiment,
            "compound": compound,
            "positive": pos_norm,
            "negative": neg_norm,
            "neutral": 1 - (pos_norm + neg_norm),
            "intensity": intensity
        }
    
    def detect_sentiment_peaks(
        self, 
        transcript: List[Dict[str, Any]]
    ) -> List[Tuple[float, float, float, Dict[str, Any]]]:
        """
        Detect peaks in sentiment intensity from a transcript.
        
        Args:
            transcript: List of transcript segments with text and timestamps
            
        Returns:
            List of tuples (start_time, end_time, intensity, metadata)
        """
        if not transcript:
            logger.warning("Empty transcript provided")
            return []
        
        # Analyze sentiment for each segment
        segments_with_sentiment = []
        for segment in transcript:
            text = segment.get("text", "")
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # Skip empty segments
            if not text.strip():
                continue
            
            # Use transformers if available, fallback to NLTK
            if self._has_transformers:
                sentiment = self.analyze_sentiment_transformers(text)
            else:
                sentiment = self.analyze_sentiment_nltk(text)
            
            segments_with_sentiment.append({
                "text": text,
                "start": start_time,
                "end": end_time,
                "sentiment": sentiment
            })
        
        # Find segments with high sentiment intensity
        results = []
        
        for segment in segments_with_sentiment:
            sentiment = segment["sentiment"]
            intensity = sentiment["intensity"]
            
            # Only consider segments above threshold
            if intensity >= self.sentiment_threshold:
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Create metadata
                metadata = {
                    "text": segment["text"],
                    "sentiment_type": sentiment["sentiment"],
                    "positive_score": sentiment["positive"],
                    "negative_score": sentiment["negative"],
                    "analysis_method": "transformers" if self._has_transformers else "nltk" if self._has_nltk else "simple"
                }
                
                results.append((start_time, end_time, intensity, metadata))
        
        # Also detect sentiment shifts (from positive to negative or vice versa)
        if len(segments_with_sentiment) > 1:
            for i in range(1, len(segments_with_sentiment)):
                prev_segment = segments_with_sentiment[i-1]
                curr_segment = segments_with_sentiment[i]
                
                prev_sentiment = prev_segment["sentiment"]["sentiment"]
                curr_sentiment = curr_segment["sentiment"]["sentiment"]
                
                # Check for sentiment shift
                if ((prev_sentiment == "positive" and curr_sentiment == "negative") or
                    (prev_sentiment == "negative" and curr_sentiment == "positive")):
                    
                    # Create a moment that spans both segments
                    start_time = prev_segment["start"]
                    end_time = curr_segment["end"]
                    
                    # Calculate intensity based on the average of both segments
                    avg_intensity = (prev_segment["sentiment"]["intensity"] + 
                                     curr_segment["sentiment"]["intensity"]) / 2
                    
                    # Add bonus for the shift
                    shift_intensity = min(avg_intensity + 0.2, 1.0)
                    
                    metadata = {
                        "text": f"{prev_segment['text']} ... {curr_segment['text']}",
                        "sentiment_shift": f"{prev_sentiment}_to_{curr_sentiment}",
                        "analysis_method": "transformers" if self._has_transformers else "nltk" if self._has_nltk else "simple"
                    }
                    
                    results.append((start_time, end_time, shift_intensity, metadata))
        
        logger.info(f"Detected {len(results)} sentiment peaks")
        return results


class KeywordAnalyzer:
    """
    Analyzes transcripts for important keywords and phrases.
    
    Features:
    - Keyword importance scoring
    - Phrase significance detection
    - Named entity recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the keyword analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.importance_threshold = config.get("keyword_importance_threshold", 0.6)
        
        # Load language processing tools if available
        try:
            import nltk
            from nltk.corpus import stopwords
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.stopwords = set(stopwords.words('english'))
            self._has_nltk = True
        except ImportError:
            logger.warning("NLTK not available. Using simplified keyword analysis.")
            # Define a minimal set of stopwords
            self.stopwords = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
                              "at", "from", "by", "for", "with", "about", "against", "between",
                              "into", "through", "during", "before", "after", "above", "below",
                              "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
                              "being", "have", "has", "had", "having", "do", "does", "did",
                              "doing", "i", "you", "he", "she", "it", "we", "they", "their",
                              "this", "that", "these", "those", "am", "is", "are", "was", "were"}
            self._has_nltk = False
        
        # Try to load spaCy for better NER and keyword extraction
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self._has_spacy = True
        except (ImportError, OSError):
            logger.warning("spaCy not available. Using simplified keyword extraction.")
            self._has_spacy = False
        
        logger.info("Initialized KeywordAnalyzer")
    
    def extract_keywords_spacy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords using spaCy with NER and noun chunk extraction.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries with keyword information
        """
        if not self._has_spacy or not text.strip():
            return []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            keywords = []
            
            # Extract named entities
            for entity in doc.ents:
                importance = 0.7  # Base importance for named entities
                
                # Adjust importance based on entity type
                if entity.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                    importance += 0.1
                
                keywords.append({
                    "text": entity.text,
                    "type": "entity",
                    "entity_type": entity.label_,
                    "importance": min(importance, 1.0),
                    "start_char": entity.start_char,
                    "end_char": entity.end_char
                })
            
            # Extract noun chunks (meaningful phrases)
            for chunk in doc.noun_chunks:
                # Skip chunks that are already covered by entities
                if any(k["start_char"] <= chunk.start_char and k["end_char"] >= chunk.end_char 
                       for k in keywords):
                    continue
                
                # Skip chunks that are just stopwords
                if all(token.is_stop for token in chunk):
                    continue
                
                # Calculate importance based on chunk properties
                importance = 0.5  # Base importance for noun chunks
                
                # Bonus for longer chunks (more likely to be meaningful)
                if len(chunk) > 2:
                    importance += 0.1
                
                # Bonus for chunks with proper nouns
                if any(token.pos_ == "PROPN" for token in chunk):
                    importance += 0.1
                
                keywords.append({
                    "text": chunk.text,
                    "type": "noun_chunk",
                    "importance": min(importance, 1.0),
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                })
            
            # Extract other important tokens (adjectives, verbs)
            for token in doc:
                # Skip tokens already covered
                if any(k["start_char"] <= token.idx and k["end_char"] >= token.idx + len(token.text) 
                       for k in keywords):
                    continue
                
                # Skip stopwords and punctuation
                if token.is_stop or token.is_punct:
                    continue
                
                # Focus on adjectives, adverbs, and verbs
                if token.pos_ in ["ADJ", "ADV", "VERB"]:
                    importance = 0.4  # Base importance
                    
                    # Adjust based on part of speech
                    if token.pos_ == "ADJ":
                        importance += 0.1
                    elif token.pos_ == "VERB":
                        # Check if it's a significant verb (not auxiliary)
                        if not token.dep_ in ["aux", "auxpass"]:
                            importance += 0.1
                    
                    keywords.append({
                        "text": token.text,
                        "type": "token",
                        "pos": token.pos_,
                        "importance": min(importance, 1.0),
                        "start_char": token.idx,
                        "end_char": token.idx + len(token.text)
                    })
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords with spaCy: {str(e)}")
            return []
    
    def extract_keywords_simple(self, text: str) -> List[Dict[str, Any]]:
        """
        Simple keyword extraction using frequency and basic rules.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries with keyword information
        """
        if not text.strip():
            return []
        
        try:
            # Clean and tokenize text
            text = text.lower()
            words = re.findall(r'\b\w+\b', text)
            
            # Remove stopwords
            filtered_words = [w for w in words if w not in self.stopwords and len(w) > 2]
            
            # Count word frequencies
            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Find bigrams (simple consecutive pairs)
            bigrams = []
            for i in range(len(words) - 1):
                if words[i] not in self.stopwords or words[i+1] not in self.stopwords:
                    bigram = f"{words[i]} {words[i+1]}"
                    bigrams.append(bigram)
            
            # Count bigram frequencies
            bigram_counts = {}
            for bigram in bigrams:
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            
            # Create keyword entries
            keywords = []
            
            # Add unigrams
            for word, count in word_counts.items():
                # Skip very common words
                if count > 5:
                    continue
                
                # Calculate importance based on frequency and length
                importance = 0.4  # Base importance
                
                # Bonus for capitalized words (potential proper nouns)
                if any(c in text for c in [f" {word.capitalize()}", f"{word.capitalize()} "]):
                    importance += 0.2
                
                # Bonus for longer words
                if len(word) > 6:
                    importance += 0.1
                
                keywords.append({
                    "text": word,
                    "type": "unigram",
                    "importance": min(importance, 1.0),
                    "frequency": count,
                    "start_char": text.find(word),
                    "end_char": text.find(word) + len(word)
                })
            
            # Add bigrams
            for bigram, count in bigram_counts.items():
                # Skip infrequent bigrams
                if count < 2:
                    continue
                
                # Calculate importance
                importance = 0.5  # Base importance for bigrams
                
                # Bonus for capitalized bigrams
                if any(c in text for c in [f" {bigram.capitalize()}", f"{bigram.capitalize()} "]):
                    importance += 0.2
                
                keywords.append({
                    "text": bigram,
                    "type": "bigram",
                    "importance": min(importance, 1.0),
                    "frequency": count,
                    "start_char": text.find(bigram),
                    "end_char": text.find(bigram) + len(bigram)
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error in simple keyword extraction: {str(e)}")
            return []
    
    def detect_important_phrases(
        self, 
        transcript: List[Dict[str, Any]]
    ) -> List[Tuple[float, float, float, Dict[str, Any]]]:
        """
        Detect important keywords and phrases in a transcript.
        
        Args:
            transcript: List of transcript segments with text and timestamps
            
        Returns:
            List of tuples (start_time, end_time, importance, metadata)
        """
        if not transcript:
            logger.warning("Empty transcript provided")
            return []
        
        results = []
        
        for segment in transcript:
            text = segment.get("text", "")
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # Skip empty segments
            if not text.strip():
                continue
            
            # Extract keywords using spaCy if available, otherwise use simple method
            if self._has_spacy:
                keywords = self.extract_keywords_spacy(text)
            else:
                keywords = self.extract_keywords_simple(text)
            
            # Filter by importance threshold
            important_keywords = [k for k in keywords if k["importance"] >= self.importance_threshold]
            
            # If we have important keywords, create a moment
            if important_keywords:
                # Sort by importance (descending)
                important_keywords.sort(key=lambda k: k["importance"], reverse=True)
                
                # Calculate average importance
                avg_importance = sum(k["importance"] for k in important_keywords) / len(important_keywords)
                
                # Create metadata
                metadata = {
                    "text": text,
                    "keywords": [k["text"] for k in important_keywords[:5]],  # Top 5 keywords
                    "top_keyword": important_keywords[0]["text"],
                    "top_keyword_importance": important_keywords[0]["importance"],
                    "analysis_method": "spacy" if self._has_spacy else "simple"
                }
                
                results.append((start_time, end_time, avg_importance, metadata))
        
        logger.info(f"Detected {len(results)} segments with important keywords")
        return results


class TranscriptAnalyzer:
    """
    Main class for analyzing transcripts to detect interesting moments.
    
    Coordinates various analysis components including sentiment analysis,
    keyword analysis, and narrative cohesion analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transcript analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.temp_dir = Path(config.get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(config)
        
        # Initialize keyword analyzer
        self.keyword_analyzer = KeywordAnalyzer(config)
        
        logger.info("Initialized TranscriptAnalyzer")
    
    def load_transcript(self, transcript_path: str) -> List[Dict[str, Any]]:
        """
        Load a transcript file.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            List of transcript segments
        """
        if not os.path.exists(transcript_path):
            logger.error(f"Transcript file not found: {transcript_path}")
            return []
        
        try:
            # Determine file format based on extension
            ext = os.path.splitext(transcript_path)[1].lower()
            
            if ext == '.json':
                # Load JSON transcript
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(transcript_data, list):
                    # List of segments
                    return transcript_data
                elif isinstance(transcript_data, dict):
                    # Check for common transcript formats
                    if 'segments' in transcript_data:
                        return transcript_data['segments']
                    elif 'results' in transcript_data:
                        return transcript_data['results']
                    else:
                        logger.warning(f"Unknown JSON transcript format: {transcript_path}")
                        return []
                else:
                    logger.warning(f"Invalid JSON transcript format: {transcript_path}")
                    return []
            
            elif ext in ['.vtt', '.srt']:
                # Parse WebVTT or SRT format
                segments = []
                
                with open(transcript_path, 'r') as f:
                    content = f.read()
                
                # Simple regex pattern for timestamps and text
                # This is a basic parser and might need enhancement for edge cases
                if ext == '.vtt':
                    pattern = r'(\d{2}:\d{2}:\d{2}.\d{3}) --> (\d{2}:\d{2}:\d{2}.\d{3})\s*\n((?:.+\n)+)'
                else:  # .srt
                    pattern = r'\d+\s*\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*\n((?:.+\n)+)'
                
                matches = re.findall(pattern, content)
                
                for start_str, end_str, text in matches:
                    # Convert timestamp strings to seconds
                    start_seconds = self._timestamp_to_seconds(start_str)
                    end_seconds = self._timestamp_to_seconds(end_str)
                    
                    # Clean text
                    text = text.strip()
                    
                    segments.append({
                        "start": start_seconds,
                        "end": end_seconds,
                        "text": text
                    })
                
                return segments
            
            else:
                # Text file - treat as raw transcript without timestamps
                with open(transcript_path, 'r') as f:
                    content = f.read()
                
                # Create a single segment for the whole content
                return [{
                    "start": 0,
                    "end": 0,  # Will be updated if video duration is known
                    "text": content
                }]
                
        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            return []
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """
        Convert a timestamp string to seconds.
        
        Args:
            timestamp: Timestamp string (HH:MM:SS.mmm or HH:MM:SS,mmm)
            
        Returns:
            Seconds as float
        """
        # Normalize format
        timestamp = timestamp.replace(',', '.')
        
        # Parse components
        parts = timestamp.split(':')
        
        if len(parts) == 3:
            # HH:MM:SS.mmm
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # MM:SS.mmm
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            # Invalid format
            logger.warning(f"Invalid timestamp format: {timestamp}")
            return 0.0
    
    def analyze_transcript(
        self, 
        transcript_path: str
    ) -> List[Tuple[str, float, float, float, Dict[str, Any]]]:
        """
        Analyze a transcript to find interesting moments.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            List of tuples (moment_type, start_time, end_time, score, metadata)
        """
        if not os.path.exists(transcript_path):
            logger.error(f"Transcript file not found: {transcript_path}")
            return []
        
        logger.info(f"Analyzing transcript: {transcript_path}")
        
        # Load transcript
        transcript = self.load_transcript(transcript_path)
        
        if not transcript:
            logger.warning(f"Failed to load transcript or empty transcript: {transcript_path}")
            return []
        
        results = []
        
        # Analyze sentiment
        sentiment_results = self.sentiment_analyzer.detect_sentiment_peaks(transcript)
        for start_time, end_time, score, metadata in sentiment_results:
            moment_type = "sentiment_peak"
            results.append((moment_type, start_time, end_time, score, metadata))
        
        # Analyze keywords
        keyword_results = self.keyword_analyzer.detect_important_phrases(transcript)
        for start_time, end_time, score, metadata in keyword_results:
            moment_type = "keyword"
            results.append((moment_type, start_time, end_time, score, metadata))
        
        # Note: Narrative cohesion analysis will be added in subsequent steps
        
        logger.info(f"Transcript analysis complete. Detected {len(results)} potential moments")
        return results 