"""
Sound Effects Recommender

This module provides context-aware sound effect recommendations based on
video content, transcript analysis, and scene context. It uses semantic
analysis and machine learning to suggest appropriate sound effects.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
import re
import numpy as np
from collections import Counter

# Optional imports for enhanced NLP capabilities
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class SoundEffectsRecommender:
    """
    Context-aware sound effects recommender that analyzes video content,
    transcripts, and scene context to suggest appropriate sound effects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sound effects recommender.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize sound effects library
        from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary
        self.sound_library = SoundEffectsLibrary(config)
        
        # Context detection weights
        self.weights = {
            "transcript_keywords": 0.4,
            "scene_context": 0.3,
            "visual_elements": 0.2,
            "genre": 0.1
        }
        
        # Pre-define common sound context mappings
        self.context_mappings = {
            # Location contexts
            "indoor": ["room tone", "interior", "indoor ambience"],
            "outdoor": ["wind", "birds", "traffic", "outdoor ambience"],
            "urban": ["traffic", "people", "city", "construction"],
            "nature": ["birds", "wind", "water", "leaves", "forest"],
            "underwater": ["bubbles", "water", "swimming"],
            "space": ["void", "ambient", "sci-fi"],
            
            # Action contexts
            "walking": ["footsteps", "movement"],
            "running": ["fast footsteps", "heavy breathing"],
            "driving": ["engine", "car interior", "road noise"],
            "flying": ["wind", "engine", "airplane"],
            "falling": ["whoosh", "air", "impact"],
            "fighting": ["punch", "impact", "grunt", "struggle"],
            "eating": ["chewing", "silverware", "drinking"],
            "typing": ["keyboard", "computer", "office"],
            
            # Emotional contexts
            "tense": ["heartbeat", "low drone", "suspense"],
            "happy": ["bright", "upbeat", "positive"],
            "sad": ["somber", "quiet", "minimal"],
            "scary": ["creepy", "horror", "tension"],
            "romantic": ["soft", "gentle", "intimate"],
            
            # Event contexts
            "explosion": ["blast", "debris", "destruction"],
            "crash": ["impact", "glass", "metal", "destruction"],
            "rain": ["water", "thunder", "weather"],
            "fire": ["flames", "burning", "crackling"],
            "party": ["crowd", "music", "celebration"],
            "meeting": ["office", "people", "business"],
            "concert": ["audience", "applause", "venue"],
            "sports": ["crowd", "cheering", "stadium"]
        }
        
        # Initialize keyword extraction tools
        if NLTK_AVAILABLE:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set([
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
                "you", "your", "yours", "yourself", "yourselves", "he", "him", 
                "his", "himself", "she", "her", "hers", "herself", "it", "its", 
                "itself", "they", "them", "their", "theirs", "themselves", 
                "what", "which", "who", "whom", "this", "that", "these", 
                "those", "am", "is", "are", "was", "were", "be", "been", 
                "being", "have", "has", "had", "having", "do", "does", "did", 
                "doing", "a", "an", "the", "and", "but", "if", "or", "because", 
                "as", "until", "while", "of", "at", "by", "for", "with", 
                "about", "against", "between", "into", "through", "during", 
                "before", "after", "above", "below", "to", "from", "up", 
                "down", "in", "out", "on", "off", "over", "under", "again", 
                "further", "then", "once", "here", "there", "when", "where", 
                "why", "how", "all", "any", "both", "each", "few", "more", 
                "most", "other", "some", "such", "no", "nor", "not", "only", 
                "own", "same", "so", "than", "too", "very", "s", "t", "can", 
                "will", "just", "don", "should", "now"
            ])
    
    def recommend_for_transcript(
        self,
        transcript: str,
        scene_context: Optional[Dict[str, Any]] = None,
        visual_elements: Optional[List[str]] = None,
        genre: Optional[str] = None,
        max_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Recommend sound effects based on transcript text and optional context information.
        
        Args:
            transcript: Transcript text to analyze
            scene_context: Optional scene context information (indoor/outdoor, etc.)
            visual_elements: Optional list of visual elements detected in the scene
            genre: Optional genre of the content
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            Dictionary with recommended sound effects
        """
        # Start timing
        start_time = time.time()
        
        # Extract keywords from transcript
        keywords = self._extract_keywords(transcript)
        
        # Identify potential trigger words
        trigger_words = self._identify_trigger_words(keywords)
        
        # Determine scene context if not provided
        if not scene_context:
            scene_context = self._infer_scene_context(transcript, keywords)
        
        # Get candidate sound effects
        candidates = self._get_candidate_effects(
            keywords, 
            trigger_words,
            scene_context,
            visual_elements,
            genre
        )
        
        # Rank candidates
        ranked_effects = self._rank_effects(
            candidates,
            keywords,
            scene_context,
            visual_elements,
            genre
        )
        
        # Get top recommendations
        recommendations = ranked_effects[:max_recommendations]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "keywords": keywords[:10],  # Return top 10 keywords
            "trigger_words": trigger_words,
            "inferred_context": scene_context,
            "processing_time": processing_time,
            "total_candidates": len(candidates),
            "total_recommendations": len(recommendations)
        }
    
    def recommend_for_scene(
        self,
        scene_context: Dict[str, Any],
        transcript: Optional[str] = None,
        visual_elements: Optional[List[str]] = None,
        genre: Optional[str] = None,
        duration: Optional[float] = None,
        max_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Recommend sound effects based on scene context and optional transcript.
        
        Args:
            scene_context: Scene context information (indoor/outdoor, etc.)
            transcript: Optional transcript text to analyze
            visual_elements: Optional list of visual elements detected in the scene
            genre: Optional genre of the content
            duration: Optional duration of the scene in seconds
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            Dictionary with recommended sound effects
        """
        # If transcript is provided, use recommend_for_transcript
        if transcript:
            return self.recommend_for_transcript(
                transcript,
                scene_context,
                visual_elements,
                genre,
                max_recommendations
            )
        
        # Start timing
        start_time = time.time()
        
        # Convert scene context to keywords
        keywords = []
        for context_type, context_value in scene_context.items():
            if isinstance(context_value, str):
                keywords.append(context_value)
            elif isinstance(context_value, list):
                keywords.extend(context_value)
        
        # Get candidate sound effects
        candidates = self._get_candidate_effects(
            keywords, 
            [],  # No trigger words without transcript
            scene_context,
            visual_elements,
            genre
        )
        
        # Rank candidates
        ranked_effects = self._rank_effects(
            candidates,
            keywords,
            scene_context,
            visual_elements,
            genre
        )
        
        # Add duration filtering if provided
        if duration is not None:
            # Filter effects that are appropriate for the scene duration
            ranked_effects = [
                effect for effect in ranked_effects
                if self._is_appropriate_duration(effect, duration)
            ]
        
        # Get top recommendations
        recommendations = ranked_effects[:max_recommendations]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "inferred_context": scene_context,
            "processing_time": processing_time,
            "total_candidates": len(candidates),
            "total_recommendations": len(recommendations)
        }
    
    def _extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract and rank keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of (keyword, weight) tuples
        """
        if not text:
            return []
        
        # Lowercase the text
        text = text.lower()
        
        # Tokenize text
        if NLTK_AVAILABLE:
            words = word_tokenize(text)
        else:
            # Simple tokenization fallback
            words = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords and short words
        filtered_words = [
            word for word in words 
            if word not in self.stopwords and len(word) > 2
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Calculate TF (Term Frequency) scores
        total_words = len(filtered_words)
        keywords = [(word, count / total_words) for word, count in word_counts.items()]
        
        # Sort by score
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords
    
    def _identify_trigger_words(self, keywords: List[Tuple[str, float]]) -> List[str]:
        """
        Identify trigger words from keywords that directly map to sound effects.
        
        Args:
            keywords: List of (keyword, weight) tuples
            
        Returns:
            List of identified trigger words
        """
        trigger_words = []
        
        # Get all trigger words from library
        library_triggers = self.sound_library.trigger_words.keys()
        
        # Check each keyword against trigger words
        for keyword, _ in keywords:
            if keyword in library_triggers:
                trigger_words.append(keyword)
                
            # Also check if keyword is part of a multi-word trigger
            for trigger in library_triggers:
                if ' ' in trigger and keyword in trigger.split():
                    trigger_words.append(trigger)
        
        return list(set(trigger_words))  # Remove duplicates
    
    def _infer_scene_context(
        self, 
        transcript: str, 
        keywords: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Infer scene context from transcript and keywords.
        
        Args:
            transcript: Transcript text
            keywords: Extracted keywords
            
        Returns:
            Dictionary with inferred context information
        """
        context = {
            "location": None,
            "time_of_day": None,
            "weather": None,
            "activity": None,
            "mood": None
        }
        
        # Get keyword words only
        keyword_words = [k[0] for k in keywords]
        
        # Check for location indicators
        indoor_indicators = ["room", "inside", "indoor", "house", "building", "office", "home"]
        outdoor_indicators = ["outside", "outdoor", "street", "park", "garden", "forest", "beach"]
        
        for word in indoor_indicators:
            if word in keyword_words or word in transcript.lower():
                context["location"] = "indoor"
                break
                
        if context["location"] is None:
            for word in outdoor_indicators:
                if word in keyword_words or word in transcript.lower():
                    context["location"] = "outdoor"
                    break
        
        # Check for time of day
        time_indicators = {
            "morning": ["morning", "sunrise", "dawn", "early"],
            "day": ["day", "afternoon", "daylight", "noon"],
            "evening": ["evening", "sunset", "dusk"],
            "night": ["night", "midnight", "dark"]
        }
        
        for time, indicators in time_indicators.items():
            for indicator in indicators:
                if indicator in keyword_words or indicator in transcript.lower():
                    context["time_of_day"] = time
                    break
            if context["time_of_day"]:
                break
        
        # Check for weather indicators
        weather_indicators = {
            "rain": ["rain", "rainy", "raining", "thunderstorm", "storm", "wet"],
            "snow": ["snow", "snowy", "snowing", "winter", "cold", "freezing"],
            "windy": ["wind", "windy", "breeze", "gust", "blowing"],
            "sunny": ["sun", "sunny", "clear", "bright", "hot"],
            "cloudy": ["cloud", "cloudy", "overcast", "gray", "grey"]
        }
        
        for weather, indicators in weather_indicators.items():
            for indicator in indicators:
                if indicator in keyword_words or indicator in transcript.lower():
                    context["weather"] = weather
                    break
            if context["weather"]:
                break
        
        # Check for activity
        activity_indicators = {
            "walking": ["walk", "walking", "strolling", "hiking"],
            "running": ["run", "running", "jog", "jogging", "sprint"],
            "driving": ["drive", "driving", "car", "vehicle", "road"],
            "eating": ["eat", "eating", "food", "meal", "dining"],
            "talking": ["talk", "talking", "conversation", "discussing", "chat"],
            "working": ["work", "working", "job", "office", "typing"],
            "sleeping": ["sleep", "sleeping", "bed", "rest", "nap"]
        }
        
        for activity, indicators in activity_indicators.items():
            for indicator in indicators:
                if indicator in keyword_words or indicator in transcript.lower():
                    context["activity"] = activity
                    break
            if context["activity"]:
                break
        
        # Infer mood
        mood_indicators = {
            "happy": ["happy", "joy", "laugh", "smile", "excited", "celebration"],
            "sad": ["sad", "unhappy", "cry", "tears", "depressed", "sorrow"],
            "angry": ["angry", "mad", "furious", "rage", "shouting", "yelling"],
            "tense": ["tense", "nervous", "anxious", "worry", "scared", "fear"],
            "calm": ["calm", "peaceful", "quiet", "relaxed", "gentle", "serene"],
            "romantic": ["love", "romantic", "kiss", "embrace", "passion"]
        }
        
        for mood, indicators in mood_indicators.items():
            for indicator in indicators:
                if indicator in keyword_words or indicator in transcript.lower():
                    context["mood"] = mood
                    break
            if context["mood"]:
                break
        
        return context
    
    def _get_candidate_effects(
        self,
        keywords: List[Any],
        trigger_words: List[str],
        scene_context: Dict[str, Any],
        visual_elements: Optional[List[str]] = None,
        genre: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get candidate sound effects based on keywords and context.
        
        Args:
            keywords: Extracted keywords
            trigger_words: Identified trigger words
            scene_context: Scene context information
            visual_elements: Visual elements detected in the scene
            genre: Genre of the content
            
        Returns:
            List of candidate sound effects
        """
        candidates = []
        
        # Extract keywords as words only
        if isinstance(keywords[0], tuple) if keywords else False:
            keyword_words = [k[0] for k in keywords]
        else:
            keyword_words = keywords
        
        # 1. First, find effects that match trigger words directly
        for trigger in trigger_words:
            if trigger in self.sound_library.trigger_words:
                for effect_id in self.sound_library.trigger_words[trigger]:
                    if effect_id in self.sound_library.metadata:
                        candidates.append(self.sound_library.metadata[effect_id])
        
        # 2. Search by context mappings
        context_keywords = []
        
        # Add location context
        location = scene_context.get("location")
        if location and location in self.context_mappings:
            context_keywords.extend(self.context_mappings[location])
        
        # Add activity context
        activity = scene_context.get("activity")
        if activity and activity in self.context_mappings:
            context_keywords.extend(self.context_mappings[activity])
            
        # Add mood context
        mood = scene_context.get("mood")
        if mood and mood in self.context_mappings:
            context_keywords.extend(self.context_mappings[mood])
            
        # Add weather context
        weather = scene_context.get("weather")
        if weather and weather in self.context_mappings:
            context_keywords.extend(self.context_mappings[weather])
            
        # Search for effects matching context keywords
        for keyword in context_keywords:
            # Search in tags, name, and description
            search_results = self.sound_library.search_effects({
                "query": keyword,
                "fields": ["tags", "name", "description"],
                "limit": 10
            })
            
            if search_results.get("status") == "success":
                for effect in search_results.get("effects", []):
                    candidates.append(effect)
        
        # 3. Search by extracted keywords
        for keyword in keyword_words[:10]:  # Use top 10 keywords
            search_results = self.sound_library.search_effects({
                "query": keyword,
                "fields": ["tags", "name", "description"],
                "limit": 5
            })
            
            if search_results.get("status") == "success":
                for effect in search_results.get("effects", []):
                    candidates.append(effect)
        
        # 4. Include visual elements if provided
        if visual_elements:
            for element in visual_elements:
                search_results = self.sound_library.search_effects({
                    "query": element,
                    "fields": ["tags", "name", "description"],
                    "limit": 3
                })
                
                if search_results.get("status") == "success":
                    for effect in search_results.get("effects", []):
                        candidates.append(effect)
        
        # 5. Include genre-specific effects if provided
        if genre:
            search_results = self.sound_library.search_effects({
                "query": genre,
                "fields": ["tags", "genre", "category"],
                "limit": 5
            })
            
            if search_results.get("status") == "success":
                for effect in search_results.get("effects", []):
                    candidates.append(effect)
        
        # Remove duplicates by effect_id
        unique_candidates = {}
        for effect in candidates:
            effect_id = effect.get("id")
            if effect_id and effect_id not in unique_candidates:
                unique_candidates[effect_id] = effect
        
        return list(unique_candidates.values())
    
    def _rank_effects(
        self,
        candidates: List[Dict[str, Any]],
        keywords: List[Any],
        scene_context: Dict[str, Any],
        visual_elements: Optional[List[str]] = None,
        genre: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate sound effects based on relevance to context.
        
        Args:
            candidates: Candidate sound effects
            keywords: Extracted keywords
            scene_context: Scene context information
            visual_elements: Visual elements detected in the scene
            genre: Genre of the content
            
        Returns:
            List of ranked sound effects
        """
        if not candidates:
            return []
        
        # Extract keywords as words only
        if isinstance(keywords[0], tuple) if keywords else False:
            keyword_words = [k[0] for k in keywords]
        else:
            keyword_words = keywords
        
        # Create a scoring function
        def score_effect(effect):
            score = 0.0
            
            # 1. Score based on keyword matches
            effect_text = " ".join([
                effect.get("name", ""),
                effect.get("description", ""),
                " ".join(effect.get("tags", []))
            ]).lower()
            
            keyword_matches = sum(1 for keyword in keyword_words if keyword in effect_text)
            keyword_score = min(keyword_matches / max(1, len(keyword_words)), 1.0)
            score += keyword_score * self.weights["transcript_keywords"]
            
            # 2. Score based on scene context
            context_score = 0.0
            
            # Check if effect matches location context
            location = scene_context.get("location")
            if location:
                location_match = location in effect_text or any(
                    location_keyword in effect_text 
                    for location_keyword in self.context_mappings.get(location, [])
                )
                context_score += 0.3 if location_match else 0
            
            # Check if effect matches activity context
            activity = scene_context.get("activity")
            if activity:
                activity_match = activity in effect_text or any(
                    activity_keyword in effect_text 
                    for activity_keyword in self.context_mappings.get(activity, [])
                )
                context_score += 0.3 if activity_match else 0
            
            # Check if effect matches mood context
            mood = scene_context.get("mood")
            if mood:
                mood_match = mood in effect_text or any(
                    mood_keyword in effect_text 
                    for mood_keyword in self.context_mappings.get(mood, [])
                )
                context_score += 0.2 if mood_match else 0
                
            # Check if effect matches weather context
            weather = scene_context.get("weather")
            if weather:
                weather_match = weather in effect_text or any(
                    weather_keyword in effect_text 
                    for weather_keyword in self.context_mappings.get(weather, [])
                )
                context_score += 0.2 if weather_match else 0
            
            # Normalize context score
            context_components = sum([
                0.3 if location else 0,
                0.3 if activity else 0,
                0.2 if mood else 0,
                0.2 if weather else 0
            ])
            
            if context_components > 0:
                context_score = context_score / context_components
            
            score += context_score * self.weights["scene_context"]
            
            # 3. Score based on visual elements
            visual_score = 0.0
            if visual_elements:
                visual_matches = sum(1 for element in visual_elements if element in effect_text)
                visual_score = min(visual_matches / len(visual_elements), 1.0)
            
            score += visual_score * self.weights["visual_elements"]
            
            # 4. Score based on genre
            genre_score = 0.0
            if genre and (genre.lower() in effect_text or genre.lower() in effect.get("genre", "").lower()):
                genre_score = 1.0
            
            score += genre_score * self.weights["genre"]
            
            # Add a small random factor for variety (0.05 max)
            score += np.random.random() * 0.05
            
            # Return the effect with its score
            effect_with_score = effect.copy()
            effect_with_score["relevance_score"] = score
            effect_with_score["match_details"] = {
                "keyword_score": keyword_score,
                "context_score": context_score,
                "visual_score": visual_score,
                "genre_score": genre_score
            }
            
            return effect_with_score, score
        
        # Score and rank effects
        scored_effects = [score_effect(effect) for effect in candidates]
        scored_effects.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked effects (without the score)
        return [effect for effect, _ in scored_effects]
    
    def _is_appropriate_duration(self, effect: Dict[str, Any], scene_duration: float) -> bool:
        """
        Check if a sound effect has an appropriate duration for a scene.
        
        Args:
            effect: Sound effect data
            scene_duration: Duration of the scene in seconds
            
        Returns:
            True if appropriate, False otherwise
        """
        # If effect has no duration info, assume it's appropriate
        if "duration" not in effect:
            return True
        
        effect_duration = effect["duration"]
        
        # For very short scenes (< 2 seconds), effect should be short
        if scene_duration < 2.0:
            return effect_duration <= 2.0
            
        # For ambient/background effects, they should be at least as long as the scene
        # or have loop capability
        if "ambient" in effect.get("tags", []) or "background" in effect.get("tags", []):
            return effect_duration >= scene_duration or effect.get("loopable", False)
            
        # For most sound effects, they should be shorter than the scene
        return effect_duration <= scene_duration * 0.8 