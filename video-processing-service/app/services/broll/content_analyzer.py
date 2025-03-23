"""
Content Analyzer for B-Roll suggestions.

This module analyzes transcript content to extract key topics, entities,
actions, and concepts that can be visualized with b-roll footage.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import json
import os
from collections import Counter
import asyncio

# Optional NLP libraries - will be imported conditionally
try:
    import spacy
    import nltk
    from nltk.corpus import stopwords
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """
    Analyzes transcript content to suggest appropriate b-roll footage.
    
    This class extracts key topics, entities, actions, and concepts from
    transcript text that can be visualized with b-roll footage. It uses
    natural language processing to identify important elements in speech.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ContentAnalyzer.
        
        Args:
            config: Configuration options for content analysis
        """
        self.config = config or {}
        self._nlp = None
        self._stopwords = set()
        
        # Initialize NLP components if available
        self._initialize_nlp()
        
        # Load visual keyword database
        self.visual_keywords = self._load_visual_keywords()
        
        # Common action verbs that often indicate visual content
        self.action_verbs = {
            "walk", "run", "jump", "swim", "fly", "drive", "eat", "drink",
            "build", "create", "dance", "sing", "play", "climb", "dive",
            "explore", "travel", "hike", "kayak", "surf", "ski", "cook",
            "paint", "draw", "write", "type", "read", "study"
        }
        
        # Visual topic categories
        self.topic_categories = {
            "nature": ["mountain", "ocean", "forest", "river", "lake", "animal", "wildlife", "sunset", "sunrise"],
            "urban": ["city", "building", "skyscraper", "street", "traffic", "downtown", "skyline"],
            "technology": ["computer", "phone", "device", "screen", "tech", "digital", "software", "hardware"],
            "business": ["office", "meeting", "presentation", "conference", "executive", "workplace"],
            "lifestyle": ["home", "family", "friends", "party", "celebration", "gathering"],
            "sports": ["game", "match", "athlete", "competition", "team", "stadium", "field"],
            "food": ["restaurant", "meal", "cooking", "chef", "dish", "recipe", "ingredient"],
            "travel": ["destination", "tourism", "vacation", "hotel", "resort", "beach", "tourist"]
        }
    
    def _initialize_nlp(self):
        """Initialize NLP components if the required libraries are available."""
        if not SPACY_AVAILABLE:
            logger.warning("Spacy or NLTK not available. Using basic keyword extraction instead.")
            return
        
        try:
            # Download NLTK resources if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            
            # Load stopwords
            self._stopwords = set(stopwords.words('english'))
            
            # Load spaCy model
            model_name = self.config.get('spacy_model', 'en_core_web_sm')
            try:
                import spacy.cli
                try:
                    self._nlp = spacy.load(model_name)
                except OSError:
                    logger.info(f"Downloading spaCy model: {model_name}")
                    spacy.cli.download(model_name)
                    self._nlp = spacy.load(model_name)
            except Exception as e:
                logger.warning(f"Error loading spaCy model: {str(e)}")
                self._nlp = None
        
        except Exception as e:
            logger.warning(f"Error initializing NLP components: {str(e)}")
    
    def _load_visual_keywords(self) -> Dict[str, List[str]]:
        """
        Load database of keywords that have strong visual representations.
        
        Returns:
            Dictionary mapping categories to lists of visual keywords
        """
        # Default simple keyword database
        default_keywords = {
            "locations": [
                "beach", "mountain", "city", "forest", "desert", "ocean",
                "lake", "river", "office", "home", "kitchen", "park"
            ],
            "activities": [
                "running", "swimming", "hiking", "cooking", "driving",
                "working", "studying", "exercising", "playing", "building"
            ],
            "objects": [
                "car", "boat", "plane", "computer", "phone", "book",
                "food", "animal", "building", "tree", "flower", "tool"
            ],
            "concepts": [
                "freedom", "success", "teamwork", "growth", "innovation",
                "technology", "nature", "family", "health", "adventure"
            ]
        }
        
        # Try to load custom keyword database if specified
        custom_db_path = self.config.get('visual_keyword_db')
        if custom_db_path and os.path.exists(custom_db_path):
            try:
                with open(custom_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading custom visual keyword database: {str(e)}")
        
        return default_keywords
    
    async def analyze_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transcript to extract key topics for b-roll visualization.
        
        Args:
            transcript: Transcript dictionary with segments containing text
            
        Returns:
            Dictionary with analysis results including suggested b-roll topics
        """
        # Extract all text from transcript segments
        segments = transcript.get("segments", [])
        all_text = " ".join([segment.get("text", "") for segment in segments])
        
        # Get segment-level suggestions
        segment_suggestions = await self._analyze_segments(segments)
        
        # Get overall content suggestions
        content_topics = await self._extract_key_topics(all_text)
        
        # Get named entities if NLP is available
        entities = await self._extract_entities(all_text) if self._nlp else {}
        
        # Get key action verbs
        actions = self._extract_actions(all_text)
        
        # Organize results
        analysis_results = {
            "overall_topics": content_topics,
            "segment_suggestions": segment_suggestions,
            "named_entities": entities,
            "key_actions": actions,
            "timestamp": segments[0].get("start", 0) if segments else 0,
            "duration": segments[-1].get("end", 0) - segments[0].get("start", 0) if segments else 0
        }
        
        return analysis_results
    
    async def _analyze_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze individual transcript segments for b-roll suggestions.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of segment analysis results with b-roll suggestions
        """
        segment_results = []
        
        # Process segments concurrently for better performance
        tasks = [self._analyze_segment(segment) for segment in segments]
        segment_results = await asyncio.gather(*tasks)
        
        return segment_results
    
    async def _analyze_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single transcript segment for b-roll suggestions.
        
        Args:
            segment: Transcript segment with text and timing
            
        Returns:
            Segment analysis with b-roll suggestions
        """
        text = segment.get("text", "")
        
        # Extract key terms from this segment
        topics = await self._extract_key_topics(text, top_n=3)
        
        # Get visual keywords from this segment
        keywords = self._extract_visual_keywords(text)
        
        # Extract actions
        actions = self._extract_actions(text)
        
        # Determine visual category
        categories = self._categorize_content(text)
        
        # Calculate relevance score (0-10) for b-roll opportunity
        # Higher score = stronger visual opportunity
        relevance = self._calculate_broll_relevance(text, topics, keywords, actions)
        
        # Return segment analysis with b-roll suggestions
        return {
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": text,
            "topics": topics,
            "visual_keywords": keywords,
            "actions": actions,
            "categories": categories,
            "broll_relevance": relevance,
            "suggested_shots": self._suggest_shots(topics, keywords, actions, categories)
        }
    
    async def _extract_key_topics(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key topics from text.
        
        Args:
            text: Text to analyze
            top_n: Number of top topics to return
            
        Returns:
            List of key topics
        """
        if not text.strip():
            return []
        
        if self._nlp:
            # Use spaCy for better topic extraction
            doc = self._nlp(text)
            
            # Extract noun phrases and named entities as potential topics
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                           if not all(token.is_stop for token in chunk)]
            
            # Extract key nouns that aren't stopwords
            nouns = [token.lemma_.lower() for token in doc 
                    if token.pos_ == "NOUN" and token.lemma_.lower() not in self._stopwords 
                    and len(token.lemma_) > 2]
            
            # Combine and count frequencies
            all_topics = noun_phrases + nouns
            topic_counter = Counter(all_topics)
            
            # Return the most common topics
            return [topic for topic, _ in topic_counter.most_common(top_n)]
        else:
            # Fallback to simple keyword extraction
            # Tokenize and clean text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove common stopwords
            basic_stopwords = {"the", "and", "for", "this", "that", "with", "from", "have", "they", "will", "what"}
            filtered_words = [w for w in words if w not in basic_stopwords]
            
            # Count word frequencies
            word_counter = Counter(filtered_words)
            
            # Return the most common words
            return [word for word, _ in word_counter.most_common(top_n)]
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types and values
        """
        if not self._nlp or not text.strip():
            return {}
        
        doc = self._nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            # Avoid duplicates
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        # Map to more user-friendly categories
        user_friendly = {
            "PERSON": "people",
            "GPE": "locations",
            "LOC": "locations",
            "ORG": "organizations",
            "PRODUCT": "products",
            "EVENT": "events",
            "WORK_OF_ART": "works",
            "FAC": "facilities"
        }
        
        # Reorganize with user-friendly keys
        result = {}
        for key, values in entities.items():
            friendly_key = user_friendly.get(key, key.lower())
            if friendly_key not in result:
                result[friendly_key] = []
            result[friendly_key].extend(values)
        
        return result
    
    def _extract_visual_keywords(self, text: str) -> List[str]:
        """
        Extract keywords that have strong visual representations.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of visual keywords found in the text
        """
        text_lower = text.lower()
        found_keywords = []
        
        # Search for visual keywords in each category
        for category, keywords in self.visual_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_actions(self, text: str) -> List[str]:
        """
        Extract action verbs that could be visualized with b-roll.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of action verbs found in the text
        """
        if self._nlp:
            # Use spaCy for better verb extraction
            doc = self._nlp(text)
            
            # Get verbs that match our action verb list or are categorized as actions
            actions = []
            for token in doc:
                if token.pos_ == "VERB":
                    lemma = token.lemma_.lower()
                    # Include if it's a known action verb or seems to be an action
                    if lemma in self.action_verbs or (token.dep_ in ["ROOT", "xcomp"] and len(lemma) > 2):
                        # Include the verb and its object if available
                        verb_phrase = self._get_verb_phrase(token)
                        if verb_phrase and verb_phrase not in actions:
                            actions.append(verb_phrase)
                        elif lemma not in actions:
                            actions.append(lemma)
            
            return actions
        else:
            # Simple action verb extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return [word for word in words if word in self.action_verbs]
    
    def _get_verb_phrase(self, verb_token) -> Optional[str]:
        """
        Extract a verb phrase including the verb and its direct object.
        
        Args:
            verb_token: The verb token from spaCy
            
        Returns:
            Verb phrase or None
        """
        if not verb_token or verb_token.pos_ != "VERB":
            return None
        
        # Find direct object
        obj = None
        for child in verb_token.children:
            if child.dep_ in ["dobj", "obj"]:
                obj = child
                break
        
        if obj:
            # Get the full object phrase
            obj_phrase = ' '.join([t.text for t in obj.subtree])
            return f"{verb_token.lemma_} {obj_phrase}"
        
        return None
    
    def _categorize_content(self, text: str) -> List[str]:
        """
        Categorize content into visual themes.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of visual theme categories
        """
        text_lower = text.lower()
        categories = []
        
        # Check each category for relevant keywords
        for category, keywords in self.topic_categories.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', text_lower) for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _calculate_broll_relevance(
        self, 
        text: str, 
        topics: List[str], 
        keywords: List[str],
        actions: List[str]
    ) -> int:
        """
        Calculate the relevance score for b-roll opportunity.
        
        Args:
            text: Original text
            topics: Extracted topics
            keywords: Visual keywords
            actions: Action verbs
            
        Returns:
            Relevance score from 0-10
        """
        # Initialize score
        score = 0
        
        # Award points for various factors
        
        # Length of text (longer text = more opportunity)
        text_length = len(text.split())
        if text_length > 25:
            score += 1
        
        # Visual keywords
        score += min(3, len(keywords))
        
        # Action verbs
        score += min(3, len(actions))
        
        # Presence of descriptive adjectives
        if self._nlp:
            doc = self._nlp(text)
            descriptive_adj = [token.text for token in doc if token.pos_ == "ADJ"]
            score += min(2, len(descriptive_adj))
        
        # Specific locations or settings mentioned
        loc_indicators = ["in", "at", "on the", "near", "around"]
        if any(indicator in text.lower() for indicator in loc_indicators):
            score += 1
        
        # Cap score at 10
        return min(10, score)
    
    def _suggest_shots(
        self,
        topics: List[str],
        keywords: List[str],
        actions: List[str],
        categories: List[str]
    ) -> List[str]:
        """
        Suggest specific b-roll shots based on analysis.
        
        Args:
            topics: Extracted topics
            keywords: Visual keywords
            actions: Action verbs
            categories: Content categories
            
        Returns:
            List of suggested b-roll shots
        """
        suggestions = []
        
        # Combine all elements
        all_elements = topics + keywords + actions
        
        # Generate specific shot suggestions
        for element in all_elements:
            if element in actions:
                suggestions.append(f"Person {element}")
                
                # Add variations for actions
                if "travel" in element or "journey" in element:
                    suggestions.append("Timelapse of travel scene")
                elif "build" in element or "create" in element:
                    suggestions.append("Close-up of hands working")
            else:
                suggestions.append(f"{element.title()} - wide shot")
                suggestions.append(f"{element.title()} - close-up")
        
        # Add category-specific suggestions
        for category in categories:
            if category == "nature":
                suggestions.append("Nature scenery - wide angle")
                suggestions.append("Wildlife in natural habitat")
            elif category == "urban":
                suggestions.append("City skyline")
                suggestions.append("Street level urban scene")
            elif category == "technology":
                suggestions.append("Technology in use - close-up")
                suggestions.append("Digital interface animation")
            elif category == "business":
                suggestions.append("Professional workplace environment")
                suggestions.append("Business meeting or presentation")
        
        # Remove duplicates and limit to reasonable number
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:10] 