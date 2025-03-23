"""
Semantic Matcher for B-Roll content.

This module provides advanced semantic matching between transcript content
and visual concepts to improve B-Roll suggestions relevance.
"""

import logging
import os
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import asyncio
from collections import defaultdict
import math

# Optional NLP libraries for advanced matching
try:
    import spacy
    import nltk
    from nltk.corpus import wordnet
    from nltk.metrics import edit_distance
    ADVANCED_NLP = True
except ImportError:
    ADVANCED_NLP = False

logger = logging.getLogger(__name__)

class SemanticMatcher:
    """
    Matches transcript content with visual concepts using semantic analysis.
    
    This class enhances B-Roll suggestions by finding deeper connections
    between speech content and potential visual representations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SemanticMatcher.
        
        Args:
            config: Configuration options for semantic matching
        """
        self.config = config or {}
        self._nlp = None
        self._word_vectors = None
        self._concept_database = None
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Load concept database
        self._load_concept_database()
        
        # Visual concept categories and their related terms
        self.visual_categories = {
            "landscape": ["mountain", "ocean", "forest", "river", "lake", "desert", "valley", "field", "sunset", "sunrise"],
            "urban": ["city", "building", "skyscraper", "street", "traffic", "downtown", "skyline", "architecture", "bridge"],
            "people": ["person", "crowd", "meeting", "presentation", "team", "audience", "interview", "conversation"],
            "technology": ["computer", "device", "phone", "screen", "digital", "software", "hardware", "robot", "circuit"],
            "abstract": ["concept", "idea", "creativity", "innovation", "growth", "success", "failure", "challenge", "process"],
            "business": ["office", "meeting", "presentation", "conference", "corporate", "management", "executive", "startup"],
            "education": ["school", "university", "classroom", "student", "teacher", "learning", "study", "research", "book"],
            "science": ["laboratory", "experiment", "research", "scientist", "chemical", "biology", "physics", "astronomy"],
            "nature": ["animal", "plant", "flower", "tree", "wildlife", "ecosystem", "environment", "biodiversity"],
            "health": ["medical", "hospital", "doctor", "patient", "treatment", "exercise", "fitness", "wellness", "medicine"]
        }
        
        # Pre-compute similarity threshold
        self.similarity_threshold = self.config.get('similarity_threshold', 0.65)
        
        # Cache for computed similarities
        self.similarity_cache = {}
    
    def _initialize_nlp(self):
        """Initialize NLP components for semantic matching."""
        if not ADVANCED_NLP:
            logger.warning("Advanced NLP libraries not available. Using basic matching.")
            return
        
        try:
            # Ensure NLTK data is available
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
            
            # Initialize spaCy with word vectors
            model_name = self.config.get('spacy_model', 'en_core_web_md')
            try:
                import spacy.cli
                try:
                    self._nlp = spacy.load(model_name)
                except OSError:
                    logger.info(f"Downloading spaCy model: {model_name}")
                    spacy.cli.download(model_name)
                    self._nlp = spacy.load(model_name)
                
                # Check if the model has word vectors
                if not self._nlp.has_pipe('tok2vec'):
                    logger.warning(f"Model {model_name} does not have word vectors. Semantic matching will be limited.")
            
            except Exception as e:
                logger.warning(f"Error loading spaCy model: {str(e)}")
                self._nlp = None
        
        except Exception as e:
            logger.warning(f"Error initializing NLP components: {str(e)}")
    
    def _load_concept_database(self):
        """
        Load visual concept database with semantic information.
        """
        # Default simple concept database
        default_concepts = {
            "abstract_concepts": {
                "freedom": ["liberty", "freedom", "independence", "open space", "flying", "soaring", "bird", "sky", "breaking chains"],
                "success": ["achievement", "winning", "celebration", "trophy", "medal", "summit", "top", "goal", "accomplishment"],
                "growth": ["plant growing", "chart rising", "child growing", "development", "increase", "expansion", "evolution"],
                "innovation": ["light bulb", "new idea", "breakthrough", "invention", "creativity", "technology", "progress"],
                "challenge": ["obstacle", "mountain climbing", "competition", "difficult path", "struggle", "overcoming"]
            },
            "emotions": {
                "happiness": ["smile", "laughter", "joy", "celebration", "sunny", "bright", "dancing", "party"],
                "sadness": ["tears", "rain", "dark clouds", "lonely", "isolation", "slow motion", "downward"],
                "anger": ["red", "fire", "intense", "storm", "conflict", "explosion", "tension"],
                "fear": ["darkness", "shadow", "running", "hiding", "suspense", "thriller", "horror"],
                "surprise": ["shock", "amazement", "unexpected", "revelation", "sudden", "discovery"]
            },
            "actions": {
                "communication": ["talking", "speaking", "dialogue", "phone call", "message", "conversation", "discussion"],
                "movement": ["walking", "running", "driving", "flying", "swimming", "journey", "travel"],
                "creation": ["building", "making", "crafting", "designing", "writing", "painting", "coding"],
                "analysis": ["studying", "researching", "examining", "measuring", "calculating", "investigating"],
                "collaboration": ["teamwork", "partnership", "cooperation", "together", "meeting", "joint effort"]
            },
            "themes": {
                "time": ["clock", "hourglass", "calendar", "aging", "history", "future", "deadlines", "evolution"],
                "space": ["universe", "galaxy", "stars", "planet", "cosmos", "exploration", "vastness", "astronomy"],
                "power": ["strength", "authority", "influence", "energy", "control", "leadership", "domination"],
                "knowledge": ["books", "library", "learning", "wisdom", "information", "data", "education"],
                "transformation": ["change", "metamorphosis", "evolution", "development", "transition", "before and after"]
            }
        }
        
        # Try to load custom concept database if specified
        custom_db_path = self.config.get('concept_database_path')
        if custom_db_path and os.path.exists(custom_db_path):
            try:
                with open(custom_db_path, 'r') as f:
                    self._concept_database = json.load(f)
                    logger.info(f"Loaded custom concept database from {custom_db_path}")
            except Exception as e:
                logger.warning(f"Error loading custom concept database: {str(e)}")
                self._concept_database = default_concepts
        else:
            self._concept_database = default_concepts
    
    async def match_concepts_to_transcript(
        self, 
        transcript_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find semantic matches between transcript segments and visual concepts.
        
        Args:
            transcript_segments: List of transcript segments with text content
            
        Returns:
            Dictionary mapping segments to visual concept matches
        """
        # Process segments concurrently
        tasks = [self._match_segment_concepts(segment) for segment in transcript_segments]
        segment_matches = await asyncio.gather(*tasks)
        
        # Combine results
        segments_with_concepts = []
        for i, (segment, matches) in enumerate(zip(transcript_segments, segment_matches)):
            segments_with_concepts.append({
                "segment_id": i,
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", ""),
                "concept_matches": matches
            })
        
        # Get global theme consistency
        theme_consistency = self._analyze_theme_consistency(segment_matches)
        
        return {
            "segments": segments_with_concepts,
            "theme_consistency": theme_consistency
        }
    
    async def _match_segment_concepts(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match visual concepts to a single transcript segment.
        
        Args:
            segment: Transcript segment with text content
            
        Returns:
            Dictionary of matched concepts with confidence scores
        """
        text = segment.get("text", "")
        if not text:
            return {}
        
        # Extract key terms from the segment
        key_terms = await self._extract_key_terms(text)
        
        # Match each term to visual concepts
        concept_matches = {}
        
        # Match abstract concepts
        for concept_category, concepts in self._concept_database.items():
            for concept, related_terms in concepts.items():
                confidence = self._calculate_concept_match_confidence(key_terms, concept, related_terms)
                if confidence > self.similarity_threshold:
                    concept_matches[concept] = {
                        "confidence": confidence,
                        "category": concept_category,
                        "related_terms": related_terms[:5],  # Limit to top 5 related terms
                        "visual_suggestions": self._get_visual_suggestions_for_concept(concept, related_terms)
                    }
        
        # Match visual categories
        for category, terms in self.visual_categories.items():
            confidence = self._calculate_category_match_confidence(key_terms, terms)
            if confidence > self.similarity_threshold:
                concept_matches[category] = {
                    "confidence": confidence,
                    "category": "visual_category",
                    "related_terms": terms[:5],  # Limit to top 5 related terms
                    "visual_suggestions": self._get_visual_suggestions_for_category(category)
                }
        
        # Sort matches by confidence and limit the number of results
        sorted_matches = dict(sorted(
            concept_matches.items(), 
            key=lambda item: item[1]["confidence"], 
            reverse=True
        )[:10])  # Limit to top 10 matches
        
        return sorted_matches
    
    async def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text that might have visual representations.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key terms
        """
        if not text.strip():
            return []
        
        if self._nlp:
            # Use spaCy for more advanced extraction
            doc = self._nlp(text)
            
            # Extract key terms (nouns, verbs, adjectives, named entities)
            key_terms = []
            
            # Add named entities
            key_terms.extend([ent.text.lower() for ent in doc.ents])
            
            # Add nouns, verbs, and adjectives
            key_terms.extend([
                token.lemma_.lower() for token in doc 
                if token.pos_ in ["NOUN", "VERB", "ADJ"] 
                and len(token.lemma_) > 2
                and not token.is_stop
            ])
            
            # Add noun chunks (compound terms)
            key_terms.extend([
                chunk.text.lower() for chunk in doc.noun_chunks
                if len(chunk.text) > 3
            ])
            
            return key_terms
        else:
            # Basic term extraction
            # Tokenize and clean text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove common stopwords
            basic_stopwords = {"the", "and", "for", "this", "that", "with", "from", "have", "they", "will", "what"}
            return [w for w in words if w not in basic_stopwords]
    
    def _calculate_concept_match_confidence(
        self, 
        key_terms: List[str],
        concept: str,
        related_terms: List[str]
    ) -> float:
        """
        Calculate confidence score for a match between key terms and a visual concept.
        
        Args:
            key_terms: Key terms extracted from text
            concept: Visual concept
            related_terms: Terms related to the visual concept
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not key_terms:
            return 0.0
        
        # Check exact matches first
        if concept in key_terms:
            return 1.0
        
        # Check if any related terms are exact matches
        exact_matches = [term for term in related_terms if term in key_terms]
        if exact_matches:
            return 0.9
        
        # Calculate semantic similarity using word vectors or WordNet
        max_similarity = 0.0
        
        for term in key_terms:
            # Check cache first
            cache_key = f"{term}_{concept}"
            if cache_key in self.similarity_cache:
                similarity = self.similarity_cache[cache_key]
            else:
                similarity = self._calculate_term_similarity(term, concept)
                self.similarity_cache[cache_key] = similarity
            
            max_similarity = max(max_similarity, similarity)
            
            # Also check similarity with related terms
            for related_term in related_terms:
                cache_key = f"{term}_{related_term}"
                if cache_key in self.similarity_cache:
                    similarity = self.similarity_cache[cache_key]
                else:
                    similarity = self._calculate_term_similarity(term, related_term)
                    self.similarity_cache[cache_key] = similarity
                
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_category_match_confidence(
        self, 
        key_terms: List[str],
        category_terms: List[str]
    ) -> float:
        """
        Calculate confidence score for a match between key terms and a visual category.
        
        Args:
            key_terms: Key terms extracted from text
            category_terms: Terms associated with the visual category
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not key_terms:
            return 0.0
        
        # Check exact matches with category terms
        exact_matches = [term for term in category_terms if term in key_terms]
        if exact_matches:
            return 0.9
        
        # Calculate semantic similarity with category terms
        max_similarity = 0.0
        for term in key_terms:
            for category_term in category_terms:
                cache_key = f"{term}_{category_term}"
                if cache_key in self.similarity_cache:
                    similarity = self.similarity_cache[cache_key]
                else:
                    similarity = self._calculate_term_similarity(term, category_term)
                    self.similarity_cache[cache_key] = similarity
                
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_term_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate semantic similarity between two terms.
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # If terms are identical
        if term1 == term2:
            return 1.0
        
        # Check if one term contains the other
        if term1 in term2 or term2 in term1:
            return 0.8
        
        if self._nlp and self._nlp.has_pipe('tok2vec'):
            # Use word vectors for semantic similarity
            try:
                doc1 = self._nlp(term1)
                doc2 = self._nlp(term2)
                
                # Calculate vector similarity
                similarity = doc1.similarity(doc2)
                return float(similarity)
            except Exception as e:
                logger.debug(f"Error calculating vector similarity: {str(e)}")
        
        if ADVANCED_NLP:
            # Fallback to WordNet similarity
            try:
                # Get WordNet synsets for the terms
                synsets1 = wordnet.synsets(term1)
                synsets2 = wordnet.synsets(term2)
                
                if synsets1 and synsets2:
                    # Calculate maximum path similarity
                    max_similarity = 0.0
                    for s1 in synsets1:
                        for s2 in synsets2:
                            try:
                                similarity = s1.path_similarity(s2)
                                if similarity is not None:
                                    max_similarity = max(max_similarity, similarity)
                            except:
                                pass
                    
                    return max_similarity
            except Exception as e:
                logger.debug(f"Error calculating WordNet similarity: {str(e)}")
        
        # Fallback to string similarity (Levenshtein distance)
        # Normalize by the length of the longest term
        distance = edit_distance(term1, term2)
        max_length = max(len(term1), len(term2))
        
        # Convert to similarity score
        if max_length > 0:
            return 1.0 - (distance / max_length)
        else:
            return 0.0
    
    def _get_visual_suggestions_for_concept(
        self, 
        concept: str, 
        related_terms: List[str]
    ) -> List[str]:
        """
        Get specific visual suggestions for a matched concept.
        
        Args:
            concept: The matched concept
            related_terms: Terms related to the concept
            
        Returns:
            List of visual suggestion strings
        """
        suggestions = []
        
        # Add concept-specific suggestions
        suggestions.append(f"{concept.title()} - wide shot")
        suggestions.append(f"{concept.title()} - close-up")
        
        # Add suggestions using related terms
        for term in related_terms[:3]:  # Use top 3 related terms
            suggestions.append(f"{term.title()} visual")
        
        # Add specific suggestions based on concept type
        if concept in ["success", "achievement", "winning"]:
            suggestions.extend([
                "Celebration scene",
                "Trophy or award",
                "Team celebrating",
                "Summit or peak reached"
            ])
        elif concept in ["challenge", "obstacle", "difficulty"]:
            suggestions.extend([
                "Person climbing mountain",
                "Overcoming barrier",
                "Struggle visual metaphor",
                "Problem-solving scene"
            ])
        elif concept in ["growth", "development", "progress"]:
            suggestions.extend([
                "Plant growing timelapse",
                "Bar chart rising",
                "Building construction",
                "Child growing up"
            ])
        elif concept in ["innovation", "creativity", "idea"]:
            suggestions.extend([
                "Light bulb moment",
                "Brainstorming session",
                "New technology demonstration",
                "Creative design process"
            ])
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _get_visual_suggestions_for_category(self, category: str) -> List[str]:
        """
        Get specific visual suggestions for a matched category.
        
        Args:
            category: The matched visual category
            
        Returns:
            List of visual suggestion strings
        """
        category_suggestions = {
            "landscape": [
                "Beautiful landscape - wide shot",
                "Natural scenery - aerial view",
                "Mountains or ocean vista",
                "Serene nature scene",
                "Sunset or sunrise over landscape"
            ],
            "urban": [
                "City skyline",
                "Urban street scene",
                "Downtown area - timelapse",
                "Modern architecture",
                "City life - busy intersection"
            ],
            "people": [
                "People in conversation",
                "Diverse crowd scene",
                "Professional meeting",
                "Person working",
                "Team collaboration"
            ],
            "technology": [
                "Modern technology in use",
                "Digital interface closeup",
                "Person using device",
                "Tech manufacturing",
                "Computer code or circuits"
            ],
            "abstract": [
                "Abstract visual metaphor",
                "Conceptual animation",
                "Symbolic imagery",
                "Artistic representation",
                "Meaningful visual symbol"
            ],
            "business": [
                "Modern office environment",
                "Business meeting",
                "Professional presentation",
                "Workspace scene",
                "Corporate building exterior"
            ],
            "education": [
                "Learning environment",
                "Classroom or lecture",
                "Study session",
                "Research activity",
                "Educational materials"
            ],
            "science": [
                "Scientific laboratory",
                "Research equipment",
                "Experiment in progress",
                "Scientific visualization",
                "Natural phenomenon"
            ],
            "nature": [
                "Wildlife in natural habitat",
                "Plant life close-up",
                "Ecosystem visualization",
                "Natural processes",
                "Biodiversity scene"
            ],
            "health": [
                "Healthcare environment",
                "Wellness activity",
                "Medical research",
                "Healthy lifestyle scene",
                "Medical professional at work"
            ]
        }
        
        return category_suggestions.get(category, ["Generic visual", "Relevant scene", "Appropriate imagery"])
    
    def _analyze_theme_consistency(self, segment_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze matched concepts across segments for theme consistency.
        
        Args:
            segment_matches: Matched concepts for each segment
            
        Returns:
            Dictionary with theme consistency analysis
        """
        if not segment_matches:
            return {"consistent_themes": [], "consistency_score": 0.0}
        
        # Collect all concepts across segments
        all_concepts = {}
        all_categories = defaultdict(int)
        
        for matches in segment_matches:
            for concept, data in matches.items():
                if concept not in all_concepts:
                    all_concepts[concept] = 0
                all_concepts[concept] += 1
                
                category = data.get("category", "")
                if category:
                    all_categories[category] += 1
        
        # Find consistent themes (concepts that appear in multiple segments)
        total_segments = len(segment_matches)
        consistent_themes = {}
        
        for concept, count in all_concepts.items():
            # Calculate consistency as percentage of segments where the concept appears
            consistency = count / total_segments
            if consistency >= 0.25:  # Consider consistent if appears in at least 25% of segments
                consistent_themes[concept] = consistency
        
        # Calculate overall theme consistency score
        num_consistent_themes = len(consistent_themes)
        
        # Score is higher when we have a moderate number of consistent themes
        # Too few = not enough consistency, too many = too generic
        if num_consistent_themes == 0:
            consistency_score = 0.0
        elif num_consistent_themes <= 3:
            consistency_score = 0.6 + (0.1 * num_consistent_themes)
        elif num_consistent_themes <= 5:
            consistency_score = 0.9
        else:
            consistency_score = 0.9 - (0.05 * (num_consistent_themes - 5))
        
        # Cap the score
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        # Get dominant categories
        dominant_categories = []
        if all_categories:
            max_count = max(all_categories.values())
            dominant_categories = [
                category for category, count in all_categories.items()
                if count >= max_count * 0.7  # At least 70% as frequent as the most dominant
            ]
        
        return {
            "consistent_themes": [
                {"concept": concept, "consistency": consistency}
                for concept, consistency in sorted(consistent_themes.items(), key=lambda x: x[1], reverse=True)
            ],
            "dominant_categories": dominant_categories,
            "consistency_score": consistency_score
        }
    
    async def enhance_broll_suggestions(
        self, 
        suggestions: Dict[str, Any],
        transcript_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhance B-Roll suggestions with semantic matching.
        
        Args:
            suggestions: Original B-Roll suggestions
            transcript_segments: Transcript segments
            
        Returns:
            Enhanced B-Roll suggestions
        """
        # Match concepts to transcript
        concept_matches = await self.match_concepts_to_transcript(transcript_segments)
        
        # Get B-Roll plan
        b_roll_plan = suggestions.get('b_roll_plan', [])
        
        # Enhance each insertion point with semantic matches
        for i, insertion in enumerate(b_roll_plan):
            timestamp = insertion.get('timestamp', 0)
            segment_text = insertion.get('segment_text', '')
            
            # Find the matching segment and its concepts
            segment_id = None
            for j, segment in enumerate(concept_matches.get('segments', [])):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                
                if start <= timestamp <= end:
                    segment_id = j
                    break
            
            if segment_id is not None:
                # Get concept matches for this segment
                segment_concepts = concept_matches.get('segments', [])[segment_id].get('concept_matches', {})
                
                # Add semantic matches to insertion
                insertion['semantic_matches'] = {
                    'concepts': segment_concepts,
                    'visual_suggestions': []
                }
                
                # Add visual suggestions from top concepts
                for concept, data in list(segment_concepts.items())[:3]:  # Use top 3 concepts
                    insertion['semantic_matches']['visual_suggestions'].extend(
                        data.get('visual_suggestions', [])
                    )
                
                # De-duplicate suggestions
                insertion['semantic_matches']['visual_suggestions'] = list(set(
                    insertion['semantic_matches']['visual_suggestions']
                ))[:5]  # Limit to top 5
                
                # Update the plan
                b_roll_plan[i] = insertion
        
        # Add theme consistency to overall suggestions
        suggestions['theme_consistency'] = concept_matches.get('theme_consistency', {})
        suggestions['b_roll_plan'] = b_roll_plan
        
        return suggestions 