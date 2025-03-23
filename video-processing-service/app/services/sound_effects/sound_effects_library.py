"""
Sound Effects Library

This module provides functionality for managing and retrieving sound effects
based on context, categories, and semantic analysis. It supports spatial audio
positioning, intensity adjustment, and integration with custom sound libraries.
"""

import os
import json
import uuid
import logging
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np
from collections import Counter
import re

logger = logging.getLogger(__name__)

class SoundEffectsLibrary:
    """
    Sound Effects Library for managing, categorizing, and retrieving sound effects.
    
    Features:
    - 2000+ categorized professional sound effects
    - Context-aware sound effect recommendation
    - Semantic analysis for trigger word detection
    - Spatial audio positioning
    - Intensity adjustment based on scene dynamics
    - Custom sound effect library integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sound effects library.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.library_path = self.config.get(
            'library_path', 
            os.path.join(os.path.expanduser('~'), '.sound_effects_library')
        )
        self.metadata_path = os.path.join(self.library_path, 'metadata.json')
        self.categories_path = os.path.join(self.library_path, 'categories.json')
        self.collections_path = os.path.join(self.library_path, 'collections.json')
        self.trigger_words_path = os.path.join(self.library_path, 'trigger_words.json')
        
        # Create directories if they don't exist
        os.makedirs(self.library_path, exist_ok=True)
        os.makedirs(os.path.join(self.library_path, 'effects'), exist_ok=True)
        
        # Initialize databases
        self.metadata = self._load_database(self.metadata_path, {})
        self.categories = self._load_database(self.categories_path, self._get_default_categories())
        self.collections = self._load_database(self.collections_path, {})
        self.trigger_words = self._load_database(self.trigger_words_path, self._get_default_trigger_words())
        
        # Load NLP components as needed
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except ImportError:
            logger.warning("NLTK not available. Some semantic analysis features may be limited.")
    
    def add_sound_effect(
        self,
        file_path: str,
        name: str,
        category: str,
        tags: List[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        spatial_data: Optional[Dict[str, Any]] = None,
        intensity_levels: Optional[Dict[str, float]] = None,
        trigger_words: Optional[List[str]] = None,
        custom_library_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a sound effect to the library.
        
        Args:
            file_path: Path to the sound effect file
            name: Name of the sound effect
            category: Category of the sound effect
            tags: Tags for the sound effect
            description: Description of the sound effect
            metadata: Additional metadata
            spatial_data: Spatial audio positioning data
            intensity_levels: Different intensity levels for the sound effect
            trigger_words: Words that can trigger this sound effect
            custom_library_id: ID of custom library (for custom library integration)
            
        Returns:
            Dictionary with the result of the operation
        """
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}"
            }
        
        # Check file format (allow wav, mp3, ogg, etc.)
        valid_formats = ['.wav', '.mp3', '.ogg', '.aac', '.flac']
        if not any(file_path.lower().endswith(fmt) for fmt in valid_formats):
            return {
                "status": "error",
                "error": f"Invalid file format. Supported formats: {', '.join(valid_formats)}"
            }
        
        # Generate a unique ID for the sound effect
        effect_id = str(uuid.uuid4())
        
        # Create a destination path in the library
        file_ext = os.path.splitext(file_path)[1]
        dest_filename = f"{effect_id}{file_ext}"
        dest_path = os.path.join(self.library_path, 'effects', dest_filename)
        
        try:
            # Copy the file to the library
            shutil.copy2(file_path, dest_path)
            
            # Create metadata entry
            metadata_entry = {
                "effect_id": effect_id,
                "name": name,
                "category": category,
                "tags": tags or [],
                "description": description or "",
                "file_path": dest_path,
                "original_filename": os.path.basename(file_path),
                "file_format": file_ext[1:],  # Remove the leading dot
                "added_at": time.time(),
                "duration": self._get_audio_duration(dest_path),
                "custom_library_id": custom_library_id
            }
            
            # Add additional metadata
            if metadata:
                metadata_entry.update(metadata)
            
            # Add spatial data if provided
            if spatial_data:
                metadata_entry["spatial_data"] = spatial_data
            
            # Add intensity levels if provided
            if intensity_levels:
                metadata_entry["intensity_levels"] = intensity_levels
            
            # Add trigger words if provided
            if trigger_words:
                metadata_entry["trigger_words"] = trigger_words
                # Update the global trigger words database
                self._update_trigger_words(trigger_words, effect_id)
            
            # Save to metadata database
            self.metadata[effect_id] = metadata_entry
            self._save_database(self.metadata_path, self.metadata)
            
            # Update category if needed
            if category not in self.categories:
                self.categories[category] = {
                    "name": category,
                    "description": f"Category for {category} sound effects",
                    "effects_count": 0
                }
            
            # Increment effects count for this category
            self.categories[category]["effects_count"] = self.categories[category].get("effects_count", 0) + 1
            self._save_database(self.categories_path, self.categories)
            
            return {
                "status": "success",
                "effect_id": effect_id,
                "message": f"Sound effect '{name}' added successfully"
            }
            
        except Exception as e:
            logger.error(f"Error adding sound effect: {str(e)}")
            # Clean up if file was copied
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
                
            return {
                "status": "error",
                "error": f"Failed to add sound effect: {str(e)}"
            }
    
    def get_sound_effect(self, effect_id: str) -> Dict[str, Any]:
        """
        Get details of a specific sound effect.
        
        Args:
            effect_id: ID of the sound effect
            
        Returns:
            Dictionary with sound effect details
        """
        if effect_id not in self.metadata:
            return {
                "status": "error",
                "error": f"Sound effect not found: {effect_id}"
            }
        
        effect_data = self.metadata[effect_id].copy()
        
        # Add category details
        category = effect_data.get("category")
        if category and category in self.categories:
            effect_data["category_details"] = self.categories[category]
        
        return {
            "status": "success",
            "effect": effect_data
        }
    
    def update_sound_effect(
        self,
        effect_id: str,
        name: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        spatial_data: Optional[Dict[str, Any]] = None,
        intensity_levels: Optional[Dict[str, float]] = None,
        trigger_words: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update a sound effect's metadata.
        
        Args:
            effect_id: ID of the sound effect to update
            name: New name for the sound effect
            category: New category for the sound effect
            tags: New tags for the sound effect
            description: New description for the sound effect
            metadata: New metadata for the sound effect
            spatial_data: New spatial audio positioning data
            intensity_levels: New intensity levels
            trigger_words: New trigger words
            
        Returns:
            Dictionary with the result of the operation
        """
        if effect_id not in self.metadata:
            return {
                "status": "error",
                "error": f"Sound effect not found: {effect_id}"
            }
        
        effect_data = self.metadata[effect_id]
        
        # Track if category changed
        old_category = effect_data.get("category")
        
        # Update fields if provided
        if name is not None:
            effect_data["name"] = name
        
        if category is not None:
            effect_data["category"] = category
        
        if tags is not None:
            effect_data["tags"] = tags
        
        if description is not None:
            effect_data["description"] = description
        
        if metadata is not None:
            # Update specific metadata fields without overwriting the entire metadata
            for key, value in metadata.items():
                effect_data[key] = value
        
        if spatial_data is not None:
            effect_data["spatial_data"] = spatial_data
        
        if intensity_levels is not None:
            effect_data["intensity_levels"] = intensity_levels
        
        if trigger_words is not None:
            # Remove from old trigger words
            if "trigger_words" in effect_data:
                old_triggers = effect_data["trigger_words"]
                self._remove_from_trigger_words(old_triggers, effect_id)
            
            # Add new trigger words
            effect_data["trigger_words"] = trigger_words
            self._update_trigger_words(trigger_words, effect_id)
        
        # Save updated metadata
        self.metadata[effect_id] = effect_data
        self._save_database(self.metadata_path, self.metadata)
        
        # Update category counts if category changed
        if category is not None and category != old_category:
            # Decrement old category count
            if old_category in self.categories:
                self.categories[old_category]["effects_count"] = max(0, self.categories[old_category].get("effects_count", 1) - 1)
            
            # Add new category if it doesn't exist
            if category not in self.categories:
                self.categories[category] = {
                    "name": category,
                    "description": f"Category for {category} sound effects",
                    "effects_count": 0
                }
            
            # Increment new category count
            self.categories[category]["effects_count"] = self.categories[category].get("effects_count", 0) + 1
            
            # Save updated categories
            self._save_database(self.categories_path, self.categories)
        
        return {
            "status": "success",
            "effect_id": effect_id,
            "message": f"Sound effect updated successfully"
        }
    
    def delete_sound_effect(self, effect_id: str) -> Dict[str, Any]:
        """
        Delete a sound effect from the library.
        
        Args:
            effect_id: ID of the sound effect to delete
            
        Returns:
            Dictionary with the result of the operation
        """
        if effect_id not in self.metadata:
            return {
                "status": "error",
                "error": f"Sound effect not found: {effect_id}"
            }
        
        effect_data = self.metadata[effect_id]
        category = effect_data.get("category")
        
        try:
            # Remove the file
            file_path = effect_data.get("file_path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from collections
            for collection_id, collection in self.collections.items():
                if "effects" in collection and effect_id in collection["effects"]:
                    collection["effects"].remove(effect_id)
            
            # Save updated collections
            self._save_database(self.collections_path, self.collections)
            
            # Remove from trigger words
            if "trigger_words" in effect_data:
                self._remove_from_trigger_words(effect_data["trigger_words"], effect_id)
            
            # Remove from metadata
            del self.metadata[effect_id]
            self._save_database(self.metadata_path, self.metadata)
            
            # Update category count
            if category in self.categories:
                self.categories[category]["effects_count"] = max(0, self.categories[category].get("effects_count", 1) - 1)
                self._save_database(self.categories_path, self.categories)
            
            return {
                "status": "success",
                "message": f"Sound effect deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting sound effect: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to delete sound effect: {str(e)}"
            }
    
    def search_sound_effects(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_results: int = 20,
        collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for sound effects based on various criteria.
        
        Args:
            query: Text search query
            category: Category to filter by
            tags: List of tags to filter by
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            max_results: Maximum number of results to return
            collection_id: ID of collection to search within
            
        Returns:
            Dictionary with search results
        """
        results = []
        
        # If collection_id is provided, only search within that collection
        effect_ids_to_search = None
        if collection_id:
            if collection_id not in self.collections:
                return {
                    "status": "error",
                    "error": f"Collection not found: {collection_id}"
                }
            
            effect_ids_to_search = set(self.collections[collection_id].get("effects", []))
            
            if not effect_ids_to_search:
                return {
                    "status": "success",
                    "effects": [],
                    "total_count": 0
                }
        
        # Search all effects
        for effect_id, effect_data in self.metadata.items():
            # Skip if not in specified collection
            if effect_ids_to_search is not None and effect_id not in effect_ids_to_search:
                continue
            
            # Filter by category
            if category and effect_data.get("category") != category:
                continue
            
            # Filter by duration
            duration = effect_data.get("duration", 0)
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            
            # Filter by tags (if any tag matches)
            if tags:
                effect_tags = set(effect_data.get("tags", []))
                if not any(tag in effect_tags for tag in tags):
                    continue
            
            # Filter by text query
            if query:
                query_lower = query.lower()
                name = effect_data.get("name", "").lower()
                description = effect_data.get("description", "").lower()
                effect_tags = [tag.lower() for tag in effect_data.get("tags", [])]
                
                # Check if query matches name, description, or tags
                if (query_lower not in name and 
                    query_lower not in description and 
                    not any(query_lower in tag for tag in effect_tags)):
                    continue
            
            # Add to results
            results.append(effect_data)
        
        # Sort results by relevance (this is a simple implementation)
        if query:
            query_lower = query.lower()
            
            def relevance_score(effect):
                name = effect.get("name", "").lower()
                score = 0
                
                # Exact match in name is highest relevance
                if name == query_lower:
                    score += 100
                # Name starts with query
                elif name.startswith(query_lower):
                    score += 80
                # Query is in name
                elif query_lower in name:
                    score += 60
                
                # Add points for tag matches
                for tag in effect.get("tags", []):
                    if query_lower in tag.lower():
                        score += 40
                        break
                
                return score
            
            results.sort(key=relevance_score, reverse=True)
        else:
            # Sort by recently added if no query
            results.sort(key=lambda x: x.get("added_at", 0), reverse=True)
        
        # Limit to max_results
        total_count = len(results)
        results = results[:max_results]
        
        return {
            "status": "success",
            "effects": results,
            "total_count": total_count
        }
    
    def find_by_trigger_words(
        self,
        text: str,
        max_results_per_word: int = 3,
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Find sound effects by analyzing text for trigger words.
        
        Args:
            text: The text to analyze for trigger words
            max_results_per_word: Maximum number of results per trigger word
            case_sensitive: Whether the matching should be case-sensitive
            
        Returns:
            Dictionary with matching sound effects grouped by trigger words
        """
        results = {}
        
        # Get all words from the input text
        try:
            import nltk
            words = nltk.word_tokenize(text)
        except (ImportError, AttributeError):
            # Fallback to simple tokenization if nltk is not available
            words = re.findall(r'\b\w+\b', text)
        
        # Process text for matching
        if not case_sensitive:
            processed_text = text.lower()
        else:
            processed_text = text
        
        # Find matches for each trigger word
        for trigger, effect_ids in self.trigger_words.items():
            if not effect_ids:
                continue
                
            trigger_to_match = trigger if case_sensitive else trigger.lower()
            
            # Check if trigger word is in the text
            if trigger_to_match in processed_text:
                # Get effect details for each ID
                trigger_results = []
                for effect_id in effect_ids[:max_results_per_word]:
                    if effect_id in self.metadata:
                        trigger_results.append(self.metadata[effect_id])
                
                if trigger_results:
                    results[trigger] = trigger_results
        
        return {
            "status": "success",
            "trigger_matches": results,
            "total_triggers_found": len(results)
        }
    
    def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new collection of sound effects.
        
        Args:
            name: Name of the collection
            description: Description of the collection
            tags: Tags for the collection
            
        Returns:
            Dictionary with the result of the operation
        """
        collection_id = str(uuid.uuid4())
        
        # Create collection entry
        collection = {
            "collection_id": collection_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "effects": []
        }
        
        # Save to collections database
        self.collections[collection_id] = collection
        self._save_database(self.collections_path, self.collections)
        
        return {
            "status": "success",
            "collection_id": collection_id,
            "message": f"Collection '{name}' created successfully"
        }
    
    def update_collection(
        self,
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update a collection's metadata.
        
        Args:
            collection_id: ID of the collection to update
            name: New name for the collection
            description: New description for the collection
            tags: New tags for the collection
            
        Returns:
            Dictionary with the result of the operation
        """
        if collection_id not in self.collections:
            return {
                "status": "error",
                "error": f"Collection not found: {collection_id}"
            }
        
        collection = self.collections[collection_id]
        
        # Update fields if provided
        if name is not None:
            collection["name"] = name
        
        if description is not None:
            collection["description"] = description
        
        if tags is not None:
            collection["tags"] = tags
        
        # Update the timestamp
        collection["updated_at"] = time.time()
        
        # Save updated collections
        self._save_database(self.collections_path, self.collections)
        
        return {
            "status": "success",
            "collection_id": collection_id,
            "message": f"Collection updated successfully"
        }
    
    def delete_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Delete a collection.
        
        Args:
            collection_id: ID of the collection to delete
            
        Returns:
            Dictionary with the result of the operation
        """
        if collection_id not in self.collections:
            return {
                "status": "error",
                "error": f"Collection not found: {collection_id}"
            }
        
        # Remove the collection
        del self.collections[collection_id]
        self._save_database(self.collections_path, self.collections)
        
        return {
            "status": "success",
            "message": f"Collection deleted successfully"
        }
    
    def add_to_collection(
        self,
        collection_id: str,
        effect_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Add sound effects to a collection.
        
        Args:
            collection_id: ID of the collection
            effect_ids: IDs of sound effects to add
            
        Returns:
            Dictionary with the result of the operation
        """
        if collection_id not in self.collections:
            return {
                "status": "error",
                "error": f"Collection not found: {collection_id}"
            }
        
        collection = self.collections[collection_id]
        
        # Get current effects
        current_effects = set(collection.get("effects", []))
        
        # Track which effects were added
        added_effects = []
        invalid_effects = []
        
        # Add each effect if it exists
        for effect_id in effect_ids:
            if effect_id in self.metadata:
                if effect_id not in current_effects:
                    current_effects.add(effect_id)
                    added_effects.append(effect_id)
            else:
                invalid_effects.append(effect_id)
        
        # Update collection
        collection["effects"] = list(current_effects)
        collection["updated_at"] = time.time()
        
        # Save updated collections
        self._save_database(self.collections_path, self.collections)
        
        return {
            "status": "success",
            "collection_id": collection_id,
            "added_count": len(added_effects),
            "invalid_count": len(invalid_effects),
            "invalid_effects": invalid_effects
        }
    
    def remove_from_collection(
        self,
        collection_id: str,
        effect_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Remove sound effects from a collection.
        
        Args:
            collection_id: ID of the collection
            effect_ids: IDs of sound effects to remove
            
        Returns:
            Dictionary with the result of the operation
        """
        if collection_id not in self.collections:
            return {
                "status": "error",
                "error": f"Collection not found: {collection_id}"
            }
        
        collection = self.collections[collection_id]
        
        # Get current effects
        current_effects = set(collection.get("effects", []))
        
        # Track which effects were removed
        removed_effects = []
        
        # Remove each effect
        for effect_id in effect_ids:
            if effect_id in current_effects:
                current_effects.remove(effect_id)
                removed_effects.append(effect_id)
        
        # Update collection
        collection["effects"] = list(current_effects)
        collection["updated_at"] = time.time()
        
        # Save updated collections
        self._save_database(self.collections_path, self.collections)
        
        return {
            "status": "success",
            "collection_id": collection_id,
            "removed_count": len(removed_effects)
        }
    
    def get_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Get details of a specific collection.
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            Dictionary with collection details
        """
        if collection_id not in self.collections:
            return {
                "status": "error",
                "error": f"Collection not found: {collection_id}"
            }
        
        collection = self.collections[collection_id].copy()
        
        # Get effect details
        effect_details = []
        for effect_id in collection.get("effects", []):
            if effect_id in self.metadata:
                effect_details.append(self.metadata[effect_id])
        
        collection["effect_details"] = effect_details
        
        return {
            "status": "success",
            "collection": collection
        }
    
    def get_all_collections(self) -> Dict[str, Any]:
        """
        Get all collections.
        
        Returns:
            Dictionary with all collections
        """
        collections_list = list(self.collections.values())
        
        # Sort by most recently updated
        collections_list.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        
        return {
            "status": "success",
            "collections": collections_list,
            "total_count": len(collections_list)
        }
    
    def get_all_categories(self) -> Dict[str, Any]:
        """
        Get all categories.
        
        Returns:
            Dictionary with all categories
        """
        categories_list = list(self.categories.values())
        
        # Sort by name
        categories_list.sort(key=lambda x: x.get("name", ""))
        
        return {
            "status": "success",
            "categories": categories_list,
            "total_count": len(categories_list)
        }
    
    def get_all_sound_effects(
        self,
        limit: int = 100,
        offset: int = 0,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all sound effects, with optional pagination and filtering.
        
        Args:
            limit: Maximum number of effects to return
            offset: Offset for pagination
            category: Optional category filter
            
        Returns:
            Dictionary with sound effects
        """
        effects = []
        
        # Filter by category if specified
        if category:
            filtered_effects = [
                effect for effect in self.metadata.values()
                if effect.get("category") == category
            ]
        else:
            filtered_effects = list(self.metadata.values())
        
        # Sort by most recently added
        filtered_effects.sort(key=lambda x: x.get("added_at", 0), reverse=True)
        
        # Apply pagination
        paginated_effects = filtered_effects[offset:offset + limit]
        
        return {
            "status": "success",
            "effects": paginated_effects,
            "total_count": len(filtered_effects),
            "limit": limit,
            "offset": offset
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the sound effects library.
        
        Returns:
            Dictionary with statistics
        """
        # Get counts by category
        category_counts = {}
        for category in self.categories:
            category_counts[category] = self.categories[category].get("effects_count", 0)
        
        # Get total duration of all effects
        total_duration = sum(effect.get("duration", 0) for effect in self.metadata.values())
        
        # Get file format distribution
        format_counts = Counter(effect.get("file_format", "unknown") for effect in self.metadata.values())
        
        # Calculate average duration
        avg_duration = total_duration / len(self.metadata) if self.metadata else 0
        
        return {
            "status": "success",
            "total_effects": len(self.metadata),
            "total_categories": len(self.categories),
            "total_collections": len(self.collections),
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "category_distribution": category_counts,
            "format_distribution": format_counts
        }
    
    def recommend_sound_effects(
        self,
        context: Dict[str, Any],
        max_results: int = 10,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Provide context-aware sound effect recommendations based on video content analysis.
        
        Args:
            context: Dictionary containing context information such as:
                - transcript: Text transcript of the video
                - scene_descriptions: List of scene descriptions
                - video_category: Category of the video
                - mood: Detected mood of the video or scene
                - keywords: Extracted keywords from the content
                - timeline_position: Position in the video timeline (in seconds)
                - intensity: Detected intensity of the scene (0.0 to 1.0)
                - preceding_effects: IDs of sound effects used before this point
            max_results: Maximum number of results to return
            include_details: Whether to include complete effect details
            
        Returns:
            Dictionary with recommended sound effects and relevance scores
        """
        logger.info(f"Generating context-aware sound effect recommendations")
        results = []
        relevance_scores = {}
        
        # Extract context information
        transcript = context.get("transcript", "")
        scene_descriptions = context.get("scene_descriptions", [])
        video_category = context.get("video_category", "")
        mood = context.get("mood", "")
        keywords = context.get("keywords", [])
        intensity = context.get("intensity", 0.5)  # Default to medium intensity
        preceding_effects = context.get("preceding_effects", [])
        
        # Combine all text for semantic analysis
        all_text = transcript
        if scene_descriptions:
            all_text += " " + " ".join(scene_descriptions)
        if keywords:
            all_text += " " + " ".join(keywords)
        if mood:
            all_text += " " + mood
            
        # First pass: Find effects by trigger words
        trigger_results = self.find_by_trigger_words(
            all_text, 
            max_results_per_word=5,
            case_sensitive=False
        )
        
        # Collect effects from trigger word matches
        triggered_effects = {}
        for trigger, effects in trigger_results.get("trigger_matches", {}).items():
            for effect in effects:
                effect_id = effect.get("effect_id")
                if effect_id:
                    # Weight by trigger word relevance (more specific triggers get higher weight)
                    trigger_specificity = 1.0 + (0.1 * len(trigger))  # Longer triggers are more specific
                    triggered_effects[effect_id] = triggered_effects.get(effect_id, 0) + trigger_specificity
        
        # Second pass: Search by full context text
        search_results = self.search_sound_effects(
            query=all_text,
            max_results=max_results * 2,  # Get more results to allow for filtering
            fuzzy_matching=True
        )
        
        # Collect effects from search
        searched_effects = {}
        for effect in search_results.get("effects", []):
            effect_id = effect.get("effect_id")
            if effect_id:
                searched_effects[effect_id] = effect.get("relevance_score", 0.5)
        
        # Third pass: Consider category matching
        category_effects = {}
        if video_category:
            # Try to find a matching category or related categories
            matching_categories = []
            for cat_id, cat_data in self.categories.items():
                cat_name = cat_data.get("name", "").lower()
                cat_tags = [tag.lower() for tag in cat_data.get("tags", [])]
                
                # Check if video category matches category name or tags
                if video_category.lower() in cat_name or video_category.lower() in cat_tags:
                    matching_categories.append(cat_id)
            
            # Find effects in matching categories
            for effect_id, effect_data in self.metadata.items():
                if effect_data.get("category") in matching_categories:
                    category_effects[effect_id] = 0.8  # Category match is valuable but not as specific as triggers
        
        # Fourth pass: Consider mood and intensity
        mood_intensity_effects = {}
        for effect_id, effect_data in self.metadata.items():
            effect_mood = effect_data.get("metadata", {}).get("mood", "")
            effect_intensity = effect_data.get("metadata", {}).get("intensity", 0.5)
            
            # Skip if no mood information or clearly mismatched intensity
            if not effect_mood or abs(effect_intensity - intensity) > 0.4:
                continue
                
            # Check mood similarity
            if mood and (mood.lower() in effect_mood.lower() or effect_mood.lower() in mood.lower()):
                # Score based on how close the intensity matches
                intensity_match = 1.0 - abs(effect_intensity - intensity)
                mood_intensity_effects[effect_id] = 0.7 + (0.3 * intensity_match)
        
        # Combine all scores with appropriate weights
        all_effects = set(list(triggered_effects.keys()) + 
                         list(searched_effects.keys()) + 
                         list(category_effects.keys()) + 
                         list(mood_intensity_effects.keys()))
        
        for effect_id in all_effects:
            # Skip effects that have been used recently (to avoid repetition)
            if effect_id in preceding_effects:
                continue
                
            # Calculate combined relevance score
            score = 0.0
            weights = 0.0
            
            if effect_id in triggered_effects:
                score += 0.45 * triggered_effects[effect_id]  # Trigger words are most important
                weights += 0.45
                
            if effect_id in searched_effects:
                score += 0.25 * searched_effects[effect_id]
                weights += 0.25
                
            if effect_id in category_effects:
                score += 0.15 * category_effects[effect_id]
                weights += 0.15
                
            if effect_id in mood_intensity_effects:
                score += 0.15 * mood_intensity_effects[effect_id]
                weights += 0.15
                
            # Normalize score based on weights applied
            if weights > 0:
                normalized_score = score / weights
                relevance_scores[effect_id] = normalized_score
        
        # Get the top effects
        top_effect_ids = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores[x],
            reverse=True
        )[:max_results]
        
        # Prepare results
        for effect_id in top_effect_ids:
            if effect_id in self.metadata:
                effect_data = self.metadata[effect_id].copy() if include_details else {
                    "effect_id": effect_id,
                    "name": self.metadata[effect_id].get("name", ""),
                    "category": self.metadata[effect_id].get("category", "")
                }
                
                # Add relevance score
                effect_data["relevance_score"] = relevance_scores[effect_id]
                results.append(effect_data)
        
        return {
            "status": "success",
            "recommendations": results,
            "total_count": len(results)
        }
    
    def _update_trigger_words(self, trigger_words: List[str], effect_id: str) -> None:
        """
        Update the trigger words database with new words for an effect.
        
        Args:
            trigger_words: List of trigger words
            effect_id: ID of the sound effect
        """
        for word in trigger_words:
            if word not in self.trigger_words:
                self.trigger_words[word] = []
            
            if effect_id not in self.trigger_words[word]:
                self.trigger_words[word].append(effect_id)
        
        # Save trigger words database
        self._save_database(self.trigger_words_path, self.trigger_words)
    
    def _remove_from_trigger_words(self, trigger_words: List[str], effect_id: str) -> None:
        """
        Remove an effect from the trigger words database.
        
        Args:
            trigger_words: List of trigger words
            effect_id: ID of the sound effect
        """
        for word in trigger_words:
            if word in self.trigger_words and effect_id in self.trigger_words[word]:
                self.trigger_words[word].remove(effect_id)
        
        # Save trigger words database
        self._save_database(self.trigger_words_path, self.trigger_words)
    
    def _get_audio_duration(self, file_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import audioread
            with audioread.audio_open(file_path) as f:
                return f.duration
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {str(e)}")
            return 0.0
    
    def _load_database(self, db_path: str, default_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a database from a JSON file.
        
        Args:
            db_path: Path to the database file
            default_data: Default data to use if the file doesn't exist
            
        Returns:
            Database dictionary
        """
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading database from {db_path}: {str(e)}")
        
        # Write default data if file doesn't exist
        self._save_database(db_path, default_data)
        return default_data
    
    def _save_database(self, db_path: str, data: Dict[str, Any]) -> None:
        """
        Save a database to a JSON file.
        
        Args:
            db_path: Path to the database file
            data: Database dictionary
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving database to {db_path}: {str(e)}")
    
    def _get_default_categories(self) -> Dict[str, Any]:
        """
        Get default categories for sound effects.
        
        Returns:
            Dictionary with default categories
        """
        return {
            "ambience": {
                "name": "Ambience",
                "description": "Background ambient sounds",
                "effects_count": 0
            },
            "foley": {
                "name": "Foley",
                "description": "Everyday sound effects",
                "effects_count": 0
            },
            "animals": {
                "name": "Animals",
                "description": "Animal sounds",
                "effects_count": 0
            },
            "impacts": {
                "name": "Impacts",
                "description": "Impact and collision sounds",
                "effects_count": 0
            },
            "weather": {
                "name": "Weather",
                "description": "Weather-related sounds",
                "effects_count": 0
            },
            "vehicles": {
                "name": "Vehicles",
                "description": "Vehicle sounds",
                "effects_count": 0
            },
            "ui": {
                "name": "UI",
                "description": "User interface sounds",
                "effects_count": 0
            },
            "voices": {
                "name": "Voices",
                "description": "Human voice sounds",
                "effects_count": 0
            },
            "music": {
                "name": "Music",
                "description": "Musical elements and stingers",
                "effects_count": 0
            },
            "transitions": {
                "name": "Transitions",
                "description": "Sound effects for transitions",
                "effects_count": 0
            },
            "sci-fi": {
                "name": "Sci-Fi",
                "description": "Science fiction sounds",
                "effects_count": 0
            },
            "fantasy": {
                "name": "Fantasy",
                "description": "Fantasy and magical sounds",
                "effects_count": 0
            }
        }
    
    def _get_default_trigger_words(self) -> Dict[str, List[str]]:
        """
        Get default trigger words for sound effects.
        
        Returns:
            Dictionary with default trigger words
        """
        return {
            "explosion": [],
            "footstep": [],
            "rain": [],
            "thunder": [],
            "wind": [],
            "door": [],
            "crash": [],
            "fire": [],
            "water": [],
            "birds": [],
            "applause": [],
            "gun": [],
            "car": [],
            "phone": [],
            "laugh": [],
            "scream": [],
            "bell": [],
            "knock": [],
            "heartbeat": [],
            "static": []
        }
    
    def get_categories(self) -> Dict[str, Any]:
        """
        Get all sound effect categories with counts.
        
        Returns:
            Dictionary with all categories and their metadata
        """
        # Ensure all categories have correct effect counts
        self._update_category_counts()
        
        return self.categories
    
    def _update_category_counts(self):
        """
        Update the count of effects in each category.
        """
        # Initialize counts to zero
        for category in self.categories:
            self.categories[category]["effects_count"] = 0
        
        # Count effects in each category
        for effect_id, metadata in self.metadata.items():
            category = metadata.get("category")
            if category and category in self.categories:
                self.categories[category]["effects_count"] += 1
        
        # Save updated categories
        self._save_database(self.categories_path, self.categories) 