"""
Stock Footage Integration for B-Roll Engine.

This module provides integration with various stock footage libraries
to search for and download appropriate B-Roll clips based on content analysis.
"""

import logging
import os
import json
import tempfile
import asyncio
import aiohttp
import time
import random
import hashlib
import shutil
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode
from pathlib import Path

logger = logging.getLogger(__name__)

class StockFootageProvider:
    """
    Integration with stock footage libraries for B-Roll suggestions.
    
    This class provides methods to search for and retrieve appropriate
    B-Roll footage from various stock footage providers and APIs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StockFootageProvider.
        
        Args:
            config: Configuration options for stock footage integration
        """
        self.config = config or {}
        
        # API keys for different providers
        self.api_keys = self.config.get('api_keys', {})
        
        # Cache directory for downloaded previews
        self.cache_dir = self.config.get('cache_dir', os.path.join(os.path.expanduser('~'), '.broll_cache'))
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, 'search'), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, 'previews'), exist_ok=True)
        
        # Cache for search results
        self.search_cache = {}
        self.cache_expiry = self.config.get('cache_expiry', 24 * 60 * 60)  # 24 hours in seconds
        
        # Supported providers
        self.supported_providers = self.config.get('supported_providers', ['pexels', 'pixabay', 'local'])
        
        # Default provider if none specified
        self.default_provider = self.config.get('default_provider', 'pixabay')
        
        # Base URLs for API requests
        self.base_urls = {
            'pexels': 'https://api.pexels.com/videos',
            'pixabay': 'https://pixabay.com/api/videos',
            'local': ''
        }
        
        # User library paths
        self.user_library_paths = self.config.get('user_library_paths', [])
        
        # Initialize HTTP session
        self.session = None
        
        # Load user library index if available
        self.user_library_index = {}
        self._load_user_library_index()
    
    async def _init_session(self):
        """Initialize aiohttp session if not already initialized."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the HTTP client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def _get_cache_path(self, query: str, provider: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Get cache file path for a query.
        
        Args:
            query: Search query
            provider: Stock provider name
            options: Search options
            
        Returns:
            Path to cache file
        """
        # Create a unique hash for the query and options
        hash_base = f"{query}_{provider}_{str(options)}"
        query_hash = hashlib.md5(hash_base.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, 'search', f"{query_hash}.json")
    
    def _load_user_library_index(self):
        """Load the user's library index from disk."""
        index_path = os.path.join(self.cache_dir, 'user_library_index.json')
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    self.user_library_index = json.load(f)
                logger.info(f"Loaded user library index with {len(self.user_library_index.get('clips', []))} clips")
            except Exception as e:
                logger.error(f"Error loading user library index: {str(e)}")
                self.user_library_index = {}
        else:
            self.user_library_index = {}
    
    async def _index_user_library(self, force: bool = False):
        """
        Index user's library of B-Roll footage.
        
        Args:
            force: Whether to force reindexing
        """
        index_path = os.path.join(self.cache_dir, 'user_library_index.json')
        
        # Check if index needs to be updated
        if not force and os.path.exists(index_path):
            index_mtime = os.path.getmtime(index_path)
            current_time = time.time()
            
            if current_time - index_mtime < 3600:  # Only update if older than 1 hour
                return
        
        clips = []
        
        # Scan library paths
        for lib_path in self.user_library_paths:
            if not os.path.exists(lib_path):
                logger.warning(f"User library path does not exist: {lib_path}")
                continue
            
            # Get all video files
            for root, _, files in os.walk(lib_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                        file_path = os.path.join(root, file)
                        
                        # Extract metadata from filename and path
                        relative_path = os.path.relpath(file_path, lib_path)
                        tags = self._extract_tags_from_path(relative_path)
                        
                        # Extract folder structure as categories
                        categories = relative_path.split(os.path.sep)[:-1]
                        
                        # Add to clips list
                        clip = {
                            'id': hashlib.md5(file_path.encode()).hexdigest(),
                            'file_path': file_path,
                            'filename': file,
                            'relative_path': relative_path,
                            'library_path': lib_path,
                            'tags': tags,
                            'categories': categories,
                            'provider': 'local'
                        }
                        
                        clips.append(clip)
        
        # Save the index
        self.user_library_index = {
            'clips': clips,
            'updated_at': time.time()
        }
        
        with open(index_path, 'w') as f:
            json.dump(self.user_library_index, f, indent=2)
        
        logger.info(f"Indexed {len(clips)} clips in user library")
    
    def _extract_tags_from_path(self, path: str) -> List[str]:
        """
        Extract tags from file path.
        
        Args:
            path: File path
            
        Returns:
            List of extracted tags
        """
        # Remove extension
        path = os.path.splitext(path)[0]
        
        # Replace separators with spaces
        path = path.replace('_', ' ').replace('-', ' ').replace('/', ' ').replace('\\', ' ')
        
        # Split into words and filter out short words
        words = [word.lower() for word in path.split() if len(word) > 2]
        
        return words
    
    async def search_footage(
        self, 
        query: str, 
        provider: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for footage matching the query.
        
        Args:
            query: Search query
            provider: Stock provider (pexels, pixabay, or local)
            options: Search options like per_page, page, orientation
            
        Returns:
            List of footage metadata
        """
        await self._init_session()
        options = options or {}
        
        # Default to using all supported providers if not specified
        providers = [provider] if provider else self.supported_providers
        
        results = []
        
        for p in providers:
            if p not in self.supported_providers:
                logger.warning(f"Unsupported provider: {p}")
                continue
            
            # Check cache first
            cache_path = self._get_cache_path(query, p, options)
            
            if os.path.exists(cache_path):
                cache_age = time.time() - os.path.getmtime(cache_path)
                
                if cache_age < self.cache_expiry:
                    try:
                        with open(cache_path, 'r') as f:
                            cached_results = json.load(f)
                            for item in cached_results:
                                item['provider'] = p
                                item['source'] = 'cache'
                            results.extend(cached_results)
                            continue
                    except Exception as e:
                        logger.error(f"Error loading cache for {query} from {p}: {str(e)}")
            
            # If not in cache or cache expired, search API
            provider_results = []
            
            if p == 'pexels':
                provider_results = await self._search_pexels(query, options)
            elif p == 'pixabay':
                provider_results = await self._search_pixabay(query, options)
            elif p == 'local':
                provider_results = await self._search_user_library(query, options)
            
            # Update provider and save to cache
            for item in provider_results:
                item['provider'] = p
                item['source'] = 'api'
            
            # Save to cache
            if provider_results:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(provider_results, f)
                except Exception as e:
                    logger.error(f"Error saving cache for {query} from {p}: {str(e)}")
            
            results.extend(provider_results)
        
        return results
    
    async def _search_pexels(
        self, 
        query: str, 
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search Pexels API for stock footage.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            List of footage metadata
        """
        if 'pexels' not in self.api_keys:
            logger.warning("Pexels API key not found")
            return []
        
        api_key = self.api_keys['pexels']
        
        # Set up request parameters
        per_page = options.get('per_page', 15)
        page = options.get('page', 1)
        orientation = options.get('orientation', '')
        
        # Build URL
        url = f"{self.base_urls['pexels']}/search?query={query}&per_page={per_page}&page={page}"
        
        if orientation:
            url += f"&orientation={orientation}"
        
        # Make API request
        headers = {'Authorization': api_key}
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Pexels API error: {response.status}")
                    return []
                
                data = await response.json()
                
                # Process results
                results = []
                
                for video in data.get('videos', []):
                    # Find a suitable video file
                    video_files = video.get('video_files', [])
                    main_file = None
                    
                    for file in video_files:
                        if file.get('quality') == 'hd' and file.get('file_type') == 'video/mp4':
                            main_file = file
                            break
                    
                    if not main_file and video_files:
                        # Just use the first file
                        main_file = video_files[0]
                    
                    if not main_file:
                        continue
                    
                    # Create result object
                    result = {
                        'id': str(video.get('id')),
                        'title': f"Pexels video {video.get('id')}",
                        'description': '',
                        'url': video.get('url', ''),
                        'download_url': main_file.get('link', ''),
                        'preview_url': video.get('image', ''),
                        'duration': video.get('duration', 0),
                        'width': main_file.get('width', 0),
                        'height': main_file.get('height', 0),
                        'tags': [query] + query.split(),
                        'user': video.get('user', {}).get('name', ''),
                        'user_url': video.get('user', {}).get('url', '')
                    }
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching Pexels: {str(e)}")
            return []
    
    async def _search_pixabay(
        self, 
        query: str, 
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search Pixabay API for stock footage.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            List of footage metadata
        """
        if 'pixabay' not in self.api_keys:
            logger.warning("Pixabay API key not found")
            return []
        
        api_key = self.api_keys['pixabay']
        
        # Set up request parameters
        per_page = options.get('per_page', 15)
        page = options.get('page', 1)
        
        # Build URL
        url = f"https://pixabay.com/api/videos/?key={api_key}&q={query}&per_page={per_page}&page={page}"
        
        # Make API request
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Pixabay API error: {response.status}")
                    return []
                
                data = await response.json()
                
                # Process results
                results = []
                
                for hit in data.get('hits', []):
                    # Find a suitable video file (medium size)
                    video_url = hit.get('videos', {}).get('medium', {}).get('url', '')
                    
                    if not video_url:
                        continue
                    
                    # Create result object
                    result = {
                        'id': str(hit.get('id')),
                        'title': f"Pixabay video {hit.get('id')}",
                        'description': '',
                        'url': hit.get('pageURL', ''),
                        'download_url': video_url,
                        'preview_url': hit.get('userImageURL', ''),
                        'duration': 0,  # Pixabay doesn't provide duration
                        'width': hit.get('videos', {}).get('medium', {}).get('width', 0),
                        'height': hit.get('videos', {}).get('medium', {}).get('height', 0),
                        'tags': hit.get('tags', '').split(', '),
                        'user': hit.get('user', ''),
                        'user_url': ''
                    }
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching Pixabay: {str(e)}")
            return []
    
    async def _search_user_library(
        self, 
        query: str, 
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search user's local library for stock footage.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            List of footage metadata
        """
        # Ensure library is indexed
        await self._index_user_library()
        
        clips = self.user_library_index.get('clips', [])
        
        if not clips:
            return []
        
        # Filter clips by query terms
        query_terms = query.lower().split()
        results = []
        
        for clip in clips:
            # Check if any query terms match tags or categories
            tags = clip.get('tags', [])
            categories = clip.get('categories', [])
            
            match_score = 0
            for term in query_terms:
                # Check for exact matches in tags and categories
                if term in tags:
                    match_score += 3
                if term in categories:
                    match_score += 2
                
                # Check for partial matches
                for tag in tags:
                    if term in tag:
                        match_score += 1
            
            if match_score > 0:
                # Create result object
                result = {
                    'id': clip.get('id', ''),
                    'title': clip.get('filename', ''),
                    'description': '',
                    'url': '',
                    'download_url': clip.get('file_path', ''),
                    'preview_url': clip.get('file_path', ''),
                    'duration': 0,  # We don't have duration info
                    'width': 0,     # We don't have dimension info
                    'height': 0,    # We don't have dimension info
                    'tags': tags,
                    'categories': categories,
                    'match_score': match_score,
                    'local': True
                }
                
                results.append(result)
        
        # Sort by match score
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        # Apply pagination
        per_page = options.get('per_page', 15)
        page = options.get('page', 1)
        start = (page - 1) * per_page
        end = start + per_page
        
        return results[start:end]
    
    async def get_footage_details(
        self, 
        footage_id: str, 
        provider: str
    ) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific footage item.
        
        Args:
            footage_id: Footage ID
            provider: Stock provider
            
        Returns:
            Dict containing footage metadata
        """
        await self._init_session()
        
        # For local footage, lookup in library index
        if provider == 'local':
            clips = self.user_library_index.get('clips', [])
            for clip in clips:
                if clip.get('id') == footage_id:
                    return clip
            return {}
        
        # For other providers, we need API calls
        # Details are often included in the search results, so we don't implement
        # separate details methods for now. In a production system, these would
        # be implemented.
        
        logger.warning(f"Detailed footage lookup not implemented for {provider}")
        return {}
    
    async def download_preview(
        self, 
        url: str, 
        output_path: str,
        provider: str
    ) -> Optional[str]:
        """
        Download preview of stock footage.
        
        Args:
            url: URL to download
            output_path: Path to save the preview
            provider: Stock provider
            
        Returns:
            Path to the downloaded file, or None if download failed
        """
        await self._init_session()
        
        # Handle local files by copying them
        if provider == 'local' and os.path.exists(url):
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy(url, output_path)
                return output_path
            except Exception as e:
                logger.error(f"Error copying local file: {str(e)}")
                return None
        
        # Download remote file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error downloading preview: {response.status}")
                    return None
                
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading preview: {str(e)}")
            return None
    
    async def find_footage_for_topic(
        self, 
        topic: str,
        options: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find stock footage for a specific topic.
        
        Args:
            topic: Topic to search for
            options: Search options
            provider: Stock provider
            
        Returns:
            List of footage metadata
        """
        # Clean up topic (remove special characters, etc.)
        search_query = topic.replace('-', ' ').replace('_', ' ')
        
        return await self.search_footage(search_query, provider, options)
    
    async def get_suggestions_for_content(
        self, 
        content_analysis: Dict[str, Any],
        max_results: int = 5,
        provider: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get suggested footage for analyzed content.
        
        Args:
            content_analysis: Content analysis results
            max_results: Maximum results per topic
            provider: Stock provider
            
        Returns:
            Dict mapping topics to lists of suggested footage
        """
        suggestions = {}
        
        # Get main topics from content analysis
        main_topics = content_analysis.get('overall_topics', [])[:5]
        
        # Add suggestions for each main topic
        for topic in main_topics:
            results = await self.find_footage_for_topic(
                topic,
                options={'per_page': max_results},
                provider=provider
            )
            
            suggestions[topic] = results
        
        # Get entities
        entities = content_analysis.get('entities', [])[:5]
        
        # Add suggestions for key entities
        for entity in entities:
            results = await self.find_footage_for_topic(
                entity,
                options={'per_page': max_results},
                provider=provider
            )
            
            suggestions[entity] = results
        
        # Get actions
        actions = content_analysis.get('key_actions', [])[:3]
        
        # Add suggestions for key actions
        for action in actions:
            results = await self.find_footage_for_topic(
                action,
                options={'per_page': max_results},
                provider=provider
            )
            
            suggestions[action] = results
        
        return suggestions
    
    def calculate_visual_similarity(
        self, 
        footage_a: Dict[str, Any],
        footage_b: Dict[str, Any]
    ) -> float:
        """
        Calculate visual similarity between two pieces of footage.
        
        Args:
            footage_a: First footage metadata
            footage_b: Second footage metadata
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Extract tags
        tags_a = set(footage_a.get('tags', []))
        tags_b = set(footage_b.get('tags', []))
        
        # Calculate tag overlap
        tag_overlap = len(tags_a.intersection(tags_b))
        tag_union = len(tags_a.union(tags_b))
        
        if tag_union == 0:
            return 0.0
        
        return tag_overlap / tag_union 