"""
External Music Service Integration Module

This module provides functionality for integrating with external music services
and APIs to expand the available music library with diverse, high-quality tracks
for video content creation.
"""

import os
import logging
import json
import time
import requests
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import hashlib
import base64

logger = logging.getLogger(__name__)

class ExternalMusicService:
    """
    Integration with external music services and APIs.
    
    This class provides methods to search, download, and manage music tracks
    from various external services, expanding the available music library
    for the video processing service.
    """
    
    # Supported external services
    SUPPORTED_SERVICES = {
        "jamendo": "Commercial use allowed API for Creative Commons music",
        "freemusicarchive": "Free Music Archive API for CC licensed music",
        "musopen": "Classical music in the public domain",
        "audiojungle": "Commercial music marketplace (requires license)",
        "artlist": "Commercial music marketplace (requires subscription)",
        "epidemic_sound": "Commercial music marketplace (requires subscription)",
        "bensound": "Royalty-free music with free and premium options"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the external music service integration.
        
        Args:
            config: Configuration options including API keys for services
        """
        self.config = config or {}
        
        # Set default parameters
        self.download_dir = self.config.get('download_dir', os.path.join(os.path.expanduser('~'), 'music_downloads'))
        self.cache_dir = self.config.get('cache_dir', os.path.join(os.path.expanduser('~'), 'music_cache'))
        self.results_cache_ttl = self.config.get('results_cache_ttl', 86400)  # 24 hours
        
        # Create directories if they don't exist
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set API credentials
        self.api_keys = {
            "jamendo": self.config.get('jamendo_api_key', ''),
            "freemusicarchive": self.config.get('freemusicarchive_api_key', ''),
            "musopen": self.config.get('musopen_api_key', ''),
            "audiojungle": self.config.get('audiojungle_api_key', ''),
            "artlist": self.config.get('artlist_api_key', ''),
            "epidemic_sound": self.config.get('epidemic_sound_api_key', ''),
            "bensound": self.config.get('bensound_api_key', '')
        }
        
        # Service-specific configurations
        self.service_config = {
            "jamendo": {
                "base_url": "https://api.jamendo.com/v3.0",
                "search_endpoint": "/tracks/",
                "download_endpoint": "/tracks/file/"
            },
            "freemusicarchive": {
                "base_url": "https://freemusicarchive.org/api",
                "search_endpoint": "/tracks.json",
                "download_endpoint": "/tracks/{track_id}/download"
            },
            # Other services configurations
        }
        
        # Initialize the request session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VideoProcessingService/1.0'
        })
        
        # Initialize cache
        self.search_cache = self._load_cache('search_cache.json')
    
    def search_tracks(
        self,
        query: str,
        service: Optional[str] = None,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        bpm: Optional[int] = None,
        duration: Optional[int] = None,
        license_type: Optional[str] = None,
        max_results: int = 20,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Search for tracks across external music services.
        
        Args:
            query: Search query term
            service: Specific service to search (None for all available)
            mood: Filter by mood
            genre: Filter by genre
            bpm: Filter by BPM (beats per minute)
            duration: Approximate duration in seconds
            license_type: Filter by license type (e.g., 'cc', 'commercial')
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with search results
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            query=query,
            service=service,
            mood=mood,
            genre=genre,
            bpm=bpm,
            duration=duration,
            license_type=license_type
        )
        
        # Check cache if enabled
        if use_cache and cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.results_cache_ttl:
                logger.info(f"Using cached results for query: {query}")
                return cache_entry['results']
        
        results = {
            "status": "success",
            "query": query,
            "total_results": 0,
            "tracks": []
        }
        
        try:
            # Determine which services to query
            services_to_query = [service] if service else self._get_available_services()
            
            all_tracks = []
            
            # Query each service
            for service_name in services_to_query:
                if service_name not in self.SUPPORTED_SERVICES:
                    logger.warning(f"Unsupported service: {service_name}")
                    continue
                
                if not self.api_keys.get(service_name):
                    logger.warning(f"Missing API key for service: {service_name}")
                    continue
                
                service_results = self._search_service(
                    service=service_name,
                    query=query,
                    mood=mood,
                    genre=genre,
                    bpm=bpm,
                    duration=duration,
                    license_type=license_type,
                    max_results=max_results
                )
                
                if service_results.get("status") == "success":
                    all_tracks.extend(service_results.get("tracks", []))
            
            # Sort tracks by relevance and limit results
            all_tracks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            results["tracks"] = all_tracks[:max_results]
            results["total_results"] = len(all_tracks)
            
            # Update cache
            if use_cache:
                self.search_cache[cache_key] = {
                    'timestamp': time.time(),
                    'results': results
                }
                self._save_cache('search_cache.json', self.search_cache)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            return {
                "status": "error",
                "query": query,
                "error": str(e)
            }
    
    def download_track(
        self,
        track_id: str,
        service: str,
        output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Download a track from an external service.
        
        Args:
            track_id: ID of the track to download
            service: Service to download from
            output_path: Path to save the downloaded track
            metadata: Additional metadata to save with the track
            
        Returns:
            Dictionary with download results
        """
        try:
            # Ensure service is supported
            if service not in self.SUPPORTED_SERVICES:
                return {
                    "status": "error",
                    "error": f"Unsupported service: {service}"
                }
            
            # Ensure API key is available
            if not self.api_keys.get(service):
                return {
                    "status": "error",
                    "error": f"Missing API key for service: {service}"
                }
            
            # Generate output path if not provided
            if not output_path:
                output_filename = f"{service}_{track_id}.mp3"
                output_path = os.path.join(self.download_dir, output_filename)
            
            # Download track based on service
            download_result = self._download_from_service(
                service=service,
                track_id=track_id,
                output_path=output_path
            )
            
            if download_result.get("status") != "success":
                return download_result
            
            # Save metadata if provided
            if metadata:
                metadata_path = f"{output_path}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            return {
                "status": "success",
                "track_id": track_id,
                "service": service,
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "metadata_saved": metadata is not None
            }
            
        except Exception as e:
            logger.error(f"Error downloading track: {str(e)}")
            return {
                "status": "error",
                "track_id": track_id,
                "service": service,
                "error": str(e)
            }
    
    def import_to_library(
        self,
        track_id: str,
        service: str,
        collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a track from an external service to the local music library.
        
        Args:
            track_id: ID of the track to import
            service: Service to import from
            collection_id: Optional collection ID to add the track to
            
        Returns:
            Dictionary with import results
        """
        try:
            # Download the track first
            download_result = self.download_track(
                track_id=track_id,
                service=service
            )
            
            if download_result.get("status") != "success":
                return download_result
            
            # Get track metadata
            track_info = self._get_track_info(
                track_id=track_id,
                service=service
            )
            
            if track_info.get("status") != "success":
                return track_info
            
            # Import to local library
            from app.services.music.music_library import MusicLibrary
            music_library = MusicLibrary()
            
            # Prepare track metadata for import
            track_data = {
                "file_path": download_result["file_path"],
                "title": track_info["title"],
                "artist": track_info["artist"],
                "genre": track_info.get("genre", "Unknown"),
                "mood": track_info.get("mood", "Neutral"),
                "bpm": track_info.get("bpm", 0),
                "duration": track_info.get("duration", 0),
                "copyright_free": track_info.get("copyright_free", True),
                "license_info": track_info.get("license_info", {}),
                "source": f"{service}:{track_id}",
                "tags": track_info.get("tags", [])
            }
            
            # Add track to library
            add_result = music_library.add_track(
                file_path=track_data["file_path"],
                metadata=track_data
            )
            
            # Add to collection if specified
            if collection_id and add_result.get("status") == "success":
                collection_result = music_library.add_tracks_to_collection(
                    collection_id=collection_id,
                    track_ids=[add_result["track_id"]]
                )
                
                if collection_result.get("status") == "success":
                    add_result["added_to_collection"] = True
            
            return add_result
            
        except Exception as e:
            logger.error(f"Error importing track to library: {str(e)}")
            return {
                "status": "error",
                "track_id": track_id,
                "service": service,
                "error": str(e)
            }
    
    def get_service_info(self, service: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about supported services.
        
        Args:
            service: Optional specific service to get info for
            
        Returns:
            Dictionary with service information
        """
        if service:
            if service not in self.SUPPORTED_SERVICES:
                return {
                    "status": "error",
                    "error": f"Unsupported service: {service}"
                }
            
            return {
                "status": "success",
                "service": service,
                "description": self.SUPPORTED_SERVICES[service],
                "api_key_configured": bool(self.api_keys.get(service)),
                "configuration": self.service_config.get(service, {})
            }
        
        # Return info for all services
        services_info = {}
        for service_name, description in self.SUPPORTED_SERVICES.items():
            services_info[service_name] = {
                "description": description,
                "api_key_configured": bool(self.api_keys.get(service_name)),
                "configuration": self.service_config.get(service_name, {})
            }
        
        return {
            "status": "success",
            "services": services_info
        }
    
    def _search_service(
        self,
        service: str,
        query: str,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        bpm: Optional[int] = None,
        duration: Optional[int] = None,
        license_type: Optional[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search for tracks within a specific service.
        
        Args:
            service: Service to search
            query: Search query term
            mood: Filter by mood
            genre: Filter by genre
            bpm: Filter by BPM
            duration: Approximate duration in seconds
            license_type: Filter by license type
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        if service == "jamendo":
            return self._search_jamendo(
                query=query,
                mood=mood,
                genre=genre,
                bpm=bpm,
                duration=duration,
                license_type=license_type,
                max_results=max_results
            )
        elif service == "freemusicarchive":
            return self._search_freemusicarchive(
                query=query,
                mood=mood,
                genre=genre,
                duration=duration,
                license_type=license_type,
                max_results=max_results
            )
        # Implement other services as needed
        else:
            return {
                "status": "error",
                "error": f"Search not implemented for service: {service}"
            }
    
    def _download_from_service(
        self,
        service: str,
        track_id: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Download a track from a specific service.
        
        Args:
            service: Service to download from
            track_id: ID of the track to download
            output_path: Path to save the downloaded track
            
        Returns:
            Dictionary with download results
        """
        if service == "jamendo":
            return self._download_from_jamendo(
                track_id=track_id,
                output_path=output_path
            )
        elif service == "freemusicarchive":
            return self._download_from_freemusicarchive(
                track_id=track_id,
                output_path=output_path
            )
        # Implement other services as needed
        else:
            return {
                "status": "error",
                "error": f"Download not implemented for service: {service}"
            }
    
    def _get_track_info(
        self,
        track_id: str,
        service: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a track.
        
        Args:
            track_id: ID of the track
            service: Service the track belongs to
            
        Returns:
            Dictionary with track information
        """
        if service == "jamendo":
            return self._get_jamendo_track_info(track_id)
        elif service == "freemusicarchive":
            return self._get_freemusicarchive_track_info(track_id)
        # Implement other services as needed
        else:
            return {
                "status": "error",
                "error": f"Getting track info not implemented for service: {service}"
            }
    
    def _search_jamendo(
        self, 
        query: str,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        bpm: Optional[int] = None,
        duration: Optional[int] = None,
        license_type: Optional[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search for tracks on Jamendo.
        
        Args:
            query: Search query
            mood: Filter by mood
            genre: Filter by genre
            bpm: Filter by BPM
            duration: Approximate duration in seconds
            license_type: Filter by license type
            max_results: Maximum results
            
        Returns:
            Dictionary with search results
        """
        try:
            api_key = self.api_keys["jamendo"]
            base_url = self.service_config["jamendo"]["base_url"]
            endpoint = self.service_config["jamendo"]["search_endpoint"]
            
            # Build request parameters
            params = {
                "client_id": api_key,
                "format": "json",
                "limit": max_results,
                "name": query,
                "include": "musicinfo+stats"
            }
            
            # Add filters if provided
            if genre:
                params["tags"] = genre
            
            if mood:
                params["mood"] = mood
            
            if license_type:
                # Map license type to Jamendo license terms
                license_map = {
                    "cc": "ccby,ccbysa,ccbync,ccbyncsa,ccbyncnd",
                    "commercial": "ccby,ccbysa"
                }
                params["license"] = license_map.get(license_type, "")
            
            # Make API request
            url = f"{base_url}{endpoint}"
            response = self.session.get(url, params=params)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"Jamendo API error: {response.status_code}"
                }
            
            data = response.json()
            
            # Process results
            tracks = []
            for track in data.get("results", []):
                # Calculate relevance score based on popularity and match
                popularity = track.get("stats", {}).get("listened", 0) / 10000
                relevance_score = min(1.0, popularity)
                
                # Format track data
                track_data = {
                    "track_id": track["id"],
                    "title": track["name"],
                    "artist": track["artist_name"],
                    "duration": track.get("duration", 0),
                    "file_path": None,  # To be filled when downloaded
                    "preview_url": track.get("audio", ""),
                    "service": "jamendo",
                    "genre": track.get("musicinfo", {}).get("tags", ["unknown"])[0] if track.get("musicinfo", {}).get("tags") else "unknown",
                    "mood": mood or "unknown",
                    "bpm": track.get("musicinfo", {}).get("bpm", 0),
                    "license_info": {
                        "type": track.get("license_ccurl", "").split("/")[-2] if track.get("license_ccurl") else "unknown",
                        "url": track.get("license_ccurl", ""),
                        "attribution": f"{track['artist_name']} - {track['name']} (CC License via Jamendo)"
                    },
                    "copyright_free": True,  # All Jamendo tracks are CC licensed
                    "relevance_score": relevance_score,
                    "source_url": track.get("shareurl", ""),
                    "tags": track.get("musicinfo", {}).get("tags", [])
                }
                
                tracks.append(track_data)
            
            return {
                "status": "success",
                "query": query,
                "service": "jamendo",
                "total_results": data.get("headers", {}).get("results_count", 0),
                "tracks": tracks
            }
            
        except Exception as e:
            logger.error(f"Error searching Jamendo: {str(e)}")
            return {
                "status": "error",
                "service": "jamendo",
                "error": str(e)
            }
    
    def _search_freemusicarchive(
        self, 
        query: str,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        duration: Optional[int] = None,
        license_type: Optional[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search for tracks on Free Music Archive.
        
        Args:
            query: Search query
            mood: Filter by mood
            genre: Filter by genre
            duration: Approximate duration in seconds
            license_type: Filter by license type
            max_results: Maximum results
            
        Returns:
            Dictionary with search results
        """
        # Implementation for Free Music Archive search
        # Note: Free Music Archive API structure would be implemented here
        # This is a placeholder
        
        return {
            "status": "error",
            "error": "Free Music Archive integration not fully implemented yet"
        }
    
    def _download_from_jamendo(
        self,
        track_id: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Download a track from Jamendo.
        
        Args:
            track_id: ID of the track to download
            output_path: Path to save the downloaded track
            
        Returns:
            Dictionary with download results
        """
        try:
            api_key = self.api_keys["jamendo"]
            base_url = self.service_config["jamendo"]["base_url"]
            endpoint = f"/tracks/file/?client_id={api_key}&id={track_id}"
            
            # Make API request
            url = f"{base_url}{endpoint}"
            response = self.session.get(url, stream=True)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"Jamendo API error: {response.status_code}"
                }
            
            # Save file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "status": "success",
                "track_id": track_id,
                "service": "jamendo",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path)
            }
            
        except Exception as e:
            logger.error(f"Error downloading from Jamendo: {str(e)}")
            return {
                "status": "error",
                "service": "jamendo",
                "track_id": track_id,
                "error": str(e)
            }
    
    def _download_from_freemusicarchive(
        self,
        track_id: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Download a track from Free Music Archive.
        
        Args:
            track_id: ID of the track to download
            output_path: Path to save the downloaded track
            
        Returns:
            Dictionary with download results
        """
        # Implementation for Free Music Archive download
        # This is a placeholder
        
        return {
            "status": "error",
            "error": "Free Music Archive download not fully implemented yet"
        }
    
    def _get_jamendo_track_info(self, track_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a Jamendo track.
        
        Args:
            track_id: ID of the track
            
        Returns:
            Dictionary with track information
        """
        try:
            api_key = self.api_keys["jamendo"]
            base_url = self.service_config["jamendo"]["base_url"]
            endpoint = f"/tracks/?client_id={api_key}&id={track_id}&include=musicinfo+stats"
            
            # Make API request
            url = f"{base_url}{endpoint}"
            response = self.session.get(url)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"Jamendo API error: {response.status_code}"
                }
            
            data = response.json()
            
            if not data.get("results"):
                return {
                    "status": "error",
                    "error": "Track not found"
                }
            
            track = data["results"][0]
            
            # Format track data
            track_info = {
                "status": "success",
                "track_id": track["id"],
                "title": track["name"],
                "artist": track["artist_name"],
                "duration": track.get("duration", 0),
                "preview_url": track.get("audio", ""),
                "service": "jamendo",
                "genre": track.get("musicinfo", {}).get("tags", ["unknown"])[0] if track.get("musicinfo", {}).get("tags") else "unknown",
                "bpm": track.get("musicinfo", {}).get("bpm", 0),
                "license_info": {
                    "type": track.get("license_ccurl", "").split("/")[-2] if track.get("license_ccurl") else "unknown",
                    "url": track.get("license_ccurl", ""),
                    "attribution": f"{track['artist_name']} - {track['name']} (CC License via Jamendo)"
                },
                "copyright_free": True,  # All Jamendo tracks are CC licensed
                "source_url": track.get("shareurl", ""),
                "tags": track.get("musicinfo", {}).get("tags", [])
            }
            
            return track_info
            
        except Exception as e:
            logger.error(f"Error getting Jamendo track info: {str(e)}")
            return {
                "status": "error",
                "service": "jamendo",
                "track_id": track_id,
                "error": str(e)
            }
    
    def _get_freemusicarchive_track_info(self, track_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a Free Music Archive track.
        
        Args:
            track_id: ID of the track
            
        Returns:
            Dictionary with track information
        """
        # Implementation for Free Music Archive track info
        # This is a placeholder
        
        return {
            "status": "error",
            "error": "Free Music Archive track info not fully implemented yet"
        }
    
    def _generate_cache_key(self, **params) -> str:
        """
        Generate a cache key from search parameters.
        
        Args:
            **params: Search parameters
            
        Returns:
            Cache key string
        """
        # Convert params to sorted string
        param_str = json.dumps(params, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _load_cache(self, cache_filename: str) -> Dict[str, Any]:
        """
        Load cache from file.
        
        Args:
            cache_filename: Name of cache file
            
        Returns:
            Cache dictionary
        """
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
        
        return {}
    
    def _save_cache(self, cache_filename: str, cache_data: Dict[str, Any]) -> None:
        """
        Save cache to file.
        
        Args:
            cache_filename: Name of cache file
            cache_data: Cache data to save
        """
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _get_available_services(self) -> List[str]:
        """
        Get list of available services with configured API keys.
        
        Returns:
            List of available service names
        """
        return [name for name, key in self.api_keys.items() if key] 