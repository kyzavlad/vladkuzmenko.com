"""
External Music Service Example

This example demonstrates how to use the External Music Service to:
1. Search for music tracks across external services
2. Download tracks from external services
3. Import tracks to the local music library
4. Get information about supported services

Note: For this example to work, you need to have API keys configured for the
services you want to use. See the README.md for more information on how to
configure API keys.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List, Optional

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the required modules
from app.services.music.external_music_service import ExternalMusicService
from app.services.music.music_library import MusicLibrary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_external_music_service(config: Optional[Dict[str, Any]] = None) -> ExternalMusicService:
    """
    Initialize the External Music Service with the provided configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized ExternalMusicService instance
    """
    if config is None:
        # Default configuration with example API keys
        # In a real application, these would be loaded from environment variables
        # or a configuration file
        config = {
            'jamendo_api_key': os.environ.get('JAMENDO_API_KEY', ''),
            'freemusicarchive_api_key': os.environ.get('FMA_API_KEY', ''),
            'download_dir': os.path.join(os.path.dirname(__file__), 'downloads'),
            'cache_dir': os.path.join(os.path.dirname(__file__), 'cache')
        }
    
    # Create the External Music Service
    external_service = ExternalMusicService(config)
    
    return external_service

def search_example(
    external_service: ExternalMusicService,
    query: str,
    service: Optional[str] = None,
    mood: Optional[str] = None,
    genre: Optional[str] = None,
    license_type: Optional[str] = None,
    max_results: int = 5
) -> None:
    """
    Example of searching for tracks across external services.
    
    Args:
        external_service: ExternalMusicService instance
        query: Search query
        service: Optional service to search
        mood: Optional mood filter
        genre: Optional genre filter
        license_type: Optional license type filter
        max_results: Maximum number of results to return
    """
    logger.info(f"Searching for '{query}'...")
    
    # Search for tracks
    results = external_service.search_tracks(
        query=query,
        service=service,
        mood=mood,
        genre=genre,
        license_type=license_type,
        max_results=max_results
    )
    
    # Check if search was successful
    if results.get('status') != 'success':
        logger.error(f"Search failed: {results.get('error')}")
        return
    
    # Print search results
    logger.info(f"Found {results.get('total_results', 0)} tracks:")
    
    for i, track in enumerate(results.get('tracks', [])):
        logger.info(f"Track {i+1}:")
        logger.info(f"  ID: {track.get('track_id')}")
        logger.info(f"  Title: {track.get('title')}")
        logger.info(f"  Artist: {track.get('artist')}")
        logger.info(f"  Service: {track.get('service')}")
        logger.info(f"  Genre: {track.get('genre')}")
        logger.info(f"  Duration: {track.get('duration')} seconds")
        logger.info(f"  Preview URL: {track.get('preview_url')}")
        logger.info(f"  Copyright free: {track.get('copyright_free', False)}")
        logger.info(f"  License: {track.get('license_info', {}).get('type')}")
        logger.info("")
    
    return results

def download_example(
    external_service: ExternalMusicService,
    track_id: str,
    service: str
) -> None:
    """
    Example of downloading a track from an external service.
    
    Args:
        external_service: ExternalMusicService instance
        track_id: ID of the track to download
        service: Service to download from
    """
    logger.info(f"Downloading track {track_id} from {service}...")
    
    # Download the track
    results = external_service.download_track(
        track_id=track_id,
        service=service
    )
    
    # Check if download was successful
    if results.get('status') != 'success':
        logger.error(f"Download failed: {results.get('error')}")
        return
    
    # Print download results
    logger.info(f"Download successful:")
    logger.info(f"  File path: {results.get('file_path')}")
    logger.info(f"  File size: {results.get('file_size')} bytes")
    
    return results

def import_example(
    external_service: ExternalMusicService,
    track_id: str,
    service: str,
    collection_id: Optional[str] = None
) -> None:
    """
    Example of importing a track to the local music library.
    
    Args:
        external_service: ExternalMusicService instance
        track_id: ID of the track to import
        service: Service to import from
        collection_id: Optional collection ID to add the track to
    """
    logger.info(f"Importing track {track_id} from {service}...")
    
    # Import the track
    results = external_service.import_to_library(
        track_id=track_id,
        service=service,
        collection_id=collection_id
    )
    
    # Check if import was successful
    if results.get('status') != 'success':
        logger.error(f"Import failed: {results.get('error')}")
        return
    
    # Print import results
    logger.info(f"Import successful:")
    logger.info(f"  Library track ID: {results.get('track_id')}")
    logger.info(f"  File path: {results.get('file_path')}")
    
    # Check if the track was added to a collection
    if results.get('added_to_collection'):
        logger.info(f"  Added to collection: {collection_id}")
    
    return results

def service_info_example(
    external_service: ExternalMusicService,
    service: Optional[str] = None
) -> None:
    """
    Example of getting information about supported services.
    
    Args:
        external_service: ExternalMusicService instance
        service: Optional specific service to get info for
    """
    logger.info(f"Getting service info for {service if service else 'all services'}...")
    
    # Get service info
    results = external_service.get_service_info(service)
    
    # Check if getting info was successful
    if results.get('status') != 'success':
        logger.error(f"Getting service info failed: {results.get('error')}")
        return
    
    # Print service info
    if service:
        # Info for a specific service
        logger.info(f"Service info for {service}:")
        logger.info(f"  Description: {results.get('description')}")
        logger.info(f"  API key configured: {results.get('api_key_configured')}")
        logger.info(f"  Configuration: {results.get('configuration')}")
    else:
        # Info for all services
        logger.info("Supported services:")
        for service_name, info in results.get('services', {}).items():
            logger.info(f"  {service_name}:")
            logger.info(f"    Description: {info.get('description')}")
            logger.info(f"    API key configured: {info.get('api_key_configured')}")
    
    return results

def create_collection_example(collection_name: str) -> str:
    """
    Example of creating a collection in the local music library.
    
    Args:
        collection_name: Name of the collection to create
        
    Returns:
        ID of the created collection
    """
    logger.info(f"Creating collection '{collection_name}'...")
    
    # Initialize the Music Library
    music_library = MusicLibrary()
    
    # Create the collection
    results = music_library.create_collection(
        name=collection_name,
        description=f"Collection created by external_music_service_example.py"
    )
    
    # Check if creation was successful
    if results.get('status') != 'success':
        logger.error(f"Creating collection failed: {results.get('error')}")
        return ""
    
    # Print creation results
    logger.info(f"Collection created:")
    logger.info(f"  Collection ID: {results.get('collection_id')}")
    
    return results.get('collection_id', '')

def main():
    """
    Main function to parse command line arguments and run the example.
    """
    parser = argparse.ArgumentParser(description='External Music Service Example')
    parser.add_argument('operation', choices=['search', 'download', 'import', 'info', 'all'],
                      help='Operation to perform')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--service', help='Service to use')
    parser.add_argument('--track-id', help='Track ID to download or import')
    parser.add_argument('--mood', help='Mood filter for search')
    parser.add_argument('--genre', help='Genre filter for search')
    parser.add_argument('--license-type', help='License type filter for search')
    parser.add_argument('--collection-name', help='Name for a new collection to create')
    
    args = parser.parse_args()
    
    # Initialize the External Music Service
    external_service = init_external_music_service()
    
    # Check for API keys
    service_info = external_service.get_service_info()
    available_services = [
        name for name, info in service_info.get('services', {}).items()
        if info.get('api_key_configured')
    ]
    
    if not available_services:
        logger.warning("No API keys configured for any service. Examples may not work.")
        logger.info("Set environment variables with API keys to use specific services.")
        logger.info("E.g., JAMENDO_API_KEY=your_api_key FMA_API_KEY=your_api_key python external_music_service_example.py")
    else:
        logger.info(f"Available services: {', '.join(available_services)}")
    
    # Run the requested operation
    if args.operation == 'search' or args.operation == 'all':
        if not args.query:
            if args.operation == 'search':
                logger.error("Search operation requires a query. Use --query.")
                return
            query = "piano"  # Default query for 'all' operation
        else:
            query = args.query
        
        # Use the first available service if none specified
        service = args.service or (available_services[0] if available_services else None)
        
        search_example(
            external_service=external_service,
            query=query,
            service=service,
            mood=args.mood,
            genre=args.genre,
            license_type=args.license_type
        )
    
    if args.operation == 'download' or args.operation == 'all':
        if not args.track_id and args.operation == 'download':
            logger.error("Download operation requires a track ID. Use --track-id.")
            return
        
        # If 'all' operation, perform a search first to get a track ID
        if args.operation == 'all':
            query = args.query or "piano"
            service = args.service or (available_services[0] if available_services else None)
            
            if not service:
                logger.error("No available service for search.")
                return
            
            # Search for tracks
            search_results = search_example(
                external_service=external_service,
                query=query,
                service=service,
                max_results=1
            )
            
            # Get the first track
            if search_results and search_results.get('tracks'):
                track = search_results['tracks'][0]
                track_id = track['track_id']
                service = track['service']
            else:
                logger.error("No tracks found in search.")
                return
        else:
            # Use the provided track ID and service
            track_id = args.track_id
            service = args.service
            
            if not service:
                logger.error("Download operation requires a service. Use --service.")
                return
        
        download_example(
            external_service=external_service,
            track_id=track_id,
            service=service
        )
    
    if args.operation == 'import' or args.operation == 'all':
        # For 'import' operation, we need a collection
        collection_id = None
        if args.collection_name:
            collection_id = create_collection_example(args.collection_name)
        
        if not args.track_id and args.operation == 'import':
            logger.error("Import operation requires a track ID. Use --track-id.")
            return
        
        # If 'all' operation, use the track from download
        if args.operation == 'all':
            query = args.query or "piano"
            service = args.service or (available_services[0] if available_services else None)
            
            if not service:
                logger.error("No available service for search.")
                return
            
            # Search for tracks
            search_results = search_example(
                external_service=external_service,
                query=query,
                service=service,
                max_results=1
            )
            
            # Get the first track
            if search_results and search_results.get('tracks'):
                track = search_results['tracks'][0]
                track_id = track['track_id']
                service = track['service']
            else:
                logger.error("No tracks found in search.")
                return
        else:
            # Use the provided track ID and service
            track_id = args.track_id
            service = args.service
            
            if not service:
                logger.error("Import operation requires a service. Use --service.")
                return
        
        import_example(
            external_service=external_service,
            track_id=track_id,
            service=service,
            collection_id=collection_id
        )
    
    if args.operation == 'info' or args.operation == 'all':
        service_info_example(
            external_service=external_service,
            service=args.service
        )

if __name__ == "__main__":
    main() 