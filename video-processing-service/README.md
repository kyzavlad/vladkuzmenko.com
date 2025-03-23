# Video Processing Service

A comprehensive video processing service with advanced subtitle generation, B-Roll insertion capabilities, and AI avatar creation.

## Features

### Subtitle Generation
- Automatic speech recognition and transcription
- Precise timestamp alignment
- Multiple language support
- Customizable subtitle formatting

### B-Roll Insertion Engine
- Content analysis for relevant B-Roll suggestions
- Automatic scene detection and transition points
- Stock footage integration
- User library integration for personalized B-Roll
- AI-powered scene composition recommendations

### Audio Enhancement Suite
- Voice clarity improvement
- Noise reduction and removal
- Environmental sound classification
- Audio normalization to broadcast standards
- Dynamic sound balancing

### AI Avatar Creation

The service includes state-of-the-art AI avatar creation capabilities:

#### 3D Face Reconstruction
- High-fidelity 3D face modeling from 2D images or videos
- 4K resolution texture mapping for realistic skin details
- Detailed feature preservation algorithm for authentic appearances
  * Region-based feature weighting system (eyes, nose, mouth, contour)
  * Geometric constraints to maintain facial proportions
  * Local detail preservation via Laplacian mesh editing
  * Statistical anatomical constraints for realism
- StyleGAN-3 implementation with custom enhancements
- Expression range calibration for natural animations

#### Animation Framework
- First Order Motion Model with temporal consistency
  * Advanced motion transfer from driving videos
  * Smooth transitions between expressions
  * Reduced jitter and artifacts
- 68-point facial landmark tracking
  * Accurate tracking of key facial points
  * Region-based movement control
  * Preservation of facial geometry during animation
- Natural micro-expression synthesis
  * Subtle facial movements for increased realism
  * Randomized micro-movements based on emotional context
  * Blink and mouth micro-movements
- Gaze direction modeling and control
  * Natural eye movement patterns
  * Attention-based gaze behaviors
  * Saccadic eye movements
- Head pose variation with natural limits
  * Physics-based head movement constraints
  * Natural rotation and translation limits
  * Momentum-based movement for realistic motion
- Emotion intensity adjustment
  * Fine-grained control over emotional expressions
  * Blending between different emotional states
  * Context-sensitive expression modulation
- Person-specific gesture and mannerism learning
  * Capture and reproduction of unique personal traits
  * Style transfer from reference videos
  * Characteristic movement patterns
- Identity consistency verification
- High-resolution detail refinement for ultra-realistic skin features
  * Multi-layered skin pore simulation with variable sizing and distribution
  * Fine wrinkle generation based on facial anatomy (forehead, eye areas)
  * Surface irregularity mapping for natural skin appearance
  * Subsurface scattering simulation for translucent skin effects
  * Adjustable detail levels from low to ultra quality

#### Expression Range Calibration
The expression range calibration system provides sophisticated control over facial animations:

- **Personalized Expression Ranges**: Automatically calibrates expression intensity limits based on individual facial morphology
- **Muscle Group Mapping**: Maps expressions to specific facial muscle groups for anatomically correct animations
- **Expression Compatibility Matrix**: Calculates how different expressions can blend together naturally
- **Multiple Expression Types**: Supports a wide range of expressions including:
  * Basic: smile, frown, surprise, anger
  * Advanced: squint, pout, jaw movements
- **Anatomical Constraints**: Prevents unrealistic deformations by applying biomechanical limits
- **Proportion-based Calibration**: Adjusts expression ranges based on facial proportions (e.g., wider mouths have different smile parameters)
- **Expression Visualization**: Provides visual analytics of expression intensity ranges and compatibility

Examples:
```bash
# Analyze expressions from an image
python examples/expression_calibration_example.py --image path/to/face.jpg

# Generate expression analysis from video
python examples/expression_calibration_example.py --video path/to/face.mp4

# Compare and visualize different expressions
python examples/expression_calibration_example.py --image path/to/face.jpg --compare-expressions

# Verify identity consistency with a reference image
python examples/identity_verification_example.py --image path/to/face.jpg --reference path/to/reference.jpg
```

Example output includes detailed expression data in JSON format and visualizations of expression intensity ranges and compatibility matrices.

### Music Selection & Integration

The service includes advanced music integration capabilities to enhance videos with appropriate audio tracks:

### Dynamic Volume Adjustment
- Automatically adjusts background music volume during speech segments
- Ensures dialogue intelligibility while maintaining pleasant background music
- Smooth fade-in and fade-out transitions between volume levels
- Advanced speech detection with configurable sensitivity
- Preserves original audio quality while mixing in music tracks

### Genre Classification
- Classifies audio tracks into 20+ musical genres
- Analyzes audio features to identify genre characteristics
- Provides probability scores for multiple matching genres
- Audio feature extraction including tempo, timbre, and dynamics

### Music Integration API

```
POST /api/v1/audio/adjust-music-volume
```

Dynamically adjust music volume during speech segments in a video.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/adjust-music-volume",
    json={
        "video_path": "/path/to/video.mp4",
        "music_path": "/path/to/music.mp3",
        "music_start_time": 0.0,
        "default_volume": 0.7,
        "ducking_amount": 0.3,
        "fade_in_time": 0.5,
        "fade_out_time": 0.8,
        "keep_original_audio": true
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "message": "Volume adjustment job started",
    "job_id": "volume_adj_1623456789"
}
```

### Content Mood Analysis
- Advanced video mood analysis using the valence-arousal model
- Multi-modal analysis combining audio, visual, and transcript cues
- Emotional timeline generation for segment-by-segment mood tracking
- AI-powered sentiment analysis of dialogue and narration
- Considers color palettes, motion dynamics, and audio characteristics
- Direct music mood matching based on emotional content

### Music Integration API

```
POST /api/v1/audio/analyze-mood
```

Analyze the emotional mood and tone of video content.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/analyze-mood",
    json={
        "file_path": "/path/to/video.mp4",
        "include_audio_analysis": True,
        "include_visual_analysis": True,
        "include_transcript_analysis": True,
        "transcript_path": "/path/to/transcript.json",
        "segment_duration": 5,
        "detailed_results": True
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "file_path": "/path/to/video.mp4",
    "primary_mood": "inspiring",
    "valence": 0.67,
    "arousal": 0.58,
    "mood_scores": {
        "inspiring": 0.85,
        "happy": 0.72,
        "nostalgic": 0.45,
        "relaxed": 0.32,
        "dramatic": 0.18
    },
    "recommended_music_moods": ["inspiring", "uplifting", "motivational"],
    "detailed_results": {
        "timeline": [
            {"start_time": 0.0, "end_time": 5.0, "mood": "neutral"},
            {"start_time": 5.0, "end_time": 10.0, "mood": "happy"},
            {"start_time": 10.0, "end_time": 15.0, "mood": "inspiring"}
        ]
    }
}
```

### Emotional Arc Mapping
- Maps how emotions evolve over time throughout video content
- Identifies story patterns such as rising action, climax, and resolution
- Detects key emotional moments for synchronized music changes
- Creates optimized music cue points for professional soundtracks
- Generates visual timeline of emotional journey
- Analyzes emotional complexity and dynamics

### Music Integration API

```
POST /api/v1/audio/map-emotional-arc
```

Map the emotional arc of video content over time.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/map-emotional-arc",
    json={
        "file_path": "/path/to/video.mp4",
        "transcript_path": "/path/to/transcript.json",
        "segment_duration": 5,
        "detect_key_moments": True,
        "smooth_arc": True
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "file_path": "/path/to/video.mp4",
    "arc_pattern": "arc",
    "arc_confidence": 0.83,
    "pattern_description": "Rise followed by fall (classic narrative arc)",
    "emotional_dynamics": {
        "emotional_range": {
            "valence_range": [-0.5, 0.8],
            "arousal_range": [0.2, 0.9]
        },
        "emotional_variability": 0.45,
        "mood_diversity": 5,
        "emotional_complexity": "moderate"
    },
    "key_moments": [
        {
            "time": 65.5,
            "type": "peak",
            "moment_mood": "excited",
            "transition_type": "intensification",
            "description": "Intensification transition to excited"
        }
    ],
    "music_cues": [
        {
            "time": 0,
            "cue_type": "intro",
            "mood": "neutral",
            "description": "Start with neutral music"
        },
        {
            "time": 65.5,
            "cue_type": "transition",
            "mood": "excited",
            "description": "Transition to excited at key moment"
        },
        {
            "time": 150.0,
            "cue_type": "outro",
            "mood": "nostalgic",
            "description": "End with nostalgic music"
        }
    ]
}
```

### Custom Music Library
- Organizes music tracks in a central library with rich metadata
- Groups tracks into custom collections for different projects
- Searches music by mood, genre, BPM, duration, and keywords  
- Manages copyright status and licensing information
- Provides intelligent track recommendations with advanced matching algorithms

```
POST /api/v1/audio/library/add-track
```

Add a track to the music library with detailed metadata.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/add-track",
    json={
        "file_path": "/path/to/music.mp3",
        "title": "Inspiring Journey",
        "artist": "Audio Composer",
        "mood": "inspiring",
        "genre": "cinematic",
        "bpm": 120.5,
        "tags": ["uplifting", "corporate", "background"],
        "description": "An inspiring track with modern cinematic elements",
        "copyright_free": True,
        "license": "Creative Commons Attribution 4.0"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "track_id": "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
    "message": "Track 'Inspiring Journey' added to library"
}
```

```
POST /api/v1/audio/library/search
```

Search for tracks based on various criteria with intelligent matching.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/search",
    json={
        "mood": "inspiring",
        "tempo": 120,
        "genre": "cinematic",
        "duration": 180,
        "keywords": ["uplifting", "corporate"],
        "max_results": 5,
        "copyright_free_only": True,
        "collection_id": "5c4a6b9d-21e3-4c88-9f45-8c3d4f7a8e5b"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "tracks": [
        {
            "id": "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
            "title": "Inspiring Journey",
            "artist": "Audio Composer",
            "mood": "inspiring",
            "genre": "cinematic",
            "bpm": 120.5,
            "duration": 185.3,
            "copyright_free": true,
            "relevance_score": 0.92
        },
        {...}
    ],
    "total_matches": 12
}
```

```
POST /api/v1/audio/library/collection/create
```

Create a music collection to organize tracks.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/collection/create",
    json={
        "name": "Corporate Videos",
        "description": "Uplifting tracks for corporate video content",
        "tags": ["corporate", "business", "professional"]
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "collection_id": "5c4a6b9d-21e3-4c88-9f45-8c3d4f7a8e5b",
    "message": "Collection 'Corporate Videos' created"
}
```

```
POST /api/v1/audio/library/collection/add-tracks
```

Add tracks to a collection.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/collection/add-tracks",
    json={
        "collection_id": "5c4a6b9d-21e3-4c88-9f45-8c3d4f7a8e5b",
        "track_ids": [
            "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
            "fa3e7c2d-9b5a-4e81-8d67-0c12f3b9ae56"
        ]
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "message": "Added 2 tracks to collection 'Corporate Videos'",
    "added_tracks": [
        "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
        "fa3e7c2d-9b5a-4e81-8d67-0c12f3b9ae56"
    ]
}
```

Other library endpoints:
- `GET /api/v1/audio/library/tracks` - List all tracks in the library
- `GET /api/v1/audio/library/track/{track_id}` - Get details of a specific track
- `DELETE /api/v1/audio/library/track/{track_id}` - Remove a track from the library
- `PATCH /api/v1/audio/library/track` - Update track metadata
- `GET /api/v1/audio/library/collections` - List all collections
- `GET /api/v1/audio/library/collection/{collection_id}` - Get details of a specific collection with its tracks
- `PATCH /api/v1/audio/library/collection` - Update collection metadata
- `DELETE /api/v1/audio/library/collection/{collection_id}` - Delete a collection
- `POST /api/v1/audio/library/collection/remove-tracks` - Remove tracks from a collection

### External Music Service Integration

The External Music Service Integration enables you to expand your music library by connecting to various external music services and APIs. This feature allows searching, downloading, and importing tracks from platforms like Jamendo, Free Music Archive, and others, providing access to a vast collection of music tracks for your videos.

**Key Features:**
- Integration with multiple music services through a unified API
- Rich metadata retrieval with track details, including mood, genre, BPM
- License information management for copyright compliance
- Intelligent caching to improve performance and reduce API calls
- Seamless import to the local music library, with collection organization

#### API Endpoints

**Search for tracks:**
```
POST /api/v1/audio/external/search
```
Example request:
```json
{
  "query": "upbeat guitar",
  "service": "jamendo",
  "mood": "happy",
  "genre": "rock",
  "license_type": "commercial",
  "max_results": 10
}
```
Example response:
```json
{
  "status": "success",
  "query": "upbeat guitar",
  "total_results": 3,
  "tracks": [
    {
      "track_id": "1234567",
      "title": "Summer Vibes",
      "artist": "Guitar Hero",
      "service": "jamendo",
      "genre": "rock",
      "mood": "happy",
      "duration": 180,
      "preview_url": "https://example.com/preview.mp3",
      "copyright_free": true,
      "license_info": {
        "type": "ccby",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Guitar Hero - Summer Vibes (CC License via Jamendo)"
      }
    },
    // Additional tracks...
  ]
}
```

**Download a track:**
```
POST /api/v1/audio/external/download
```
Example request:
```json
{
  "track_id": "1234567",
  "service": "jamendo",
  "include_metadata": true
}
```
Example response:
```json
{
  "status": "success",
  "track_id": "1234567",
  "service": "jamendo",
  "file_path": "/path/to/downloaded/track.mp3",
  "file_size": 3456789,
  "metadata_saved": true
}
```

**Import to library:**
```
POST /api/v1/audio/external/import
```
Example request:
```json
{
  "track_id": "1234567",
  "service": "jamendo",
  "collection_id": "my-collection-id"
}
```
Example response:
```json
{
  "status": "success",
  "track_id": "local-library-track-id",
  "file_path": "/path/to/library/track.mp3",
  "added_to_collection": true
}
```

**Get service information:**
```
POST /api/v1/audio/external/service-info
```
Example request:
```json
{
  "service": "jamendo"
}
```
Example response:
```json
{
  "status": "success",
  "service": "jamendo",
  "description": "Commercial use allowed API for Creative Commons music",
  "api_key_configured": true,
  "configuration": {
    "base_url": "https://api.jamendo.com/v3.0",
    "search_endpoint": "/tracks/",
    "download_endpoint": "/tracks/file/"
  }
}
```

#### Example Usage

The following example demonstrates how to search, download, and import a track from an external service:

```python
from app.services.music.external_music_service import ExternalMusicService

# Initialize the service
external_service = ExternalMusicService()

# Search for tracks
search_results = external_service.search_tracks(
    query="upbeat guitar",
    service="jamendo",
    mood="happy",
    genre="rock",
    max_results=5
)

# Download the first track
if search_results["status"] == "success" and search_results["tracks"]:
    track = search_results["tracks"][0]
    download_result = external_service.download_track(
        track_id=track["track_id"],
        service=track["service"]
    )
    
    # Import to library
    if download_result["status"] == "success":
        import_result = external_service.import_to_library(
            track_id=track["track_id"],
            service=track["service"],
            collection_id="my-collection-id"
        )
```

For a complete example, see the `external_music_service_example.py` script in the `examples` directory.

## System Requirements

- Python 3.10+
- FFmpeg and FFprobe
- OpenCV (for advanced visual analysis)
- Required Python packages (see requirements.txt)

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/video-processing-service.git
cd video-processing-service

# Build and start the Docker containers
docker-compose up -d
```

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/video-processing-service.git
cd video-processing-service

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-subtitle.txt
pip install -r requirements-broll.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Subtitle Generation

```
POST /api/subtitles/generate - Generate subtitles
POST /api/subtitles/multiple-outputs - Generate multiple subtitle outputs
POST /api/subtitles/upload - Upload a video for subtitle processing
GET /api/subtitles/batch/{batch_id} - Get batch processing status
```

### B-Roll Insertion

```
POST /api/broll/upload - Upload a video for B-Roll processing
POST /api/broll/process - Process a video for B-Roll suggestions and insertion
GET /api/broll/jobs/{job_id} - Get job status
GET /api/broll/jobs/{job_id}/preview - Get preview video with B-Roll
POST /api/broll/analyze - Analyze a video for B-Roll opportunities
```

## API Usage Examples

### Generate Subtitles with B-Roll Suggestions

```python
import requests

# Upload a video file
with open('myvideo.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/broll/upload',
        files={'file': f},
        headers={'Authorization': 'Bearer YOUR_API_KEY'}
    )
video_path = response.json()['file_path']

# Process the video for B-Roll suggestions
transcript = {
    'segments': [
        {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'This is the first segment.'},
        {'id': 1, 'start': 5.1, 'end': 10.0, 'text': 'This is the second segment.'}
    ]
}

response = requests.post(
    'http://localhost:8000/api/broll/process',
    json={
        'transcript': transcript,
        'generate_preview': True,
        'max_suggestions': 3
    },
    params={'video_path': video_path},
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

job_id = response.json()['job_id']

# Check job status
response = requests.get(
    f'http://localhost:8000/api/broll/jobs/{job_id}',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
job_status = response.json()

# Get preview video when completed
if job_status['status'] == 'completed':
    response = requests.get(
        f'http://localhost:8000/api/broll/jobs/{job_id}/preview',
        headers={'Authorization': 'Bearer YOUR_API_KEY'}
    )
    with open('preview_with_broll.mp4', 'wb') as f:
        f.write(response.content)
```

## Example Scripts

The `examples` directory contains several example scripts:

- `subtitle_example.py` - Basic subtitle generation
- `multiple_output_example.py` - Generate multiple subtitle outputs
- `batch_processing_example.py` - Process multiple videos in batch
- `broll_example.py` - B-Roll suggestion and insertion

Run an example:

```bash
python examples/broll_example.py path/to/your/video.mp4
```

## Configuration

The service can be configured using environment variables or a `.env` file:

- `FFMPEG_PATH` - Path to FFmpeg binary
- `FFPROBE_PATH` - Path to FFprobe binary
- `TEMP_PATH` - Path for temporary files
- `OUTPUT_PATH` - Default path for output files
- `PEXELS_API_KEY` - API key for Pexels stock footage
- `PIXABAY_API_KEY` - API key for Pixabay stock footage
- `BROLL_LIBRARY_PATH` - Path to your personal B-Roll library

## Project Structure

```
video-processing-service/
├── app/                       # Main application code
│   ├── api/                   # API endpoints
│   ├── core/                  # Core configuration
│   ├── services/              # Service modules
│   │   ├── subtitles/         # Subtitle generation services
│   │   ├── broll/             # B-Roll insertion services
│   │   └── common/            # Shared utilities
│   └── main.py                # FastAPI application
├── examples/                  # Example scripts
├── tests/                     # Unit and integration tests
├── .env.example               # Example environment variables
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Core requirements
├── requirements-subtitle.txt  # Subtitle-specific requirements
└── requirements-broll.txt     # B-Roll-specific requirements
```

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Audio Enhancement

The service includes a comprehensive Audio Enhancement Suite to improve audio quality in videos, offering:

### Noise Reduction
- Advanced spectral subtraction algorithms for removing background noise
- Auto-detection of noise profile segments
- Customizable noise reduction strength

### Voice Enhancement
- Adaptive EQ for improved voice clarity
- De-essing to reduce sibilance
- Harmonic enhancement for better presence
- Separate optimization for male and female voices

### Dynamics Processing
- Multi-stage dynamics control with compression, limiting, expansion, and gating
- Broadcast-standard loudness normalization (-14 LUFS)
- Content-aware processing with automatic detection of content type
- Presets for different audio types:
  - `voice_broadcast`: For voiceovers and narration
  - `voice_intimate`: For podcasts and interviews
  - `music_master`: For music content
  - `dialog_leveler`: For movies and dialogue
  - `transparent`: For subtle processing

### Environmental Sound Classification
- AI-based detection of background sound types
- Optimized noise reduction strategies based on detected sounds
- Selective preservation of important environmental sounds

### Usage

You can access the audio enhancement features via the API:

```bash
# Process a video file with audio enhancement
curl -X POST "http://localhost:8000/api/v1/audio/enhance" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/video.mp4",
    "apply_noise_reduction": true,
    "apply_dynamics_processing": true,
    "dynamics_processing": {
      "preset": "voice_broadcast",
      "target_loudness": -14
    }
  }'
```

Or use the example scripts:

```bash
# Example 1: Basic audio enhancement
python examples/audio_enhancement_example.py input.mp4 output.mp4

# Example 2: Focused dynamics processing
python examples/dynamics_processor_demo.py input.mp4 output.mp4 --preset voice_broadcast --target-loudness -14
```

## Audio Enhancement Suite

The service includes a comprehensive Audio Enhancement Suite to improve audio quality in videos, offering:

### Noise Reduction
- Advanced spectral subtraction algorithms for removing background noise
- Auto-detection of noise profile segments
- Customizable noise reduction strength

### Voice Enhancement
- Adaptive EQ for improved voice clarity
- De-essing to reduce sibilance
- Harmonic enhancement for better presence
- Separate optimization for male and female voices

### Dynamics Processing
- Multi-stage dynamics control with compression, limiting, expansion, and gating
- Broadcast-standard loudness normalization (-14 LUFS)
- Content-aware processing with automatic detection of content type
- Presets for different audio types:
  - `voice_broadcast`: For voiceovers and narration
  - `voice_intimate`: For podcasts and interviews
  - `music_master`: For music content
  - `dialog_leveler`: For movies and dialogue
  - `transparent`: For subtle processing

### Environmental Sound Classification
- AI-based detection of background sound types
- Optimized noise reduction strategies based on detected sounds
- Selective preservation of important environmental sounds

### Audio Enhancement API

```
POST /api/v1/audio/enhance
```

Enhance audio quality in a video or audio file.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/enhance",
    json={
        "file_path": "/path/to/video.mp4",
        "apply_noise_reduction": True,
        "apply_voice_enhancement": True,
        "noise_reduction": {
            "strength": 0.7,
            "auto_detect": True
        },
        "voice_enhancement": {
            "clarity": 0.6,
            "warmth": 0.4
        }
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
```

## Music Genre Classification & Recommendation

The service includes music analysis capabilities for helping content creators select appropriate music tracks:

### Genre Classification
- Classifies audio tracks into 20+ musical genres
- Analyzes audio features to identify genre characteristics
- Provides probability scores for multiple matching genres
- Audio feature extraction including tempo, timbre, and dynamics

### Music Recommendation 
- Recommends music genres based on video content mood and genre
- Intelligent genre matching for different video content types
- Helps creators find appropriate music that enhances their content

### BPM Detection & Matching
- Accurately detects tempo (BPM) in audio tracks
- Categorizes tempos into ranges (very slow, slow, moderate, fast, very fast)
- Matches music tracks based on target BPM with adjustable tolerance
- Supports multiple matching styles:
  - Exact: Matches within specific BPM range
  - Double: Considers half/double tempo relationships
  - Harmonic: Matches harmonically related tempos (2/3, 3/4, etc.)
- Provides content-specific BPM recommendations for different video types

### Music Analysis API

```
POST /api/v1/audio/classify-genre
```

Classify the genre of an audio track.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/classify-genre",
    json={
        "top_n": 5  # Return top 5 genre matches
    },
    params={"file_path": "/path/to/music.mp3"},
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "file_path": "/path/to/music.mp3",
    "primary_genre": "electronic",
    "top_genres": [
        {"genre": "electronic", "probability": 0.72},
        {"genre": "dance", "probability": 0.58},
        {"genre": "pop", "probability": 0.23},
        {"genre": "indie", "probability": 0.18},
        {"genre": "ambient", "probability": 0.12}
    ],
    "audio_features": {
        "tempo": "Fast",
        "bpm": 128
    }
}
```

```
POST /api/v1/audio/recommend-genres
```

Get music genre recommendations for video content.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/recommend-genres",
    json={
        "video_mood": "energetic",
        "video_genre": "documentary",
        "top_n": 5  # Return top 5 recommendations
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "video_mood": "energetic",
    "video_genre": "documentary",
    "recommendations": [
        {"genre": "electronic", "score": 0.85},
        {"genre": "soundtrack", "score": 0.78},
        {"genre": "rock", "score": 0.65},
        {"genre": "classical", "score": 0.52},
        {"genre": "world", "score": 0.45}
    ]
}
```

```
POST /api/v1/audio/detect-bpm
```

Detect the tempo (BPM) of an audio track.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/detect-bpm",
    json={
        "file_path": "/path/to/music.mp3"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "file_path": "/path/to/music.mp3",
    "bpm": 128.5,
    "category": "fast",
    "confidence": 0.85,
    "range": [120, 150]
}
```

```
POST /api/v1/audio/match-bpm
```

Find tracks that match a target BPM.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/match-bpm",
    json={
        "target_bpm": 120.0,
        "file_paths": [
            "/path/to/music1.mp3",
            "/path/to/music2.mp3",
            "/path/to/music3.mp3"
        ],
        "tolerance": 5.0,
        "match_style": "harmonic"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "target_bpm": 120.0,
    "tolerance": 5.0,
    "match_style": "harmonic",
    "matches": [
        {
            "file_path": "/path/to/music2.mp3",
            "bpm": 121.3,
            "category": "fast",
            "match_score": 0.94,
            "bpm_difference": 1.3
        },
        {
            "file_path": "/path/to/music1.mp3",
            "bpm": 90.2,
            "category": "moderate",
            "match_score": 0.85,
            "bpm_difference": 2.3
        }
    ],
    "errors": []
}
```

```
POST /api/v1/audio/suggest-bpm
```

Get BPM recommendations based on content type.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/suggest-bpm",
    json={
        "content_type": "documentary"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "content_type": "documentary",
    "bpm_range": [80, 110],
    "suggested_bpm": 95.0,
    "category": "moderate"
}
```

## Example Scripts

The repository includes example scripts to demonstrate common use cases:

```bash
# Generate subtitles for a video
python examples/generate_subtitles.py input.mp4

# Process multiple videos in batch
python examples/batch_process.py input_directory output_directory

# Enhance audio in a video
python examples/audio_enhancement_example.py input.mp4 output.mp4

# Focused dynamics processing
python examples/dynamics_processor_demo.py input.mp4 output.mp4 --preset voice_broadcast --target-loudness -14
```

## Configuration

The service can be configured using environment variables or a `.env` file:

```
FFMPEG_PATH=/usr/bin/ffmpeg
FFPROBE_PATH=/usr/bin/ffprobe
TEMP_DIRECTORY=/tmp/video-processing
```

## Project Structure

```
video-processing-service/
├── app/
│   ├── api/
│   │   └── endpoints/
│   ├── core/
│   ├── services/
│   │   ├── subtitles/
│   │   ├── broll/
│   │   └── audio/
│   └── main.py
├── examples/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Audio Fingerprinting

The service includes Audio Fingerprinting capabilities to identify music:

### Audio Fingerprinting 
- Generates unique audio fingerprints for tracks to identify music
- Prevents copyright issues by detecting similar or matching audio
- Compares audio content for similarity regardless of format, quality, or modifications
- Builds a local fingerprint database for rapid identification
- Provides detailed acoustic feature analysis of audio content

```
POST /api/v1/audio/fingerprint/generate
```

Generate an audio fingerprint for a file.

**Request:**
```json
{
  "file_path": "/path/to/audio/file.mp3"
}
```

**Response:**
```json
{
  "status": "success",
  "fingerprint": {
    "hash": "a1b2c3d4e5f6...",
    "vector_summary": "[768 values summarized]"
  }
}
```

```
POST /api/v1/audio/fingerprint/identify
```

Identify an audio file by comparing its fingerprint to the database.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/fingerprint/identify",
    json={
        "file_path": "/path/to/audio.mp3",
        "top_n": 5,
        "min_similarity": 0.8
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "input_path": "/path/to/audio.mp3",
    "matches": [
        {
            "track_id": "tr_123456",
            "title": "Awesome Track",
            "artist": "Famous Artist",
            "similarity": 0.95,
            "distance": 0.15,
            "is_match": true
        }
    ],
    "match_found": true
}
```

```
POST /api/v1/audio/fingerprint/compare
```

Compare two audio files to determine their similarity.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/fingerprint/compare",
    json={
        "file_path1": "/path/to/audio1.mp3",
        "file_path2": "/path/to/audio2.mp3",
        "detailed_results": true
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "file_path1": "/path/to/audio1.mp3",
    "file_path2": "/path/to/audio2.mp3",
    "similarity": 0.83,
    "distance": 0.32,
    "is_match": false
}
```

```
POST /api/v1/audio/fingerprint/add-to-database
```

Add an audio fingerprint to the database for future identification.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/fingerprint/add-to-database",
    json={
        "file_path": "/path/to/audio.mp3",
        "track_id": "tr_123456",
        "title": "Awesome Track",
        "artist": "Famous Artist",
        "metadata": {
            "genre": "rock",
            "year": 2022,
            "album": "Great Album"
        }
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "track_id": "tr_123456",
    "message": "Fingerprint added to database"
}
```

Other fingerprint endpoints:
- `POST /api/v1/audio/fingerprint/remove-from-database` - Remove a fingerprint from the database
- `GET /api/v1/audio/fingerprint/database-info` - Get information about the fingerprint database

### Custom Music Library

```
POST /api/v1/audio/library/add-track
```

Add a track to the music library with detailed metadata.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/add-track",
    json={
        "file_path": "/path/to/music.mp3",
        "title": "Inspiring Journey",
        "artist": "Audio Composer",
        "mood": "inspiring",
        "genre": "cinematic",
        "bpm": 120.5,
        "tags": ["uplifting", "corporate", "background"],
        "description": "An inspiring track with modern cinematic elements",
        "copyright_free": True,
        "license": "Creative Commons Attribution 4.0"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "track_id": "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
    "message": "Track 'Inspiring Journey' added to library"
}
```

```
POST /api/v1/audio/library/search
```

Search for tracks based on various criteria with intelligent matching.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/search",
    json={
        "mood": "inspiring",
        "tempo": 120,
        "genre": "cinematic",
        "duration": 180,
        "keywords": ["uplifting", "corporate"],
        "max_results": 5,
        "copyright_free_only": True,
        "collection_id": "5c4a6b9d-21e3-4c88-9f45-8c3d4f7a8e5b"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "tracks": [
        {
            "id": "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
            "title": "Inspiring Journey",
            "artist": "Audio Composer",
            "mood": "inspiring",
            "genre": "cinematic",
            "bpm": 120.5,
            "duration": 185.3,
            "copyright_free": true,
            "relevance_score": 0.92
        },
        {...}
    ],
    "total_matches": 12
}
```

```
POST /api/v1/audio/library/collection/create
```

Create a music collection to organize tracks.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/collection/create",
    json={
        "name": "Corporate Videos",
        "description": "Uplifting tracks for corporate video content",
        "tags": ["corporate", "business", "professional"]
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "collection_id": "5c4a6b9d-21e3-4c88-9f45-8c3d4f7a8e5b",
    "message": "Collection 'Corporate Videos' created"
}
```

```
POST /api/v1/audio/library/collection/add-tracks
```

Add tracks to a collection.

Example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/library/collection/add-tracks",
    json={
        "collection_id": "5c4a6b9d-21e3-4c88-9f45-8c3d4f7a8e5b",
        "track_ids": [
            "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
            "fa3e7c2d-9b5a-4e81-8d67-0c12f3b9ae56"
        ]
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# Response example
{
    "status": "success",
    "message": "Added 2 tracks to collection 'Corporate Videos'",
    "added_tracks": [
        "d8f0a53c-15a9-4c7d-8560-a1e692f39b87",
        "fa3e7c2d-9b5a-4e81-8d67-0c12f3b9ae56"
    ]
}
```

Other library endpoints:
- `GET /api/v1/audio/library/tracks` - List all tracks in the library
- `GET /api/v1/audio/library/track/{track_id}` - Get details of a specific track
- `DELETE /api/v1/audio/library/track/{track_id}` - Remove a track from the library
- `PATCH /api/v1/audio/library/track` - Update track metadata
- `GET /api/v1/audio/library/collections` - List all collections
- `GET /api/v1/audio/library/collection/{collection_id}` - Get details of a specific collection with its tracks
- `PATCH /api/v1/audio/library/collection` - Update collection metadata
- `DELETE /api/v1/audio/library/collection/{collection_id}` - Delete a collection
- `POST /api/v1/audio/library/collection/remove-tracks` - Remove tracks from a collection

### Music Recommendation Engine

The Music Recommendation Engine provides intelligent music suggestions for video content by combining multiple recommendation approaches including content-based filtering, collaborative filtering, and emotional arc matching. It learns from user feedback to continuously improve recommendations over time.

**Key Features:**
- Comprehensive video content analysis for music matching
- Personalized recommendations based on user preferences and feedback
- Dynamic soundtrack generation following the emotional arc of videos
- Cross-source recommendations from local library and external services
- Diversity-aware track selection to avoid repetitive recommendations
- Similarity-based recommendation for finding tracks like ones you enjoy

#### API Endpoints

**Recommend for video:**
```
POST /api/v1/audio/recommend/for-video
```
Example request:
```json
{
  "file_path": "/path/to/video.mp4",
  "user_id": "user_123",
  "use_emotional_arc": true,
  "max_results": 10,
  "copyright_free_only": true,
  "include_external_services": true,
  "diversity_level": 0.3
}
```
Example response:
```json
{
  "status": "success",
  "recommendations": [
    {
      "track_id": "track_456",
      "title": "Summer Breeze",
      "artist": "Ambient Vibes",
      "mood": "relaxed",
      "genre": "ambient",
      "final_score": 0.85,
      "recommendation_source": "content",
      "file_path": "/path/to/library/summer_breeze.mp3"
    },
    // Additional tracks...
  ],
  "mood_analysis": {
    "mood": "relaxed",
    "valence": 0.7,
    "arousal": 0.3
  },
  "emotional_timeline": [
    {
      "start_time": 0.0,
      "end_time": 45.2,
      "duration": 45.2,
      "mood": "peaceful",
      "cue_type": "establishing"
    },
    // Additional segments...
  ]
}
```

**Recommend similar tracks:**
```
POST /api/v1/audio/recommend/similar-to-track
```
Example request:
```json
{
  "track_id": "track_456",
  "source": "library",
  "user_id": "user_123",
  "max_results": 5,
  "include_external_services": true
}
```
Example response:
```json
{
  "status": "success",
  "reference_track": {
    "title": "Summer Breeze",
    "artist": "Ambient Vibes",
    "mood": "relaxed",
    "genre": "ambient"
  },
  "recommendations": [
    {
      "track_id": "track_789",
      "title": "Ocean Waves",
      "artist": "Chill Sessions",
      "mood": "relaxed",
      "genre": "ambient",
      "final_score": 0.92,
      "file_path": "/path/to/library/ocean_waves.mp3"
    },
    // Additional tracks...
  ]
}
```

**Submit feedback:**
```
POST /api/v1/audio/recommend/submit-feedback
```
Example request:
```json
{
  "user_id": "user_123",
  "track_id": "track_456",
  "source": "library",
  "rating": 5,
  "liked": true,
  "used_in_project": true,
  "context": {
    "project_id": "project_789",
    "video_id": "video_012"
  }
}
```
Example response:
```json
{
  "status": "success",
  "user_id": "user_123",
  "track_id": "track_456",
  "preferences_updated": true
}
```

**Get user preferences:**
```
POST /api/v1/audio/recommend/user-preferences
```
Example request:
```json
{
  "user_id": "user_123"
}
```
Example response:
```json
{
  "status": "success",
  "user_id": "user_123",
  "preferences": {
    "moods": {
      "relaxed": 0.9,
      "upbeat": 0.7,
      "inspiring": 0.8,
      "melancholic": 0.3
    },
    "genres": {
      "ambient": 0.9,
      "electronic": 0.6,
      "classical": 0.8
    },
    "artists": {
      "Ambient Vibes": 0.9,
      "Chill Sessions": 0.8
    },
    "favorite_tracks": [
      {
        "track_id": "track_456",
        "source": "library"
      },
      {
        "track_id": "track_789",
        "source": "library"
      }
    ]
  }
}
```

#### Example Usage

The following example demonstrates how to recommend music tracks for a video:

```python
from app.services.music.music_recommender import MusicRecommender

# Initialize the recommender
recommender = MusicRecommender()

# Get recommendations for a video
results = recommender.recommend_for_video(
    video_path="/path/to/video.mp4",
    user_id="user_123",
    max_results=5,
    copyright_free_only=True,
    use_emotional_arc=True
)

# Print recommendations
for track in results["recommendations"]:
    print(f"Track: {track['title']} by {track['artist']}")
    print(f"Mood: {track['mood']}, Genre: {track['genre']}")
    print(f"Score: {track['final_score']}")
    print()

# Get the emotional timeline
if "emotional_timeline" in results:
    for segment in results["emotional_timeline"]:
        print(f"Segment: {segment['start_time']}s - {segment['end_time']}s")
        print(f"Mood: {segment['mood']}, Cue Type: {segment['cue_type']}")
        print()
```

For a complete example, see the `music_recommendation_example.py` script in the `examples` directory.

## Sound Effects Library

The Sound Effects Library provides a comprehensive suite of tools for adding and managing professional sound effects in your video content:

### Context-Aware Sound Effect Recommendation

The system uses advanced contextual analysis to recommend the most appropriate sound effects for your content:

- **Transcript Analysis**: Processes dialogue and narration to identify sound effect opportunities
- **Scene Understanding**: Analyzes scene descriptions to match with appropriate sound effects
- **Mood Matching**: Recommends sound effects that match the emotional tone of your scene
- **Intensity Scaling**: Adapts sound effect selection based on scene dynamics
- **Trigger Word Detection**: Identifies specific words or phrases that commonly indicate a need for certain sound effects
- **Repetition Avoidance**: Intelligently avoids recommending recently used effects to prevent monotony

This recommendation engine combines multiple scoring mechanisms:
1. Trigger word matching with semantic analysis
2. Full-context text search with relevance scoring
3. Category and mood matching
4. Intensity appropriateness evaluation

#### Sound Effects Library API

```json
POST /api/v1/sound-effects/recommend
```

Provides context-aware sound effect recommendations based on video content analysis.

Request:
```json
{
  "transcript": "The car speeds down the highway as sirens wail in the distance.",
  "scene_descriptions": ["High-speed car chase", "Police pursuit"],
  "video_category": "action",
  "mood": "tense",
  "keywords": ["car", "chase", "police", "speed"],
  "intensity": 0.8,
  "max_results": 5
}
```

Response:
```json
{
  "status": "success",
  "recommendations": [
    {
      "effect_id": "sfx-12345",
      "name": "Police Siren Wail",
      "category": "vehicles",
      "tags": ["police", "siren", "emergency", "alarm"],
      "description": "Police car siren with doppler effect",
      "relevance_score": 0.92
    },
    {
      "effect_id": "sfx-23456",
      "name": "Car Engine Rev High",
      "category": "vehicles",
      "tags": ["car", "engine", "acceleration", "speed"],
      "description": "Sports car engine revving at high RPM",
      "relevance_score": 0.87
    },
    // Additional sound effects...
  ],
  "total_count": 5
}
```

```json
POST /api/v1/sound-effects/library/search
```

Search for sound effects by category, tags, or text search.

**Request:**
```json
{
  "category": "weather",
  "tags": ["storm"],
  "search_term": "thunder",
  "limit": 10,
  "offset": 0
}
```

**Response:**
```json
{
  "status": "success",
  "total_count": 5,
  "sound_effects": [
    {
      "effect_id": "sf12345",
      "name": "Thunder Crack",
      "category": "weather",
      "tags": ["thunder", "storm", "loud", "dramatic"],
      "description": "Powerful thunder crack for storm scenes",
      "duration": 3.5,
      "file_format": "wav",
      "sample_rate": 48000,
      "channels": 2
    },
    // ... other sound effects
  ]
}
```

### Avatar Generation API

#### 3D Face Reconstruction

The API provides endpoints for generating high-fidelity 3D face models from images and videos.

##### Generate 3D Face from Image

```
POST /api/v1/avatars/face/reconstruct-from-image
```

Generates a 3D face model from a single image.

**Request:**
- `image` (file): Image file containing a face
- `detail_level` (string, optional): Detail level for face reconstruction (low, medium, high, ultra). Default: "high"
  * `low`: Basic skin texture with minimal pore simulation
  * `medium`: Enhanced skin texture with pores and basic wrinkle simulation
  * `high`: Detailed skin texture with varied pore sizes, wrinkles, and texture variations
  * `ultra`: Ultra-realistic skin with multi-layered details, fine wrinkles, and subsurface scattering
- `enable_texture_mapping` (boolean, optional): Enable high-fidelity 4K texture mapping. Default: true
- `enable_detail_refinement` (boolean, optional): Enable detailed feature preservation algorithm. Default: true
- `enable_identity_verification` (boolean, optional): Enable identity consistency verification. Default: true
- `enable_stylegan_enhancements` (boolean, optional): Enable StyleGAN-3 enhancements. Default: true
- `enable_expression_calibration` (boolean, optional): Enable expression range calibration. Default: true

**Response:**
```json
{
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_url": "https://storage.example.com/avatars/550e8400-e29b-41d4-a716-446655440000/model.glb",
  "texture_url": "https://storage.example.com/avatars/550e8400-e29b-41d4-a716-446655440000/texture.png",
  "quality_score": 0.92,
  "processing_time": 8.45,
  "identity_verification_score": 0.91,
  "landmarks": {
    "nose_tip": [0.5, 0.5, 0.2],
    "left_eye_center": [0.3, 0.4, 0.1],
    "right_eye_center": [0.7, 0.4, 0.1],
    "mouth_center": [0.5, 0.7, 0.1]
  },
  "metadata": {
    "face_confidence": 0.99,
    "source_resolution": [1280, 720]
  }
}
```

##### Generate 3D Face from Video

```
POST /api/v1/avatars/face/reconstruct-from-video
```

Generates a high-quality 3D face model from a video, utilizing multiple frames for improved accuracy.

**Request:**
- `video` (file): Video file containing a face
- `detail_level` (string, optional): Detail level for face reconstruction (low, medium, high, ultra). Default: "high"
- `enable_texture_mapping` (boolean, optional): Enable high-fidelity 4K texture mapping. Default: true
- `enable_detail_refinement` (boolean, optional): Enable detailed feature preservation algorithm. Default: true
- `enable_identity_verification` (boolean, optional): Enable identity consistency verification. Default: true
- `enable_stylegan_enhancements` (boolean, optional): Enable StyleGAN-3 enhancements. Default: true
- `enable_expression_calibration` (boolean, optional): Enable expression range calibration. Default: true

**Response:**
Same format as the response for the image-based reconstruction.

## Example Scripts

The repository includes example scripts to demonstrate various features:

### Avatar Generation Examples

#### Face Modeling Example

The `face_modeling_example.py` script demonstrates how to use the FaceModeling component to create high-fidelity 3D face models from images and videos.

**Example usage:**

```bash
# Generate from image
python face_modeling_example.py --image path/to/face/image.jpg

# Generate from video with ultra detail level
python face_modeling_example.py --video path/to/face/video.mp4 --detail ultra
```

#### Face Detail Refinement Example

The `face_detail_refinement_example.py` script demonstrates the high-resolution detail refinement capabilities, focusing on skin features like pores, wrinkles, and texture variations.

**Example usage:**

```bash
# Generate with high detail level (default)
python face_detail_refinement_example.py --image path/to/face/image.jpg

# Generate with ultra detail level for maximum detail
python face_detail_refinement_example.py --image path/to/face/image.jpg --detail-level ultra

# Compare all detail levels (low, medium, high, ultra) with visualization
python face_detail_refinement_example.py --image path/to/face/image.jpg --compare-all
```
