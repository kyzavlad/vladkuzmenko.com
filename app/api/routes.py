from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import time
from werkzeug.utils import secure_filename

from app.core.transcription import transcribe_video
from app.core.pause_detection import detect_and_remove_pauses
from app.core.subtitle_generator import generate_subtitles
from app.core.video_enhancement import enhance_video
from app.core.b_roll_suggestions import suggest_b_roll
from app.core.music_recommendation import recommend_music
from app.core.sound_effects import suggest_sound_effects

# Create blueprint
api_bp = Blueprint('api', __name__)

# Helper functions
def allowed_file(filename):
    """Check if the file extension is allowed."""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_path(filename):
    """Get the full path for a file in the upload folder."""
    return os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

# API Routes
@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'success',
        'message': 'API is running'
    })

@api_bp.route('/upload', methods=['POST'])
def upload_video():
    """Upload a video file."""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = get_file_path(filename)
        
        file.save(file_path)
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'data': {
                'filename': filename,
                'original_filename': original_filename,
                'file_path': file_path
            }
        })
    
    return jsonify({
        'status': 'error',
        'message': 'File type not allowed'
    }), 400

@api_bp.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe a video using Whisper API."""
    data = request.json
    
    if not data or 'filename' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No filename provided'
        }), 400
    
    filename = data['filename']
    file_path = get_file_path(filename)
    
    if not os.path.exists(file_path):
        return jsonify({
            'status': 'error',
            'message': 'File not found'
        }), 404
    
    try:
        # Process the transcription
        transcription_result = transcribe_video(file_path)
        
        return jsonify({
            'status': 'success',
            'data': transcription_result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Transcription failed: {str(e)}'
        }), 500

@api_bp.route('/detect-pauses', methods=['POST'])
def detect_pauses():
    """Detect and remove pauses in a video."""
    data = request.json
    
    if not data or 'filename' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No filename provided'
        }), 400
    
    filename = data['filename']
    file_path = get_file_path(filename)
    
    # Get the threshold from request or use default
    min_pause = data.get('min_pause', 0.5)
    max_pause = data.get('max_pause', 2.0)
    
    if not os.path.exists(file_path):
        return jsonify({
            'status': 'error',
            'message': 'File not found'
        }), 404
    
    try:
        # Process pause detection
        output_filename = f"nopause_{filename}"
        output_path = get_file_path(output_filename)
        
        result = detect_and_remove_pauses(file_path, output_path, min_pause, max_pause)
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_filename': filename,
                'processed_filename': output_filename,
                'pauses_removed': result['pauses_removed'],
                'time_saved': result['time_saved']
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Pause detection failed: {str(e)}'
        }), 500

@api_bp.route('/generate-subtitles', methods=['POST'])
def subtitles():
    """Generate subtitles for a video."""
    data = request.json
    
    if not data or 'filename' not in data or 'transcription' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameters'
        }), 400
    
    filename = data['filename']
    transcription = data['transcription']
    file_path = get_file_path(filename)
    
    # Get subtitle styling options
    style_options = data.get('style_options', {})
    
    if not os.path.exists(file_path):
        return jsonify({
            'status': 'error',
            'message': 'File not found'
        }), 404
    
    try:
        # Generate subtitles
        output_filename = f"subtitled_{filename}"
        output_path = get_file_path(output_filename)
        
        subtitle_result = generate_subtitles(file_path, output_path, transcription, style_options)
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_filename': filename,
                'processed_filename': output_filename,
                'subtitle_file': subtitle_result['subtitle_file']
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Subtitle generation failed: {str(e)}'
        }), 500

@api_bp.route('/enhance-video', methods=['POST'])
def enhance():
    """Enhance video quality and audio clarity."""
    data = request.json
    
    if not data or 'filename' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No filename provided'
        }), 400
    
    filename = data['filename']
    file_path = get_file_path(filename)
    
    # Get enhancement options
    enhancement_options = data.get('enhancement_options', {})
    
    if not os.path.exists(file_path):
        return jsonify({
            'status': 'error',
            'message': 'File not found'
        }), 404
    
    try:
        # Enhance the video
        output_filename = f"enhanced_{filename}"
        output_path = get_file_path(output_filename)
        
        enhancement_result = enhance_video(file_path, output_path, enhancement_options)
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_filename': filename,
                'processed_filename': output_filename,
                'enhancements_applied': enhancement_result['enhancements_applied']
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Video enhancement failed: {str(e)}'
        }), 500

@api_bp.route('/suggest-b-roll', methods=['POST'])
def b_roll():
    """Suggest B-roll footage based on content analysis."""
    data = request.json
    
    if not data or 'transcription' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No transcription provided'
        }), 400
    
    transcription = data['transcription']
    
    try:
        suggestions = suggest_b_roll(transcription)
        
        return jsonify({
            'status': 'success',
            'data': {
                'suggestions': suggestions
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'B-roll suggestion failed: {str(e)}'
        }), 500

@api_bp.route('/recommend-music', methods=['POST'])
def music():
    """Recommend background music based on video content and mood."""
    data = request.json
    
    if not data or 'transcription' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No transcription provided'
        }), 400
    
    transcription = data['transcription']
    mood = data.get('mood', None)
    
    try:
        recommendations = recommend_music(transcription, mood)
        
        return jsonify({
            'status': 'success',
            'data': {
                'recommendations': recommendations
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Music recommendation failed: {str(e)}'
        }), 500

@api_bp.route('/suggest-sound-effects', methods=['POST'])
def sound_effects():
    """Suggest sound effects based on video content."""
    data = request.json
    
    if not data or 'transcription' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No transcription provided'
        }), 400
    
    transcription = data['transcription']
    
    try:
        effects = suggest_sound_effects(transcription)
        
        return jsonify({
            'status': 'success',
            'data': {
                'sound_effects': effects
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Sound effect suggestion failed: {str(e)}'
        }), 500 