import os
import json
import re
from flask import current_app
import openai
from collections import Counter

def extract_sound_triggers(text):
    """
    Extract potential sound effect triggers from text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        list: Potential sound effect triggers
    """
    # Common sound effect triggers
    sound_triggers = [
        # Actions and movements
        "walk", "run", "jump", "fall", "crash", "bang", "slam", "knock", "tap",
        "slide", "whoosh", "swing", "hit", "punch", "kick", "stomp", "clap",
        "snap", "click", "pop", "explosion", "blast", "boom", "crackle", "crunch",
        
        # Nature and environment
        "rain", "thunder", "lightning", "wind", "storm", "ocean", "waves", "fire",
        "water", "splash", "drip", "pour", "stream", "river", "forest", "birds",
        "animals", "insects", "leaves", "trees", "grass", "beach", "snow", "ice",
        
        # Objects and machines
        "car", "engine", "motor", "vehicle", "train", "airplane", "helicopter",
        "boat", "ship", "door", "window", "phone", "computer", "keyboard", "typing",
        "bell", "alarm", "siren", "horn", "whistle", "drum", "piano", "guitar",
        
        # Human sounds
        "laugh", "cry", "scream", "shout", "whisper", "sigh", "breath", "cough",
        "sneeze", "snore", "eat", "drink", "sip", "chew", "swallow", "gulp",
        
        # Abstract concepts that often have associated sounds
        "success", "failure", "victory", "defeat", "celebration", "party",
        "tension", "suspense", "reveal", "surprise", "shock", "horror"
    ]
    
    # Find triggers in the text
    text = text.lower()
    found_triggers = []
    
    for trigger in sound_triggers:
        # Look for the trigger word with word boundaries
        matches = re.findall(r'\b' + trigger + r'\b', text)
        if matches:
            found_triggers.extend(matches)
    
    # Remove duplicates
    found_triggers = list(set(found_triggers))
    
    return found_triggers

def analyze_sounds_with_ai(transcription):
    """
    Analyze transcript with AI to identify potential sound effect placements.
    
    Args:
        transcription (dict): Transcription data
        
    Returns:
        dict: Analysis results with sound effect suggestions
    """
    # Set OpenAI API key
    api_key = current_app.config['OPENAI_API_KEY']
    if not api_key:
        raise ValueError("OpenAI API key is not set in the configuration")
    
    openai.api_key = api_key
    
    # Extract the text from transcription
    text = transcription.get("transcription", "")
    if not text:
        return {
            "sound_effect_suggestions": []
        }
    
    # Prepare the prompt for OpenAI
    prompt = f"""
    Analyze the following transcript and suggest sound effects that would enhance the video:
    
    Transcript:
    {text}
    
    Please identify:
    1. Key moments where sound effects would enhance the content
    2. Type of sound effect that would be appropriate
    3. Approximate timestamp or position in the content
    4. Intensity or volume level recommendation
    
    Format your response as a JSON object with the following structure:
    {{
        "sound_effect_suggestions": [
            {{
                "description": "brief description of the sound effect",
                "type": "category of sound effect",
                "position": "approximate position or timestamp",
                "intensity": "low|medium|high",
                "trigger_phrase": "phrase or word that triggered this suggestion"
            }},
            ...
        ]
    }}
    
    Common sound effect categories include:
    - ambient (background sounds like room tone, nature, etc.)
    - impact (crashes, bangs, hits, etc.)
    - transition (whooshes, swipes, etc.)
    - ui (clicks, beeps, notifications, etc.)
    - human (laughs, gasps, applause, etc.)
    - machine (car engines, computers, devices, etc.)
    - nature (rain, thunder, animals, etc.)
    - musical (stingers, accents, etc.)
    """
    
    # Call OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    # Parse the response
    try:
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing AI response: {str(e)}")
        # Fallback to basic sound trigger detection
        triggers = extract_sound_triggers(text)
        return {
            "sound_effect_suggestions": [
                {
                    "description": f"{trigger} sound",
                    "type": "general",
                    "position": "unknown",
                    "intensity": "medium",
                    "trigger_phrase": trigger
                } for trigger in triggers[:10]  # Limit to 10 suggestions
            ]
        }

def map_sound_effects_to_timestamps(sound_suggestions, transcription):
    """
    Map sound effect suggestions to timestamps in the video.
    
    Args:
        sound_suggestions (list): Sound effect suggestions
        transcription (dict): Transcription data with segments
        
    Returns:
        list: Sound effect suggestions with timestamps
    """
    # Extract segments with timestamps
    segments = transcription.get("segments", [])
    if not segments:
        return sound_suggestions
    
    # Create a word to timestamp mapping
    word_timestamps = {}
    for segment in segments:
        text = segment.get("text", "").lower()
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        # Split into words
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if word not in word_timestamps:
                word_timestamps[word] = []
            word_timestamps[word].append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
    
    # Map suggestions to timestamps
    for suggestion in sound_suggestions:
        trigger = suggestion.get("trigger_phrase", "").lower()
        if trigger and trigger in word_timestamps:
            # Find the timestamps for this trigger
            timestamps = word_timestamps[trigger]
            if timestamps:
                # Use the first occurrence by default
                suggestion["timestamp"] = timestamps[0]["start"]
                suggestion["segment_text"] = timestamps[0]["text"]
    
    return sound_suggestions

def get_sound_effect_library():
    """
    Get the available sound effect library.
    
    Returns:
        dict: Sound effect library by category
    """
    # This would typically load from a database or file
    # For now, we'll return a static library
    return {
        "ambient": [
            {"name": "Room Tone", "duration": 10.0, "tags": ["room", "indoor", "quiet"]},
            {"name": "Office Ambience", "duration": 15.0, "tags": ["office", "work", "typing", "phones"]},
            {"name": "Nature Ambience", "duration": 20.0, "tags": ["nature", "forest", "birds", "wind"]},
            {"name": "City Traffic", "duration": 12.0, "tags": ["city", "traffic", "cars", "urban"]},
            {"name": "Cafe Background", "duration": 18.0, "tags": ["cafe", "restaurant", "people", "talking"]}
        ],
        "impact": [
            {"name": "Door Slam", "duration": 1.0, "tags": ["door", "slam", "impact", "house"]},
            {"name": "Glass Break", "duration": 2.0, "tags": ["glass", "break", "shatter", "crash"]},
            {"name": "Punch Impact", "duration": 0.5, "tags": ["punch", "hit", "impact", "fight"]},
            {"name": "Metal Crash", "duration": 1.5, "tags": ["metal", "crash", "impact", "industrial"]},
            {"name": "Wood Break", "duration": 1.2, "tags": ["wood", "break", "snap", "impact"]}
        ],
        "transition": [
            {"name": "Fast Whoosh", "duration": 0.8, "tags": ["whoosh", "transition", "fast", "swipe"]},
            {"name": "Slide Transition", "duration": 1.0, "tags": ["slide", "transition", "smooth", "swipe"]},
            {"name": "Pop Transition", "duration": 0.3, "tags": ["pop", "transition", "quick", "bubble"]},
            {"name": "Warp Effect", "duration": 1.5, "tags": ["warp", "transition", "sci-fi", "distortion"]},
            {"name": "Page Turn", "duration": 0.7, "tags": ["page", "turn", "transition", "paper"]}
        ],
        "ui": [
            {"name": "Button Click", "duration": 0.2, "tags": ["click", "button", "ui", "interface"]},
            {"name": "Notification", "duration": 0.5, "tags": ["notification", "alert", "ui", "app"]},
            {"name": "Message Sent", "duration": 0.4, "tags": ["message", "sent", "ui", "communication"]},
            {"name": "Error Alert", "duration": 0.6, "tags": ["error", "alert", "ui", "warning"]},
            {"name": "Typing", "duration": 1.0, "tags": ["typing", "keyboard", "ui", "computer"]}
        ],
        "human": [
            {"name": "Applause", "duration": 3.0, "tags": ["applause", "clapping", "audience", "approval"]},
            {"name": "Laughter", "duration": 2.5, "tags": ["laugh", "laughter", "funny", "humor"]},
            {"name": "Gasp", "duration": 0.7, "tags": ["gasp", "surprise", "shock", "reaction"]},
            {"name": "Footsteps", "duration": 2.0, "tags": ["footsteps", "walking", "movement", "steps"]},
            {"name": "Whisper", "duration": 1.5, "tags": ["whisper", "quiet", "secret", "voice"]}
        ],
        "machine": [
            {"name": "Car Engine Start", "duration": 2.0, "tags": ["car", "engine", "start", "vehicle"]},
            {"name": "Computer Startup", "duration": 1.5, "tags": ["computer", "startup", "boot", "tech"]},
            {"name": "Phone Ring", "duration": 2.0, "tags": ["phone", "ring", "call", "communication"]},
            {"name": "Camera Shutter", "duration": 0.3, "tags": ["camera", "shutter", "photo", "click"]},
            {"name": "Printer Working", "duration": 3.0, "tags": ["printer", "printing", "office", "machine"]}
        ],
        "nature": [
            {"name": "Thunder Crack", "duration": 2.0, "tags": ["thunder", "storm", "weather", "rain"]},
            {"name": "Water Stream", "duration": 3.0, "tags": ["water", "stream", "river", "nature"]},
            {"name": "Wind Gust", "duration": 2.5, "tags": ["wind", "gust", "weather", "storm"]},
            {"name": "Bird Chirp", "duration": 1.0, "tags": ["bird", "chirp", "nature", "animal"]},
            {"name": "Fire Crackling", "duration": 3.0, "tags": ["fire", "crackle", "flame", "camp"]}
        ],
        "musical": [
            {"name": "Success Stinger", "duration": 1.0, "tags": ["success", "win", "achievement", "musical"]},
            {"name": "Fail Stinger", "duration": 0.8, "tags": ["fail", "failure", "loss", "musical"]},
            {"name": "Suspense Rise", "duration": 2.0, "tags": ["suspense", "tension", "rise", "musical"]},
            {"name": "Comedy Boing", "duration": 0.5, "tags": ["comedy", "funny", "boing", "musical"]},
            {"name": "Mystery Chord", "duration": 1.2, "tags": ["mystery", "chord", "intrigue", "musical"]}
        ]
    }

def find_matching_sound_effects(suggestion, library):
    """
    Find matching sound effects in the library.
    
    Args:
        suggestion (dict): Sound effect suggestion
        library (dict): Sound effect library
        
    Returns:
        list: Matching sound effects
    """
    suggestion_type = suggestion.get("type", "").lower()
    trigger_phrase = suggestion.get("trigger_phrase", "").lower()
    description = suggestion.get("description", "").lower()
    
    # Collect keywords from description and trigger phrase
    keywords = set()
    if trigger_phrase:
        keywords.add(trigger_phrase)
    if description:
        # Extract words from description
        words = re.findall(r'\b\w+\b', description)
        keywords.update(words)
    
    # Find matching sound effects
    matches = []
    
    # Check the specific category first
    if suggestion_type in library:
        for sound in library[suggestion_type]:
            score = 0
            for tag in sound["tags"]:
                if tag in keywords:
                    score += 1
            if score > 0:
                matches.append({
                    "sound": sound,
                    "score": score,
                    "category": suggestion_type
                })
    
    # If no good matches, check all categories
    if not matches:
        for category, sounds in library.items():
            for sound in sounds:
                score = 0
                for tag in sound["tags"]:
                    if tag in keywords:
                        score += 1
                if score > 0:
                    matches.append({
                        "sound": sound,
                        "score": score,
                        "category": category
                    })
    
    # Sort by score
    matches.sort(key=lambda x: x["score"], reverse=True)
    
    # Take the top matches
    return matches[:3]

def suggest_sound_effects(transcription):
    """
    Suggest sound effects based on video content.
    
    Args:
        transcription (dict): Transcription data
        
    Returns:
        dict: Sound effect suggestions
    """
    try:
        # Analyze the transcript for sound effect suggestions
        analysis = analyze_sounds_with_ai(transcription)
        
        # Get the sound effect suggestions
        suggestions = analysis.get("sound_effect_suggestions", [])
        
        # Map suggestions to timestamps
        suggestions = map_sound_effects_to_timestamps(suggestions, transcription)
        
        # Get the sound effect library
        library = get_sound_effect_library()
        
        # Find matching sound effects for each suggestion
        for suggestion in suggestions:
            matches = find_matching_sound_effects(suggestion, library)
            suggestion["matches"] = matches
        
        # Generate the final suggestions
        result = {
            "sound_effect_suggestions": suggestions,
            "library_categories": list(library.keys()),
            "total_suggestions": len(suggestions)
        }
        
        return result
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Sound effect suggestion error: {str(e)}")
        raise 