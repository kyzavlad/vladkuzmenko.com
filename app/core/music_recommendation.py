import os
import json
import re
from flask import current_app
import openai
from collections import Counter

def analyze_mood_from_text(text):
    """
    Analyze the mood of the text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Detected mood and confidence
    """
    # Common mood keywords
    mood_keywords = {
        "happy": ["happy", "joy", "celebration", "exciting", "fun", "cheerful", "upbeat", "positive"],
        "sad": ["sad", "sorrow", "grief", "depressing", "melancholy", "gloomy", "tearful"],
        "inspirational": ["inspire", "motivation", "success", "achievement", "determination", "courage"],
        "dramatic": ["dramatic", "intense", "suspense", "tension", "serious", "emotional"],
        "calm": ["calm", "peaceful", "relaxing", "gentle", "soothing", "tranquil", "quiet"],
        "energetic": ["energy", "dynamic", "powerful", "strong", "fast", "action", "quick"],
        "romantic": ["love", "romance", "affection", "tender", "intimate", "passion"],
        "mysterious": ["mystery", "curious", "unknown", "enigmatic", "suspenseful", "intriguing"],
        "epic": ["epic", "grand", "majestic", "powerful", "heroic", "vast", "cinematic"],
        "funny": ["funny", "humorous", "comedy", "laughing", "joke", "amusing", "hilarious"]
    }
    
    # Count occurrences of mood keywords
    text = text.lower()
    mood_counts = {mood: 0 for mood in mood_keywords}
    
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', text))
            mood_counts[mood] += count
    
    # Find the most common mood
    if all(count == 0 for count in mood_counts.values()):
        # No clear mood detected
        return {
            "primary_mood": "neutral",
            "secondary_mood": None,
            "confidence": 0.5,
            "mood_scores": {"neutral": 1.0}
        }
    
    # Calculate total and normalize scores
    total_mentions = sum(mood_counts.values())
    mood_scores = {mood: count / total_mentions for mood, count in mood_counts.items() if count > 0}
    
    # Sort moods by score
    sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
    primary_mood = sorted_moods[0][0]
    primary_score = sorted_moods[0][1]
    
    # Secondary mood
    secondary_mood = sorted_moods[1][0] if len(sorted_moods) > 1 else None
    
    return {
        "primary_mood": primary_mood,
        "secondary_mood": secondary_mood,
        "confidence": primary_score,
        "mood_scores": mood_scores
    }

def analyze_mood_with_ai(transcription):
    """
    Analyze mood with AI.
    
    Args:
        transcription (dict): Transcription data
        
    Returns:
        dict: Analysis results with mood information
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
            "primary_mood": "neutral",
            "secondary_mood": None,
            "confidence": 0.5
        }
    
    # Prepare the prompt for OpenAI
    prompt = f"""
    Analyze the following transcript and determine the overall mood of the content:
    
    Transcript:
    {text}
    
    Please identify:
    1. The primary mood or emotion of the content
    2. Any secondary moods or emotions
    3. Key phrases that indicate the mood
    4. The intensity or confidence level in this mood assessment (0.0 to 1.0)
    
    Format your response as a JSON object with the following structure:
    {{
        "primary_mood": "mood_name",
        "secondary_mood": "mood_name",
        "mood_phrases": ["phrase1", "phrase2", ...],
        "confidence": 0.0 to 1.0,
        "intensity": "low|medium|high",
        "mood_changes": [
            {{
                "approximate_position": "start|middle|end or timestamp",
                "mood": "mood_name",
                "description": "brief description of the change"
            }},
            ...
        ]
    }}
    
    Possible mood categories include (but are not limited to):
    happy, sad, inspirational, dramatic, calm, energetic, romantic, mysterious, epic, funny, tense, educational, informative, nostalgic
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
        # Fallback to basic mood detection
        return analyze_mood_from_text(text)

def get_music_recommendations(mood, tempo=None, genre=None):
    """
    Get music recommendations based on mood.
    
    Args:
        mood (str): Mood to base recommendations on
        tempo (str, optional): Tempo preference (slow, medium, fast)
        genre (str, optional): Genre preference
        
    Returns:
        list: Music recommendations
    """
    # This would typically call a music API to get actual recommendations
    # For now, we'll return a static list based on the mood
    
    # Sample music tracks by mood
    mood_tracks = {
        "happy": [
            {"title": "Happy Day", "artist": "Various Artists", "tempo": "medium", "genres": ["pop", "electronic"]},
            {"title": "Sunny Side Up", "artist": "Various Artists", "tempo": "fast", "genres": ["pop", "rock"]},
            {"title": "Good Times", "artist": "Various Artists", "tempo": "medium", "genres": ["funk", "disco"]}
        ],
        "sad": [
            {"title": "Melancholy Sunset", "artist": "Various Artists", "tempo": "slow", "genres": ["instrumental", "piano"]},
            {"title": "Rainy Days", "artist": "Various Artists", "tempo": "slow", "genres": ["ambient", "piano"]},
            {"title": "Blue Memories", "artist": "Various Artists", "tempo": "medium", "genres": ["jazz", "blues"]}
        ],
        "inspirational": [
            {"title": "Rise Up", "artist": "Various Artists", "tempo": "medium", "genres": ["orchestral", "electronic"]},
            {"title": "Triumph", "artist": "Various Artists", "tempo": "medium", "genres": ["orchestral", "cinematic"]},
            {"title": "New Horizons", "artist": "Various Artists", "tempo": "medium", "genres": ["electronic", "ambient"]}
        ],
        "dramatic": [
            {"title": "Tension Build", "artist": "Various Artists", "tempo": "medium", "genres": ["orchestral", "cinematic"]},
            {"title": "Epic Moment", "artist": "Various Artists", "tempo": "fast", "genres": ["orchestral", "trailer"]},
            {"title": "Dramatic Impact", "artist": "Various Artists", "tempo": "slow", "genres": ["orchestral", "cinematic"]}
        ],
        "calm": [
            {"title": "Peaceful Morning", "artist": "Various Artists", "tempo": "slow", "genres": ["ambient", "piano"]},
            {"title": "Gentle Waves", "artist": "Various Artists", "tempo": "slow", "genres": ["ambient", "nature"]},
            {"title": "Soft Meditation", "artist": "Various Artists", "tempo": "slow", "genres": ["ambient", "meditation"]}
        ],
        "energetic": [
            {"title": "Power Up", "artist": "Various Artists", "tempo": "fast", "genres": ["electronic", "rock"]},
            {"title": "Adrenaline Rush", "artist": "Various Artists", "tempo": "fast", "genres": ["electronic", "dubstep"]},
            {"title": "High Energy", "artist": "Various Artists", "tempo": "fast", "genres": ["electronic", "dance"]}
        ],
        "romantic": [
            {"title": "Love Theme", "artist": "Various Artists", "tempo": "slow", "genres": ["piano", "orchestral"]},
            {"title": "Eternal Bond", "artist": "Various Artists", "tempo": "medium", "genres": ["piano", "instrumental"]},
            {"title": "Sweet Romance", "artist": "Various Artists", "tempo": "slow", "genres": ["jazz", "lounge"]}
        ],
        "mysterious": [
            {"title": "Unknown Territory", "artist": "Various Artists", "tempo": "medium", "genres": ["electronic", "ambient"]},
            {"title": "Dark Mystery", "artist": "Various Artists", "tempo": "slow", "genres": ["electronic", "cinematic"]},
            {"title": "Curious Mind", "artist": "Various Artists", "tempo": "medium", "genres": ["electronic", "ambient"]}
        ],
        "epic": [
            {"title": "Epic Adventure", "artist": "Various Artists", "tempo": "fast", "genres": ["orchestral", "trailer"]},
            {"title": "Heroic Journey", "artist": "Various Artists", "tempo": "medium", "genres": ["orchestral", "cinematic"]},
            {"title": "Grand Victory", "artist": "Various Artists", "tempo": "fast", "genres": ["orchestral", "trailer"]}
        ],
        "funny": [
            {"title": "Comedy Sketch", "artist": "Various Artists", "tempo": "medium", "genres": ["quirky", "electronic"]},
            {"title": "Silly Moments", "artist": "Various Artists", "tempo": "fast", "genres": ["quirky", "whimsical"]},
            {"title": "Light Humor", "artist": "Various Artists", "tempo": "medium", "genres": ["jazz", "whimsical"]}
        ],
        "neutral": [
            {"title": "Background Atmosphere", "artist": "Various Artists", "tempo": "medium", "genres": ["ambient", "background"]},
            {"title": "Subtle Presence", "artist": "Various Artists", "tempo": "medium", "genres": ["ambient", "background"]},
            {"title": "Gentle Flow", "artist": "Various Artists", "tempo": "medium", "genres": ["ambient", "background"]}
        ]
    }
    
    # Use neutral mood if the specified mood isn't found
    if mood.lower() not in mood_tracks:
        mood = "neutral"
    
    # Get tracks for the mood
    tracks = mood_tracks[mood.lower()]
    
    # Filter by tempo if specified
    if tempo:
        tracks = [track for track in tracks if track["tempo"] == tempo]
    
    # Filter by genre if specified
    if genre:
        tracks = [track for track in tracks if genre in track["genres"]]
    
    # If no tracks match the filters, return the original list
    if not tracks and (tempo or genre):
        tracks = mood_tracks[mood.lower()]
    
    # Add license information (for a real application, this would be actual license info)
    for track in tracks:
        track["license"] = "Royalty-free for commercial use"
        track["source"] = "AI Video Platform Library"
    
    return tracks

def recommend_music(transcription, mood=None):
    """
    Recommend background music based on video content and mood.
    
    Args:
        transcription (dict): Transcription data
        mood (str, optional): Explicitly specified mood
        
    Returns:
        dict: Music recommendations
    """
    try:
        # If mood is explicitly provided, use it
        if mood:
            detected_mood = {
                "primary_mood": mood,
                "secondary_mood": None,
                "confidence": 1.0
            }
        else:
            # Analyze the mood from the transcription
            detected_mood = analyze_mood_with_ai(transcription)
        
        # Get primary and secondary recommendations
        primary_recommendations = get_music_recommendations(detected_mood["primary_mood"])
        
        secondary_recommendations = []
        if detected_mood.get("secondary_mood"):
            secondary_recommendations = get_music_recommendations(detected_mood["secondary_mood"])
        
        # Generate the final recommendations
        result = {
            "detected_mood": {
                "primary": detected_mood["primary_mood"],
                "secondary": detected_mood.get("secondary_mood"),
                "confidence": detected_mood.get("confidence", 0.7)
            },
            "primary_recommendations": primary_recommendations,
            "secondary_recommendations": secondary_recommendations,
            "mood_phrases": detected_mood.get("mood_phrases", []),
            "mood_changes": detected_mood.get("mood_changes", [])
        }
        
        return result
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Music recommendation error: {str(e)}")
        raise 