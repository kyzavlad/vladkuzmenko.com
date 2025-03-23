import os
import json
import re
from flask import current_app
import openai
from collections import Counter

def extract_keywords(text, max_keywords=10):
    """
    Extract important keywords from text.
    
    Args:
        text (str): Text to extract keywords from
        max_keywords (int): Maximum number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    # Remove common stop words
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", 
        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", 
        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
        "about", "against", "between", "into", "through", "during", "before", "after", 
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
        "under", "again", "further", "then", "once", "here", "there", "when", "where", 
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
        "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", 
        "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", 
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", 
        "wasn", "weren", "won", "wouldn", "um", "uh", "like", "okay", "actually", "basically"
    ])
    
    # Clean and tokenize the text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    
    # Filter stop words and count occurrences
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    
    # Get the most common keywords
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords

def analyze_content_with_ai(transcription):
    """
    Analyze content with AI to identify key topics and visuals.
    
    Args:
        transcription (dict): Transcription data
        
    Returns:
        dict: Analysis results with topics and suggested visuals
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
            "topics": [],
            "suggested_visuals": []
        }
    
    # Prepare the prompt for OpenAI
    prompt = f"""
    Analyze the following transcript and suggest B-roll footage that would enhance the video:
    
    Transcript:
    {text}
    
    Please identify:
    1. Key topics discussed
    2. Specific visual elements that would complement the content
    3. Suggested B-roll footage for each segment
    
    Format your response as a JSON object with the following structure:
    {{
        "topics": ["topic1", "topic2", ...],
        "visual_suggestions": [
            {{
                "timeframe": "approximate timestamp or segment description",
                "suggestion": "detailed description of suggested b-roll",
                "keywords": ["keyword1", "keyword2", ...]
            }},
            ...
        ]
    }}
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
        # Fallback to basic keyword extraction
        keywords = extract_keywords(text, max_keywords=20)
        return {
            "topics": keywords[:5],
            "visual_suggestions": [
                {
                    "timeframe": "entire video",
                    "suggestion": f"Visual elements related to {', '.join(keywords[:5])}",
                    "keywords": keywords
                }
            ]
        }

def map_keywords_to_timestamps(transcription):
    """
    Map keywords to their timestamps in the video.
    
    Args:
        transcription (dict): Transcription data with segments and words
        
    Returns:
        dict: Keywords mapped to timestamps
    """
    result = {}
    
    # Extract keywords from each segment and map to timestamps
    segments = transcription.get("segments", [])
    for segment in segments:
        text = segment.get("text", "")
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        # Extract keywords from this segment
        keywords = extract_keywords(text, max_keywords=5)
        
        # Map each keyword to this timestamp
        for keyword in keywords:
            if keyword not in result:
                result[keyword] = []
            result[keyword].append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
    
    return result

def suggest_b_roll(transcription):
    """
    Suggest B-roll footage based on spoken content analysis.
    
    Args:
        transcription (dict): Transcription data
        
    Returns:
        dict: B-roll suggestions
    """
    try:
        # Get AI analysis of the content
        analysis = analyze_content_with_ai(transcription)
        
        # Map keywords to timestamps
        keyword_timestamps = map_keywords_to_timestamps(transcription)
        
        # Enhance visual suggestions with timestamps
        visual_suggestions = analysis.get("visual_suggestions", [])
        for suggestion in visual_suggestions:
            keywords = suggestion.get("keywords", [])
            
            # Find timestamps for these keywords
            timestamp_info = []
            for keyword in keywords:
                if keyword in keyword_timestamps:
                    timestamp_info.extend(keyword_timestamps[keyword])
            
            # Add timestamp information if available
            if timestamp_info:
                suggestion["timestamps"] = timestamp_info
        
        # Generate the final suggestions
        result = {
            "topics": analysis.get("topics", []),
            "suggestions": visual_suggestions,
            "keyword_mapping": keyword_timestamps
        }
        
        return result
    
    except Exception as e:
        # Log the error and re-raise
        print(f"B-roll suggestion error: {str(e)}")
        raise

def get_stock_footage_suggestions(keyword, category=None):
    """
    Get stock footage suggestions for a keyword.
    
    Args:
        keyword (str): Keyword to search for
        category (str, optional): Category to filter by
        
    Returns:
        list: Stock footage suggestions
    """
    # This is a placeholder function that would typically call a stock footage API
    # For now, we'll return a static response based on the keyword
    
    # Some common categories of b-roll footage
    common_categories = {
        "nature": ["mountains", "forests", "beaches", "rivers", "lakes", "animals", "sunset", "sunrise"],
        "city": ["skyline", "streets", "buildings", "traffic", "people", "night", "architecture"],
        "technology": ["computers", "coding", "devices", "screens", "data", "gadgets", "innovation"],
        "business": ["office", "meeting", "presentation", "handshake", "teamwork", "conference"],
        "lifestyle": ["family", "friends", "eating", "shopping", "exercise", "relaxation"],
        "travel": ["airport", "airplane", "hotel", "tourism", "landmarks", "exploration"]
    }
    
    # Find which category the keyword matches best
    matched_category = category
    if not matched_category:
        for cat, keywords in common_categories.items():
            if keyword.lower() in keywords or any(k in keyword.lower() for k in keywords):
                matched_category = cat
                break
        
        if not matched_category:
            matched_category = "general"
    
    # Return mock suggestions
    return {
        "keyword": keyword,
        "category": matched_category,
        "suggestions": [
            f"{keyword} in {matched_category} context - wide shot",
            f"{keyword} in {matched_category} context - close-up",
            f"{keyword} in {matched_category} context - slow motion",
            f"People interacting with {keyword}",
            f"{keyword} from different angles"
        ]
    } 