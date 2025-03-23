#!/usr/bin/env python3
"""
Emotion Markup Language Parser

This module provides tools for parsing and processing emotion markup language,
which allows for fine-grained control over emotions and expressions in text.
"""

import re
from typing import Dict, List, Tuple, Optional, Union


class EmotionMarkupParser:
    """
    Parser for emotion markup language that converts tagged text into
    animation events with timing and emotion intensity information.
    """

    def __init__(self):
        """Initialize the emotion markup parser."""
        # Supported emotion tags and their default intensities
        self.supported_emotions = {
            "happy": 0.7,
            "sad": 0.7,
            "angry": 0.7,
            "surprised": 0.7,
            "disgusted": 0.7,
            "fearful": 0.7,
            "neutral": 0.7
        }
        
        # Compile regex patterns for parsing
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for parsing markup."""
        # Pattern for matching emotion tags with attributes
        self.tag_pattern = re.compile(
            r'<(?P<emotion>' + '|'.join(self.supported_emotions.keys()) + r')' +
            r'(?P<attributes>(\s+[a-z_]+="[^"]*")*)\s*>' +
            r'(?P<content>.*?)' +
            r'</(?P=emotion)>', 
            re.DOTALL
        )
        
        # Pattern for parsing attributes
        self.attribute_pattern = re.compile(r'([a-z_]+)="([^"]*)"')
    
    def parse(self, markup: str) -> Tuple[str, List[Dict]]:
        """
        Parse the emotion markup and extract plain text and emotion events.
        
        Args:
            markup: String containing emotion markup language
            
        Returns:
            Tuple of (plain_text, emotion_events)
        """
        plain_text = markup
        emotion_events = []
        
        # Keep track of current position in text
        text_pos = 0
        
        # Process all emotion tags
        for match in self.tag_pattern.finditer(markup):
            # Extract information from match
            emotion = match.group('emotion')
            attributes_str = match.group('attributes')
            content = match.group('content')
            start_pos = match.start()
            end_pos = match.end()
            
            # Parse attributes
            attributes = {}
            for attr_match in self.attribute_pattern.finditer(attributes_str):
                key, value = attr_match.groups()
                if key == 'intensity':
                    attributes[key] = float(value)
                elif key == 'transition':
                    attributes[key] = float(value)
                else:
                    attributes[key] = value
            
            # Get intensity (default if not specified)
            intensity = attributes.get('intensity', self.supported_emotions[emotion])
            
            # Get transition time (default to 0.3 seconds if not specified)
            transition = attributes.get('transition', 0.3)
            
            # Calculate character positions in plain text
            content_start_pos = text_pos + (start_pos - text_pos) - (match.start() - match.start('content'))
            content_len = len(content)
            
            # Create emotion event
            event = {
                'emotion': emotion,
                'intensity': intensity,
                'transition': transition,
                'text_range': (content_start_pos, content_start_pos + content_len)
            }
            emotion_events.append(event)
            
            # Update text position
            text_pos += (end_pos - start_pos) - (len(match.group(0)) - len(content))
        
        # Remove all tags to get plain text
        plain_text = re.sub(r'</?[a-z_]+(\s+[a-z_]+="[^"]*")*\s*>', '', markup)
        
        return plain_text, emotion_events
    
    def convert_to_animation_events(self, markup: str, 
                                   start_time: float, 
                                   speech_rate: float = 0.07, 
                                   fps: int = 30) -> List[Dict]:
        """
        Convert markup to animation events with timing information.
        
        Args:
            markup: String containing emotion markup language
            start_time: Start time in seconds for the speech
            speech_rate: Average time in seconds per character
            fps: Frames per second
            
        Returns:
            List of animation events
        """
        plain_text, emotion_events = self.parse(markup)
        animation_events = []
        
        total_text_length = len(plain_text)
        
        for event in emotion_events:
            text_start, text_end = event['text_range']
            
            # Calculate time based on character position and speech rate
            event_start_time = start_time + (text_start * speech_rate)
            event_end_time = start_time + (text_end * speech_rate)
            
            # Create animation event
            animation_event = {
                "type": "emotion",
                "start_time": event_start_time,
                "end_time": event_end_time,
                "emotion": event['emotion'],
                "intensity": event['intensity'],
                "transition": event['transition']
            }
            
            animation_events.append(animation_event)
        
        return animation_events
    
    def extract_text(self, markup: str) -> str:
        """
        Extract plain text from markup by removing all tags.
        
        Args:
            markup: String containing emotion markup language
            
        Returns:
            Plain text without markup
        """
        return re.sub(r'</?[a-z_]+(\s+[a-z_]+="[^"]*")*\s*>', '', markup)
    
    def validate(self, markup: str) -> Tuple[bool, Optional[str]]:
        """
        Validate emotion markup for correct syntax.
        
        Args:
            markup: String containing emotion markup language
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for unmatched opening tags
        opening_tags = re.findall(r'<([a-z_]+)(\s+[a-z_]+="[^"]*")*\s*>', markup)
        closing_tags = re.findall(r'</([a-z_]+)>', markup)
        
        opening_emotions = [tag[0] for tag in opening_tags]
        
        if len(opening_emotions) != len(closing_tags):
            return False, "Unmatched tags: number of opening and closing tags doesn't match"
        
        # Check for tags that aren't in our supported emotions
        for emotion in opening_emotions + closing_tags:
            if emotion not in self.supported_emotions:
                return False, f"Unsupported emotion tag: {emotion}"
        
        # Check for nested tags (simplified check)
        tag_positions = []
        for match in re.finditer(r'</?([a-z_]+)(\s+[a-z_]+="[^"]*")*\s*>', markup):
            is_closing = match.group(0).startswith('</')
            tag = match.group(1)
            
            if is_closing:
                # Check if the last opening tag matches this closing tag
                if not tag_positions or tag_positions[-1][0] != tag:
                    return False, f"Improperly nested tags at position {match.start()}"
                tag_positions.pop()
            else:
                tag_positions.append((tag, match.start()))
        
        # There should be no unclosed tags
        if tag_positions:
            return False, f"Unclosed tag: {tag_positions[0][0]} at position {tag_positions[0][1]}"
        
        # Validate attributes
        for match in re.finditer(r'<([a-z_]+)(\s+[a-z_]+="[^"]*")*\s*>', markup):
            attributes_str = match.group(2) or ""
            for attr_match in re.finditer(r'([a-z_]+)="([^"]*)"', attributes_str):
                key, value = attr_match.groups()
                
                if key == 'intensity':
                    try:
                        intensity = float(value)
                        if not 0 <= intensity <= 1:
                            return False, f"Intensity must be between 0 and 1, got {value}"
                    except ValueError:
                        return False, f"Invalid intensity value: {value}"
                
                elif key == 'transition':
                    try:
                        transition = float(value)
                        if transition < 0:
                            return False, f"Transition time cannot be negative, got {value}"
                    except ValueError:
                        return False, f"Invalid transition value: {value}"
        
        return True, None


def parse_emotion_markup(markup: str, start_time: float = 0.0, fps: int = 30) -> List[Dict]:
    """
    Utility function to parse emotion markup and convert to animation events.
    
    Args:
        markup: String containing emotion markup language
        start_time: Start time in seconds
        fps: Frames per second
        
    Returns:
        List of animation events
    """
    parser = EmotionMarkupParser()
    return parser.convert_to_animation_events(markup, start_time, fps=fps)


if __name__ == "__main__":
    # Example usage
    test_markup = """
    <happy intensity="0.8" transition="0.5">I'm so excited</happy> to show you these 
    <surprised intensity="0.6">amazing</surprised> features!
    """
    
    parser = EmotionMarkupParser()
    text, events = parser.parse(test_markup)
    
    print("Plain text:", text)
    print("Emotion events:")
    for event in events:
        print(f"  - {event['emotion']} (intensity: {event['intensity']}) at {event['text_range']}")
    
    animation_events = parser.convert_to_animation_events(test_markup, 1.0)
    print("\nAnimation events:")
    for event in animation_events:
        print(f"  - {event['emotion']} from {event['start_time']:.2f}s to {event['end_time']:.2f}s") 