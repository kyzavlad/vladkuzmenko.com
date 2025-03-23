# Script Timing Preservation Module

## Overview

The Script Timing Preservation Module provides functionality for maintaining proper timing information when translating scripts for video dubbing and subtitling. It ensures that translated content matches the timing of the original content as closely as possible, taking into account the differences in language structure, word length, and speech rate across different languages.

## Key Features

- **Language-Specific Timing Profiles**: Contains timing data for various languages including average speaking rates, syllable patterns, and timing adjustment factors
- **Automatic Time Adjustment**: Automatically adjusts segment durations based on translation length and language characteristics
- **Syllable Counting**: Implements language-specific syllable counting for more accurate timing predictions
- **Timing Issue Detection**: Identifies potential problems like speaking rates that are too fast or too slow
- **Segment Optimization**: Resolves overlaps and optimizes a sequence of timed segments
- **Multi-Language Support**: Includes profiles for multiple languages with expansion capabilities

## Core Components

### TimedSegment Class

The `TimedSegment` class represents a segment of text with timing information:

```python
@dataclass
class TimedSegment:
    text: str               # The text content
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds
    segment_id: Optional[str] = None  # Optional identifier
    
    # Properties: duration, characters_per_second, word_count, words_per_minute
    # Methods: adjust_end_time, adjust_start_time, split
```

### LanguageTimingProfile Class

The `LanguageTimingProfile` class holds timing profiles for different languages:

```python
@dataclass
class LanguageTimingProfile:
    language_code: str                                # ISO language code
    avg_chars_per_second: float                       # Average spoken characters per second
    avg_words_per_minute: float                       # Average spoken words per minute
    syllable_pattern: str                             # Regex pattern for syllable detection
    avg_syllables_per_word: float                     # Average syllables per word
    timing_adjustment_factors: Dict[str, float]       # Adjustment factors for language pairs
    max_comfortable_chars_per_second: float = 15.0    # Maximum comfortable speaking rate
    properties: Dict[str, Any] = field(default_factory=dict)  # Additional properties
```

### ScriptTimingPreserver Class

The main class that provides timing preservation functionality:

```python
class ScriptTimingPreserver:
    # Methods:
    # - adjust_timing: Adjusts timing for a translated segment
    # - count_syllables: Counts syllables in text for a specific language
    # - optimize_segments: Optimizes a sequence of segments
    # - check_timing_issues: Detects potential timing problems
    # - process_script: Processes an entire script with timing info
```

## Usage Examples

### Basic Usage: Adjusting Timing for a Translated Segment

```python
from app.avatar_creation.video_translation.timing import (
    ScriptTimingPreserver, TimedSegment
)

# Create a timed segment
segment = TimedSegment(
    text="Welcome to our demonstration of the avatar creation system.",
    start_time=0.0,
    end_time=3.0,
    segment_id="intro_1"
)

# Spanish translation (typically longer than English)
translation = "Bienvenido a nuestra demostración del sistema de creación de avatares."

# Initialize timing preserver
preserver = ScriptTimingPreserver()

# Adjust timing for translated segment
adjusted = preserver.adjust_timing(segment, translation, "en", "es")

print(f"Original duration: {segment.duration:.2f}s")
print(f"Adjusted duration: {adjusted.duration:.2f}s")
```

### Processing an Entire Script

```python
# Load a script with timed segments
script = {
    "title": "Product Demo",
    "language": "en",
    "segments": [
        {
            "id": "intro_1",
            "text": "Welcome to our product demonstration.",
            "start_time": 0.0,
            "end_time": 2.0
        },
        # More segments...
    ]
}

# Translations for each segment
translations = {
    "es": {
        "intro_1": "Bienvenido a nuestra demostración de producto.",
        # More translations...
    }
}

# Process the entire script
preserver = ScriptTimingPreserver()
translated_script = preserver.process_script(script, translations, "en", "es")

# The translated_script now contains all segments with adjusted timing
```

### Checking for Timing Issues

```python
# Check if a segment has timing issues
segment = TimedSegment(
    text="Este texto es muy largo para el tiempo asignado y podría ser difícil de leer rápidamente.",
    start_time=0.0,
    end_time=2.5,
    segment_id="example"
)

preserver = ScriptTimingPreserver()
issues = preserver.check_timing_issues(segment, "es")

if issues["has_issues"]:
    if issues["too_fast"]:
        print(f"Speaking rate too fast: {issues['chars_per_second']:.2f} chars/sec")
        print(f"Recommended duration: {issues['recommended_duration']:.2f}s")
```

## Language Support

The module includes timing profiles for the following languages:

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)
- Russian (ru)
- Arabic (ar)

Additional languages can be added by extending the `_load_language_profiles` method.

## Integration with the Video Translation Module

The Script Timing Preservation Module is designed to work seamlessly with the rest of the Video Translation Module:

```python
from app.avatar_creation.video_translation import (
    NeuralTranslator, TranslationOptions, 
    TerminologyManager, ScriptTimingPreserver
)

# Initialize components
translator = NeuralTranslator()
term_manager = TerminologyManager()
timing_preserver = ScriptTimingPreserver()

# Configure translation options
options = TranslationOptions(
    preserve_technical_terms=True
)

# Translate script segments
# ...

# Adjust timing for the translation
translated_script = timing_preserver.process_script(
    script, translations, source_lang, target_lang
)
```

## Best Practices

1. **Language-Specific Adjustments**: Always specify the correct source and target languages to get proper timing adjustments.

2. **Terminology Preservation**: Use the terminology management features alongside timing preservation for technical content.

3. **Review Fast Segments**: Always review segments flagged as "too fast" as they may be difficult to read or speak.

4. **Testing with Speakers**: Test adjusted timing with native speakers to ensure comfort levels.

5. **Context Consideration**: Consider contextual factors like speaker speed, audience, and content type when finalizing timing.

## API Reference

### TimedSegment

| Property/Method | Description |
|-----------------|-------------|
| `text` | The segment text content |
| `start_time` | Start time in seconds |
| `end_time` | End time in seconds |
| `segment_id` | Optional identifier |
| `duration` | Calculated duration in seconds |
| `characters_per_second` | Calculated character rate |
| `word_count` | Number of words in the segment |
| `words_per_minute` | Speaking rate in words per minute |
| `adjust_end_time(new_duration)` | Adjust end time to match a new duration |
| `adjust_start_time(new_start_time)` | Adjust start time while maintaining duration |
| `split(at_time)` | Split the segment at the specified time |

### ScriptTimingPreserver

| Method | Description |
|--------|-------------|
| `adjust_timing(segment, translation, source_lang, target_lang)` | Adjust timing for a translated segment |
| `count_syllables(text, language_code)` | Count syllables in text for a specific language |
| `optimize_segments(segments, translations, source_lang, target_lang)` | Optimize a sequence of segments |
| `check_timing_issues(segment, language_code)` | Detect potential timing issues |
| `process_script(script, translations, source_lang, target_lang)` | Process an entire script with timing information |
| `_resolve_timing_conflicts(segments)` | Internal method to resolve timing conflicts | 