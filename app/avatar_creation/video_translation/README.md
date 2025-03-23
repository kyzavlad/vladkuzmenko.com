# Video Translation Module

## Overview

The Video Translation Module provides comprehensive functionality for translating video content across multiple languages while preserving technical terminology, cultural context, and proper timing for dubbing and subtitling.

This module is designed for the Avatar Creation system to enable multi-language video content with high-quality, context-aware translations that maintain the integrity of technical vocabulary and proper timing.

## Key Features

- **Multi-Language Processing**: Support for 50+ languages with dialect variants
- **Context-Aware Neural Machine Translation**: Maintains proper context across sentences and paragraphs
- **Industry-Specific Terminology Preservation**: Ensures technical terms are translated consistently
- **Cultural Adaptation**: Handles cultural references appropriately for target audiences
- **Idiomatic Expression Handling**: Translates expressions to maintain natural language flow
- **Script Timing Preservation**: Adjusts timing for video dubbing and subtitling
- **Named Entity Recognition**: Preserves names of people, organizations, and products
- **Technical Vocabulary Handling**: Manages domain-specific technical terms

## Components

### Core Translation

- `NeuralTranslator`: Base translator class for neural machine translation
- `ContextAwareTranslator`: Enhanced translator that considers document context
- `TranslationOptions`: Configuration options for translation behavior

### Language Management

- `LanguageManager`: Manages available languages and language pairs
- `LanguageProfile`: Contains properties for a specific language
- `DialectVariant`: Represents dialect variations of a base language

### Terminology Management

- `TerminologyManager`: Manages technical terminology across languages
- `IndustryTerminology`: Industry-specific terminology collections
- `NamedEntityRecognizer`: Identifies and preserves named entities
- `TechnicalVocabulary`: Domain-specific vocabulary handling
- `TermDefinition`: Definition of a term with translations and metadata

### Cultural Adaptation

- `CulturalAdapter`: Adapts cultural references for target audiences
- `IdiomaticExpressionHandler`: Manages translation of idiomatic expressions
- `CulturalReference`: Definition of a cultural reference with alternatives

### Timing Preservation

- `ScriptTimingPreserver`: Maintains proper timing when translating scripts
- `TimedSegment`: Represents a segment of text with timing information
- `LanguageTimingProfile`: Holds timing profiles for different languages

## Usage Examples

### Basic Translation

```python
from app.avatar_creation.video_translation import NeuralTranslator

translator = NeuralTranslator()
translated_text, metadata = translator.translate_text(
    "Welcome to the Avatar Creation System.",
    source_lang="en",
    target_lang="es"
)

print(translated_text)  # "Bienvenido al Sistema de Creación de Avatares."
print(metadata["confidence_score"])  # e.g., 0.92
```

### Context-Aware Translation

```python
from app.avatar_creation.video_translation import ContextAwareTranslator

translator = ContextAwareTranslator(context_window=3)
document = [
    "The bank is closed today.",  # 'bank' could be financial or river bank
    "People were hoping to withdraw some money."  # This clarifies the meaning
]

results = translator.translate_document(document, "en", "es")
for translated, metadata in results:
    print(translated)
    print(f"Context used: {metadata['context_used']}")
```

### Preserving Terminology

```python
from app.avatar_creation.video_translation import (
    TerminologyManager, TermDefinition, ContextAwareTranslator, TranslationOptions
)

# Set up terminology
term_manager = TerminologyManager()
term_manager.add_term(TermDefinition(
    source_term="neural network",
    translations={"es": "red neuronal", "fr": "réseau de neurones"},
    domain="ai",
    part_of_speech="noun"
))

# Configure translator with terminology preservation
translator = ContextAwareTranslator()
options = TranslationOptions(preserve_technical_terms=True)

# Translate with terminology preservation
text = "Our system uses advanced neural networks."
translated, metadata = translator.translate_text(
    text, "en", "es", options, terminology_manager=term_manager
)
```

### Script Timing Preservation

```python
from app.avatar_creation.video_translation import (
    ScriptTimingPreserver, TimedSegment
)

# Create a timed segment
segment = TimedSegment(
    text="Welcome to our product demonstration.",
    start_time=0.0,
    end_time=2.5,
    segment_id="intro_1"
)

# Spanish translation (typically longer than English)
translation = "Bienvenido a nuestra demostración de producto."

# Adjust timing for the translation
preserver = ScriptTimingPreserver()
adjusted = preserver.adjust_timing(segment, translation, "en", "es")

print(f"Original duration: {segment.duration:.2f}s")
print(f"Adjusted duration: {adjusted.duration:.2f}s")
```

## Complete Example: Translating a Video Script

```python
from app.avatar_creation.video_translation import (
    ContextAwareTranslator, TerminologyManager, ScriptTimingPreserver,
    TranslationOptions, TermDefinition
)

# 1. Set up components
translator = ContextAwareTranslator()
term_manager = TerminologyManager()
timing_preserver = ScriptTimingPreserver()

# 2. Add technical terms
term_manager.add_term(TermDefinition(
    source_term="Avatar Creation System",
    translations={"es": "Sistema de Creación de Avatares"},
    domain="product"
))

# 3. Load a script with timing information
script = {
    "title": "Product Demo",
    "language": "en",
    "segments": [
        {
            "id": "intro_1",
            "text": "Welcome to the Avatar Creation System.",
            "start_time": 0.0,
            "end_time": 2.5
        },
        # More segments...
    ]
}

# 4. Translate each segment
options = TranslationOptions(preserve_technical_terms=True)
translations = {"es": {}}

for segment in script["segments"]:
    translated, _ = translator.translate_text(
        segment["text"], "en", "es", options, terminology_manager=term_manager
    )
    translations["es"][segment["id"]] = translated

# 5. Adjust timing for translations
translated_script = timing_preserver.process_script(
    script, translations, "en", "es"
)

# 6. Save or use the translated script with adjusted timing
# ...
```

## Running the Demo

The module includes a comprehensive demo that showcases all its features:

```bash
python -m app.avatar_creation.video_translation.demo --source-lang en --target-lang es
```

## Extending the Module

### Adding New Languages

To add support for a new language:

1. Update the `LanguageManager` with the new language details
2. Add a corresponding timing profile in the `ScriptTimingPreserver`
3. Add relevant terminology translations for the new language

### Adding Industry-Specific Terminology

To add terminology for a specific domain:

```python
term_manager = TerminologyManager()
term_manager.add_term(TermDefinition(
    source_term="my technical term",
    translations={...},
    domain="my_industry",
    description="Description of the term",
    part_of_speech="noun"
))
```

## Documentation

For more detailed documentation, see:

- `docs/TIMING_PRESERVATION.md` - Documentation on the script timing preservation module
- Example scripts in `app/avatar_creation/video_translation/examples/` 