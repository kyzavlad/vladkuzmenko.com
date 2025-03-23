#!/usr/bin/env python3
"""
Cultural Adaptation Module

This module provides functionality for adapting cultural references and
idiomatic expressions during the translation process, ensuring that the
translated content is culturally appropriate and natural.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field


@dataclass
class CulturalReference:
    """Representation of a cultural reference with alternatives for different cultures."""
    source_reference: str  # The original reference
    source_culture: str  # Culture/region of the original reference
    alternatives: Dict[str, str] = field(default_factory=dict)  # Alternative references by culture code
    explanation: str = ""  # Explanation of the reference
    context: str = ""  # Context in which the reference is used
    tags: List[str] = field(default_factory=list)  # Tags for categorization (e.g., "holiday", "food", "celebrity")
    
    def get_alternative(self, target_culture: str) -> Optional[str]:
        """Get an alternative reference for the target culture."""
        if target_culture in self.alternatives:
            return self.alternatives[target_culture]
        
        # Try with just the base culture code (without region)
        if '-' in target_culture:
            base_culture = target_culture.split('-')[0]
            if base_culture in self.alternatives:
                return self.alternatives[base_culture]
        
        return None
    
    def add_alternative(self, culture: str, alternative: str) -> None:
        """Add an alternative reference for a specific culture."""
        self.alternatives[culture] = alternative
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source_reference": self.source_reference,
            "source_culture": self.source_culture,
            "alternatives": self.alternatives,
            "explanation": self.explanation,
            "context": self.context,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CulturalReference':
        """Create instance from dictionary data."""
        return cls(
            source_reference=data.get("source_reference", ""),
            source_culture=data.get("source_culture", ""),
            alternatives=data.get("alternatives", {}),
            explanation=data.get("explanation", ""),
            context=data.get("context", ""),
            tags=data.get("tags", [])
        )


@dataclass
class IdiomaticExpression:
    """Representation of an idiomatic expression with translations for different languages."""
    source_expression: str  # The original expression
    source_language: str  # Language of the original expression
    translations: Dict[str, str] = field(default_factory=dict)  # Translations by language code
    literal_meaning: str = ""  # Literal meaning of the expression
    figurative_meaning: str = ""  # Figurative meaning of the expression
    example_usage: str = ""  # Example of usage in a sentence
    
    def get_translation(self, target_language: str) -> Optional[str]:
        """Get a translation for the target language."""
        if target_language in self.translations:
            return self.translations[target_language]
        
        # Try with just the base language code (without dialect)
        if '-' in target_language:
            base_lang = target_language.split('-')[0]
            if base_lang in self.translations:
                return self.translations[base_lang]
        
        return None
    
    def add_translation(self, language: str, translation: str) -> None:
        """Add a translation for a specific language."""
        self.translations[language] = translation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source_expression": self.source_expression,
            "source_language": self.source_language,
            "translations": self.translations,
            "literal_meaning": self.literal_meaning,
            "figurative_meaning": self.figurative_meaning,
            "example_usage": self.example_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IdiomaticExpression':
        """Create instance from dictionary data."""
        return cls(
            source_expression=data.get("source_expression", ""),
            source_language=data.get("source_language", ""),
            translations=data.get("translations", {}),
            literal_meaning=data.get("literal_meaning", ""),
            figurative_meaning=data.get("figurative_meaning", ""),
            example_usage=data.get("example_usage", "")
        )


class CulturalAdapter:
    """
    Manages cultural references and adaptations for different target cultures.
    """
    
    def __init__(self, reference_dir: str = "data/cultural_references"):
        """
        Initialize the cultural adapter.
        
        Args:
            reference_dir: Directory containing cultural reference files
        """
        self.reference_dir = reference_dir
        self.references: Dict[str, CulturalReference] = {}  # Key is the source reference
        self.tagged_references: Dict[str, List[str]] = {}  # Tag -> list of reference keys
        
        # Load cultural references if available
        self._load_references()
    
    def _load_references(self) -> None:
        """Load cultural references from files in the reference directory."""
        if not os.path.exists(self.reference_dir):
            print(f"Cultural reference directory not found: {self.reference_dir}")
            return
        
        for filename in os.listdir(self.reference_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.reference_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reference_data = json.load(f)
                        
                        if isinstance(reference_data, list):
                            # List of reference definitions
                            for item in reference_data:
                                try:
                                    reference = CulturalReference.from_dict(item)
                                    self.add_reference(reference)
                                except Exception as e:
                                    print(f"Error loading cultural reference: {e}")
                        elif isinstance(reference_data, dict):
                            # Single reference definition
                            try:
                                reference = CulturalReference.from_dict(reference_data)
                                self.add_reference(reference)
                            except Exception as e:
                                print(f"Error loading cultural reference: {e}")
                except Exception as e:
                    print(f"Error loading reference file {filepath}: {e}")
    
    def add_reference(self, reference: CulturalReference) -> None:
        """
        Add a cultural reference to the adapter.
        
        Args:
            reference: Cultural reference to add
        """
        key = reference.source_reference.lower()
        self.references[key] = reference
        
        # Add to tag index
        for tag in reference.tags:
            if tag not in self.tagged_references:
                self.tagged_references[tag] = []
            
            self.tagged_references[tag].append(key)
    
    def get_reference(self, reference: str) -> Optional[CulturalReference]:
        """
        Get a cultural reference by name.
        
        Args:
            reference: Reference to look up
            
        Returns:
            Cultural reference or None if not found
        """
        return self.references.get(reference.lower())
    
    def get_references_by_tag(self, tag: str) -> List[CulturalReference]:
        """
        Get all cultural references with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of cultural references with the tag
        """
        if tag not in self.tagged_references:
            return []
        
        return [self.references[key] for key in self.tagged_references[tag]]
    
    def get_alternative(self, reference: str, target_culture: str) -> Optional[str]:
        """
        Get an alternative for a cultural reference in the target culture.
        
        Args:
            reference: Source reference to adapt
            target_culture: Target culture code
            
        Returns:
            Alternative reference or None if not found
        """
        ref = self.get_reference(reference)
        if ref:
            return ref.get_alternative(target_culture)
        
        return None
    
    def process_text(self, text: str, source_culture: str, target_culture: str) -> Tuple[str, List[Dict]]:
        """
        Process text to identify and adapt cultural references.
        
        Args:
            text: Text to process
            source_culture: Source culture code
            target_culture: Target culture code
            
        Returns:
            Tuple of (processed_text, adaptations)
        """
        adaptations = []
        processed_text = text
        
        # Sort references by length (longest first) to avoid partial matches
        sorted_refs = sorted(self.references.keys(), key=len, reverse=True)
        
        for ref_key in sorted_refs:
            ref = self.references[ref_key]
            
            # Skip references from different source cultures
            if ref.source_culture != source_culture:
                continue
            
            # Look for the reference in the text
            pattern = re.compile(r'\b' + re.escape(ref.source_reference) + r'\b', re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.span()
                
                # Get the alternative for the target culture
                alternative = ref.get_alternative(target_culture)
                
                adaptation_info = {
                    "reference": ref.source_reference,
                    "position": (start, end),
                    "alternative": alternative,
                    "tags": ref.tags,
                    "explanation": ref.explanation
                }
                
                adaptations.append(adaptation_info)
                
                # In a real implementation, you might want to replace the reference in the text
                # with the alternative. For this example, we just track the adaptations.
        
        return processed_text, adaptations
    
    def save_references(self, output_path: str) -> bool:
        """
        Save all cultural references to a file.
        
        Args:
            output_path: Path to save the references file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            ref_list = [ref.to_dict() for ref in self.references.values()]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ref_list, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving references to {output_path}: {e}")
            return False


class IdiomaticExpressionHandler:
    """
    Manages idiomatic expressions and their translations for different languages.
    """
    
    def __init__(self, expression_dir: str = "data/idiomatic_expressions"):
        """
        Initialize the idiomatic expression handler.
        
        Args:
            expression_dir: Directory containing idiomatic expression files
        """
        self.expression_dir = expression_dir
        self.expressions: Dict[str, Dict[str, IdiomaticExpression]] = {}  # Language -> Dict of expressions
        
        # Load idiomatic expressions if available
        self._load_expressions()
    
    def _load_expressions(self) -> None:
        """Load idiomatic expressions from files in the expression directory."""
        if not os.path.exists(self.expression_dir):
            print(f"Idiomatic expression directory not found: {self.expression_dir}")
            return
        
        for filename in os.listdir(self.expression_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.expression_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        expression_data = json.load(f)
                        
                        if isinstance(expression_data, list):
                            # List of expression definitions
                            for item in expression_data:
                                try:
                                    expression = IdiomaticExpression.from_dict(item)
                                    self.add_expression(expression)
                                except Exception as e:
                                    print(f"Error loading idiomatic expression: {e}")
                        elif isinstance(expression_data, dict):
                            # Single expression definition
                            try:
                                expression = IdiomaticExpression.from_dict(expression_data)
                                self.add_expression(expression)
                            except Exception as e:
                                print(f"Error loading idiomatic expression: {e}")
                except Exception as e:
                    print(f"Error loading expression file {filepath}: {e}")
    
    def add_expression(self, expression: IdiomaticExpression) -> None:
        """
        Add an idiomatic expression to the handler.
        
        Args:
            expression: Idiomatic expression to add
        """
        language = expression.source_language
        
        if language not in self.expressions:
            self.expressions[language] = {}
        
        key = expression.source_expression.lower()
        self.expressions[language][key] = expression
    
    def get_expression(self, expression: str, language: str) -> Optional[IdiomaticExpression]:
        """
        Get an idiomatic expression by text and language.
        
        Args:
            expression: Expression text to look up
            language: Language of the expression
            
        Returns:
            Idiomatic expression or None if not found
        """
        if language not in self.expressions:
            return None
        
        return self.expressions[language].get(expression.lower())
    
    def get_translation(self, expression: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Get the translation of an idiomatic expression.
        
        Args:
            expression: Source expression to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated expression or None if not found
        """
        expr = self.get_expression(expression, source_language)
        if expr:
            return expr.get_translation(target_language)
        
        return None
    
    def process_text(self, text: str, source_language: str, target_language: str) -> Tuple[str, List[Dict]]:
        """
        Process text to identify and translate idiomatic expressions.
        
        Args:
            text: Text to process
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Tuple of (processed_text, translations)
        """
        translations = []
        processed_text = text
        
        # Make sure we have expressions for this language
        if source_language not in self.expressions:
            return processed_text, translations
        
        # Sort expressions by length (longest first) to avoid partial matches
        sorted_expressions = sorted(self.expressions[source_language].keys(), key=len, reverse=True)
        
        for expr_key in sorted_expressions:
            expr = self.expressions[source_language][expr_key]
            
            # Look for the expression in the text
            # This simple pattern matching is just for demonstration; in a real implementation,
            # you'd want to use more sophisticated matching for idiomatic expressions
            pattern = re.compile(r'\b' + re.escape(expr.source_expression) + r'\b', re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.span()
                
                # Get the translation for the target language
                translation = expr.get_translation(target_language)
                
                translation_info = {
                    "expression": expr.source_expression,
                    "position": (start, end),
                    "translation": translation,
                    "literal_meaning": expr.literal_meaning,
                    "figurative_meaning": expr.figurative_meaning
                }
                
                translations.append(translation_info)
                
                # In a real implementation, you might want to replace the expression in the text
                # with the translation. For this example, we just track the translations.
        
        return processed_text, translations
    
    def save_expressions(self, language: str, output_path: str) -> bool:
        """
        Save idiomatic expressions for a specific language to a file.
        
        Args:
            language: Language code for the expressions to save
            output_path: Path to save the expressions file
            
        Returns:
            True if saved successfully, False otherwise
        """
        if language not in self.expressions:
            print(f"No expressions found for language: {language}")
            return False
        
        try:
            expr_list = [expr.to_dict() for expr in self.expressions[language].values()]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(expr_list, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving expressions to {output_path}: {e}")
            return False


if __name__ == "__main__":
    # Example usage of CulturalAdapter
    adapter = CulturalAdapter()
    
    # Add some example cultural references
    adapter.add_reference(CulturalReference(
        source_reference="Super Bowl",
        source_culture="en-US",
        alternatives={
            "en-GB": "American Football Championship",
            "es-ES": "Final del fútbol americano",
            "de-DE": "Amerikanisches Football-Finale"
        },
        explanation="The annual championship game of the National Football League (NFL) in the United States",
        tags=["sports", "event", "american"]
    ))
    
    adapter.add_reference(CulturalReference(
        source_reference="Thanksgiving",
        source_culture="en-US",
        alternatives={
            "en-GB": "Harvest Festival",
            "es-ES": "Día de Acción de Gracias",
            "de-DE": "Erntedankfest"
        },
        explanation="An American holiday celebrated on the fourth Thursday of November",
        tags=["holiday", "american", "cultural"]
    ))
    
    # Process a text with cultural references
    text = "They always watch the Super Bowl and have a big Thanksgiving dinner."
    processed_text, adaptations = adapter.process_text(text, "en-US", "es-ES")
    
    print("Original text:", text)
    print("\nCultural adaptations for Spanish (Spain):")
    for adaptation in adaptations:
        print(f"  - {adaptation['reference']} → {adaptation['alternative']}")
        print(f"    ({', '.join(adaptation['tags'])})")
    
    # Example usage of IdiomaticExpressionHandler
    idiom_handler = IdiomaticExpressionHandler()
    
    # Add some example idiomatic expressions
    idiom_handler.add_expression(IdiomaticExpression(
        source_expression="kick the bucket",
        source_language="en",
        translations={
            "es": "estirar la pata",
            "fr": "casser sa pipe",
            "de": "den Löffel abgeben"
        },
        literal_meaning="To strike a bucket with one's foot",
        figurative_meaning="To die",
        example_usage="Unfortunately, my neighbor's old cat kicked the bucket last week."
    ))
    
    idiom_handler.add_expression(IdiomaticExpression(
        source_expression="break the ice",
        source_language="en",
        translations={
            "es": "romper el hielo",
            "fr": "briser la glace",
            "de": "das Eis brechen"
        },
        literal_meaning="To break frozen water into pieces",
        figurative_meaning="To initiate social interaction and reduce tension",
        example_usage="Let's play a game to break the ice and get everyone talking."
    ))
    
    # Process a text with idiomatic expressions
    idiom_text = "She tried to break the ice at the party, but the old man kicked the bucket before responding."
    processed_idiom_text, translations = idiom_handler.process_text(idiom_text, "en", "es")
    
    print("\nOriginal text:", idiom_text)
    print("\nIdiomatic expression translations for Spanish:")
    for translation in translations:
        print(f"  - {translation['expression']} → {translation['translation']}")
        print(f"    (Figurative meaning: {translation['figurative_meaning']})") 