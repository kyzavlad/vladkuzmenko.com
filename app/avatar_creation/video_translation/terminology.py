#!/usr/bin/env python3
"""
Terminology Management Module

This module provides functionality for managing, recognizing, and preserving 
specific terminology during translation, including industry terms, named entities,
and technical vocabulary.
"""

import os
import re
import json
import csv
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field


@dataclass
class TermDefinition:
    """Definition of a terminology entry."""
    source_term: str  # Original term
    translations: Dict[str, str] = field(default_factory=dict)  # Translations by language code
    domain: str = ""  # Industry/domain (e.g., "medical", "legal", "tech")
    description: str = ""  # Description or context
    part_of_speech: str = ""  # e.g., "noun", "verb", "phrase"
    do_not_translate: bool = False  # Whether to keep the term in original language
    
    def get_translation(self, language_code: str) -> Optional[str]:
        """Get the translation for a specific language."""
        if language_code in self.translations:
            return self.translations[language_code]
        
        # Try with just the base language code (without dialect)
        if '-' in language_code:
            base_lang = language_code.split('-')[0]
            if base_lang in self.translations:
                return self.translations[base_lang]
        
        return None
    
    def add_translation(self, language_code: str, translation: str) -> None:
        """Add a translation for a specific language."""
        self.translations[language_code] = translation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source_term": self.source_term,
            "translations": self.translations,
            "domain": self.domain,
            "description": self.description,
            "part_of_speech": self.part_of_speech,
            "do_not_translate": self.do_not_translate
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TermDefinition':
        """Create instance from dictionary data."""
        return cls(
            source_term=data.get("source_term", ""),
            translations=data.get("translations", {}),
            domain=data.get("domain", ""),
            description=data.get("description", ""),
            part_of_speech=data.get("part_of_speech", ""),
            do_not_translate=data.get("do_not_translate", False)
        )


class TerminologyManager:
    """
    Manages terminology dictionaries for consistent translation of specific terms.
    """
    
    def __init__(self, terminology_dir: str = "data/terminology"):
        """
        Initialize the terminology manager.
        
        Args:
            terminology_dir: Directory containing terminology files
        """
        self.terminology_dir = terminology_dir
        self.terms: Dict[str, TermDefinition] = {}  # Key is the source term (case insensitive)
        self.domain_terms: Dict[str, Set[str]] = {}  # Domain -> set of terms
        
        # Load terminology files if available
        self._load_terminology()
    
    def _load_terminology(self) -> None:
        """Load terminology from files in the terminology directory."""
        if not os.path.exists(self.terminology_dir):
            print(f"Terminology directory not found: {self.terminology_dir}")
            return
        
        # Load JSON terminology files
        for filename in os.listdir(self.terminology_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.terminology_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        term_data = json.load(f)
                        
                        if isinstance(term_data, list):
                            # List of term definitions
                            for item in term_data:
                                try:
                                    term = TermDefinition.from_dict(item)
                                    self.add_term(term)
                                except Exception as e:
                                    print(f"Error loading term definition: {e}")
                        elif isinstance(term_data, dict):
                            # Single term definition
                            try:
                                term = TermDefinition.from_dict(term_data)
                                self.add_term(term)
                            except Exception as e:
                                print(f"Error loading term definition: {e}")
                except Exception as e:
                    print(f"Error loading terminology file {filepath}: {e}")
            
            # Load CSV terminology files
            elif filename.endswith('.csv'):
                filepath = os.path.join(self.terminology_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        
                        for row in reader:
                            if 'source_term' in row:
                                translations = {}
                                
                                # Extract translations from columns
                                for key, value in row.items():
                                    if key.startswith('translation_') and value:
                                        lang_code = key[12:]  # Remove 'translation_' prefix
                                        translations[lang_code] = value
                                
                                term = TermDefinition(
                                    source_term=row['source_term'],
                                    translations=translations,
                                    domain=row.get('domain', ''),
                                    description=row.get('description', ''),
                                    part_of_speech=row.get('part_of_speech', ''),
                                    do_not_translate=row.get('do_not_translate', '').lower() in ('true', 'yes', '1')
                                )
                                
                                self.add_term(term)
                except Exception as e:
                    print(f"Error loading terminology CSV file {filepath}: {e}")
    
    def add_term(self, term: TermDefinition) -> None:
        """
        Add a term to the terminology dictionary.
        
        Args:
            term: Term definition to add
        """
        term_key = term.source_term.lower()
        self.terms[term_key] = term
        
        # Add to domain index
        if term.domain:
            if term.domain not in self.domain_terms:
                self.domain_terms[term.domain] = set()
            
            self.domain_terms[term.domain].add(term_key)
    
    def get_term(self, term: str) -> Optional[TermDefinition]:
        """
        Get the definition for a specific term.
        
        Args:
            term: Term to look up
            
        Returns:
            Term definition or None if not found
        """
        return self.terms.get(term.lower())
    
    def get_translation(self, term: str, target_language: str) -> Optional[str]:
        """
        Get the translation of a term for a specific target language.
        
        Args:
            term: Source term to translate
            target_language: Target language code
            
        Returns:
            Translated term or None if not found
        """
        term_def = self.get_term(term)
        if term_def:
            if term_def.do_not_translate:
                return term  # Return the original term
            
            translation = term_def.get_translation(target_language)
            return translation
        
        return None
    
    def get_terms_by_domain(self, domain: str) -> List[TermDefinition]:
        """
        Get all terms for a specific domain.
        
        Args:
            domain: Domain to filter by
            
        Returns:
            List of term definitions in the domain
        """
        if domain not in self.domain_terms:
            return []
        
        return [self.terms[term_key] for term_key in self.domain_terms[domain]]
    
    def save_terminology(self, output_path: str) -> bool:
        """
        Save all terminology to a file.
        
        Args:
            output_path: Path to save the terminology file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            term_list = [term.to_dict() for term in self.terms.values()]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(term_list, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving terminology to {output_path}: {e}")
            return False
    
    def process_text(self, text: str, source_language: str, target_language: str) -> Tuple[str, List[Dict]]:
        """
        Process text to identify and track terminology.
        
        Args:
            text: Text to process
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Tuple of (processed_text, term_occurrences)
        """
        term_occurrences = []
        processed_text = text
        
        # Sort terms by length (longest first) to avoid partial matches
        sorted_terms = sorted(self.terms.keys(), key=len, reverse=True)
        
        for term_key in sorted_terms:
            term_def = self.terms[term_key]
            pattern = re.compile(r'\b' + re.escape(term_def.source_term) + r'\b', re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.span()
                
                # Get the translation
                translation = term_def.get_translation(target_language)
                
                term_info = {
                    "term": term_def.source_term,
                    "position": (start, end),
                    "translation": translation,
                    "domain": term_def.domain,
                    "do_not_translate": term_def.do_not_translate
                }
                
                term_occurrences.append(term_info)
        
        return processed_text, term_occurrences


class IndustryTerminology(TerminologyManager):
    """
    Specialized terminology manager for industry-specific terms.
    """
    
    def __init__(self, industry: str, terminology_dir: str = "data/terminology"):
        """
        Initialize industry-specific terminology manager.
        
        Args:
            industry: Industry domain (e.g., "medical", "legal", "tech")
            terminology_dir: Directory containing terminology files
        """
        super().__init__(terminology_dir)
        
        self.industry = industry
        
        # Try to load industry-specific terminology file
        industry_file = os.path.join(terminology_dir, f"{industry}.json")
        if os.path.exists(industry_file):
            try:
                with open(industry_file, 'r', encoding='utf-8') as f:
                    term_data = json.load(f)
                    
                    if isinstance(term_data, list):
                        for item in term_data:
                            try:
                                term = TermDefinition.from_dict(item)
                                self.add_term(term)
                            except Exception as e:
                                print(f"Error loading industry term: {e}")
            except Exception as e:
                print(f"Error loading industry terminology file {industry_file}: {e}")
    
    def process_industry_text(self, text: str, source_language: str, target_language: str) -> Tuple[str, List[Dict]]:
        """
        Process text with industry-specific term handling.
        
        Args:
            text: Text to process
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Tuple of (processed_text, term_occurrences)
        """
        # Get all terms specific to this industry
        industry_terms = self.get_terms_by_domain(self.industry)
        
        # Use base implementation with special handling for industry terms
        return self.process_text(text, source_language, target_language)


class NamedEntityRecognizer:
    """
    Recognizes and preserves named entities in the translation process.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the named entity recognizer.
        
        Args:
            model_path: Path to the NER model (if any)
        """
        self.model_path = model_path
        self.entity_types = {
            "PERSON": "People, including fictional characters",
            "ORG": "Organizations, companies, institutions",
            "GPE": "Geopolitical entities (countries, cities, states)",
            "LOC": "Non-GPE locations (mountains, water bodies)",
            "PRODUCT": "Products, objects, vehicles, foods",
            "EVENT": "Named events (battles, wars, sports events)",
            "WORK_OF_ART": "Titles of books, songs, etc.",
            "DATE": "Absolute or relative dates",
            "TIME": "Times smaller than a day",
            "MONEY": "Monetary values",
            "QUANTITY": "Measurements",
            "PERCENT": "Percentage",
            "LANGUAGE": "Named languages"
        }
        
        # Custom entity dictionary for special cases
        self.custom_entities = {}
        
        # In a real implementation, this would load a NER model
        # For now, we'll use a simplified approach
    
    def add_custom_entity(self, text: str, entity_type: str) -> None:
        """
        Add a custom named entity to the recognizer.
        
        Args:
            text: Entity text
            entity_type: Entity type
        """
        self.custom_entities[text] = entity_type
    
    def recognize_entities(self, text: str, source_language: str) -> List[Dict]:
        """
        Recognize named entities in the text.
        
        Args:
            text: Text to analyze
            source_language: Source language code
            
        Returns:
            List of recognized entities with positions
        """
        entities = []
        
        # In a real implementation, this would use a NER model
        # For now, we'll check against our custom entities dictionary
        
        for entity_text, entity_type in self.custom_entities.items():
            # Simple pattern matching
            pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.span()
                
                entity_info = {
                    "text": entity_text,
                    "type": entity_type,
                    "position": (start, end)
                }
                
                entities.append(entity_info)
        
        return entities
    
    def process_text(self, text: str, source_language: str) -> Tuple[str, List[Dict]]:
        """
        Process text to identify and mark named entities.
        
        Args:
            text: Text to process
            source_language: Source language code
            
        Returns:
            Tuple of (processed_text, entities)
        """
        # Recognize entities
        entities = self.recognize_entities(text, source_language)
        
        # In a real implementation, this might add markup or placeholders
        # For this example, we just return the text and entities
        return text, entities


class TechnicalVocabulary:
    """
    Manages technical vocabulary across domains and ensures consistent translation.
    """
    
    def __init__(self, vocabulary_dir: str = "data/vocabulary"):
        """
        Initialize the technical vocabulary manager.
        
        Args:
            vocabulary_dir: Directory containing vocabulary files
        """
        self.vocabulary_dir = vocabulary_dir
        
        # Dictionary of technical terms by domain
        # Each term maps to its translations in different languages
        self.vocabulary: Dict[str, Dict[str, Dict[str, str]]] = {}
        
        # Load vocabulary if available
        self._load_vocabulary()
    
    def _load_vocabulary(self) -> None:
        """Load technical vocabulary from files."""
        if not os.path.exists(self.vocabulary_dir):
            print(f"Vocabulary directory not found: {self.vocabulary_dir}")
            return
        
        for filename in os.listdir(self.vocabulary_dir):
            if filename.endswith('.json'):
                domain = os.path.splitext(filename)[0]
                filepath = os.path.join(self.vocabulary_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        domain_vocab = json.load(f)
                        self.vocabulary[domain] = domain_vocab
                except Exception as e:
                    print(f"Error loading vocabulary file {filepath}: {e}")
    
    def get_translation(self, term: str, source_language: str, target_language: str, domain: str = None) -> Optional[str]:
        """
        Get the translation of a technical term.
        
        Args:
            term: Source term to translate
            source_language: Source language code
            target_language: Target language code
            domain: Optional domain to limit the search
            
        Returns:
            Translated term or None if not found
        """
        # If domain is specified, only look in that domain
        if domain and domain in self.vocabulary:
            return self._get_term_translation(self.vocabulary[domain], term, source_language, target_language)
        
        # Otherwise, look in all domains
        for domain_vocab in self.vocabulary.values():
            translation = self._get_term_translation(domain_vocab, term, source_language, target_language)
            if translation:
                return translation
        
        return None
    
    def _get_term_translation(self, domain_vocab: Dict, term: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Helper method to get a term translation from a domain vocabulary.
        
        Args:
            domain_vocab: Domain vocabulary dictionary
            term: Source term to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated term or None if not found
        """
        # Check if term exists in the source language
        if term in domain_vocab.get(source_language, {}):
            # Get translations for this term
            translations = domain_vocab[source_language][term]
            
            # Check if translation exists for target language
            if target_language in translations:
                return translations[target_language]
        
        return None
    
    def add_term(self, term: str, source_language: str, translations: Dict[str, str], domain: str) -> None:
        """
        Add a technical term to the vocabulary.
        
        Args:
            term: Source term
            source_language: Source language code
            translations: Dictionary of translations by language code
            domain: Domain for the term
        """
        # Ensure domain exists
        if domain not in self.vocabulary:
            self.vocabulary[domain] = {}
        
        # Ensure source language exists
        if source_language not in self.vocabulary[domain]:
            self.vocabulary[domain][source_language] = {}
        
        # Add the term with its translations
        self.vocabulary[domain][source_language][term] = translations
    
    def save_vocabulary(self, domain: str = None) -> bool:
        """
        Save vocabulary to file(s).
        
        Args:
            domain: Optional domain to save, if None save all domains
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not os.path.exists(self.vocabulary_dir):
            try:
                os.makedirs(self.vocabulary_dir)
            except Exception as e:
                print(f"Error creating vocabulary directory: {e}")
                return False
        
        try:
            if domain:
                # Save specific domain
                if domain in self.vocabulary:
                    filepath = os.path.join(self.vocabulary_dir, f"{domain}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(self.vocabulary[domain], f, ensure_ascii=False, indent=2)
            else:
                # Save all domains
                for domain, vocab in self.vocabulary.items():
                    filepath = os.path.join(self.vocabulary_dir, f"{domain}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(vocab, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving vocabulary: {e}")
            return False


if __name__ == "__main__":
    # Example usage of TerminologyManager
    term_manager = TerminologyManager()
    
    # Add some example terms
    term_manager.add_term(TermDefinition(
        source_term="artificial intelligence",
        translations={"fr": "intelligence artificielle", "es": "inteligencia artificial", "de": "künstliche Intelligenz"},
        domain="tech",
        description="The simulation of human intelligence in machines",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="machine learning",
        translations={"fr": "apprentissage automatique", "es": "aprendizaje automático", "de": "maschinelles Lernen"},
        domain="tech",
        description="Subset of AI focused on learning from data",
        part_of_speech="noun"
    ))
    
    # Process a text
    text = "Artificial intelligence and machine learning are transforming industries."
    processed_text, terms = term_manager.process_text(text, "en", "fr")
    
    print("Original text:", text)
    print("\nDetected terminology:")
    for term in terms:
        print(f"  - {term['term']} → {term['translation']} (domain: {term['domain']})")
    
    # Example of NamedEntityRecognizer
    ner = NamedEntityRecognizer()
    
    # Add some custom entities
    ner.add_custom_entity("Google", "ORG")
    ner.add_custom_entity("New York", "GPE")
    
    # Process a text with entities
    entity_text = "Google has offices in New York and many other cities."
    processed_entity_text, entities = ner.process_text(entity_text, "en")
    
    print("\nOriginal text:", entity_text)
    print("\nDetected entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")
        
    # Example of TechnicalVocabulary
    tech_vocab = TechnicalVocabulary()
    
    # Add some terms
    tech_vocab.add_term(
        term="neural network",
        source_language="en",
        translations={"fr": "réseau de neurones", "de": "neuronales Netzwerk", "es": "red neuronal"},
        domain="ai"
    )
    
    # Get translation
    translation = tech_vocab.get_translation("neural network", "en", "fr", "ai")
    print("\nTechnical term translation:")
    print(f"  - neural network (en) → {translation} (fr)")
    
    # Save vocabulary
    tech_vocab.save_vocabulary("ai") 