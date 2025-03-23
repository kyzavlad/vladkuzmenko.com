#!/usr/bin/env python3
"""
Language Manager Module

This module provides functionality for managing different languages, dialects,
and their specific properties relevant to translation.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class DialectVariant:
    """Representation of a dialect variant of a language."""
    code: str  # ISO code (e.g., 'en-US')
    name: str  # Human-readable name (e.g., 'American English')
    region: str  # Geographic region (e.g., 'North America')
    properties: Dict = field(default_factory=dict)  # Specific properties of this dialect
    
    def __post_init__(self):
        """Initialize default properties if not provided."""
        default_props = {
            "use_diacritics": True,
            "script": "latin",
            "text_direction": "ltr",  # left-to-right
            "uses_capitalization": True,
            "default_for_language": False
        }
        
        for key, value in default_props.items():
            if key not in self.properties:
                self.properties[key] = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "name": self.name,
            "region": self.region,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DialectVariant':
        """Create instance from dictionary data."""
        return cls(
            code=data.get("code", ""),
            name=data.get("name", ""),
            region=data.get("region", ""),
            properties=data.get("properties", {})
        )


@dataclass
class LanguageProfile:
    """Complete profile of a language with its dialects and properties."""
    code: str  # ISO 639-1 code (e.g., 'en')
    name: str  # Human-readable name (e.g., 'English')
    native_name: str  # Name in the language itself (e.g., 'English')
    family: str  # Language family (e.g., 'Indo-European')
    dialects: List[DialectVariant] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default properties if not provided."""
        default_props = {
            "has_gendered_nouns": False,
            "has_formal_address": False,
            "has_complex_conjugation": False,
            "has_tone_system": False,
            "default_dialect": None
        }
        
        for key, value in default_props.items():
            if key not in self.properties:
                self.properties[key] = value
        
        # Set the default dialect if one is marked as default
        for dialect in self.dialects:
            if dialect.properties.get("default_for_language", False):
                self.properties["default_dialect"] = dialect.code
                break
        
        # If no default dialect is marked, use the first one
        if not self.properties["default_dialect"] and self.dialects:
            self.properties["default_dialect"] = self.dialects[0].code
    
    def get_default_dialect(self) -> Optional[DialectVariant]:
        """Get the default dialect for this language."""
        default_code = self.properties.get("default_dialect")
        if default_code:
            for dialect in self.dialects:
                if dialect.code == default_code:
                    return dialect
        
        # If no default is set, return the first dialect if available
        return self.dialects[0] if self.dialects else None
    
    def get_dialect(self, dialect_code: str) -> Optional[DialectVariant]:
        """Get a specific dialect by code."""
        for dialect in self.dialects:
            if dialect.code == dialect_code:
                return dialect
        return None
    
    def add_dialect(self, dialect: DialectVariant) -> None:
        """Add a dialect to this language profile."""
        # Check if dialect already exists
        for i, existing in enumerate(self.dialects):
            if existing.code == dialect.code:
                # Replace existing dialect
                self.dialects[i] = dialect
                return
        
        # Add new dialect
        self.dialects.append(dialect)
        
        # If this is the first dialect, set it as default
        if len(self.dialects) == 1:
            self.properties["default_dialect"] = dialect.code
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "name": self.name,
            "native_name": self.native_name,
            "family": self.family,
            "dialects": [d.to_dict() for d in self.dialects],
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LanguageProfile':
        """Create instance from dictionary data."""
        dialects = [DialectVariant.from_dict(d) for d in data.get("dialects", [])]
        return cls(
            code=data.get("code", ""),
            name=data.get("name", ""),
            native_name=data.get("native_name", ""),
            family=data.get("family", ""),
            dialects=dialects,
            properties=data.get("properties", {})
        )


class LanguageManager:
    """
    Manager for language profiles and dialect information.
    Provides functionality for language-specific customization of translations.
    """
    
    def __init__(self, config_dir: str = "config/languages"):
        """
        Initialize the language manager.
        
        Args:
            config_dir: Directory containing language configuration files
        """
        self.config_dir = config_dir
        self.languages: Dict[str, LanguageProfile] = {}
        self.dialects: Dict[str, DialectVariant] = {}
        
        # Load predefined language profiles
        self._load_predefined_languages()
    
    def _load_predefined_languages(self) -> None:
        """
        Load predefined language profiles from configuration files or internal definitions.
        """
        # Load language profiles from config files if available
        if os.path.exists(self.config_dir):
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.config_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            language_data = json.load(f)
                            self._add_language_from_data(language_data)
                    except Exception as e:
                        print(f"Error loading language profile from {filepath}: {e}")
        
        # If no languages loaded, initialize with basic predefined set
        if not self.languages:
            self._initialize_default_languages()
    
    def _initialize_default_languages(self) -> None:
        """
        Initialize with a set of default language profiles.
        This is used when no configuration files are available.
        """
        # Define a few key languages with their dialects
        # English
        english = LanguageProfile(
            code="en",
            name="English",
            native_name="English",
            family="Indo-European",
            properties={
                "has_gendered_nouns": False,
                "has_formal_address": False,
                "has_complex_conjugation": False,
                "has_tone_system": False
            }
        )
        
        english.add_dialect(DialectVariant(
            code="en-US",
            name="American English",
            region="North America",
            properties={"default_for_language": True}
        ))
        
        english.add_dialect(DialectVariant(
            code="en-GB",
            name="British English",
            region="Europe"
        ))
        
        english.add_dialect(DialectVariant(
            code="en-AU",
            name="Australian English",
            region="Oceania"
        ))
        
        english.add_dialect(DialectVariant(
            code="en-CA",
            name="Canadian English",
            region="North America"
        ))
        
        # Spanish
        spanish = LanguageProfile(
            code="es",
            name="Spanish",
            native_name="Español",
            family="Indo-European",
            properties={
                "has_gendered_nouns": True,
                "has_formal_address": True,
                "has_complex_conjugation": True,
                "has_tone_system": False
            }
        )
        
        spanish.add_dialect(DialectVariant(
            code="es-ES",
            name="Castilian Spanish",
            region="Europe",
            properties={"default_for_language": True}
        ))
        
        spanish.add_dialect(DialectVariant(
            code="es-MX",
            name="Mexican Spanish",
            region="North America"
        ))
        
        spanish.add_dialect(DialectVariant(
            code="es-AR",
            name="Argentine Spanish",
            region="South America"
        ))
        
        # Chinese
        chinese = LanguageProfile(
            code="zh",
            name="Chinese",
            native_name="中文",
            family="Sino-Tibetan",
            properties={
                "has_gendered_nouns": False,
                "has_formal_address": True,
                "has_complex_conjugation": False,
                "has_tone_system": True
            }
        )
        
        chinese.add_dialect(DialectVariant(
            code="zh-CN",
            name="Mandarin (Simplified)",
            region="Asia",
            properties={
                "script": "hans",  # Simplified Chinese
                "default_for_language": True
            }
        ))
        
        chinese.add_dialect(DialectVariant(
            code="zh-TW",
            name="Mandarin (Traditional)",
            region="Asia",
            properties={
                "script": "hant"  # Traditional Chinese
            }
        ))
        
        # Add these languages to the manager
        self.add_language(english)
        self.add_language(spanish)
        self.add_language(chinese)
        
        # Additional languages would be added similarly
    
    def _add_language_from_data(self, data: Dict) -> None:
        """
        Add a language profile from dictionary data.
        
        Args:
            data: Dictionary with language data
        """
        try:
            profile = LanguageProfile.from_dict(data)
            self.add_language(profile)
        except Exception as e:
            print(f"Error adding language from data: {e}")
    
    def add_language(self, language: LanguageProfile) -> None:
        """
        Add a language profile to the manager.
        
        Args:
            language: Language profile to add
        """
        self.languages[language.code] = language
        
        # Also add all dialects to the dialect lookup
        for dialect in language.dialects:
            self.dialects[dialect.code] = dialect
    
    def get_language(self, code: str) -> Optional[LanguageProfile]:
        """
        Get a language profile by code.
        
        Args:
            code: ISO 639-1 language code
            
        Returns:
            Language profile or None if not found
        """
        return self.languages.get(code)
    
    def get_dialect(self, code: str) -> Optional[DialectVariant]:
        """
        Get a dialect by code.
        
        Args:
            code: Dialect code (e.g., 'en-US')
            
        Returns:
            Dialect variant or None if not found
        """
        return self.dialects.get(code)
    
    def get_language_from_dialect(self, dialect_code: str) -> Optional[LanguageProfile]:
        """
        Get the language profile for a given dialect.
        
        Args:
            dialect_code: Dialect code (e.g., 'en-US')
            
        Returns:
            Language profile or None if not found
        """
        if '-' not in dialect_code:
            return self.get_language(dialect_code)
        
        language_code = dialect_code.split('-')[0]
        return self.get_language(language_code)
    
    def get_all_languages(self) -> List[LanguageProfile]:
        """
        Get all language profiles.
        
        Returns:
            List of all language profiles
        """
        return list(self.languages.values())
    
    def get_all_dialects(self) -> List[DialectVariant]:
        """
        Get all dialect variants.
        
        Returns:
            List of all dialect variants
        """
        return list(self.dialects.values())
    
    def save_language_profile(self, language: LanguageProfile) -> bool:
        """
        Save a language profile to configuration file.
        
        Args:
            language: Language profile to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir)
            except Exception as e:
                print(f"Error creating language config directory: {e}")
                return False
        
        filename = f"{language.code}.json"
        filepath = os.path.join(self.config_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(language.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving language profile to {filepath}: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    manager = LanguageManager()
    
    # Get all languages
    languages = manager.get_all_languages()
    print(f"Loaded {len(languages)} language profiles")
    
    # Print some language details
    for language in languages:
        print(f"\n{language.name} ({language.native_name}, {language.code}):")
        print(f"  Family: {language.family}")
        print(f"  Dialects: {len(language.dialects)}")
        
        # Print dialect information
        for dialect in language.dialects:
            print(f"    - {dialect.name} ({dialect.code}): {dialect.region}")
            
        # Show additional properties
        print("  Properties:")
        for key, value in language.properties.items():
            print(f"    - {key}: {value}")
        
    # Example of getting a specific language and dialect
    en = manager.get_language("en")
    if en:
        print("\nEnglish language details:")
        print(f"  Default dialect: {en.get_default_dialect().name if en.get_default_dialect() else 'None'}")
        
        en_us = en.get_dialect("en-US")
        if en_us:
            print(f"  en-US dialect properties: {en_us.properties}")
    
    # Example of adding a new dialect to an existing language
    if en:
        en.add_dialect(DialectVariant(
            code="en-IN",
            name="Indian English",
            region="Asia"
        ))
        print(f"\nAdded Indian English dialect to {en.name}")
        print(f"  Total dialects: {len(en.dialects)}") 