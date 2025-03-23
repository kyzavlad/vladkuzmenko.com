#!/usr/bin/env python3
"""
Smart Text Breaking Example

This script demonstrates the functionality of the SmartTextBreaker class
for intelligent subtitle text breaking with various strategies and languages.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import SmartTextBreaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_lines(title, lines):
    """Helper function to print lines with formatting."""
    print(f"\n{title}:")
    print("-" * len(title))
    for i, line in enumerate(lines, 1):
        print(f"{i}: '{line}' ({len(line)} chars)")
    print(f"Total lines: {len(lines)}")

def main():
    """Run the smart text breaking examples."""
    print("Smart Text Breaking Examples")
    print("===========================")
    
    # Create different text breakers with various configurations
    basic_breaker = SmartTextBreaker(config={"breaking_strategy": "basic"})
    linguistic_breaker = SmartTextBreaker(config={"breaking_strategy": "linguistic"})
    balanced_breaker = SmartTextBreaker(config={"breaking_strategy": "balanced"})
    chinese_breaker = SmartTextBreaker(config={"language": "zh"})
    
    # Example 1: Simple sentence breaking with different strategies
    simple_text = "This is a simple sentence that needs to be broken into two lines."
    
    print_lines("Basic breaking", basic_breaker.break_text(simple_text, max_chars_per_line=30))
    print_lines("Linguistic breaking", linguistic_breaker.break_text(simple_text, max_chars_per_line=30))
    print_lines("Balanced breaking", balanced_breaker.break_text(simple_text, max_chars_per_line=30))
    
    # Example 2: Complex text with natural breaks
    complex_text = "This is a more complex text with multiple sentences. It should be broken at natural points. Linguistic analysis should help with this task."
    
    print_lines("Basic breaking (complex)", basic_breaker.break_text(complex_text, max_chars_per_line=40, max_lines=3))
    print_lines("Linguistic breaking (complex)", linguistic_breaker.break_text(complex_text, max_chars_per_line=40, max_lines=3))
    
    # Example 3: Unbalanced lines
    unbalanced_text = "Short first part. But this second part is significantly longer and needs better balancing for readability."
    
    print_lines("Without balancing", basic_breaker.break_text(unbalanced_text, max_chars_per_line=40))
    print_lines("With balancing", balanced_breaker.break_text(unbalanced_text, max_chars_per_line=40))
    
    # Example 4: Chinese text (no spaces)
    chinese_text = "这是一个中文测试文本，用来测试没有空格的语言。智能文本分割应该能够正确处理这种情况。"
    
    print_lines("Chinese text breaking", chinese_breaker.break_text(chinese_text, max_chars_per_line=15, max_lines=3))
    
    # Example 5: Reading complexity analysis
    simple_sentence = "This is a simple sentence."
    complex_sentence = "The intricate interplay of syntactic structures and semantic nuances contributes significantly to the overall comprehensibility of this deliberately complex sentence."
    
    simple_complexity = linguistic_breaker.analyze_reading_complexity(simple_sentence)
    complex_complexity = linguistic_breaker.analyze_reading_complexity(complex_sentence)
    
    print("\nReading Complexity Analysis:")
    print("--------------------------")
    print(f"Simple sentence: {simple_complexity}")
    print(f"Complex sentence: {complex_complexity}")
    
    # Example 6: Line count optimization
    short_text = "Short text."
    medium_text = "This is medium length text that could be one or two lines."
    long_text = "This is a very long piece of text that should definitely use multiple lines. It contains several sentences and should be split across multiple lines for optimal readability."
    
    print("\nLine Count Optimization:")
    print("----------------------")
    print(f"Short text: {linguistic_breaker.optimize_line_count(short_text)} lines")
    print(f"Medium text: {linguistic_breaker.optimize_line_count(medium_text)} lines")
    print(f"Long text: {linguistic_breaker.optimize_line_count(long_text, max_lines=3)} lines")

if __name__ == "__main__":
    main() 