"""
Text processing utilities for content analysis and enhancement.
"""
import re
import string
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from textstat import flesch_reading_ease, flesch_kincaid_grade
from transformers import pipeline

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """Advanced text processing for content analysis."""
    
    def __init__(self):
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize NLP pipeline for summarization
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
        except Exception as e:
            logger.warning(f"Could not initialize summarizer: {e}")
            self.summarizer = None
        
        # Technical terms specific to MOSDAC/satellite domain
        self.technical_terms = {
            'satellites': [
                'insat', 'kalpana', 'megha-tropiques', 'scatsat', 'oceansat',
                'cartosat', 'resourcesat', 'risat', 'astrosat', 'chandrayaan',
                'mangalyaan', 'aditya', 'isro', 'gslv', 'pslv'
            ],
            'instruments': [
                'vhrr', 'ccd', 'liss', 'awi', 'msmr', 'madras', 'saral',
                'altika', 'scatterometer', 'radiometer', 'imager', 'sounder',
                'payload', 'sensor', 'detector', 'spectroradiometer'
            ],
            'parameters': [
                'sst', 'chlorophyll', 'aerosol', 'precipitation', 'humidity',
                'temperature', 'pressure', 'wind', 'ozone', 'radiation',
                'reflectance', 'brightness', 'normalized', 'index', 'anomaly'
            ],
            'applications': [
                'meteorology', 'oceanography', 'agriculture', 'forestry',
                'disaster', 'cyclone', 'drought', 'flood', 'tsunami',
                'climate', 'weather', 'monsoon', 'forecast', 'monitoring'
            ]
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'omw-1.4'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download NLTK data {data}: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        try:
            sentences = sent_tokenize(text)
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except Exception as e:
            logger.error(f"Error extracting sentences: {e}")
            return [text]
    
    def extract_keywords(
        self,
        text: str,
        max_keywords: int = 20,
        min_length: int = 3
    ) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            min_length: Minimum keyword length
            
        Returns:
            List of keywords
        """
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            
            # Filter tokens
            filtered_tokens = [
                token for token in tokens
                if (token not in self.stop_words and
                    token not in string.punctuation and
                    len(token) >= min_length and
                    token.isalpha())
            ]
            
            # Lemmatize
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(token) for token in filtered_tokens
            ]
            
            # Count frequency
            word_freq = Counter(lemmatized_tokens)
            
            # Get most common keywords
            keywords = [word for word, freq in word_freq.most_common(max_keywords)]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and entities
        """
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            entities = ne_chunk(pos_tags)
            
            # Extract entities by type
            entity_dict = {}
            current_entity = []
            current_label = None
            
            for item in entities:
                if hasattr(item, 'label'):
                    if current_label:
                        # Save previous entity
                        entity_text = ' '.join([token for token, pos in current_entity])
                        if current_label not in entity_dict:
                            entity_dict[current_label] = []
                        entity_dict[current_label].append(entity_text)
                    
                    # Start new entity
                    current_label = item.label()
                    current_entity = [item[0]]
                else:
                    if current_label:
                        current_entity.append(item)
                    else:
                        # Save any remaining entity
                        if current_label and current_entity:
                            entity_text = ' '.join([token for token, pos in current_entity])
                            if current_label not in entity_dict:
                                entity_dict[current_label] = []
                            entity_dict[current_label].append(entity_text)
                        current_label = None
                        current_entity = []
            
            # Save final entity
            if current_label and current_entity:
                entity_text = ' '.join([token for token, pos in current_entity])
                if current_label not in entity_dict:
                    entity_dict[current_label] = []
                entity_dict[current_label].append(entity_text)
            
            return entity_dict
            
        except Exception as e:
            logger.error(f"Error extracting named entities: {e}")
            return {}
    
    def extract_technical_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Extract technical terms specific to MOSDAC/satellite domain.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of technical term categories
        """
        text_lower = text.lower()
        found_terms = {}
        
        for category, terms in self.technical_terms.items():
            found_in_category = []
            
            for term in terms:
                # Look for exact term matches
                if term.lower() in text_lower:
                    found_in_category.append(term)
                
                # Look for term variations (plurals, etc.)
                term_variations = [
                    term + 's',
                    term + 'es',
                    term.replace('-', ' '),
                    term.replace(' ', '-'),
                ]
                
                for variation in term_variations:
                    if variation.lower() in text_lower:
                        found_in_category.append(variation)
            
            if found_in_category:
                found_terms[category] = list(set(found_in_category))
        
        return found_terms
    
    def extract_satellite_info(self, text: str) -> Dict[str, Any]:
        """
        Extract satellite-specific information from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of satellite information
        """
        satellite_info = {
            'satellites_mentioned': [],
            'instruments_mentioned': [],
            'parameters_mentioned': [],
            'launch_dates': [],
            'orbit_info': [],
            'mission_info': []
        }
        
        text_lower = text.lower()
        
        # Extract satellite names
        satellite_patterns = [
            r'insat[-\s]?(\d+[a-z]?)',
            r'kalpana[-\s]?(\d+)?',
            r'megha[-\s]?tropiques',
            r'scatsat[-\s]?(\d+)?',
            r'oceansat[-\s]?(\d+)?',
            r'cartosat[-\s]?(\d+[a-z]?)?',
            r'resourcesat[-\s]?(\d+[a-z]?)?',
            r'risat[-\s]?(\d+[a-z]?)?',
        ]
        
        for pattern in satellite_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                satellite_info['satellites_mentioned'].extend(matches)
        
        # Extract launch dates
        date_patterns = [
            r'launch(?:ed)?\s+(?:on\s+)?(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'launch(?:ed)?\s+(?:on\s+)?(\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})\s+launch',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                satellite_info['launch_dates'].extend(matches)
        
        # Extract orbit information
        orbit_patterns = [
            r'(geostationary|polar|sun[-\s]?synchronous|elliptical)\s+orbit',
            r'orbit(?:al)?\s+(?:period|altitude|inclination)',
            r'(\d+)\s*km\s+(?:altitude|orbit)',
        ]
        
        for pattern in orbit_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                satellite_info['orbit_info'].extend(matches)
        
        # Extract mission information
        mission_patterns = [
            r'mission\s+(?:life|duration)?\s*:?\s*(\d+)\s*(?:years?|months?)',
            r'(?:primary|secondary)\s+mission',
            r'mission\s+(?:objectives?|goals?)',
        ]
        
        for pattern in mission_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                satellite_info['mission_info'].extend(matches)
        
        # Extract technical terms
        technical_terms = self.extract_technical_terms(text)
        if 'instruments' in technical_terms:
            satellite_info['instruments_mentioned'] = technical_terms['instruments']
        if 'parameters' in technical_terms:
            satellite_info['parameters_mentioned'] = technical_terms['parameters']
        
        return satellite_info
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate text readability metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability scores
        """
        try:
            readability_scores = {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text),
                'word_count': len(word_tokenize(text)),
                'sentence_count': len(sent_tokenize(text)),
                'avg_sentence_length': 0,
                'avg_word_length': 0,
            }
            
            # Calculate average sentence length
            sentences = sent_tokenize(text)
            if sentences:
                total_words = len(word_tokenize(text))
                readability_scores['avg_sentence_length'] = total_words / len(sentences)
            
            # Calculate average word length
            words = word_tokenize(text)
            if words:
                total_chars = sum(len(word) for word in words if word.isalpha())
                alpha_words = [word for word in words if word.isalpha()]
                if alpha_words:
                    readability_scores['avg_word_length'] = total_chars / len(alpha_words)
            
            return readability_scores
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {}
    
    def summarize_text(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50
    ) -> str:
        """
        Generate text summary.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Text summary
        """
        try:
            if not self.summarizer:
                # Fallback to extractive summarization
                sentences = sent_tokenize(text)
                if len(sentences) <= 3:
                    return text
                
                # Simple extractive summary - take first and last sentences
                summary = sentences[0] + " " + sentences[-1]
                return summary[:max_length]
            
            # Use transformers summarization
            if len(text) < min_length:
                return text
            
            # Chunk text if too long
            max_chunk_size = 1000
            if len(text) > max_chunk_size:
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                summaries = []
                
                for chunk in chunks:
                    try:
                        summary = self.summarizer(
                            chunk,
                            max_length=max_length//len(chunks),
                            min_length=min_length//len(chunks),
                            do_sample=False
                        )
                        summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        logger.warning(f"Error summarizing chunk: {e}")
                        continue
                
                return " ".join(summaries)
            else:
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return summary[0]['summary_text']
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to first sentence
            sentences = sent_tokenize(text)
            return sentences[0] if sentences else text[:max_length]
    
    def extract_questions_and_answers(self, text: str) -> List[Dict[str, str]]:
        """
        Extract question-answer pairs from text.
        
        Args:
            text: Input text
            
        Returns:
            List of question-answer pairs
        """
        qa_pairs = []
        
        # Pattern for explicit Q&A format
        qa_pattern = r'Q\s*:?\s*(.+?)\s*A\s*:?\s*(.+?)(?=Q\s*:?|$)'
        matches = re.findall(qa_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for question, answer in matches:
            qa_pairs.append({
                'question': question.strip(),
                'answer': answer.strip()
            })
        
        # Pattern for FAQ format
        faq_pattern = r'(\d+\.\s*)?(.+\?)\s*\n\s*(.+?)(?=\n\s*\d+\.|$)'
        faq_matches = re.findall(faq_pattern, text, re.DOTALL)
        
        for _, question, answer in faq_matches:
            if question.strip() and answer.strip():
                qa_pairs.append({
                    'question': question.strip(),
                    'answer': answer.strip()
                })
        
        return qa_pairs
    
    def detect_language(self, text: str) -> str:
        """
        Detect text language.
        
        Args:
            text: Input text
            
        Returns:
            Language code
        """
        try:
            # Simple language detection based on common patterns
            # This is a basic implementation - for production use langdetect
            
            # Check for common English words
            english_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'her', 'was', 'one', 'our', 'had', 'but', 'words'
            }
            
            # Check for common Hindi words (in English script)
            hindi_words = {
                'aur', 'hai', 'hain', 'ke', 'ki', 'ka', 'se', 'me',
                'ko', 'par', 'tha', 'thi', 'the', 'kya', 'kaise', 'kahan'
            }
            
            words = word_tokenize(text.lower())
            english_count = sum(1 for word in words if word in english_words)
            hindi_count = sum(1 for word in words if word in hindi_words)
            
            if english_count > hindi_count:
                return 'en'
            elif hindi_count > 0:
                return 'hi'
            else:
                return 'en'  # Default to English
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'en'
    
    def calculate_content_quality(self, text: str) -> float:
        """
        Calculate content quality score.
        
        Args:
            text: Input text
            
        Returns:
            Quality score (0-1)
        """
        try:
            score = 0.0
            
            # Length factor (optimal around 500-2000 words)
            word_count = len(word_tokenize(text))
            if 100 <= word_count <= 3000:
                length_score = min(word_count / 1000, 1.0)
            else:
                length_score = 0.5
            
            score += length_score * 0.2
            
            # Readability factor
            readability = self.calculate_readability(text)
            flesch_score = readability.get('flesch_reading_ease', 50)
            readability_score = min(flesch_score / 100, 1.0)
            score += readability_score * 0.2
            
            # Technical content factor
            technical_terms = self.extract_technical_terms(text)
            technical_score = min(len(technical_terms) * 0.1, 1.0)
            score += technical_score * 0.3
            
            # Structure factor (presence of sentences, paragraphs)
            sentence_count = len(sent_tokenize(text))
            structure_score = min(sentence_count / 20, 1.0)
            score += structure_score * 0.15
            
            # Information density factor
            unique_words = len(set(word_tokenize(text.lower())))
            total_words = len(word_tokenize(text))
            density_score = unique_words / total_words if total_words > 0 else 0
            score += density_score * 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating content quality: {e}")
            return 0.5
    
    def create_text_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping text chunks.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        sentence_start = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            if current_size + sentence_length > chunk_size and current_chunk:
                # Create chunk
                chunk_info = {
                    'text': current_chunk.strip(),
                    'start_sentence': sentence_start,
                    'end_sentence': i - 1,
                    'word_count': len(word_tokenize(current_chunk)),
                    'char_count': len(current_chunk),
                }
                chunks.append(chunk_info)
                
                # Start new chunk with overlap
                overlap_sentences = sentences[max(0, i - overlap//100):i]
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_size = len(current_chunk)
                sentence_start = max(0, i - overlap//100)
            else:
                current_chunk += " " + sentence
                current_size += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunk_info = {
                'text': current_chunk.strip(),
                'start_sentence': sentence_start,
                'end_sentence': len(sentences) - 1,
                'word_count': len(word_tokenize(current_chunk)),
                'char_count': len(current_chunk),
            }
            chunks.append(chunk_info)
        
        return chunks
