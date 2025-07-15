"""
Helper utilities and common functions.
"""
import hashlib
import json
import re
import string
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, urljoin
from uuid import UUID

import bleach
from bs4 import BeautifulSoup

from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_hash(text: str, algorithm: str = "md5") -> str:
    """
    Generate hash for given text.
    
    Args:
        text: Input text
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        str: Generated hash
    """
    try:
        if algorithm == "md5":
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(text.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        logger.error(f"Error generating hash: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    try:
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text


def sanitize_html(html: str, allowed_tags: Optional[List[str]] = None) -> str:
    """
    Sanitize HTML content.
    
    Args:
        html: Input HTML
        allowed_tags: List of allowed HTML tags
        
    Returns:
        str: Sanitized HTML
    """
    try:
        if allowed_tags is None:
            allowed_tags = [
                'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
                'code', 'pre', 'a', 'img'
            ]
        
        allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title'],
            '*': ['class', 'id']
        }
        
        return bleach.clean(
            html,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
    except Exception as e:
        logger.error(f"Error sanitizing HTML: {e}")
        return html


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        List[str]: List of extracted URLs
    """
    try:
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.findall(text)
        
    except Exception as e:
        logger.error(f"Error extracting URLs: {e}")
        return []


def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid
    """
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    except Exception:
        return False


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Normalize URL.
    
    Args:
        url: URL to normalize
        base_url: Base URL for relative URLs
        
    Returns:
        str: Normalized URL
    """
    try:
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        parsed = urlparse(url)
        
        # Remove fragment
        normalized = parsed._replace(fragment='').geturl()
        
        # Remove trailing slash
        if normalized.endswith('/') and normalized.count('/') > 2:
            normalized = normalized[:-1]
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing URL: {e}")
        return url


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: Input URL
        
    Returns:
        str: Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception as e:
        logger.error(f"Error extracting domain: {e}")
        return ""


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append
        
    Returns:
        str: Truncated text
    """
    try:
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
        
    except Exception as e:
        logger.error(f"Error truncating text: {e}")
        return text


def generate_slug(text: str, max_length: int = 50) -> str:
    """
    Generate URL-friendly slug from text.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        str: Generated slug
    """
    try:
        # Convert to lowercase and normalize
        slug = text.lower().strip()
        
        # Remove special characters and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        
        # Truncate to max length
        slug = slug[:max_length].strip('-')
        
        return slug
        
    except Exception as e:
        logger.error(f"Error generating slug: {e}")
        return ""


def parse_json_safely(json_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON string safely.
    
    Args:
        json_string: JSON string to parse
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON or None
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error parsing JSON: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size
    """
    try:
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
        
    except Exception as e:
        logger.error(f"Error formatting file size: {e}")
        return "Unknown"


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename
        
    Returns:
        str: File extension
    """
    try:
        return filename.split('.')[-1].lower() if '.' in filename else ""
    except Exception as e:
        logger.error(f"Error getting file extension: {e}")
        return ""


def is_uuid(value: str) -> bool:
    """
    Check if string is a valid UUID.
    
    Args:
        value: String to check
        
    Returns:
        bool: True if valid UUID
    """
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        str: Formatted datetime
    """
    try:
        return dt.strftime(format_str)
    except Exception as e:
        logger.error(f"Error formatting datetime: {e}")
        return ""


def parse_datetime(date_string: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse datetime from string.
    
    Args:
        date_string: Date string
        format_str: Format string
        
    Returns:
        Optional[datetime]: Parsed datetime or None
    """
    try:
        return datetime.strptime(date_string, format_str)
    except ValueError as e:
        logger.error(f"Error parsing datetime: {e}")
        return None


def utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        datetime: Current UTC datetime
    """
    return datetime.now(timezone.utc)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: Input list
        chunk_size: Chunk size
        
    Returns:
        List[List[Any]]: List of chunks
    """
    try:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    except Exception as e:
        logger.error(f"Error chunking list: {e}")
        return [lst]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Input dictionary
        parent_key: Parent key prefix
        sep: Separator
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    try:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    except Exception as e:
        logger.error(f"Error flattening dictionary: {e}")
        return d


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    try:
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
        
    except Exception as e:
        logger.error(f"Error merging dictionaries: {e}")
        return dict1


def remove_duplicates(lst: List[Any], key: Optional[str] = None) -> List[Any]:
    """
    Remove duplicates from list.
    
    Args:
        lst: Input list
        key: Key function for complex objects
        
    Returns:
        List[Any]: List without duplicates
    """
    try:
        if key:
            seen = set()
            result = []
            for item in lst:
                item_key = getattr(item, key) if hasattr(item, key) else item.get(key)
                if item_key not in seen:
                    seen.add(item_key)
                    result.append(item)
            return result
        else:
            return list(dict.fromkeys(lst))
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        return lst


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        
    Returns:
        List[str]: List of keywords
    """
    try:
        # Simple keyword extraction
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        
        # Split into words
        words = text.split()
        
        # Filter common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'this', 'these', 'those',
            'they', 'them', 'their', 'there', 'then', 'than', 'or', 'but',
            'not', 'have', 'had', 'do', 'does', 'did', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'will'
        }
        
        # Extract keywords
        keywords = []
        for word in words:
            if len(word) >= min_length and word not in stop_words:
                keywords.append(word)
        
        # Remove duplicates and return
        return list(set(keywords))
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard index.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score (0-1)
    """
    try:
        # Extract keywords from both texts
        keywords1 = set(extract_keywords(text1))
        keywords2 = set(extract_keywords(text2))
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
        
    except Exception as e:
        logger.error(f"Error calculating text similarity: {e}")
        return 0.0


def mask_sensitive_data(text: str, patterns: Optional[List[str]] = None) -> str:
    """
    Mask sensitive data in text.
    
    Args:
        text: Input text
        patterns: List of regex patterns to mask
        
    Returns:
        str: Text with masked sensitive data
    """
    try:
        if patterns is None:
            patterns = [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{10}\b',  # Phone number
            ]
        
        masked_text = text
        for pattern in patterns:
            masked_text = re.sub(pattern, '[MASKED]', masked_text)
        
        return masked_text
        
    except Exception as e:
        logger.error(f"Error masking sensitive data: {e}")
        return text
