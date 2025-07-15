"""
Validation utilities for data validation and sanitization.
"""
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from datetime import datetime

from pydantic import BaseModel, ValidationError, validator
from email_validator import validate_email, EmailNotValidError

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Validation result container."""
    
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def __bool__(self):
        return self.is_valid
    
    def __repr__(self):
        return f"ValidationResult(is_valid={self.is_valid}, errors={self.errors})"


def validate_email_address(email: str) -> ValidationResult:
    """
    Validate email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        validate_email(email)
        return ValidationResult(True)
    except EmailNotValidError as e:
        return ValidationResult(False, [str(e)])


def validate_url(url: str) -> ValidationResult:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            return ValidationResult(False, ["URL must have a scheme (http/https)"])
        
        if not parsed.netloc:
            return ValidationResult(False, ["URL must have a valid domain"])
        
        if parsed.scheme not in ['http', 'https']:
            return ValidationResult(False, ["URL scheme must be http or https"])
        
        return ValidationResult(True)
        
    except Exception as e:
        return ValidationResult(False, [f"Invalid URL format: {str(e)}"])


def validate_phone_number(phone: str, country_code: Optional[str] = None) -> ValidationResult:
    """
    Validate phone number.
    
    Args:
        phone: Phone number to validate
        country_code: Optional country code
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        # Remove common formatting characters
        cleaned_phone = re.sub(r'[^\d+]', '', phone)
        
        # Basic validation patterns
        patterns = [
            r'^\+?1?[2-9]\d{9}$',  # US format
            r'^\+?[1-9]\d{1,14}$',  # International format
            r'^\d{10}$',  # Simple 10-digit
        ]
        
        for pattern in patterns:
            if re.match(pattern, cleaned_phone):
                return ValidationResult(True)
        
        return ValidationResult(False, ["Invalid phone number format"])
        
    except Exception as e:
        return ValidationResult(False, [f"Phone validation error: {str(e)}"])


def validate_password(password: str, min_length: int = 8) -> ValidationResult:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        min_length: Minimum password length
        
    Returns:
        ValidationResult: Validation result
    """
    errors = []
    
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters long")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    
    if re.search(r'\s', password):
        errors.append("Password cannot contain spaces")
    
    return ValidationResult(len(errors) == 0, errors)


def validate_file_type(filename: str, allowed_types: List[str]) -> ValidationResult:
    """
    Validate file type by extension.
    
    Args:
        filename: Filename to validate
        allowed_types: List of allowed file extensions
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        if not filename:
            return ValidationResult(False, ["Filename is required"])
        
        extension = filename.split('.')[-1].lower() if '.' in filename else ""
        
        if not extension:
            return ValidationResult(False, ["File must have an extension"])
        
        if extension not in [ext.lower() for ext in allowed_types]:
            return ValidationResult(
                False,
                [f"File type '{extension}' not allowed. Allowed types: {', '.join(allowed_types)}"]
            )
        
        return ValidationResult(True)
        
    except Exception as e:
        return ValidationResult(False, [f"File type validation error: {str(e)}"])


def validate_file_size(file_size: int, max_size: int) -> ValidationResult:
    """
    Validate file size.
    
    Args:
        file_size: File size in bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        if file_size <= 0:
            return ValidationResult(False, ["File size must be greater than 0"])
        
        if file_size > max_size:
            return ValidationResult(
                False,
                [f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)"]
            )
        
        return ValidationResult(True)
        
    except Exception as e:
        return ValidationResult(False, [f"File size validation error: {str(e)}"])


def validate_json(json_string: str) -> ValidationResult:
    """
    Validate JSON string.
    
    Args:
        json_string: JSON string to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        import json
        json.loads(json_string)
        return ValidationResult(True)
    except json.JSONDecodeError as e:
        return ValidationResult(False, [f"Invalid JSON: {str(e)}"])


def validate_date_range(start_date: datetime, end_date: datetime) -> ValidationResult:
    """
    Validate date range.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        if start_date >= end_date:
            return ValidationResult(False, ["Start date must be before end date"])
        
        # Check if dates are not too far in the future
        now = datetime.utcnow()
        if start_date > now:
            return ValidationResult(False, ["Start date cannot be in the future"])
        
        return ValidationResult(True)
        
    except Exception as e:
        return ValidationResult(False, [f"Date range validation error: {str(e)}"])


def validate_search_query(query: str) -> ValidationResult:
    """
    Validate search query.
    
    Args:
        query: Search query to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        if not query or not query.strip():
            return ValidationResult(False, ["Search query cannot be empty"])
        
        if len(query.strip()) < 2:
            return ValidationResult(False, ["Search query must be at least 2 characters long"])
        
        if len(query) > 1000:
            return ValidationResult(False, ["Search query cannot exceed 1000 characters"])
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'<script',
            r'javascript:',
            r'onload=',
            r'onerror=',
            r'eval\(',
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ValidationResult(False, ["Search query contains potentially harmful content"])
        
        return ValidationResult(True)
        
    except Exception as e:
        return ValidationResult(False, [f"Search query validation error: {str(e)}"])


def validate_pagination(offset: int, limit: int, max_limit: int = 100) -> ValidationResult:
    """
    Validate pagination parameters.
    
    Args:
        offset: Offset value
        limit: Limit value
        max_limit: Maximum allowed limit
        
    Returns:
        ValidationResult: Validation result
    """
    errors = []
    
    if offset < 0:
        errors.append("Offset must be non-negative")
    
    if limit <= 0:
        errors.append("Limit must be greater than 0")
    
    if limit > max_limit:
        errors.append(f"Limit cannot exceed {max_limit}")
    
    return ValidationResult(len(errors) == 0, errors)


def validate_tags(tags: List[str]) -> ValidationResult:
    """
    Validate tags list.
    
    Args:
        tags: List of tags to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        errors = []
        
        if len(tags) > 10:
            errors.append("Cannot have more than 10 tags")
        
        for tag in tags:
            if not tag.strip():
                errors.append("Tags cannot be empty")
                continue
            
            if len(tag) > 50:
                errors.append(f"Tag '{tag}' exceeds maximum length of 50 characters")
            
            if not re.match(r'^[a-zA-Z0-9\s\-_]+$', tag):
                errors.append(f"Tag '{tag}' contains invalid characters")
        
        return ValidationResult(len(errors) == 0, errors)
        
    except Exception as e:
        return ValidationResult(False, [f"Tags validation error: {str(e)}"])


def validate_mosdac_query(query: str) -> ValidationResult:
    """
    Validate MOSDAC-specific query.
    
    Args:
        query: MOSDAC query to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        # First, run basic search query validation
        basic_validation = validate_search_query(query)
        if not basic_validation:
            return basic_validation
        
        # MOSDAC-specific validation
        errors = []
        
        # Check for reasonable length
        if len(query) > 500:
            errors.append("MOSDAC query should not exceed 500 characters")
        
        # Check for potentially problematic patterns
        problematic_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'UPDATE\s+SET',
            r'INSERT\s+INTO',
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append("Query contains potentially harmful SQL patterns")
        
        return ValidationResult(len(errors) == 0, errors)
        
    except Exception as e:
        return ValidationResult(False, [f"MOSDAC query validation error: {str(e)}"])


def validate_satellite_data_request(request_data: Dict[str, Any]) -> ValidationResult:
    """
    Validate satellite data request.
    
    Args:
        request_data: Request data to validate
        
    Returns:
        ValidationResult: Validation result
    """
    try:
        errors = []
        
        # Required fields
        required_fields = ['satellite', 'instrument', 'product_type']
        for field in required_fields:
            if field not in request_data:
                errors.append(f"Missing required field: {field}")
        
        # Date range validation
        if 'start_date' in request_data and 'end_date' in request_data:
            try:
                start_date = datetime.fromisoformat(request_data['start_date'])
                end_date = datetime.fromisoformat(request_data['end_date'])
                
                date_validation = validate_date_range(start_date, end_date)
                if not date_validation:
                    errors.extend(date_validation.errors)
                    
            except ValueError:
                errors.append("Invalid date format. Use ISO format (YYYY-MM-DD)")
        
        # Geographic bounds validation
        if 'bounds' in request_data:
            bounds = request_data['bounds']
            if not isinstance(bounds, dict):
                errors.append("Bounds must be a dictionary")
            else:
                required_bounds = ['north', 'south', 'east', 'west']
                for bound in required_bounds:
                    if bound not in bounds:
                        errors.append(f"Missing bound: {bound}")
                
                # Validate coordinate ranges
                if all(bound in bounds for bound in required_bounds):
                    if not (-90 <= bounds['north'] <= 90):
                        errors.append("North bound must be between -90 and 90")
                    if not (-90 <= bounds['south'] <= 90):
                        errors.append("South bound must be between -90 and 90")
                    if not (-180 <= bounds['east'] <= 180):
                        errors.append("East bound must be between -180 and 180")
                    if not (-180 <= bounds['west'] <= 180):
                        errors.append("West bound must be between -180 and 180")
                    
                    if bounds['north'] <= bounds['south']:
                        errors.append("North bound must be greater than south bound")
                    if bounds['east'] <= bounds['west']:
                        errors.append("East bound must be greater than west bound")
        
        return ValidationResult(len(errors) == 0, errors)
        
    except Exception as e:
        return ValidationResult(False, [f"Satellite data request validation error: {str(e)}"])


class QueryValidator:
    """Query validator with configurable rules."""
    
    def __init__(self, max_length: int = 1000, min_length: int = 2):
        self.max_length = max_length
        self.min_length = min_length
        self.forbidden_patterns = [
            r'<script',
            r'javascript:',
            r'onload=',
            r'onerror=',
            r'eval\(',
            r'document\.cookie',
            r'window\.location',
        ]
    
    def validate(self, query: str) -> ValidationResult:
        """
        Validate query with configured rules.
        
        Args:
            query: Query to validate
            
        Returns:
            ValidationResult: Validation result
        """
        errors = []
        
        # Basic checks
        if not query or not query.strip():
            errors.append("Query cannot be empty")
            return ValidationResult(False, errors)
        
        # Length checks
        if len(query.strip()) < self.min_length:
            errors.append(f"Query must be at least {self.min_length} characters long")
        
        if len(query) > self.max_length:
            errors.append(f"Query cannot exceed {self.max_length} characters")
        
        # Pattern checks
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append("Query contains forbidden content")
                break
        
        return ValidationResult(len(errors) == 0, errors)


def sanitize_user_input(user_input: str) -> str:
    """
    Sanitize user input to remove potentially harmful content.
    
    Args:
        user_input: User input to sanitize
        
    Returns:
        str: Sanitized input
    """
    try:
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]*>', '', user_input)
        
        # Remove script tags and javascript
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove event handlers
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Error sanitizing user input: {e}")
        return user_input
