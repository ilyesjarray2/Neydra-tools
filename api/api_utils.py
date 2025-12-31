"""
Utility Functions Module
Data Processing, Formatting, Validation, and Helper Functions
NEYDRA Platform - Enterprise Grade
Founder & CEO: Ilyes Jarray
© 2025 - All Rights Reserved
"""

import os
import re
import json
import math
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import statistics

import numpy as np
import pandas as pd
from flask import request, jsonify, g

# ===== LOGGING =====
logger = logging.getLogger(__name__)


# ===================================================================
# DATA VALIDATION UTILITIES
# ===================================================================

class ValidationError(Exception):
    """Custom validation exception"""
    pass


class DataValidator:
    """
    Validates input data and ensures data quality
    """

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format
        
        Args:  
            email:  Email address to validate
            
        Returns:  
            bool: True if valid
        """
        pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_price(price: Union[int, float, str]) -> bool:
        """
        Validate price is positive number
        
        Args: 
            price: Price value
            
        Returns:  
            bool: True if valid
        """
        try: 
            price_float = float(price)
            return price_float > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_percentage(value: Union[int, float, str]) -> bool:
        """
        Validate percentage is between 0 and 100
        
        Args:
            value: Percentage value
            
        Returns:
            bool: True if valid
        """
        try: 
            value_float = float(value)
            return 0 <= value_float <= 100
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_confidence(confidence: Union[int, float, str]) -> bool:
        """
        Validate confidence is between 0 and 1
        
        Args:  
            confidence: Confidence value
            
        Returns:
            bool: True if valid
        """
        try:
            conf_float = float(confidence)
            return 0 <= conf_float <= 1
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate API key format
        
        Args:  
            api_key: API key to validate
            
        Returns: 
            bool: True if valid
        """
        if not isinstance(api_key, str):
            return False
        # API key should be at least 20 chars, alphanumeric with underscores/hyphens
        pattern = r'^[A-Za-z0-9_-]{20,}$'
        return bool(re.match(pattern, api_key))

    @staticmethod
    def validate_date(date_string: str, format: str = '%Y-%m-%d') -> bool:
        """
        Validate date string format
        
        Args:  
            date_string: Date string
            format: Expected date format
            
        Returns: 
            bool: True if valid
        """
        try:
            datetime.strptime(date_string, format)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_json(json_string: str) -> bool:
        """
        Validate JSON string
        
        Args: 
            json_string: JSON string
            
        Returns:
            bool: True if valid
        """
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def validate_dict_keys(data: Dict, required_keys: List[str]) -> Tuple[bool, Optional[List[str]]]:
        """
        Validate dictionary contains required keys
        
        Args:  
            data: Dictionary to validate
            required_keys: List of required keys
            
        Returns: 
            Tuple: (is_valid, missing_keys)
        """
        missing_keys = [key for key in required_keys if key not in data]
        return len(missing_keys) == 0, missing_keys if missing_keys else None

    @staticmethod
    def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
        """
        Validate value is within range
        
        Args:  
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            
        Returns: 
            bool: True if valid
        """
        try:
            value_float = float(value)
            min_float = float(min_val)
            max_float = float(max_val)
            return min_float <= value_float <= max_float
        except (ValueError, TypeError):
            return False


# ===================================================================
# NUMERICAL UTILITIES
# ===================================================================

class NumberUtils:
    """
    Utilities for numerical calculations and formatting
    """

    @staticmethod
    def round_price(price: float, decimals: int = 5) -> float:
        """
        Round price to specific decimals
        
        Args:  
            price: Price value
            decimals: Number of decimal places
            
        Returns:  
            float: Rounded price
        """
        try:
            decimal_price = Decimal(str(price))
            quantize_exp = Decimal(10) ** -decimals
            return float(decimal_price.quantize(quantize_exp, rounding=ROUND_HALF_UP))
        except (ValueError, TypeError):
            return price

    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """
        Calculate percentage change
        
        Args:  
            old_value: Original value
            new_value: New value
            
        Returns: 
            float: Percentage change
        """
        try:
            if old_value == 0:
                return 0
            return ((new_value - old_value) / abs(old_value)) * 100
        except (ValueError, TypeError, ZeroDivisionError):
            return 0

    @staticmethod
    def calculate_volatility(values: List[float]) -> float:
        """
        Calculate standard deviation (volatility)
        
        Args: 
            values: List of values
            
        Returns: 
            float: Volatility (standard deviation)
        """
        try:
            if len(values) < 2:
                return 0
            return statistics.stdev(values)
        except (ValueError, statistics.StatisticsError):
            return 0

    @staticmethod
    def calculate_moving_average(values: List[float], window: int = 20) -> List[float]:
        """
        Calculate moving average
        
        Args: 
            values: List of values
            window:  Moving average window size
            
        Returns: 
            List:  Moving averages
        """
        try: 
            if len(values) < window:
                return values

            moving_avg = []
            for i in range(len(values)):
                if i < window:
                    window_values = values[:  i + 1]
                else:
                    window_values = values[i - window + 1:  i + 1]
                moving_avg.append(sum(window_values) / len(window_values))

            return moving_avg
        except (ValueError, TypeError):
            return values

    @staticmethod
    def calculate_rsi(values: List[float], period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI)
        
        Args: 
            values: List of closing prices
            period: RSI period (default 14)
            
        Returns:  
            List: RSI values
        """
        try: 
            if len(values) < period + 1:
                return [50] * len(values)  # Return neutral RSI

            deltas = np.diff(values)
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]

            avg_gain = [np.mean(gains[: period])]
            avg_loss = [np.mean(losses[:period])]

            for i in range(period, len(gains)):
                avg_gain.append((avg_gain[-1] * (period - 1) + gains[i]) / period)
                avg_loss.append((avg_loss[-1] * (period - 1) + losses[i]) / period)

            rs = []
            for i in range(len(avg_gain)):
                if avg_loss[i] == 0:
                    rs.append(100)
                else:
                    rs.append(100 - (100 / (1 + avg_gain[i] / avg_loss[i])))

            # Pad initial values
            rsi_values = [50] * period + rs

            return rsi_values[: len(values)]

        except (ValueError, TypeError, ZeroDivisionError):
            return [50] * len(values)

    @staticmethod
    def calculate_macd(values: List[float], fast:  int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]: 
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:  
            values: List of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns: 
            Tuple: (MACD, Signal, Histogram)
        """
        try:
            values_array = np.array(values, dtype=float)

            # Calculate EMAs
            ema_fast = pd.Series(values_array).ewm(span=fast).mean().values
            ema_slow = pd.Series(values_array).ewm(span=slow).mean().values

            # Calculate MACD
            macd_line = ema_fast - ema_slow

            # Calculate Signal line
            macd_series = pd.Series(macd_line)
            signal_line = macd_series.ewm(span=signal).mean().values

            # Calculate Histogram
            histogram = macd_line - signal_line

            return macd_line. tolist(), signal_line.tolist(), histogram.tolist()

        except (ValueError, TypeError):
            return [0] * len(values), [0] * len(values), [0] * len(values)

    @staticmethod
    def calculate_bollinger_bands(values:  List[float], window: int = 20, std_dev: int = 2) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Bollinger Bands
        
        Args: 
            values: List of closing prices
            window: Moving average window
            std_dev:  Standard deviation multiplier
            
        Returns: 
            Tuple: (Upper Band, Middle Band, Lower Band)
        """
        try: 
            values_series = pd.Series(values)
            
            # Middle band (SMA)
            middle_band = values_series.rolling(window=window).mean()

            # Standard deviation
            std = values_series.rolling(window=window).std()

            # Upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            return upper_band.tolist(), middle_band.tolist(), lower_band.tolist()

        except (ValueError, TypeError):
            return values, values, values

    @staticmethod
    def normalize_values(values: List[float]) -> List[float]:
        """
        Normalize values to 0-1 range
        
        Args: 
            values:  List of values
            
        Returns: 
            List: Normalized values
        """
        try: 
            values_array = np.array(values, dtype=float)
            min_val = np.min(values_array)
            max_val = np.max(values_array)

            if max_val == min_val:
                return [0.5] * len(values)

            normalized = (values_array - min_val) / (max_val - min_val)
            return normalized.tolist()

        except (ValueError, TypeError):
            return values

    @staticmethod
    def calculate_correlation(series1: List[float], series2: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient
        
        Args: 
            series1: First data series
            series2: Second data series
            
        Returns: 
            float: Correlation coefficient (-1 to 1)
        """
        try:
            if len(series1) != len(series2) or len(series1) < 2:
                return 0

            return float(np.corrcoef(series1, series2)[0, 1])

        except (ValueError, TypeError):
            return 0


# ===================================================================
# STRING & TEXT UTILITIES
# ===================================================================

class StringUtils:
    """
    Utilities for string manipulation and formatting
    """

    @staticmethod
    def slugify(text: str) -> str:
        """
        Convert text to URL-safe slug
        
        Args: 
            text: Text to slugify
            
        Returns: 
            str:  Slugified text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')

    @staticmethod
    def truncate(text: str, length: int = 100, suffix: str = '...') -> str:
        """
        Truncate text to specified length
        
        Args: 
            text: Text to truncate
            length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:  
            str: Truncated text
        """
        if len(text) <= length:
            return text
        return text[: length - len(suffix)] + suffix

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """
        Convert camelCase to snake_case
        
        Args: 
            name: CamelCase string
            
        Returns: 
            str: snake_case string
        """
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    @staticmethod
    def snake_to_camel(name: str) -> str:
        """
        Convert snake_case to camelCase
        
        Args: 
            name: snake_case string
            
        Returns: 
            str: camelCase string
        """
        components = name.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """
        Extract all numbers from text
        
        Args:  
            text: Text to search
            
        Returns: 
            List:  Extracted numbers
        """
        try:
            pattern = r'-?\d+\. ?\d*'
            numbers = re.findall(pattern, text)
            return [float(n) for n in numbers]
        except (ValueError, TypeError):
            return []

    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """
        Sanitize user input for safety
        
        Args: 
            text: User input
            max_length: Maximum length
            
        Returns: 
            str: Sanitized text
        """
        # Remove dangerous characters
        text = re.sub(r'[<>\"\'%;()&+]', '', text)
        # Limit length
        text = text[:max_length]
        # Strip whitespace
        text = text. strip()
        return text


# ===================================================================
# DATE & TIME UTILITIES
# ===================================================================

class DateTimeUtils:
    """
    Utilities for date and time operations
    """

    @staticmethod
    def get_current_timestamp() -> str:
        """
        Get current timestamp in ISO format
        
        Returns:  
            str: ISO format timestamp
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
        """
        Parse ISO format datetime string
        
        Args:  
            iso_string: ISO format string
            
        Returns:  
            datetime: Parsed datetime or None
        """
        try: 
            return datetime.fromisoformat(iso_string. replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def get_time_ago(past_datetime: datetime) -> str:
        """
        Get human-readable time difference
        
        Args: 
            past_datetime: Past datetime object
            
        Returns: 
            str: Human-readable time difference
        """
        try:
            now = datetime.now(timezone.utc) if past_datetime. tzinfo else datetime.now()
            if past_datetime.tzinfo is None:
                past_datetime = past_datetime.replace(tzinfo=timezone.utc)

            diff = now - past_datetime

            if diff.days > 365:
                return f"{diff. days // 365} year(s) ago"
            elif diff.days > 30:
                return f"{diff.days // 30} month(s) ago"
            elif diff.days > 0:
                return f"{diff.days} day(s) ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600} hour(s) ago"
            elif diff. seconds > 60:
                return f"{diff.seconds // 60} minute(s) ago"
            else:
                return "Just now"

        except (TypeError, AttributeError):
            return "Unknown"

    @staticmethod
    def get_market_hours(timezone_str: str = 'UTC') -> Dict[str, str]:
        """
        Get market trading hours
        
        Args: 
            timezone_str: Timezone name
            
        Returns: 
            Dict: Market hours info
        """
        return {
            'forex_open': '17:00 Sun',
            'forex_close': '17:00 Fri',
            'market_hours': '24 hours (Monday-Friday)',
            'timezone': 'UTC'
        }

    @staticmethod
    def is_market_open() -> bool:
        """
        Check if forex market is currently open
        
        Returns: 
            bool: True if market is open
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0 = Monday, 4 = Friday
        hour = now.hour

        # Forex market is open Monday-Friday
        if weekday >= 5:  # Saturday or Sunday
            return False

        # Forex market is typically open 17:00 Sunday to 17:00 Friday UTC
        if weekday == 0 and hour < 17:  # Monday before 17:00
            return False

        return True


# ===================================================================
# RESPONSE FORMATTING UTILITIES
# ===================================================================

class ResponseFormatter:
    """
    Utilities for formatting API responses
    """

    @staticmethod
    def success_response(data: Any, message: str = 'Success', status_code: int = 200) -> Tuple[Dict, int]:
        """
        Format success response
        
        Args: 
            data: Response data
            message: Success message
            status_code: HTTP status code
            
        Returns: 
            Tuple: (response_dict, status_code)
        """
        response = {
            'success': True,
            'message': message,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if hasattr(g, 'request_id'):
            response['request_id'] = g.request_id

        return response, status_code

    @staticmethod
    def error_response(error: str, status_code: int = 400, details: Optional[Dict] = None) -> Tuple[Dict, int]:
        """
        Format error response
        
        Args: 
            error: Error message
            status_code: HTTP status code
            details:  Additional error details
            
        Returns:  
            Tuple: (response_dict, status_code)
        """
        response = {
            'success': False,
            'error': error,
            'status_code': status_code,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if details:
            response['details'] = details

        if hasattr(g, 'request_id'):
            response['request_id'] = g.request_id

        return response, status_code

    @staticmethod
    def paginated_response(
        data: List,
        page: int = 1,
        page_size: int = 20,
        total_count: int = 0
    ) -> Dict:
        """
        Format paginated response
        
        Args:  
            data: Data list
            page: Current page
            page_size: Items per page
            total_count:  Total items
            
        Returns:  
            Dict: Paginated response
        """
        total_pages = (total_count + page_size - 1) // page_size

        return {
            'data': data,
            'pagination': {
                'page': page,
                'page_size':  page_size,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }

    @staticmethod
    def list_response(items: List, count: Optional[int] = None) -> Dict:
        """
        Format list response
        
        Args: 
            items: List of items
            count: Total count (if different from len)
            
        Returns: 
            Dict: List response
        """
        return {
            'items': items,
            'count':  count if count is not None else len(items)
        }


# ===================================================================
# DATA TRANSFORMATION UTILITIES
# ===================================================================

class DataTransformer:
    """
    Utilities for transforming data formats
    """

    @staticmethod
    def dict_to_dataframe(data:  List[Dict]) -> pd.DataFrame:
        """
        Convert list of dicts to DataFrame
        
        Args: 
            data: List of dictionaries
            
        Returns: 
            pd.DataFrame: Converted DataFrame
        """
        try:
            return pd.DataFrame(data)
        except (ValueError, TypeError) as e:
            logger.error(f"❌ DataFrame conversion error: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def dataframe_to_dict(df: pd.DataFrame, orient: str = 'records') -> Union[List[Dict], Dict]: 
        """
        Convert DataFrame to dict or list of dicts
        
        Args: 
            df: DataFrame to convert
            orient:  Orientation ('records', 'list', 'dict', etc.)
            
        Returns: 
            Union[List[Dict], Dict]: Converted data
        """
        try:
            return df.to_dict(orient=orient)
        except (ValueError, AttributeError) as e:
            logger.error(f"❌ Dict conversion error: {str(e)}")
            return {}

    @staticmethod
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dictionary
        
        Args: 
            d: Dictionary to flatten
            parent_key:  Parent key prefix
            sep: Separator for nested keys
            
        Returns:  
            Dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items. extend(DataTransformer.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def unflatten_dict(d: Dict, sep: str = '. ') -> Dict:
        """
        Unflatten dictionary with separator
        
        Args: 
            d:  Flattened dictionary
            sep:  Separator used in keys
            
        Returns: 
            Dict: Nested dictionary
        """
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]: 
                if part not in current: 
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    @staticmethod
    def rename_keys(data: Dict, mapping:  Dict[str, str]) -> Dict:
        """
        Rename dictionary keys
        
        Args: 
            data: Dictionary with data
            mapping: Key mapping {old_key: new_key}
            
        Returns: 
            Dict: Dictionary with renamed keys
        """
        return {mapping.get(k, k): v for k, v in data.items()}

    @staticmethod
    def filter_dict(data: Dict, keys: List[str], exclude: bool = False) -> Dict:
        """
        Filter dictionary by keys
        
        Args: 
            data: Dictionary to filter
            keys: Keys to include/exclude
            exclude: If True, exclude keys; if False, include only keys
            
        Returns: 
            Dict: Filtered dictionary
        """
        if exclude:
            return {k: v for k, v in data.items() if k not in keys}
        else:
            return {k: v for k, v in data.items() if k in keys}


# ===================================================================
# HASHING & SECURITY UTILITIES
# ===================================================================

class SecurityUtils:
    """
    Utilities for security operations
    """

    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """
        Hash a string
        
        Args: 
            text: Text to hash
            algorithm: Hash algorithm
            
        Returns: 
            str: Hashed string
        """
        try:
            if algorithm == 'sha256': 
                return hashlib.sha256(text.encode()).hexdigest()
            elif algorithm == 'md5':
                return hashlib. md5(text.encode()).hexdigest()
            elif algorithm == 'sha512':
                return hashlib.sha512(text.encode()).hexdigest()
            else:
                return hashlib.sha256(text.encode()).hexdigest()
        except Exception as e:
            logger. error(f"❌ Hashing error: {str(e)}")
            return ""

    @staticmethod
    def generate_uuid() -> str:
        """
        Generate UUID string
        
        Returns: 
            str: UUID string
        """
        import uuid
        return str(uuid. uuid4())

    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """
        Mask sensitive data like API keys
        
        Args: 
            data: Data to mask
            visible_chars: Number of visible characters
            
        Returns: 
            str: Masked data
        """
        if len(data) <= visible_chars:
            return "*" * len(data)
        return data[:visible_chars] + "*" * (len(data) - visible_chars)


# ===================================================================
# FILE & EXPORT UTILITIES
# ===================================================================

class FileUtils:
    """
    Utilities for file operations
    """

    @staticmethod
    def export_to_csv(data: List[Dict], filename: str) -> bool:
        """
        Export data to CSV file
        
        Args: 
            data: List of dictionaries
            filename: Output filename
            
        Returns: 
            bool:  Success status
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logger.info(f"✅ Data exported to {filename}")
            return True
        except Exception as e: 
            logger.error(f"❌ CSV export error: {str(e)}")
            return False

    @staticmethod
    def export_to_json(data: Any, filename: str) -> bool:
        """
        Export data to JSON file
        
        Args:  
            data: Data to export
            filename: Output filename
            
        Returns: 
            bool: Success status
        """
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger. info(f"✅ Data exported to {filename}")
            return True
        except Exception as e: 
            logger.error(f"❌ JSON export error: {str(e)}")
            return False

    @staticmethod
    def export_to_excel(data: List[Dict], filename: str, sheet_name: str = 'Data') -> bool:
        """
        Export data to Excel file
        
        Args: 
            data: List of dictionaries
            filename: Output filename
            sheet_name: Excel sheet name
            
        Returns:  
            bool: Success status
        """
        try:
            df = pd.DataFrame(data)
            df.to_excel(filename, sheet_name=sheet_name, index=False)
            logger.info(f"✅ Data exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Excel export error:  {str(e)}")
            return False


# ===================================================================
# CACHING UTILITIES
# ===================================================================

class CacheUtils:
    """
    Utilities for caching operations
    """

    @staticmethod
    def cache_key(*args, prefix: str = 'cache') -> str:
        """
        Generate cache key from arguments
        
        Args: 
            args: Arguments to include in key
            prefix: Key prefix
            
        Returns: 
            str: Cache key
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ': '.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def is_cache_expired(cached_time: datetime, ttl_seconds: int) -> bool:
        """
        Check if cache entry has expired
        
        Args:  
            cached_time: When cache was created
            ttl_seconds: Time to live in seconds
            
        Returns: 
            bool: True if expired
        """
        try:
            elapsed = (datetime.now(timezone.utc) - cached_time).total_seconds()
            return elapsed > ttl_seconds
        except (TypeError, AttributeError):
            return True


# ===================================================================
# LOGGING UTILITIES
# ===================================================================

class LoggingUtils:
    """
    Utilities for logging operations
    """

    @staticmethod
    def log_request_info(endpoint: str, method: str, params: Dict = None):
        """
        Log request information
        
        Args: 
            endpoint: API endpoint
            method: HTTP method
            params: Request parameters
        """
        logger.info(f"📥 Request: {method} {endpoint}")
        if params:
            logger.debug(f"   Params: {json.dumps(params, default=str)}")

    @staticmethod
    def log_response_info(endpoint: str, status_code: int, response_time:  float):
        """
        Log response information
        
        Args:  
            endpoint: API endpoint
            status_code: HTTP status code
            response_time: Response time in seconds
        """
        logger. info(f"📤 Response: {endpoint} - {status_code} ({response_time:.3f}s)")

    @staticmethod
    def log_error(error: Exception, context: str = ""):
        """
        Log error with context
        
        Args: 
            error: Exception object
            context: Error context
        """
        logger.error(f"❌ Error {context}: {str(error)}", exc_info=True)


if __name__ == '__main__':
    logger. info("🔧 API Utilities module loaded successfully")