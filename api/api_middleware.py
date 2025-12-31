"""
API Middleware Layer
Authentication, Authorization, Rate Limiting, and Request/Response Processing
NEYDRA Platform - Enterprise Grade
Founder & CEO: Ilyes Jarray
© 2025 - All Rights Reserved
"""

import os
import time
import json
import logging
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple, Any
from functools import wraps
from collections import defaultdict

import jwt
from flask import request, jsonify, g, current_app
from werkzeug. exceptions import HTTPException
import redis

# ===== LOGGING CONFIGURATION =====
logger = logging.getLogger(__name__)


# ===================================================================
# AUTHENTICATION MIDDLEWARE
# ===================================================================

class AuthenticationMiddleware:
    """
    Handles JWT token validation and API key authentication
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = 'HS256',
        expiration_hours: int = 24,
        enable_api_keys: bool = True
    ):
        """
        Initialize authentication middleware
        
        Args: 
            secret_key: JWT secret key
            algorithm: JWT algorithm (default: HS256)
            expiration_hours: Token expiration time
            enable_api_keys: Enable API key authentication
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_hours = expiration_hours
        self.enable_api_keys = enable_api_keys
        self. valid_api_keys = self._load_api_keys()

        logger.info("🔐 Authentication Middleware initialized")

    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        Load valid API keys from environment or configuration
        
        Returns:
            Dict:  Valid API keys with metadata
        """
        try:
            api_keys_env = os.getenv('API_KEYS', '{}')
            api_keys = json.loads(api_keys_env)
            logger.info(f"✅ Loaded {len(api_keys)} API keys")
            return api_keys
        except json.JSONDecodeError:
            logger.warning("⚠️  Failed to parse API_KEYS environment variable")
            return {}

    def generate_token(
        self,
        user_id: str,
        username: str,
        is_admin: bool = False,
        custom_claims: Optional[Dict] = None
    ) -> str:
        """
        Generate JWT token
        
        Args:
            user_id: User ID
            username: Username
            is_admin: Admin flag
            custom_claims: Additional claims
            
        Returns: 
            str: JWT token
        """
        try:
            payload = {
                'user_id': user_id,
                'username': username,
                'is_admin': is_admin,
                'iat': datetime.now(timezone.utc),
                'exp': datetime.now(timezone.utc) + timedelta(hours=self.expiration_hours)
            }

            if custom_claims:
                payload.update(custom_claims)

            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"🎫 Token generated for user:  {username}")
            return token

        except Exception as e:
            logger.error(f"❌ Token generation error: {str(e)}")
            raise

    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Verify JWT token
        
        Args:
            token: JWT token to verify
            
        Returns: 
            Tuple: (is_valid, payload, error_message)
        """
        try:
            payload = jwt. decode(token, self.secret_key, algorithms=[self.algorithm])
            logger.debug(f"✅ Token verified for user: {payload.get('username')}")
            return True, payload, None

        except jwt.ExpiredSignatureError:
            error = "Token has expired"
            logger.warning(f"⚠️  {error}")
            return False, None, error

        except jwt.InvalidTokenError as e:
            error = f"Invalid token: {str(e)}"
            logger.warning(f"⚠️  {error}")
            return False, None, error

        except Exception as e:
            error = f"Token verification error: {str(e)}"
            logger.error(f"❌ {error}")
            return False, None, error

    def verify_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Verify API key
        
        Args:
            api_key: API key to verify
            
        Returns:
            Tuple: (is_valid, user_data, error_message)
        """
        try:
            if not self.enable_api_keys:
                return False, None, "API key authentication disabled"

            if api_key not in self.valid_api_keys:
                logger.warning(f"⚠️  Invalid API key: {api_key[: 8]}...")
                return False, None, "Invalid API key"

            key_data = self.valid_api_keys[api_key]
            
            # Check if key is active
            if not key_data.get('active', True):
                return False, None, "API key is deactivated"

            # Check expiration
            if 'expires_at' in key_data: 
                expires = datetime.fromisoformat(key_data['expires_at'])
                if datetime.now(timezone.utc) > expires:
                    return False, None, "API key has expired"

            logger.debug(f"✅ API key verified for user: {key_data. get('username')}")
            return True, key_data, None

        except Exception as e:
            error = f"API key verification error:  {str(e)}"
            logger.error(f"❌ {error}")
            return False, None, error

    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret:  str
    ) -> bool:
        """
        Verify HMAC-SHA256 signature
        
        Args:
            payload: Request payload
            signature:  Signature to verify
            secret: Secret key
            
        Returns: 
            bool: True if signature is valid
        """
        try: 
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            is_valid = hmac.compare_digest(expected_signature, signature)
            
            if not is_valid: 
                logger.warning("⚠️  Invalid signature")
            else:
                logger.debug("✅ Signature verified")

            return is_valid

        except Exception as e:
            logger. error(f"❌ Signature verification error: {str(e)}")
            return False


# ===================================================================
# RATE LIMITING MIDDLEWARE
# ===================================================================

class RateLimitMiddleware:
    """
    Handles API rate limiting using token bucket algorithm
    """

    def __init__(
        self,
        requests_per_minute: int = 100,
        requests_per_hour: int = 5000,
        redis_client: Optional[redis.Redis] = None,
        storage_backend: str = 'memory'
    ):
        """
        Initialize rate limit middleware
        
        Args: 
            requests_per_minute:  Requests per minute limit
            requests_per_hour:  Requests per hour limit
            redis_client: Redis client for distributed rate limiting
            storage_backend: 'memory' or 'redis'
        """
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
        self.redis_client = redis_client
        self.storage_backend = storage_backend
        
        # In-memory storage (fallback)
        self.request_counts = defaultdict(lambda: {
            'minute': [],
            'hour': []
        })

        logger.info(
            f"🚦 Rate Limit Middleware initialized "
            f"(RPM: {requests_per_minute}, RPH:  {requests_per_hour})"
        )

    def get_client_identifier(self, request_obj) -> str:
        """
        Get unique client identifier (IP or API key)
        
        Args: 
            request_obj: Flask request object
            
        Returns: 
            str: Client identifier
        """
        # Try to get from API key first
        api_key = request_obj.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key[: 16]}"

        # Get from JWT token
        auth_header = request_obj.headers.get('Authorization', '')
        if auth_header. startswith('Bearer '):
            token = auth_header[7:]
            return f"token:{token[:16]}"

        # Fallback to IP address
        if request_obj.headers.get('X-Forwarded-For'):
            ip = request_obj.headers.get('X-Forwarded-For').split(',')[0]
        else:
            ip = request_obj.remote_addr

        return f"ip:{ip}"

    def is_rate_limited(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if client is rate limited
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple: (is_limited, metadata)
        """
        try: 
            current_time = time.time()
            minute_ago = current_time - 60
            hour_ago = current_time - 3600

            if self.storage_backend == 'redis' and self.redis_client:
                return self._check_redis_limit(client_id, current_time, minute_ago, hour_ago)
            else:
                return self._check_memory_limit(client_id, current_time, minute_ago, hour_ago)

        except Exception as e:
            logger.error(f"❌ Rate limit check error: {str(e)}")
            # Fail open - allow request on error
            return False, {}

    def _check_redis_limit(
        self,
        client_id: str,
        current_time: float,
        minute_ago: float,
        hour_ago: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit using Redis
        
        Args: 
            client_id: Client identifier
            current_time: Current timestamp
            minute_ago: Timestamp 1 minute ago
            hour_ago: Timestamp 1 hour ago
            
        Returns: 
            Tuple: (is_limited, metadata)
        """
        rpm_key = f"rate_limit: rpm:{client_id}"
        rph_key = f"rate_limit:rph:{client_id}"

        try:
            # Increment counters
            rpm_count = self.redis_client.incr(rpm_key)
            rph_count = self.redis_client.incr(rph_key)

            # Set expiration
            if rpm_count == 1:
                self.redis_client.expire(rpm_key, 60)
            if rph_count == 1:
                self.redis_client.expire(rph_key, 3600)

            rpm_remaining = max(0, self.rpm_limit - rpm_count)
            rph_remaining = max(0, self.rph_limit - rph_count)

            metadata = {
                'rpm_limit': self.rpm_limit,
                'rpm_remaining': rpm_remaining,
                'rph_limit': self.rph_limit,
                'rph_remaining': rph_remaining
            }

            is_limited = rpm_count > self.rpm_limit or rph_count > self.rph_limit

            return is_limited, metadata

        except Exception as e:
            logger.error(f"❌ Redis rate limit error: {str(e)}")
            return False, {}

    def _check_memory_limit(
        self,
        client_id: str,
        current_time: float,
        minute_ago: float,
        hour_ago: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit using in-memory storage
        
        Args:
            client_id: Client identifier
            current_time: Current timestamp
            minute_ago: Timestamp 1 minute ago
            hour_ago:  Timestamp 1 hour ago
            
        Returns:
            Tuple: (is_limited, metadata)
        """
        counts = self.request_counts[client_id]

        # Clean old entries
        counts['minute'] = [t for t in counts['minute'] if t > minute_ago]
        counts['hour'] = [t for t in counts['hour'] if t > hour_ago]

        # Add current request
        counts['minute']. append(current_time)
        counts['hour'].append(current_time)

        rpm_count = len(counts['minute'])
        rph_count = len(counts['hour'])

        rpm_remaining = max(0, self.rpm_limit - rpm_count)
        rph_remaining = max(0, self.rph_limit - rph_count)

        metadata = {
            'rpm_limit': self.rpm_limit,
            'rpm_remaining': rpm_remaining,
            'rph_limit': self.rph_limit,
            'rph_remaining': rph_remaining
        }

        is_limited = rpm_count > self.rpm_limit or rph_count > self.rph_limit

        return is_limited, metadata

    def get_reset_time(self, client_id: str) -> Dict[str, int]:
        """
        Get rate limit reset times
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dict: Reset times in seconds
        """
        if self.storage_backend == 'redis' and self.redis_client:
            rpm_key = f"rate_limit:rpm:{client_id}"
            rph_key = f"rate_limit:rph:{client_id}"

            rpm_ttl = self.redis_client. ttl(rpm_key)
            rph_ttl = self.redis_client.ttl(rph_key)

            return {
                'rpm_reset_in': max(0, rpm_ttl),
                'rph_reset_in': max(0, rph_ttl)
            }

        return {
            'rpm_reset_in': 60,
            'rph_reset_in': 3600
        }


# ===================================================================
# REQUEST/RESPONSE MIDDLEWARE
# ===================================================================

class RequestResponseMiddleware:
    """
    Handles request/response logging, timing, and transformation
    """

    def __init__(self, log_request_body: bool = False, log_response_body: bool = False):
        """
        Initialize request/response middleware
        
        Args:
            log_request_body: Log request body
            log_response_body: Log response body
        """
        self. log_request_body = log_request_body
        self.log_response_body = log_response_body

        logger.info("📝 Request/Response Middleware initialized")

    def log_request(self, request_obj):
        """
        Log incoming request
        
        Args: 
            request_obj: Flask request object
        """
        try:
            g.start_time = time.time()

            request_data = {
                'method': request_obj.method,
                'path': request_obj.path,
                'remote_addr': request_obj.remote_addr,
                'user_agent': request_obj.user_agent.string if request_obj.user_agent else 'Unknown'
            }

            if self.log_request_body and request_obj.is_json:
                request_data['body'] = request_obj. get_json(silent=True)

            logger.debug(f"📥 Incoming request: {json.dumps(request_data)}")

        except Exception as e:
            logger.error(f"❌ Request logging error: {str(e)}")

    def log_response(self, response):
        """
        Log outgoing response
        
        Args: 
            response: Flask response object
            
        Returns:
            response: Modified response object
        """
        try:
            elapsed_time = time.time() - getattr(g, 'start_time', time.time())

            response_data = {
                'status_code': response.status_code,
                'elapsed_ms': round(elapsed_time * 1000, 2)
            }

            if self.log_response_body and response.is_json:
                try:
                    response_data['body'] = response.get_json()
                except:
                    pass

            logger.debug(f"📤 Outgoing response:  {json.dumps(response_data)}")

            # Add timing header
            response. headers['X-Response-Time'] = f"{elapsed_time:.3f}s"

        except Exception as e:
            logger.error(f"❌ Response logging error: {str(e)}")

        return response


# ===================================================================
# CORS MIDDLEWARE
# ===================================================================

class CORSMiddleware:
    """
    Handles Cross-Origin Resource Sharing (CORS)
    """

    def __init__(
        self,
        allowed_origins: List[str],
        allowed_methods: List[str],
        allowed_headers: List[str],
        allow_credentials: bool = True,
        max_age: int = 86400
    ):
        """
        Initialize CORS middleware
        
        Args:
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            allow_credentials: Allow credentials
            max_age: Max age for preflight cache (seconds)
        """
        self. allowed_origins = allowed_origins
        self.allowed_methods = allowed_methods
        self.allowed_headers = allowed_headers
        self.allow_credentials = allow_credentials
        self.max_age = max_age

        logger.info("🔄 CORS Middleware initialized")

    def get_cors_headers(self, origin: str) -> Dict[str, str]: 
        """
        Get CORS headers for response
        
        Args:
            origin: Request origin
            
        Returns: 
            Dict: CORS headers
        """
        headers = {}

        # Check if origin is allowed
        if origin in self.allowed_origins or '*' in self.allowed_origins:
            headers['Access-Control-Allow-Origin'] = origin
            headers['Access-Control-Allow-Methods'] = ', '.join(self. allowed_methods)
            headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            headers['Access-Control-Max-Age'] = str(self. max_age)

            if self.allow_credentials:
                headers['Access-Control-Allow-Credentials'] = 'true'

        return headers

    def handle_preflight(self, request_obj):
        """
        Handle preflight OPTIONS request
        
        Args:
            request_obj: Flask request object
            
        Returns:
            Response: CORS response
        """
        origin = request_obj.headers.get('Origin')
        cors_headers = self.get_cors_headers(origin)

        if not cors_headers:
            logger.warning(f"⚠️  CORS request from disallowed origin: {origin}")
            return jsonify({'error': 'CORS origin not allowed'}), 403

        response = jsonify({'status': 'ok'})
        for key, value in cors_headers.items():
            response.headers[key] = value

        logger.debug(f"✅ CORS preflight handled for origin: {origin}")
        return response, 200


# ===================================================================
# DECORATOR FUNCTIONS
# ===================================================================

def require_auth(auth_middleware:  AuthenticationMiddleware):
    """
    Decorator to require authentication
    
    Args:
        auth_middleware: AuthenticationMiddleware instance
    """
    def decorator(f:  Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                auth_header = request.headers.get('Authorization', '')

                # Check API Key first
                api_key = request.headers.get('X-API-Key')
                if api_key: 
                    is_valid, user_data, error = auth_middleware.verify_api_key(api_key)
                    if is_valid:
                        g.user = user_data
                        g.auth_method = 'api_key'
                        return f(*args, **kwargs)
                    else:
                        logger.warning(f"❌ API key auth failed:  {error}")
                        return jsonify({'error': error}), 401

                # Check JWT Token
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    is_valid, payload, error = auth_middleware.verify_token(token)
                    if is_valid:
                        g. user = payload
                        g. auth_method = 'jwt'
                        return f(*args, **kwargs)
                    else: 
                        logger.warning(f"❌ JWT auth failed: {error}")
                        return jsonify({'error': error}), 401

                # No valid authentication
                logger.warning("❌ No authentication provided")
                return jsonify({'error': 'Authentication required'}), 401

            except Exception as e: 
                logger.error(f"❌ Authentication error: {str(e)}")
                return jsonify({'error': 'Authentication error'}), 500

        return decorated_function
    return decorator


def require_admin(auth_middleware: AuthenticationMiddleware):
    """
    Decorator to require admin role
    
    Args:
        auth_middleware: AuthenticationMiddleware instance
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # First require authentication
                auth_header = request.headers.get('Authorization', '')
                api_key = request.headers.get('X-API-Key')

                user_data = None

                if api_key:
                    is_valid, user_data, _ = auth_middleware.verify_api_key(api_key)
                elif auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    is_valid, user_data, _ = auth_middleware.verify_token(token)

                if not user_data: 
                    return jsonify({'error': 'Authentication required'}), 401

                # Check admin status
                if not user_data.get('is_admin', False):
                    logger.warning(f"❌ Admin access denied for user: {user_data.get('username')}")
                    return jsonify({'error': 'Admin access required'}), 403

                g. user = user_data
                return f(*args, **kwargs)

            except Exception as e:
                logger.error(f"❌ Admin auth error: {str(e)}")
                return jsonify({'error':  'Authorization error'}), 500

        return decorated_function
    return decorator


def rate_limit(rate_limiter: RateLimitMiddleware):
    """
    Decorator to apply rate limiting
    
    Args: 
        rate_limiter: RateLimitMiddleware instance
    """
    def decorator(f:  Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                client_id = rate_limiter.get_client_identifier(request)
                is_limited, metadata = rate_limiter.is_rate_limited(client_id)

                # Add rate limit headers
                response_data = f(*args, **kwargs)
                
                if isinstance(response_data, tuple):
                    response_obj, status_code = response_data
                else:
                    response_obj = response_data
                    status_code = 200

                if not isinstance(response_obj, dict):
                    response_obj = {'data': response_obj}

                # Add rate limit info to response headers
                if isinstance(response_obj, dict) or hasattr(response_obj, 'headers'):
                    if hasattr(response_obj, 'headers'):
                        response_obj.headers['X-RateLimit-Limit'] = str(metadata.get('rpm_limit', self.rpm_limit))
                        response_obj.headers['X-RateLimit-Remaining'] = str(metadata.get('rpm_remaining', 0))
                        reset_time = rate_limiter.get_reset_time(client_id)
                        response_obj.headers['X-RateLimit-Reset'] = str(reset_time. get('rpm_reset_in', 60))

                if is_limited:
                    logger.warning(f"⚠️  Rate limit exceeded for client:  {client_id}")
                    reset_time = rate_limiter. get_reset_time(client_id)
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': reset_time.get('rpm_reset_in', 60)
                    }), 429

                return response_data

            except Exception as e: 
                logger.error(f"❌ Rate limiting error: {str(e)}")
                # Fail open - allow request on error
                return f(*args, **kwargs)

        return decorated_function
    return decorator


# ===================================================================
# ERROR HANDLER
# ===================================================================

class ErrorHandler:
    """
    Handles API errors and exceptions
    """

    @staticmethod
    def handle_error(error: Exception, status_code: int = 500) -> Tuple[Dict, int]:
        """
        Handle and format error responses
        
        Args: 
            error: Exception object
            status_code: HTTP status code
            
        Returns: 
            Tuple: (error_response, status_code)
        """
        try:
            if isinstance(error, HTTPException):
                status_code = error.code
                message = error.description
            else:
                message = str(error)

            error_response = {
                'success': False,
                'error':  message,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status_code': status_code
            }

            # Add request ID if available
            if hasattr(g, 'request_id'):
                error_response['request_id'] = g.request_id

            logger.error(f"❌ Error (HTTP {status_code}): {message}")

            return error_response, status_code

        except Exception as e: 
            logger.error(f"❌ Error handling error: {str(e)}")
            return {
                'success': False,
                'error': 'Internal server error',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status_code': 500
            }, 500


# ===================================================================
# MIDDLEWARE FACTORY
# ===================================================================

class MiddlewareFactory:
    """
    Factory for creating and configuring middleware instances
    """

    @staticmethod
    def create_auth_middleware(app) -> AuthenticationMiddleware:
        """Create authentication middleware from app config"""
        return AuthenticationMiddleware(
            secret_key=os.getenv('JWT_SECRET_KEY', 'change-this-secret-key'),
            algorithm=os.getenv('JWT_ALGORITHM', 'HS256'),
            expiration_hours=int(os.getenv('JWT_EXPIRATION_HOURS', 24)),
            enable_api_keys=os.getenv('ENABLE_API_KEYS', 'true').lower() == 'true'
        )

    @staticmethod
    def create_rate_limiter(
        app,
        redis_client: Optional[redis.Redis] = None
    ) -> RateLimitMiddleware:
        """Create rate limiter from app config"""
        return RateLimitMiddleware(
            requests_per_minute=int(os.getenv('RATE_LIMIT_RPM', 100)),
            requests_per_hour=int(os.getenv('RATE_LIMIT_RPH', 5000)),
            redis_client=redis_client,
            storage_backend='redis' if redis_client else 'memory'
        )

    @staticmethod
    def create_cors_middleware(app) -> CORSMiddleware:
        """Create CORS middleware from app config"""
        cors_origins_env = os.getenv('CORS_ORIGINS', 'http://localhost:3000')
        cors_origins = [origin.strip() for origin in cors_origins_env.split(',')]

        cors_methods_env = os.getenv('CORS_ALLOW_METHODS', 'GET,POST,PUT,DELETE,OPTIONS')
        cors_methods = [method.strip() for method in cors_methods_env.split(',')]

        cors_headers_env = os.getenv(
            'CORS_ALLOW_HEADERS',
            'Content-Type,Authorization,X-API-Key,X-Request-ID'
        )
        cors_headers = [header.strip() for header in cors_headers_env.split(',')]

        return CORSMiddleware(
            allowed_origins=cors_origins,
            allowed_methods=cors_methods,
            allowed_headers=cors_headers,
            allow_credentials=os.getenv('CORS_ALLOW_CREDENTIALS', 'true').lower() == 'true'
        )

    @staticmethod
    def create_request_response_middleware(app) -> RequestResponseMiddleware:
        """Create request/response middleware from app config"""
        return RequestResponseMiddleware(
            log_request_body=os.getenv('LOG_REQUEST_BODY', 'false').lower() == 'true',
            log_response_body=os.getenv('LOG_RESPONSE_BODY', 'false').lower() == 'true'
        )


if __name__ == '__main__': 
    logger.info("🔧 API Middleware module loaded successfully")