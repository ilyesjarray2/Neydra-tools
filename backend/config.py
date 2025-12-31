"""
Configuration Module
XAU/USD Predictive Analytics Engine - NEYDRA Platform
Handles all configuration, environment variables, and settings
Founder & CEO: Ilyes Jarray
© 2025 - All Rights Reserved
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging

# ===== LOAD ENVIRONMENT VARIABLES =====
# Load from . env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# ===== LOGGING CONFIGURATION =====
@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file: str = os.getenv('LOG_FILE', 'logs/app.log')
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = os.getenv('CONSOLE_OUTPUT', 'true').lower() == 'true'


# ===== DATABASE CONFIGURATION =====
@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    # PostgreSQL
    postgres_enabled: bool = os.getenv('POSTGRES_ENABLED', 'true').lower() == 'true'
    postgres_host: str = os.getenv('POSTGRES_HOST', 'localhost')
    postgres_port: int = int(os.getenv('POSTGRES_PORT', 5432))
    postgres_user: str = os.getenv('POSTGRES_USER', 'postgres')
    postgres_password: str = os.getenv('POSTGRES_PASSWORD', 'password')
    postgres_database: str = os.getenv('POSTGRES_DATABASE', 'neydra')
    postgres_ssl_mode: str = os.getenv('POSTGRES_SSL_MODE', 'prefer')
    postgres_pool_size: int = int(os.getenv('POSTGRES_POOL_SIZE', 10))
    postgres_max_overflow: int = int(os.getenv('POSTGRES_MAX_OVERFLOW', 20))
    
    # Redis Cache
    redis_enabled: bool = os.getenv('REDIS_ENABLED', 'true').lower() == 'true'
    redis_host: str = os. getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os. getenv('REDIS_PORT', 6379))
    redis_db: int = int(os. getenv('REDIS_DB', 0))
    redis_password: str = os.getenv('REDIS_PASSWORD', None)
    redis_ttl: int = int(os. getenv('REDIS_TTL', 3600))  # 1 hour
    
    # MongoDB
    mongodb_enabled: bool = os.getenv('MONGODB_ENABLED', 'false').lower() == 'true'
    mongodb_uri: str = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    mongodb_database: str = os.getenv('MONGODB_DATABASE', 'neydra')

    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self. postgres_password}"
            f"@{self. postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )

    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL"""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}"
                f"@{self. redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# ===== BROKER API CONFIGURATION =====
@dataclass
class BrokerConfig: 
    """Broker API configuration"""
    
    # OANDA Configuration
    oanda_enabled: bool = os.getenv('OANDA_ENABLED', 'true').lower() == 'true'
    oanda_api_key: str = os.getenv('OANDA_API_KEY', '')
    oanda_account_id: str = os. getenv('OANDA_ACCOUNT_ID', '')
    oanda_endpoint: str = os.getenv('OANDA_ENDPOINT', 'https://api-fxpractice.oanda.com/v3')
    oanda_instruments: str = 'XAU_USD'
    oanda_timeout: int = 10
    oanda_weight:  float = 0.40  # 40% weight in ensemble
    
    # Alpha Vantage Configuration
    alphavantage_enabled: bool = os.getenv('ALPHAVANTAGE_ENABLED', 'true').lower() == 'true'
    alphavantage_api_key: str = os.getenv('ALPHAVANTAGE_API_KEY', '')
    alphavantage_endpoint: str = 'https://www.alphavantage.co/query'
    alphavantage_timeout: int = 10
    alphavantage_weight: float = 0.30  # 30% weight
    
    # Finnhub Configuration
    finnhub_enabled: bool = os.getenv('FINNHUB_ENABLED', 'true').lower() == 'true'
    finnhub_api_key: str = os.getenv('FINNHUB_API_KEY', '')
    finnhub_endpoint:  str = 'https://finnhub.io/api/v1'
    finnhub_timeout:  int = 10
    finnhub_weight: float = 0.20  # 20% weight
    
    # IEX Cloud Configuration
    iex_enabled: bool = os.getenv('IEX_ENABLED', 'false').lower() == 'true'
    iex_api_key: str = os.getenv('IEX_API_KEY', '')
    iex_endpoint: str = 'https://cloud.iexapis.com/stable'
    iex_timeout: int = 10
    iex_weight: float = 0.10  # 10% weight
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = int(os.getenv('RATE_LIMIT_RPM', 100))
    rate_limit_requests_per_hour: int = int(os.getenv('RATE_LIMIT_RPH', 5000))
    
    @property
    def enabled_brokers(self) -> List[str]:
        """Get list of enabled brokers"""
        enabled = []
        if self.oanda_enabled:
            enabled.append('oanda')
        if self.alphavantage_enabled:
            enabled.append('alphavantage')
        if self.finnhub_enabled:
            enabled.append('finnhub')
        if self.iex_enabled:
            enabled.append('iex')
        return enabled


# ===== ML MODEL CONFIGURATION =====
@dataclass
class ModelConfig:
    """Machine Learning model configuration"""
    
    # Model Paths
    models_dir: str = os.getenv('MODELS_DIR', 'backend/models')
    model_save_interval: int = int(os.getenv('MODEL_SAVE_INTERVAL', 3600))  # seconds
    
    # Random Forest Parameters
    rf_n_estimators: int = int(os.getenv('RF_N_ESTIMATORS', 200))
    rf_max_depth: int = int(os.getenv('RF_MAX_DEPTH', 15))
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    rf_random_state: int = 42
    rf_weight:  float = 0.35  # 35% weight in ensemble
    
    # Gradient Boosting Parameters
    gb_n_estimators: int = int(os.getenv('GB_N_ESTIMATORS', 150))
    gb_learning_rate: float = float(os.getenv('GB_LEARNING_RATE', 0.1))
    gb_max_depth: int = 7
    gb_min_samples_split: int = 5
    gb_subsample: float = 0.8
    gb_random_state: int = 42
    gb_weight: float = 0.35  # 35% weight in ensemble
    
    # LSTM Parameters
    lstm_layers: List[int] = None
    lstm_epochs: int = int(os.getenv('LSTM_EPOCHS', 50))
    lstm_batch_size: int = int(os.getenv('LSTM_BATCH_SIZE', 32))
    lstm_validation_split: float = 0.2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_weight:  float = 0.30  # 30% weight in ensemble
    
    # Training Configuration
    train_test_split: float = 0.8
    validation_split: float = 0.2
    look_back_window: int = int(os.getenv('LOOK_BACK_WINDOW', 20))
    historical_days: int = int(os.getenv('HISTORICAL_DAYS', 365))
    
    # Model Performance Thresholds
    min_accuracy: float = 0.87  # 87% minimum accuracy
    min_r2_score: float = 0.80
    max_rmse: float = 50.0
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.lstm_layers is None:
            self. lstm_layers = [64, 32]


# ===== PREDICTION ENGINE CONFIGURATION =====
@dataclass
class PredictionConfig: 
    """Prediction engine configuration"""
    
    # Update Intervals
    price_update_interval: int = int(os.getenv('PRICE_UPDATE_INTERVAL', 3))  # seconds
    prediction_interval: int = int(os.getenv('PREDICTION_INTERVAL', 300))  # 5 minutes
    model_retraining_interval: int = int(os.getenv('MODEL_RETRAIN_INTERVAL', 86400))  # 24 hours
    
    # Prediction Parameters
    confidence_threshold: float = 0.87
    signal_strength_threshold: float = 0.70
    enable_real_time_updates: bool = True
    enable_backtesting: bool = os.getenv('ENABLE_BACKTESTING', 'true').lower() == 'true'
    
    # Signal Generation
    rsi_oversold_threshold: int = 30
    rsi_overbought_threshold: int = 70
    macd_threshold: float = 0.0001
    bollinger_bands_std_dev: int = 2
    atr_period: int = 14
    
    # Price Targets
    take_profit_percent: float = 2.0  # 2%
    stop_loss_percent:  float = 1.0  # 1%
    trailing_stop_enabled: bool = True
    trailing_stop_percent: float = 0.5  # 0.5%


# ===== API CONFIGURATION =====
@dataclass
class ApiConfig:
    """API server configuration"""
    
    # Server
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', 5000))
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    workers: int = int(os.getenv('WORKERS', 4))
    
    # CORS
    cors_origins: List[str] = None
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = None
    cors_allow_headers: List[str] = None
    
    # Rate Limiting
    rate_limit_enabled:  bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_requests_per_hour: int = 5000
    
    # Authentication
    jwt_secret_key: str = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    jwt_algorithm: str = 'HS256'
    jwt_expiration_hours: int = 24
    api_keys: Dict[str, str] = None
    
    # Timeouts
    request_timeout: int = 30
    response_timeout: int = 60
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.cors_origins is None:
            self.cors_origins = [
                'http://localhost:3000',
                'http://127.0.0.1:3000',
                'http://localhost:8080',
                os.getenv('FRONTEND_URL', 'http://localhost:3000')
            ]
        
        if self.cors_allow_methods is None:
            self.cors_allow_methods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        
        if self.cors_allow_headers is None:
            self.cors_allow_headers = [
                'Content-Type',
                'Authorization',
                'X-API-Key',
                'X-Request-ID'
            ]
        
        if self.api_keys is None:
            api_keys_env = os.getenv('API_KEYS', '{}')
            try:
                self.api_keys = json.loads(api_keys_env)
            except json. JSONDecodeError:
                self.api_keys = {}


# ===== WEBHOOK CONFIGURATION =====
@dataclass
class WebhookConfig:
    """Webhook configuration"""
    
    # Security
    signature_verification_enabled: bool = True
    webhook_secret_key: str = os.getenv('WEBHOOK_SECRET_KEY', 'webhook-secret-key')
    allowed_ips: List[str] = None
    
    # Queue Configuration
    queue_enabled: bool = True
    queue_max_retries: int = 3
    queue_retry_delay: int = 5  # seconds
    queue_dir: str = 'backend/queue'
    
    # Event Configuration
    events_enabled: bool = True
    event_log_file: str = 'logs/events.log'
    
    # Notification Endpoints
    slack_webhook_url: str = os.getenv('SLACK_WEBHOOK_URL', '')
    slack_enabled: bool = bool(os.getenv('SLACK_WEBHOOK_URL'))
    
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    telegram_enabled: bool = bool(os.getenv('TELEGRAM_BOT_TOKEN'))
    
    email_enabled: bool = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    email_smtp_host: str = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
    email_smtp_port: int = int(os.getenv('EMAIL_SMTP_PORT', 587))
    email_sender:  str = os.getenv('EMAIL_SENDER', '')
    email_password: str = os.getenv('EMAIL_PASSWORD', '')
    email_recipient:  str = os.getenv('EMAIL_RECIPIENT', '')
    
    # WebSocket
    websocket_enabled: bool = os.getenv('WEBSOCKET_ENABLED', 'true').lower() == 'true'
    websocket_port: int = int(os.getenv('WEBSOCKET_PORT', 8080))
    websocket_host: str = '0.0.0.0'
    websocket_max_connections: int = 1000
    websocket_ping_interval: int = 30
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.allowed_ips is None:
            allowed_ips_env = os. getenv('WEBHOOK_ALLOWED_IPS', '127.0.0.1,localhost')
            self.allowed_ips = [ip.strip() for ip in allowed_ips_env.split(',')]


# ===== MONITORING & ALERTING CONFIGURATION =====
@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = int(os.getenv('METRICS_PORT', 9090))
    metrics_path: str = '/metrics'
    
    # Health Checks
    health_check_interval: int = 60  # seconds
    health_check_timeout: int = 10
    
    # Alerting
    alert_price_change_percent: float = 5.0  # Alert if price changes >5%
    alert_accuracy_drop_percent: float = 5.0  # Alert if accuracy drops >5%
    alert_broker_down_minutes: int = 5  # Alert if broker down >5 minutes
    alert_queue_size_threshold: int = 1000  # Alert if queue size >1000
    
    # Performance Tracking
    track_prediction_accuracy: bool = True
    track_broker_latency: bool = True
    track_api_performance: bool = True
    performance_history_days: int = 30


# ===== SECURITY CONFIGURATION =====
@dataclass
class SecurityConfig:
    """Security configuration"""
    
    # SSL/TLS
    ssl_enabled:  bool = os.getenv('SSL_ENABLED', 'false').lower() == 'true'
    ssl_cert_file: str = os.getenv('SSL_CERT_FILE', '')
    ssl_key_file: str = os.getenv('SSL_KEY_FILE', '')
    
    # Encryption
    encrypt_sensitive_data: bool = True
    encryption_key: str = os.getenv('ENCRYPTION_KEY', 'your-encryption-key')
    
    # Password Policy
    password_min_length:  int = 12
    password_require_uppercase: bool = True
    password_require_numbers: bool = True
    password_require_special_chars: bool = True
    
    # Two-Factor Authentication
    mfa_enabled: bool = False
    mfa_provider: str = 'totp'  # Time-based OTP
    
    # IP Whitelisting
    ip_whitelist_enabled: bool = False
    ip_whitelist:  List[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.ip_whitelist is None:
            ip_whitelist_env = os. getenv('IP_WHITELIST', '')
            self.ip_whitelist = [ip.strip() for ip in ip_whitelist_env.split(',') if ip.strip()]


# ===== DEPLOYMENT CONFIGURATION =====
@dataclass
class DeploymentConfig: 
    """Deployment configuration"""
    
    # Environment
    environment: str = os. getenv('ENVIRONMENT', 'development')  # development, staging, production
    app_name: str = 'NEYDRA XAU/USD Predictive Analytics Engine'
    app_version: str = '1.0.0'
    
    # Docker
    docker_enabled: bool = os.getenv('DOCKER_ENABLED', 'false').lower() == 'true'
    docker_image: str = 'neydra/xau-usd-engine:latest'
    docker_registry: str = os.getenv('DOCKER_REGISTRY', 'docker.io')
    
    # Kubernetes
    kubernetes_enabled: bool = os.getenv('KUBERNETES_ENABLED', 'false').lower() == 'true'
    kubernetes_namespace: str = os.getenv('KUBERNETES_NAMESPACE', 'default')
    kubernetes_replicas: int = int(os.getenv('KUBERNETES_REPLICAS', 1))
    
    # Cloud Providers
    aws_enabled: bool = os.getenv('AWS_ENABLED', 'false').lower() == 'true'
    aws_region: str = os.getenv('AWS_REGION', 'us-east-1')
    aws_access_key: str = os.getenv('AWS_ACCESS_KEY', '')
    aws_secret_key: str = os.getenv('AWS_SECRET_KEY', '')
    
    gcp_enabled: bool = os. getenv('GCP_ENABLED', 'false').lower() == 'true'
    gcp_project_id: str = os.getenv('GCP_PROJECT_ID', '')
    gcp_credentials_file: str = os.getenv('GCP_CREDENTIALS_FILE', '')
    
    azure_enabled: bool = os.getenv('AZURE_ENABLED', 'false').lower() == 'true'
    azure_subscription_id: str = os.getenv('AZURE_SUBSCRIPTION_ID', '')
    azure_resource_group: str = os.getenv('AZURE_RESOURCE_GROUP', '')
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment. lower() == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == 'development'


# ===== MAIN CONFIGURATION CLASS =====
class Config:
    """Main configuration class aggregating all sub-configurations"""
    
    def __init__(self):
        """Initialize all configuration sections"""
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        self.broker = BrokerConfig()
        self.model = ModelConfig()
        self.prediction = PredictionConfig()
        self.api = ApiConfig()
        self.webhook = WebhookConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        self.deployment = DeploymentConfig()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.logging.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get logger
        logger = logging.getLogger('neydra')
        logger.setLevel(getattr(logging, self.logging. level))
        
        # File handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.logging.log_file,
            maxBytes=self.logging.max_bytes,
            backupCount=self.logging.backup_count
        )
        file_handler.setFormatter(logging.Formatter(self.logging.format))
        logger.addHandler(file_handler)
        
        # Console handler
        if self.logging.console_output:
            console_handler = logging. StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            logger.addHandler(console_handler)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'logging': asdict(self.logging),
            'database': asdict(self.database),
            'broker': asdict(self. broker),
            'model': asdict(self.model),
            'prediction': asdict(self. prediction),
            'api': asdict(self.api),
            'webhook': asdict(self. webhook),
            'monitoring': asdict(self.monitoring),
            'security': asdict(self. security),
            'deployment': asdict(self.deployment)
        }
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, filepath: str = 'config.json'):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self. to_json())
        logging.info(f'Configuration saved to {filepath}')
    
    def validate(self) -> bool:
        """Validate critical configuration"""
        errors = []
        
        # Check required database settings
        if self.database.postgres_enabled and not all([
            self. database.postgres_host,
            self.database.postgres_user,
            self.database.postgres_database
        ]):
            errors. append('PostgreSQL configuration incomplete')
        
        # Check required broker settings
        if not self.broker.enabled_brokers:
            errors. append('No brokers configured')
        
        # Check API configuration
        if not self.api.jwt_secret_key or self.api.jwt_secret_key == 'your-secret-key-change-in-production':
            errors.append('JWT secret key not properly configured (change in production! )')
        
        # Check webhook configuration
        if self.webhook.signature_verification_enabled and not self.webhook.webhook_secret_key:
            errors. append('Webhook secret key not configured')
        
        if errors:
            logger = logging.getLogger('neydra')
            for error in errors:
                logger. error(f'Configuration error: {error}')
            return False
        
        logger = logging.getLogger('neydra')
        logger.info('Configuration validation successful')
        return True


# ===== SINGLETON INSTANCE =====
# Create global configuration instance
config = Config()

# Validate configuration on import
config.validate()


# ===== HELPER FUNCTIONS =====
def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return config.logging


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config. database


def get_broker_config() -> BrokerConfig:
    """Get broker configuration"""
    return config.broker


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.model


def get_prediction_config() -> PredictionConfig:
    """Get prediction configuration"""
    return config.prediction


def get_api_config() -> ApiConfig:
    """Get API configuration"""
    return config. api


def get_webhook_config() -> WebhookConfig:
    """Get webhook configuration"""
    return config.webhook


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config.monitoring


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config.security


def get_deployment_config() -> DeploymentConfig:
    """Get deployment configuration"""
    return config.deployment


if __name__ == '__main__': 
    """Print configuration when run directly"""
    print("="*80)
    print("NEYDRA Configuration")
    print("="*80)
    print(config.to_json())
    print("\n" + "="*80)
    print(f"Environment: {config.deployment.environment}")
    print(f"Debug: {config.api.debug}")
    print(f"Brokers: {', '.join(config.broker.enabled_brokers)}")
    print(f"Database: {config.database.postgres_host}:{config.database.postgres_port}")
    print(f"API: {config.api.host}:{config.api.port}")
    print("="*80)