-- ===================================================================
-- XAU/USD Predictive Analytics Engine - Database Schema
-- NEYDRA Platform - PostgreSQL
-- Founder & CEO: Ilyes Jarray
-- © 2025 - All Rights Reserved
-- ===================================================================

-- ===== CREATE EXTENSIONS =====
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ===== CREATE SCHEMAS =====
CREATE SCHEMA IF NOT EXISTS neydra;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- ===== SET SEARCH PATH =====
SET search_path TO neydra, public;

-- ===================================================================
-- CORE TABLES
-- ===================================================================

-- ===== USERS TABLE =====
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    api_key VARCHAR(255) UNIQUE,
    api_key_created_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT username_length CHECK (LENGTH(username) >= 3),
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_api_key ON users(api_key);
CREATE INDEX idx_users_is_active ON users(is_active);

-- ===== MARKET DATA TABLE =====
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    open_price NUMERIC(10, 5) NOT NULL,
    high_price NUMERIC(10, 5) NOT NULL,
    low_price NUMERIC(10, 5) NOT NULL,
    close_price NUMERIC(10, 5) NOT NULL,
    volume BIGINT,
    bid_price NUMERIC(10, 5),
    ask_price NUMERIC(10, 5),
    mid_price NUMERIC(10, 5),
    bid_volume BIGINT,
    ask_volume BIGINT,
    spread NUMERIC(10, 6),
    spread_percent NUMERIC(8, 6),
    time_frame VARCHAR(10) DEFAULT 'D',
    source VARCHAR(50),
    data_quality VARCHAR(20) DEFAULT 'GOOD',
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT price_positive CHECK (open_price > 0 AND close_price > 0),
    CONSTRAINT high_low_valid CHECK (high_price >= low_price)
);

CREATE INDEX idx_market_data_pair ON market_data(pair);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX idx_market_data_pair_timestamp ON market_data(pair, timestamp DESC);
CREATE INDEX idx_market_data_source ON market_data(source);
CREATE INDEX idx_market_data_time_frame ON market_data(time_frame);

-- ===== PRICE HISTORY TABLE =====
CREATE TABLE IF NOT EXISTS price_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    bid NUMERIC(10, 5) NOT NULL,
    ask NUMERIC(10, 5) NOT NULL,
    mid NUMERIC(10, 5) NOT NULL,
    bid_volume BIGINT,
    ask_volume BIGINT,
    spread NUMERIC(10, 6),
    spread_percent NUMERIC(8, 6),
    broker VARCHAR(50),
    broker_timestamp TIMESTAMP WITH TIME ZONE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT bid_ask_valid CHECK (bid > 0 AND ask > 0),
    CONSTRAINT bid_ask_relationship CHECK (bid <= ask)
);

CREATE INDEX idx_price_history_pair ON price_history(pair);
CREATE INDEX idx_price_history_timestamp ON price_history(timestamp DESC);
CREATE INDEX idx_price_history_pair_timestamp ON price_history(pair, timestamp DESC);
CREATE INDEX idx_price_history_broker ON price_history(broker);

-- ===== TECHNICAL INDICATORS TABLE =====
CREATE TABLE IF NOT EXISTS technical_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    price_id UUID REFERENCES market_data(id) ON DELETE CASCADE,
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    
    -- Moving Averages
    ma_5 NUMERIC(10, 5),
    ma_20 NUMERIC(10, 5),
    ma_50 NUMERIC(10, 5),
    ma_200 NUMERIC(10, 5),
    ema_12 NUMERIC(10, 5),
    ema_26 NUMERIC(10, 5),
    
    -- Momentum Indicators
    rsi NUMERIC(8, 4),
    rsi_14 NUMERIC(8, 4),
    macd NUMERIC(10, 6),
    macd_signal NUMERIC(10, 6),
    macd_histogram NUMERIC(10, 6),
    stochastic_k NUMERIC(8, 4),
    stochastic_d NUMERIC(8, 4),
    
    -- Volatility Indicators
    bollinger_upper NUMERIC(10, 5),
    bollinger_middle NUMERIC(10, 5),
    bollinger_lower NUMERIC(10, 5),
    atr NUMERIC(10, 5),
    atr_percent NUMERIC(8, 6),
    standard_deviation NUMERIC(10, 6),
    volatility NUMERIC(8, 6),
    
    -- Volume Indicators
    volume_ma NUMERIC(15, 2),
    obv NUMERIC(15, 2),
    ad_line NUMERIC(15, 2),
    
    -- Trend Indicators
    adx NUMERIC(8, 4),
    di_plus NUMERIC(8, 4),
    di_minus NUMERIC(8, 4),
    
    -- Other Metrics
    price_range NUMERIC(10, 5),
    daily_return NUMERIC(10, 6),
    intraday_change_percent NUMERIC(8, 6),
    
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_technical_indicators_price_id ON technical_indicators(price_id);
CREATE INDEX idx_technical_indicators_pair ON technical_indicators(pair);
CREATE INDEX idx_technical_indicators_timestamp ON technical_indicators(timestamp DESC);

-- ===== PREDICTIONS TABLE =====
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Prediction Details
    prediction_type VARCHAR(50) NOT NULL,  -- 'PRICE', 'DIRECTION', 'SIGNAL'
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    
    -- Current Market State
    current_price NUMERIC(10, 5) NOT NULL,
    current_bid NUMERIC(10, 5),
    current_ask NUMERIC(10, 5),
    
    -- Predicted Values
    predicted_price NUMERIC(10, 5),
    price_change NUMERIC(10, 5),
    price_change_percent NUMERIC(8, 6),
    predicted_direction VARCHAR(20),  -- 'UP', 'DOWN', 'NEUTRAL'
    
    -- Model Performance
    model_name VARCHAR(100),
    confidence NUMERIC(8, 6),
    confidence_percent NUMERIC(8, 4),
    model_accuracy NUMERIC(8, 4),
    r2_score NUMERIC(8, 6),
    rmse NUMERIC(10, 5),
    mae NUMERIC(10, 5),
    
    -- Targets
    resistance_level NUMERIC(10, 5),
    support_level NUMERIC(10, 5),
    target_price NUMERIC(10, 5),
    stop_loss NUMERIC(10, 5),
    take_profit NUMERIC(10, 5),
    
    -- Time Frame
    time_horizon VARCHAR(50),  -- '1H', '4H', '1D', '1W', etc.
    prediction_window_minutes INT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'PENDING',  -- 'PENDING', 'ACTIVE', 'COMPLETED', 'EXPIRED'
    is_accurate BOOLEAN,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    validated_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT confidence_valid CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT price_positive CHECK (current_price > 0)
);

CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_pair ON predictions(pair);
CREATE INDEX idx_predictions_status ON predictions(status);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_confidence ON predictions(confidence DESC);
CREATE INDEX idx_predictions_expires_at ON predictions(expires_at);
CREATE INDEX idx_predictions_pair_status ON predictions(pair, status);

-- ===== SIGNALS TABLE =====
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    
    -- Signal Details
    signal_type VARCHAR(50) NOT NULL,  -- 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    
    -- Price Information
    signal_price NUMERIC(10, 5) NOT NULL,
    entry_price NUMERIC(10, 5),
    exit_price NUMERIC(10, 5),
    
    -- Signal Strength
    strength NUMERIC(8, 6),  -- 0.0 to 1.0
    strength_percent NUMERIC(8, 4),
    confidence NUMERIC(8, 6),
    confidence_percent NUMERIC(8, 4),
    
    -- Risk Assessment
    risk_level VARCHAR(20),  -- 'LOW', 'MEDIUM', 'HIGH'
    risk_score NUMERIC(8, 4),
    risk_reward_ratio NUMERIC(8, 4),
    
    -- Signal Source
    source VARCHAR(50),  -- 'ML_MODEL', 'TECHNICAL', 'SENTIMENT', 'COMBINED'
    model_name VARCHAR(100),
    
    -- Indicators Contributing to Signal
    contributing_indicators TEXT[],  -- Array of indicators
    
    -- Signal Performance
    status VARCHAR(20) DEFAULT 'PENDING',  -- 'PENDING', 'TRIGGERED', 'COMPLETED', 'FAILED', 'CANCELLED'
    actual_profit NUMERIC(10, 5),
    actual_return_percent NUMERIC(8, 6),
    win_probability NUMERIC(8, 6),
    
    -- Time Information
    time_frame VARCHAR(10),
    validity_period INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    triggered_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT signal_strength_valid CHECK (strength BETWEEN 0 AND 1),
    CONSTRAINT confidence_valid CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT price_positive CHECK (signal_price > 0)
);

CREATE INDEX idx_signals_user_id ON signals(user_id);
CREATE INDEX idx_signals_prediction_id ON signals(prediction_id);
CREATE INDEX idx_signals_pair ON signals(pair);
CREATE INDEX idx_signals_type ON signals(signal_type);
CREATE INDEX idx_signals_status ON signals(status);
CREATE INDEX idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX idx_signals_strength ON signals(strength DESC);
CREATE INDEX idx_signals_confidence ON signals(confidence DESC);
CREATE INDEX idx_signals_pair_status ON signals(pair, status);

-- ===== BROKER DATA TABLE =====
CREATE TABLE IF NOT EXISTS broker_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    broker_name VARCHAR(50) NOT NULL,
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    
    -- Price Data
    bid NUMERIC(10, 5) NOT NULL,
    ask NUMERIC(10, 5) NOT NULL,
    mid NUMERIC(10, 5),
    
    -- Volume Data
    bid_volume BIGINT,
    ask_volume BIGINT,
    total_volume BIGINT,
    
    -- Spread Information
    spread NUMERIC(10, 6),
    spread_percent NUMERIC(8, 6),
    
    -- Quality Metrics
    data_quality VARCHAR(20),
    latency_ms INTEGER,
    
    -- Account Information
    account_balance NUMERIC(15, 2),
    account_equity NUMERIC(15, 2),
    account_margin NUMERIC(15, 2),
    
    -- Metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    broker_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT bid_ask_valid CHECK (bid > 0 AND ask > 0),
    CONSTRAINT bid_ask_relationship CHECK (bid <= ask)
);

CREATE INDEX idx_broker_data_broker_name ON broker_data(broker_name);
CREATE INDEX idx_broker_data_pair ON broker_data(pair);
CREATE INDEX idx_broker_data_timestamp ON broker_data(timestamp DESC);
CREATE INDEX idx_broker_data_broker_pair_timestamp ON broker_data(broker_name, pair, timestamp DESC);

-- ===== ALERTS TABLE =====
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    signal_id UUID REFERENCES signals(id) ON DELETE SET NULL,
    
    -- Alert Details
    alert_type VARCHAR(50) NOT NULL,  -- 'PRICE_ALERT', 'SIGNAL_ALERT', 'RISK_ALERT', 'ERROR_ALERT'
    severity VARCHAR(20) DEFAULT 'INFO',  -- 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'
    title VARCHAR(255) NOT NULL,
    message TEXT,
    
    -- Alert Conditions
    pair VARCHAR(20),
    price_threshold NUMERIC(10, 5),
    condition VARCHAR(50),  -- 'ABOVE', 'BELOW', 'EQUAL'
    
    -- Alert Status
    status VARCHAR(20) DEFAULT 'ACTIVE',  -- 'ACTIVE', 'TRIGGERED', 'ACKNOWLEDGED', 'DISMISSED'
    triggered_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    
    -- Notifications Sent
    email_sent BOOLEAN DEFAULT false,
    slack_sent BOOLEAN DEFAULT false,
    telegram_sent BOOLEAN DEFAULT false,
    sms_sent BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT title_not_empty CHECK (LENGTH(title) > 0)
);

CREATE INDEX idx_alerts_user_id ON alerts(user_id);
CREATE INDEX idx_alerts_signal_id ON alerts(signal_id);
CREATE INDEX idx_alerts_pair ON alerts(pair);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_created_at ON alerts(created_at DESC);

-- ===== EVENTS TABLE =====
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Event Data
    source VARCHAR(50),  -- 'API', 'WEBHOOK', 'SCHEDULER', 'MANUAL'
    data JSONB,
    
    -- Status
    status VARCHAR(20) DEFAULT 'PROCESSED',  -- 'PENDING', 'PROCESSING', 'PROCESSED', 'FAILED'
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_user_id ON events(user_id);
CREATE INDEX idx_events_source ON events(source);
CREATE INDEX idx_events_status ON events(status);
CREATE INDEX idx_events_created_at ON events(created_at DESC);
CREATE INDEX idx_events_data ON events USING gin(data);

-- ===================================================================
-- ANALYTICS SCHEMA TABLES
-- ===================================================================

-- ===== MODEL PERFORMANCE TABLE =====
CREATE TABLE IF NOT EXISTS analytics. model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    -- Performance Metrics
    accuracy NUMERIC(8, 6),
    precision NUMERIC(8, 6),
    recall NUMERIC(8, 6),
    f1_score NUMERIC(8, 6),
    r2_score NUMERIC(8, 6),
    rmse NUMERIC(10, 5),
    mae NUMERIC(10, 5),
    mape NUMERIC(8, 6),
    
    -- Backtesting Results
    total_trades INT,
    winning_trades INT,
    losing_trades INT,
    win_rate NUMERIC(8, 6),
    profit_factor NUMERIC(8, 4),
    sharpe_ratio NUMERIC(8, 4),
    max_drawdown NUMERIC(8, 6),
    
    -- Period
    period_start DATE,
    period_end DATE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_performance_model_name ON analytics. model_performance(model_name);
CREATE INDEX idx_model_performance_period ON analytics.model_performance(period_start, period_end);

-- ===== SIGNAL PERFORMANCE TABLE =====
CREATE TABLE IF NOT EXISTS analytics.signal_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id) ON DELETE CASCADE,
    
    -- Execution Details
    entry_price NUMERIC(10, 5),
    entry_time TIMESTAMP WITH TIME ZONE,
    exit_price NUMERIC(10, 5),
    exit_time TIMESTAMP WITH TIME ZONE,
    
    -- Performance Metrics
    profit_loss NUMERIC(10, 5),
    profit_loss_percent NUMERIC(8, 6),
    return_percent NUMERIC(8, 6),
    holding_period INTERVAL,
    
    -- Risk Metrics
    max_profit NUMERIC(10, 5),
    max_loss NUMERIC(10, 5),
    risk_reward_ratio NUMERIC(8, 4),
    
    -- Accuracy
    prediction_accuracy BOOLEAN,
    actual_direction VARCHAR(20),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signal_performance_signal_id ON analytics.signal_performance(signal_id);
CREATE INDEX idx_signal_performance_created_at ON analytics.signal_performance(created_at DESC);

-- ===== DAILY STATISTICS TABLE =====
CREATE TABLE IF NOT EXISTS analytics.daily_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    pair VARCHAR(20) NOT NULL DEFAULT 'XAU/USD',
    
    -- Price Statistics
    open_price NUMERIC(10, 5),
    high_price NUMERIC(10, 5),
    low_price NUMERIC(10, 5),
    close_price NUMERIC(10, 5),
    daily_range NUMERIC(10, 5),
    daily_change NUMERIC(10, 5),
    daily_change_percent NUMERIC(8, 6),
    
    -- Volume Statistics
    total_volume BIGINT,
    average_volume BIGINT,
    
    -- Signal Statistics
    total_signals INT,
    buy_signals INT,
    sell_signals INT,
    winning_signals INT,
    signal_accuracy NUMERIC(8, 6),
    
    -- Prediction Statistics
    total_predictions INT,
    correct_predictions INT,
    prediction_accuracy NUMERIC(8, 6),
    average_confidence NUMERIC(8, 6),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_daily_statistics_date ON analytics.daily_statistics(date DESC);
CREATE INDEX idx_daily_statistics_pair ON analytics.daily_statistics(pair);

-- ===================================================================
-- MONITORING SCHEMA TABLES
-- ===================================================================

-- ===== SYSTEM METRICS TABLE =====
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- System Health
    cpu_usage NUMERIC(8, 2),
    memory_usage NUMERIC(8, 2),
    disk_usage NUMERIC(8, 2),
    
    -- Application Metrics
    active_connections INT,
    request_count INT,
    error_count INT,
    average_response_time_ms NUMERIC(10, 2),
    
    -- API Metrics
    api_requests INT,
    api_errors INT,
    api_average_latency_ms NUMERIC(10, 2),
    
    -- Database Metrics
    db_connections INT,
    db_query_time_ms NUMERIC(10, 2),
    db_slow_queries INT,
    
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp DESC);

-- ===== ERROR LOG TABLE =====
CREATE TABLE IF NOT EXISTS monitoring.error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Error Details
    error_type VARCHAR(100),
    error_message TEXT,
    error_trace TEXT,
    
    -- Context
    service VARCHAR(100),
    endpoint VARCHAR(255),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Request Details
    request_method VARCHAR(10),
    request_path VARCHAR(255),
    request_body JSONB,
    response_status INT,
    
    -- Metadata
    severity VARCHAR(20),  -- 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    resolved BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_error_logs_error_type ON monitoring. error_logs(error_type);
CREATE INDEX idx_error_logs_service ON monitoring.error_logs(service);
CREATE INDEX idx_error_logs_severity ON monitoring.error_logs(severity);
CREATE INDEX idx_error_logs_created_at ON monitoring.error_logs(created_at DESC);
CREATE INDEX idx_error_logs_resolved ON monitoring.error_logs(resolved);

-- ===================================================================
-- MATERIALIZED VIEWS
-- ===================================================================

-- ===== RECENT PREDICTIONS SUMMARY =====
CREATE MATERIALIZED VIEW IF NOT EXISTS public.v_recent_predictions AS
SELECT 
    p.id,
    p.pair,
    p.current_price,
    p.predicted_price,
    p.predicted_direction,
    p.confidence,
    p.status,
    p.created_at,
    CASE 
        WHEN p. predicted_direction = 'UP' AND p.current_price < p.predicted_price THEN true
        WHEN p.predicted_direction = 'DOWN' AND p.current_price > p.predicted_price THEN true
        ELSE false
    END as is_accurate
FROM predictions p
WHERE p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY p. created_at DESC;

CREATE INDEX idx_v_recent_predictions_pair ON v_recent_predictions(pair);

-- ===== ACTIVE SIGNALS SUMMARY =====
CREATE MATERIALIZED VIEW IF NOT EXISTS public.v_active_signals AS
SELECT 
    s.id,
    s.pair,
    s.signal_type,
    s.signal_price,
    s.strength,
    s.confidence,
    s.risk_level,
    COUNT(*) OVER (PARTITION BY s.signal_type) as signal_count,
    s.created_at
FROM signals s
WHERE s.status IN ('PENDING', 'TRIGGERED')
ORDER BY s.strength DESC;

CREATE INDEX idx_v_active_signals_pair ON v_active_signals(pair);

-- ===== BROKER COMPARISON VIEW =====
CREATE MATERIALIZED VIEW IF NOT EXISTS public. v_broker_comparison AS
SELECT 
    broker_name,
    pair,
    AVG(bid) as avg_bid,
    AVG(ask) as avg_ask,
    AVG((ask - bid)) as avg_spread,
    MIN(timestamp) as data_from,
    MAX(timestamp) as data_to,
    COUNT(*) as quote_count
FROM broker_data
GROUP BY broker_name, pair;

-- ===================================================================
-- FUNCTIONS & TRIGGERS
-- ===================================================================

-- ===== UPDATE TIMESTAMP FUNCTION =====
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ===== TRIGGER FOR USERS =====
CREATE TRIGGER trigger_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ===== TRIGGER FOR PREDICTIONS =====
CREATE TRIGGER trigger_predictions_updated_at
BEFORE UPDATE ON predictions
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ===== FUNCTION TO GET SIGNAL STATISTICS =====
CREATE OR REPLACE FUNCTION get_signal_statistics(
    p_pair VARCHAR DEFAULT 'XAU/USD',
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    total_signals BIGINT,
    buy_signals BIGINT,
    sell_signals BIGINT,
    hold_signals BIGINT,
    win_rate NUMERIC,
    average_confidence NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_signals,
        COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END)::BIGINT as buy_signals,
        COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END)::BIGINT as sell_signals,
        COUNT(CASE WHEN signal_type = 'HOLD' THEN 1 END)::BIGINT as hold_signals,
        (COUNT(CASE WHEN actual_return_percent > 0 THEN 1 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100) as win_rate,
        AVG(confidence) as average_confidence
    FROM signals
    WHERE pair = p_pair
    AND created_at > CURRENT_TIMESTAMP - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- ===== FUNCTION TO GET PREDICTION ACCURACY =====
CREATE OR REPLACE FUNCTION get_prediction_accuracy(
    p_pair VARCHAR DEFAULT 'XAU/USD',
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    total_predictions BIGINT,
    accurate_predictions BIGINT,
    accuracy_percent NUMERIC,
    average_confidence NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_predictions,
        COUNT(CASE WHEN is_accurate THEN 1 END)::BIGINT as accurate_predictions,
        (COUNT(CASE WHEN is_accurate THEN 1 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100) as accuracy_percent,
        AVG(confidence) as average_confidence
    FROM predictions
    WHERE pair = p_pair
    AND created_at > CURRENT_TIMESTAMP - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- ===== FUNCTION TO GET BEST PERFORMING MODEL =====
CREATE OR REPLACE FUNCTION get_best_performing_model(
    p_period_days INT DEFAULT 30
)
RETURNS TABLE (
    model_name VARCHAR,
    accuracy NUMERIC,
    win_rate NUMERIC,
    sharpe_ratio NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ap.model_name,
        ap.accuracy,
        ap.win_rate,
        ap.sharpe_ratio
    FROM analytics. model_performance ap
    WHERE ap.period_start > CURRENT_DATE - (p_period_days || ' days')::INTERVAL
    ORDER BY ap.accuracy DESC, ap.sharpe_ratio DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ===================================================================
-- GRANTS & PERMISSIONS
-- ===================================================================

-- Create application role (replace 'neydra_app' with your actual app user)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'neydra_app') THEN
        CREATE ROLE neydra_app WITH LOGIN PASSWORD 'change_this_password_in_production';
    END IF;
END
$$;

-- Grant permissions to neydra_app
GRANT CONNECT ON DATABASE neydra TO neydra_app;
GRANT USAGE ON SCHEMA neydra, analytics, monitoring TO neydra_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA neydra TO neydra_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA analytics TO neydra_app;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO neydra_app;
GRANT SELECT ON ALL MATERIALIZED VIEWS IN SCHEMA public TO neydra_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA neydra TO neydra_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics TO neydra_app;

-- ===================================================================
-- CLEANUP & MAINTENANCE
-- ===================================================================

-- Create partition tables for large datasets (optional)
-- Uncomment if you have large volume

-- ALTER TABLE market_data PARTITION BY RANGE (date_trunc('month', timestamp));
-- ALTER TABLE price_history PARTITION BY RANGE (date_trunc('month', timestamp));
-- ALTER TABLE signals PARTITION BY RANGE (date_trunc('month', created_at));

-- ===================================================================
-- FINAL VERIFICATION
-- ===================================================================

-- Verify all tables are created
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'neydra' 
ORDER BY table_name;

-- Display summary
SELECT 
    COUNT(*) as total_tables,
    'NEYDRA Database Schema' as schema_name
FROM information_schema.tables 
WHERE table_schema = 'neydra';

-- Verify indexes
SELECT indexname FROM pg_indexes 
WHERE schemaname = 'neydra'
ORDER BY indexname;

COMMIT;