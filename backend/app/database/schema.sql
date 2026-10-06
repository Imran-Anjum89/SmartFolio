-- SmartFolio PostgreSQL Schema Definition
-- Compatible with SQLite and PostgreSQL

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolios (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(id),
    name VARCHAR(128),
    investment_amount DOUBLE PRECISION NOT NULL,
    risk_profile VARCHAR(32) NOT NULL,
    strategy VARCHAR(32) NOT NULL,
    expected_return DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_assets (
    id VARCHAR(64) PRIMARY KEY,
    portfolio_id VARCHAR(64) REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    investment_amount DOUBLE PRECISION NOT NULL,
    predicted_return DOUBLE PRECISION,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS stock_predictions (
    id VARCHAR(64) PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    prediction_date DATE NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    predicted_return DOUBLE PRECISION NOT NULL,
    actual_return DOUBLE PRECISION,
    prediction_error DOUBLE PRECISION,
    absolute_error DOUBLE PRECISION,
    squared_error DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_feedback (
    id VARCHAR(64) PRIMARY KEY,
    prediction_id VARCHAR(64),
    symbol VARCHAR(32) NOT NULL,
    predicted_return DOUBLE PRECISION NOT NULL,
    actual_return DOUBLE PRECISION NOT NULL,
    prediction_error DOUBLE PRECISION NOT NULL,
    feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(32) NOT NULL,
    used_for_retraining BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_promotions (
    id VARCHAR(64) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    champion_version VARCHAR(32) NOT NULL,
    challenger_version VARCHAR(32) NOT NULL,
    decision VARCHAR(64) NOT NULL,
    promoted BOOLEAN NOT NULL,
    champion_sharpe DOUBLE PRECISION,
    challenger_sharpe DOUBLE PRECISION,
    sharpe_diff DOUBLE PRECISION,
    reason TEXT
);
