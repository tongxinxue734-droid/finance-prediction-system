-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 创建时间戳函数
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.modified_at = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user', -- admin, user
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- 创建用户表触发器
CREATE TRIGGER update_user_modtime
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 创建股票信息表
CREATE TABLE IF NOT EXISTS stocks (
    symbol VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    exchange VARCHAR(20),
    sector VARCHAR(50),
    industry VARCHAR(100),
    country VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建股票价格表（由数据采集服务使用）
CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL REFERENCES stocks(symbol),
    date DATE NOT NULL,
    open NUMERIC(10, 4) NOT NULL,
    high NUMERIC(10, 4) NOT NULL,
    low NUMERIC(10, 4) NOT NULL,
    close NUMERIC(10, 4) NOT NULL,
    adj_close NUMERIC(10, 4),
    volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- 创建股票指标表
CREATE TABLE IF NOT EXISTS stock_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL REFERENCES stocks(symbol),
    date DATE NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value NUMERIC(10, 6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date, indicator_name)
);

-- 创建数据采集状态表（由数据采集服务使用）
CREATE TABLE IF NOT EXISTS collection_status (
    symbol VARCHAR(20) PRIMARY KEY REFERENCES stocks(symbol),
    last_collected_date DATE,
    last_success_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending', -- pending, success, failed
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建预测模型表（由预测服务使用）
CREATE TABLE IF NOT EXISTS prediction_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL REFERENCES stocks(symbol),
    model_type VARCHAR(50) NOT NULL,
    lookback_days INT NOT NULL,
    train_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    is_default BOOLEAN DEFAULT FALSE
);

-- 创建预测结果表（由预测服务使用）
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL REFERENCES stocks(symbol),
    model_name VARCHAR(100) NOT NULL REFERENCES prediction_models(model_name),
    prediction_date DATE NOT NULL,
    predicted_close FLOAT NOT NULL,
    actual_close FLOAT, -- 实际值，后续可以填充
    accuracy FLOAT, -- 准确度，实际值已知后计算
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, model_name, prediction_date)
);

-- 创建系统日志表
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL, -- api, data_collector, etc.
    log_level VARCHAR(20) NOT NULL, -- debug, info, warning, error
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引提升查询性能
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_stock_indicators_symbol_date ON stock_indicators(symbol, date);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol, prediction_date);
CREATE INDEX IF NOT EXISTS idx_system_logs_service_level ON system_logs(service_name, log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);

-- 插入初始股票数据
INSERT INTO stocks (symbol, name, exchange, sector, country) VALUES
('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology', 'USA'),
('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology', 'USA'),
('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology', 'USA'),
('AMZN', 'Amazon.com Inc.', 'NASDAQ', 'Consumer Cyclical', 'USA'),
('TSLA', 'Tesla Inc.', 'NASDAQ', 'Automotive', 'USA')
ON CONFLICT (symbol) DO NOTHING;

-- 插入管理员用户（默认密码：admin123，实际部署时应修改）
INSERT INTO users (username, email, password_hash, full_name, role) 
VALUES ('admin', 'admin@example.com', '$2b$12$9jVjP8QJz3X3WJZJZJZJZ.3JZJZJZJZJZJZJZJZJZJZJZJZJZJ', 'System Admin', 'admin')
ON CONFLICT (username) DO NOTHING;

-- 创建定期清理旧日志的函数
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS VOID AS $$
BEGIN
    DELETE FROM system_logs WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- 创建定时任务（需要pg_cron扩展）
-- 注意：实际使用前需要安装pg_cron扩展并启用
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('daily_log_cleanup', '0 0 * * *', 'SELECT cleanup_old_logs();');
