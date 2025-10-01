from fastapi import FastAPI, HTTPException, Query
import psycopg2
import redis
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_processor")

# 初始化FastAPI应用
app = FastAPI(
    title="金融数据处理服务",
    description="提供金融数据清洗和指标计算功能",
    version="1.0.0"
)

# 数据库连接
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME")
    )

# Redis连接
def get_redis_connection():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=os.getenv("REDIS_PORT", "6379"),
        decode_responses=True
    )

class DataProcessor:
    @staticmethod
    def get_stock_data(symbol: str, days: int = 90) -> pd.DataFrame:
        """获取股票数据并转换为DataFrame"""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            conn = get_db_connection()
            query = """
                SELECT date, open, high, low, close, volume 
                FROM stock_prices 
                WHERE symbol = %s AND date BETWEEN %s AND %s 
                ORDER BY date
            """
            
            df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到 {symbol} 的数据")
                return pd.DataFrame()
                
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")

    @staticmethod
    def calculate_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """计算简单移动平均线(SMA)"""
        df_copy = df.copy()
        df_copy[f'sma_{window}'] = df_copy['close'].rolling(window=window).mean()
        return df_copy

    @staticmethod
    def calculate_ema(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """计算指数移动平均线(EMA)"""
        df_copy = df.copy()
        df_copy[f'ema_{window}'] = df_copy['close'].ewm(span=window, adjust=False).mean()
        return df_copy

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """计算相对强弱指数(RSI)"""
        df_copy = df.copy()
        delta = df_copy['close'].diff(1)
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        df_copy[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        return df_copy

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
        """计算指数平滑异同平均线(MACD)"""
        df_copy = df.copy()
        
        # 计算12期和26期EMA
        ema12 = df_copy['close'].ewm(span=12, adjust=False).mean()
        ema26 = df_copy['close'].ewm(span=26, adjust=False).mean()
        
        # MACD线 = 12期EMA - 26期EMA
        df_copy['macd'] = ema12 - ema26
        
        # 信号线 = MACD线的9期EMA
        df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
        
        # 差离值 = MACD线 - 信号线
        df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
        
        return df_copy

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """计算布林带(Bollinger Bands)"""
        df_copy = df.copy()
        
        # 计算中轨（20期SMA）
        df_copy['bb_mid'] = df_copy['close'].rolling(window=window).mean()
        
        # 计算标准差
        df_copy['bb_std'] = df_copy['close'].rolling(window=window).std()
        
        # 上轨 = 中轨 + 2*标准差
        df_copy['bb_upper'] = df_copy['bb_mid'] + 2 * df_copy['bb_std']
        
        # 下轨 = 中轨 - 2*标准差
        df_copy['bb_lower'] = df_copy['bb_mid'] - 2 * df_copy['bb_std']
        
        # 不需要保留标准差列
        df_copy.drop(columns=['bb_std'], inplace=True)
        
        return df_copy

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df_with_indicators = df.copy()
        
        # 计算各种指标
        df_with_indicators = DataProcessor.calculate_sma(df_with_indicators, 20)
        df_with_indicators = DataProcessor.calculate_sma(df_with_indicators, 50)
        df_with_indicators = DataProcessor.calculate_ema(df_with_indicators, 20)
        df_with_indicators = DataProcessor.calculate_rsi(df_with_indicators, 14)
        df_with_indicators = DataProcessor.calculate_macd(df_with_indicators)
        df_with_indicators = DataProcessor.calculate_bollinger_bands(df_with_indicators, 20)
        
        return df_with_indicators

# 健康检查接口
@app.get("/health", tags=["系统"])
def health_check():
    return {"status": "healthy", "service": "data_processor"}

# 计算并获取技术指标
@app.get("/indicators", tags=["指标计算"])
def get_indicators(
    symbol: str = Query(..., description="股票代码"),
    indicator: str = Query(None, description="特定指标名称，如'sma'、'rsi'等，不指定则返回所有指标"),
    days: int = Query(90, description="计算指标的天数")
):
    try:
        # 检查缓存
        r = get_redis_connection()
        cache_key = f"indicators:{symbol}:{days}:{indicator or 'all'}"
        cached_data = r.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # 获取股票数据
        df = DataProcessor.get_stock_data(symbol, days)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的数据")
        
        # 根据请求计算指标
        if not indicator or indicator == "all":
            result_df = DataProcessor.calculate_all_indicators(df)
        elif indicator.startswith("sma"):
            window = int(indicator.split("_")[1]) if "_" in indicator else 20
            result_df = DataProcessor.calculate_sma(df, window)
        elif indicator.startswith("ema"):
            window = int(indicator.split("_")[1]) if "_" in indicator else 20
            result_df = DataProcessor.calculate_ema(df, window)
        elif indicator.startswith("rsi"):
            window = int(indicator.split("_")[1]) if "_" in indicator else 14
            result_df = DataProcessor.calculate_rsi(df, window)
        elif indicator == "macd":
            result_df = DataProcessor.calculate_macd(df)
        elif indicator == "bollinger":
            result_df = DataProcessor.calculate_bollinger_bands(df)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的指标: {indicator}")
        
        # 转换为JSON格式
        result_df.reset_index(inplace=True)
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        result = result_df.to_dict('records')
        
        # 存入缓存，有效期30分钟
        r.setex(cache_key, 1800, json.dumps(result))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"计算指标失败: {str(e)}")

# 批量处理数据
@app.post("/process-all", tags=["批量处理"])
def process_all_data():
    try:
        # 获取所有股票代码
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM stock_prices")
            symbols = [row[0] for row in cur.fetchall()]
        conn.close()
        
        # 逐个处理
        results = []
        for symbol in symbols:
            try:
                df = DataProcessor.get_stock_data(symbol)
                if not df.empty:
                    result_df = DataProcessor.calculate_all_indicators(df)
                    results.append({
                        "symbol": symbol,
                        "status": "success",
                        "records_processed": len(result_df)
                    })
                    
                    # 清除旧缓存
                    r = get_redis_connection()
                    for key in r.keys(f"indicators:{symbol}:*"):
                        r.delete(key)
                else:
                    results.append({
                        "symbol": symbol,
                        "status": "warning",
                        "message": "没有数据可处理"
                    })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "status": "completed",
            "processed_count": len(symbols),
            "results": results
        }
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")
