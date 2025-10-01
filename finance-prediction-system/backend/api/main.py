from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import redis
import os
import requests
from datetime import datetime, timedelta

# 初始化FastAPI应用
app = FastAPI(
    title="金融预测系统API",
    description="提供金融数据查询、处理和预测相关接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库连接
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME")
    )
    return conn

# Redis连接
def get_redis_connection():
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=os.getenv("REDIS_PORT", "6379"),
        decode_responses=True
    )
    return r

# 数据模型
class StockDataRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str

class PredictionRequest(BaseModel):
    symbol: str
    days: int = 7

# 健康检查接口
@app.get("/health", tags=["系统"])
def health_check():
    return {"status": "healthy", "service": "api"}

# 获取股票列表
@app.get("/stocks", tags=["股票数据"])
def get_stocks():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol, name FROM stock_prices ORDER BY symbol")
            stocks = cur.fetchall()
            return [{"symbol": s[0], "name": s[1]} for s in stocks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库错误: {str(e)}")
    finally:
        conn.close()

# 获取股票历史数据
@app.post("/stock/history", tags=["股票数据"])
def get_stock_history(request: StockDataRequest):
    try:
        # 先检查缓存
        r = get_redis_connection()
        cache_key = f"stock:history:{request.symbol}:{request.start_date}:{request.end_date}"
        cached_data = r.get(cache_key)
        
        if cached_data:
            import json
            return json.loads(cached_data)
        
        # 缓存未命中，查询数据库
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, volume 
                FROM stock_prices 
                WHERE symbol = %s AND date BETWEEN %s AND %s 
                ORDER BY date
            """, (request.symbol, request.start_date, request.end_date))
            
            columns = [desc[0] for desc in cur.description]
            data = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            # 存入缓存，有效期1小时
            r.setex(cache_key, 3600, json.dumps(data))
            
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

# 获取技术指标
@app.get("/stock/indicators/{symbol}", tags=["股票数据"])
def get_stock_indicators(symbol: str, indicator: str = None):
    try:
        # 调用数据处理服务
        processor_url = os.getenv("DATA_PROCESSOR_URL", "http://data_processor:8001")
        params = {"symbol": symbol}
        if indicator:
            params["indicator"] = indicator
            
        response = requests.get(f"{processor_url}/indicators", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")

# 获取预测结果
@app.post("/predict", tags=["预测"])
def get_prediction(request: PredictionRequest):
    try:
        # 调用预测服务
        prediction_url = os.getenv("PREDICTION_SERVICE_URL", "http://prediction_service:8002")
        response = requests.post(f"{prediction_url}/predict", json=request.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预测失败: {str(e)}")

# 获取模型列表
@app.get("/models", tags=["模型管理"])
def get_models():
    try:
        prediction_url = os.getenv("PREDICTION_SERVICE_URL", "http://prediction_service:8002")
        response = requests.get(f"{prediction_url}/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

# 触发数据采集
@app.post("/collect/{symbol}", tags=["数据管理"])
def trigger_collection(symbol: str):
    try:
        # 调用数据采集服务API（实际项目中应通过消息队列实现）
        # 这里简化处理，直接返回成功
        return {"status": "success", "message": f"已触发 {symbol} 数据采集"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"触发采集失败: {str(e)}")
