from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import psycopg2
import redis
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("prediction_service")

# 初始化FastAPI应用
app = FastAPI(
    title="金融预测服务",
    description="提供股票价格预测和模型管理功能",
    version="1.0.0"
)

# 模型保存目录
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

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

# 数据模型
class PredictionRequest(BaseModel):
    symbol: str
    days: int = 7
    model_name: str = None

class TrainRequest(BaseModel):
    symbol: str
    model_type: str = "random_forest"  # 或 "linear_regression"
    lookback_days: int = 30
    test_size: float = 0.2

# 初始化数据库表
def init_database():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 创建模型信息表
            cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                lookback_days INT NOT NULL,
                train_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mse FLOAT,
                mae FLOAT,
                r2 FLOAT,
                is_default BOOLEAN DEFAULT FALSE
            )
            """)
            
            # 创建预测结果表
            cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_close FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, model_name, prediction_date)
            )
            """)
        
        conn.commit()
        logger.info("预测服务数据库表初始化完成")
    except Exception as e:
        logger.error(f"预测服务数据库初始化失败: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# 初始化时创建表
init_database()

class PredictionModel:
    @staticmethod
    def get_stock_data_with_indicators(symbol: str, days: int = 365) -> pd.DataFrame:
        """获取股票数据及技术指标"""
        try:
            # 从数据处理服务获取带指标的数据
            # 实际项目中应通过API调用，这里简化处理
            from data_processor.service import DataProcessor
            
            df = DataProcessor.get_stock_data(symbol, days)
            if df.empty:
                return pd.DataFrame()
                
            # 计算技术指标
            df = DataProcessor.calculate_all_indicators(df)
            
            # 移除包含NaN的行
            df = df.dropna()
            
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 带指标的数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def create_features(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
        """创建用于预测的特征"""
        df_copy = df.copy()
        
        # 创建滞后特征
        for i in range(1, lookback_days + 1):
            df_copy[f'lag_{i}'] = df_copy['close'].shift(i)
        
        # 创建移动平均特征
        df_copy['sma_5'] = df_copy['close'].rolling(window=5).mean()
        df_copy['sma_10'] = df_copy['close'].rolling(window=10).mean()
        
        # 创建价格变化率特征
        df_copy['return_1d'] = df_copy['close'].pct_change(1)
        df_copy['return_5d'] = df_copy['close'].pct_change(5)
        
        # 目标变量：未来一天的收盘价
        df_copy['target'] = df_copy['close'].shift(-1)
        
        # 移除包含NaN的行
        df_copy = df_copy.dropna()
        
        return df_copy

    @staticmethod
    def train_model(symbol: str, model_type: str = "random_forest", lookback_days: int = 30, test_size: float = 0.2):
        """训练预测模型"""
        try:
            # 获取数据
            df = PredictionModel.get_stock_data_with_indicators(symbol)
            if df.empty:
                raise Exception(f"没有足够的 {symbol} 数据用于训练")
            
            # 创建特征
            feature_df = PredictionModel.create_features(df, lookback_days)
            if len(feature_df) < 100:  # 需要足够的样本
                raise Exception(f"样本数量不足，无法训练模型")
            
            # 准备特征和目标变量
            features = [col for col in feature_df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            X = feature_df[features]
            y = feature_df['target']
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # 选择模型类型
            if model_type == "linear_regression":
                model = LinearRegression()
            elif model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise Exception(f"不支持的模型类型: {model_type}")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            
            logger.info(f"{symbol} {model_type} 模型训练完成 - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # 保存模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{symbol}_{model_type}_{timestamp}"
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            joblib.dump({
                'model': model,
                'features': features,
                'lookback_days': lookback_days,
                'train_date': datetime.now()
            }, model_path)
            
            # 保存模型信息到数据库
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO prediction_models 
                (model_name, symbol, model_type, lookback_days, mse, mae, r2, is_default)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (model_name, symbol, model_type, lookback_days, mse, mae, r2, False))
                
                # 如果是该股票的第一个模型，设为默认模型
                cur.execute("""
                SELECT COUNT(*) FROM prediction_models WHERE symbol = %s
                """, (symbol,))
                count = cur.fetchone()[0]
                if count == 1:
                    cur.execute("""
                    UPDATE prediction_models 
                    SET is_default = TRUE 
                    WHERE model_name = %s
                    """, (model_name,))
            
            conn.commit()
            conn.close()
            
            # 生成预测可视化图
            plot_data = PredictionModel.generate_prediction_plot(
                y_test.values, y_pred, symbol, model_name
            )
            
            return {
                "status": "success",
                "model_name": model_name,
                "symbol": symbol,
                "model_type": model_type,
                "metrics": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "r2": float(r2)
                },
                "plot": plot_data
            }
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise

    @staticmethod
    def generate_prediction_plot(y_true, y_pred, symbol, model_name):
        """生成预测结果对比图"""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='实际价格')
            plt.plot(y_pred, label='预测价格', alpha=0.7)
            plt.title(f'{symbol} 价格预测对比 - {model_name}')
            plt.xlabel('时间')
            plt.ylabel('价格')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存到内存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # 转换为base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            
            return img_base64
        except Exception as e:
            logger.error(f"生成预测图表失败: {str(e)}")
            return None

    @staticmethod
    def load_model(model_name: str):
        """加载模型"""
        try:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                raise Exception(f"模型 {model_name} 不存在")
                
            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    @staticmethod
    def get_default_model(symbol: str):
        """获取股票的默认模型"""
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                SELECT model_name FROM prediction_models 
                WHERE symbol = %s AND is_default = TRUE
                """, (symbol,))
                result = cur.fetchone()
            
            conn.close()
            
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"获取默认模型失败: {str(e)}")
            return None

    @staticmethod
    def predict_future(symbol: str, days: int = 7, model_name: str = None):
        """预测未来价格"""
        try:
            # 如果没有指定模型，使用默认模型
            if not model_name:
                model_name = PredictionModel.get_default_model(symbol)
                if not model_name:
                    raise Exception(f"没有找到 {symbol} 的默认模型，请先训练模型")
            
            # 加载模型
            model_data = PredictionModel.load_model(model_name)
            model = model_data['model']
            features = model_data['features']
            lookback_days = model_data['lookback_days']
            
            # 获取最新数据
            df = PredictionModel.get_stock_data_with_indicators(symbol, lookback_days + 30)
            if df.empty:
                raise Exception(f"获取 {symbol} 数据失败")
            
            # 创建特征
            feature_df = PredictionModel.create_features(df, lookback_days)
            if feature_df.empty:
                raise Exception(f"无法为 {symbol} 创建特征")
            
            # 获取最新的特征数据
            last_features = feature_df.iloc[-1][features].values.reshape(1, -1)
            
            # 预测未来价格
            predictions = []
            current_date = df.index[-1]
            
            # 使用滚动预测
            temp_df = df.copy()
            for i in range(days):
                # 预测下一天
                next_pred = model.predict(last_features)[0]
                
                # 生成下一个日期（跳过周末）
                next_date = current_date + timedelta(days=i+1)
                while next_date.weekday() >= 5:  # 5是周六，6是周日
                    next_date += timedelta(days=1)
                
                predictions.append({
                    "date": next_date.strftime("%Y-%m-%d"),
                    "predicted_close": float(next_pred)
                })
                
                # 更新特征数据用于下一次预测
                current_date = next_date
                
                # 这是一个简化的滚动预测方法，实际应用中应更复杂
                # 这里我们只是简单地将预测值添加到临时数据中
                
            # 保存预测结果到数据库
            conn = get_db_connection()
            with conn.cursor() as cur:
                for pred in predictions:
                    cur.execute("""
                    INSERT INTO predictions 
                    (symbol, model_name, prediction_date, predicted_close)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (symbol, model_name, prediction_date) DO UPDATE 
                    SET predicted_close = EXCLUDED.predicted_close,
                        created_at = CURRENT_TIMESTAMP
                    """, (symbol, model_name, pred["date"], pred["predicted_close"]))
            
            conn.commit()
            conn.close()
            
            # 清除缓存
            r = get_redis_connection()
            r.delete(f"predictions:{symbol}:{model_name}")
            
            return {
                "status": "success",
                "symbol": symbol,
                "model_name": model_name,
                "predictions": predictions
            }
        except Exception as e:
            logger.error(f"价格预测失败: {str(e)}")
            raise

    @staticmethod
    def get_all_models(symbol: str = None):
        """获取所有模型列表"""
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                    SELECT model_name, symbol, model_type, lookback_days, 
                           train_date, mse, mae, r2, is_default 
                    FROM prediction_models 
                    WHERE symbol = %s
                    ORDER BY train_date DESC
                    """, (symbol,))
                else:
                    cur.execute("""
                    SELECT model_name, symbol, model_type, lookback_days, 
                           train_date, mse, mae, r2, is_default 
                    FROM prediction_models 
                    ORDER BY train_date DESC
                    """)
                
                columns = [desc[0] for desc in cur.description]
                models = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            conn.close()
            return models
        except Exception as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            raise

    @staticmethod
    def set_default_model(model_name: str):
        """设置默认模型"""
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # 获取模型信息
                cur.execute("""
                SELECT symbol FROM prediction_models 
                WHERE model_name = %s
                """, (model_name,))
                result = cur.fetchone()
                
                if not result:
                    conn.close()
                    raise Exception(f"模型 {model_name} 不存在")
                
                symbol = result[0]
                
                # 将该股票的所有模型设为非默认
                cur.execute("""
                UPDATE prediction_models 
                SET is_default = FALSE 
                WHERE symbol = %s
                """, (symbol,))
                
                # 将指定模型设为默认
                cur.execute("""
                UPDATE prediction_models 
                SET is_default = TRUE 
                WHERE model_name = %s
                """, (model_name,))
            
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "message": f"模型 {model_name} 已设为 {symbol} 的默认模型"
            }
        except Exception as e:
            logger.error(f"设置默认模型失败: {str(e)}")
            if conn:
                conn.rollback()
            raise

# 健康检查接口
@app.get("/health", tags=["系统"])
def health_check():
    return {"status": "healthy", "service": "prediction_service"}

# 训练模型接口
@app.post("/train", tags=["模型管理"])
def train_model(request: TrainRequest):
    try:
        result = PredictionModel.train_model(
            symbol=request.symbol,
            model_type=request.model_type,
            lookback_days=request.lookback_days,
            test_size=request.test_size
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取模型列表
@app.get("/models", tags=["模型管理"])
def get_models(symbol: str = None):
    try:
        models = PredictionModel.get_all_models(symbol)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 设置默认模型
@app.post("/models/default", tags=["模型管理"])
def set_default_model(model_name: str = Form(...)):
    try:
        result = PredictionModel.set_default_model(model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 预测接口
@app.post("/predict", tags=["预测"])
def predict(request: PredictionRequest):
    try:
        # 检查缓存
        r = get_redis_connection()
        cache_key = f"predictions:{request.symbol}:{request.model_name or 'default'}"
        cached_data = r.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # 执行预测
        result = PredictionModel.predict_future(
            symbol=request.symbol,
            days=request.days,
            model_name=request.model_name
        )
        
        # 存入缓存，有效期1小时
        r.setex(cache_key, 3600, json.dumps(result))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取历史预测
@app.get("/predictions/history", tags=["预测"])
def get_prediction_history(symbol: str, model_name: str = None):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if model_name:
                cur.execute("""
                SELECT prediction_date, predicted_close, created_at 
                FROM predictions 
                WHERE symbol = %s AND model_name = %s
                ORDER BY prediction_date DESC
                """, (symbol, model_name))
            else:
                cur.execute("""
                SELECT prediction_date, predicted_close, model_name, created_at 
                FROM predictions 
                WHERE symbol = %s
                ORDER BY prediction_date DESC
                """, (symbol,))
            
            columns = [desc[0] for desc in cur.description]
            predictions = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        conn.close()
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
