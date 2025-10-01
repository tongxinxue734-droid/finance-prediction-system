import os
import time
import requests
import psycopg2
import redis
from datetime import datetime, timedelta
import logging
from apscheduler.schedulers.blocking import BlockingScheduler

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_collector")

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

# 需要采集的股票列表
STOCKS_TO_COLLECT = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    # 可以添加更多股票
]

class FinanceDataCollector:
    def __init__(self):
        self.api_key = os.getenv("FINANCE_API_KEY")
        if not self.api_key:
            logger.warning("未设置FINANCE_API_KEY环境变量")
        
        # 初始化数据库表
        self.init_database()
        
        # 初始化调度器
        self.scheduler = BlockingScheduler()
        collect_interval = int(os.getenv("COLLECT_INTERVAL", 3600))  # 默认1小时
        self.scheduler.add_job(
            self.collect_all_stocks, 
            'interval', 
            seconds=collect_interval,
            name="collect_all_stocks"
        )
        
        logger.info(f"数据采集服务初始化完成，采集间隔: {collect_interval}秒")

    def init_database(self):
        """初始化数据库表结构"""
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # 创建股票价格表
                cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    date DATE NOT NULL,
                    open NUMERIC(10, 4) NOT NULL,
                    high NUMERIC(10, 4) NOT NULL,
                    low NUMERIC(10, 4) NOT NULL,
                    close NUMERIC(10, 4) NOT NULL,
                    adj_close NUMERIC(10, 4),
                    volume BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
                """)
                
                # 创建采集状态表
                cur.execute("""
                CREATE TABLE IF NOT EXISTS collection_status (
                    symbol VARCHAR(20) PRIMARY KEY,
                    last_collected_date DATE,
                    last_success_time TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending'
                )
                """)
                
                # 插入股票列表（如果不存在）
                for stock in STOCKS_TO_COLLECT:
                    cur.execute("""
                    INSERT INTO collection_status (symbol) 
                    VALUES (%s) 
                    ON CONFLICT (symbol) DO NOTHING
                    """, (stock["symbol"],))
                
            conn.commit()
            logger.info("数据库表初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def fetch_stock_data(self, symbol, name, start_date=None, end_date=None):
        """从金融API获取股票数据"""
        try:
            # 这里使用模拟数据，实际项目中应替换为真实的API调用
            # 例如Alpha Vantage, Yahoo Finance等API
            
            # 生成模拟的日期范围
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                # 默认获取最近30天数据
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            logger.info(f"获取 {symbol} 数据: {start_date} 至 {end_date}")
            
            # 模拟API返回数据
            data = []
            current_date = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            # 为了演示，只生成最近5天的数据
            count = 0
            while current_date <= end and count < 5:
                # 生成随机价格数据（模拟）
                open_price = 100 + (count * 1.5) + (current_date.day * 0.1)
                close_price = open_price + (0.5 - (count % 3) * 0.3)
                
                data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "open": round(open_price, 2),
                    "high": round(open_price + 2.5, 2),
                    "low": round(open_price - 1.5, 2),
                    "close": round(close_price, 2),
                    "adj_close": round(close_price * 0.98, 2),
                    "volume": 1000000 + (count * 50000)
                })
                
                current_date += timedelta(days=1)
                # 跳过周末
                while current_date.weekday() >= 5:  # 5是周六，6是周日
                    current_date += timedelta(days=1)
                
                count += 1
            
            return data
            
            # 实际API调用示例（注释掉，使用模拟数据）
            # url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={self.api_key}"
            # response = requests.get(url)
            # response.raise_for_status()
            # result = response.json()
            # return result.get("historical", [])
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {str(e)}")
            return []

    def save_stock_data(self, symbol, name, data):
        """保存股票数据到数据库"""
        if not data:
            logger.warning(f"没有要保存的 {symbol} 数据")
            return 0
            
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                inserted = 0
                for item in data:
                    try:
                        cur.execute("""
                        INSERT INTO stock_prices 
                        (symbol, name, date, open, high, low, close, adj_close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, date) DO UPDATE 
                        SET open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                        """, (
                            symbol,
                            name,
                            item["date"],
                            item["open"],
                            item["high"],
                            item["low"],
                            item["close"],
                            item.get("adj_close"),
                            item["volume"]
                        ))
                        inserted += 1
                    except Exception as e:
                        logger.warning(f"保存 {symbol} {item['date']} 数据失败: {str(e)}")
                        continue
                
                # 更新采集状态
                cur.execute("""
                UPDATE collection_status 
                SET last_collected_date = %s,
                    last_success_time = CURRENT_TIMESTAMP,
                    status = 'success'
                WHERE symbol = %s
                """, (data[-1]["date"], symbol))
                
                conn.commit()
                logger.info(f"{symbol} 数据保存完成，新增/更新 {inserted} 条记录")
                return inserted
        except Exception as e:
            logger.error(f"保存 {symbol} 数据失败: {str(e)}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if conn:
                conn.close()

    def collect_stock_data(self, stock):
        """采集单只股票数据"""
        symbol = stock["symbol"]
        name = stock["name"]
        
        try:
            # 获取最后一次采集的日期
            last_date = None
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                SELECT last_collected_date 
                FROM collection_status 
                WHERE symbol = %s
                """, (symbol,))
                result = cur.fetchone()
                if result and result[0]:
                    last_date = result[0]
            conn.close()
            
            # 确定采集的日期范围
            start_date = None
            if last_date:
                # 从最后一次采集的次日开始
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # 获取数据
            data = self.fetch_stock_data(symbol, name, start_date)
            
            # 保存数据
            if data:
                self.save_stock_data(symbol, name, data)
                
                # 清除相关缓存
                r = get_redis_connection()
                pattern = f"stock:history:{symbol}:*"
                for key in r.keys(pattern):
                    r.delete(key)
                logger.info(f"已清除 {symbol} 的缓存数据")
                
        except Exception as e:
            logger.error(f"{symbol} 数据采集失败: {str(e)}")
            # 更新状态为失败
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                    UPDATE collection_status 
                    SET status = 'failed'
                    WHERE symbol = %s
                    """, (symbol,))
                conn.commit()
            except:
                pass
            finally:
                if conn:
                    conn.close()

    def collect_all_stocks(self):
        """采集所有股票数据"""
        logger.info("开始批量采集股票数据")
        for stock in STOCKS_TO_COLLECT:
            self.collect_stock_data(stock)
            time.sleep(2)  # 避免请求过于频繁
        logger.info("批量采集完成")

    def start(self):
        """启动采集服务"""
        try:
            # 先立即执行一次采集
            self.collect_all_stocks()
            
            # 启动调度器
            logger.info("启动数据采集调度器...")
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("数据采集服务已停止")
        except Exception as e:
            logger.error(f"调度器启动失败: {str(e)}")

if __name__ == "__main__":
    collector = FinanceDataCollector()
    collector.start()
