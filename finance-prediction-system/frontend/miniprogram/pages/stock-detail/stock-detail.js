// pages/stock-detail/stock-detail.js
const app = getApp();
const dateUtil = require('../../utils/date.js');

Page({
  data: {
    symbol: '',
    stockName: '',
    stockData: [], // 股票历史数据
    indicators: [], // 技术指标
    predictions: [], // 预测数据
    loading: true,
    error: '',
    dateRange: '1M', // 1W, 1M, 3M, 1Y
    chartType: 'price', // price, indicators, prediction
    selectedIndicator: 'sma_20',
    indicatorTypes: [
      { value: 'sma_20', name: '20日移动平均' },
      { value: 'sma_50', name: '50日移动平均' },
      { value: 'rsi_14', name: 'RSI(14)' },
      { value: 'macd', name: 'MACD' }
    ],
    // 图表配置
    chartOption: {}
  },

  onLoad(options) {
    // 从路由参数获取股票代码
    if (options.symbol) {
      this.setData({
        symbol: options.symbol,
        stockName: options.name || ''
      });
      
      // 设置页面标题
      wx.setNavigationBarTitle({
        title: `${options.symbol} - ${options.name || '股票详情'}`
      });
      
      // 加载数据
      this.loadStockData();
    } else {
      this.setData({
        loading: false,
        error: '未指定股票代码'
      });
    }
  },

  // 加载股票数据
  loadStockData() {
    this.setData({ loading: true });
    
    // 根据选择的时间范围计算开始日期
    const today = new Date();
    let days = 30; // 默认1个月
    
    switch (this.data.dateRange) {
      case '1W':
        days = 7;
        break;
      case '1M':
        days = 30;
        break;
      case '3M':
        days = 90;
        break;
      case '1Y':
        days = 365;
        break;
    }
    
    const endDate = today.toISOString().split('T')[0];
    const startDate = dateUtil.addDays(today, -days).toISOString().split('T')[0];
    
    // 并行加载历史数据和预测数据
    Promise.all([
      // 获取历史数据
      app.getStockHistory(this.data.symbol, startDate, endDate),
      // 获取预测数据
      app.getPrediction(this.data.symbol, 7)
    ]).then(([historyData, predictionData]) => {
      this.setData({
        stockData: historyData,
        predictions: predictionData.predictions || []
      });
      
      // 加载指标数据
      return app.getStockIndicators(this.data.symbol);
    }).then(indicatorData => {
      this.setData({
        indicators: indicatorData
      });
      
      // 初始化图表
      this.updateChart();
      
      this.setData({ loading: false });
    }).catch(err => {
      console.error('加载数据失败:', err);
      this.setData({
        loading: false,
        error: '加载数据失败，请重试'
      });
    });
  },

  // 更新图表
  updateChart() {
    const { stockData, indicators, predictions, chartType, selectedIndicator } = this.data;
    
    if (stockData.length === 0) return;
    
    // 提取日期和价格数据
    const dates = stockData.map(item => {
      const date = new Date(item.date);
      return `${date.getMonth() + 1}/${date.getDate()}`;
    });
    const closePrices = stockData.map(item => item.close);
    
    let series = [
      {
        name: '收盘价',
        type: 'line',
        data: closePrices,
        smooth: true,
        lineStyle: { width: 2 },
        emphasis: { focus: 'series' }
      }
    ];
    
    // 根据图表类型添加不同的数据系列
    if (chartType === 'indicators' && indicators.length > 0) {
      // 添加指标数据
      const indicatorData = indicators.map(item => item[selectedIndicator] || null);
      
      series.push({
        name: this.getIndicatorName(selectedIndicator),
        type: 'line',
        data: indicatorData,
        smooth: true,
        lineStyle: { width: 2, type: 'dashed' },
        emphasis: { focus: 'series' },
        yAxisIndex: 1 // 使用第二个Y轴
      });
    } else if (chartType === 'prediction' && predictions.length > 0) {
      // 添加预测数据
      const lastDate = new Date(stockData[stockData.length - 1].date);
      
      // 生成预测日期标签
      const predictionDates = predictions.map(item => {
        const date = new Date(item.date);
        return `${date.getMonth() + 1}/${date.getDate()}`;
      });
      
      // 预测价格
      const predictionPrices = predictions.map(item => item.predicted_close);
      
      // 将历史数据和预测数据合并显示
      const allDates = [...dates, ...predictionDates];
      const allPrices = [...closePrices];
      
      // 添加一个空数据点作为历史和预测的连接点
      allPrices.push(null);
      allPrices.push(...predictionPrices);
      
      series = [
        {
          name: '历史价格',
          type: 'line',
          data: [...closePrices, null],
          smooth: true,
          lineStyle: { width: 2 },
          emphasis: { focus: 'series' }
        },
        {
          name: '预测价格',
          type: 'line',
          data: [...Array(closePrices.length + 1).fill(null), ...predictionPrices],
          smooth: true,
          lineStyle: { width: 2, color: '#ff4d4f' },
          emphasis: { focus: 'series' },
          itemStyle: { color: '#ff4d4f' }
        }
      ];
    }
    
    // 配置Y轴
    let yAxis = [
      {
        type: 'value',
        name: '价格',
        min: 'dataMin',
        max: 'dataMax',
        axisLine: { show: true }
      }
    ];
    
    // 指标图表需要第二个Y轴
    if (chartType === 'indicators') {
      yAxis.push({
        type: 'value',
        name: this.getIndicatorName(selectedIndicator),
        min: 'dataMin',
        max: 'dataMax',
        position: 'right',
        axisLine: { show: true }
      });
    }
    
    // 更新图表配置
    this.setData({
      chartOption: {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross'
          }
        },
        legend: {
          data: series.map(s => s.name),
          top: 0
        },
        grid: {
          left: '3%',
          right: chartType === 'indicators' ? '10%' : '3%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: chartType === 'prediction' ? [...dates, '', ...predictionDates] : dates,
          boundaryGap: false,
          axisLabel: {
            interval: Math.ceil(dates.length / 10) // 控制X轴标签显示密度
          }
        },
        yAxis: yAxis,
        series: series
      }
    });
  },

  // 根据指标代码获取显示名称
  getIndicatorName(indicatorCode) {
    const indicator = this.data.indicatorTypes.find(item => item.value === indicatorCode);
    return indicator ? indicator.name : indicatorCode;
  },

  // 切换日期范围
  onDateRangeChange(e) {
    this.setData({ dateRange: e.detail.value });
    this.loadStockData();
  },

  // 切换图表类型
  onChartTypeChange(e) {
    this.setData({ chartType: e.detail.value });
    this.updateChart();
  },

  // 切换指标类型
  onIndicatorChange(e) {
    this.setData({ selectedIndicator: e.detail.value });
    this.updateChart();
  },

  // 刷新数据
  onRefresh() {
    this.loadStockData();
  }
});
