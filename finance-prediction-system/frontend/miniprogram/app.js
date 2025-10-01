// app.js
App({
  onLaunch() {
    // 初始化云开发环境
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力');
    } else {
      wx.cloud.init({
        traceUser: true,
      });
    }

    // 检查登录状态
    this.checkLoginStatus();
    
    // 初始化全局数据
    this.globalData = {
      userInfo: null,
      apiBaseUrl: 'http://localhost:8000', // API基础地址
      // 可根据环境切换
      // apiBaseUrl: 'https://api.finance-prediction-system.com'
    };
  },

  // 检查登录状态
  checkLoginStatus() {
    wx.getSetting({
      success: res => {
        if (res.authSetting['scope.userInfo']) {
          // 已经授权，可以直接调用 getUserInfo 获取头像昵称
          wx.getUserInfo({
            success: res => {
              this.globalData.userInfo = res.userInfo;
              
              // 触发回调
              if (this.userInfoReadyCallback) {
                this.userInfoReadyCallback(res);
              }
            }
          });
        }
      }
    });
  },

  // 网络请求封装
  request(url, method = 'GET', data = {}) {
    const baseUrl = this.globalData.apiBaseUrl;
    const fullUrl = `${baseUrl}${url}`;
    
    return new Promise((resolve, reject) => {
      wx.showLoading({
        title: '加载中...',
        mask: true
      });
      
      wx.request({
        url: fullUrl,
        method: method,
        data: data,
        header: {
          'content-type': 'application/json'
        },
        success: res => {
          wx.hideLoading();
          
          if (res.statusCode === 200) {
            resolve(res.data);
          } else {
            wx.showToast({
              title: res.data.detail || '请求失败',
              icon: 'none',
              duration: 2000
            });
            reject(res.data);
          }
        },
        fail: err => {
          wx.hideLoading();
          wx.showToast({
            title: '网络错误',
            icon: 'none',
            duration: 2000
          });
          reject(err);
        }
      });
    });
  },

  // 获取股票列表
  getStocks() {
    return this.request('/stocks');
  },

  // 获取股票历史数据
  getStockHistory(symbol, startDate, endDate) {
    return this.request('/stock/history', 'POST', {
      symbol,
      start_date: startDate,
      end_date: endDate
    });
  },

  // 获取股票指标
  getStockIndicators(symbol, indicator) {
    return this.request(`/stock/indicators/${symbol}`, 'GET', {
      indicator
    });
  },

  // 获取预测结果
  getPrediction(symbol, days = 7, modelName) {
    return this.request('/predict', 'POST', {
      symbol,
      days,
      model_name: modelName
    });
  }
});
