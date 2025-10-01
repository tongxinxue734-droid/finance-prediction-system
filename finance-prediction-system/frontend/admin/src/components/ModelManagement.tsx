import React, { useState, useEffect } from 'react';
import { 
  Box, Button, Typography, Paper, Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow, IconButton, Snackbar, 
  Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  FormControl, InputLabel, Select, MenuItem, CircularProgress,
  Alert
} from '@mui/material';
import { 
  Add as AddIcon, 
  Delete as DeleteIcon, 
  Edit as EditIcon, 
  Check as CheckIcon,
  Autorenew as RefreshIcon
} from '@mui/icons-material';
import axios from 'axios';

// API基础URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 模型接口类型定义
interface PredictionModel {
  model_name: string;
  symbol: string;
  model_type: string;
  lookback_days: number;
  train_date: string;
  mse: number;
  mae: number;
  r2: number;
  is_default: boolean;
}

// 训练模型请求参数
interface TrainModelRequest {
  symbol: string;
  model_type: string;
  lookback_days: number;
  test_size: number;
}

// 股票列表类型
interface Stock {
  symbol: string;
  name: string;
}

const ModelManagement: React.FC = () => {
  // 状态管理
  const [models, setModels] = useState<PredictionModel[]>([]);
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error'}>({
    open: false,
    message: '',
    severity: 'success'
  });
  
  // 对话框状态
  const [trainDialogOpen, setTrainDialogOpen] = useState<boolean>(false);
  const [trainModelData, setTrainModelData] = useState<TrainModelRequest>({
    symbol: '',
    model_type: 'random_forest',
    lookback_days: 30,
    test_size: 0.2
  });
  const [training, setTraining] = useState<boolean>(false);
  const [trainResult, setTrainResult] = useState<{success: boolean, plot?: string} | null>(null);

  // 获取模型列表
  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/models`);
      setModels(response.data);
      setError(null);
    } catch (err) {
      console.error('获取模型列表失败:', err);
      setError('获取模型列表失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 获取股票列表
  const fetchStocks = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stocks`);
      setStocks(response.data);
      if (response.data.length > 0 && !trainModelData.symbol) {
        setTrainModelData(prev => ({...prev, symbol: response.data[0].symbol}));
      }
    } catch (err) {
      console.error('获取股票列表失败:', err);
    }
  };

  // 组件加载时获取数据
  useEffect(() => {
    fetchModels();
    fetchStocks();
  }, []);

  // 设置默认模型
  const handleSetDefault = async (modelName: string) => {
    try {
      await axios.post(`${API_BASE_URL}/models/default`, { model_name: modelName });
      setSnackbar({
        open: true,
        message: '模型已设为默认',
        severity: 'success'
      });
      fetchModels(); // 刷新列表
    } catch (err) {
      console.error('设置默认模型失败:', err);
      setSnackbar({
        open: true,
        message: '设置默认模型失败',
        severity: 'error'
      });
    }
  };

  // 训练新模型
  const handleTrainModel = async () => {
    try {
      setTraining(true);
      const response = await axios.post(`${API_BASE_URL}/train`, trainModelData);
      setTrainResult({
        success: true,
        plot: response.data.plot
      });
      setSnackbar({
        open: true,
        message: `模型训练成功: ${response.data.model_name}`,
        severity: 'success'
      });
      fetchModels(); // 刷新模型列表
      setTrainDialogOpen(false);
    } catch (err: any) {
      console.error('模型训练失败:', err);
      setTrainResult({
        success: false
      });
      setSnackbar({
        open: true,
        message: `模型训练失败: ${err.response?.data || err.message}`,
        severity: 'error'
      });
    } finally {
      setTraining(false);
    }
  };

  // 处理输入变化
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | {name?: string; value: unknown}>) => {
    const { name, value } = e.target;
    if (name) {
      setTrainModelData(prev => ({
        ...prev,
        [name]: name === 'lookback_days' || name === 'test_size' ? Number(value) : value
      }));
    }
  };

  // 关闭对话框
  const handleCloseDialog = () => {
    setTrainDialogOpen(false);
    setTrainResult(null);
  };

  // 关闭提示
  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({...prev, open: false}));
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h1">
          模型管理
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => setTrainDialogOpen(true)}
        >
          训练新模型
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        <TableContainer>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>模型名称</TableCell>
                <TableCell>股票代码</TableCell>
                <TableCell>模型类型</TableCell>
                <TableCell>回溯天数</TableCell>
                <TableCell>训练日期</TableCell>
                <TableCell>MSE</TableCell>
                <TableCell>MAE</TableCell>
                <TableCell>R²</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={10} align="center">
                    <CircularProgress />
                  </TableCell>
                </TableRow>
              ) : models.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={10} align="center">
                    暂无模型数据，请训练新模型
                  </TableCell>
                </TableRow>
              ) : (
                models.map((model) => (
                  <TableRow key={model.model_name}>
                    <TableCell>{model.model_name}</TableCell>
                    <TableCell>{model.symbol}</TableCell>
                    <TableCell>
                      {model.model_type === 'random_forest' ? '随机森林' : '线性回归'}
                    </TableCell>
                    <TableCell>{model.lookback_days}</TableCell>
                    <TableCell>
                      {new Date(model.train_date).toLocaleString()}
                    </TableCell>
                    <TableCell>{model.mse.toFixed(4)}</TableCell>
                    <TableCell>{model.mae.toFixed(4)}</TableCell>
                    <TableCell>{model.r2.toFixed(4)}</TableCell>
                    <TableCell>
                      {model.is_default && (
                        <Badge color="success" variant="contained">
                          默认模型
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <IconButton
                        color={model.is_default ? "default" : "primary"}
                        onClick={() => !model.is_default && handleSetDefault(model.model_name)}
                        disabled={model.is_default}
                      >
                        <CheckIcon />
                      </IconButton>
                      <IconButton color="secondary" onClick={() => {}}>
                        <RefreshIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* 训练模型对话框 */}
      <Dialog open={trainDialogOpen} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>训练新模型</DialogTitle>
        <DialogContent dividers>
          {trainResult ? (
            <Box>
              {trainResult.success ? (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    模型训练成功！
                  </Typography>
                  {trainResult.plot && (
                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                      <img 
                        src={`data:image/png;base64,${trainResult.plot}`} 
                        alt="预测结果对比" 
                        style={{ maxWidth: '100%', maxHeight: '500px' }}
                      />
                    </Box>
                  )}
                </Box>
              ) : (
                <Alert severity="error">
                  模型训练失败，请检查参数并重试
                </Alert>
              )}
            </Box>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth>
                <InputLabel>股票代码</InputLabel>
                <Select
                  name="symbol"
                  value={trainModelData.symbol}
                  label="股票代码"
                  onChange={handleInputChange}
                >
                  {stocks.map(stock => (
                    <MenuItem key={stock.symbol} value={stock.symbol}>
                      {stock.symbol} - {stock.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth>
                <InputLabel>模型类型</InputLabel>
                <Select
                  name="model_type"
                  value={trainModelData.model_type}
                  label="模型类型"
                  onChange={handleInputChange}
                >
                  <MenuItem value="random_forest">随机森林</MenuItem>
                  <MenuItem value="linear_regression">线性回归</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                name="lookback_days"
                label="回溯天数"
                type="number"
                value={trainModelData.lookback_days}
                onChange={handleInputChange}
                InputProps={{ inputProps: { min: 5, max: 180 } }}
                helperText="用于创建特征的历史数据天数"
              />

              <TextField
                fullWidth
                name="test_size"
                label="测试集比例"
                type="number"
                step="0.05"
                value={trainModelData.test_size}
                onChange={handleInputChange}
                InputProps={{ inputProps: { min: 0.1, max: 0.5 } }}
                helperText="用于模型评估的测试数据比例"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} disabled={training}>
            关闭
          </Button>
          {!trainResult && (
            <Button 
              onClick={handleTrainModel} 
              color="primary" 
              disabled={training}
            >
              {training ? <CircularProgress size={20} /> : '开始训练'}
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* 提示信息 */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

// 自定义Badge组件（解决上面代码中未定义的问题）
const Badge = ({ color, variant, children }: {
  color: string;
  variant: string;
  children: React.ReactNode;
}) => (
  <Box 
    sx={{
      backgroundColor: color === 'success' ? '#4caf50' : '#f50057',
      color: 'white',
      borderRadius: 1,
      px: 1,
      py: 0.5,
      fontSize: '0.75rem',
    }}
  >
    {children}
  </Box>
);

export default ModelManagement;
