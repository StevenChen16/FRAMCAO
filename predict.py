import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from data import download_and_prepare_data, rolling_normalize
from train import generate_event_data
from fusion_model import FRAMCAOCNNFusion
from model import prepare_feature_groups
from tqdm import tqdm

def setup_logging(log_dir='logs'):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"prediction_{datetime.now():%Y%m%d_%H%M%S}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def combine_stock_data(symbols, start_date, end_date):
    """
    下载多只股票的数据并拼接
    
    Args:
        symbols (list): 股票代码列表
        start_date (str): 开始日期
        end_date (str): 结束日期
    
    Returns:
        pd.DataFrame: 拼接后的数据
    """
    all_data = []
    
    for symbol in tqdm(symbols):
        # 获取单个股票数据
        data = download_and_prepare_data(symbol, start_date, end_date)

        if not data.empty:
            # 将数据添加到列表中
            all_data.append(data)
    
    # 直接拼接所有数据
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    
    return combined_data

class InferenceFusionDataset(Dataset):
    """用于推理的数据集"""
    def __init__(self, data, events, sequence_length=250):
        self.feature_order = [
            'Close', 'Open', 'High', 'Low',
            'MA5', 'MA20',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI', 'Upper', 'Middle', 'Lower',
            'CRSI', 'Kalman_Price', 'Kalman_Trend',
            'FFT_21', 'FFT_63',
            'Volume', 'Volume_MA5'
        ]
        
        self.data = data
        self.events = events
        self.sequence_length = sequence_length
        
        # 组织2D特征数据
        self.feature_data = self._organize_features()
        
        # 计算时间距离矩阵
        self.time_distances = self._compute_time_distances()
    
    def _organize_features(self):
        """组织特征为2D张量格式"""
        feature_data = []
        for feature in self.feature_order:
            values = self.data[feature].values
            mean = np.mean(values)
            std = np.std(values)
            normalized = (values - mean) / (std + 1e-8)
            feature_data.append(normalized)
        
        return torch.FloatTensor(np.stack(feature_data, axis=0))
    
    def _compute_time_distances(self):
        """计算事件时间距离"""
        distances = np.zeros((len(self.data), 1))
        last_event_idx = -1
        
        for i in range(len(self.data)):
            if self.events[i].any():
                last_event_idx = i
            distances[i] = i - last_event_idx if last_event_idx != -1 else 100
            
        distances = distances / 100.0
        return distances
    
    def __len__(self):
        return 1  # 推理时只需要一个样本
    
    def __getitem__(self, idx):
        # 获取最后sequence_length天的数据
        feature_seq = self.feature_data[:, -self.sequence_length:]
        sequence = torch.FloatTensor(self.data.iloc[-self.sequence_length:].values)
        events = torch.LongTensor(self.events[-self.sequence_length:])
        time_distances = torch.FloatTensor(self.time_distances[-self.sequence_length:])
        current_price = self.feature_data[0, -1]
        
        return {
            'feature_seq': feature_seq,
            'sequence': sequence,
            'events': events,
            'time_distances': time_distances,
            'current_price': current_price
        }

def predict_next_day(model, symbols, lookback_days=250, device='cuda'):
    """预测下一天的股票价格
    
    Args:
        model: 训练好的模型
        symbols: 股票代码列表
        lookback_days: 历史数据长度
        device: 运行设备
    
    Returns:
        predictions: 预测结果字典
    """
    logger = setup_logging()
    model.eval()
    
    # 设置日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)
    
    result_dict = {}  # 改名以避免混淆
    
    # 获取数据
    logger.info("Loading stock data...")
    data = combine_stock_data(
        symbols,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # 生成事件数据
    logger.info("Generating event data...")
    events = generate_event_data(data)
    
    # 创建推理数据集
    dataset = InferenceFusionDataset(data, events)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 进行预测
    logger.info("Making predictions...")
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            sequence = batch['sequence'].to(device)
            events = batch['events'].to(device)
            time_distances = batch['time_distances'].to(device)
            current_price = batch['current_price'].to(device)
            
            # 模型预测
            predictions, _, _ = model(sequence, events, time_distances)
            
            # 获取最后一个时间步的预测并转换为标量
            next_day_pred = predictions[0, -1, 0].cpu().item()
            current_price = current_price.cpu().item()
            
            # 记录预测结果
            result_dict['next_day'] = next_day_pred
            result_dict['current_price'] = current_price
            result_dict['predicted_change'] = (next_day_pred / current_price - 1) * 100
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'].values[-30:], label='Historical')
    plt.scatter(len(data['Close'].values[-30:]), next_day_pred, color='red', label='Prediction')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.savefig('prediction_plot.png')
    plt.close()
    
    logger.info(f"Prediction completed successfully.")
    logger.info(f"Predicted next day price: {next_day_pred:.2f}")
    logger.info(f"Predicted change: {result_dict['predicted_change']:.2f}%")
    
    return result_dict

def main():
    # 参数设置
    input_dim = 21  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    event_dim = 32  # 事件嵌入维度
    num_event_types = 10  # 事件类型数量
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    print("Loading model...")
    model = FRAMCAOCNNFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load('checkpoints/fusion/stage1_best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 股票列表
    symbols = ["GOOGL"]
    
    # 进行预测
    print("Making predictions...")
    predictions = predict_next_day(model, symbols, device=device)
    print("\nPrediction Results:")
    if predictions['predicted_change'] > 0:
        print("Up")
    else:
        print("Down")

if __name__ == "__main__":
    main()