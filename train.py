import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

class EnhancedStockDataset(Dataset):
    def __init__(self, data, events, seq_length=10, prediction_horizon=1):
        self.data = data
        self.events = events
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        
        # 生成有效的序列起始索引
        self.indices = range(len(data) - seq_length - prediction_horizon + 1)
        
        # 计算时间距离矩阵
        self.time_distances = self._compute_time_distances()
        
    def _compute_time_distances(self):
        # 这里应该根据实际的事件时间戳计算距离
        # 现在用简单的线性距离代替
        distances = np.zeros((len(self.data), 1))
        last_event_idx = -1
        
        for i in range(len(self.data)):
            if self.events[i].any():  # 如果有事件发生
                last_event_idx = i
            distances[i] = i - last_event_idx if last_event_idx != -1 else 999
            
        return distances
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length
        
        # 获取输入序列
        sequence = torch.FloatTensor(
            self.data.iloc[start_idx:end_idx].values
        )
        
        # 获取目标值 (预测未来的收盘价)
        target_idx = end_idx + self.prediction_horizon - 1
        target = torch.FloatTensor([
            self.data.iloc[target_idx]['Close']
        ])
        
        # 获取事件数据
        events = torch.LongTensor(
            self.events[start_idx:end_idx]
        )
        
        # 获取时间距离
        time_distances = torch.FloatTensor(
            self.time_distances[start_idx:end_idx]
        )
        
        return sequence, events, time_distances, target

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, d_model, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch + 1
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        
        return [base_lr * self.d_model ** (-0.5) * min(arg1, arg2)
                for base_lr in self.base_lrs]

class EnhancedCombinedLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=0.9, gamma=0, delta=0.05, epsilon=0,
                 memory_weight=0.1, coupling_weight=0.1):
        super().__init__()
        self.alpha = alpha   # MSE权重
        self.beta = beta     # 方向预测权重
        self.gamma = gamma   # 连续性权重
        self.delta = delta   # 特征正则化权重
        self.epsilon = epsilon # 特征组一致性权重
        self.memory_weight = memory_weight
        self.coupling_weight = coupling_weight
        
    def _compute_group_consistency(self, group_features):
        """计算特征组内的一致性损失"""
        consistency_loss = 0
        for i in range(group_features.size(1)):  # 遍历每个时间步
            # 获取当前时间步的特征
            time_step_features = group_features[:, i, :]
            # 计算均值
            mean_feature = time_step_features.mean(dim=-1, keepdim=True)
            # 计算与均值的差异
            consistency_loss += torch.mean(
                torch.abs(time_step_features - mean_feature)
            )
        # 对时间步取平均
        return consistency_loss / group_features.size(1)
        
    def forward(self, predictions, targets, prev_price, features, group_features):
        # 计算基础MSE损失
        mse_loss = F.mse_loss(predictions[:, -1, :], targets)
        
        # 计算方向预测损失
        pred_diff = predictions[:, -1, :] - prev_price.unsqueeze(-1)
        target_diff = targets - prev_price.unsqueeze(-1)
        direction_loss = F.binary_cross_entropy_with_logits(
            (pred_diff > 0).float(),
            (target_diff > 0).float()
        )
        
        # 计算连续性损失
        smoothness_loss = torch.mean(torch.abs(
            predictions[:, -1, :] - prev_price.unsqueeze(-1)
        ))
        if smoothness_loss > 0.5:  # 当平滑度超过阈值时增加惩罚
            smoothness_loss *= 1.5
            
        # 特征正则化
        mcao_features = torch.mean(torch.abs(features))
        
        # 特征组一致性损失
        group_consistency_loss = self._compute_group_consistency(group_features)
        
        # 组合所有损失
        total_loss = (self.alpha * mse_loss +
                     self.beta * direction_loss +
                     self.gamma * smoothness_loss +
                     self.delta * mcao_features +
                     self.epsilon * group_consistency_loss)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'smoothness': smoothness_loss.item(),
            'mcao_features': mcao_features.item(),
            'group_consistency': group_consistency_loss.item()
        }

def train_enhanced_model(model, train_loader, val_loader, n_epochs,
                        device, learning_rate=0.001):
    criterion = EnhancedCombinedLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 使用余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # 初始化早停
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        model.train()
        total_metrics = {
            'mse': 0, 'direction': 0, 'smoothness': 0,
            'tmdo_reg': 0, 'group_consistency': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for sequence, events, time_distances, target in pbar:
            # 移动数据到设备
            sequence = sequence.to(device)
            events = events.to(device)
            time_distances = time_distances.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions, tmdo_features, group_features = model(
                sequence, events, time_distances
            )
            
            # 计算损失
            loss, metrics = criterion(
                predictions,
                target,
                sequence[:, -1, 3],  # 假设收盘价是第4列
                tmdo_features,
                group_features
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 更新指标
            for k, v in metrics.items():
                total_metrics[k] += v
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                **{k: v/len(pbar) for k, v in total_metrics.items()}
            })
        
        # 打印训练指标
        print(f"\nEpoch {epoch+1} Training Metrics:")
        for k, v in total_metrics.items():
            print(f"{k}: {v/len(train_loader):.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        val_metrics = {k: 0 for k in total_metrics.keys()}
        
        with torch.no_grad():
            for sequence, events, time_distances, target in val_loader:
                sequence = sequence.to(device)
                events = events.to(device)
                time_distances = time_distances.to(device)
                target = target.to(device)
                
                predictions, tmdo_features, group_features = model(
                    sequence, events, time_distances
                )
                
                loss, metrics = criterion(
                    predictions,
                    target,
                    sequence[:, -1, 3],
                    tmdo_features,
                    group_features
                )
                
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] += v
        
        val_loss /= len(val_loader)
        print(f"\nValidation Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"Val {k}: {v/len(val_loader):.4f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def generate_event_data(data):
    """
    生成股票相关的事件数据
    
    参数:
    data: DataFrame, 包含股票的OHLCV等数据
    
    返回:
    events: ndarray, shape (n_samples, n_event_types)
    其中每一列代表一种事件类型:
    0: 无事件
    1: 财报事件
    2: 分红派息
    3: 大额交易
    4: 价格异常
    5: 成交异常
    6: 高管变动
    7: 监管事件
    8: 停复牌
    9: 其他重大事件
    """
    n_samples = len(data)
    n_event_types = 10  # 总共10种事件类型
    events = np.zeros((n_samples, n_event_types))
    
    # 1. 检测财报事件 (通常是每季度)
    # 简单起见，假设每90个交易日发布一次财报
    for i in range(0, n_samples, 90):
        if i < n_samples:
            events[i, 1] = 1
    
    # 2. 检测分红派息 (根据价格跳跃)
    returns = data['Close'].pct_change()
    for i in range(1, n_samples):
        if abs(returns.iloc[i]) > 0.1:  # 价格跳跃超过10%
            events[i, 2] = 1
    
    # 3. 检测大额交易 (基于成交量异常)
    volume_mean = data['Volume'].rolling(window=20).mean()
    volume_std = data['Volume'].rolling(window=20).std()
    volume_zscore = (data['Volume'] - volume_mean) / volume_std
    events[volume_zscore > 3, 3] = 1  # 成交量超过3个标准差
    
    # 4. 检测价格异常
    price_std = data['Close'].rolling(window=20).std()
    price_returns = data['Close'].pct_change()
    events[abs(price_returns) > 2 * price_std, 4] = 1
    
    # 5. 检测成交量异常
    vol_ratio = data['Volume'] / data['Volume'].rolling(window=5).mean()
    events[vol_ratio > 3, 5] = 1  # 成交量是5日平均的3倍以上
    
    # 6-9: 其他事件可以基于具体数据来源添加
    # 这里用随机生成来模拟
    np.random.seed(42)  # 保证可重复性
    
    # 高管变动 (假设平均每年4次)
    n_management_events = n_samples // 250 * 4
    management_event_days = np.random.choice(
        n_samples, n_management_events, replace=False
    )
    events[management_event_days, 6] = 1
    
    # 监管事件 (假设平均每月1次)
    n_regulatory_events = n_samples // 20
    regulatory_event_days = np.random.choice(
        n_samples, n_regulatory_events, replace=False
    )
    events[regulatory_event_days, 7] = 1
    
    # 停复牌
    # 通过检测价格是否变化来判断
    price_change = data['Close'].diff()
    events[price_change == 0, 8] = 1
    
    # 其他重大事件 (随机生成，较少)
    n_other_events = n_samples // 500
    other_event_days = np.random.choice(
        n_samples, n_other_events, replace=False
    )
    events[other_event_days, 9] = 1
    
    return events

def add_real_events(events, news_data=None, filings_data=None):
    """
    添加真实的事件数据（如果有的话）
    
    参数:
    events: 基础事件矩阵
    news_data: 新闻数据（可选）
    filings_data: 公告数据（可选）
    
    返回:
    更新后的事件矩阵
    """
    if news_data is not None:
        # 处理新闻数据
        for date, news in news_data.items():
            if date in events.index:
                # 根据新闻类型设置相应的事件标志
                if '财报' in news or '业绩' in news:
                    events.loc[date, 1] = 1
                if '分红' in news or '派息' in news:
                    events.loc[date, 2] = 1
                # ... 处理其他类型的新闻
    
    if filings_data is not None:
        # 处理公告数据
        for date, filing in filings_data.items():
            if date in events.index:
                # 根据公告类型设置相应的事件标志
                if '董事会' in filing or '高管' in filing:
                    events.loc[date, 6] = 1
                if '问询函' in filing or '监管' in filing:
                    events.loc[date, 7] = 1
                # ... 处理其他类型的公告
    
    return events