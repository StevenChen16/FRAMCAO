# fusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import numpy as np
import pandas as pd
import pickle
from CNN import ResidualFinancialBlock, EnhancedStockDataset
from train import EnhancedCombinedLoss, train_enhanced_model, generate_event_data#, combine_stock_data
from data import load_data_from_csv, download_and_prepare_data
from sklearn.model_selection import train_test_split
from model import EnhancedMCAOLSTMCell, EventProcessor, MCAO, prepare_feature_groups, MCAOEnhancedLSTM
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def combine_stock_data(symbols, start_date, end_date):
    """
    下载多只股票的数据并拼接
    
    Args:
        symbols (list): 股票代码列表
        start_date (str): 开始日期/
        
        end_date (str): 结束日期
    
    Returns:
        pd.DataFrame: 拼接后的数据
    """
    all_data = []
    
    for symbol in tqdm(symbols):
        # 获取单个股票数据
        # data = download_and_prepare_data(symbol, start_date, end_date)
        data = load_data_from_csv(f"./data/{symbol}.csv")

        if not data.empty:
            # 将数据添加到列表中
            all_data.append(data)
    
    # 直接拼接所有数据
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    
    return combined_data

class CNNBranch(nn.Module):
    """分离的CNN分支"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 移除独立的input_norm，改为在CNN branch中使用BatchNorm
        self.cnn_branch = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 第二个卷积块
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # 添加预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        # [batch, seq, features] -> [batch, 1, seq, features]
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)  # [batch, 1, features, seq]
        
        # CNN特征提取
        features = self.cnn_branch(x)  # [batch, hidden_dim, features, seq]
        
        # 重排维度并进行特征聚合
        features = features.mean(dim=2)  # [batch, hidden_dim, seq]
        features = features.transpose(1, 2)  # [batch, seq, hidden_dim]
        
        # 预测
        predictions = []
        for t in range(features.size(1)):
            pred = self.predictor(features[:, t])
            predictions.append(pred)
            
        predictions = torch.stack(predictions, dim=1)
        return predictions, features


class LSTMBranch(nn.Module):
    """分离的LSTM分支"""
    def __init__(self, input_dim, hidden_dim, event_dim, num_event_types):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.atlas = EnhancedMCAOLSTMCell(input_dim, hidden_dim)
        self.event_processor = EventProcessor(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_event_types=num_event_types
        )
        self.mcao = MCAO(input_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, events, time_distances):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # MCAO特征提取
        mcao_features, memory_term = self.mcao(x)
        
        # 初始化状态
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        features = []
        
        for t in range(seq_len):
            current_x = x[:, t]
            current_events = events[:, t]
            current_distances = time_distances[:, t]
            
            # 将输入投影到hidden_dim维度
            projected_x = self.input_proj(current_x)
            
            # 事件处理
            event_impact = self.event_processor(
                current_events,
                h,
                current_distances
            )
            
            # LSTM步进
            h, c = self.atlas(
                projected_x,
                h, c,
                current_x,
                event_impact
            )
            
            pred = self.predictor(h)
            outputs.append(pred)
            features.append(h)
        
        predictions = torch.stack(outputs, dim=1)
        features = torch.stack(features, dim=1)
        
        return predictions, features, mcao_features


def train_cnn_branch(model, train_loader, val_loader, 
                    n_epochs=30, device='cuda', learning_rate=0.0001,
                    checkpoint_dir='checkpoints/cnn'):
    """训练CNN分支"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter('runs/cnn_branch')
    
    # 使用MSE损失
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'CNN Epoch {epoch+1}/{n_epochs}')
        for batch_idx, batch in enumerate(pbar):
            feature_seq = batch['feature_seq'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(feature_seq)
            
            loss = criterion(predictions[:, -1], target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Avg Training Loss: {avg_loss:.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                feature_seq = batch['feature_seq'].to(device)
                target = batch['target'].to(device)
                
                predictions, _ = model(feature_seq)
                loss = criterion(predictions[:, -1], target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    writer.close()
    return model

class LSTMCombinedLoss(nn.Module):
    """专门用于LSTM训练的损失函数"""
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.05, delta=0.05):
        super().__init__()
        self.alpha = alpha   # MSE权重
        self.beta = beta     # 方向预测权重
        self.gamma = gamma   # 连续性权重
        self.delta = delta   # MCAO正则化权重
        
    def forward(self, predictions, targets, prev_price, mcao_features):
        # 取最后一个时间步的预测值
        final_predictions = predictions[:, -1, :]
        
        # 基础MSE损失
        mse_loss = F.mse_loss(final_predictions, targets)
        
        # 方向预测损失
        pred_diff = final_predictions - prev_price.unsqueeze(-1)
        target_diff = targets - prev_price.unsqueeze(-1)
        direction_loss = F.binary_cross_entropy_with_logits(
            (pred_diff > 0).float(),
            (target_diff > 0).float()
        )
        
        # 连续性损失
        smoothness_loss = torch.mean(torch.abs(final_predictions - prev_price.unsqueeze(-1)))
        
        # MCAO特征正则化
        mcao_reg = torch.mean(torch.abs(mcao_features))
        
        # 组合所有损失
        total_loss = (self.alpha * mse_loss +
                     self.beta * direction_loss +
                     self.gamma * smoothness_loss +
                     self.delta * mcao_reg)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'smoothness': smoothness_loss.item(),
            'mcao_reg': mcao_reg.item()
        }

def train_mcao_lstm(model, train_loader, val_loader, 
                    n_epochs=30, device='cuda', learning_rate=0.0001,
                    checkpoint_dir='checkpoints/mcao_lstm'):
    """训练MCAO增强的LSTM模型
    
    Args:
        model: MCAO-LSTM模型实例
        train_loader: 训练数据加载器 
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        device: 训练设备
        learning_rate: 学习率
        checkpoint_dir: 模型保存路径
    """
    import os
    from torch.utils.tensorboard import SummaryWriter
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化tensorboard
    writer = SummaryWriter('runs/mcao_lstm')
    
    # 损失函数
    criterion = LSTMCombinedLoss(
        alpha=0.4,    # MSE权重
        beta=0.4,     # 方向预测权重
        gamma=0.1,    # 平滑度权重
        delta=0.1     # MCAO正则化权重
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # 初始化训练状态
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    global_step = 0
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        total_metrics = {
            'mse': 0, 'direction': 0, 'smoothness': 0,
            'mcao_reg': 0
        }
        epoch_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'MCAO-LSTM Epoch {epoch+1}/{n_epochs}')
        for batch_idx, batch in enumerate(train_pbar):
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            current_price = batch['current_price'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions, mcao_features = model(sequence)
            
            # 计算损失
            loss, metrics = criterion(
                predictions,
                target,
                current_price,
                mcao_features
            )
            
            # 检查NaN
            if torch.isnan(loss):
                print("NaN loss detected!")
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 更新指标
            epoch_loss += loss.item()
            for k, v in metrics.items():
                if not torch.isnan(torch.tensor(v)):
                    total_metrics[k] += v
            
            # 记录训练指标
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', 
                            optimizer.param_groups[0]['lr'], global_step)
            
            # 更新进度条
            avg_loss = epoch_loss / (batch_idx + 1)
            current_metrics = {
                k: v / (batch_idx + 1)
                for k, v in total_metrics.items()
            }
            train_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                **{k: f'{v:.4f}' for k, v in current_metrics.items()}
            })
            
            global_step += 1
        
        # 打印训练阶段摘要
        print(f"\nEpoch {epoch+1} Training Summary:")
        print(f"Average Loss: {epoch_loss/len(train_loader):.4f}")
        for k, v in total_metrics.items():
            print(f"{k}: {v/len(train_loader):.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_metrics = {k: 0 for k in total_metrics.keys()}
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, batch in enumerate(val_pbar):
                sequence = batch['sequence'].to(device)
                target = batch['target'].to(device)
                current_price = batch['current_price'].to(device)
                
                # 前向传播
                predictions, mcao_features = model(sequence)
                
                # 计算损失
                loss, metrics = criterion(
                    predictions,
                    target,
                    current_price,
                    mcao_features
                )
                
                val_loss += loss.item()
                
                # 更新验证指标
                for k, v in metrics.items():
                    if not torch.isnan(torch.tensor(v)):
                        val_metrics[k] += v
                
                # 收集预测结果
                val_predictions.append(predictions[:, -1].cpu())
                val_targets.append(target.cpu())
                
                # 更新进度条
                avg_val_loss = val_loss / (batch_idx + 1)
                current_val_metrics = {
                    k: v / (batch_idx + 1)
                    for k, v in val_metrics.items()
                }
                val_pbar.set_postfix({
                    'val_loss': f'{avg_val_loss:.4f}',
                    **{f'val_{k}': f'{v:.4f}' for k, v in current_val_metrics.items()}
                })
        
        # 计算验证指标
        val_loss /= len(val_loader)
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        # 记录验证指标
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PredictionMean', 
                         val_predictions.mean().item(), epoch)
        writer.add_scalar('Validation/PredictionStd', 
                         val_predictions.std().item(), epoch)
        
        # 打印验证结果
        print(f"\nValidation Results:")
        print(f"Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            v_avg = v/len(val_loader)
            print(f"Val {k}: {v_avg:.4f}")
            writer.add_scalar(f'Validation/{k}', v_avg, epoch)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from checkpoint")
    
    writer.close()
    return model

def train_fusion_model_progressive(model, train_loader, val_loader,
                                 cnn_state_dict, lstm_state_dict,
                                 n_epochs=50, device='cuda', learning_rate=0.0001,
                                 checkpoint_dir='checkpoints/fusion'):
    """MCAO增强版fusion model的渐进式训练
    
    Args:
        model: ATLASCNNFusion 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 
        cnn_state_dict: CNN预训练权重路径
        lstm_state_dict: LSTM预训练权重路径
        n_epochs: 训练轮数
        device: 训练设备
        learning_rate: 学习率
        checkpoint_dir: 模型保存路径
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter('runs/fusion_model')
    
    # 加载预训练的权重
    print("Loading pretrained weights...")
    
    # 加载CNN分支权重
    cnn_weights = torch.load(cnn_state_dict)['model_state_dict']
    cnn_weights_filtered = {
        k.replace('cnn_branch.', ''): v 
        for k, v in cnn_weights.items() 
        if k.startswith('cnn_branch')
    }
    model.cnn_branch.load_state_dict(cnn_weights_filtered)
    
    # 加载MCAO-LSTM权重 
    lstm_weights = torch.load(lstm_state_dict)['model_state_dict']
    
    # 初始化EnhancedMCAOLSTMCell的权重
    mcao_lstm_state = {}
    
    # MCAO相关权重
    mcao_state = {
        k: v for k, v in lstm_weights.items() 
        if k.startswith('mcao.')
    }
    mcao_lstm_state.update(mcao_state)
    
    # LSTM基础组件权重
    lstm_base_state = {
        k: v for k, v in lstm_weights.items()
        if any(k.startswith(prefix) for prefix in 
              ['input_gate.', 'forget_gate.', 'cell_gate.', 'output_gate.'])
    }
    mcao_lstm_state.update(lstm_base_state)
    
    # 投影层权重
    proj_state = {
        k: v for k, v in lstm_weights.items()
        if k.startswith('mcao_proj.') or k.startswith('memory_gate.') or k.startswith('global_modulation.')
    }
    mcao_lstm_state.update(proj_state)
    
    # 加载权重到模型
    try:
        model.atlas.load_state_dict(mcao_lstm_state, strict=False)
        print("Successfully loaded MCAO-LSTM weights!")
    except Exception as e:
        print(f"Warning: Error loading MCAO-LSTM weights: {e}")
        print("Will initialize these components randomly")
    
    # 加载事件处理器权重
    try:
        event_state = {
            k.replace('event_processor.', ''): v
            for k, v in lstm_weights.items()
            if k.startswith('event_processor.')
        }
        model.event_processor.load_state_dict(event_state)
        print("Successfully loaded event processor weights!")
    except Exception as e:
        print(f"Warning: Error loading event processor weights: {e}")
        print("Will initialize event processor randomly")
    
    # 第一阶段：冻结CNN和LSTM分支
    for param in model.cnn_branch.parameters():
        param.requires_grad = False
    for param in model.atlas.parameters():
        param.requires_grad = False
    for param in model.event_processor.parameters():
        param.requires_grad = False
    
    # 使用增强版损失函数
    criterion = EnhancedCombinedLoss()
    
    # 优化器 - 只优化未冻结的参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    print("Stage 1: Training fusion layers only...")
    
    # 训练融合层
    for epoch in range(n_epochs // 2):
        model.train()
        total_metrics = {
            'mse': 0,
            'direction': 0, 
            'smoothness': 0,
            'feature_reg': 0,
            'group_consistency': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Fusion Stage 1 - Epoch {epoch+1}/{n_epochs//2}')
        train_one_epoch(model, pbar, optimizer, scheduler, criterion, writer, 
                       total_metrics, epoch, device)
        
        # 验证
        val_loss = validate_model(model, val_loader, criterion, writer, epoch, device)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'stage1_best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered for Stage 1")
                break
    
    # 第二阶段：解冻所有层进行微调
    epoch = n_epochs // 2
    print("\nStage 2: Fine-tuning all layers...")
    
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
    
    # 使用较小的学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate * 0.1,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )
    
    patience_counter = 0
    best_val_loss = float('inf')
    remaining_epochs = n_epochs - epoch - 1
    
    for epoch in range(remaining_epochs):
        model.train()
        total_metrics = {
            'mse': 0, 'direction': 0, 'smoothness': 0,
            'mcao_reg': 0, 'group_consistency': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Fusion Stage 2 - Epoch {epoch+1}/{remaining_epochs}')
        train_one_epoch(model, pbar, optimizer, scheduler, criterion, writer, 
                       total_metrics, epoch + n_epochs//2, device)
        
        # 验证
        val_loss = validate_model(model, val_loader, criterion, writer, 
                                epoch + n_epochs//2, device)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch + n_epochs//2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'stage2_best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered for Stage 2")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    writer.close()
    return model

def train_one_epoch(model, pbar, optimizer, scheduler, criterion, writer,
                   total_metrics, epoch, device):
    """Helper function for training one epoch"""
    for batch_idx, batch in enumerate(pbar):
        sequence = batch['sequence'].to(device)
        events = batch['events'].to(device)
        time_distances = batch['time_distances'].to(device)
        target = batch['target'].to(device)
        current_price = batch['current_price'].to(device)
        
        optimizer.zero_grad()
        predictions, mcao_features, group_features = model(
            sequence, events, time_distances
        )

        if predictions is None or mcao_features is None or group_features is None:
            print("Skip this batch due to NaN")
            continue
        
        loss, metrics = criterion(
            predictions,
            target,
            current_price,
            mcao_features,
            group_features
        )
        
        if torch.isnan(loss):
            print("NaN loss detected!")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step()
        
        # 确保metrics的键与total_metrics一致
        for k, v in metrics.items():
            if k in total_metrics and not torch.isnan(torch.tensor(v)):
                total_metrics[k] += v
        
        valid_metrics = {
            k: v/len(pbar) 
            for k, v in total_metrics.items() 
            if not np.isnan(v/len(pbar))
        }
        pbar.set_postfix({'loss': loss.item(), **valid_metrics})
        
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(pbar) + batch_idx)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], 
                         epoch * len(pbar) + batch_idx)

def validate_model(model, val_loader, criterion, writer, epoch, device):
    """Helper function for model validation"""
    model.eval()
    val_loss = 0
    val_metrics = {'mse': 0, 'direction': 0, 'smoothness': 0,
                  'mcao_reg': 0, 'group_consistency': 0}
    
    with torch.no_grad():
        for batch in val_loader:
            sequence = batch['sequence'].to(device)
            events = batch['events'].to(device)
            time_distances = batch['time_distances'].to(device)
            target = batch['target'].to(device)
            current_price = batch['current_price'].to(device)
            
            predictions, mcao_features, group_features = model(
                sequence, events, time_distances
            )
            
            loss, metrics = criterion(
                predictions,
                target,
                current_price,
                mcao_features,
                group_features
            )
            
            if not torch.isnan(loss):
                val_loss += loss.item()
                for k, v in metrics.items():
                    if not torch.isnan(torch.tensor(v)):
                        val_metrics[k] += v
    
    val_loss /= len(val_loader)
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    
    print(f"\nValidation Loss: {val_loss:.4f}")
    for k, v in val_metrics.items():
        v_avg = v/len(val_loader)
        if not np.isnan(v_avg):
            print(f"Val {k}: {v_avg:.4f}")
            writer.add_scalar(f'Validation/{k}', v_avg, epoch)
    
    return val_loss



class FusionStockDataset(Dataset):
    """
    融合数据集类,同时支持CNN和LSTM特性
    """
    def __init__(self, data, events, sequence_length=250, prediction_horizon=5):
        """
        Args:
            data: DataFrame containing stock data
            events: Event data matrix
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of days to predict ahead
        """
        # 定义特征顺序
        self.feature_order = [
            'Close', 'Open', 'High', 'Low',  # 价格指标
            'MA5', 'MA20',                   # 均线指标
            'MACD', 'MACD_Signal', 'MACD_Hist',  # MACD族
            'RSI', 'Upper', 'Middle', 'Lower',    # RSI和布林带
            'CRSI', 'Kalman_Price', 'Kalman_Trend',  # 高级指标
            'FFT_21', 'FFT_63',              # FFT指标
            'Volume', 'Volume_MA5'           # 成交量指标
        ]
        
        self.data = data
        self.events = events
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 组织2D特征数据
        self.feature_data = self._organize_features()
        
        # 计算时间距离矩阵
        self.time_distances = self._compute_time_distances()
        
    def _organize_features(self):
        """组织特征为2D张量格式"""
        # 检查特征完整性
        for feature in self.feature_order:
            if feature not in self.data.columns:
                raise ValueError(f"Missing feature: {feature}")
        
        # 提取并堆叠特征
        feature_data = []
        for feature in self.feature_order:
            values = self.data[feature].values
            # 标准化数据
            mean = np.mean(values)
            std = np.std(values)
            normalized = (values - mean) / (std + 1e-8)  # 避免除零
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
            
        # 标准化时间距离
        distances = distances / 100.0  # 归一化到[0,1]范围
        return distances
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # 获取特征序列
        feature_seq = self.feature_data[:, idx:idx + self.sequence_length]
        
        # 获取DataFrame格式的序列(用于LSTM)
        df_seq = self.data.iloc[idx:idx + self.sequence_length]
        sequence = torch.FloatTensor(df_seq.values)
        
        # 获取事件数据
        events = torch.LongTensor(
            self.events[idx:idx + self.sequence_length]
        )
        
        # 获取时间距离
        time_distances = torch.FloatTensor(
            self.time_distances[idx:idx + self.sequence_length]
        )
        
        # 计算目标值
        future_idx = idx + self.sequence_length + self.prediction_horizon - 1
        future_price = self.feature_data[0, future_idx]  # Close price
        current_price = self.feature_data[0, idx + self.sequence_length - 1]
        
        # 计算收益率
        returns = (future_price - current_price) / current_price
        
        # 生成分类标签
        if returns < -0.02:
            label = 0  # 下跌
        elif returns > 0.02:
            label = 2  # 上涨
        else:
            label = 1  # 横盘
            
        # 回归目标
        target = torch.FloatTensor([self.data.iloc[future_idx]['Close']])
        
        return {
            'feature_seq': feature_seq,      # CNN使用
            'sequence': sequence,            # LSTM使用
            'events': events,                # 事件数据
            'time_distances': time_distances,  # 时间距离
            'label': label,                  # 分类标签
            'target': target,                # 回归目标
            'current_price': current_price   # 当前价格
        }

class ATLASCNNFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, event_dim, num_event_types, feature_groups):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 添加输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 修改：先做通道调整，将input_dim通道转换为1通道
        self.channel_adjust = nn.Conv2d(input_dim, 1, kernel_size=1)
        
        # CNN分支 - 保持与预训练模型相同的结构
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        
        # 初始化权重
        for m in self.cnn_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 其他组件
        self.atlas = EnhancedMCAOLSTMCell(input_dim, hidden_dim)
        self.event_processor = EventProcessor(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_event_types=num_event_types
        )
        self.mcao = MCAO(input_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.market_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化所有Linear层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_cnn(self, x):
        """CNN特征提取"""
        batch_size, seq_len, features = x.shape
        
        # [batch, seq, features] -> [batch, features, seq, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        # 新增：通道调整
        x = self.channel_adjust(x)
        
        # CNN特征提取
        cnn_features = self.cnn_branch(x)
        
        # [batch, hidden, seq, 1] -> [batch, seq, hidden]
        return cnn_features.squeeze(-1).permute(0, 2, 1)

    
    def forward(self, x, events, time_distances):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # CNN特征提取
        cnn_features = self._apply_cnn(x)
        if torch.isnan(cnn_features).any():
            print("NaN detected in CNN features")
            return None, None, None
            
        # MCAO特征提取
        mcao_features, memory_term = self.mcao(x)
        if torch.isnan(mcao_features).any():
            print("NaN detected in MCAO features")
            return None, None, None
            
        # 初始化LSTM状态
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        combined_features = []
        
        for t in range(seq_len):
            current_x = x[:, t]
            current_cnn = cnn_features[:, t]
            current_events = events[:, t]
            current_distances = time_distances[:, t]
            
            # 处理事件影响
            event_impact = self.event_processor(
                current_events,
                h,
                current_distances
            )
            
            h, c, mcao_cell_features = self.atlas(
                current_x,
                h,
                c
            )
            
            # 在LSTM之后添加event_impact
            h = h + event_impact

            if torch.isnan(event_impact).any():
                print(f"NaN detected in event_impact at step {t}")
                return None, None, None
            
            if torch.isnan(h).any() or torch.isnan(c).any():
                print(f"NaN detected in LSTM state at step {t}")
                return None, None, None
            
            # 交叉注意力处理
            h_query = h.unsqueeze(1)
            cnn_kv = current_cnn.unsqueeze(1)
            
            h_enhanced, _ = self.cross_attention(
                h_query,
                cnn_kv,
                cnn_kv
            )
            h_enhanced = h_enhanced.squeeze(1)
            if torch.isnan(h_enhanced).any():
                print(f"NaN detected in attention output at step {t}")
                return None, None, None
                
            # 特征融合
            fusion_weight = self.fusion_gate(
                torch.cat([h_enhanced, current_cnn], dim=-1)
            )
            if torch.isnan(fusion_weight).any():
                print(f"NaN detected in fusion_weight at step {t}")
                return None, None, None
                
            market_impact = self.market_state(
                torch.cat([h_enhanced, current_cnn], dim=-1)
            )
            if torch.isnan(market_impact).any():
                print(f"NaN detected in market_impact at step {t}")
                return None, None, None
                
            # 最终特征组合
            combined = fusion_weight * h_enhanced + (1 - fusion_weight) * current_cnn
            combined = combined * market_impact
            
            if torch.isnan(combined).any():
                print(f"NaN detected in combined features at step {t}")
                return None, None, None
                
            pred = self.predictor(combined)
            if torch.isnan(pred).any():
                print(f"NaN detected in predictions at step {t}")
                return None, None, None
                
            outputs.append(pred)
            combined_features.append(combined)
        
        predictions = torch.stack(outputs, dim=1)
        combined_features = torch.stack(combined_features, dim=1)
        
        return predictions, mcao_features, combined_features

# 训练函数
def train_fusion_model(model, train_loader, val_loader, 
                      n_epochs=50, device='cuda', learning_rate=0.0001,
                      checkpoint_dir='checkpoints'):
    """
    训练融合模型的函数
    
    Args:
        model: ATLASCNNFusion模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        device: 训练设备
        learning_rate: 学习率
        checkpoint_dir: 模型保存路径
    """
    from torch.utils.tensorboard import SummaryWriter
    import os
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter()
    
    criterion = EnhancedCombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    global_step = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_metrics = {
            'mse': 0, 'direction': 0, 'smoothness': 0,
            'mcao_reg': 0, 'group_consistency': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch_idx, batch in enumerate(pbar):
            sequence = batch['sequence'].to(device)
            events = batch['events'].to(device)
            time_distances = batch['time_distances'].to(device)
            target = batch['target'].to(device)
            current_price = batch['current_price'].to(device)
            
            optimizer.zero_grad()
            
            try:
                predictions, mcao_features, group_features = model(
                    sequence, events, time_distances
                )

                if predictions is None or mcao_features is None or group_features is None:
                    print("Skip this batch due to NaN")
                    continue
                
                loss, metrics = criterion(
                    predictions,
                    target,
                    current_price,
                    mcao_features,
                    group_features
                )
                
                if torch.isnan(loss):
                    print("NaN loss detected!")
                    continue
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item()
                if grad_norm > 10:
                    print(f"Large gradient norm: {grad_norm}")
                
                optimizer.step()
                scheduler.step()
                
                # Log training metrics to tensorboard
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/GradientNorm', grad_norm, global_step)
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                
                for k, v in metrics.items():
                    if not torch.isnan(torch.tensor(v)):
                        total_metrics[k] += v
                        writer.add_scalar(f'Train/{k}', v, global_step)
                
                global_step += 1
                
            except RuntimeError as e:
                print(f"Error during training: {e}")
                continue
            
            valid_metrics = {
                k: v/len(pbar) 
                for k, v in total_metrics.items() 
                if not np.isnan(v/len(pbar))
            }
            pbar.set_postfix({'loss': loss.item(), **valid_metrics})
        
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {k: 0 for k in total_metrics.keys()}
        
        with torch.no_grad():
            for batch in val_loader:
                sequence = batch['sequence'].to(device)
                events = batch['events'].to(device)
                time_distances = batch['time_distances'].to(device)
                target = batch['target'].to(device)
                current_price = batch['current_price'].to(device)
                
                predictions, mcao_features, group_features = model(
                    sequence, events, time_distances
                )
                
                loss, metrics = criterion(
                    predictions,
                    target,
                    current_price,
                    mcao_features,
                    group_features
                )
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    for k, v in metrics.items():
                        if not torch.isnan(torch.tensor(v)):
                            val_metrics[k] += v
        
        val_loss /= len(val_loader)
        
        # Log validation metrics to tensorboard
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Validation/{k}', v/len(val_loader), epoch)
        
        print(f"\nEpoch {epoch+1} Validation Metrics:")
        print(f"Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v/len(val_loader):.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'global_step': global_step
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'global_step': global_step
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, 'best_model.pt')
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    writer.close()
    return model

# 在fusion_model.py中修改main函数

def main():
    # 参数设置
    input_dim = 21  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    event_dim = 32  # 事件嵌入维度
    num_event_types = 10  # 事件类型数量
    
    # 数据准备
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", 
        "^GSPC", "^NDX", "^DJI", "^IXIC",
        "UNH", "ABBV", "LLY",
        "FANG", "DLR", "PSA", "BABA", "JD", "BIDU",
        "QQQ"
    ]
    data = combine_stock_data(symbols, '2020-01-01', '2024-01-01')
    events = generate_event_data(data)
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_events, val_events = train_test_split(events, test_size=0.2, shuffle=False)
    
    # 创建数据集
    train_dataset = FusionStockDataset(train_data, train_events)
    val_dataset = FusionStockDataset(val_data, val_events)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=3200,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=3200,
        shuffle=False,
        num_workers=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # Step 1: 训练CNN分支
    print("\nStep 1: Training CNN Branch...")
    cnn_model = CNNBranch(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    cnn_model = train_cnn_branch(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=30,
        device=device,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints/cnn'
    )
    
    # Step 2: 训练LSTM分支
    print("\nStep 2: Training LSTM Branch...")
    model = MCAOEnhancedLSTM(
        input_size=21,    # 输入特征维度
        hidden_size=128   # 隐藏层维度
    ).to(device)

    # 训练模型
    trained_model = train_mcao_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=30,
        device=device,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints/mcao_lstm'
    )
    
    # Step 3: 融合训练
    print("\nStep 3: Progressive Fusion Training...")
    fusion_model = ATLASCNNFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device)
    
    # 使用预训练的分支模型进行融合训练
    trained_model = train_fusion_model_progressive(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        cnn_state_dict='checkpoints/cnn/best_model.pt',
        lstm_state_dict='checkpoints/mcao_lstm/best_model.pt',
        n_epochs=50,
        device=device,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints/fusion'
    )
    
    # 保存最终模型
    torch.save(
        trained_model.state_dict(),
        'checkpoints/final_model.pt'
    )# 在fusion_model.py中修改main函数

def main():
    # 参数设置
    input_dim = 21  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    event_dim = 32  # 事件嵌入维度
    num_event_types = 10  # 事件类型数量
    
    # 数据准备
    symbols_200 = [# 科技股
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", 
    "ADBE", "NFLX", "CSCO", "ORCL", "QCOM", "IBM", "AMAT", "MU", "NOW", "SNOW",
    
    # 金融股
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "V", "MA",
    "COF", "USB", "PNC", "SCHW", "BK", "TFC", "AIG", "MET", "PRU", "ALL",
    
    # 医疗保健
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "LLY",
    "AMGN", "GILD", "ISRG", "CVS", "CI", "HUM", "BIIB", "VRTX", "REGN", "ZTS",
    
    # 消费品
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "COST", "DIS", "CMCSA", "VZ", "T", "CL", "EL", "KMB", "GIS", "K", "PDD", "GOTU",
    
    # 工业
    "BA", "GE", "MMM", "CAT", "HON", "UPS", "LMT", "RTX", "DE", "EMR",
    "FDX", "NSC", "UNP", "WM", "ETN", "PH", "ROK", "CMI", "IR", "GD",
    
    # 能源
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY",
    "KMI", "WMB", "EP", "HAL", "DVN", "HES", "MRO", "APA", "FANG", "BKR",
    
    # 材料
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "DOW",
    "DD", "PPG", "ALB", "EMN", "CE", "CF", "MOS", "IFF", "FMC", "SEE",
    
    # 房地产
    "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "AVB", "EQR",
    "SPG", "VTR", "BXP", "ARE", "MAA", "UDR", "HST", "KIM", "REG", "ESS",
    
    # 中概股
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "TME", "BILI", "IQ",
    
    # ETF
    "SPY", "QQQ", "DIA", "IWM", "VOO", "IVV", "ARKK", "XLF", "XLK", "XLE", 
    "VNQ", "TLT", "HYG", "EEM", "GDX", "VTI", "IEMG", "XLY", "XLP", "USO",

    # 指数
    "^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", 
    "^IXIC", "^HSI", "000001.SS", "^GDAXI", "^FTSE",
    ]
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", 
        "^GSPC", "^NDX", "^DJI", "^IXIC",
        "UNH", "ABBV", "LLY",
        "FANG", "DLR", "PSA", "BABA", "JD", "BIDU",
        "QQQ"
    ]
    data = combine_stock_data(symbols, '2020-01-01', '2024-01-01')
    events = generate_event_data(data)
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_events, val_events = train_test_split(events, test_size=0.2, shuffle=False)
    
    # 创建数据集
    train_dataset = FusionStockDataset(train_data, train_events)
    val_dataset = FusionStockDataset(val_data, val_events)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4
    )

    #CNN分支数据加载器
    train_loader_CNN = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader_CNN = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    #LSTM分支数据加载器
    train_loader_lstm = DataLoader(
        train_dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=4
    )
    val_loader_lstm = DataLoader(
        val_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # Step 1: 训练CNN分支
    print("\nStep 1: Training CNN Branch...")
    cnn_model = CNNBranch(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    cnn_model = train_cnn_branch(
        model=cnn_model,
        train_loader=train_loader_CNN,
        val_loader=val_loader_CNN,
        n_epochs=1,
        device=device,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints/cnn'
    )
    
    # Step 2: 训练LSTM分支
    print("\nStep 2: Training LSTM Branch...")
    model = MCAOEnhancedLSTM(
    input_size=21,    # 输入特征维度
    hidden_size=128   # 隐藏层维度
    ).to(device)

    # 训练模型
    trained_model = train_mcao_lstm(
        model=model,
        train_loader=train_loader_lstm,
        val_loader=val_loader_lstm,
        n_epochs=1,
        device=device,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints/mcao_lstm'
    )

    
    # Step 3: 融合训练
    print("\nStep 3: Progressive Fusion Training...")
    fusion_model = ATLASCNNFusion(  # 这里使用ATLASCNNFusion是正确的
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device)
    
    # 使用预训练的分支模型进行融合训练
    trained_model = train_fusion_model_progressive(
        model=fusion_model,  # 传入ATLASCNNFusion实例
        train_loader=train_loader,
        val_loader=val_loader,
        cnn_state_dict='checkpoints/cnn/best_model.pt',
        lstm_state_dict='checkpoints/mcao_lstm/best_model.pt',
        n_epochs=50,
        device=device,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints/fusion'
    )
    
    # 保存最终模型
    torch.save(
        trained_model.state_dict(),
        'checkpoints/final_model.pt'
    )
    
if __name__ == "__main__":
    main()